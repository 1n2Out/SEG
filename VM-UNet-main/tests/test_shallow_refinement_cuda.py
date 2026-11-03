"""Real selective-scan, AMP and checkpoint tests; no mock CUDA backend."""
import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA selective scan required")


@pytest.fixture(autouse=True)
def production_cudnn_settings():
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    # Match train_brats.run: AMP convolution backward otherwise permits
    # nondeterministic kernels, obscuring checkpoint round-trip checks.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    yield
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def assert_state_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("mode", ["simple", "gated"])
def test_real_backbone_identity_amp_learning_and_resume(tmp_path, mode):
    from train_brats import SOURCE, build_model, build_criterion, load_checkpoint, save_checkpoint
    config = json.loads((SOURCE / "configs/brats2020.json").read_text())
    torch.manual_seed(42)
    baseline = build_model(config)
    rng = torch.get_rng_state().clone()
    modified = {**config, "shallow_refinement": {"mode": mode, "hidden_channels": 24}}
    torch.manual_seed(42)
    model = build_model(modified)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    baseline.cuda().eval()
    model.cuda().eval()
    image = torch.randn(2, 4, 64, 64, device="cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        torch.testing.assert_close(model(image), baseline(image), atol=0, rtol=0)
    # Same drop-path RNG also preserves the initial training function.
    baseline.train()
    model.train()
    with torch.no_grad():
        torch.manual_seed(87)
        reference = baseline(image)
        torch.manual_seed(87)
        torch.testing.assert_close(model(image), reference, atol=0, rtol=0)
    del baseline, reference
    labels = torch.zeros(2, 64, 64, dtype=torch.long, device="cuda")
    labels[1, 20:25, 20:25] = 3
    criterion = build_criterion(modified)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(image), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step == 0:
            assert model.shallow_refinement.alpha.grad.abs().sum() > 0
        else:
            assert model.shallow_refinement.projection.weight.grad.abs().sum() > 0
        scaler.step(optimizer)
        scaler.update()
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, {"config": modified, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict()})
    saved = load_checkpoint(path)
    resumed = build_model(saved["config"]).cuda()
    resumed.load_state_dict(saved["model_state_dict"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    resumed_optimizer.load_state_dict(saved["optimizer_state_dict"])
    resumed_scaler = torch.cuda.amp.GradScaler()
    resumed_scaler.load_state_dict(saved["scaler_state_dict"])
    assert_state_equal(resumed.state_dict(), model.state_dict())
    assert_state_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    assert_state_equal(resumed_scaler.state_dict(), scaler.state_dict())
    model.eval()
    resumed.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        torch.testing.assert_close(resumed(image), model(image), atol=0, rtol=0)
    # Exact state/forward equality is required. Independent CUDA backwards are
    # not bitwise repeatable even for the unchanged baseline on this host, so
    # test successful continuation rather than asserting identical future steps.
    model.train()
    resumed.train()
    torch.manual_seed(71)
    with torch.no_grad(), torch.cuda.amp.autocast():
        expected_loss = criterion(model(image), labels)
    torch.manual_seed(71)
    resumed_optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss = criterion(resumed(image), labels)
    torch.testing.assert_close(loss.detach(), expected_loss, atol=0, rtol=0)
    alpha_before = resumed.shallow_refinement.alpha.detach().clone()
    resumed_scaler.scale(loss).backward()
    resumed_scaler.unscale_(resumed_optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in resumed.parameters())
    resumed_scaler.step(resumed_optimizer)
    resumed_scaler.update()
    assert not torch.equal(alpha_before, resumed.shallow_refinement.alpha)


def test_disabled_model_and_exported_legacy_checkpoint():
    from train_brats import SOURCE, build_model, load_checkpoint
    config = json.loads((SOURCE / "configs/brats2020.json").read_text())
    torch.manual_seed(42)
    baseline = build_model(config)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    disabled = build_model({**config, "shallow_refinement": {"mode": "off"}})
    assert type(baseline) is type(disabled)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[name])
    path = SOURCE / "results/brats2020_region_presence_full_1103_1004/exports/best_inference.pth"
    if path.is_file():
        exported = load_checkpoint(path)
        restored = build_model(exported["config"])
        assert type(restored) is type(baseline)
        restored.load_state_dict(exported["model_state_dict"], strict=True)
