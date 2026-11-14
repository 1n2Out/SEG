"""Real selective-scan, AMP, warm-start and checkpoint checks; no mock CUDA backend."""
import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA selective scan required")


@pytest.fixture(autouse=True)
def production_cudnn_settings():
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    # Match train_brats.run so exact forward comparisons use deterministic kernels.
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


def labels(device="cuda"):
    target = torch.zeros(2, 64, 64, dtype=torch.long, device=device)
    target[1, 18:30, 18:30] = 2
    target[1, 20:25, 20:25] = 3
    return target


@pytest.mark.parametrize("mode", ["concat", "gated"])
def test_real_backbone_identity_amp_learning_and_resume(tmp_path, mode):
    from train_brats import SOURCE, build_criterion, build_model, load_checkpoint, save_checkpoint
    config = json.loads((SOURCE / "configs/brats2020.json").read_text())
    torch.manual_seed(42)
    baseline = build_model(config)
    rng = torch.get_rng_state().clone()
    modified = {**config, "detail_fusion": {"mode": mode, "full_channels": 16, "half_channels": 32}}
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
    baseline.train()
    model.train()
    with torch.no_grad():  # same drop-path RNG: identical initial training function
        torch.manual_seed(87)
        reference = baseline(image)
        torch.manual_seed(87)
        torch.testing.assert_close(model(image), reference, atol=0, rtol=0)
    del baseline, reference
    target = labels()
    criterion = build_criterion(modified)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    branch = model.detail_fusion
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(image), target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step == 0:
            assert branch.head.weight.grad.abs().sum() > 0
            assert branch.stem_full.body[0].weight.grad.abs().sum() == 0
        else:
            assert branch.stem_full.body[0].weight.grad.abs().sum() > 0
            assert branch.up_half.project.weight.grad.abs().sum() > 0
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
    assert_state_equal(resumed.state_dict(), model.state_dict())
    assert_state_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    model.eval()
    resumed.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        output = model(image)
        torch.testing.assert_close(resumed(image), output, atol=0, rtol=0)
    assert output.shape == (2, 4, 64, 64) and torch.isfinite(output).all()


def test_warm_start_reproduces_source_model_and_groups_learning_rates(tmp_path):
    from train_brats import (SOURCE, apply_warm_start, build_model, extension_prefixes,
                             optimizer_parameter_groups, save_checkpoint)
    config = json.loads((SOURCE / "configs/brats2020.json").read_text())
    torch.manual_seed(1)
    source = build_model(config)
    with torch.no_grad():  # move away from the shared seed so loading is observable
        for parameter in source.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    path = tmp_path / "best.pth"
    save_checkpoint(path, {"model_state_dict": source.state_dict(), "manifest_sha256": "split", "epoch": 55})
    modified = {**config, "pretrained": None, "detail_fusion": {"mode": "gated"},
                "warm_start": {"checkpoint": str(path), "new_parameter_lr": 1e-4}}
    torch.manual_seed(2)
    model = build_model(modified)
    report = apply_warm_start(model, modified["warm_start"], "split", extension_prefixes(modified))
    assert report["new_tensors"] and all(name.startswith("detail_fusion.") for name in report["new_tensors"])
    groups = optimizer_parameter_groups(model, modified)
    assert sum(p.numel() for p in groups[1]["params"]) == 70462
    source.cuda().eval()
    model.cuda().eval()
    image = torch.randn(2, 4, 64, 64, device="cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        torch.testing.assert_close(model(image), source(image), atol=0, rtol=0)


def test_stacking_on_a_trained_gated_model_starts_at_its_output_then_learns(tmp_path):
    from train_brats import (SOURCE, apply_warm_start, build_criterion, build_model,
                             new_parameter_prefixes, optimizer_parameter_groups, save_checkpoint)
    config = json.loads((SOURCE / "configs/brats2020.json").read_text())
    gated = {**config, "pretrained": None, "detail_fusion": {"mode": "gated"}}
    torch.manual_seed(1)
    source = build_model(gated)
    with torch.no_grad():  # a trained-like source: the detail head is no longer zero
        for parameter in source.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    path = tmp_path / "best.pth"
    save_checkpoint(path, {"model_state_dict": source.state_dict(), "manifest_sha256": "split", "epoch": 95})
    new = ["detail_fusion.boundary_head.", "detail_fusion.context."]
    stacked = {**config, "pretrained": None,
               "detail_fusion": {"mode": "gated", "boundary_weight": 0.5, "context_slices": 1},
               "warm_start": {"checkpoint": str(path), "new_parameter_lr": 1e-4, "new_parameters": new}}
    torch.manual_seed(2)
    model = build_model(stacked)
    report = apply_warm_start(model, stacked["warm_start"], "split", new_parameter_prefixes(stacked))
    assert report["new_tensors"] and all(name.startswith(tuple(new)) for name in report["new_tensors"])
    groups = optimizer_parameter_groups(model, stacked)
    assert sum(p.numel() for p in groups[1]["params"]) == 75105 - 70462
    source.cuda().eval()
    model.cuda().eval()
    image = torch.randn(2, 12, 64, 64, device="cuda")  # current slice first, then its neighbours
    with torch.no_grad(), torch.cuda.amp.autocast():
        torch.testing.assert_close(model(image), source(image[:, :4]), atol=0, rtol=0)
    model.train()
    target = labels()
    criterion = build_criterion(stacked)
    optimizer = torch.optim.AdamW(groups, lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    branch = model.detail_fusion
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(image), target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert branch.boundary_head.weight.grad.abs().sum() > 0
        assert branch.context.project.weight.grad.abs().sum() > 0
        assert (branch.context.block.body[0].weight.grad.abs().sum() > 0) == (step == 1)
        scaler.step(optimizer)
        scaler.update()
