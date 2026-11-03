"""Configuration, identity initialization, branch learning and loss invariance."""
import json

import pytest

from brats.innovations.shallow_refinement import resolve_settings
from train_brats import SOURCE, load_training_config, validate_config, innovation_settings


@pytest.mark.parametrize("settings", [None, False, {"mode": "full"}, {"alpha_init": 1},
    {"hidden_channels": 0}, {"hidden_channels": True}, {"hidden_channels": 1.5}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError, match="shallow_refinement"):
        resolve_settings(settings)


def test_configs_only_change_model_extension_and_output_location(tmp_path):
    baseline = json.loads((SOURCE / "configs/brats2020.json").read_text())
    assert "region_presence" not in baseline and "shallow_refinement" not in baseline
    for mode in ("simple", "gated"):
        path = SOURCE / "configs/shallow_refinement" / (mode + ".json")
        experiment = json.loads(path.read_text())
        assert experiment.pop("shallow_refinement")["mode"] == mode
        assert experiment["launch"]["work_dir"] != baseline["launch"]["work_dir"]
        experiment["launch"]["work_dir"] = baseline["launch"]["work_dir"]
        assert experiment == baseline
        validate_config(load_training_config(path))
    baseline["shallow_refinement"] = {"mode": "gated"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(baseline))
    assert load_training_config(path)["shallow_refinement"] == resolve_settings({"mode": "gated"})


def test_retired_configuration_cannot_silently_become_baseline():
    with pytest.raises(ValueError, match="retired"):
        innovation_settings({"region_presence": {"mode": "full"}})


torch = pytest.importorskip("torch")
from brats.innovations.shallow_refinement.block import ShallowRefinement
from brats.losses import BraTSCEDiceLoss
from train_brats import build_criterion


@pytest.mark.parametrize("mode", ["simple", "gated"])
def test_zero_residual_is_identity_and_learns_in_two_steps(mode):
    torch.manual_seed(11)
    block = ShallowRefinement(8, mode, 4)
    decoder = torch.randn(2, 6, 6, 8)
    shallow = torch.randn_like(decoder)
    torch.testing.assert_close(block(decoder, shallow), decoder, atol=0, rtol=0)
    optimizer = torch.optim.SGD(block.parameters(), lr=0.2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        block(decoder, shallow).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        if step == 0:
            assert block.alpha.grad.abs().sum() > 0
            assert block.projection.weight.grad.abs().sum() == 0
        else:
            assert block.projection.weight.grad.abs().sum() > 0
            if mode == "gated":
                assert block.gate.weight.grad.abs().sum() > 0
                assert block.local3[0].weight.grad.abs().sum() > 0
                assert block.local5[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(block(decoder, shallow), decoder)
    block.eval()
    assert not torch.equal(block(decoder, shallow), decoder)  # active at inference too


@pytest.mark.parametrize("mode,count", [("simple", 9408), ("gated", 26520)])
def test_parameter_budget_and_feature_alignment(mode, count):
    block = ShallowRefinement(96, mode, 24)
    assert sum(p.numel() for p in block.parameters()) == count
    with pytest.raises(ValueError, match="matching BHWC"):
        block(torch.zeros(1, 8, 8, 96), torch.zeros(1, 4, 4, 96))


@pytest.mark.parametrize("mode", [None, "off", "simple", "gated"])
def test_every_mode_uses_exact_original_loss_and_gradient(mode):
    config = {} if mode is None else {"shallow_refinement": {"mode": mode}}
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    labels = torch.randint(0, 4, (2, 8, 8))
    baseline = BraTSCEDiceLoss()(logits, labels)
    criterion = build_criterion(config)
    assert type(criterion) is BraTSCEDiceLoss
    actual = criterion(logits, labels)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0],
                               torch.autograd.grad(baseline, logits)[0], atol=0, rtol=0)


def test_baseline_does_not_import_optional_package(monkeypatch):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("brats.innovations"):
            pytest.fail("Baseline must keep working without optional modules")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert innovation_settings({}) is None
    assert type(build_criterion({})) is BraTSCEDiceLoss
