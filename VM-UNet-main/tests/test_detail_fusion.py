"""Configuration, zero-residual identity, branch learning and model wiring on CPU."""
import json

import pytest

from brats.innovations.detail_fusion import resolve_settings
from train_brats import (SOURCE, extension_prefixes, innovation_settings, load_training_config,
                         validate_config)


@pytest.mark.parametrize("settings", [None, [], {"mode": "full"}, {"mode": "gated", "gate": True},
    {"full_channels": 1}, {"half_channels": 0}, {"full_channels": True}, {"half_channels": 8.0}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError, match="detail_fusion"):
        resolve_settings(settings)


def test_defaults_are_resolved_persisted_and_mark_new_parameters(tmp_path):
    assert resolve_settings({}) == {"mode": "off", "full_channels": 16, "half_channels": 32}
    config = json.loads((SOURCE / "configs/brats2020.json").read_text(encoding="utf-8"))
    config["detail_fusion"] = {"mode": "gated"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_training_config(path)
    assert loaded["detail_fusion"] == {"mode": "gated", "full_channels": 16, "half_channels": 32}
    validate_config(loaded)
    assert extension_prefixes(loaded) == ("detail_fusion.",)
    assert extension_prefixes({"detail_fusion": {"mode": "off"}}) == ()
    assert extension_prefixes({"shallow_refinement": {"mode": "gated"}}) == ("shallow_refinement.",)
    assert extension_prefixes({}) == ()


def test_formal_configs_only_add_the_extension_and_output_location():
    baseline = json.loads((SOURCE / "configs/brats2020.json").read_text(encoding="utf-8"))
    assert "detail_fusion" not in baseline
    gpu_count = len(baseline["launch"].pop("gpus"))
    for mode in ("concat", "gated"):
        path = SOURCE / "configs/detail_fusion" / (mode + ".json")
        experiment = json.loads(path.read_text(encoding="utf-8"))
        assert experiment.pop("detail_fusion") == {"mode": mode, "full_channels": 16, "half_channels": 32}
        assert experiment["launch"]["work_dir"] != baseline["launch"]["work_dir"]
        experiment["launch"]["work_dir"] = baseline["launch"]["work_dir"]
        # Any free GPU may be chosen; the GPU count fixes the protocol's global batch.
        assert len(experiment["launch"].pop("gpus")) == gpu_count
        assert experiment == baseline
        validate_config(load_training_config(path))


def test_one_extension_per_run_and_retired_option_still_rejected():
    with pytest.raises(ValueError, match="at most one"):
        innovation_settings({"shallow_refinement": {"mode": "off"}, "detail_fusion": {"mode": "gated"}})
    with pytest.raises(ValueError, match="retired"):
        innovation_settings({"region_presence": {"mode": "full"}, "detail_fusion": {"mode": "gated"}})
    name, settings = innovation_settings({"detail_fusion": {"mode": "concat"}})
    assert name == "detail_fusion" and settings["mode"] == "concat"
    assert innovation_settings({}) is None


torch = pytest.importorskip("torch")
from brats.innovations.detail_fusion.block import DetailFusion
from brats.losses import BraTSCEDiceLoss
from train_brats import build_criterion

BRANCH = ("stem_full", "stem_half", "up_half", "fuse_half", "up_full", "fuse_full")


def first_conv(module):
    return next(m for m in module.modules() if isinstance(m, torch.nn.Conv2d))


@pytest.mark.parametrize("mode", ["concat", "gated"])
def test_zero_head_is_identity_then_whole_branch_learns(mode):
    torch.manual_seed(5)
    block = DetailFusion(4, 12, 4, mode, full_channels=4, half_channels=8)
    image, decoder = torch.randn(2, 4, 16, 16), torch.randn(2, 4, 4, 12)
    target = torch.randn(2, 4, 16, 16)
    assert torch.equal(block(image, decoder), torch.zeros(2, 4, 16, 16))
    optimizer = torch.optim.SGD(block.parameters(), lr=0.2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (block(image, decoder) - target).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        if step == 0:
            # Only the zero head learns first; upstream gradients are exact zeros, not missing.
            assert block.head.weight.grad.abs().sum() > 0 and block.head.bias.grad.abs().sum() > 0
            assert all(p.grad.abs().sum() == 0 for name, p in block.named_parameters()
                       if not name.startswith("head."))
        else:
            gates = ("gate_half", "gate_full") if mode == "gated" else ()
            for name in BRANCH + gates:
                assert first_conv(getattr(block, name)).weight.grad.abs().sum() > 0, name
        optimizer.step()
    block.eval()
    assert block(image, decoder).abs().sum() > 0  # active at inference too


@pytest.mark.parametrize("mode,count", [("concat", 69476), ("gated", 70462)])
def test_parameter_budget_for_the_vmunet_decoder(mode, count):
    block = DetailFusion(4, 96, 4, mode)
    assert sum(p.numel() for p in block.parameters()) == count
    assert hasattr(block, "gate_half") == (mode == "gated")


def test_each_slice_is_independent_of_its_batch():
    # GroupNorm, not BatchNorm: training, inference and DDP shards see identical per-slice results.
    torch.manual_seed(3)
    block = DetailFusion(4, 8, 4, "gated", 4, 8)
    torch.nn.init.normal_(block.head.weight)
    image, decoder = torch.randn(3, 4, 16, 16), torch.randn(3, 4, 4, 8)
    together = block(image, decoder)
    alone = torch.cat([block(image[i:i + 1], decoder[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(together, alone)


def test_rejects_misaligned_inputs_and_disabled_mode():
    block = DetailFusion(4, 8, 4, "gated", 4, 8)
    with pytest.raises(ValueError, match="1/4-resolution"):
        block(torch.zeros(1, 4, 16, 16), torch.zeros(1, 2, 2, 8))
    with pytest.raises(ValueError, match="1/4-resolution"):
        block(torch.zeros(2, 4, 16, 16), torch.zeros(1, 4, 4, 8))
    with pytest.raises(ValueError, match="BCHW"):
        block(torch.zeros(4, 16, 16), torch.zeros(1, 4, 4, 8))
    with pytest.raises(ValueError, match="concat or gated"):
        DetailFusion(4, 8, 4, "off")


@pytest.mark.parametrize("mode", ["off", "concat", "gated"])
def test_every_mode_uses_the_original_loss(mode):
    criterion = build_criterion({"detail_fusion": {"mode": mode}})
    assert type(criterion) is BraTSCEDiceLoss


KWARGS = dict(input_channels=4, num_classes=4, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1],
              drop_path_rate=0.2)


@pytest.fixture
def vmunet_classes():
    # Constructing VSSM needs timm/einops but no selective scan; forward is never run here.
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    from models.vmunet.vmunet import VMUNet
    from brats.innovations.detail_fusion.model import DetailFusionVMUNet
    return VMUNet, DetailFusionVMUNet


def test_construction_preserves_baseline_weights_and_random_stream(vmunet_classes):
    VMUNet, DetailFusionVMUNet = vmunet_classes
    torch.manual_seed(42)
    baseline = VMUNet(**KWARGS)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    model = DetailFusionVMUNet(detail_fusion={"mode": "gated"}, **KWARGS)
    assert torch.equal(expected_rng, torch.get_rng_state())
    state = model.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in baseline.state_dict().items())
    added = set(state) - set(baseline.state_dict())
    assert added and all(name.startswith("detail_fusion.") for name in added)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == 70462
    with pytest.raises(ValueError, match="off mode"):
        DetailFusionVMUNet(detail_fusion={"mode": "off"}, **KWARGS)


def test_forward_adds_branch_logits_to_unchanged_backbone_logits(vmunet_classes, monkeypatch):
    _, DetailFusionVMUNet = vmunet_classes
    torch.manual_seed(0)
    model = DetailFusionVMUNet(detail_fusion={"mode": "gated", "full_channels": 4, "half_channels": 8},
                               **KWARGS)
    image = torch.randn(2, 4, 32, 32)
    decoded = torch.randn(2, 8, 8, 96, requires_grad=True)
    logits = torch.randn(2, 4, 32, 32)
    seen = {}

    def features(x):
        seen["image"] = x
        return "encoded", ["skips"]

    # Stub only the backbone stages; this checks wiring, not the selective scan.
    monkeypatch.setattr(model.vmunet, "forward_features", features)
    monkeypatch.setattr(model.vmunet, "forward_features_up", lambda encoded, skips: decoded)
    monkeypatch.setattr(model.vmunet, "forward_final", lambda features: logits)
    torch.testing.assert_close(model(image), logits, atol=0, rtol=0)
    assert seen["image"] is image
    torch.nn.init.normal_(model.detail_fusion.head.weight)
    output = model(image)
    torch.testing.assert_close(output, logits + model.detail_fusion(image, decoded))
    output.sum().backward()
    assert decoded.grad.abs().sum() > 0  # the branch reads the final decoder features
    with pytest.raises(ValueError, match="Expected 4 input"):
        model(torch.zeros(1, 1, 32, 32))
