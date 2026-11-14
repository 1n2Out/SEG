"""Configuration, zero-residual identity, branch learning and model wiring on CPU."""
import json

import pytest

from brats.innovations.detail_fusion import resolve_settings
from train_brats import (SOURCE, extension_prefixes, innovation_settings, input_context,
                         load_training_config, validate_config)


@pytest.mark.parametrize("settings", [None, [], {"mode": "full"}, {"mode": "gated", "gate": True},
    {"full_channels": 1}, {"half_channels": 0}, {"full_channels": True}, {"half_channels": 8.0},
    {"mode": "gated", "boundary_weight": -0.1}, {"mode": "gated", "boundary_weight": True},
    {"mode": "gated", "boundary_weight": float("nan")}, {"mode": "gated", "boundary_weight": "0.5"},
    {"mode": "gated", "context_slices": 2}, {"mode": "gated", "context_slices": -1},
    {"mode": "gated", "context_slices": 1.0}, {"mode": "gated", "context_slices": True},
    {"mode": "off", "boundary_weight": 0.5}, {"mode": "off", "context_slices": 1}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError, match="detail_fusion"):
        resolve_settings(settings)


DEFAULTS = {"mode": "off", "full_channels": 16, "half_channels": 32, "boundary_weight": 0.0, "context_slices": 0}


def test_defaults_are_resolved_persisted_and_mark_new_parameters(tmp_path):
    assert resolve_settings({}) == DEFAULTS
    config = json.loads((SOURCE / "configs/brats2020.json").read_text(encoding="utf-8"))
    config["detail_fusion"] = {"mode": "gated"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_training_config(path)
    assert loaded["detail_fusion"] == {**DEFAULTS, "mode": "gated"}
    validate_config(loaded)
    assert extension_prefixes(loaded) == ("detail_fusion.",)
    assert extension_prefixes({"detail_fusion": {"mode": "off"}}) == ()
    assert extension_prefixes({"shallow_refinement": {"mode": "gated"}}) == ("shallow_refinement.",)
    assert extension_prefixes({}) == ()


def test_only_the_context_option_changes_the_model_input():
    assert input_context({}) == 0
    assert input_context({"shallow_refinement": {"mode": "gated"}}) == 0
    assert input_context({"detail_fusion": {"mode": "gated", "boundary_weight": 0.5}}) == 0
    assert input_context({"detail_fusion": {"mode": "concat", "context_slices": 1}}) == 1


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


STACKING = {"boundary": {"boundary_weight": 0.5}, "context": {"context_slices": 1},
            "boundary-context": {"boundary_weight": 0.5, "context_slices": 1}}


@pytest.mark.parametrize("name", sorted(STACKING))
def test_stacking_formal_configs_only_add_their_options_to_gated(name):
    gated = json.loads((SOURCE / "configs/detail_fusion/gated.json").read_text(encoding="utf-8"))
    path = SOURCE / "configs/detail_fusion" / ("gated-" + name + ".json")
    experiment = json.loads(path.read_text(encoding="utf-8"))
    assert experiment["detail_fusion"] == {**gated["detail_fusion"], **STACKING[name]}
    assert experiment["launch"].pop("work_dir") == "results/brats2020-gated-" + name
    gated["launch"].pop("work_dir")
    for config in (experiment, gated):
        config.pop("detail_fusion")
    assert experiment == gated
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


@pytest.mark.parametrize("options,count", [({"boundary": True}, 70513), ({"context_slices": 1}, 75054),
                                           ({"boundary": True, "context_slices": 1}, 75105)])
def test_optional_parts_are_small_and_leave_the_shared_branch_initialization(options, count):
    torch.manual_seed(11)
    plain = DetailFusion(4, 96, 4, "gated")
    torch.manual_seed(11)
    block = DetailFusion(4, 96, 4, "gated", **options)
    assert sum(p.numel() for p in block.parameters()) == count
    state = block.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in plain.state_dict().items())
    added = set(state) - set(plain.state_dict())
    assert added and all(name.startswith(("boundary_head.", "context.")) for name in added)


def test_context_is_a_zero_residual_on_a_trained_branch_then_learns():
    torch.manual_seed(7)
    plain = DetailFusion(4, 12, 4, "gated", 4, 8)
    torch.nn.init.normal_(plain.head.weight)  # a trained source branch, as in EXP-006
    block = DetailFusion(4, 12, 4, "gated", 4, 8, context_slices=1)
    block.load_state_dict(plain.state_dict(), strict=False)
    stacked, decoder = torch.randn(2, 12, 16, 16), torch.randn(2, 4, 4, 12)
    # The current slice occupies the first four channels; neighbours start with no effect.
    torch.testing.assert_close(block(stacked, decoder), plain(stacked[:, :4], decoder), atol=0, rtol=0)
    target = torch.randn(2, 4, 16, 16)
    optimizer = torch.optim.SGD(block.parameters(), lr=0.2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (block(stacked, decoder) - target).square().mean().backward()
        project, first = block.context.project.weight.grad, block.context.block.body[0].weight.grad
        assert project.abs().sum() > 0  # the zero projection learns at once
        assert (first.abs().sum() > 0) == (step == 1)  # its input block one step later
        optimizer.step()
    with pytest.raises(ValueError, match="context slices"):
        block(stacked[:, :4], decoder)


def test_boundary_head_is_an_extra_training_output_only():
    torch.manual_seed(9)
    block = DetailFusion(4, 12, 4, "gated", 4, 8, boundary=True)
    image, decoder = torch.randn(2, 4, 16, 16), torch.randn(2, 4, 4, 12)
    residual, boundary = block(image, decoder, with_boundary=True)
    torch.testing.assert_close(residual, block(image, decoder), atol=0, rtol=0)
    assert boundary.shape == (2, 3, 16, 16) and torch.equal(boundary, torch.zeros_like(boundary))
    boundary.sum().backward()
    assert block.boundary_head.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="no boundary head"):
        DetailFusion(4, 12, 4, "gated", 4, 8)(image, decoder, with_boundary=True)


def test_boundary_bands_cover_both_sides_of_each_region_edge():
    from brats.innovations.detail_fusion.loss import boundary_targets
    target = torch.zeros(1, 9, 9, dtype=torch.long)
    target[0, 2:7, 2:7] = 2  # edema square
    target[0, 4, 4] = 3      # one enhancing voxel inside
    bands = boundary_targets(target)
    assert bands.shape == (1, 3, 9, 9)
    wt = bands[0, 0]
    assert wt[1, 4] == 1 and wt[2, 4] == 1 and wt[3, 4] == 0  # outer and inner ring, not the interior
    assert bands[0, 1].sum() == bands[0, 2].sum() == 9  # 3x3 band around the single TC/ET voxel
    assert torch.equal(boundary_targets(torch.zeros(2, 5, 5, dtype=torch.long)), torch.zeros(2, 3, 5, 5))


def test_boundary_loss_adds_a_weighted_term_to_the_unchanged_main_loss():
    from brats.innovations.detail_fusion.loss import BoundaryAuxLoss
    criterion = build_criterion({"detail_fusion": {"mode": "gated", "boundary_weight": 0.5}})
    assert type(criterion) is BoundaryAuxLoss and criterion.component_names == ("ce", "dice", "boundary")
    torch.manual_seed(4)
    logits = torch.randn(2, 4, 16, 16)
    boundary = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 16, 16))
    total, components = criterion((logits, boundary), target, return_components=True)
    main = BraTSCEDiceLoss()(logits, target)
    torch.testing.assert_close(total, main + 0.5 * components["boundary"])
    assert set(components) == {"ce", "dice", "boundary"} and components["boundary"] > 0
    total.backward()
    assert torch.isfinite(boundary.grad).all() and boundary.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="training outputs"):
        criterion(logits, target)
    with pytest.raises(ValueError, match="positive"):
        BoundaryAuxLoss(0)


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


STACKED = {"mode": "gated", "boundary_weight": 0.5, "context_slices": 1}


def test_stacked_options_keep_baseline_weights_and_random_stream(vmunet_classes):
    VMUNet, DetailFusionVMUNet = vmunet_classes
    torch.manual_seed(42)
    baseline = VMUNet(**KWARGS)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    model = DetailFusionVMUNet(detail_fusion=STACKED, **KWARGS)
    assert torch.equal(expected_rng, torch.get_rng_state())
    state = model.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in baseline.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == 75105


def test_backbone_reads_the_current_slice_and_training_adds_boundary_logits(vmunet_classes, monkeypatch):
    _, DetailFusionVMUNet = vmunet_classes
    torch.manual_seed(0)
    model = DetailFusionVMUNet(detail_fusion={**STACKED, "full_channels": 4, "half_channels": 8}, **KWARGS)
    stacked = torch.randn(2, 12, 32, 32)
    decoded = torch.randn(2, 8, 8, 96)
    logits = torch.randn(2, 4, 32, 32)
    seen = {}

    def features(x):
        seen["image"] = x
        return "encoded", ["skips"]

    monkeypatch.setattr(model.vmunet, "forward_features", features)
    monkeypatch.setattr(model.vmunet, "forward_features_up", lambda encoded, skips: decoded)
    monkeypatch.setattr(model.vmunet, "forward_final", lambda features: logits)
    model.train()
    output, boundary = model(stacked)
    assert torch.equal(seen["image"], stacked[:, :4])
    torch.testing.assert_close(output, logits, atol=0, rtol=0)  # zero-initialized branch
    assert boundary.shape == (2, 3, 32, 32)
    model.eval()  # evaluation and 3D inference receive plain logits
    torch.testing.assert_close(model(stacked), logits, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Expected 12 input"):
        model(torch.zeros(1, 4, 32, 32))
