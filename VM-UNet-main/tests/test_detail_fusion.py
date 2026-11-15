"""Configuration, zero-residual identity, branch learning and model wiring on CPU."""
import json

import pytest

from brats.innovations.detail_fusion import resolve_settings
from train_brats import (SOURCE, extension_prefixes, innovation_settings, load_training_config,
                         validate_config)


@pytest.mark.parametrize("settings", [None, [], {"mode": "full"}, {"mode": "gated", "gate": True},
    {"full_channels": 1}, {"half_channels": 0}, {"full_channels": True}, {"half_channels": 8.0},
    {"mode": "gated", "deep_supervision": 1}, {"mode": "gated", "input_stem": "yes"},
    {"mode": "off", "deep_supervision": True}, {"input_stem": True}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError, match="detail_fusion"):
        resolve_settings(settings)


RESOLVED_GATED = {"mode": "gated", "full_channels": 16, "half_channels": 32,
                  "deep_supervision": False, "input_stem": False}


def test_defaults_are_resolved_persisted_and_mark_new_parameters(tmp_path):
    assert resolve_settings({}) == {**RESOLVED_GATED, "mode": "off"}
    config = json.loads((SOURCE / "configs/brats2020.json").read_text(encoding="utf-8"))
    config["detail_fusion"] = {"mode": "gated"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_training_config(path)
    assert loaded["detail_fusion"] == RESOLVED_GATED
    validate_config(loaded)
    assert extension_prefixes(loaded) == ("detail_fusion.",)
    assert extension_prefixes({"detail_fusion": {"mode": "off"}}) == ()
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


@pytest.mark.parametrize("name,options", [("gated-ds", {"deep_supervision": True}),
    ("gated-stem", {"input_stem": True}), ("gated-ds-stem", {"deep_supervision": True, "input_stem": True})])
def test_option_configs_differ_from_gated_only_by_options_and_output(name, options):
    # TASK-005 ablations: same protocol and branch as EXP-006, plus the named options.
    gated = json.loads((SOURCE / "configs/detail_fusion/gated.json").read_text(encoding="utf-8"))
    path = SOURCE / "configs/detail_fusion" / (name + ".json")
    experiment = json.loads(path.read_text(encoding="utf-8"))
    assert experiment["launch"].pop("work_dir") == "results/brats2020_detail_" + name.replace("-", "_")
    gated["launch"].pop("work_dir")
    gated["detail_fusion"].update(options)
    assert experiment == gated
    assert load_training_config(path)["detail_fusion"] == {**RESOLVED_GATED, **options}
    validate_config(load_training_config(path))


def test_one_extension_per_run_and_retired_options_still_rejected(monkeypatch):
    import train_brats
    with monkeypatch.context() as patch:
        patch.setattr(train_brats, "EXTENSIONS", ("detail_fusion", "future_extension"))
        with pytest.raises(ValueError, match="at most one"):
            innovation_settings({"future_extension": {"mode": "off"}, "detail_fusion": {"mode": "gated"}})
    for retired in ("region_presence", "shallow_refinement"):
        with pytest.raises(ValueError, match="retired"):
            innovation_settings({retired: {"mode": "gated"}})
        with pytest.raises(ValueError, match="retired"):
            innovation_settings({retired: {"mode": "gated"}, "detail_fusion": {"mode": "gated"}})
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


OPTION_SHAPES = dict(decoder_stage_channels=(384, 192), embed_channels=96)
BOTH = {"deep_supervision": True, "input_stem": True}


@pytest.mark.parametrize("options,extra", [({"deep_supervision": True}, 2444),
                                           ({"input_stem": True}, 63968), (BOTH, 66412)])
def test_options_add_parameters_without_changing_the_shared_branch(options, extra):
    torch.manual_seed(11)
    plain = DetailFusion(4, 96, 4, "gated")
    rng = torch.get_rng_state().clone()
    torch.manual_seed(11)
    block = DetailFusion(4, 96, 4, "gated", **options, **OPTION_SHAPES)
    assert torch.equal(rng, torch.get_rng_state())
    state = block.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in plain.state_dict().items())
    assert sum(p.numel() for p in block.parameters()) - sum(p.numel() for p in plain.parameters()) == extra


def test_each_option_is_initialized_independently_of_the_other():
    def build(**options):
        torch.manual_seed(13)
        return DetailFusion(4, 96, 4, "gated", **options, **OPTION_SHAPES).state_dict()
    both = build(**BOTH)
    for options in ({"deep_supervision": True}, {"input_stem": True}):
        assert all(torch.equal(value, both[name]) for name, value in build(**options).items())


def test_input_stem_starts_at_zero_then_learns_through_the_shared_detail():
    torch.manual_seed(7)
    block = DetailFusion(4, 12, 4, "gated", 4, 8, input_stem=True, embed_channels=6)
    image, target = torch.randn(2, 4, 16, 16), torch.randn(2, 4, 4, 6)
    tokens = block.tokens(block.extract(image)[1])
    assert tokens.shape == (2, 4, 4, 6) and torch.equal(tokens, torch.zeros_like(tokens))
    optimizer = torch.optim.SGD(block.parameters(), lr=0.2)
    stem = block.input_stem
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (block.tokens(block.extract(image)[1]) - target).square().mean().backward()
        assert stem.project.weight.grad.abs().sum() > 0
        if step == 0:
            assert first_conv(stem.block).weight.grad.abs().sum() == 0
        else:
            assert first_conv(stem.block).weight.grad.abs().sum() > 0
            assert first_conv(block.stem_half).weight.grad.abs().sum() > 0  # shared with the fusion
        optimizer.step()
    with pytest.raises(ValueError, match="no input stem"):
        DetailFusion(4, 12, 4, "gated", 4, 8).tokens(torch.zeros(1, 8, 8, 8))
    with pytest.raises(ValueError, match="patch-embedding"):
        DetailFusion(4, 12, 4, "gated", 4, 8, input_stem=True)


def test_auxiliary_heads_read_the_fused_half_and_the_decoder_stages():
    torch.manual_seed(9)
    block = DetailFusion(4, 12, 4, "gated", 4, 8, deep_supervision=True, decoder_stage_channels=(10, 6))
    image, decoder = torch.randn(2, 4, 16, 16), torch.randn(2, 4, 4, 12)
    full, half = block.extract(image)
    residual, auxiliary = block.fuse(full, half, decoder, with_auxiliary=True)
    torch.testing.assert_close(residual, block(image, decoder), atol=0, rtol=0)
    assert torch.equal(residual, torch.zeros_like(residual))  # the output stays at zero
    assert auxiliary.shape == (2, 4, 8, 8) and auxiliary.abs().sum() > 0  # heads train at once
    auxiliary.sum().backward()
    assert first_conv(block.fuse_half).weight.grad.abs().sum() > 0 and block.head.weight.grad is None
    sixteenth, eighth = block.decoder_auxiliary((torch.randn(2, 1, 1, 10), torch.randn(2, 2, 2, 6)))
    assert sixteenth.shape == (2, 4, 1, 1) and eighth.shape == (2, 4, 2, 2)
    with pytest.raises(ValueError, match="1/16 and 1/8"):
        block.decoder_auxiliary((torch.randn(2, 1, 1, 10),))
    with pytest.raises(ValueError, match="1/16 and 1/8"):
        DetailFusion(4, 12, 4, "gated", 4, 8, deep_supervision=True)
    with pytest.raises(ValueError, match="no deep supervision"):
        DetailFusion(4, 12, 4, "gated", 4, 8).fuse(full, half, decoder, with_auxiliary=True)


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


def test_only_deep_supervision_changes_the_loss():
    from brats.innovations.detail_fusion.loss import DeepSupervisionLoss
    assert type(build_criterion({"detail_fusion": {"mode": "gated", "input_stem": True}})) is BraTSCEDiceLoss
    assert type(build_criterion({"detail_fusion": {"mode": "concat", "deep_supervision": True}})) is DeepSupervisionLoss


def test_deep_supervision_loss_weights_auxiliary_outputs_at_label_resolution():
    from brats.innovations.detail_fusion.loss import AUXILIARY_WEIGHTS, DeepSupervisionLoss
    torch.manual_seed(4)
    target = torch.randint(0, 4, (2, 16, 16))
    logits = torch.randn(2, 4, 16, 16)
    auxiliary = (torch.randn(2, 4, 8, 8), torch.randn(2, 4, 16, 16),
                 torch.randn(2, 4, 2, 2), torch.randn(2, 4, 1, 1))
    criterion = DeepSupervisionLoss()
    total, components = criterion((logits, auxiliary), target, return_components=True)
    base = BraTSCEDiceLoss()
    main, parts = base(logits, target, return_components=True)

    def at_labels(output):
        if output.shape[-1] == 16:
            return output
        return torch.nn.functional.interpolate(output, size=(16, 16), mode="bilinear", align_corners=False)

    deep = sum(weight * base(at_labels(output), target) for weight, output in zip(AUXILIARY_WEIGHTS, auxiliary))
    torch.testing.assert_close(total, main + deep)
    torch.testing.assert_close(components["deep"], deep)
    assert torch.equal(components["ce"], parts["ce"]) and torch.equal(components["dice"], parts["dice"])
    assert AUXILIARY_WEIGHTS == (0.5, 0.25, 0.125, 0.0625)  # halved per scale, as in nnU-Net
    assert criterion.component_names == ("ce", "dice", "deep")
    for bad in (logits, (logits,), (logits, auxiliary[:3]), (logits, (torch.randn(2, 3, 8, 8),) + auxiliary[1:])):
        with pytest.raises(ValueError, match="(?i)deep supervision|auxiliary"):
            criterion(bad, target)


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


@pytest.mark.parametrize("options,extra", [({"deep_supervision": True}, 2444),
                                           ({"input_stem": True}, 63968), (BOTH, 66412)])
def test_options_keep_gated_weights_and_baseline_random_stream(vmunet_classes, options, extra):
    VMUNet, DetailFusionVMUNet = vmunet_classes
    torch.manual_seed(42)
    VMUNet(**KWARGS)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    model = DetailFusionVMUNet(detail_fusion={"mode": "gated", **options}, **KWARGS)
    assert torch.equal(rng, torch.get_rng_state())
    torch.manual_seed(42)
    gated_model = DetailFusionVMUNet(detail_fusion={"mode": "gated"}, **KWARGS)
    gated, state = gated_model.state_dict(), model.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in gated.items())  # EXP-006 init
    assert all(name.startswith("detail_fusion.") for name in set(state) - set(gated))
    count = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in gated_model.parameters())
    assert count == extra


def skip_selective_scan(model):
    """Keep the real embedding, merging, expanding and final layers; skip only VSS blocks."""
    for layer in model.vmunet.layers:
        layer.forward = lambda x, layer=layer: x + 0.5 if layer.downsample is None else layer.downsample(x)
    for layer in model.vmunet.layers_up:
        layer.forward = lambda x, layer=layer: x - 0.25 if layer.upsample is None else layer.upsample(x)


def test_input_stem_adds_tokens_and_deep_supervision_returns_every_scale(vmunet_classes):
    _, DetailFusionVMUNet = vmunet_classes
    torch.manual_seed(0)
    model = DetailFusionVMUNet(detail_fusion={"mode": "gated", "full_channels": 4, "half_channels": 8,
                                              **BOTH}, **KWARGS)
    skip_selective_scan(model)
    backbone, branch = model.vmunet, model.detail_fusion
    image = torch.randn(2, 4, 32, 32)

    def reference(tokens):
        x = backbone.pos_drop(backbone.patch_embed(image) + tokens)
        skips = []
        for layer in backbone.layers:
            skips.append(x)
            x = layer(x)
        decoded = backbone.forward_features_up(x, skips)
        return backbone.forward_final(decoded), decoded

    model.eval()
    with torch.no_grad():
        tokens = branch.tokens(branch.extract(image)[1])
        assert torch.equal(tokens, torch.zeros_like(tokens))
        # A zero projection leaves the original encoder exactly unchanged.
        encoded, skips = model.encode(image, tokens)
        expected, expected_skips = backbone.forward_features(image)
        assert torch.equal(encoded, expected) and all(map(torch.equal, skips, expected_skips))
        torch.nn.init.normal_(branch.input_stem.project.weight)
        tokens = branch.tokens(branch.extract(image)[1])
        coarse, decoded = reference(tokens)
        assert not torch.equal(coarse, reference(torch.zeros_like(tokens))[0])  # tokens reach the encoder
        evaluated = model(image)
        torch.testing.assert_close(evaluated, coarse + branch(image, decoded), atol=0, rtol=0)
        decoded_again, stages = model.decode(*model.encode(image, tokens))
        assert torch.equal(decoded_again, decoded) and torch.equal(stages[-1], decoded)
        assert [tuple(s.shape) for s in stages] == [(2, 1, 1, 768), (2, 2, 2, 384), (2, 4, 4, 192), (2, 8, 8, 96)]
    model.train()
    logits, auxiliary = model(image)
    torch.testing.assert_close(logits, evaluated, atol=0, rtol=0)  # supervision never changes the output
    half, coarse_out, eighth, sixteenth = auxiliary
    torch.testing.assert_close(coarse_out, coarse, atol=0, rtol=0)  # the backbone's own logits
    assert [tuple(a.shape) for a in auxiliary] == [(2, 4, 16, 16), (2, 4, 32, 32), (2, 4, 4, 4), (2, 4, 2, 2)]
    target = torch.randint(0, 4, (2, 32, 32))
    build_criterion({"detail_fusion": {"mode": "gated", **BOTH}})((logits, auxiliary), target).backward()
    # DDP needs every branch parameter in each training graph.
    assert all(p.grad is not None for p in branch.parameters())


def test_deep_supervision_alone_keeps_the_original_inference_path(vmunet_classes, monkeypatch):
    _, DetailFusionVMUNet = vmunet_classes
    torch.manual_seed(1)
    model = DetailFusionVMUNet(detail_fusion={"mode": "concat", "full_channels": 4, "half_channels": 8,
                                              "deep_supervision": True}, **KWARGS)
    skip_selective_scan(model)
    image = torch.randn(2, 4, 32, 32)
    model.eval()
    monkeypatch.setattr(model, "decode", lambda *args: pytest.fail("evaluation must not run the stage loop"))
    with torch.no_grad():
        evaluated = model(image)
    monkeypatch.undo()
    assert torch.is_tensor(evaluated) and evaluated.shape == (2, 4, 32, 32)
    model.train()
    logits, auxiliary = model(image)
    torch.testing.assert_close(logits, evaluated, atol=0, rtol=0)
    assert len(auxiliary) == 4 and not hasattr(model.detail_fusion, "input_stem")
