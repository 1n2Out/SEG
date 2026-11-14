"""Run on the training machine; no CUDA needed for these functional tests."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is not installed on the data-inspection host")
from brats.data import MODALITIES, PREPROCESS_VERSION, write_json
from brats.dataset import BraTSSlices
from brats.engine import predict_volume
from brats.losses import BraTSCEDiceLoss


def test_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (2, 32, 32))
    loss = BraTSCEDiceLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def write_case(root):
    """One cached 3-slice case whose one-hot image encodes its own labels."""
    case_id = "BraTS20_Training_001"
    directory = root / case_id
    directory.mkdir()
    label = np.zeros((3, 24, 22), np.uint8)
    label[1, 3:8, 4:9] = 4
    exclusive = label.copy()
    exclusive[exclusive == 4] = 3
    images = np.eye(4, dtype=np.float32)[exclusive].transpose(0, 3, 1, 2)
    np.save(directory / "image.npy", images.astype(np.float16))
    np.save(directory / "label.npy", label)
    write_json(directory / "metadata.json", {"preprocess_version": PREPROCESS_VERSION,
               "case_id": case_id, "modalities": list(MODALITIES), "shape_xyz": [24, 22, 3],
               "spacing_xyz_mm": [1, 1, 1], "spatial_unit": "mm", "affine": np.eye(4).tolist()})
    manifest = {"splits": {"train": [case_id]}, "cases": {case_id: {"id": case_id, "label": "unused"}}}
    return manifest, images, label


def test_slice_mapping_and_prediction_reconstruct_all_slices(tmp_path):
    manifest, images, label = write_case(tmp_path)
    dataset = BraTSSlices(manifest, tmp_path, input_size=32, augment=False)
    assert len(dataset) == 3  # both tumor-negative slices are retained
    item = dataset[1]
    assert tuple(item["image"].shape) == (4, 32, 32)
    assert item["label"].max().item() == 3
    model = torch.nn.Identity()
    prediction = predict_volume(model, images, torch.device("cpu"), input_size=32, batch_size=2)
    assert np.array_equal(prediction, label.transpose(1, 2, 0))


@pytest.fixture
def vmunet_components(monkeypatch):
    class DummyVSSM(torch.nn.Module):
        def __init__(self, in_chans, num_classes, **kwargs):
            super().__init__()
            self.patch_embed = torch.nn.Module()
            self.patch_embed.proj = torch.nn.Conv2d(in_chans, 4, 1)
            self.final_conv = torch.nn.Conv2d(4, num_classes, 1)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
            self.layers_up = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])

        def forward(self, x):
            return self.final_conv(self.patch_embed.proj(x))

    package = types.ModuleType("test_vmunet_package")
    package.__path__ = []
    dependency = types.ModuleType("test_vmunet_package.vmamba")
    dependency.VSSM = DummyVSSM
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    filename = Path(__file__).resolve().parents[1] / "models/vmunet/vmunet.py"
    spec = importlib.util.spec_from_file_location("test_vmunet_package.vmunet", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, DummyVSSM


def test_channel_adaptation_and_shape_safe_pretraining(tmp_path, vmunet_components):
    module, DummyVSSM = vmunet_components
    path = tmp_path / "pretrained.pth"
    source = DummyVSSM(3, 7)
    torch.save({"model": source.state_dict()}, path)
    model = module.VMUNet(input_channels=4, num_classes=4, load_ckpt_path=path)
    report = model.load_from()
    assert report["adapted"] == ["patch_embed.proj.weight"]
    assert "final_conv.weight" in report["skipped"]
    assert torch.allclose(model.vmunet.patch_embed.proj.weight.sum(1), source.patch_embed.proj.weight.sum(1))
    assert torch.equal(model.vmunet.layers_up[3].weight, source.layers[0].weight)
    assert model(torch.zeros(1, 4, 32, 32)).shape == (1, 4, 32, 32)
    with pytest.raises(ValueError, match="Expected 4 input"):
        model(torch.zeros(1, 1, 32, 32))
    single = module.VMUNet(input_channels=1, num_classes=4)
    assert single(torch.zeros(1, 1, 32, 32)).shape == (1, 4, 32, 32)


def test_pretraining_prefers_ema_and_records_coverage_and_source(tmp_path, vmunet_components):
    import hashlib
    module, DummyVSSM = vmunet_components
    ordinary, ema = DummyVSSM(3, 7), DummyVSSM(3, 7)
    with torch.no_grad():
        for p in ordinary.parameters():
            p.fill_(-0.5)
        for p in ema.parameters():
            p.fill_(0.25)
    path = tmp_path / 'ema.pth'
    torch.save({'model': ordinary.state_dict(), 'model_ema': ema.state_dict()}, path)
    model = module.VMUNet(input_channels=4, num_classes=4, load_ckpt_path=path)
    head_before = model.vmunet.final_conv.weight.detach().clone()
    report = model.load_from()
    assert report['checkpoint_key'] == 'model_ema'
    assert report['source_path'] == str(path.resolve())
    assert report['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert 0 < report['loaded_fraction'] < 1
    assert report['loaded_numel'] < report['target_numel']
    assert 'final_conv.weight' in report['missing']
    assert torch.equal(model.vmunet.final_conv.weight, head_before)
    assert torch.equal(model.vmunet.layers_up[0].weight, ema.layers[3].weight)
    assert torch.allclose(model.vmunet.patch_embed.proj.weight.sum(1), ema.patch_embed.proj.weight.sum(1))


def test_empty_ema_falls_back_to_model(tmp_path, vmunet_components):
    module, DummyVSSM = vmunet_components
    path = tmp_path / 'model.pth'
    torch.save({'model': DummyVSSM(3, 7).state_dict(), 'model_ema': {}}, path)
    model = module.VMUNet(input_channels=4, num_classes=4, load_ckpt_path=path)
    assert model.load_from()['checkpoint_key'] == 'model'


def test_pretraining_missing_file_and_incompatible_checkpoint_fail(tmp_path, vmunet_components):
    module, _ = vmunet_components
    path = tmp_path / 'missing.pth'
    model = module.VMUNet(input_channels=4, num_classes=4, load_ckpt_path=path)
    with pytest.raises(FileNotFoundError, match='Pretrained checkpoint not found'):
        model.load_from()
    torch.save({'optimizer': {}}, path)
    with pytest.raises(ValueError, match='model_ema or model'):
        model.load_from()
    torch.save({'model': {'unrelated.weight': torch.ones(2)}}, path)
    with pytest.raises(ValueError, match='No compatible'):
        model.load_from()


def test_context_slices_stack_current_first_and_repeat_edge_slices():
    from brats.dataset import context_offsets, stack_slices
    assert context_offsets(0) == [0] and context_offsets(1) == [0, -1, 1]
    volume = np.arange(3, dtype=np.float16)[:, None, None, None] * np.ones((3, 4, 2, 2), np.float16)
    stacked = stack_slices(volume, [0, 1, 2], 1)
    assert stacked.shape == (3, 12, 2, 2) and stacked.dtype == np.float32
    # First modality of each 4-channel block: current, previous, next slice index.
    assert [stacked[i, ::4, 0, 0].tolist() for i in range(3)] == [[0, 0, 1], [1, 0, 2], [2, 1, 2]]
    assert np.array_equal(stack_slices(volume, [1], 0)[0], volume[1].astype(np.float32))
    with pytest.raises(ValueError, match="context_slices"):
        context_offsets(-1)


def test_context_dataset_and_inference_stack_slices_alike(tmp_path):
    manifest, images, label = write_case(tmp_path)
    plain = BraTSSlices(manifest, tmp_path, input_size=32, augment=False)
    context = BraTSSlices(manifest, tmp_path, input_size=32, augment=False, context_slices=1)
    item = context[1]
    assert tuple(item["image"].shape) == (12, 32, 32)
    for block, z in enumerate((1, 0, 2)):
        assert torch.equal(item["image"][4 * block:4 * block + 4], plain[z]["image"])
    # One random rotation/flip moves every stacked slice and the label together.
    augmented = BraTSSlices(manifest, tmp_path, input_size=32, context_slices=1)
    reference = BraTSSlices(manifest, tmp_path, input_size=32)
    np.random.seed(5)
    moved = augmented[1]
    for block, z in enumerate((1, 0, 2)):
        np.random.seed(5)
        expected = reference[z]
        assert torch.equal(moved["image"][4 * block:4 * block + 4], expected["image"])
    np.random.seed(5)
    assert torch.equal(moved["label"], reference[1]["label"])

    class CurrentSlice(torch.nn.Module):
        def forward(self, x):
            assert x.shape[1] == 12
            return x[:, :4]

    prediction = predict_volume(CurrentSlice(), images, torch.device("cpu"), input_size=32, batch_size=2,
                                context_slices=1)
    assert np.array_equal(prediction, label.transpose(1, 2, 0))
