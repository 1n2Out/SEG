"""Non-training checks for stopping, checkpointing and distributed input sampling."""
import json
import random
from pathlib import Path

import pytest

from brats.training import EarlyStopping


def test_patience_counts_validation_checks_and_restores(tmp_path):
    stopping = EarlyStopping(enabled=True, patience=2, min_delta=0.01)
    assert not stopping.update(0.5)  # epoch 1
    assert not stopping.update(0.505)  # epoch 5: small gain still counts as stale
    path = tmp_path / "stopping.json"
    path.write_text(json.dumps(stopping.state_dict()), encoding="utf-8")
    resumed = EarlyStopping(enabled=True, patience=2, min_delta=0.01)
    resumed.load_state_dict(json.loads(path.read_text(encoding="utf-8")))
    assert resumed.update(0.49)  # epoch 10, not epoch 3
    assert resumed.bad_validations == 2


def test_significant_cumulative_gain_resets_patience():
    stopping = EarlyStopping(enabled=True, patience=2, min_delta=0.01)
    for score in (0.5, 0.505, 0.511):
        assert not stopping.update(score)
    assert stopping.bad_validations == 0
    assert stopping.reference_score == 0.511
    assert not stopping.update(0.511)
    assert stopping.update(0.511)


def test_zero_delta_and_disabled_stopping():
    stopping = EarlyStopping(enabled=True, patience=1, min_delta=0)
    assert not stopping.update(0.5)
    assert not stopping.update(0.500001)
    assert stopping.update(0.500001)
    legacy = EarlyStopping()  # Old configs omit early_stopping and retain fixed epochs.
    for _ in range(20):
        assert not legacy.update(0.5)


@pytest.mark.parametrize("kwargs", [
    {"enabled": "true"}, {"patience": 0}, {"patience": True}, {"patience": 1.5},
    {"min_delta": -1}, {"min_delta": float("nan")}, {"min_delta": float("inf")},
])
def test_reject_invalid_stopping_config(kwargs):
    with pytest.raises(ValueError):
        EarlyStopping(**kwargs)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_validation(score):
    with pytest.raises(ValueError, match="finite"):
        EarlyStopping().update(score)


def test_default_config_and_legacy_config():
    from train_brats import SOURCE, validate_config
    config = json.loads((SOURCE / "configs/brats2020.json").read_text(encoding="utf-8"))
    validate_config(config)
    assert EarlyStopping(**config["early_stopping"]).enabled
    del config["early_stopping"]
    validate_config(config)


def test_distributed_sampler_coverage_padding_and_resume():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, DistributedSampler
    dataset = list(range(11))
    samplers = [DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42) for rank in range(2)]
    for sampler in samplers:
        sampler.set_epoch(5)
    partitions = [list(sampler) for sampler in samplers]
    assert set(partitions[0] + partitions[1]) == set(dataset)
    assert len(partitions[0]) == len(partitions[1]) == 6
    assert len(set(partitions[0]) & set(partitions[1])) == 1  # one padded slice
    assert [len(DataLoader(dataset, batch_size=4, sampler=s)) for s in samplers] == [2, 2]
    samplers[0].set_epoch(6)
    assert list(samplers[0]) != partitions[0]
    resumed = DistributedSampler(dataset, num_replicas=2, rank=0, seed=42)
    resumed.set_epoch(6)
    assert list(resumed) == list(samplers[0])


def test_rank_rng_roundtrip_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    import numpy as np
    from train_brats import capture_rng, restore_rng
    fake_cuda = torch.Generator().manual_seed(123)
    monkeypatch.setattr(torch.cuda, "get_rng_state", fake_cuda.get_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", fake_cuda.set_state)
    loader = torch.Generator().manual_seed(42)
    state = capture_rng(loader)

    def draw():
        return (random.random(), np.random.random(), torch.rand(1).item(),
                torch.rand(1, generator=fake_cuda).item(), torch.rand(1, generator=loader).item())

    expected = draw()
    restore_rng(state, loader)
    assert draw() == expected


def test_checkpoint_roundtrip_preserves_stopping_and_plain_keys(tmp_path):
    torch = pytest.importorskip("torch")
    from train_brats import load_checkpoint, save_checkpoint
    stopping = EarlyStopping(enabled=True, patience=2)
    stopping.update(0.5)
    stopping.update(0.4)
    model = torch.nn.Linear(4, 4)
    path = tmp_path / "latest.pth"
    save_checkpoint(path, {"model_state_dict": model.state_dict(),
                           "early_stopping_state": stopping.state_dict(), "world_size": 2})
    loaded = load_checkpoint(path)
    # A DDP run saves raw_model's state, consumable by single-GPU evaluation.
    restored_model = torch.nn.Linear(4, 4)
    restored_model.load_state_dict(loaded["model_state_dict"], strict=True)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored_model.parameters()))
    resumed = EarlyStopping(enabled=True, patience=2)
    resumed.load_state_dict(loaded["early_stopping_state"])
    assert resumed.update(0.4)
    assert not Path(str(path) + ".tmp").exists()
