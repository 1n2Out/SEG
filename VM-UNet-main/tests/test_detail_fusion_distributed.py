"""Opt-in two-rank Gloo check: the zero-initialized detail parts start learning under DDP."""
import os
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from brats.innovations.detail_fusion.block import DetailFusion

# Plain gated branch, and the stacked options whose training forward returns two outputs.
OPTIONS = ({}, {"boundary": True, "context_slices": 1})


def inputs(rank, options):
    channels = 12 if options.get("context_slices") else 4
    image = torch.linspace(-1, 1, channels * 16 * 16).reshape(1, channels, 16, 16) + rank * 0.1
    decoder = torch.linspace(0, 1, 8).repeat(1, 4, 4, 1) + rank
    target = torch.cos(torch.linspace(0, 6, 4 * 16 * 16)).reshape(1, 4, 16, 16) * (rank + 1)
    return image, decoder, target


def build(options):
    torch.manual_seed(29)
    return DetailFusion(4, 8, 4, "gated", 4, 8, **options)


def train(model, options, image, decoder, target):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        if options.get("boundary"):
            residual, boundary = model(image, decoder, with_boundary=True)
            loss = (residual - target).square().mean() + (boundary - 1).square().mean()
        else:
            loss = (model(image, decoder) - target).square().mean()
        loss.backward()
        optimizer.step()


def worker(rank, rendezvous, folder, index):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        options = OPTIONS[index]
        model = DistributedDataParallel(build(options))
        train(model, options, *inputs(rank, options))
        torch.save(model.module.state_dict(), Path(folder) / (str(rank) + ".pth"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("BRATS_TEST_DISTRIBUTED") != "1", reason="Set BRATS_TEST_DISTRIBUTED=1 for two-process Gloo")
@pytest.mark.parametrize("index", range(len(OPTIONS)))
def test_zero_parts_can_start_learning_under_ddp(tmp_path, index):
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), index), nprocs=2, join=True)
    options = OPTIONS[index]
    initial = build(options).stem_full.body[0].weight.detach().clone()
    model = build(options)
    batches = [inputs(rank, options) for rank in range(2)]
    train(model, options, *(torch.cat(parts) for parts in zip(*batches)))
    assert model.head.weight.abs().sum() > 0
    assert not torch.equal(model.stem_full.body[0].weight, initial)  # upstream learned on step two
    if options:
        assert model.boundary_head.weight.abs().sum() > 0 and model.context.project.weight.abs().sum() > 0
    for rank in range(2):
        actual = torch.load(tmp_path / (str(rank) + ".pth"), weights_only=False)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(actual[name], expected)
