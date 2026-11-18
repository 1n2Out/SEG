"""Opt-in two-rank Gloo check: the zero-initialized detail head starts learning under DDP."""
import os
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from brats.innovations.detail_fusion.block import DetailFusion


def inputs(rank):
    image = torch.linspace(-1, 1, 4 * 16 * 16).reshape(1, 4, 16, 16) + rank * 0.1
    decoder = torch.linspace(0, 1, 8).repeat(1, 4, 4, 1) + rank
    target = torch.cos(torch.linspace(0, 6, 4 * 16 * 16)).reshape(1, 4, 16, 16) * (rank + 1)
    return image, decoder, target


def build():
    torch.manual_seed(29)
    return DetailFusion(4, 8, 4, "gated", 4, 8)


def train(model, image, decoder, target):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model(image, decoder) - target).square().mean().backward()
        optimizer.step()


def worker(rank, rendezvous, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        model = DistributedDataParallel(build())
        train(model, *inputs(rank))
        torch.save(model.module.state_dict(), Path(folder) / (str(rank) + ".pth"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("BRATS_TEST_DISTRIBUTED") != "1", reason="Set BRATS_TEST_DISTRIBUTED=1 for two-process Gloo")
def test_zero_head_can_start_learning_under_ddp(tmp_path):
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=2, join=True)
    initial = build().stem_full.body[0].weight.detach().clone()
    model = build()
    batches = [inputs(rank) for rank in range(2)]
    train(model, *(torch.cat(parts) for parts in zip(*batches)))
    assert model.head.weight.abs().sum() > 0
    assert not torch.equal(model.stem_full.body[0].weight, initial)  # upstream learned on step two
    for rank in range(2):
        actual = torch.load(tmp_path / (str(rank) + ".pth"), weights_only=False)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(actual[name], expected)
