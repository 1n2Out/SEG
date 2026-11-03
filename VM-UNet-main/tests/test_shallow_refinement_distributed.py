"""Opt-in two-rank Gloo check of zero-initialized residual training."""
import os
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from brats.innovations.shallow_refinement.block import ShallowRefinement


def inputs(rank):
    decoder = torch.full((1, 4, 4, 8), float(rank + 1))
    shallow = torch.linspace(-1, 1, 128).reshape(1, 4, 4, 8) + rank * 0.1
    return decoder, shallow


def worker(rank, rendezvous, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(29)
        model = DistributedDataParallel(ShallowRefinement(8, "gated", 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(*inputs(rank)).square().mean().backward()
            optimizer.step()
        torch.save(model.module.state_dict(), Path(folder) / (str(rank) + ".pth"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("BRATS_TEST_DISTRIBUTED") != "1", reason="Set BRATS_TEST_DISTRIBUTED=1 for two-process Gloo")
def test_zero_gate_can_start_learning_under_ddp(tmp_path):
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=2, join=True)
    torch.manual_seed(29)
    model = ShallowRefinement(8, "gated", 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    decoder = torch.cat((inputs(0)[0], inputs(1)[0]))
    shallow = torch.cat((inputs(0)[1], inputs(1)[1]))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(decoder, shallow).square().mean().backward()
        optimizer.step()
    assert model.alpha.abs().sum() > 0
    for rank in range(2):
        actual = torch.load(tmp_path / (str(rank) + ".pth"), weights_only=False)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(actual[name], expected)
