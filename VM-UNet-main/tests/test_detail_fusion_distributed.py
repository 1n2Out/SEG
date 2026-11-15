"""Opt-in two-rank Gloo check: the zero-initialized detail head starts learning under DDP."""
import os
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from brats.innovations.detail_fusion.block import DetailFusion


def inputs(rank):
    image = torch.linspace(-1, 1, 4 * 16 * 16).reshape(1, 4, 16, 16) + rank * 0.1
    decoder = torch.linspace(0, 1, 8).repeat(1, 4, 4, 1) + rank
    target = torch.cos(torch.linspace(0, 6, 4 * 16 * 16)).reshape(1, 4, 16, 16) * (rank + 1)
    return image, decoder, target


class AllOutputs(nn.Module):
    """One DDP forward through every option: residual, input-stem tokens and auxiliary heads."""

    def __init__(self):
        super().__init__()
        self.block = DetailFusion(4, 8, 4, "gated", 4, 8, deep_supervision=True,
                                  decoder_stage_channels=(6, 6), input_stem=True, embed_channels=6)

    def forward(self, image, decoder):
        full, half = self.block.extract(image)
        residual, auxiliary = self.block.fuse(full, half, decoder, with_auxiliary=True)
        stage = decoder[..., :6]
        deep = self.block.decoder_auxiliary((stage[:, :1, :1], stage[:, :2, :2]))
        # The tokens start at exactly zero, so they need a nonzero target to receive gradients.
        loss_terms = [(self.block.tokens(half) - 0.5).square().mean(), auxiliary.square().mean()]
        return residual, sum(loss_terms) + sum(output.square().mean() for output in deep)


def build(options=False):
    torch.manual_seed(29)
    return AllOutputs() if options else DetailFusion(4, 8, 4, "gated", 4, 8)


def train(model, image, decoder, target):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(image, decoder)
        if isinstance(output, tuple):
            residual, auxiliary = output
            ((residual - target).square().mean() + 0.1 * auxiliary).backward()
        else:
            (output - target).square().mean().backward()
        optimizer.step()


def worker(rank, rendezvous, folder, options):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        model = DistributedDataParallel(build(options))
        train(model, *inputs(rank))
        torch.save(model.module.state_dict(), Path(folder) / (str(rank) + ".pth"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("BRATS_TEST_DISTRIBUTED") != "1", reason="Set BRATS_TEST_DISTRIBUTED=1 for two-process Gloo")
@pytest.mark.parametrize("options", [False, True], ids=["branch", "options"])
def test_zero_head_can_start_learning_under_ddp(tmp_path, options):
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), options), nprocs=2, join=True)
    model = build(options)
    block = model.block if options else model
    initial = block.stem_full.body[0].weight.detach().clone()
    initial_stem = block.input_stem.block.body[0].weight.detach().clone() if options else None
    batches = [inputs(rank) for rank in range(2)]
    train(model, *(torch.cat(parts) for parts in zip(*batches)))
    assert block.head.weight.abs().sum() > 0
    assert not torch.equal(block.stem_full.body[0].weight, initial)  # upstream learned on step two
    if options:  # the zero input-stem projection opens the same way
        assert block.input_stem.project.weight.abs().sum() > 0
        assert not torch.equal(block.input_stem.block.body[0].weight, initial_stem)
    for rank in range(2):
        actual = torch.load(tmp_path / (str(rank) + ".pth"), weights_only=False)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(actual[name], expected)
