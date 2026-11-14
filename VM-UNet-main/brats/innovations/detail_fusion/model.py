"""VMUNet extension: residual logits from a 1/2 and 1/1 detail branch."""
import torch

from models.vmunet.vmunet import VMUNet
from . import resolve_settings
from .block import DetailFusion


class DetailFusionVMUNet(VMUNet):
    """With context_slices=k the input stacks 2k+1 slices (current first); the backbone
    reads only the current slice. With a boundary weight, training forwards return
    (logits, boundary_logits); evaluation always returns logits only."""

    def __init__(self, detail_fusion, **kwargs):
        settings = resolve_settings(detail_fusion)
        if settings["mode"] == "off":
            raise ValueError("Use the original VMUNet for off mode")
        super().__init__(**kwargs)
        self.context_slices = settings["context_slices"]
        self.boundary_supervision = settings["boundary_weight"] > 0
        # Construct the original model first. Isolate the new branch's random
        # draws so identical seeds keep baseline weights and later RNG streams.
        with torch.random.fork_rng(devices=[]):
            self.detail_fusion = DetailFusion(
                self.input_channels, self.vmunet.dims[0], self.num_classes, settings["mode"],
                settings["full_channels"], settings["half_channels"],
                boundary=self.boundary_supervision, context_slices=self.context_slices)

    def forward(self, x):
        expected = self.input_channels * (2 * self.context_slices + 1)
        if x.size(1) != expected:
            raise ValueError(f"Expected {expected} input channels, got {x.size(1)}")
        center = x[:, :self.input_channels] if self.context_slices else x
        encoded, skips = self.vmunet.forward_features(center)
        decoded = self.vmunet.forward_features_up(encoded, skips)
        logits = self.vmunet.forward_final(decoded)
        if self.training and self.boundary_supervision:
            residual, boundary = self.detail_fusion(x, decoded, with_boundary=True)
            return logits + residual.to(dtype=logits.dtype), boundary
        # Keep the baseline logits dtype under autocast; never skip the branch.
        logits = logits + self.detail_fusion(x, decoded).to(dtype=logits.dtype)
        return logits.sigmoid() if self.num_classes == 1 else logits
