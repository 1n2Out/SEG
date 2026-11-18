"""VMUNet extension: residual logits from a 1/2 and 1/1 detail branch."""
import torch

from models.vmunet.vmunet import VMUNet
from . import resolve_settings
from .block import DetailFusion


class DetailFusionVMUNet(VMUNet):
    def __init__(self, detail_fusion, freeze_base=False, **kwargs):
        settings = resolve_settings(detail_fusion)
        if settings["mode"] == "off":
            raise ValueError("Use the original VMUNet for off mode")
        super().__init__(**kwargs)
        # Construct the original model first. Isolate the new branch's random
        # draws so identical seeds keep baseline weights and later RNG streams.
        with torch.random.fork_rng(devices=[]):
            self.detail_fusion = DetailFusion(
                self.input_channels, self.vmunet.dims[0], self.num_classes, settings["mode"],
                settings["full_channels"], settings["half_channels"], settings.get("et_rollback"))
        self.freeze_base = freeze_base
        if freeze_base:
            self.requires_grad_(False)
            self.detail_fusion.et_rollback.requires_grad_(True)
            self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.vmunet.eval()
            self.detail_fusion.eval()
            self.detail_fusion.et_rollback.train(mode)
        return self

    def forward(self, x):
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")
        encoded, skips = self.vmunet.forward_features(x)
        decoded = self.vmunet.forward_features_up(encoded, skips)
        logits = self.vmunet.forward_final(decoded)
        if self.detail_fusion.et_rollback is not None:
            return self.detail_fusion(x, decoded, logits)
        # Keep the baseline logits dtype under autocast; never skip the branch.
        logits = logits + self.detail_fusion(x, decoded).to(dtype=logits.dtype)
        return logits.sigmoid() if self.num_classes == 1 else logits
