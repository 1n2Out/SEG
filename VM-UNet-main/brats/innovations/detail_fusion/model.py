"""VMUNet extension: residual logits from a 1/2 and 1/1 detail branch.

Options (see the package docstring): with input_stem the branch's 1/2 detail also
feeds the patch embedding; with deep_supervision, training forwards return
(logits, (aux_1/2, coarse, aux_1/8, aux_1/16)), ordered from fine to coarse, where
coarse is the backbone's own logits before the residual. Evaluation always returns
logits only, so inference and checkpoints for evaluation are unchanged.
"""
import torch

from models.vmunet.vmunet import VMUNet
from . import resolve_settings
from .block import DetailFusion


class DetailFusionVMUNet(VMUNet):
    def __init__(self, detail_fusion, **kwargs):
        settings = resolve_settings(detail_fusion)
        if settings["mode"] == "off":
            raise ValueError("Use the original VMUNet for off mode")
        super().__init__(**kwargs)
        self.deep_supervision = settings["deep_supervision"]
        self.input_stem = settings["input_stem"]
        dims = self.vmunet.dims
        # Construct the original model first. Isolate the new branch's random
        # draws so identical seeds keep baseline weights and later RNG streams.
        with torch.random.fork_rng(devices=[]):
            self.detail_fusion = DetailFusion(
                self.input_channels, dims[0], self.num_classes, settings["mode"],
                settings["full_channels"], settings["half_channels"],
                deep_supervision=self.deep_supervision, decoder_stage_channels=(dims[2], dims[1]),
                input_stem=self.input_stem, embed_channels=dims[0])

    def encode(self, x, tokens):
        """VSSM.forward_features with a residual added to the patch embedding."""
        backbone = self.vmunet
        x = backbone.patch_embed(x)
        x = x + tokens.to(dtype=x.dtype)
        if backbone.ape:
            x = x + backbone.absolute_pos_embed
        x = backbone.pos_drop(x)
        skips = []
        for layer in backbone.layers:
            skips.append(x)
            x = layer(x)
        return x, skips

    def decode(self, x, skips):
        """VSSM.forward_features_up that also returns every stage (1/32 ... 1/4)."""
        stages = []
        for index, layer in enumerate(self.vmunet.layers_up):
            x = layer(x) if index == 0 else layer(x + skips[-index])
            stages.append(x)
        return x, stages

    def forward(self, x):
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")
        branch = self.detail_fusion
        auxiliary = self.training and self.deep_supervision
        if not (self.input_stem or auxiliary):
            # The original EXP-006 path, call for call.
            encoded, skips = self.vmunet.forward_features(x)
            decoded = self.vmunet.forward_features_up(encoded, skips)
            logits = self.vmunet.forward_final(decoded)
            # Keep the baseline logits dtype under autocast; never skip the branch.
            logits = logits + branch(x, decoded).to(dtype=logits.dtype)
            return logits.sigmoid() if self.num_classes == 1 else logits
        full, half = branch.extract(x)
        if self.input_stem:
            encoded, skips = self.encode(x, branch.tokens(half))
        else:
            encoded, skips = self.vmunet.forward_features(x)
        if auxiliary:
            decoded, stages = self.decode(encoded, skips)
        else:
            decoded = self.vmunet.forward_features_up(encoded, skips)
        coarse = self.vmunet.forward_final(decoded)
        if not auxiliary:
            logits = coarse + branch.fuse(full, half, decoded).to(dtype=coarse.dtype)
            return logits.sigmoid() if self.num_classes == 1 else logits
        residual, half_logits = branch.fuse(full, half, decoded, with_auxiliary=True)
        sixteenth, eighth = branch.decoder_auxiliary((stages[1], stages[2]))
        return coarse + residual.to(dtype=coarse.dtype), (half_logits, coarse, eighth, sixteenth)
