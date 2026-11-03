"""VMUNet extension with unchanged backbone names, pretraining and logits API."""
import torch

from models.vmunet.vmunet import VMUNet
from . import resolve_settings
from .block import ShallowRefinement


class ShallowRefinementVMUNet(VMUNet):
    def __init__(self, shallow_refinement, **kwargs):
        settings = resolve_settings(shallow_refinement)
        if settings["mode"] == "off":
            raise ValueError("Use the original VMUNet for off mode")
        super().__init__(**kwargs)
        # Construct the original model first. Isolate initialization of the new
        # branch so identical seeds preserve baseline weights and subsequent RNG.
        with torch.random.fork_rng(devices=[]):
            self.shallow_refinement = ShallowRefinement(
                self.vmunet.dims[0], settings["mode"], settings["hidden_channels"])

    def forward(self, x):
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")
        encoded, skips = self.vmunet.forward_features(x)
        decoded = self.vmunet.forward_features_up(encoded, skips)
        decoded = self.shallow_refinement(decoded, skips[0])
        logits = self.vmunet.forward_final(decoded)
        return logits.sigmoid() if self.num_classes == 1 else logits
