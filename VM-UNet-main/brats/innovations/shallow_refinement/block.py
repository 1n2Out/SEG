"""Lightweight BHWC skip injection at the final 1/4-resolution decoder stage."""
import torch
from torch import nn


class ShallowRefinement(nn.Module):
    def __init__(self, channels, mode="gated", hidden_channels=24):
        super().__init__()
        if mode not in ("simple", "gated"):
            raise ValueError("ShallowRefinement requires simple or gated mode")
        self.mode = mode
        if mode == "simple":
            self.projection = nn.Conv2d(channels, channels, 1)
        else:
            self.reduce = nn.Sequential(nn.Conv2d(channels, hidden_channels, 1), nn.GELU())
            self.local3 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3,
                                                padding=1, groups=hidden_channels), nn.GELU())
            self.local5 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 5,
                                                padding=2, groups=hidden_channels), nn.GELU())
            self.projection = nn.Conv2d(2 * hidden_channels, channels, 1)
            self.gate = nn.Conv2d(2 * channels, channels, 1)
        # Only alpha starts at zero. Zeroing both alpha and the residual would
        # block learning. Alpha learns first; the residual receives gradients
        # once alpha becomes nonzero. Parameters stay in the DDP graph throughout.
        self.alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, decoder, shallow):
        if decoder.ndim != 4 or shallow.shape != decoder.shape:
            raise ValueError("Expected matching BHWC decoder and shallow features")
        d, s = decoder.permute(0, 3, 1, 2), shallow.permute(0, 3, 1, 2)
        if self.mode == "simple":
            residual = self.projection(s)
        else:
            reduced = self.reduce(s)
            residual = self.projection(torch.cat((self.local3(reduced), self.local5(reduced)), dim=1))
            gate = self.gate(torch.cat((d, residual), dim=1)).sigmoid()
            residual = gate * residual
        # Keep the baseline activation dtype even when autocast changes the
        # convolution dtype. Do not detach or skip computation when alpha=0.
        alpha = self.alpha.to(dtype=d.dtype)[None, :, None, None]
        refined = d + alpha * residual.to(dtype=d.dtype)
        return refined.permute(0, 2, 3, 1).contiguous()
