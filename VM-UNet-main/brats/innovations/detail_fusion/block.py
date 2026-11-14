"""Full/half-resolution detail branch fused with the final decoder semantics.

VM-UNet sees the image only through 4x4 patch tokens and returns to full
resolution with a linear 4x patch expansion. This branch adds the missing
1/2 and 1/1 scales: a light convolutional stem extracts detail from the input,
the last 1/4 decoder feature is upsampled by sub-pixel convolution and fused
with that detail scale by scale. The output is a residual on the original
logits. Its 1x1 head starts at zero, so the initial model equals the baseline.

In gated mode an LGAG-style gate (EMCAD, CVPR 2024) lets decoder semantics
weight each detail map before concatenation; concat mode omits only the gate.
GroupNorm replaces BatchNorm to stay independent of batch composition.

Two optional parts are built after the shared branch, so they never change its
initialization, and both start at zero, so a trained branch keeps its output:
a boundary head on the final 1/1 features (an auxiliary training target only),
and a slice-context block that adds through-plane detail from the stacked
neighbouring slices to the 1/1 detail map.
"""
import math

import torch
from torch import nn


def group_norm(channels):
    return nn.GroupNorm(math.gcd(4, channels), channels)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with a projected residual; stride on the first."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            group_norm(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            group_norm(out_channels))
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                group_norm(out_channels))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x) + self.shortcut(x))


class SubPixelUp(nn.Module):
    """1x1 projection and 2x pixel shuffle, the BCHW analogue of PatchExpand2D."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, 4 * out_channels, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = group_norm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.norm(self.shuffle(self.project(x))))


class SemanticGate(nn.Module):
    """LGAG-style gate: one spatial weight map from semantics and detail.

    As in EMCAD, both inputs pass through grouped 3x3 convolutions with C/2
    outputs and C/2 groups, are summed and rectified, and a 1x1 convolution
    gives the sigmoid weight for the detail map. GroupNorm replaces BatchNorm;
    no normalization precedes the sigmoid, so a whole slice may close its gate.
    """

    def __init__(self, semantic_channels, detail_channels):
        super().__init__()
        hidden = max(1, detail_channels // 2)
        groups = math.gcd(math.gcd(semantic_channels, detail_channels), hidden)
        self.semantic = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden, 3, padding=1, groups=groups, bias=False),
            group_norm(hidden))
        self.detail = nn.Sequential(
            nn.Conv2d(detail_channels, hidden, 3, padding=1, groups=groups, bias=False),
            group_norm(hidden))
        self.activation = nn.ReLU()
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, semantic, detail):
        weight = torch.sigmoid(self.score(self.activation(self.semantic(semantic) + self.detail(detail))))
        return detail * weight


class SliceContext(nn.Module):
    """Stacked current and neighbouring slices -> zero-initialized 1/1 detail residual."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.project = nn.Conv2d(out_channels, out_channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        return self.project(self.block(x))


class DetailFusion(nn.Module):
    def __init__(self, in_channels, decoder_channels, num_classes, mode="gated",
                 full_channels=16, half_channels=32, boundary=False, context_slices=0):
        super().__init__()
        if mode not in ("concat", "gated"):
            raise ValueError("DetailFusion requires concat or gated mode")
        if type(context_slices) is not int or context_slices < 0:
            raise ValueError("context_slices must be a nonnegative integer")
        self.mode = mode
        self.in_channels = in_channels
        self.context_slices = context_slices
        self.stem_full = ConvBlock(in_channels, full_channels)
        self.stem_half = ConvBlock(full_channels, half_channels, stride=2)
        self.up_half = SubPixelUp(decoder_channels, half_channels)
        self.fuse_half = ConvBlock(2 * half_channels, half_channels)
        self.up_full = SubPixelUp(half_channels, full_channels)
        self.fuse_full = ConvBlock(2 * full_channels, full_channels)
        if mode == "gated":
            self.gate_half = SemanticGate(half_channels, half_channels)
            self.gate_full = SemanticGate(full_channels, full_channels)
        self.head = nn.Conv2d(full_channels, num_classes, 1)
        # Only the last projection starts at zero. The head receives gradients
        # at once; the rest of the branch does after the first head update.
        # Parameters always stay in the graph, keeping DDP's parameter set fixed.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Optional parts come last: the shared branch above keeps its random draws.
        if context_slices:
            self.context = SliceContext(in_channels * (2 * context_slices + 1), full_channels)
        if boundary:
            self.boundary_head = nn.Conv2d(full_channels, 3, 1)  # WT, TC, ET boundary logits
            nn.init.zeros_(self.boundary_head.weight)
            nn.init.zeros_(self.boundary_head.bias)

    def forward(self, image, decoder, with_boundary=False):
        """Residual logits; with_boundary also returns the auxiliary boundary logits."""
        if image.ndim != 4 or decoder.ndim != 4:
            raise ValueError("Expected BCHW image and BHWC decoder features")
        batch, channels, height, width = image.shape
        if channels != self.in_channels * (2 * self.context_slices + 1):
            raise ValueError("Image channels must stack the current slice and its context slices")
        if (height % 4 or width % 4 or decoder.shape[0] != batch
                or tuple(decoder.shape[1:3]) != (height // 4, width // 4)):
            raise ValueError("Decoder features must be the 1/4-resolution BHWC map of the image")
        if with_boundary and not hasattr(self, "boundary_head"):
            raise ValueError("This branch has no boundary head")
        # The current slice always occupies the first channels (see brats.dataset.stack_slices).
        full = self.stem_full(image[:, :self.in_channels] if self.context_slices else image)
        if self.context_slices:
            full = full + self.context(image)
        half = self.stem_half(full)
        semantic = self.up_half(decoder.permute(0, 3, 1, 2))
        if self.mode == "gated":
            half = self.gate_half(semantic, half)
        semantic = self.fuse_half(torch.cat((semantic, half), dim=1))
        semantic = self.up_full(semantic)
        if self.mode == "gated":
            full = self.gate_full(semantic, full)
        semantic = self.fuse_full(torch.cat((semantic, full), dim=1))
        if with_boundary:
            return self.head(semantic), self.boundary_head(semantic)
        return self.head(semantic)
