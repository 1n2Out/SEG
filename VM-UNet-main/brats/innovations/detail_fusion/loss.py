"""Auxiliary WT/TC/ET boundary supervision for the detail branch (IDEA-005).

The main CE + regional Dice loss is unchanged. Boundary targets are two-pixel
bands (3x3 dilation minus 3x3 erosion of each region mask) computed on the fly
from the augmented label, so they always match the training input.
"""
import torch
import torch.nn.functional as F

from brats.losses import BraTSCEDiceLoss

# Exclusive training labels (original 4 mapped to 3) forming WT, TC and ET.
REGION_LABELS = ((1, 2, 3), (1, 3), (3,))


def boundary_targets(target):
    """B,H,W exclusive labels -> B,3,H,W float boundary bands."""
    if target.ndim != 3:
        raise ValueError("Boundary targets require B,H,W labels")
    masks = []
    for labels in REGION_LABELS:
        mask = torch.zeros_like(target, dtype=torch.bool)
        for label in labels:
            mask = mask | (target == label)
        masks.append(mask)
    masks = torch.stack(masks, dim=1).float()
    # max_pool2d pads with -inf, so pixels outside the canvas never count.
    dilated = F.max_pool2d(masks, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-masks, 3, stride=1, padding=1)
    return dilated - eroded


class BoundaryAuxLoss(BraTSCEDiceLoss):
    """CE + regional Dice + weight * (BCE + batch soft Dice) on the boundary bands."""
    component_names = ("ce", "dice", "boundary")

    def __init__(self, weight):
        super().__init__()
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not weight > 0:
            raise ValueError("Boundary loss weight must be positive")
        self.weight = float(weight)

    def forward(self, outputs, target, return_components=False):
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise ValueError("Boundary supervision expects (logits, boundary_logits) training outputs")
        logits, boundary = outputs
        total, components = super().forward(logits, target, return_components=True)
        truth = boundary_targets(target)
        if boundary.shape != truth.shape:
            raise ValueError("Boundary logits must be B,3,H,W like the labels")
        boundary = boundary.float()
        bce = F.binary_cross_entropy_with_logits(boundary, truth)
        probability = boundary.sigmoid()
        # Batch-level Dice: single slices often contain no boundary at all.
        dims = (0, 2, 3)
        overlap = 2 * (probability * truth).sum(dim=dims) + 1
        dice = 1 - (overlap / (probability.sum(dim=dims) + truth.sum(dim=dims) + 1)).mean()
        auxiliary = bce + dice
        total = total + self.weight * auxiliary
        if return_components:
            return total, {**components, "boundary": auxiliary.detach()}
        return total
