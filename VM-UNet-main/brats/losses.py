import torch
import torch.nn.functional as F
from torch import nn


class BraTSCEDiceLoss(nn.Module):
    """Four exclusive logits: CE plus Dice on overlapping WT/TC/ET regions."""
    def forward(self, logits, target, return_components=False):
        if logits.ndim != 4 or logits.shape[1] != 4:
            raise ValueError("BraTS loss requires B,4,H,W logits")
        probabilities = logits.float().softmax(dim=1)
        dice_losses = []
        for labels in ((1, 2, 3), (1, 3), (3,)):
            prediction = probabilities[:, labels].sum(dim=1)
            truth = torch.zeros_like(target, dtype=torch.bool)
            for label in labels:
                truth = truth | (target == label)
            truth = truth.float()
            dims = (1, 2)
            numerator = 2 * (prediction * truth).sum(dim=dims) + 1e-5
            denominator = prediction.sum(dim=dims) + truth.sum(dim=dims) + 1e-5
            dice_losses.append(1 - (numerator / denominator).mean())
        ce = F.cross_entropy(logits.float(), target.long())
        dice = torch.stack(dice_losses).mean()
        total = ce + dice
        if return_components:
            return total, {"ce": ce.detach(), "dice": dice.detach()}
        return total
