"""Deep supervision for the detail-fusion model (IDEA-007).

Every supervised output uses the unchanged CE + regional Dice. Weights halve per
scale, as in nnU-Net: the final output 1, the 1/2 branch head 1/2, the backbone's
coarse logits (1/4 tokens) 1/4, and the 1/8 and 1/16 decoder heads 1/8 and 1/16.
nnU-Net additionally normalizes the weights to sum 1, a constant factor that
AdamW's update is insensitive to. Auxiliary logits are bilinearly upsampled to the
label size (as mmsegmentation does for auxiliary heads), so thin ET labels are
never aliased by label downsampling.
"""
import torch.nn.functional as F

from brats.losses import BraTSCEDiceLoss

# Relative to the final output; order of the model's auxiliary outputs (fine -> coarse).
AUXILIARY_WEIGHTS = (0.5, 0.25, 0.125, 0.0625)


class DeepSupervisionLoss(BraTSCEDiceLoss):
    """Main CE + Dice plus the weighted auxiliary terms, logged together as "deep"."""
    component_names = ("ce", "dice", "deep")

    def forward(self, outputs, target, return_components=False):
        if (not isinstance(outputs, tuple) or len(outputs) != 2
                or len(outputs[1]) != len(AUXILIARY_WEIGHTS)):
            raise ValueError("Deep supervision expects (logits, four auxiliary logits) training outputs")
        logits, auxiliary = outputs
        total, components = super().forward(logits, target, return_components=True)
        size = tuple(target.shape[-2:])
        deep = 0.0
        for weight, output in zip(AUXILIARY_WEIGHTS, auxiliary):
            if output.ndim != 4 or output.shape[:2] != logits.shape[:2]:
                raise ValueError("Auxiliary logits must be B,4,h,w like the main logits")
            if tuple(output.shape[-2:]) != size:
                output = F.interpolate(output.float(), size=size, mode="bilinear", align_corners=False)
            deep = deep + weight * super().forward(output, target)
        total = total + deep
        if return_components:
            return total, {**components, "deep": deep.detach()}
        return total
