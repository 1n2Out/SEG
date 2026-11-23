"""Group-wise fusion of spatially aligned SS2D outputs, in H+, H-, V+, V- order."""
import torch
from torch import nn
from torch.nn import functional as F


class DirectionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groups = 8
        hidden = 16
        # Use the same channel projection for each direction before taking contrasts.
        self.describe = nn.Conv1d(channels, self.groups, 1, groups=self.groups, bias=False)
        self.compress = nn.Conv2d(7 * self.groups, hidden, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv2d(hidden, hidden, 1)
        self.score = nn.Conv2d(hidden, 4 * self.groups, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    # FP16 can round away the descriptor gradients just after the zero-initialized
    # score layer's first update. Keep this branch in the scan outputs' FP32 dtype.
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, directions, height, width):
        batch, channels, length = directions[0].shape
        hf, hr, vf, vr = [self.describe(y) for y in directions]
        contrasts = ((hf - hr).abs(), (vf - vr).abs(), ((hf + hr - vf - vr) * 0.5).abs())
        features = torch.cat((hf, hr, vf, vr, *contrasts), dim=1)
        features = F.silu(self.compress(features.reshape(batch, -1, height, width)))
        context = self.context(features.mean(dim=(2, 3), keepdim=True))
        scores = self.score(F.silu(self.local(features) + context))
        weights = 4 * scores.reshape(batch, 4, self.groups, 1, length).float().softmax(dim=1)
        # Keep the baseline summation order and avoid stacking four full-width maps.
        def weighted(index):
            grouped = directions[index].reshape(batch, self.groups, -1, length)
            return (grouped * weights[:, index]).reshape(batch, channels, length)
        return weighted(0) + weighted(1) + weighted(2) + weighted(3)
