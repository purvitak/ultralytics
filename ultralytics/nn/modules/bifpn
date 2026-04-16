import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class BiFPN_Add(nn.Module):
    """
    BiFPN weighted feature fusion node.
    Receives list of feature maps from YAML multi-input syntax.
    All inputs upsampled/downsampled to match first input spatial size.
    Output channels = c2.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.relu = nn.ReLU()
        self.conv = Conv(c1, c2, 1, 1)
        self.eps = 1e-4

    def forward(self, x):
        # x is list of tensors from multiple layers
        w = self.relu(self.w)
        w = w / (w.sum() + self.eps)
        # Resize all to first tensor's spatial size
        h, ww = x[0].shape[2:]
        out = sum(
            w[i] * F.interpolate(
                xi, size=(h, ww),
                mode="bilinear", align_corners=False
            )
            for i, xi in enumerate(x[:2])
        )
        return self.conv(out)