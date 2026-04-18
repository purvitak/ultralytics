import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class BiFPN_Add(nn.Module):
    """
    BiFPN weighted feature fusion node.
    Receives list of feature maps from YAML multi-input syntax.
    Handles different channel sizes by projecting them to match c2.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c2 = c2
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.relu = nn.ReLU()
        self.eps = 1e-4
        self.conv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        # Project inputs to have matching channels if needed
        projected = []
        for xi in x:
            if xi.shape[1] != self.c2:
                temp_conv = Conv(xi.shape[1], self.c2, 1, 1).to(xi.device)
                xi = temp_conv(xi)
            projected.append(xi)
        
        # Weighted fusion
        w = self.relu(self.w)
        w = w / (w.sum() + self.eps)
        
        # Resize all to first tensor's spatial size
        h, ww = projected[0].shape[2:]
        out = w[0] * projected[0]
        for i in range(1, min(2, len(projected))):
            resized = F.interpolate(
                projected[i], 
                size=(h, ww),
                mode="bilinear", 
                align_corners=False
            )
            out = out + w[i] * resized
        
        return self.conv(out)
