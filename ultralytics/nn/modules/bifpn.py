# =============================================================================
# bifpn.py - BiFPN_Add module (resolved merge conflict + no in-place operations)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

class BiFPN_Add(nn.Module):
    """
    BiFPN weighted feature fusion node.
    Handles multi-input feature maps from YAML syntax with learnable fusion weights.
    
    Args:
        c1: Input channels (reference channel count)
        c2: Output channels after fusion and optional convolution
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c2 = c2
        # Learnable weight for inputs (uses functional ReLU, not in-place)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = 1e-4
        # 1x1 convolution to project channels if needed
        self.conv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        """
        Forward pass with weighted fusion of input tensors.
        
        Args:
            x: List or tuple of tensors from multiple YAML layers
            
        Returns:
            Fused tensor after weighted summation and convolution
        """
        # Project inputs to have matching channels if needed
        projected = []
        for xi in x:
            if xi.shape[1] != self.c2:
                # Dynamically create conv layer for channel projection
                temp_conv = Conv(xi.shape[1], self.c2, 1, 1).to(xi.device)
                xi = temp_conv(xi)
            projected.append(xi)
        
        # Apply ReLU using torch.relu() (non-in-place) to avoid leaf variable errors
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        
        # Resize all inputs to the first tensor's spatial size
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
