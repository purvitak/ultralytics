# =============================================================================
# bifpn.py - Corrected BiFPN_Add module (no in-place operations on leaf variables)
# =============================================================================

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class BiFPN_Add(nn.Module):
    """
<<<<<<< HEAD
    Bidirectional Feature Pyramid Network (BiFPN) weighted fusion module.
    Combines two input feature maps with learnable fusion weights.
    
    Args:
        c1: Input channels (same for both inputs)
        c2: Output channels after fusion and optional convolution
    """
    def __init__(self, c1, c2):
        super().__init__()
        # Learnable weight for two inputs (no in-place operations will be used)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.epsilon = 0.0001  # Small constant for numerical stability
        # Optional 1x1 convolution to adjust channels if needed
        self.conv = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        """
        Forward pass with weighted fusion of two input tensors.
        
        Args:
            x: List or tuple of two tensors [x1, x2] to be fused
            
        Returns:
            Fused tensor after weighted summation and optional convolution
        """
        # Apply ReLU non-linearity WITHOUT in-place operation
        # Using torch.relu() instead of nn.ReLU() avoids in-place modification
        w = torch.relu(self.w)
        
        # Normalize weights to sum to 1 (softmax-style without softmax)
        weight = w / (torch.sum(w) + self.epsilon)
        
        # Weighted sum of the two input tensors
        fused = weight[0] * x[0] + weight[1] * x[1]
        
        # Apply optional convolution and return
        return self.conv(fused)
=======
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
>>>>>>> d6e29a949724b31d732b2e801e841557e61b03d3
