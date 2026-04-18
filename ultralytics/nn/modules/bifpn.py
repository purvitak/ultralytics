# =============================================================================
# bifpn.py - Corrected BiFPN_Add module (no in-place operations on leaf variables)
# =============================================================================

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class BiFPN_Add(nn.Module):
    """
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