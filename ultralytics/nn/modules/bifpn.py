# Cell: Force overwrite BiFPN_Add module (no in-place operations)
import os
import sys
import importlib

# The corrected BiFPN_Add implementation
corrected_bifpn_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

class BiFPN_Add(nn.Module):
    """
    BiFPN weighted feature fusion node with learnable weights.
    No in-place operations on leaf variables.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c2 = c2
        # Learnable weight parameter (will be used with torch.relu, not nn.ReLU)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = 1e-4
        self.conv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        # Project inputs to target channels if needed
        projected = []
        for xi in x:
            if xi.shape[1] != self.c2:
                temp_conv = Conv(xi.shape[1], self.c2, 1, 1).to(xi.device)
                xi = temp_conv(xi)
            projected.append(xi)
        
        # CRITICAL: Use torch.relu (functional, non-in-place) NOT nn.ReLU()
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        
        # Resize all inputs to first tensor's spatial size
        h, w_dim = projected[0].shape[2:]
        out = w[0] * projected[0]
        for i in range(1, min(2, len(projected))):
            resized = F.interpolate(
                projected[i], 
                size=(h, w_dim),
                mode="bilinear", 
                align_corners=False
            )
            out = out + w[i] * resized
        
        return self.conv(out)
'''

# Paths to update
paths_to_update = [
    "/kaggle/working/ultralytics/ultralytics/nn/modules/bifpn.py",
    "/usr/local/lib/python3.12/dist-packages/ultralytics/nn/modules/bifpn.py"
]

for path in paths_to_update:
    if os.path.exists(path):
        with open(path, 'w') as f:
            f.write(corrected_bifpn_code)
        print(f"✅ Updated: {path}")
    else:
        print(f"⚠️ Not found: {path}")

# Force clear the cached module from sys.modules
if 'ultralytics.nn.modules.bifpn' in sys.modules:
    del sys.modules['ultralytics.nn.modules.bifpn']
if 'ultralytics.nn.modules' in sys.modules:
    importlib.reload(sys.modules['ultralytics.nn.modules'])

print("\n✅ BiFPN_Add module has been force-updated. No in-place ReLU operations remain.")
