import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class RGBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        if self.add:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c1, c2, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.add:
            b, c, h, w = x.shape
            g = self.gate(x).view(b, c, 1, 1)
            return x + y * g
        return y


class RGC2f(nn.Module):
    """Residual-Gated C2f block — Jin et al. 2025."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            RGBottleneck(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))