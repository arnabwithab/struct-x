"""3D-ResNet18 control (commercial-safe path). See design sec 6/8.

Input (N,2,B,H,W): T=B frames, C=2 polarity channels. Stem surgery replaces
the 3-channel conv with a 2-channel one (weight-mean init); temporal extent is
pooled away so any B in 5..9 works.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class ResNet3DEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.backbone = r3d_18(weights=None)
        stem = self.backbone.stem[0]
        assert stem.in_channels == 3
        new_stem = nn.Conv3d(
            2,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            bias=False,
        )
        with torch.no_grad():
            new_stem.weight.copy_(stem.weight.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1, -1))
        self.backbone.stem[0] = new_stem
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))
