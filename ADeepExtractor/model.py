"""Model definition for ADeepExtractor.

ADeepExtractor is a shared-encoder dual-branch U-Net with:
1) adjacent-window joint input,
2) bottleneck cross-window feature fusion,
3) attention-gated skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    """Two Conv2d-BatchNorm-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for U-Net skip connections.

    g: decoder feature after upsampling.
    x: encoder skip feature.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CrossWindowFusion2D(nn.Module):
    """Lightweight bottleneck-level cross-window feature fusion.

    The two bottleneck features are concatenated along the channel dimension,
    transformed by 1x1 and 3x3 convolutions, split back into two residual terms,
    and added to the original features.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([z1, z2], dim=1)
        z = self.fuse(z)
        dz1, dz2 = torch.chunk(z, 2, dim=1)
        return z1 + dz1, z2 + dz2


class SharedCrossWindowUNet(nn.Module):
    """ADeepExtractor main network.

    Parameters
    ----------
    in_channels:
        Number of input channels. Default 2: log-magnitude and phase.
    out_channels:
        Number of output channels. Default 2: predicted magnitude and phase.
    features:
        Channel sizes for the U-Net encoder.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv2D(ch, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            ch = feat

        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)
        self.cross_fusion = CrossWindowFusion2D(features[-1] * 2)

        decoder_in_ch = features[-1] * 2
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(decoder_in_ch, feat, kernel_size=2, stride=2))
            self.att_gates.append(AttentionGate(F_g=feat, F_l=feat, F_int=max(feat // 2, 1)))
            self.up_blocks.append(DoubleConv2D(feat * 2, feat))
            decoder_in_ch = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def encode_one(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        return x, skips

    def decode_one(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        skips = skips[::-1]
        for up, att, block, skip in zip(self.ups, self.att_gates, self.up_blocks, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            gated_skip = att(x, skip)
            x = torch.cat([gated_skip, x], dim=1)
            x = block(x)
        return self.final_conv(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1, skips1 = self.encode_one(x1)
        z2, skips2 = self.encode_one(x2)
        z1, z2 = self.cross_fusion(z1, z2)
        out1 = self.decode_one(z1, skips1)
        out2 = self.decode_one(z2, skips2)
        return out1, out2
