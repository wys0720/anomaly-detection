import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothNoiseAE(nn.Module):
    """A lightweight 1-D convolutional autoencoder for stationary noise reconstruction.

    The network follows the structure of the original WhiteNoiseAE script:
    three downsampling encoder blocks, one non-downsampling bottleneck block,
    and three transposed-convolution decoder blocks.
    """

    def __init__(self, input_length: int = 4096):
        super().__init__()
        self.input_length = input_length

        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.dec4 = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    @staticmethod
    def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Crop or pad the last dimension to match the input signal length."""
        current_length = x.shape[-1]
        if current_length == target_length:
            return x
        if current_length > target_length:
            return x[..., :target_length]
        return F.pad(x, (0, target_length - current_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        out = self.dec4(d3)

        return self._match_length(out, x.shape[-1])


# Backward-compatible alias for the original class name.
WhiteNoiseAE = SmoothNoiseAE
