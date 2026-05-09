import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """Load 1-D signal samples from .txt or .npy files.

    Each sample is returned as a tensor with shape [1, length], which is the
    expected input shape for a 1-D convolutional autoencoder.
    """

    def __init__(
        self,
        root_dir: str,
        scale: float = 1e19,
        extensions: Iterable[str] = (".txt", ".npy"),
        expected_length: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.scale = scale
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.expected_length = expected_length

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.root_dir}")

        self.file_paths = sorted(
            p for p in self.root_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.extensions
        )

        if not self.file_paths:
            raise RuntimeError(
                f"No signal files found in {self.root_dir}. "
                f"Supported extensions: {self.extensions}"
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]

        if file_path.suffix.lower() == ".npy":
            data = np.load(file_path, allow_pickle=True)
        else:
            data = np.loadtxt(file_path)

        data = np.asarray(data, dtype=np.float32).squeeze()
        if data.ndim != 1:
            raise ValueError(f"Expected a 1-D signal, but got shape {data.shape}: {file_path}")

        if self.expected_length is not None and len(data) != self.expected_length:
            raise ValueError(
                f"Unexpected signal length in {file_path}. "
                f"Expected {self.expected_length}, got {len(data)}."
            )

        data = data * self.scale
        return torch.from_numpy(data).float().unsqueeze(0)
