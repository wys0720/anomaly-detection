from __future__ import annotations

from pathlib import Path
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SignalNoiseDataset(Dataset):
    """Dataset for center-cropped signal+noise mixtures and target noise."""

    def __init__(
        self,
        signal_dir: str,
        noise_dir: str,
        noise_len: int = 122880,
        signal_len: int = 20480,
        input_len: int = 20480,
        use_signal_for_stats: bool = False,
    ) -> None:
        self.signal_files = sorted(glob.glob(os.path.join(signal_dir, '*.txt')))
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, '*.txt')))
        if not self.signal_files:
            raise FileNotFoundError(f'No signal files found in {signal_dir}')
        if not self.noise_files:
            raise FileNotFoundError(f'No noise files found in {noise_dir}')

        self.noise_len = noise_len
        self.signal_len = signal_len
        self.input_len = input_len
        self.use_signal_for_stats = use_signal_for_stats

        self.mean, self.std = self._compute_stats()

    def _compute_stats(self) -> tuple[float, float]:
        all_series: list[np.ndarray] = []
        for idx, noise_path in enumerate(self.noise_files):
            noise = np.loadtxt(noise_path)
            if len(noise) != self.noise_len:
                raise ValueError(f'{noise_path} length {len(noise)} != {self.noise_len}')

            series = noise.copy()
            if self.use_signal_for_stats:
                signal_path = self.signal_files[idx % len(self.signal_files)]
                signal = np.loadtxt(signal_path)
                if len(signal) != self.signal_len:
                    raise ValueError(f'{signal_path} length {len(signal)} != {self.signal_len}')
                start = (self.noise_len - self.signal_len) // 2
                end = start + self.signal_len
                series[start:end] += signal
            all_series.append(series)

        concat = np.concatenate(all_series)
        return float(concat.mean()), float(concat.std() + 1e-9)

    def __len__(self) -> int:
        return len(self.noise_files)

    def __getitem__(self, idx: int):
        noise_path = self.noise_files[idx % len(self.noise_files)]
        signal_path = self.signal_files[idx % len(self.signal_files)]

        noise = np.loadtxt(noise_path)
        signal = np.loadtxt(signal_path)
        if len(noise) != self.noise_len:
            raise ValueError(f'{noise_path} length {len(noise)} != {self.noise_len}')
        if len(signal) != self.signal_len:
            raise ValueError(f'{signal_path} length {len(signal)} != {self.signal_len}')

        mix = noise.copy()
        noise_center = self.noise_len // 2
        sig_start = noise_center - self.signal_len // 2
        sig_end = sig_start + self.signal_len
        mix[sig_start:sig_end] += signal

        in_center = self.noise_len // 2
        in_start = in_center - self.input_len // 2
        in_end = in_start + self.input_len

        mix_crop = mix[in_start:in_end]
        noise_crop = noise[in_start:in_end]
        signal_crop = signal.copy()

        mix_crop_norm = (mix_crop - self.mean) / self.std
        noise_crop_norm = (noise_crop - self.mean) / self.std

        return (
            torch.tensor(mix_crop_norm, dtype=torch.float32),
            torch.tensor(noise_crop_norm, dtype=torch.float32),
            torch.tensor(signal_crop, dtype=torch.float32),
            torch.tensor(self.mean, dtype=torch.float32),
            torch.tensor(self.std, dtype=torch.float32),
        )
