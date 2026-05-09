"""Dataset classes for adjacent-window ADeepExtractor training."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class AdjacentWindowPairDataset(Dataset):
    """Adjacent-window pair dataset with random signal injection.

    Each item returns two adjacent windows. The mixed windows are used as model
    inputs, and the corresponding clean noise windows are used as targets.
    """

    def __init__(
        self,
        signal_dir: str,
        noise_dir: str,
        model_len: int,
        signal_len: int,
        noise_total_len: int,
        pair_stride: int,
        noise_scale: float = 80.0,
        signal_scale: float = 0.83,
        p_signal_pair: float = 0.7,
        stats_seed: int = 1234,
    ):
        self.signal_files = sorted(glob.glob(os.path.join(signal_dir, "*.txt")))
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "*.txt")))

        if len(self.signal_files) == 0:
            raise ValueError(f"No signal files found in {signal_dir}")
        if len(self.noise_files) == 0:
            raise ValueError(f"No noise files found in {noise_dir}")

        self.model_len = model_len
        self.signal_len = signal_len
        self.noise_total_len = noise_total_len
        self.pair_stride = pair_stride
        self.noise_scale = noise_scale
        self.signal_scale = signal_scale
        self.p_signal_pair = p_signal_pair

        self.valid_pair_starts = np.arange(
            0,
            self.noise_total_len - self.model_len - self.pair_stride + 1,
            self.pair_stride,
            dtype=np.int64,
        )
        if len(self.valid_pair_starts) == 0:
            raise ValueError("No valid adjacent-window pair starts. Check lengths and stride.")

        self.mean, self.std = self._compute_dataset_stats(seed=stats_seed)

    def __len__(self) -> int:
        return len(self.noise_files)

    @staticmethod
    def _read_1d_txt(path: str) -> np.ndarray:
        arr = np.loadtxt(path).astype(np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr

    @staticmethod
    def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
        return max(a0, b0) < min(a1, b1)

    def _compute_dataset_stats(self, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        mix_list: list[np.ndarray] = []
        noise_list: list[np.ndarray] = []

        for idx in tqdm(range(len(self.noise_files)), desc="Computing mean/std"):
            noise = self._read_1d_txt(self.noise_files[idx]) * self.noise_scale
            signal = self._read_1d_txt(self.signal_files[idx % len(self.signal_files)]) * self.signal_scale
            self._validate_lengths(noise, signal)
            w1_mix, w2_mix, w1_noise, w2_noise = self._build_one_pair(noise, signal, rng)
            mix_list.extend([w1_mix, w2_mix])
            noise_list.extend([w1_noise, w2_noise])

        all_vals = np.concatenate(mix_list + noise_list)
        mean = float(all_vals.mean())
        std = float(all_vals.std() + 1e-12)
        print(f"[DEBUG] raw dataset mean = {mean:.6e}")
        print(f"[DEBUG] raw dataset std  = {std:.6e}")
        print(f"[DEBUG] p_signal_pair = {self.p_signal_pair:.2f}")
        return mean, std

    def _validate_lengths(self, noise: np.ndarray, signal: np.ndarray) -> None:
        if len(noise) != self.noise_total_len:
            raise ValueError(f"Noise length {len(noise)} != {self.noise_total_len}")
        if len(signal) != self.signal_len:
            raise ValueError(f"Signal length {len(signal)} != {self.signal_len}")

    def _build_one_pair(
        self,
        noise: np.ndarray,
        signal: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mix_full = noise.copy()

        inject_start = int(rng.integers(0, self.noise_total_len - self.signal_len + 1))
        inject_end = inject_start + self.signal_len
        mix_full[inject_start:inject_end] += signal

        overlap_candidates: list[int] = []
        pure_candidates: list[int] = []
        for s in self.valid_pair_starts:
            w1 = (int(s), int(s) + self.model_len)
            w2 = (int(s) + self.pair_stride, int(s) + self.pair_stride + self.model_len)
            w1_hit = self._interval_overlap(w1[0], w1[1], inject_start, inject_end)
            w2_hit = self._interval_overlap(w2[0], w2[1], inject_start, inject_end)
            if w1_hit or w2_hit:
                overlap_candidates.append(int(s))
            else:
                pure_candidates.append(int(s))

        if rng.random() < self.p_signal_pair and overlap_candidates:
            pair_start = int(rng.choice(overlap_candidates))
        elif pure_candidates:
            pair_start = int(rng.choice(pure_candidates))
        else:
            pair_start = int(rng.choice(self.valid_pair_starts))

        s1 = pair_start
        e1 = s1 + self.model_len
        s2 = s1 + self.pair_stride
        e2 = s2 + self.model_len

        return mix_full[s1:e1], mix_full[s2:e2], noise[s1:e1], noise[s2:e2]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        noise = self._read_1d_txt(self.noise_files[idx]) * self.noise_scale
        signal = self._read_1d_txt(self.signal_files[idx % len(self.signal_files)]) * self.signal_scale
        self._validate_lengths(noise, signal)

        rng = np.random.default_rng()
        mix1, mix2, noise1, noise2 = self._build_one_pair(noise, signal, rng)

        mix1_norm = (mix1 - self.mean) / self.std
        mix2_norm = (mix2 - self.mean) / self.std
        noise1_norm = (noise1 - self.mean) / self.std
        noise2_norm = (noise2 - self.mean) / self.std

        return {
            "mix1": torch.tensor(mix1_norm, dtype=torch.float32),
            "mix2": torch.tensor(mix2_norm, dtype=torch.float32),
            "noise1": torch.tensor(noise1_norm, dtype=torch.float32),
            "noise2": torch.tensor(noise2_norm, dtype=torch.float32),
            "noise1_raw": torch.tensor(noise1, dtype=torch.float32),
            "noise2_raw": torch.tensor(noise2, dtype=torch.float32),
            "mean": torch.tensor(self.mean, dtype=torch.float32),
            "std": torch.tensor(self.std, dtype=torch.float32),
        }
