from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.signal import welch

from config import STFTConfig


DEFAULT_STFT = STFTConfig()


def stft_to_tensor(x: torch.Tensor, cfg: STFTConfig = DEFAULT_STFT) -> torch.Tensor:
    window = torch.hann_window(cfg.win_length).to(x.device)
    x = x.squeeze(1)
    X = torch.stft(
        x,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True,
    )
    mag = X.abs()
    phase = X.angle()
    return torch.stack([mag, phase], dim=1)


def tensor_to_istft(mag_phase: torch.Tensor, cfg: STFTConfig = DEFAULT_STFT) -> torch.Tensor:
    mag, phase = mag_phase[:, 0], mag_phase[:, 1]
    X_complex = torch.polar(mag, phase)
    window = torch.hann_window(cfg.win_length).to(mag.device)
    return torch.istft(
        X_complex,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        length=cfg.signal_len,
    )


def estimate_psd(noise: np.ndarray, fs: int, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = welch(noise, fs=fs, nperseg=nperseg)
    return freqs, psd


def compute_psd_weighted_inner(
    g1: np.ndarray,
    g2: np.ndarray,
    psd: np.ndarray,
    fs: int,
    f_low: float = 20,
    f_high: float | None = None,
) -> float:
    if len(g1) != len(g2):
        raise ValueError(f'g1({len(g1)}) vs g2({len(g2)}) length mismatch')

    N = len(g1)
    G1 = np.fft.rfft(g1)
    G2 = np.fft.rfft(g2)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    mask = freqs >= f_low
    if f_high is not None:
        mask &= freqs < f_high

    psd = np.asarray(psd)
    if psd.shape[0] != G1.shape[0]:
        interp_psd = interp1d(
            np.linspace(freqs[0], freqs[-1], len(psd)),
            psd,
            bounds_error=False,
            fill_value='extrapolate',
        )
        psd = interp_psd(freqs)

    G1 = G1[mask]
    G2 = G2[mask]
    psd = psd[mask]
    df = freqs[1] - freqs[0]
    inner = 4 * np.real(np.sum(G1 * np.conj(G2) / psd)) * df
    norm1 = 4 * np.real(np.sum(G1 * np.conj(G1) / psd)) * df
    norm2 = 4 * np.real(np.sum(G2 * np.conj(G2) / psd)) * df
    return float(inner / (np.sqrt(norm1 * norm2) + 1e-12))


def compute_mismatch(
    g1: np.ndarray,
    g2: np.ndarray,
    psd: np.ndarray,
    fs: int,
    f_low: float = 20,
    f_high: float | None = None,
) -> float:
    match = compute_psd_weighted_inner(g1, g2, psd, fs, f_low, f_high)
    return float(1 - match)


def save_norm_params(path: str | Path, params: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)


def load_norm_params(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
