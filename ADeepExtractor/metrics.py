"""Metrics and plotting helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import welch


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    mse = F.mse_loss(pred, target)
    max_val = torch.max(target).clamp_min(eps)
    return (20 * torch.log10(max_val / torch.sqrt(mse + eps))).item()


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
        raise ValueError(f"g1({len(g1)}) vs g2({len(g2)}) length mismatch")

    N = len(g1)
    G1 = np.fft.rfft(g1)
    G2 = np.fft.rfft(g2)
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    mask = freqs >= f_low
    if f_high is not None:
        mask &= freqs < f_high

    psd = np.asarray(psd)
    if psd.shape[0] != G1.shape[0]:
        from scipy.interpolate import interp1d

        interp_psd = interp1d(
            np.linspace(freqs[0], freqs[-1], len(psd)),
            psd,
            bounds_error=False,
            fill_value="extrapolate",
        )
        psd = interp_psd(freqs)

    G1 = G1[mask]
    G2 = G2[mask]
    psd = np.maximum(psd[mask], 1e-20)

    df = freqs[1] - freqs[0]
    inner = 4 * np.real(np.sum(G1 * np.conj(G2) / psd)) * df
    norm1 = 4 * np.real(np.sum(G1 * np.conj(G1) / psd)) * df
    norm2 = 4 * np.real(np.sum(G2 * np.conj(G2) / psd)) * df
    return float(inner / (np.sqrt(norm1 * norm2) + 1e-12))


def compute_mismatch(g1: np.ndarray, g2: np.ndarray, psd: np.ndarray, fs: int) -> float:
    return 1.0 - compute_psd_weighted_inner(g1, g2, psd, fs)


def plot_mag_phase(mag: torch.Tensor, phase: torch.Tensor, save_path: str, title_prefix: str = "") -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mag[0].detach().cpu().numpy(), aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(f"{title_prefix} Magnitude")

    plt.subplot(1, 2, 2)
    plt.imshow(phase[0].detach().cpu().numpy(), aspect="auto", origin="lower", cmap="twilight")
    plt.colorbar()
    plt.title(f"{title_prefix} Phase")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_waveform(original: np.ndarray, reconstructed: np.ndarray, save_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(reconstructed, linewidth=1)
    plt.title("Reconstructed Noise")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(original, linewidth=1)
    plt.title("True Noise")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
