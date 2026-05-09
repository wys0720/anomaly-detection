"""STFT/iSTFT feature utilities for ADeepExtractor."""

from __future__ import annotations

import numpy as np
import torch


def stft_complex(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Return complex STFT of a batch of time-domain signals."""
    if x.dim() == 3:
        x = x.squeeze(1)
    window = torch.hann_window(win_length, device=x.device)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )


def stft_to_mag_phase(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    X = stft_complex(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return X.abs(), X.angle()


def tensor_to_istft_from_mag_phase(
    mag: torch.Tensor,
    phase: torch.Tensor,
    length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    X_complex = torch.polar(mag, phase)
    window = torch.hann_window(win_length, device=mag.device)
    return torch.istft(
        X_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def encode_input_features(
    x_time: torch.Tensor,
    log_min: float,
    log_max: float,
    mag_scale: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode time signal into [log_mag_norm, phase_norm] model input."""
    mag, phase = stft_to_mag_phase(
        x_time,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    log_mag = torch.log1p(mag * mag_scale)
    log_mag_norm = (log_mag - log_min) / (log_max - log_min + 1e-9)
    phase_norm = phase / np.pi
    model_in = torch.stack([log_mag_norm, phase_norm], dim=1)
    return model_in, log_mag_norm, phase


def decode_spec_prediction(
    pred_spec: torch.Tensor,
    log_min: float,
    log_max: float,
    length: int,
    mag_scale: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode predicted spectrogram channels into time-domain noise."""
    pred_mag_norm = torch.clamp(pred_spec[:, 0], 0.0, 1.0)
    pred_phase_norm = torch.clamp(pred_spec[:, 1], -1.0, 1.0)

    pred_log_mag = pred_mag_norm * (log_max - log_min) + log_min
    pred_mag = (torch.expm1(pred_log_mag) / mag_scale).clamp_min(0.0)
    pred_phase = pred_phase_norm * np.pi

    x_hat = tensor_to_istft_from_mag_phase(
        pred_mag,
        pred_phase,
        length=length,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    return x_hat, pred_mag_norm, pred_phase
