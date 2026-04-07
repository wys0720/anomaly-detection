from __future__ import annotations

import numpy as np
import torch

from config import DEVICE, PathConfig, STFTConfig
from models import UNET2D
from signal_utils import load_norm_params, stft_to_tensor


def run_inference(
    model_path: str,
    norm_params_path: str,
    signal_file: str,
    noise_file: str,
    output_file: str,
    reconstructed_noise_file: str,
    stft_cfg: STFTConfig | None = None,
) -> None:
    stft_cfg = stft_cfg or STFTConfig()

    params = load_norm_params(norm_params_path)
    mean = params['mean']
    std = params['std']
    log_min = params['log_mag_min']
    log_max = params['log_mag_max']
    scale = 1.0
    epsilon = 1e-30

    model = UNET2D().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    signal = np.loadtxt(signal_file)
    noise = np.loadtxt(noise_file)
    if noise.ndim > 1:
        noise = noise[:, 0]

    n_len = len(noise)
    s_len = len(signal)
    start = (n_len - s_len) // 2
    end = start + s_len
    h_raw = noise.copy()
    h_raw[start:end] += signal
    h_norm = (h_raw - mean) / std

    h_tensor = torch.tensor(h_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        h_stft = stft_to_tensor(h_tensor, stft_cfg)
        pred = model(h_stft)
        pred_mag_norm = pred[:, 0]
        pred_phase = pred[:, 1]
        log_mag = pred_mag_norm * (log_max - log_min) + log_min
        mag_restored = (torch.expm1(log_mag) - epsilon) / scale
        mag_restored = torch.clamp(mag_restored, min=0.0)
        X_complex = torch.polar(mag_restored, pred_phase)
        window = torch.hann_window(stft_cfg.win_length).to(X_complex.device)
        n_hat = torch.istft(
            X_complex,
            n_fft=stft_cfg.n_fft,
            hop_length=stft_cfg.hop_length,
            win_length=stft_cfg.win_length,
            window=window,
            length=n_len,
        )
        n_hat_real = n_hat.squeeze().cpu().numpy() * std + mean
        g_hat = h_raw - n_hat_real

    np.savetxt(output_file, g_hat)
    np.savetxt(reconstructed_noise_file, n_hat_real)
    print(f'Residual saved to {output_file}')
    print(f'Reconstructed noise saved to {reconstructed_noise_file}')


if __name__ == '__main__':
    paths = PathConfig()
    run_inference(
        paths.model_path,
        paths.norm_params_path,
        paths.test_signal,
        paths.test_noise,
        paths.output_file,
        paths.reconstructed_noise_file,
    )
