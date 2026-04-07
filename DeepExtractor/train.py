from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DEVICE, PathConfig, STFTConfig, TrainConfig
from dataset import SignalNoiseDataset
from models import UNET2D
from signal_utils import (
    compute_mismatch,
    estimate_psd,
    save_norm_params,
    stft_to_tensor,
)


def check_gradients(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f'No gradient for {name}')
        else:
            print(f'Gradient norm for {name}: {param.grad.norm().item()}')


def train_model(
    signal_dir: str,
    noise_dir: str,
    model_path: str,
    norm_params_path: str,
    train_cfg: TrainConfig | None = None,
    stft_cfg: STFTConfig | None = None,
) -> None:
    train_cfg = train_cfg or TrainConfig()
    stft_cfg = stft_cfg or STFTConfig()

    dataset = SignalNoiseDataset(signal_dir, noise_dir, use_signal_for_stats=False)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    model = UNET2D().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    best_loss = float('inf')
    loss_history = {'mag': [], 'phase': [], 'total': []}

    for epoch in range(train_cfg.epochs):
        model.train()
        total_loss = 0.0

        for h, n, g, mean, std in tqdm(loader, desc=f'Epoch {epoch + 1}/{train_cfg.epochs}'):
            h, n, g = h.to(DEVICE), n.to(DEVICE), g.to(DEVICE)
            mean = mean.to(DEVICE)
            std = std.to(DEVICE)
            h = h.unsqueeze(1)
            n = n.unsqueeze(1)

            h_stft = stft_to_tensor(h, stft_cfg)
            n_stft = stft_to_tensor(n, stft_cfg)
            pred = model(h_stft)

            pred_mag, pred_phase = pred[:, 0], pred[:, 1]
            true_mag_raw, true_phase = n_stft[:, 0], n_stft[:, 1]

            epsilon = 1e-30
            scale = 1.0
            true_mag_log = torch.log1p(true_mag_raw * scale + epsilon)
            log_min = true_mag_log.min().item()
            log_max = true_mag_log.max().item()
            true_mag_norm = (true_mag_log - log_min) / (log_max - log_min + 1e-9)

            pred_mag = torch.clamp(pred_mag, 0.0, 1.0)
            loss_mag = torch.nn.functional.mse_loss(pred_mag, true_mag_norm)
            phase_diff = pred_phase - true_phase
            loss_phase = 1 - torch.mean(torch.cos(phase_diff))
            loss = loss_mag * train_cfg.loss_mag_weight + loss_phase * train_cfg.loss_phase_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / len(loader)
        loss_history['mag'].append(float(loss_mag.item()))
        loss_history['phase'].append(float(loss_phase.item()))
        loss_history['total'].append(avg_loss)

        print(f'Epoch {epoch + 1} - Loss: {avg_loss:.6f}')
        print(f'Magnitude Loss: {loss_mag.item():.6f}, Phase Loss: {loss_phase.item():.6f}')
        print(f'Pred Mag Mean: {pred_mag.mean().item()}, True Mag Mean: {true_mag_norm.mean().item()}')
        print(f'Pred phase Mean: {pred_phase.mean().item()}, True phase Mean: {true_phase.mean().item()}')

        with torch.no_grad():
            pred_demo = model(h_stft[:1])
            pred_mag_demo, pred_phase_demo = pred_demo[:, 0], pred_demo[:, 1]
            log_mag = pred_mag_demo * (log_max - log_min) + log_min
            mag_restored = (torch.expm1(log_mag) - epsilon) / scale
            mag_restored = torch.clamp(mag_restored, min=0.0)
            X_complex = torch.polar(mag_restored, pred_phase_demo)
            window = torch.hann_window(stft_cfg.win_length).to(X_complex.device)
            n_hat = torch.istft(
                X_complex,
                n_fft=stft_cfg.n_fft,
                hop_length=stft_cfg.hop_length,
                win_length=stft_cfg.win_length,
                window=window,
                length=stft_cfg.signal_len,
            )
            n_hat_real = n_hat.squeeze().cpu().numpy() * std[0].cpu().numpy() + mean[0].cpu().numpy()
            n_true = n[0].squeeze().cpu().numpy() * std[0].cpu().numpy() + mean[0].cpu().numpy()
            _, psd = estimate_psd(n_true, stft_cfg.sample_rate)
            mismatch = compute_mismatch(n_hat_real, n_true, psd, stft_cfg.sample_rate)
            print(f'Epoch {epoch + 1} - mismatch: {mismatch:.6e}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            save_norm_params(norm_params_path, {
                'mean': dataset.mean,
                'std': dataset.std,
                'log_mag_min': log_min,
                'log_mag_max': log_max,
            })
            print(f'Best model saved to {model_path}')

    plt.figure()
    plt.plot(loss_history['mag'], label='Magnitude Loss')
    plt.plot(loss_history['phase'], label='Phase Loss')
    plt.plot(loss_history['total'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    Path('./outputs').mkdir(exist_ok=True)
    plt.savefig('./outputs/loss_curves.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    paths = PathConfig()
    train_model(paths.signal_dir, paths.noise_dir, paths.model_path, paths.norm_params_path)
