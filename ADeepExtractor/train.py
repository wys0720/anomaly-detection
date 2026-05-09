"""Training script for ADeepExtractor."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AdjacentWindowPairDataset
from metrics import compute_mismatch, compute_psnr, estimate_psd, plot_mag_phase, plot_waveform
from model import SharedCrossWindowUNet
from stft_utils import decode_spec_prediction, encode_input_features, stft_to_mag_phase


@dataclass
class TrainConfig:
    signal_dir: str
    noise_dir: str
    output_dir: str = "outputs"
    model_name: str = "best_model_adeepextractor.pth"
    norm_name: str = "norm_params_adeepextractor.json"

    epochs: int = 50
    batch_size: int = 8
    lr: float = 3e-4

    lambda_mag: float = 100.0
    lambda_phase: float = 50.0
    lambda_time: float = 300.0
    lambda_overlap: float = 200.0

    noise_scale: float = 80.0
    signal_scale: float = 0.83
    p_signal_pair: float = 0.7

    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024

    model_len: int = 28672
    signal_len: int = 20480
    noise_total_len: int = 122880
    pair_stride: int = 4096
    fs: int = 4096
    mag_scale: float = 1e20

    num_workers: int = 0
    seed: int = 2025

    @property
    def overlap_len(self) -> int:
        return self.model_len - self.pair_stride

    @property
    def model_path(self) -> str:
        return os.path.join(self.output_dir, self.model_name)

    @property
    def norm_params_path(self) -> str:
        return os.path.join(self.output_dir, self.norm_name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_global_log_mag_range_pair(dataset: AdjacentWindowPairDataset, cfg: TrainConfig, device: torch.device) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    log_min = float("inf")
    log_max = float("-inf")

    for batch in tqdm(loader, desc="Computing global log-mag range"):
        for key in ["noise1", "noise2"]:
            noise = batch[key].to(device).unsqueeze(1)
            mag, _ = stft_to_mag_phase(noise, cfg.n_fft, cfg.hop_length, cfg.win_length)
            log_mag = torch.log1p(mag * cfg.mag_scale)
            log_min = min(log_min, log_mag.min().item())
            log_max = max(log_max, log_mag.max().item())
    return log_min, log_max


def save_norm_params(dataset: AdjacentWindowPairDataset, cfg: TrainConfig, log_min: float, log_max: float) -> None:
    norm_params = {
        "mean": dataset.mean,
        "std": dataset.std,
        "log_mag_min": log_min,
        "log_mag_max": log_max,
        "mag_scale": cfg.mag_scale,
        "model_len": cfg.model_len,
        "signal_len": cfg.signal_len,
        "noise_total_len": cfg.noise_total_len,
        "pair_stride": cfg.pair_stride,
        "overlap_len": cfg.overlap_len,
        "p_signal_pair": cfg.p_signal_pair,
        "noise_scale": cfg.noise_scale,
        "signal_scale": cfg.signal_scale,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "win_length": cfg.win_length,
    }
    with open(cfg.norm_params_path, "w", encoding="utf-8") as f:
        json.dump(norm_params, f, indent=2)


def train_model(cfg: TrainConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = AdjacentWindowPairDataset(
        signal_dir=cfg.signal_dir,
        noise_dir=cfg.noise_dir,
        model_len=cfg.model_len,
        signal_len=cfg.signal_len,
        noise_total_len=cfg.noise_total_len,
        pair_stride=cfg.pair_stride,
        noise_scale=cfg.noise_scale,
        signal_scale=cfg.signal_scale,
        p_signal_pair=cfg.p_signal_pair,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Mean={dataset.mean:.6e}, Std={dataset.std:.6e}")

    log_min, log_max = compute_global_log_mag_range_pair(dataset, cfg, device)
    print(f"log_mag range: [{log_min:.6f}, {log_max:.6f}]")
    save_norm_params(dataset, cfg, log_min, log_max)

    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    model = SharedCrossWindowUNet(in_channels=2, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=True,
    )

    best_loss = float("inf")
    loss_history = {"mag": [], "phase": [], "time": [], "overlap": [], "total": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_mag = epoch_phase = epoch_time = epoch_overlap = epoch_total = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch in loop:
            mix1 = batch["mix1"].to(device).unsqueeze(1)
            mix2 = batch["mix2"].to(device).unsqueeze(1)
            noise1 = batch["noise1"].to(device).unsqueeze(1)
            noise2 = batch["noise2"].to(device).unsqueeze(1)

            in1, _, true_phase1 = encode_input_features(
                mix1, log_min, log_max, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )
            in2, _, true_phase2 = encode_input_features(
                mix2, log_min, log_max, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )
            _, true_mag_norm1, _ = encode_input_features(
                noise1, log_min, log_max, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )
            _, true_mag_norm2, _ = encode_input_features(
                noise2, log_min, log_max, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )

            pred1, pred2 = model(in1, in2)

            noise_hat1, pred_mag_norm1, pred_phase1 = decode_spec_prediction(
                pred1, log_min, log_max, cfg.model_len, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )
            noise_hat2, pred_mag_norm2, pred_phase2 = decode_spec_prediction(
                pred2, log_min, log_max, cfg.model_len, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )

            loss_mag = 0.5 * (
                F.mse_loss(pred_mag_norm1, true_mag_norm1) + F.mse_loss(pred_mag_norm2, true_mag_norm2)
            )
            loss_phase = 0.5 * (
                (1.0 - torch.mean(torch.cos(pred_phase1 - true_phase1)))
                + (1.0 - torch.mean(torch.cos(pred_phase2 - true_phase2)))
            )
            loss_time = 0.5 * (
                F.l1_loss(noise_hat1, noise1.squeeze(1)) + F.l1_loss(noise_hat2, noise2.squeeze(1))
            )
            loss_overlap = F.l1_loss(noise_hat1[:, cfg.pair_stride :], noise_hat2[:, : -cfg.pair_stride])

            loss = (
                cfg.lambda_mag * loss_mag
                + cfg.lambda_phase * loss_phase
                + cfg.lambda_time * loss_time
                + cfg.lambda_overlap * loss_overlap
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_mag += loss_mag.item()
            epoch_phase += loss_phase.item()
            epoch_time += loss_time.item()
            epoch_overlap += loss_overlap.item()
            epoch_total += loss.item()

            loop.set_postfix(
                total=f"{loss.item():.6f}",
                mag=f"{loss_mag.item():.6f}",
                phase=f"{loss_phase.item():.6f}",
                time=f"{loss_time.item():.8f}",
                overlap=f"{loss_overlap.item():.8f}",
            )

        num_batches = len(loader)
        avg_mag = epoch_mag / num_batches
        avg_phase = epoch_phase / num_batches
        avg_time = epoch_time / num_batches
        avg_overlap = epoch_overlap / num_batches
        avg_total = epoch_total / num_batches

        loss_history["mag"].append(avg_mag)
        loss_history["phase"].append(avg_phase)
        loss_history["time"].append(avg_time)
        loss_history["overlap"].append(avg_overlap)
        loss_history["total"].append(avg_total)
        scheduler.step(avg_total)

        print(
            f"[Epoch {epoch + 1}] total={avg_total:.10f}, mag={avg_mag:.10f}, "
            f"phase={avg_phase:.10f}, time={avg_time:.10f}, overlap={avg_overlap:.10f}"
        )
        print(
            f"weighted -> mag={cfg.lambda_mag * avg_mag:.10f}, "
            f"phase={cfg.lambda_phase * avg_phase:.10f}, "
            f"time={cfg.lambda_time * avg_time:.10f}, "
            f"overlap={cfg.lambda_overlap * avg_overlap:.10f}"
        )

        # Quick diagnostic on a mini-batch.
        with torch.no_grad():
            batch = next(iter(loader))
            mix1 = batch["mix1"].to(device).unsqueeze(1)
            mix2 = batch["mix2"].to(device).unsqueeze(1)
            noise1 = batch["noise1"].to(device).unsqueeze(1)
            noise1_raw = batch["noise1_raw"].numpy()
            mean = batch["mean"].numpy()
            std = batch["std"].numpy()

            in1, _, true_phase1 = encode_input_features(
                mix1, log_min, log_max, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )
            in2, _, _ = encode_input_features(
                mix2, log_min, log_max, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )
            _, true_mag_norm1, _ = encode_input_features(
                noise1, log_min, log_max, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )

            pred1, _ = model(in1, in2)
            noise_hat1, pred_mag_norm1, pred_phase1 = decode_spec_prediction(
                pred1, log_min, log_max, cfg.model_len, cfg.mag_scale, cfg.n_fft, cfg.hop_length, cfg.win_length
            )

            psnr_value = compute_psnr(pred_mag_norm1, true_mag_norm1)
            n_hat_real = noise_hat1[0].cpu().numpy() * std[0] + mean[0]
            n_true_real = noise1_raw[0]
            _, psd = estimate_psd(n_true_real, cfg.fs)
            mismatch = compute_mismatch(n_hat_real, n_true_real, psd, cfg.fs)
            print(f"PSNR(mag_norm): {psnr_value:.2f} dB")
            print(f"Mismatch(window1): {mismatch:.6e}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), cfg.model_path)
            print(f"Saved best model to {cfg.model_path}")

        if epoch == cfg.epochs - 1:
            plot_mag_phase(true_mag_norm1[:1], true_phase1[:1], os.path.join(cfg.output_dir, "true_noise_spec_pair.png"), "True Noise W1")
            plot_mag_phase(pred_mag_norm1[:1], pred_phase1[:1], os.path.join(cfg.output_dir, "pred_noise_spec_pair.png"), "Pred Noise W1")
            plot_waveform(n_true_real, n_hat_real, os.path.join(cfg.output_dir, "noise_wave_compare_pair.png"))

    # Plot training curves.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history["mag"], label="Magnitude Loss")
    plt.plot(loss_history["phase"], label="Phase Loss")
    plt.plot(loss_history["time"], label="Time Loss")
    plt.plot(loss_history["overlap"], label="Overlap Consistency Loss")
    plt.plot(loss_history["total"], label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Cross-Window Consistency Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "loss_curves_cross_window_pair.png"), dpi=150)
    plt.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ADeepExtractor.")
    parser.add_argument("--signal_dir", required=True, help="Directory containing signal .txt files.")
    parser.add_argument("--noise_dir", required=True, help="Directory containing noise .txt files.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--noise_scale", type=float, default=80.0)
    parser.add_argument("--signal_scale", type=float, default=0.83)
    parser.add_argument("--p_signal_pair", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train_model(parse_args())
