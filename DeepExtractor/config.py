from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class STFTConfig:
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024
    signal_len: int = 20480
    sample_rate: int = 4096


@dataclass
class TrainConfig:
    epochs: int = 1500
    batch_size: int = 32
    learning_rate: float = 3e-4
    loss_mag_weight: float = 100.0
    loss_phase_weight: float = 100.0


@dataclass
class PathConfig:
    signal_dir: str = './data/signals'
    noise_dir: str = './data/noise_train'
    output_dir: str = './outputs'
    model_path: str = './outputs/best_model.pth'
    norm_params_path: str = './outputs/norm_params.json'
    test_signal: str = './data/test_signal.txt'
    test_noise: str = './data/test_noise.txt'
    output_file: str = './outputs/reconstruction.txt'
    reconstructed_noise_file: str = './outputs/reconstructed_noise.txt'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
