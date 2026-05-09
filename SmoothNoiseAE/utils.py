import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(no_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available() and not no_cuda:
        return torch.device("cuda")
    return torch.device("cpu")


def save_loss_history(path: str, train_losses: List[float], val_losses: List[float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([idx, train_loss, val_loss])


def plot_loss_curve(path: str, train_losses: List[float], val_losses: List[float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_checkpoint(path: str, model: torch.nn.Module, metadata: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
