from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for signals in loader:
        signals = signals.to(device)

        optimizer.zero_grad(set_to_none=True)
        recon = model(signals)
        loss = F.mse_loss(recon, signals)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for signals in loader:
        signals = signals.to(device)
        recon = model(signals)
        loss = F.mse_loss(recon, signals)
        total_loss += loss.item() * signals.size(0)

    return total_loss / len(loader.dataset)
