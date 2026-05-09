import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import SignalDataset
from engine import train_one_epoch, validate
from model import SmoothNoiseAE
from utils import (
    get_device,
    plot_loss_curve,
    save_checkpoint,
    save_loss_history,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmoothNoiseAE for 1-D noise reconstruction.")

    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing .txt or .npy signal files.")
    parser.add_argument("--output-dir", type=str, default="outputs/smoothae", help="Directory for checkpoints and logs.")
    parser.add_argument("--input-length", type=int, default=4096, help="Expected signal length.")
    parser.add_argument("--scale", type=float, default=1e19, help="Scaling factor applied to input signals.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0, help="Set torch CPU threads. Use 0 to keep PyTorch default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true", help="Force training on CPU.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(no_cuda=args.no_cuda)
    print(f"Using device: {device}")

    dataset = SignalDataset(
        root_dir=args.data_dir,
        scale=args.scale,
        expected_length=args.input_length,
    )

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Dataset is too small for the requested train/validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SmoothNoiseAE(input_length=args.input_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                metadata={
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "input_length": args.input_length,
                    "scale": args.scale,
                    "model_name": "SmoothNoiseAE",
                },
            )
            print("Best model saved.")

        save_loss_history(output_dir / "loss_history.csv", train_losses, val_losses)
        plot_loss_curve(output_dir / "loss_curve.png", train_losses, val_losses)

    print(f"Training finished. Best validation loss: {best_val_loss:.6f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
