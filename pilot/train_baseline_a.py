"""Train the independent pilot baseline; use a fresh output directory for each run."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import CharDataset, ROOT
from model import resnet18_gray_64


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(data_root, save_dir, *, epochs=30, batch_size=64, seed=42, device="cpu"):
    data_root, save_dir = Path(data_root).resolve(), Path(save_dir).resolve()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if save_dir.exists():
        raise FileExistsError(f"Use a fresh run directory: {save_dir}")
    seed_training(seed)  # Before model initialization and shuffled loading.
    ds_train = CharDataset("train", data_root)
    ds_val = CharDataset("val_select", data_root)
    generator = torch.Generator().manual_seed(seed)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device(device)
    model = resnet18_gray_64(num_classes=81).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
    config = {
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "device": str(device),
        "lr": 0.001, "weight_decay": 1e-4, "num_classes": 81,
        "deterministic_algorithms": True, "early_stop": "val_select_accuracy == 1",
        "selection": "val_select accuracy, then lower cross entropy",
        "data_manifest_sha256": hashlib.sha256((data_root / "meta/manifest.json").read_bytes()).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((ROOT / "model.py").read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__},
    }
    save_dir.mkdir(parents=True, exist_ok=False)
    (save_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    logs, best_acc, best_loss = [], -1.0, float("inf")
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss, correct_train, total_train = 0.0, 0, 0
        for imgs, labels in loader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct_train += (logits.argmax(1) == labels).sum().item()
            total_train += imgs.size(0)
        model.eval()
        correct_val, total_val, loss_val = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in loader_val:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                correct_val += (logits.argmax(1) == labels).sum().item()
                total_val += imgs.size(0)
                loss_val += criterion(logits, labels).item() * imgs.size(0)
        val_acc, val_loss = correct_val / total_val, loss_val / total_val
        scheduler.step()
        logs.append({"epoch": epoch, "train_loss": total_loss / total_train,
                     "train_acc": correct_train / total_train, "val_select_acc": val_acc,
                     "val_select_loss": val_loss, "lr": lr,
                     "train_samples": total_train, "val_select_samples": total_val})
        # Save FIRST, including a perfect first epoch or a zero-accuracy run.
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
            temp = save_dir / "best_checkpoint.pt.tmp"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_acc": best_acc, "best_val_loss": best_loss,
                        "seed": seed, "config": config}, temp)
            temp.replace(save_dir / "best_checkpoint.pt")
        (save_dir / "log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}/{epochs}: train={logs[-1]['train_acc']:.4f}; "
              f"val_select={val_acc:.4f}; loss={val_loss:.6f}", flush=True)
        if val_acc == 1.0:
            print("Perfect validation accuracy: saved best checkpoint before early stopping.")
            break
    print(f"Best val_select accuracy: {best_acc:.4f}; outputs: {save_dir}")
    return logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset_81_v2")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints_a_v2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data_root, args.save_dir, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
