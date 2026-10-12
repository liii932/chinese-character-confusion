"""ResNet18 baseline, with epoch-boundary checkpoints and explicit test gating."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import resnet18

# This module deliberately does not import the renderer: training consumes the
# immutable, checked dataset and never regenerates examples or uses test labels.
from training.io_utils import digest, write_json, verify_dataset


class GlyphDataset(Dataset):
    def __init__(self, directory, split, per_class=None):
        with np.load(Path(directory) / f"{split}.npz", allow_pickle=False) as data:
            self.images = data["images"]
            self.labels = data["labels"]
            self.condition_ids = data["condition_ids"]
        if per_class:
            indexes = np.concatenate([np.flatnonzero(self.labels == y)[:per_class]
                                      for y in np.unique(self.labels)])
            self.images, self.labels = self.images[indexes], self.labels[indexes]
            self.condition_ids = self.condition_ids[indexes]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0).float().div_(127.5).sub_(1)
        return image, int(self.labels[index]), index


def model_for(n_classes):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def loader_for(dataset, batch_size, device, seed=None):
    # No online augmentation or worker RNG. A separate per-epoch generator makes
    # the shuffle independent of evaluations and reproducible after resumption.
    generator = torch.Generator().manual_seed(0 if seed is None else seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=seed is not None,
                      num_workers=0, pin_memory=device.type == "cuda", generator=generator)


def epoch_pass(model, loader, device, n_classes, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    all_predictions = np.empty(len(loader.dataset), dtype=np.int64)
    with torch.set_grad_enabled(training):
        for images, labels, indexes in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Use AMP for training on CUDA; use fp32 for checkpoint selection.
            with torch.autocast(device_type=device.type, enabled=training and device.type == "cuda"):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; checkpoint was not advanced")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            predicted = logits.detach().argmax(1).cpu().numpy()
            truth = labels.cpu().numpy()
            np.add.at(matrix, (truth, predicted), 1)
            all_predictions[indexes.numpy()] = predicted
            total_loss += loss.item() * len(labels)
            n += len(labels)
    denominator = matrix.sum(1)
    metrics = {"loss": total_loss / n, "accuracy": float(np.trace(matrix) / n),
               "macro_accuracy": float(np.mean(np.diag(matrix)[denominator > 0] / denominator[denominator > 0])),
               "samples": n}
    return metrics, matrix, all_predictions


def train(args, manifest, labels, device):
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": args.mode, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "per_class": args.per_class,
        "model": "resnet18-gray3x3-s1-no-maxpool-random", "loss": "cross_entropy",
        "data_sha256": digest(args.data / "manifest.json"), "trainer_sha256": digest(__file__),
        "io_sha256": digest(Path(__file__).with_name("io_utils.py")),
        "environment": environment(device), "amp": device.type == "cuda",
    }
    if args.mode == "baseline" and args.per_class is not None:
        raise ValueError("Per-class truncation is only allowed for smoke/overfit modes")
    if not args.resume and any(out.iterdir()):
        raise ValueError("Run directory is nonempty. Use --resume or choose a new run name.")
    train_data = GlyphDataset(args.data, "train", args.per_class)
    validation = train_data if args.mode == "overfit" else GlyphDataset(args.data, "val_select", args.per_class)
    validation_loader = loader_for(validation, args.batch_size, device)
    model = model_for(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start, history, best_score = 0, [], (-math.inf, -math.inf)
    if args.resume:
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("Resume configuration/environment/data/code changed. Restore the original settings or use a new run.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, history, best_score = saved["epoch"], saved["history"], tuple(saved["best_score"])
        if not (out / "best.pt").exists():
            raise ValueError("best.pt is missing; restore the complete run directory")
        print(f"Resuming after completed epoch {start}", flush=True)
    write_json(out / "config.json", config)
    write_json(out / "history.json", history)
    write_json(out / "labels.json", labels)
    write_json(out / "data_manifest.json", manifest)
    stop = min(args.epochs, start + args.stop_after) if args.stop_after else args.epochs
    for epoch in range(start, stop):
        beginning = time.perf_counter()
        train_loader = loader_for(train_data, args.batch_size, device, seed=args.seed + epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_metrics, _, _ = epoch_pass(model, train_loader, device, len(labels), optimizer, scaler)
        val_metrics, _, _ = epoch_pass(model, validation_loader, device, len(labels))
        scheduler.step()
        record = {"epoch": epoch + 1, "lr": lr, "train": train_metrics,
                  "validation_split": "train-subset" if args.mode == "overfit" else "val_select",
                  "validation": val_metrics, "seconds_compute": time.perf_counter() - beginning}
        history.append(record)
        score = (val_metrics["macro_accuracy"], -val_metrics["loss"])
        if score > best_score:
            best_score = score
            save_checkpoint(out / "best.pt", {"model": model.state_dict(), "config": config,
                                               "epoch": epoch + 1, "metrics": val_metrics})
        save_checkpoint(out / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch + 1, "config": config, "history": history, "best_score": best_score,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })
        write_json(out / "history.json", history)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        print(f"Checkpoint saved: {out / 'last.pt'}; epoch wall time incl. saves: {time.perf_counter() - beginning:.1f}s", flush=True)
    print(f"Completed {len(history)}/{args.epochs} epochs. Best macro accuracy: {best_score[0]:.4f}", flush=True)


def evaluate(args, manifest, label_data, device):
    if args.split == "test" and not args.final_test:
        raise ValueError("Test is reserved. Explicit --final-test is required after the protocol is frozen.")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"]["mode"] != "baseline":
        raise ValueError("Only a baseline checkpoint can be used for development/final reports")
    if checkpoint["config"]["data_sha256"] != digest(args.data / "manifest.json"):
        raise ValueError("Checkpoint and dataset do not match")
    if checkpoint["config"]["environment"] != environment(device):
        raise ValueError("Evaluation environment differs from the training environment")
    if checkpoint["config"]["trainer_sha256"] != digest(__file__):
        raise ValueError("Evaluation code differs from the checkpoint source")
    labels = label_data["characters"]
    dataset = GlyphDataset(args.data, args.split)
    model = model_for(len(labels)).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, matrix, prediction = epoch_pass(model, loader_for(dataset, args.batch_size, device), device, len(labels))
    conditions = json.loads((args.data / "conditions.json").read_text(encoding="utf-8"))[args.split]
    sigma = np.array([conditions[int(i)]["sigma"] for i in dataset.condition_ids])
    metrics["by_sigma"] = {str(s): {"samples": int((sigma == s).sum()),
        "accuracy": float(np.mean(prediction[sigma == s] == dataset.labels[sigma == s]))}
        for s in sorted(set(sigma.tolist()))}
    metrics.update({"split": args.split, "checkpoint_epoch": checkpoint["epoch"],
                    "checkpoint_sha256": digest(args.run_dir / "best.pt")})
    char_to_id = {c["char"]: c["class_id"] for c in labels}
    pairs = []
    for pair in label_data["pairs"]:
        a, b = char_to_id[pair["a"]], char_to_id[pair["b"]]
        pairs.append({"a": pair["a"], "b": pair["b"], "a_to_b": int(matrix[a, b]),
                      "b_to_a": int(matrix[b, a]),
                      "pair_error_rate": float((matrix[a, b] + matrix[b, a]) / (matrix[a].sum() + matrix[b].sum()))})
    report_dir = args.run_dir / f"evaluation_{args.split}"
    report_dir.mkdir(exist_ok=True)
    np.save(report_dir / "confusion.npy", matrix)
    write_json(report_dir / "metrics.json", metrics)
    write_json(report_dir / "pairs.json", pairs)
    with (report_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "true_id", "true_char", "predicted_id", "predicted_char", "sigma", "dx", "dy"])
        for index, (truth, pred) in enumerate(zip(dataset.labels, prediction)):
            condition = conditions[int(dataset.condition_ids[index])]
            writer.writerow([index, truth, labels[int(truth)]["char"], pred, labels[int(pred)]["char"],
                             condition["sigma"], condition["dx"], condition["dy"]])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "overfit", "baseline"], default="baseline")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--stop-after", type=int, help="Stop after N additional epochs without changing the planned schedule")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Only for local verification")
    parser.add_argument("--split", choices=["val_select", "val_graph", "test"], default="val_select")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or (args.per_class is not None and args.per_class <= 0):
        parser.error("epochs, batch-size and per-class must be positive")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("stop-after must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError("No CUDA GPU. In Colab select Runtime > Change runtime type > GPU, then reconnect.")
    if device.type == "cpu":
        torch.set_num_threads(2)
    seed_everything(args.seed)
    manifest = verify_dataset(args.data)
    label_data = json.loads((args.data / "labels.json").read_text(encoding="utf-8"))
    print(json.dumps(environment(device), indent=2), flush=True)
    if args.command == "train":
        train(args, manifest, label_data["characters"], device)
    else:
        evaluate(args, manifest, label_data, device)


if __name__ == "__main__":
    main()
