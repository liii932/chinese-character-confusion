"""Inspect val_graph for a repaired pilot checkpoint; never evaluate test here."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CharDataset, ROOT
from model import resnet18_gray_64


def confusion_pairs(cm, characters, min_support=5):
    records = []
    for i, entry in enumerate(characters):
        row = cm[i].copy()
        row[i] = -1
        j = int(np.argmax(row))
        count = int(row[j])
        records.append({"class_id_true": i, "char_true": entry["char"],
                        "class_id_confuse": j if count > 0 else None,
                        "char_confuse": characters[j]["char"] if count > 0 else None,
                        "count": count, "valid": count >= min_support})
    return records


def evaluate(data_root, checkpoint, output, *, plot=False):
    data_root, checkpoint, output = Path(data_root), Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh evaluation directory: {output}")
    dataset = CharDataset("val_graph", data_root)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data_hash = hashlib.sha256((data_root / "meta/manifest.json").read_bytes()).hexdigest()
    if ckpt.get("config", {}).get("data_manifest_sha256") != data_hash:
        raise ValueError("Checkpoint and dataset differ; regenerate/retrain the repaired pilot")
    characters = dataset.manifest["characters"]
    model = resnet18_gray_64(num_classes=len(characters))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cm = np.zeros((len(characters), len(characters)), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(1)
            np.add.at(cm, (labels.numpy(), predictions.numpy()), 1)
    records = confusion_pairs(cm, characters)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "confusion_matrix.npy", cm)
    (output / "confuse_pairs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"split": "val_graph", "samples": int(cm.sum()),
               "accuracy": float(np.trace(cm) / cm.sum()), "min_support": 5,
               "supported_pairs": sum(r["valid"] for r in records),
               "checkpoint_epoch": ckpt["epoch"],
               "data_manifest_sha256": data_hash,
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 11))
        chart = ax.imshow(cm, cmap="Blues")
        fig.colorbar(chart, ax=ax)
        ax.set(title="Confusion Matrix val_graph", xlabel="predict class_id", ylabel="true class_id")
        fig.tight_layout()
        fig.savefig(output / "confusion_matrix.png", dpi=200)
        plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset_81_v2")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints_a_v2/best_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints_a_v2/val_graph")
    parser.add_argument("--plot", action="store_true", help="Also render a matrix plot (requires matplotlib)")
    args = parser.parse_args()
    evaluate(args.data_root, args.checkpoint, args.output, plot=args.plot)


if __name__ == "__main__":
    main()
