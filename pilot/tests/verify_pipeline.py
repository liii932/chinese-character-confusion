"""Bounded CPU integration check; writes only to a fresh verification directory.

Run after build_dataset.py. Uses two real images per class in train/val_select,
one epoch of the real ResNet18, then all val_graph images. Never reads test pixels.
These outputs are plumbing checks, not a completed baseline experiment.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import CharDataset
from model import resnet18_gray_64
import train_baseline_a as trainer
from eval_confusion import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset_81_v2")
    parser.add_argument("--output", type=Path, default=ROOT / ".verification/pipeline")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    def subset(split, data_root):
        assert split in {"train", "val_select"}
        dataset = CharDataset(split, data_root)
        counts, indices = {}, []
        for index, rec in enumerate(dataset.records):
            cid = rec["class_id"]
            if counts.get(cid, 0) < 2:
                indices.append(index)
                counts[cid] = counts.get(cid, 0) + 1
        assert len(counts) == 81 and set(counts.values()) == {2}
        return Subset(dataset, indices)

    # Check real-model initialization in addition to the small-model unit tests.
    trainer.seed_training(42)
    expected = {k: v.clone() for k, v in resnet18_gray_64().state_dict().items()}
    trainer.seed_training(42)
    actual = resnet18_gray_64().state_dict()
    assert all(torch.equal(v, actual[k]) for k, v in expected.items())
    del expected, actual

    with patch.object(trainer, "CharDataset", side_effect=subset):
        logs = trainer.train(args.data_root, args.output / "smoke", epochs=1, batch_size=16)
    ckpt_path = args.output / "smoke/best_checkpoint.pt"
    ckpt = torch.load(ckpt_path, weights_only=True)
    assert logs[0]["train_samples"] == logs[0]["val_select_samples"] == 162
    assert ckpt["epoch"] == 1 and ckpt["seed"] == 42
    assert len(ckpt["optimizer_state_dict"]["state"]) > 0
    summary = evaluate(args.data_root, ckpt_path, args.output / "val_graph")
    assert summary["split"] == "val_graph" and summary["samples"] == 648
    report = {"status": "passed", "purpose": "pipeline verification, not final accuracy",
              "device": "cpu", "real_model": "resnet18_gray_64",
              "initialization_repeat_equal": True, "train_samples": 162,
              "val_select_samples": 162, "val_graph_samples": 648,
              "epochs": 1, "test_evaluated": False,
              "checkpoint_reload": True, "logs": logs}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS: real ResNet18 forward/backward, checkpoint reload and val_graph evaluation")


if __name__ == "__main__":
    main()
