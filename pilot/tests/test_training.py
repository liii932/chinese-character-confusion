"""Regression tests for checkpoint selection and deterministic training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_baseline_a as trainer
from eval_confusion import confusion_pairs, evaluate


class ConstantModel(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([20.] + [0.] * (num_classes - 1)))

    def forward(self, images):
        return self.bias.unsqueeze(0).expand(len(images), -1)


class LaterPerfectModel(ConstantModel):
    def __init__(self, num_classes=81):
        super().__init__(num_classes)
        self.evaluations = 0

    def forward(self, images):
        logits = super().forward(images)
        if not self.training:
            self.evaluations += 1
            if self.evaluations == 1:
                return logits.roll(1, dims=1)
        return logits


class TinyModel(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, images):
        return self.fc(images.mean((2, 3)))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        (ROOT / ".verification").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".verification")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "meta").mkdir()
        (self.root / "meta/manifest.json").write_text("{}", encoding="utf-8")

    def run_case(self, name, labels, model, seed=42, epochs=3, batch_size=4):
        images = torch.arange(4 * 64 * 64, dtype=torch.float32).reshape(4, 1, 64, 64) / 16384
        dataset = TensorDataset(images, torch.tensor(labels))
        def load(split, root):
            self.assertIn(split, {"train", "val_select"})
            return dataset
        output = self.root / name
        with patch.object(trainer, "CharDataset", side_effect=load), \
             patch.object(trainer, "resnet18_gray_64", model):
            logs = trainer.train(self.root, output, epochs=epochs, batch_size=batch_size, seed=seed)
        checkpoint = torch.load(output / "best_checkpoint.pt", weights_only=True)
        self.assertEqual(json.loads((output / "log.json").read_text()), logs)
        self.assertEqual(checkpoint["seed"], seed)
        self.assertEqual(checkpoint["config"]["seed"], seed)
        self.assertIn("scheduler_state_dict", checkpoint)
        return logs, checkpoint

    def test_perfect_first_epoch_is_saved_before_stopping(self):
        logs, ckpt = self.run_case("perfect", [0] * 4, ConstantModel)
        self.assertEqual(len(logs), 1)
        self.assertEqual((ckpt["epoch"], ckpt["best_val_acc"]), (1, 1.0))
        reloaded = ConstantModel()
        reloaded.load_state_dict(ckpt["model_state_dict"])
        self.assertTrue(torch.all(reloaded(torch.zeros(4, 1, 64, 64)).argmax(1) == 0))

    def test_later_perfect_epoch_replaces_previous_best(self):
        logs, ckpt = self.run_case("later", [0] * 4, LaterPerfectModel)
        self.assertEqual([r["val_select_acc"] for r in logs], [0.0, 1.0])
        self.assertEqual((ckpt["epoch"], ckpt["best_val_acc"]), (2, 1.0))

    def test_zero_accuracy_still_saves_and_breaks_ties_by_loss(self):
        logs, ckpt = self.run_case("zero", [1] * 4, ConstantModel)
        self.assertEqual(len(logs), 3)
        self.assertTrue(all(r["val_select_acc"] == 0 for r in logs))
        self.assertEqual(ckpt["epoch"], 3)
        self.assertEqual(ckpt["best_val_loss"], min(r["val_select_loss"] for r in logs))

    def test_training_is_reproducible_and_seed_has_effect(self):
        log_a, a = self.run_case("seed_a", [0, 1, 2, 3], TinyModel, batch_size=2)
        log_b, b = self.run_case("seed_b", [0, 1, 2, 3], TinyModel, batch_size=2)
        _, c = self.run_case("seed_c", [0, 1, 2, 3], TinyModel, seed=43, batch_size=2)
        self.assertEqual(log_a, log_b)
        self.assertEqual(log_a[0]["lr"], 0.001)
        for key, value in a["model_state_dict"].items():
            self.assertTrue(torch.equal(value, b["model_state_dict"][key]))
        self.assertTrue(any(not torch.equal(value, c["model_state_dict"][key])
                            for key, value in a["model_state_dict"].items()))

    def test_existing_run_is_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "best_checkpoint.pt"
        marker.write_bytes(b"previous run")
        with self.assertRaises(FileExistsError):
            trainer.train(self.root, output)
        self.assertEqual(marker.read_bytes(), b"previous run")

    def test_no_confusion_does_not_invent_a_target(self):
        cm = np.diag([8, 8, 8])
        rows = confusion_pairs(cm, [{"char": c} for c in "甲乙丙"])
        self.assertTrue(all(r["class_id_confuse"] is None and r["count"] == 0 and not r["valid"] for r in rows))

    def test_checkpoint_data_mismatch_is_rejected(self):
        checkpoint = self.root / "wrong.pt"
        torch.save({"config": {"data_manifest_sha256": "different"}}, checkpoint)
        with patch("eval_confusion.CharDataset", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Checkpoint and dataset differ"):
                evaluate(self.root, checkpoint, self.root / "eval")
        self.assertFalse((self.root / "eval").exists())


if __name__ == "__main__":
    unittest.main()
