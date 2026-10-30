"""Focused CPU checks for subset integrity and deterministic block resumption."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from colab import learning_curve as C
from training import train as T

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/data/baseline81-v2"


class TinyData:
    def __init__(self):
        rng = np.random.default_rng(135)
        self.images = rng.normal(size=(6, 1, 4, 4)).astype(np.float32)
        self.labels = np.tile(np.arange(3), 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]), int(self.labels[index]), index


def tiny_model(classes):
    return nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU(),
                         nn.Dropout(0.25), nn.Linear(8, classes))


class LearningCurveTests(unittest.TestCase):
    def test_actual_train_subsets_and_balanced_draws(self):
        conditions = json.loads((DATA / "conditions.json").read_text(encoding="utf-8"))
        selected = C.choose_subsets(conditions["train"])
        self.assertEqual(selected, C.choose_subsets(conditions["train"]))
        dataset = T.GlyphDataset(DATA, "train")
        previous_ids = set()
        for size in C.SIZES:
            subset = selected["subsets"][str(size)]
            ids = set(subset["condition_ids"])
            self.assertEqual(len(ids), size)
            self.assertTrue(previous_ids < ids)
            previous_ids = ids
            for sigma in (0.0, 0.5, 1.0):
                group = [conditions["train"][i] for i in ids if conditions["train"][i]["sigma"] == sigma]
                self.assertEqual(len(group), size // 3)
                for axis in ("dx", "dy"):
                    values = [c[axis] for c in group]
                    self.assertLessEqual(abs(float(np.mean(values))), 0.5)
                    self.assertLessEqual(min(values), -2)
                    self.assertGreaterEqual(max(values), 2)
            unique, repeated = C.build_rows(dataset, subset)
            self.assertEqual(len(unique), size * 81)
            self.assertEqual(len(repeated), 15552)
            _, counts = np.unique(repeated, return_counts=True)
            np.testing.assert_array_equal(counts, np.full(size * 81, 192 // size))
            np.testing.assert_array_equal(np.bincount(dataset.labels[repeated]), np.full(81, 192))
            view = C.RepeatedRows(dataset, repeated)
            for index in (0, len(repeated) // 2, len(repeated) - 1):
                image, label, local_index = view[index]
                self.assertEqual(local_index, index)
                self.assertEqual(label, dataset.labels[repeated[index]])
                self.assertTrue(torch.equal(image, dataset[int(repeated[index])][0]))
        np.testing.assert_array_equal(repeated, np.arange(15552))

    def test_resume_matches_continuous_with_rng_and_update_counts(self):
        torch.set_num_threads(2)
        scratch = ROOT / "training/.verification"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="curve-", dir=scratch) as name:
            base = Path(name)
            self.assertTrue(base.resolve().is_relative_to(scratch.resolve()))
            data = TinyData()
            repeated = C.RepeatedRows(data, np.tile(np.arange(len(data)), 2))
            sigmas = np.array([0, 0, 0, 0.5, 0.5, 0.5])
            cfg = {"environment": T.environment(torch.device("cpu")), "seed": 42,
                   "n_classes": 3, "lr": 0.001, "weight_decay": 0.0001,
                   "blocks": 2, "batch_size": 3, "block_draws": 12,
                   "repeat_factor": 2, "unique_per_class": 2}
            with patch.object(T, "model_for", tiny_model):
                C.run_arm(T, torch, repeated, data, sigmas, cfg, base / "continuous")
                C.run_arm(T, torch, repeated, data, sigmas, cfg, base / "resumed", stop_after=1)
                C.run_arm(T, torch, repeated, data, sigmas, cfg, base / "resumed")
                a = torch.load(base / "continuous/last.pt", weights_only=True)
                b = torch.load(base / "resumed/last.pt", weights_only=True)
                self.assertEqual(a["optimizer_updates"], 8)
                self.assertEqual(b["optimizer_updates"], 8)
                self.assertEqual(a["scheduler"], b["scheduler"])
                self.assertEqual(a["best_score"], b["best_score"])
                self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
                for key in a["model"]:
                    self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
                for ra, rb in zip(a["history"], b["history"]):
                    self.assertEqual(ra["train"], rb["train"])
                    self.assertEqual(ra["validation"], rb["validation"])
                    self.assertEqual(ra["amp_skipped_updates_this_block"], 0)
                changed = copy.deepcopy(cfg)
                changed["lr"] = 0.002
                with self.assertRaisesRegex(ValueError, "Resume configuration"):
                    C.run_arm(T, torch, repeated, data, sigmas, changed, base / "resumed")
                before = hashlib.sha256((base / "resumed/last.pt").read_bytes()).hexdigest()
                with patch.object(T, "epoch_pass", side_effect=AssertionError("must not retrain")):
                    C.run_arm(T, torch, repeated, data, sigmas, cfg, base / "resumed")
                after = hashlib.sha256((base / "resumed/last.pt").read_bytes()).hexdigest()
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
