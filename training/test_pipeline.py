"""Integration checks using real generated pixels and small CPU training runs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from training.io_utils import digest, verify_dataset, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/data/baseline81-v2"


class PipelineTests(unittest.TestCase):
    def test_full_dataset_counts_and_no_pixel_leakage(self):
        manifest = verify_dataset(DATA)
        conditions = json.loads((DATA / "conditions.json").read_text(encoding="utf-8"))
        seen, condition_seen = set(), set()
        for split, expected_per_class in {"train": 192, "val_graph": 96, "val_select": 96, "test": 192}.items():
            with np.load(DATA / f"{split}.npz", allow_pickle=False) as d:
                self.assertEqual(d["images"].shape, (expected_per_class * 81, 64, 64))
                self.assertEqual(d["images"].dtype, np.uint8)
                np.testing.assert_array_equal(np.bincount(d["labels"]), np.full(81, expected_per_class))
                self.assertEqual(len(conditions[split]), expected_per_class)
                for condition in conditions[split]:
                    key = (condition["dx"], condition["dy"], condition["sigma"])
                    self.assertNotIn(key, condition_seen)
                    condition_seen.add(key)
                for sigma in [0, 0.5, 1.0]:
                    self.assertEqual(sum(c["sigma"] == sigma for c in conditions[split]), expected_per_class // 3)
                    for axis in ("dx", "dy"):
                        values = [c[axis] for c in conditions[split] if c["sigma"] == sigma]
                        self.assertLessEqual(abs(float(np.mean(values))), 0.25)
                        self.assertLessEqual(min(values), -2.5)
                        self.assertGreaterEqual(max(values), 2.5)
                for image in d["images"]:
                    h = hashlib.sha256(image.tobytes()).digest()
                    self.assertNotIn(h, seen)
                    seen.add(h)
        self.assertEqual(len(seen), 46656)
        self.assertEqual(manifest["audit"]["unique_pixel_hashes"], 46656)

    def test_train_resume_evaluate_and_guards(self):
        scratch_root = ROOT / "training/.verification"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="integration-", dir=scratch_root) as temp:
            base = Path(temp)
            self.assertTrue(base.resolve().is_relative_to(scratch_root.resolve()))
            fixture = base / "data"
            fixture.mkdir()
            manifest = copy.deepcopy(verify_dataset(DATA))
            conditions = json.loads((DATA / "conditions.json").read_text(encoding="utf-8"))
            for split in manifest["splits"]:
                with np.load(DATA / f"{split}.npz", allow_pickle=False) as d:
                    # One real condition, all 81 labels: intentionally tiny fixture.
                    np.savez_compressed(fixture / f"{split}.npz", images=d["images"][:81],
                                        labels=d["labels"][:81], condition_ids=np.zeros(81, dtype=np.int64))
                conditions[split] = conditions[split][:1]
                manifest["splits"][split] = {"images": 81, "per_class": 1}
            (fixture / "labels.json").write_bytes((DATA / "labels.json").read_bytes())
            write_json(fixture / "conditions.json", conditions)
            manifest["files"] = {name: digest(fixture / name) for name in manifest["files"]}
            manifest["recipe"]["purpose"] = "Tiny local integration fixture; not an experimental result"
            write_json(fixture / "manifest.json", manifest)

            def command(run, *extra, success=True, action="train"):
                result = subprocess.run([sys.executable, "-m", "training.train", action,
                    "--data", str(fixture), "--run-dir", str(base / run), "--allow-cpu",
                    "--epochs", "2", "--batch-size", "27", *extra], cwd=ROOT,
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
                if success and result.returncode:
                    self.fail(result.stdout + result.stderr)
                if not success:
                    self.assertNotEqual(result.returncode, 0)
                return result

            command("continuous")
            command("resumed", "--stop-after", "1")
            command("resumed", "--resume")
            continuous = torch.load(base / "continuous/last.pt", map_location="cpu", weights_only=True)
            resumed = torch.load(base / "resumed/last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            for key, value in continuous["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][key]), key)
            self.assertEqual(continuous["scheduler"], resumed["scheduler"])
            self.assertEqual(continuous["best_score"], resumed["best_score"])
            for a, b in zip(continuous["history"], resumed["history"]):
                self.assertEqual(a["train"], b["train"])
                self.assertEqual(a["validation"], b["validation"])
            result = command("resumed", "--resume", "--lr", "0.002", success=False)
            self.assertIn("Resume configuration", result.stderr)
            result = command("resumed", success=False)
            self.assertIn("nonempty", result.stderr)
            command("resumed", action="evaluate")
            report = json.loads((base / "resumed/evaluation_val_select/metrics.json").read_text())
            self.assertEqual(report["samples"], 81)
            matrix = np.load(base / "resumed/evaluation_val_select/confusion.npy")
            self.assertEqual(matrix.shape, (81, 81))
            self.assertEqual(int(matrix.sum()), 81)
            result = command("resumed", "--split", "test", action="evaluate", success=False)
            self.assertIn("Test is reserved", result.stderr)
            # This evaluates only the tiny fixture to exercise the explicit gate.
            command("resumed", "--split", "test", "--final-test", action="evaluate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
