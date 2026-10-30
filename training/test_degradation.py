"""Focused phase-1 checks; synthetic inference fixtures are never experimental results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from training import degradation as D
from training.io_utils import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/data/degradation81-calibration-v1"


class DegradationTests(unittest.TestCase):
    def test_identity_blur_boundary_and_resolution(self):
        pixels = np.full((64, 64), 255, dtype=np.uint8)
        pixels[8:56, 30:34] = 0
        source = Image.fromarray(pixels)
        np.testing.assert_array_equal(D.degrade(source, 0, 64), pixels)
        expected = ImageOps.expand(source, border=16, fill=255).filter(ImageFilter.GaussianBlur(1)).crop((16, 16, 80, 80))
        np.testing.assert_array_equal(D.degrade(source, 1, 64), expected)
        transformed = np.asarray(D.degrade(source, 1.5, 16))
        self.assertEqual(transformed.shape, (64, 64))
        self.assertFalse(np.array_equal(transformed, pixels))
        self.assertLess(transformed.min(), 200)

    def test_generated_grid_and_group_provenance(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertEqual(manifest["shape"], [25, 1296, 64, 64])
        self.assertEqual(manifest["samples"], 32400)
        self.assertEqual(len({(g["sigma"], g["resolution"]) for g in manifest["grids"]}), 25)
        with np.load(DATA / "sweep.npz", allow_pickle=False) as saved, np.load(
                ROOT / "training/data/baseline81-v2/val_graph.npz", allow_pickle=False) as original:
            images, truth = saved["images"], saved["labels"]
            indexes, condition_ids = saved["source_indexes"], saved["source_condition_ids"]
            self.assertEqual(images.dtype, np.uint8)
            np.testing.assert_array_equal(images[0], original["images"][indexes])
            np.testing.assert_array_equal(truth, original["labels"][indexes])
            np.testing.assert_array_equal(np.bincount(truth), np.full(81, 16))
            self.assertEqual(len(set(indexes.tolist())), 1296)
            self.assertEqual(len(set(condition_ids.tolist())), 16)
            # Each severity shares every source group and every class; no label-dependent corruption.
            for index in (0, 647, 1295):
                for grid in manifest["grids"]:
                    expected = D.degrade(Image.fromarray(images[0, index]), grid["sigma"], grid["resolution"])
                    np.testing.assert_array_equal(images[grid["grid_id"], index], expected)
        for name, expected in manifest["files"].items():
            self.assertEqual(digest(DATA / name), expected)

    def test_duplicate_labels_are_detected(self):
        images = np.zeros((1, 2, 64, 64), dtype=np.uint8)
        audit = D.audit_pixels(images, np.array([0, 1]))
        self.assertEqual(len(audit["cross_label_collisions"]), 1)
        self.assertEqual(audit["unique_pixels"], 1)

    def test_directional_pair_metrics_and_support(self):
        labels = {"characters": [{"char": c, "class_id": i} for i, c in enumerate("甲乙丙")],
                  "pairs": [{"pair_id": 1, "a": "甲", "b": "乙"}]}
        truth = np.repeat(np.arange(3), 16)
        prediction = truth.copy()
        prediction[:6], prediction[16:18], prediction[32] = 1, 0, 0
        metrics, matrix = D.summarize(truth, prediction, np.ones(48), labels)
        self.assertEqual(metrics["errors"], 9)
        self.assertEqual(metrics["pairs"][0]["a_to_b"], 6)
        self.assertEqual(metrics["pairs"][0]["b_to_a"], 2)
        self.assertEqual(metrics["pairs"][0]["pair_error_rate"], .25)
        self.assertEqual(metrics["supported_sources_ge5"], [{"source_id": 0, "target_id": 1, "count": 6}])
        self.assertAlmostEqual(metrics["macro_accuracy"], 39 / 48)
        self.assertEqual(matrix.sum(), 48)

    def test_real_torch_inference_and_report_on_tiny_fixture(self):
        import torch
        from training import train as T
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="phase1-test-") as temp:
            folder = Path(temp)
            fixture = folder / "data"
            fixture.mkdir()
            labels = {"characters": [{"char": c, "class_id": i} for i, c in enumerate("甲乙丙")],
                      "pairs": [{"pair_id": 1, "a": "甲", "b": "乙"}]}
            truth = np.tile(np.arange(3), 2)
            pixels = np.full((25, 6, 64, 64), 255, dtype=np.uint8)
            pixels[:, :, 20:44, 30:34] = 0
            np.savez_compressed(fixture / "sweep.npz", images=pixels, labels=truth,
                                source_indexes=np.arange(6), source_condition_ids=np.repeat([0, 1], 3))
            write_json(fixture / "labels.json", labels)
            manifest = copy.deepcopy(json.loads((DATA / "manifest.json").read_text()))
            manifest["files"] = {name: digest(fixture / name) for name in ("sweep.npz", "labels.json")}
            manifest["pixel_sha256"] = "synthetic-test-fixture-not-a-research-result"
            write_json(fixture / "manifest.json", manifest)
            model = T.model_for(3)
            with torch.no_grad():
                model.fc.weight.zero_()
                model.fc.bias.copy_(torch.tensor([1., 0., -1.]))
            checkpoint = folder / "fixture.pt"
            torch.save({"epoch": 0, "model": model.state_dict(), "config": {
                "mode": "baseline", "data_sha256": D.BASE_MANIFEST_SHA256,
                "trainer_sha256": digest(Path(T.__file__)), "environment": T.environment(torch.device("cpu"))}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "Checkpoint does not match"):
                D.evaluate(fixture, checkpoint, folder / "rejected", device_name="cpu")
            summary = D.evaluate(fixture, checkpoint, folder / "report", batch_size=4, device_name="cpu",
                                 expected_checkpoint_sha256=digest(checkpoint))
            self.assertEqual(summary["samples"], 150)
            self.assertEqual(summary["grids"], 25)
            self.assertAlmostEqual(summary["clean_control_accuracy"], 1 / 3)
            self.assertFalse(summary["test_evaluated"])
            metrics = json.loads((folder / "report/metrics.json").read_text())
            self.assertTrue(all(m["errors"] == 4 for m in metrics))
            for name in ("summary.json", "metrics.json", "summary.csv", "pairs.csv", "predictions.csv", "confusion.npz", "accuracy_curves.png", "report.html"):
                self.assertTrue((folder / "report" / name).is_file(), name)

    def test_notebook_is_standalone_and_model_source_bytes_match(self):
        from colab.make_degradation_notebook import build
        notebook = build()
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code), 2)
        import ast
        prepare = "".join(code[0]["source"])
        tree = ast.parse(prepare)
        sources = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in n.targets))
        for name, source in sources.items():
            self.assertEqual(source.encode("utf-8"), (ROOT / name).read_bytes())
        self.assertNotIn('archive.read("training/data/baseline81-v2/test.npz")', prepare)
        for cell in code:
            compile("".join(cell["source"]), cell["id"], "exec")
            self.assertFalse(cell["outputs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
