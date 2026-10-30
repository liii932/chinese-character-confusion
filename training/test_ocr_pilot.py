"""Focused integration checks using synthetic images and fake recognizers."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from training import ocr_pilot as pilot
from training.ocr_label import image_box


class FakeClassifier(torch.nn.Module):
    def forward(self, images):
        return torch.tensor([[5.0, 0.0]]).repeat(len(images), 1)


class FakeRecognizer:
    def __init__(self, **kwargs):
        pass

    def predict(self, input, batch_size):
        return [types.SimpleNamespace(json={"res": {"rec_text": "士", "rec_score": 0.9}})]


class PilotTest(unittest.TestCase):
    def test_paired_workflow_and_material_lock(self):
        temp_root = Path(__file__).parent / ".verification"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            labels = {"characters": [{"char": "土", "class_id": 0}, {"char": "士", "class_id": 1}],
                      "pairs": [{"a": "土", "b": "士"}]}
            (dataset / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            (dataset / "manifest.json").write_text(json.dumps({"files": {"labels.json": pilot.digest(dataset / "labels.json")}}), encoding="utf-8")
            ckpt = root / "best.pt"
            torch.save({"epoch": 38, "config": {"mode": "baseline", "data_sha256": pilot.digest(dataset / "manifest.json")},
                        "model": {}}, ckpt)
            model_dir = root / "paddle_model"
            model_dir.mkdir()
            (model_dir / "inference.yml").write_text("fake", encoding="utf-8")
            manifests = []
            for split in ("dev", "test"):
                Image.new("L", (20, 20), 255).save(root / f"{split}.png")
                (root / f"{split}.txt").write_text("土", encoding="utf-8-sig")
                data = {"split": split, "pages": [{"id": split, "source_id": split,
                        "edge_text": f"{split}.txt", "edge_status": "ok", "lines": [
                            {"id": "l1", "image": f"{split}.png", "truth": "土", "boxes": [[0, 0, 20, 20]]}]}]}
                if split == "test":
                    data["pages"].append({**data["pages"][0], "id": "test2", "source_id": "test2"})
                path = root / f"{split}.json"
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                manifests.append(path)
            self.assertEqual(pilot.validate(manifests), {"dev": 1, "test": 2})
            fake_paddle = types.SimpleNamespace(__version__="3.2.0")
            fake_ocr = types.SimpleNamespace(__version__="3.4.1", TextRecognition=FakeRecognizer)
            with patch.dict("sys.modules", {"paddle": fake_paddle, "paddleocr": fake_ocr}), \
                 patch("training.train.model_for", return_value=FakeClassifier()), \
                 patch.object(pilot, "EXPECTED_CHECKPOINT_SHA256", pilot.digest(ckpt)):
                for split, manifest in zip(("dev", "test"), manifests):
                    pilot.run_paddle(manifest, root / f"{split}_a.json", "test_model", model_dir, "cpu")
                    pilot.candidates(manifest, root / f"{split}_a.json", dataset, ckpt, root / f"{split}_c.json")
                pilot.tune(manifests[0], root / "dev_a.json", root / "dev_c.json", root / "policy.json")
                pilot.report(manifests[1], root / "test_a.json", root / "test_c.json", root / "policy.json", root / "report.json")
            report = pilot.read(root / "report.json")
            self.assertEqual(report["cer"], {"A": 1.0, "B": 0.0, "D": 0.0})
            self.assertEqual(report["replacement_counts_oracle_aligned"]["fixed"], 2)
            self.assertEqual(report["paired_bootstrap_95"]["B_minus_A"], [-1.0, -1.0])
            self.assertEqual(report["paired_bootstrap_95"]["D_minus_B"], [0.0, 0.0])
            with self.assertRaises(ValueError):
                pilot.tune(manifests[1], root / "test_a.json", root / "test_c.json", root / "forbidden.json")
            (root / "test.txt").write_text("士", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                pilot.report(manifests[1], root / "test_a.json", root / "test_c.json", root / "policy.json", root / "changed.json")

    def test_alignment_and_unmatched_insertions(self):
        distance, mapping = pilot.alignment("土木", "士大木")
        self.assertEqual(distance, 2)
        self.assertEqual(mapping, {2: 1, 1: 0})
        revised, changes = pilot.apply("士木", [{"index": 0, "original": "士", "proposed": "土", "score": 0.8}], None)
        self.assertEqual((revised, changes), ("士木", []))

    def test_assemble_annotations(self):
        temp_root = Path(__file__).parent / ".verification"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            root = Path(tmp)
            (root / "line.json").write_text('{"id":"l1","truth":"土","image":"x.png","boxes":[null]}', encoding="utf-8")
            (root / "draft.json").write_text('{"split":"dev","pages":[{"id":"p","lines":["line.json"]}]}', encoding="utf-8")
            pilot.assemble(root / "draft.json", root / "manifest.json")
            self.assertEqual(pilot.read(root / "manifest.json")["pages"][0]["lines"][0]["truth"], "土")

    def test_box_mapping_with_fractional_zoom(self):
        self.assertEqual(image_box((10, 8), (1, 1), 0.5, 16, 12), [2, 2, 16, 12])
        self.assertEqual(image_box((-5, -1), (6, 7), 2, 20, 20), [0, 0, 3, 4])


if __name__ == "__main__":
    unittest.main()
