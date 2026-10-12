"""Full pixel audit and regeneration checks, using the actual renderer."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_dataset as builder
from render_samples import checked_input, common_layout, render_high, blur_white


class DatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".verification").mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / ".verification")
        cls.output = Path(cls.temp.name) / "dataset"
        cls.manifest = builder.build_dataset(cls.output)
        cls.records = json.loads((cls.output / "meta/dataset_meta.json").read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pixels_labels_splits_and_reconstruction(self):
        data, _ = checked_input()
        chars = data["characters"]
        font, _, _, baseline = common_layout([c["char"] for c in chars])
        hashes, groups = set(), defaultdict(list)
        self.assertEqual(len(self.records), 3240)
        self.assertEqual(Counter(r["split"] for r in self.records),
                         {"train": 1620, "val_graph": 648, "val_select": 486, "test": 486})
        for r in self.records:
            with Image.open(self.output / r["file_relative"]) as image:
                self.assertEqual((image.mode, image.size), ("L", (64, 64)))
                pixels = image.tobytes()
            digest = hashlib.sha256(pixels).hexdigest()
            self.assertEqual(digest, r["pixel_sha256"])
            self.assertNotIn(digest, hashes, "Duplicate pixels within/across splits or labels")
            hashes.add(digest)
            self.assertEqual(chars[r["class_id"]]["char"], r["char"])
            high, _, cropped = render_high(r["char"], font, baseline, r["dx_out"], r["dy_out"])
            self.assertFalse(cropped)
            restored = blur_white(high.resize((64, 64), Image.Resampling.LANCZOS), r["sigma"])
            self.assertEqual(restored.tobytes(), pixels, "Stored parameters cannot reproduce pixels")
            groups[(r["class_id"], r["split"])].append((r["dx_out"], r["dy_out"], r["sigma"]))
        for split, expected in builder.SAMPLE_PER_CHAR.items():
            reference = groups[(0, split)]
            for cid in range(81):
                conditions = groups[(cid, split)]
                self.assertEqual(len(conditions), expected)
                self.assertEqual(conditions, reference)
                counts = Counter(c[2] for c in conditions)
                self.assertEqual(set(counts), {0.0, 0.5, 1.0})
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(len(hashes), 3240)

    def test_same_seed_regenerates_identical_dataset(self):
        other = Path(self.temp.name) / "repeat"
        manifest = builder.build_dataset(other)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual((other / "meta/dataset_meta.json").read_bytes(),
                         (self.output / "meta/dataset_meta.json").read_bytes())
        for r in self.records:
            self.assertEqual((other / r["file_relative"]).read_bytes(),
                             (self.output / r["file_relative"]).read_bytes())

    def test_existing_dataset_is_not_overwritten(self):
        before = (self.output / "meta/manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            builder.build_dataset(self.output, seed=43)
        self.assertEqual((self.output / "meta/manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
