"""One-off offline audit of the full round-two saved pixels; never used at training startup."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pilot.render_samples import checked_input
from training.io_utils import verify_dataset, write_json
from training.prepare_round2 import OUTPUT, TABLE


def audit(output):
    output = Path(output)
    manifest = verify_dataset(output)
    table, _ = checked_input(TABLE)
    labels = json.loads((output / "labels.json").read_text(encoding="utf-8"))
    for key in ("characters", "pairs", "source_groups", "candidate_pairs"):
        if labels[key] != table[key]:
            raise ValueError(f"Saved label metadata mismatch: {key}")
    k = len(labels["characters"])
    conditions = json.loads((output / "conditions.json").read_text(encoding="utf-8"))
    seen, seen_conditions, summary = set(), set(), {}
    for split, per_class in {"train": 192, "val_graph": 96, "val_select": 96, "test": 192}.items():
        with np.load(output / f"{split}.npz", allow_pickle=False) as saved:
            images, targets, ids = saved["images"], saved["labels"], saved["condition_ids"]
        if images.shape != (k * per_class, 64, 64) or images.dtype != np.uint8:
            raise ValueError(f"Wrong saved pixel array: {split}")
        np.testing.assert_array_equal(targets, np.tile(np.arange(k), per_class))
        np.testing.assert_array_equal(ids, np.repeat(np.arange(per_class), k))
        if len(conditions[split]) != per_class:
            raise ValueError(f"Wrong number of conditions: {split}")
        sigma_counts = Counter(c["sigma"] for c in conditions[split])
        if sigma_counts != {0: per_class // 3, 0.5: per_class // 3, 1: per_class // 3}:
            raise ValueError(f"Wrong sigma distribution: {split}")
        for c in conditions[split]:
            key = (c["dx"], c["dy"], c["sigma"])
            if key in seen_conditions:
                raise ValueError("Repeated condition within/across splits")
            seen_conditions.add(key)
            if any(not -3 <= c[a] <= 3 or c[a] * 4 != int(c[a] * 4) for a in ("dx", "dy")):
                raise ValueError("Offset outside the prescribed grid")
        for sigma in sigma_counts:
            for axis in ("dx", "dy"):
                values = [c[axis] for c in conditions[split] if c["sigma"] == sigma]
                if abs(float(np.mean(values))) > 0.25 or min(values) > -2.5 or max(values) < 2.5:
                    raise ValueError("Unbalanced geometry")
        for start in range(0, len(images), k):
            batch = images[start:start + k]
            if np.any(np.min(batch, axis=(1, 2)) == 255):
                raise ValueError("Blank image")
            if np.any(batch[:, [0, -1], :] != 255) or np.any(batch[:, :, [0, -1]] != 255):
                raise ValueError("Nonwhite image boundary")
            for image in batch:
                h = hashlib.sha256(image.tobytes()).digest()
                if h in seen:
                    raise ValueError("Repeated pixels within/across classes or splits")
                seen.add(h)
        summary[split] = {"images": len(targets), "per_class": per_class,
                          "per_class_per_sigma": dict(sigma_counts)}
        del images
        print(f"Audited {split}: {len(targets):,} decoded images", flush=True)
    if len(seen) != manifest["audit"]["unique_pixel_hashes"]:
        raise ValueError("Decoded unique-pixel count differs from generation manifest")
    report = {"automatic_checks_passed": True, "classes": k, "source_groups": len(labels["source_groups"]),
        "unique_images": len(seen), "splits": summary,
        "blank_images": 0, "nonwhite_boundaries": 0, "pixel_duplicates": 0,
        "cross_split_condition_duplicates": 0, "file_hashes_verified": True,
        "saved_label_condition_alignment_verified": True,
        "interpretation": "Pixel and provenance audit only; no model predictions or human readability measurement"}
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.data), ensure_ascii=False, indent=2))
