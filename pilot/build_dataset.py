"""Build an independent pilot dataset, deduplicated before splitting.

All classes share the same quarter-pixel conditions. Existing output directories
are never overwritten; this dataset is separate from the Colab baseline.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random

import fontTools
import PIL
from PIL import Image, features

from render_samples import (
    ROOT, SIZE, SCALE, FONT_PATH, checked_input, common_layout,
    render_high, blur_white, save_json,
)

OUT_ROOT = ROOT / "dataset_81_v2"
SAMPLE_PER_CHAR = {"train": 20, "val_graph": 8, "val_select": 6, "test": 6}
SPLITS = tuple(SAMPLE_PER_CHAR)
SIGMAS = (0.0, 0.5, 1.0)
RANDOM_SEED = 42


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def render_image(char, font, baseline, dx, dy, sigma):
    high, _, cropped = render_high(char, font, baseline, dx, dy)
    if cropped:
        raise ValueError(f"Cropped glyph: {char}")
    return blur_white(high.resize((SIZE, SIZE), Image.Resampling.LANCZOS), sigma)


def build_dataset(output=OUT_ROOT, seed=RANDOM_SEED):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    data, character_hash = checked_input()
    entries = data["characters"]
    source = json.loads((ROOT / "fonts/source.json").read_text(encoding="utf-8"))
    for resource in source["files"]:
        if sha256((ROOT / "fonts" / resource["name"]).read_bytes()) != resource["sha256"]:
            raise ValueError("Font resource hash mismatch")
    font, font_size, union_bbox, baseline = common_layout([c["char"] for c in entries])
    rng = random.Random(seed)
    # Near-equal sigma counts (difference <= 1) in each split, for every class.
    quotas = {split: [n // 3 + int(i < n % 3) for i in range(3)]
              for split, n in SAMPLE_PER_CHAR.items()}
    pools, seen = {}, set()
    rejected = 0
    for sigma_index, sigma in enumerate(SIGMAS):
        positions = [(x / SCALE, y / SCALE) for x in range(-12, 13)
                     for y in range(-12, 13)]
        rng.shuffle(positions)
        needed = sum(q[sigma_index] for q in quotas.values())
        pool = []
        for dx, dy in positions:
            images = [render_image(c["char"], font, baseline, dx, dy, sigma) for c in entries]
            hashes = [sha256(image.tobytes()) for image in images]
            # Reject the entire condition if any label collides, preserving
            # a common condition table across classes, including after dedup.
            if len(set(hashes)) != len(hashes) or seen.intersection(hashes):
                rejected += 1
                continue
            seen.update(hashes)
            pool.append((dx, dy, sigma, images, hashes))
            if len(pool) == needed:
                break
        if len(pool) != needed:
            raise ValueError(f"Insufficient unique conditions for sigma={sigma}")
        pools[sigma] = pool

    # Assign splits only AFTER global decoded-pixel deduplication.
    conditions = {split: [] for split in SPLITS}
    for i, sigma in enumerate(SIGMAS):
        cursor = 0
        for split in SPLITS:
            count = quotas[split][i]
            conditions[split].extend(pools[sigma][cursor:cursor + count])
            cursor += count
    output.mkdir(parents=True, exist_ok=False)
    (output / "meta").mkdir()
    for split in SPLITS:
        (output / split).mkdir()
    records = []
    for entry_index, entry in enumerate(entries):
        for split in SPLITS:
            for dx, dy, sigma, images, hashes in conditions[split]:
                filename = f"cid_{entry['class_id']}_{len(records):06d}.png"
                images[entry_index].save(output / split / filename)
                records.append({
                    "char": entry["char"], "class_id": entry["class_id"],
                    "split": split, "dx_out": dx, "dy_out": dy, "sigma": sigma,
                    "filename": filename, "file_relative": f"{split}/{filename}",
                    "pixel_sha256": hashes[entry_index],
                })
    meta_path = output / "meta/dataset_meta.json"
    save_json(meta_path, records)
    manifest = {
        "schema_version": 2, "purpose": "independent pilot; not the Colab baseline",
        "seed": seed, "size": SIZE, "mode": "L", "supersampling": SCALE,
        "offset_step": 1 / SCALE, "offset_limit": [-3, 3], "sigmas": list(SIGMAS),
        "samples_per_class": SAMPLE_PER_CHAR, "sigma_counts_per_class": quotas,
        "shared_conditions": {s: [{"dx_out": c[0], "dy_out": c[1], "sigma": c[2]}
                                  for c in conditions[s]] for s in SPLITS},
        "characters": entries, "characters_sha256": character_hash,
        "font_sha256": sha256(FONT_PATH.read_bytes()), "font_size_high": font_size,
        "baseline_high": baseline, "union_bbox_high": union_bbox,
        "renderer_sha256": sha256((ROOT / "render_samples.py").read_bytes()),
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "metadata_sha256": sha256(meta_path.read_bytes()),
        "samples": len(records), "unique_pixels": len(seen),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "rejected_duplicate_conditions": rejected,
        "environment": {"python": platform.python_version(), "Pillow": PIL.__version__,
                        "fonttools": fontTools.__version__,
                        "freetype": features.version_module("freetype2")},
    }
    save_json(output / "meta/manifest.json", manifest)
    print(f"Dataset saved: {output}; {len(records)} unique images")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    build_dataset(args.output, args.seed)


if __name__ == "__main__":
    main()
