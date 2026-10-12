"""Build shared, pixel-deduplicated conditions before assigning the four splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random

import fontTools
from fontTools.ttLib import TTFont
import numpy as np
import PIL
from PIL import Image, ImageChops, features

from pilot import render_samples as renderer
from training.io_utils import digest, write_json, verify_dataset

ROOT = Path(__file__).resolve().parents[1]
SPLIT_COUNTS = {"train": 64, "val_graph": 32, "val_select": 32, "test": 64}


def build(output, seed=20261012, counts=None):
    output = Path(output)
    counts = dict(SPLIT_COUNTS if counts is None else counts)
    if set(counts) != set(SPLIT_COUNTS) or any(n <= 0 for n in counts.values()):
        raise ValueError("Four positive split counts are required")
    if sum(counts.values()) > 625:
        raise ValueError("Requested more conditions than the finite position grid")
    data, character_hash = renderer.checked_input()
    if len(data["characters"]) != 81:
        raise ValueError("This baseline is explicitly limited to the approved 81-character table")
    source = json.loads((renderer.ROOT / "fonts/source.json").read_text(encoding="utf-8"))
    for item in source["files"]:
        if digest(renderer.ROOT / "fonts" / item["name"]) != item["sha256"]:
            raise ValueError(f"Font resource changed: {item['name']}")
    labels = [entry["char"] for entry in data["characters"]]
    with TTFont(renderer.FONT_PATH) as tt:
        cmap = tt.getBestCmap()
        if any(ord(c) not in cmap or tt.getGlyphID(cmap[ord(c)]) == 0 for c in labels):
            raise ValueError("Missing glyph in the approved character table")
    font, font_size, union, baseline = renderer.common_layout(labels)
    recipe = {
        "schema": 1, "purpose": "81-character baseline development",
        "seed": seed, "per_sigma_counts": counts,
        "characters_sha256": character_hash, "font_sha256": digest(renderer.FONT_PATH),
        "renderer_sha256": digest(Path(renderer.__file__)), "builder_sha256": digest(__file__),
        "size": 64, "mode": "L", "scale": 4, "font_size_high": font_size,
        "baseline_high": baseline, "union_bbox_high": union,
        "position_grid": {"min": -3, "max": 3, "step": 0.25},
        "sigmas": renderer.SIGMAS, "normalization": "(uint8 / 255 - 0.5) / 0.5",
        "allocation": "per split/sigma/axis: abs(mean)<=0.25 px, min<=-2.5 px, max>=2.5 px; deterministic rejection, max 10000 attempts",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "Pillow": PIL.__version__, "fonttools": fontTools.__version__,
                    "freetype": features.version_module("freetype2")},
    }
    if (output / "manifest.json").exists():
        manifest = verify_dataset(output)
        if manifest["recipe"] != recipe:
            raise ValueError("Recipe changed. Use a new dataset directory; do not mix datasets.")
        print(f"Verified existing dataset: {output}", flush=True)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete/nonempty dataset directory. Use a new empty directory.")
    output.mkdir(parents=True, exist_ok=True)

    # Render once at the shared baseline. Integer shifts at 4x resolution preserve
    # the approved renderer exactly; ImageChops.offset wrapping is forbidden by
    # the explicit bounding-box margin check below.
    bases, boxes = [], []
    for char in labels:
        high, box, cropped = renderer.render_high(char, font, baseline, 0, 0)
        if cropped:
            raise ValueError(f"Clipped base glyph: {char}")
        bases.append(high)
        boxes.append(box)

    rng = random.Random(seed)
    seen = {}  # Across every accepted condition, sigma, class and eventual split.
    split_arrays = {name: [] for name in counts}
    split_conditions = {name: [] for name in counts}
    rejected = []
    for sigma in renderer.SIGMAS:
        candidates = [(x, y) for x in range(-12, 13) for y in range(-12, 13)]
        rng.shuffle(candidates)
        accepted = []
        for x4, y4 in candidates:
            batch, hashes, duplicate = [], [], False
            local_hashes = {}
            for class_id, (base, box) in enumerate(zip(bases, boxes)):
                if min(box[0] + x4, box[1] + y4) < 0 or max(box[2] + x4, box[3] + y4) > 256:
                    raise ValueError("A proposed condition clips the glyph")
                low = ImageChops.offset(base, x4, y4).resize((64, 64), Image.Resampling.LANCZOS)
                image = renderer.blur_white(low, sigma)
                pixels = np.asarray(image, dtype=np.uint8)
                if pixels.min() == 255 or renderer.edge_darkness(image) != 0:
                    raise ValueError(f"Blank image or non-white edge: class {class_id}")
                h = hashlib.sha256(pixels.tobytes()).hexdigest()
                previous = seen.get(h, local_hashes.get(h))
                if previous is not None:
                    if previous != class_id:
                        raise ValueError("Different labels produce identical pixels; inspect rendering")
                    duplicate = True
                local_hashes[h] = class_id
                batch.append(pixels)
                hashes.append(h)
            if duplicate:
                rejected.append({"x4": x4, "y4": y4, "sigma": sigma})
                continue  # Replace this condition for ALL classes, before splitting.
            seen.update({h: i for i, h in enumerate(hashes)})
            accepted.append((np.stack(batch), {"dx": x4 / 4, "dy": y4 / 4, "sigma": sigma}))
            if len(accepted) == sum(counts.values()):
                break
        if len(accepted) != sum(counts.values()):
            raise ValueError("Insufficient unique conditions; revise the common recipe explicitly")
        # Balance geometric conditions only, without consulting predictions or
        # outcomes. Avoid one split being shifted noticeably to one side.
        for attempt in range(10000):
            rng.shuffle(accepted)
            cursor, balanced = 0, True
            for number in counts.values():
                subset = [c for _, c in accepted[cursor:cursor + number]]
                for axis in ("dx", "dy"):
                    values = [c[axis] for c in subset]
                    balanced &= abs(float(np.mean(values))) <= 0.25 and min(values) <= -2.5 and max(values) >= 2.5
                cursor += number
            if balanced:
                break
        else:
            raise ValueError("Could not allocate balanced geometric conditions within the declared budget")
        cursor = 0
        for name, number in counts.items():
            for pixels, condition in accepted[cursor:cursor + number]:
                split_arrays[name].append(pixels)
                split_conditions[name].append(condition)
            cursor += number
        print(f"sigma={sigma}: {len(accepted)} unique shared conditions accepted", flush=True)

    split_summary, distribution = {}, {}
    for name in counts:
        # Row order: condition first, then class 0..80. Label and condition index
        # are stored explicitly so downstream sampling cannot change alignment.
        images = np.stack(split_arrays[name]).reshape(-1, 64, 64)
        n_conditions = len(split_conditions[name])
        y = np.tile(np.arange(len(labels), dtype=np.int64), n_conditions)
        condition_ids = np.repeat(np.arange(n_conditions, dtype=np.int64), len(labels))
        np.savez_compressed(output / f"{name}.npz", images=images, labels=y, condition_ids=condition_ids)
        split_summary[name] = {"images": len(y), "per_class": n_conditions}
        distribution[name] = {}
        for sigma in renderer.SIGMAS:
            subset = [c for c in split_conditions[name] if c["sigma"] == sigma]
            distribution[name][str(sigma)] = {
                axis: {"mean": float(np.mean([c[axis] for c in subset])),
                       "min": min(c[axis] for c in subset), "max": max(c[axis] for c in subset)}
                for axis in ("dx", "dy")
            }
    write_json(output / "conditions.json", split_conditions)
    write_json(output / "labels.json", {"characters": data["characters"], "pairs": data["pairs"]})
    files = [f"{name}.npz" for name in counts] + ["conditions.json", "labels.json"]
    manifest = {
        "recipe": recipe, "splits": split_summary,
        "audit": {"unique_pixel_hashes": len(seen), "cross_label_duplicates": 0,
                  "cross_split_duplicates": 0, "rejected_shared_conditions": rejected,
                  "position_distribution": distribution},
        "files": {name: digest(output / name) for name in files},
    }
    # A manifest is the completion marker; interrupted output must not be reused.
    write_json(output / "manifest.json", manifest)
    verify_dataset(output)
    print(json.dumps(split_summary, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_dataset(args.output)
        print(json.dumps(manifest["splits"], indent=2))
    else:
        build(args.output, args.seed)


if __name__ == "__main__":
    main()
