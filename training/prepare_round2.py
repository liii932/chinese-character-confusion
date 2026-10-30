"""Prepare the full supplied character pool for round two, without training."""
from __future__ import annotations

import argparse
import copy
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pilot import render_samples as renderer
from training.io_utils import digest, write_json
from training.prepare_data import build

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "笔画微差类（长短、出头、点撇微殊）.md"
TABLE = ROOT / "training/characters1140.json"
OUTPUT = ROOT / "training/data/round2-1140-v1"


def make_table():
    original, _ = renderer.checked_input()
    if digest(SOURCE) != original["source"]["sha256"]:
        raise ValueError("Source pool changed; review its scope before generating")
    groups, members, category = [], {}, None
    for line_no, raw in enumerate(SOURCE.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if line.startswith("## "):
            category = line[3:]
        elif line and line != "---":
            chars = line.split("、")
            if category is None or len(chars) < 2 or any(len(c) != 1 for c in chars):
                raise ValueError(f"Invalid source group on line {line_no}: {line}")
            group = {"group_id": len(groups) + 1, "category": category,
                     "source_line": line_no, "source_group": line, "characters": chars}
            groups.append(group)
            for char in chars:
                members.setdefault(char, []).append(group)
    if len(groups) != 443 or len(members) != 1140 or sum(len(g["characters"]) for g in groups) != 1190:
        raise ValueError("Source pool does not match the recorded 443 groups / 1140 characters")

    # Keep the first model's 0..80 mapping, then append new characters in source order.
    labels = [row["char"] for row in original["characters"]]
    labels += [char for char in members if char not in labels]
    characters = []
    for class_id, char in enumerate(labels):
        sources = members[char]
        characters.append({"class_id": class_id, "char": char, "unicode": f"U+{ord(char):04X}",
            "category": sources[0]["category"],
            "source_group_ids": [g["group_id"] for g in sources],
            "source_categories": list(dict.fromkeys(g["category"] for g in sources))})

    candidates = {}
    for group in groups:
        if len(group["characters"]) != 2:
            continue
        a, b = sorted(group["characters"], key=ord)
        entry = candidates.setdefault((a, b), {"pair_id": len(candidates) + 1, "a": a, "b": b,
            "status": "user-listed candidate; glyph relation not newly verified",
            "source_group_ids": []})
        entry["source_group_ids"].append(group["group_id"])
    if len(candidates) != 251:
        raise ValueError("Expected 251 distinct explicit two-character source pairs")
    return {"schema_version": "2.0", "title": "第二轮完整 1140 字表",
        "created_on": "2026-10-29", "purpose": "Full supplied pool for round-two data generation",
        "source": {"file": SOURCE.name, "sha256": digest(SOURCE)},
        "class_id_convention": "Original 81 IDs unchanged; new characters appended in source order",
        "relation_rule": "Keep all source groups; no clique or transitive expansion. pairs retains the original 54 verified pairs; candidate_pairs contains explicit two-character source entries only.",
        "summary": {"unique_characters": len(characters), "source_groups": len(groups),
            "unique_source_groups": len({tuple(sorted(g["characters"])) for g in groups}),
            "character_occurrences": sum(len(g["characters"]) for g in groups),
            "explicit_candidate_pairs": len(candidates), "retained_verified_pairs": len(original["pairs"])},
        "characters": characters, "source_groups": groups,
        "pairs": copy.deepcopy(original["pairs"]), "candidate_pairs": list(candidates.values())}


def previews(output, table):
    """Preview every class and source group using actual saved train pixels."""
    output = Path(output)
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    conditions = json.loads((output / "conditions.json").read_text(encoding="utf-8"))["train"]
    n_classes = len(table["characters"])
    with np.load(output / "train.npz", allow_pickle=False) as data:
        images = data["images"]
    samples = {}
    for sigma in (0, 1.0):
        eligible = [i for i, c in enumerate(conditions) if c["sigma"] == sigma]
        condition_id = min(eligible, key=lambda i: conditions[i]["dx"] ** 2 + conditions[i]["dy"] ** 2)
        samples[sigma] = images[condition_id * n_classes:(condition_id + 1) * n_classes]
        for row in table["characters"]:
            Image.fromarray(samples[sigma][row["class_id"]]).save(preview / f"{row['class_id']:04d}_s{sigma:g}.png")
    del images
    sheets = []
    for start in range(0, n_classes, 96):
        subset = table["characters"][start:start + 96]
        sheet = Image.new("RGB", (1216, 96 + 100 * ((len(subset) + 11) // 12)), "#f5f7fb")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), f"Round 2 · 1140 characters · IDs {start}–{start + len(subset) - 1}",
                  font=renderer.font_at(24), fill="#182c44")
        draw.text((20, 50), "Actual train samples · sigma=0 · 64×64 grayscale", font=renderer.font_at(18), fill="#334155")
        for i, row in enumerate(subset):
            x, y = 20 + i % 12 * 100, 90 + i // 12 * 100
            sheet.paste(Image.fromarray(samples[0][row["class_id"]]).convert("RGB"), (x, y))
            draw.text((x, y + 66), f"{row['class_id']:04d} {row['char']}", font=renderer.font_at(17), fill="#182c44")
        name = f"overview_{start // 96 + 1:02d}.png"
        sheet.save(preview / name)
        sheets.append(name)
    by_char = {row["char"]: row["class_id"] for row in table["characters"]}
    blocks = []
    for group in table["source_groups"]:
        cells = []
        for char in group["characters"]:
            class_id = by_char[char]
            cells.append(f'<div><strong>{html.escape(char)} · {class_id}</strong><br>'
                f'<img loading="lazy" src="{class_id:04d}_s0.png" alt="{char} sigma 0">'
                f'<img loading="lazy" src="{class_id:04d}_s1.png" alt="{char} sigma 1"></div>')
        blocks.append(f'<section><h2>{group["group_id"]}. {html.escape(group["source_group"])}</h2>'
            f'<p>{html.escape(group["category"])} · 来源行 {group["source_line"]}</p><div class="group">'
            + "".join(cells) + '</div></section>')
    (preview / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>第二轮 1140 字预览</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;background:#f5f7fb;color:#182c44}'
        'section{background:white;padding:16px;margin:16px 0;border-radius:10px}.group{display:flex;gap:20px;flex-wrap:wrap}'
        'img{width:96px;height:96px;image-rendering:pixelated}h2{font-size:20px}p{color:#536278}</style>'
        '<h1>第二轮：443 个来源字组，1140 个类别</h1><p>每字左右为实际训练图 sigma=0 / 1。'
        '预览带标签，用于检查渲染；字组未自动展开成全部字对。</p>' + "".join(blocks) + '</html>', encoding="utf-8")
    write_json(preview / "preview_conditions.json", {str(s): conditions[min(
        (i for i, c in enumerate(conditions) if c["sigma"] == s),
        key=lambda i: conditions[i]["dx"] ** 2 + conditions[i]["dy"] ** 2)] for s in samples})
    return sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    table = make_table()
    if TABLE.exists() and json.loads(TABLE.read_text(encoding="utf-8")) != table:
        raise ValueError("Expanded character table changed; review before replacing it")
    write_json(TABLE, table)
    print(json.dumps(table["summary"], ensure_ascii=False), flush=True)
    if not args.table_only:
        build(args.output, seed=20261029, characters=TABLE)
        sheets = previews(args.output, table)
        print(f"Saved {len(sheets)} overview sheets and all 443 source-group previews", flush=True)


if __name__ == "__main__":
    main()
