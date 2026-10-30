"""Render a finite inspection set; this is not a train/validation/test dataset."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform

import fontTools
from fontTools.ttLib import TTFont
import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat, features

ROOT = Path(__file__).resolve().parent
SIZE, SCALE, START_FONT = 64, 4, 48
SIGMAS = [0, 0.5, 1.0]
POSITIONS = [("c", 0, 0), ("tl", -3, -3), ("tr", 3, -3), ("bl", -3, 3), ("br", 3, 3)]
FONT_PATH = ROOT / "fonts" / "SourceHanSansSC-Regular.otf"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox_union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


@lru_cache(maxsize=16)
def font_at(size):
    return ImageFont.truetype(str(FONT_PATH), size, layout_engine=ImageFont.Layout.BASIC)


def dark_bbox(image, minimum_darkness=1):
    return image.point([255 if 255 - value >= minimum_darkness else 0 for value in range(256)]).getbbox()


def edge_darkness(image):
    w, h = image.size
    pixels = list(image.crop((0, 0, w, 1)).getdata()) + list(image.crop((0, h - 1, w, h)).getdata())
    pixels += list(image.crop((0, 0, 1, h)).getdata()) + list(image.crop((w - 1, 0, w, h)).getdata())
    return 255 - min(pixels)


def checked_input(path=None):
    raw = Path(path if path is not None else ROOT / "characters.json").read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    chars, pairs = data["characters"], data["pairs"]
    labels = [row["char"] for row in chars]
    if len(labels) != len(set(labels)) or any(len(c) != 1 for c in labels):
        raise ValueError("Character labels must be distinct single code points")
    if [c["class_id"] for c in chars] != list(range(len(chars))):
        raise ValueError("Class IDs must equal array indexes")
    if any(c["unicode"] != f"U+{ord(c['char']):04X}" for c in chars):
        raise ValueError("Unicode labels do not match characters")
    if len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Pair IDs must be unique")
    edges = [tuple(sorted((p["a"], p["b"]))) for p in pairs]
    if len(edges) != len(set(edges)):
        raise ValueError("Duplicate undirected pairs")
    for a, b in edges:
        if a == b or a not in labels or b not in labels:
            raise ValueError("Invalid pair endpoints")
    return data, sha256(raw)


def common_layout(labels):
    # Every bbox is relative to the same left-baseline anchor, not its own crop.
    for size in range(START_FONT * SCALE, 0, -1):
        font = font_at(size)
        union = bbox_union([font.getbbox(c, anchor="ls") for c in labels])
        if union[2] - union[0] <= 48 * SCALE and union[3] - union[1] <= 48 * SCALE:
            break
    baseline = [math.floor(SIZE * SCALE / 2 - (union[0] + union[2]) / 2 + 0.5),
                math.floor(SIZE * SCALE / 2 - (union[1] + union[3]) / 2 + 0.5)]
    return font, size, union, baseline


def render_high(char, font, baseline, dx, dy):
    # Overscan proves no glyph ink is lost when the 256x256 canvas is extracted.
    pad, side = 64, SIZE * SCALE
    reference = Image.new("L", (side + 2 * pad, side + 2 * pad), 255)
    ImageDraw.Draw(reference).text((baseline[0] + pad + dx * SCALE,
                                   baseline[1] + pad + dy * SCALE),
                                  char, font=font, fill=0, anchor="ls")
    bbox = dark_bbox(reference)
    if bbox is None:
        raise ValueError(f"Blank glyph: {char}")
    cropped = bbox[0] < pad or bbox[1] < pad or bbox[2] > pad + side or bbox[3] > pad + side
    relative = [v - pad for v in bbox]
    return reference.crop((pad, pad, pad + side, pad + side)), relative, cropped


def blur_white(image, sigma):
    if sigma == 0:
        return image.copy()
    # Fixed white padding prevents the boundary-extension rule becoming a variable.
    pad = 16
    return ImageOps.expand(image, border=pad, fill=255).filter(
        ImageFilter.GaussianBlur(radius=sigma)).crop((pad, pad, pad + SIZE, pad + SIZE))


def title(draw, text, y=22, size=26, fill="#182c44"):
    draw.text((26, y), text, font=font_at(size), fill=fill)


def tile(sheet, sample, x, y, zoom=2):
    side = SIZE * zoom
    ImageDraw.Draw(sheet).rectangle((x - 1, y - 1, x + side, y + side), outline="#cbd5e1")
    sheet.paste(sample.resize((side, side), Image.Resampling.NEAREST).convert("RGB"), (x, y))


def make_sheets(output, data, images):
    chars, pairs = data["characters"], data["pairs"]
    files = []
    for sigma_index in [0, 2]:
        # 9 columns; all 81 characters remain legible at 2x pixel replication.
        cols, cw, ch = 9, 156, 172
        sheet = Image.new("RGB", (cols * cw + 48, math.ceil(len(chars) / cols) * ch + 128), "#f5f7fb")
        draw = ImageDraw.Draw(sheet)
        title(draw, f"81 字核验总览 · sigma={SIGMAS[sigma_index]:g} · 居中")
        title(draw, "原图 64×64；此页为 2 倍近邻放大。编号对应 characters.json。", 65, 18, "#536278")
        for i, c in enumerate(chars):
            x, y = 26 + (i % cols) * cw, 114 + (i // cols) * ch
            tile(sheet, images[(c["char"], sigma_index, "c")], x, y)
            draw.text((x, y + 132), f"{c['class_id']:02d}  {c['char']}  {c['unicode'][2:]}", font=font_at(17), fill="#334155")
        name = "overview_clear.png" if sigma_index == 0 else "overview_sigma1.png"
        sheet.save(output / name)
        files.append(name)
    by_char = {c["char"]: c for c in chars}
    for start in range(0, len(pairs), 9):
        subset = pairs[start:start + 9]
        sheet = Image.new("RGB", (1110, 144 + 178 * len(subset)), "#f5f7fb")
        draw = ImageDraw.Draw(sheet)
        title(draw, f"形近字对照 · {start + 1:02d}—{start + len(subset):02d} / {len(pairs)}")
        title(draw, "每档左右顺序一致；原图 64×64，此页 2 倍近邻放大；居中。", 65, 18, "#536278")
        for si, sigma in enumerate(SIGMAS):
            draw.text((224 + si * 288, 107), f"sigma = {sigma:g}", font=font_at(18), fill="#245b8a")
        for row, pair in enumerate(subset):
            y = 140 + row * 178
            a, b = pair["a"], pair["b"]
            draw.text((26, y + 10), f"{pair['pair_id']:02d}  {a} / {b}", font=font_at(24), fill="#182c44")
            draw.text((26, y + 49), pair["category"], font=font_at(17), fill="#536278")
            draw.text((26, y + 77), f"ID {by_char[a]['class_id']:02d}/{by_char[b]['class_id']:02d}", font=font_at(17), fill="#536278")
            for si in range(3):
                for ai, char in enumerate((a, b)):
                    tile(sheet, images[(char, si, "c")], 224 + si * 288 + ai * 136, y)
            draw.line((26, y + 155, 1080, y + 155), fill="#dce3ed")
        name = f"pairs_{start // 9 + 1:02d}.png"
        sheet.save(output / name)
        files.append(name)
    # A compact position sheet complements the all-character interactive view.
    focus = ["日", "曰", "己", "已", "巳", "赢", "羸", "籍", "藉"]
    sheet = Image.new("RGB", (920, 145 + 166 * len(focus)), "#f5f7fb")
    draw = ImageDraw.Draw(sheet)
    title(draw, "偏移核验重点样例 · sigma=1 · 2 倍近邻放大")
    title(draw, "全部 81 字的五位置对照见 index.html；边界自动检查覆盖全部 1,215 图。", 65, 18, "#536278")
    for j, (_, dx, dy) in enumerate(POSITIONS):
        draw.text((178 + 144 * j, 106), f"({dx:+d}, {dy:+d})", font=font_at(18), fill="#245b8a")
    for i, c in enumerate(focus):
        y = 142 + i * 166
        draw.text((30, y + 40), c, font=font_at(34), fill="#182c44")
        for j, (pos, _, _) in enumerate(POSITIONS):
            tile(sheet, images[(c, 2, pos)], 178 + 144 * j, y)
    sheet.save(output / "offsets_focus.png")
    files.append("offsets_focus.png")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "samples" / "v1")
    args = parser.parse_args()
    output = args.output.resolve()
    data, input_hash = checked_input()
    source = json.loads((ROOT / "fonts/source.json").read_text(encoding="utf-8"))
    for resource in source["files"]:
        if sha256((ROOT / "fonts" / resource["name"]).read_bytes()) != resource["sha256"]:
            raise ValueError("Font resource hash mismatch")
    tt = TTFont(FONT_PATH)
    cmap = tt.getBestCmap()
    labels = [c["char"] for c in data["characters"]]
    missing = [c for c in labels if ord(c) not in cmap or tt.getGlyphID(cmap[ord(c)]) == 0]
    if missing:
        raise ValueError(f"Unsupported characters: {missing}")
    font, font_size, union, baseline = common_layout(labels)
    config = {
        "purpose": "字体核验样张，未划分训练/验证/测试集", "input_sha256": input_hash,
        "font_sha256": sha256(FONT_PATH.read_bytes()), "font_name": list(font.getname()),
        "font_version": tt["name"].getDebugName(5), "font_source": source,
        "size": SIZE, "mode": "L", "background": 255, "foreground": 0, "supersampling": SCALE,
        "font_size_high": font_size, "font_size_output": font_size / SCALE,
        "layout_engine": "Pillow BASIC", "anchor": "ls", "union_bbox_high": union,
        "baseline_high": baseline, "common_baseline_rounding": "floor(x+0.5), once for whole charset",
        "resampling": "Lanczos", "sigmas": SIGMAS, "blur": "Pillow GaussianBlur after downsampling",
        "blur_white_padding": 16, "positions": [{"id": p, "dx": x, "dy": y} for p, x, y in POSITIONS],
        "boundary_darkness_threshold": 16, "preview_not_full_position_grid": True,
        "runtime": {"python": platform.python_version(), "Pillow": PIL.__version__,
                    "fonttools": fontTools.__version__, "freetype": features.version_module("freetype2")},
        "renderer_sha256": sha256(Path(__file__).read_bytes()),
    }
    # Do not silently mix a changed character/font/configuration set with old files.
    if (output / "config.json").exists():
        previous = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Changed data configuration or renderer; use a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    images, manifest, geometry = {}, [], []
    hash_groups = defaultdict(list)
    for entry in data["characters"]:
        c, class_id = entry["char"], entry["class_id"]
        folder = output / "images" / f"{class_id:03d}_{ord(c):04X}"
        folder.mkdir(parents=True, exist_ok=True)
        center_bbox, center_high = None, None
        for position, dx, dy in POSITIONS:
            high, high_bbox, cropped = render_high(c, font, baseline, dx, dy)
            if center_bbox is None:
                center_bbox, center_high = high_bbox, high.copy()
            expected_bbox = [center_bbox[0] + dx * SCALE, center_bbox[1] + dy * SCALE,
                             center_bbox[2] + dx * SCALE, center_bbox[3] + dy * SCALE]
            shifted = Image.new("L", high.size, 255)
            shifted.paste(center_high, (dx * SCALE, dy * SCALE))
            geometry.append({"char": c, "position": position, "bbox_high": high_bbox,
                             "cropped": cropped, "bbox_translation_exact": high_bbox == expected_bbox,
                             "pixel_translation_exact": ImageChops.difference(high, shifted).getbbox() is None})
            low = high.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            for si, sigma in enumerate(SIGMAS):
                im = blur_white(low, sigma)
                images[(c, si, position)] = im
                path = folder / f"s{si}_{position}.png"
                im.save(path)
                bbox = dark_bbox(im)
                row = {"class_id": class_id, "char": c, "unicode": entry["unicode"],
                       "sigma": sigma, "position": position, "dx": dx, "dy": dy,
                       "path": path.relative_to(output).as_posix(), "pixel_sha256": sha256(im.tobytes()),
                       "bbox_any_ink": bbox, "bbox_darkness_ge_16": dark_bbox(im, 16),
                       "edge_max_darkness": edge_darkness(im)}
                manifest.append(row)
                hash_groups[row["pixel_sha256"]].append({"char": c, "path": row["path"]})
    duplicates = [g for g in hash_groups.values() if len(g) > 1]
    cross_label = [g for g in duplicates if len({r["char"] for r in g}) > 1]
    pair_distances = []
    for p in data["pairs"]:
        for si, sigma in enumerate(SIGMAS):
            diff = ImageChops.difference(images[(p["a"], si, "c")], images[(p["b"], si, "c")])
            pair_distances.append({"pair_id": p["pair_id"], "a": p["a"], "b": p["b"], "sigma": sigma,
                                   "mean_absolute_pixel_difference": ImageStat.Stat(diff).mean[0]})
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "character_count": len(labels),
             "pair_count": len(data["pairs"]), "sample_count": len(manifest),
             "expected_count": len(labels) * len(SIGMAS) * len(POSITIONS), "font_missing": missing,
             "unique_pixel_images": len(hash_groups), "duplicate_groups": duplicates,
             "cross_label_duplicate_groups": cross_label,
             "high_res_cropped": [r for r in geometry if r["cropped"]],
             "translation_failures": [r for r in geometry if not r["bbox_translation_exact"] or not r["pixel_translation_exact"]],
             "blank_images": [r["path"] for r in manifest if r["bbox_any_ink"] is None],
             "max_edge_darkness": max(r["edge_max_darkness"] for r in manifest),
             "edge_darkness_ge_16": [r["path"] for r in manifest if r["edge_max_darkness"] >= 16],
             "minimum_raw_ink_margin_output_px": min(min(g["bbox_high"][0], g["bbox_high"][1],
                 SIZE * SCALE - g["bbox_high"][2], SIZE * SCALE - g["bbox_high"][3]) / SCALE for g in geometry),
             "pair_pixel_distances_center": pair_distances,
             "interpretation": "像素差仅作相同图检查与定位线索，不是人类可读性或模型难度指标。",
             "geometry": geometry}
    # Independently decode saved files: dimensions, mode and pixel payload must match.
    file_errors = []
    for row in manifest:
        with Image.open(output / row["path"]) as saved:
            if saved.size != (SIZE, SIZE) or saved.mode != "L" or sha256(saved.tobytes()) != row["pixel_sha256"]:
                file_errors.append(row["path"])
    audit["saved_file_errors"] = file_errors
    audit["automatic_checks_passed"] = (
        len(manifest) == audit["expected_count"] and not missing and not duplicates
        and not audit["high_res_cropped"] and not audit["translation_failures"]
        and not audit["blank_images"] and not audit["edge_darkness_ge_16"] and not file_errors)
    save_json(output / "config.json", config)
    save_json(output / "audit.json", audit)
    (output / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    files = make_sheets(output, data, images)
    payload = {"characters": data["characters"], "pairs": data["pairs"], "manifest": manifest,
               "config": config, "audit": {k: v for k, v in audit.items() if k not in ["geometry", "pair_pixel_distances_center"]}}
    template = (ROOT / "gallery_template.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__PILOT_DATA__", encoded), encoding="utf-8")
    print(json.dumps({"output": str(output), "config": config, "automatic_checks_passed": audit["automatic_checks_passed"],
                      "sample_count": len(manifest), "unique_pixel_images": len(hash_groups),
                      "max_edge_darkness": audit["max_edge_darkness"],
                      "minimum_ink_margin_px": audit["minimum_raw_ink_margin_output_px"], "sheets": files}, ensure_ascii=False, indent=2))
    if not audit["automatic_checks_passed"]:
        raise SystemExit("Automatic checks found an issue; inspect audit.json before accepting samples.")


if __name__ == "__main__":
    main()
