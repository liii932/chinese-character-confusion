"""Phase 1: a fixed development-only blur/resolution sweep and frozen-model evaluation."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import random

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from training.io_utils import digest, write_json

SIGMAS = [0, 1, 1.5, 2, 2.5]
RESOLUTIONS = [64, 48, 32, 24, 16]
SEED = 20261020
BASE_MANIFEST_SHA256 = "6a2d237ae03ed219f76c2d4cf827eeb6b9c852c88c54a5a77d6ca34706d84f86"
CHECKPOINT_SHA256 = "8d65412bf6065253ebce3ca5b0c542788b71e0da391cc335a815f980d828a106"
ROOT = Path(__file__).resolve().parents[1]


def degrade(image, sigma, resolution):
    """Sigma is in 64x64 pixels; retain the original renderer's white boundary."""
    if sigma:
        image = ImageOps.expand(image, border=16, fill=255).filter(
            ImageFilter.GaussianBlur(sigma)).crop((16, 16, 80, 80))
    if resolution != 64:
        image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        image = image.resize((64, 64), Image.Resampling.BICUBIC)
    return image.copy()


def select_conditions(conditions):
    clear = [i for i, c in enumerate(conditions) if c["sigma"] == 0]
    center = next(i for i in clear if conditions[i]["dx"] == conditions[i]["dy"] == 0)
    rng = random.Random(SEED)
    # Offline geometric balancing only; no model predictions enter selection.
    for _ in range(10000):
        ids = sorted([center] + rng.sample([i for i in clear if i != center], 15))
        axes = [[conditions[i][axis] for i in ids] for axis in ("dx", "dy")]
        if all(abs(sum(v) / len(v)) <= .25 and min(v) <= -2.5 and max(v) >= 2.5 for v in axes):
            return ids
    raise ValueError("Could not select 16 balanced, shared development conditions")


def generate(data, output):
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new calibration directory: {output}")
    if digest(data / "manifest.json") != BASE_MANIFEST_SHA256:
        raise ValueError("Phase 1 requires the archived baseline81-v2 data manifest")
    source_manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    # Only these development inputs are opened. Do not load train/test arrays.
    for name in ("val_graph.npz", "conditions.json", "labels.json"):
        if digest(data / name) != source_manifest["files"][name]:
            raise ValueError(f"Changed source data: {name}")
    label_data = json.loads((data / "labels.json").read_text(encoding="utf-8"))
    conditions = json.loads((data / "conditions.json").read_text(encoding="utf-8"))["val_graph"]
    ids = select_conditions(conditions)
    with np.load(data / "val_graph.npz", allow_pickle=False) as source:
        indexes = np.concatenate([np.flatnonzero(source["condition_ids"] == i) for i in ids])
        base = source["images"][indexes]
        labels = source["labels"][indexes]
        source_condition_ids = source["condition_ids"][indexes]
    np.testing.assert_array_equal(labels, np.tile(np.arange(81), 16))
    grids = [{"grid_id": i, "sigma": s, "resolution": r}
             for i, (s, r) in enumerate((s, r) for s in SIGMAS for r in RESOLUTIONS)]
    images = np.stack([np.stack([np.asarray(degrade(Image.fromarray(x), g["sigma"], g["resolution"]))
                                for x in base]) for g in grids])
    np.testing.assert_array_equal(images[0], base)
    audit = audit_pixels(images, labels)
    if audit["blank_images"] or audit["cross_label_collisions"]:
        raise ValueError(f"Invalid calibration pixels: {audit}")
    output.mkdir(parents=True)
    np.savez_compressed(output / "sweep.npz", images=images, labels=labels,
                        source_indexes=indexes, source_condition_ids=source_condition_ids)
    (output / "labels.json").write_bytes((data / "labels.json").read_bytes())
    manifest = {
        "schema": 1, "purpose": "development_calibration_only", "dataset": "degradation81-calibration-v1",
        "source_split": "val_graph", "source_manifest_sha256": BASE_MANIFEST_SHA256,
        "selection_seed": SEED, "source_condition_ids": ids,
        "conditions": [{"source_condition_id": i, **conditions[i]} for i in ids],
        "source_group_key": "baseline81-v2/val_graph/source_index; all 25 derivatives stay together",
        "grids": grids, "shape": list(images.shape), "samples": int(images.shape[0] * images.shape[1]),
        "characters": len(label_data["characters"]), "pairs": len(label_data["pairs"]),
        "samples_per_character_per_grid": 16, "normalization": "uint8 / 127.5 - 1",
        "pipeline": ["archived clear 64x64", "white-pad16 Gaussian(sigma in 64px units)",
                     "Lanczos to R", "bicubic to 64; bypass both resizes when R=64"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "Pillow": PIL.__version__},
        "files": {name: digest(output / name) for name in ("sweep.npz", "labels.json")},
        "pixel_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "model_evaluation": "pending_original_checkpoint", "human_readability": "not_measured",
        "interpretation": "Related synthetic variants, not independent real glyphs. No training or final test.",
    }
    audit.update({"clean_identity": True, "source_split": "val_graph", "test_arrays_opened": False,
                  "position_statistics": {a: {"mean": float(np.mean([conditions[i][a] for i in ids])),
                      "min": min(conditions[i][a] for i in ids), "max": max(conditions[i][a] for i in ids)}
                      for a in ("dx", "dy")}})
    write_json(output / "audit.json", audit)
    write_json(output / "manifest.json", manifest)
    return manifest, audit


def audit_pixels(images, labels):
    seen, collisions, duplicate_count = {}, [], 0
    for grid_id, batch in enumerate(images):
        for base_id, pixels in enumerate(batch):
            key = hashlib.sha256(pixels.tobytes()).digest()
            value = (int(labels[base_id]), grid_id, base_id)
            if key in seen:
                duplicate_count += 1
                if seen[key][0] != value[0]:
                    collisions.append({"first": seen[key], "second": value})
            else:
                seen[key] = value
    edge = np.concatenate((images[:, :, 0, :], images[:, :, -1, :],
                           images[:, :, :, 0], images[:, :, :, -1]), axis=-1)
    return {"images": int(np.prod(images.shape[:2])), "unique_pixels": len(seen),
            "same_label_duplicate_images": duplicate_count - len(collisions),
            "cross_label_collisions": collisions,
            "blank_images": int(np.sum(np.min(images, axis=(-2, -1)) == 255)),
            "maximum_border_darkness": int(255 - edge.min()),
            "images_with_border_darkness_ge_16": int(np.sum(255 - edge.min(axis=-1) >= 16)),
            "note": "Pixel uniqueness is a data audit, not a readability or difficulty measurement."}


def make_gallery(directory, font_path):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    with np.load(directory / "sweep.npz", allow_pickle=False) as saved:
        images, source_ids, truth = saved["images"], saved["source_condition_ids"], saved["labels"]
    center = next(c["source_condition_id"] for c in manifest["conditions"] if c["dx"] == c["dy"] == 0)
    by_char = {c["char"]: np.flatnonzero((source_ids == center) & (truth == c["class_id"]))[0]
               for c in labels["characters"]}
    preview = directory / "preview"
    preview.mkdir(exist_ok=True)
    font = ImageFont.truetype(str(font_path), 20)
    small = ImageFont.truetype(str(font_path), 15)
    for pair in labels["pairs"]:
        sheet = Image.new("RGB", (850, 650), "#f4f6fa")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 12), f"{pair['a']} / {pair['b']} · 25 档退化标定", font=font, fill="#182c44")
        draw.text((20, 44), "每字原尺寸 64×64；左／右顺序固定。此页有标签，不是人工盲认测试。", font=small, fill="#526175")
        draw.text((20, 70), pair["region"], font=small, fill="#526175")
        for g in manifest["grids"]:
            gi = g["grid_id"]
            x, y = 20 + (gi % 5) * 166, 110 + (gi // 5) * 106
            draw.text((x, y), f"σ={g['sigma']:g} / R={g['resolution']}", font=small, fill="#526175")
            for j, char in enumerate((pair["a"], pair["b"])):
                sheet.paste(Image.fromarray(images[gi, by_char[char]]).convert("RGB"), (x + j * 72, y + 26))
        sheet.save(preview / f"pair_{pair['pair_id']:02d}.png")
    options = "".join(f'<option value="{p["pair_id"]:02d}">{p["pair_id"]:02d} · {p["a"]} / {p["b"]}</option>'
                      for p in labels["pairs"])
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>81 字退化标定</title>
<style>body{{font:16px system-ui;background:#f4f6fa;color:#182c44;max-width:1100px;margin:30px auto;padding:0 16px}}
select,button{{font:inherit;padding:8px}}img{{display:block;margin-top:16px;max-width:100%}}.zoom{{max-width:none;width:1700px;image-rendering:pixelated}}</style>
<h1>81 字 · 第一阶段退化标定</h1><p>25 档 × 81 字 × 16 个共享位移 = 32,400 张开发图。模型推理待在 Colab 执行；人类可读性尚未测量。</p>
<p>以下展示居中条件。σ 按 64×64 像素计；R 是中间分辨率。全体类别共用条件，无关键笔画擦除。</p>
<label>重点字对 <select id="pair">{options}</select></label> <button id="zoom">切换 2 倍像素放大</button>
<img id="sheet" alt="字对在 25 档条件下的实际像素" src="pair_01.png">
<script>document.getElementById('pair').onchange=e=>document.getElementById('sheet').src='pair_'+e.target.value+'.png';
document.getElementById('zoom').onclick=()=>document.getElementById('sheet').classList.toggle('zoom');</script></html>'''
    (preview / "index.html").write_text(html, encoding="utf-8")


def summarize(truth, prediction, losses, label_data):
    n_classes = len(label_data["characters"])
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    count = matrix.sum(axis=1)
    accuracy = np.diag(matrix) / count
    char_to_id = {c["char"]: c["class_id"] for c in label_data["characters"]}
    pairs = []
    for p in label_data["pairs"]:
        a, b = char_to_id[p["a"]], char_to_id[p["b"]]
        pairs.append({"pair_id": p["pair_id"], "a": p["a"], "b": p["b"],
                      "a_to_b": int(matrix[a, b]), "b_to_a": int(matrix[b, a]),
                      "pair_error_rate": float((matrix[a, b] + matrix[b, a]) / (count[a] + count[b]))})
    # Original S rule: for each source class choose its most frequent wrong target.
    wrong = matrix.copy()
    np.fill_diagonal(wrong, 0)
    target = wrong.argmax(axis=1)
    support = [{"source_id": i, "target_id": int(target[i]), "count": int(wrong[i, target[i]])}
               for i in range(n_classes) if wrong[i, target[i]] >= 5]
    return {"samples": len(truth), "errors": int(np.sum(truth != prediction)),
            "accuracy": float(np.mean(truth == prediction)), "macro_accuracy": float(accuracy.mean()),
            "loss": float(np.mean(losses)), "per_class_accuracy": accuracy.tolist(),
            "supported_sources_ge5": support, "pairs": pairs}, matrix


def write_report(output, rows, wrong_examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for resolution in RESOLUTIONS:
        group = [r for r in rows if r["resolution"] == resolution]
        ax.plot([r["sigma"] for r in group], [100 * r["macro_accuracy"] for r in group],
                marker="o", label=f"R={resolution}")
    ax.set(xlabel="Gaussian sigma (64x64 pixels)", ylabel="Macro accuracy (%)", ylim=(0, 101),
           title="Frozen baseline / development calibration")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy_curves.png", dpi=160)
    plt.close(fig)
    table = "".join(f'<tr><td>{r["sigma"]}</td><td>{r["resolution"]}</td><td>{r["macro_accuracy"]:.2%}</td>'
                    f'<td>{r["errors"]}/{r["samples"]}</td><td>{len(r["supported_sources_ge5"])}</td></tr>' for r in rows)
    cards = []
    for e in wrong_examples:
        stream = io.BytesIO()
        Image.fromarray(e["pixels"]).save(stream, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        cards.append(f'<figure><img src="{url}" width="64" height="64"><figcaption>{e["true"]} → {e["predicted"]}'
                     f'<br>σ={e["sigma"]}, R={e["resolution"]}<br>source={e["source_index"]}</figcaption></figure>')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>退化标定结果</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 16px;color:#182c44}}td,th{{padding:7px 14px;border-bottom:1px solid #ddd}}
.cards{{display:flex;flex-wrap:wrap}}figure{{margin:12px;width:145px}}figcaption{{font-size:13px}}img{{max-width:100%}}</style>
<h1>第一阶段：旧模型退化扫描</h1><p>仅反映旧模型遇到新退化后的表现。尚未完成同分布重训或人类可读性标定，不能据此选定正式训练难度。</p>
<img src="accuracy_curves.png" alt="准确率随模糊强度变化"><table><tr><th>σ</th><th>R</th><th>宏准确率</th><th>错误数</th><th>有≥5次支持的源类别</th></tr>{table}</table>
<h2>错误样例</h2><p>按档位、来源顺序，每档最多 4 张；完整记录见 predictions.csv。此处不是盲认测试。</p><div class="cards">{''.join(cards)}</div></html>'''
    (output / "report.html").write_text(html, encoding="utf-8")


def evaluate(directory, checkpoint_path, output, batch_size=128, device_name="auto",
             expected_checkpoint_sha256=CHECKPOINT_SHA256):
    import torch
    from training import train as T

    directory, checkpoint_path, output = Path(directory), Path(checkpoint_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Keep the previous report; use a new output directory: {output}")
    if digest(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("Checkpoint does not match the archived epoch-38 baseline")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Calibration file changed: {name}")
    label_data = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["mode"] != "baseline" or checkpoint["config"]["data_sha256"] != manifest["source_manifest_sha256"]:
        raise ValueError("Checkpoint is not the source-data baseline")
    if checkpoint["config"]["trainer_sha256"] != digest(Path(T.__file__)):
        raise ValueError("Restore the archived model implementation before evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(2)
    T.seed_everything(42)
    model = T.model_for(len(label_data["characters"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with np.load(directory / "sweep.npz", allow_pickle=False) as saved:
        images, truth = saved["images"], saved["labels"]
        source_indexes, source_ids = saved["source_indexes"], saved["source_condition_ids"]
    output.mkdir(parents=True)
    rows, matrices, wrong_examples = [], [], []
    chars = [c["char"] for c in label_data["characters"]]
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["grid_id", "sigma", "resolution", "source_index", "source_condition_id", "true_id",
                         "true_char", "predicted_id", "predicted_char", "confidence", "true_probability_margin", "loss"])
        for grid in manifest["grids"]:
            batch_images = images[grid["grid_id"]]
            predicted, losses, confidences, margins = [], [], [], []
            with torch.inference_mode():
                for start in range(0, len(truth), batch_size):
                    x = torch.from_numpy(batch_images[start:start + batch_size].copy()).unsqueeze(1).float().div_(127.5).sub_(1).to(device)
                    y = torch.as_tensor(truth[start:start + batch_size], device=device)
                    logits = model(x)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite logits in calibration inference")
                    loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
                    probabilities = logits.softmax(1)
                    confidence, pred = probabilities.max(1)
                    true_probability = probabilities.gather(1, y[:, None]).squeeze(1)
                    probabilities.scatter_(1, y[:, None], -1)
                    margin = true_probability - probabilities.max(1).values
                    predicted.extend(pred.cpu().tolist())
                    losses.extend(loss.cpu().tolist())
                    confidences.extend(confidence.cpu().tolist())
                    margins.extend(margin.cpu().tolist())
            predicted = np.asarray(predicted)
            metrics, matrix = summarize(truth, predicted, losses, label_data)
            row = {**grid, **metrics}
            rows.append(row)
            matrices.append(matrix)
            for i, pred in enumerate(predicted):
                writer.writerow([grid["grid_id"], grid["sigma"], grid["resolution"], int(source_indexes[i]),
                                 int(source_ids[i]), int(truth[i]), chars[truth[i]], int(pred), chars[pred],
                                 confidences[i], margins[i], losses[i]])
            for i in np.flatnonzero(predicted != truth)[:4]:
                wrong_examples.append({"pixels": batch_images[i], "true": chars[truth[i]], "predicted": chars[predicted[i]],
                                       "sigma": grid["sigma"], "resolution": grid["resolution"], "source_index": int(source_indexes[i])})
            print(f"[{grid['grid_id'] + 1:02d}/25] sigma={grid['sigma']:g} R={grid['resolution']}: "
                  f"accuracy={metrics['macro_accuracy']:.2%}, errors={metrics['errors']}/{metrics['samples']}", flush=True)
    write_json(output / "metrics.json", rows)
    np.savez_compressed(output / "confusion.npz", matrices=np.stack(matrices))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["grid_id", "sigma", "resolution", "samples", "errors", "macro_accuracy", "loss", "supported_sources_ge5"])
        for r in rows:
            writer.writerow([r[k] for k in ("grid_id", "sigma", "resolution", "samples", "errors", "macro_accuracy", "loss")]
                            + [len(r["supported_sources_ge5"])])
    with (output / "pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["grid_id", "sigma", "resolution", "pair_id", "a", "b", "a_to_b", "b_to_a", "pair_error_rate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for pair in row["pairs"]:
                writer.writerow({**{k: row[k] for k in ("grid_id", "sigma", "resolution")}, **pair})
    write_report(output, rows, wrong_examples)
    summary = {"status": "completed_frozen_model_sweep", "samples": sum(r["samples"] for r in rows),
               "grids": len(rows), "checkpoint_epoch": checkpoint["epoch"],
               "checkpoint_sha256": expected_checkpoint_sha256, "calibration_manifest_sha256": digest(directory / "manifest.json"),
               "pixel_sha256": manifest["pixel_sha256"], "environment": T.environment(device),
               "original_training_environment": checkpoint["config"]["environment"], "precision": "fp32_no_autocast",
               "clean_control_accuracy": rows[0]["macro_accuracy"], "clean_control_matches_archived": rows[0]["errors"] == 0,
               "evaluator_sha256": digest(__file__), "human_readability": "not_measured",
               "retrained_baseline": "not_run", "test_evaluated": False,
               "interpretation": "New-corruption evaluation of the old model; not final-test or matched-training evidence."}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("generate")
    build.add_argument("--data", type=Path, default=ROOT / "training/data/baseline81-v2")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--font", type=Path, help="Also build all 54 pair previews using this font")
    run = sub.add_parser("evaluate")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.command == "generate":
        manifest, audit = generate(args.data, args.output)
        if args.font:
            make_gallery(args.output, args.font)
        print(json.dumps({"output": str(args.output), "samples": manifest["samples"], "audit": audit}, ensure_ascii=False, indent=2))
    else:
        if args.batch_size <= 0:
            parser.error("batch-size must be positive")
        print(json.dumps(evaluate(args.data, args.checkpoint, args.output, args.batch_size, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
