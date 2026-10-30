"""First-stage line OCR pilot: Paddle baseline, oracle crops, Edge comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

EXPECTED_CHECKPOINT_SHA256 = "8d65412bf6065253ebce3ca5b0c542788b71e0da391cc335a815f980d828a106"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def norm(text):
    # Only whitespace is removed; punctuation and Unicode variants are retained.
    return "".join(text.split())


def materials(manifest):
    path = Path(manifest).resolve()
    data = read(path)
    files = {page["edge_text"] for page in data["pages"]}
    files.update(line["image"] for page in data["pages"] for line in page["lines"])
    value = {"manifest": digest(path), "files": {f: digest(path.parent / f) for f in sorted(files)}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def directory_digest(directory):
    directory = Path(directory).resolve()
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Empty model directory: {directory}")
    hashes = {str(p.relative_to(directory)): digest(p) for p in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def prepare_pdf(pdf_path, output, dpi):
    import fitz
    doc = fitz.open(pdf_path)
    text_pages = [i + 1 for i, page in enumerate(doc) if page.get_text().strip()]
    if text_pages:
        raise ValueError(f"PDF already contains extractable text on pages {text_pages}; use an image-only PDF")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        pix.save(str(folder / f"page_{i + 1:04d}.png"))
    save(folder / "pdf_info.json", {"pdf_sha256": digest(pdf_path), "pages": len(doc), "dpi": dpi,
                                    "pymupdf": fitz.VersionBind,
                                    "note": "No extractable text found by PyMuPDF; visually review each page."})


def assemble(draft, output):
    path = Path(draft).resolve()
    data = read(path)
    for page in data["pages"]:
        page["lines"] = [read(path.parent / annotation) for annotation in page["lines"]]
    if Path(output).resolve().parent != path.parent:
        raise ValueError("Draft and output must share a directory, so image paths remain valid")
    save(output, data)


def alignment(reference, prediction):
    """Minimum edit alignment, returning predicted index -> reference index.

    Deterministic diagonal-first ties. This uses ground truth and is oracle-only.
    """
    a, b = reference, prediction
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                          d[i - 1][j] + 1, d[i][j - 1] + 1)
    mapping = {}
    i, j = len(a), len(b)
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            mapping[j - 1] = i - 1
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return d[-1][-1], mapping


def validate(manifests):
    from PIL import Image
    seen_pages, sources, counts = set(), {}, {}
    for manifest in manifests:
        path = Path(manifest).resolve()
        data = read(path)
        split = data["split"]
        if split not in {"dev", "test"} or not data["pages"]:
            raise ValueError(f"Invalid split or empty pages: {path}")
        for page in data["pages"]:
            pid, source = page["id"], page["source_id"]
            if not pid or not source or pid in seen_pages:
                raise ValueError(f"Duplicate or empty page/source: {pid}")
            seen_pages.add(pid)
            if source in sources and sources[source] != split:
                raise ValueError(f"Source crosses splits: {source}")
            sources[source] = split
            if page["edge_status"] not in {"ok", "failed"}:
                raise ValueError(f"Invalid Edge status: {pid}")
            edge = (path.parent / page["edge_text"]).read_text(encoding="utf-8-sig")
            if page["edge_status"] == "failed" and norm(edge):
                raise ValueError(f"Failed Edge page must have an empty text file: {pid}")
            if not page["lines"]:
                raise ValueError(f"Page has no annotated lines: {pid}")
            ids = set()
            for line in page["lines"]:
                if line["id"] in ids or not line["id"]:
                    raise ValueError(f"Duplicate/empty line id: {pid}/{line['id']}")
                ids.add(line["id"])
                truth = line["truth"]
                if not truth or norm(truth) != truth or len(line["boxes"]) != len(truth):
                    raise ValueError(f"Truth must be nonempty without whitespace; one box/null per character: {pid}/{line['id']}")
                with Image.open(path.parent / line["image"]) as img:
                    w, h = img.size
                for box in line["boxes"]:
                    if box is not None and (len(box) != 4 or not all(isinstance(x, int) for x in box)
                                              or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
                        raise ValueError(f"Invalid line-relative character box: {pid}/{line['id']}: {box}")
                counts[split] = counts.get(split, 0) + 1
    return counts


def run_paddle(manifest, output, model_name, model_dir, device):
    # The pilot requires a local model directory; do not probe model hosts.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition
    import paddle
    import paddleocr
    path = Path(manifest).resolve()
    data = read(path)
    model_hash = directory_digest(model_dir)
    model = TextRecognition(model_name=model_name, model_dir=str(Path(model_dir).resolve()), device=device)
    lines = {}
    for page in data["pages"]:
        for line in page["lines"]:
            result = list(model.predict(input=str(path.parent / line["image"]), batch_size=1))
            if len(result) != 1:
                raise ValueError(f"Expected one Paddle result: {page['id']}/{line['id']}")
            payload = result[0].json
            result_data = payload["res"]
            lines[f"{page['id']}/{line['id']}"] = {"text": norm(result_data["rec_text"]),
                                                        "score": float(result_data["rec_score"])}
    save(output, {"manifest_sha256": digest(path), "materials_sha256": materials(path),
                  "model": model_name, "model_sha256": model_hash, "device": device,
                  "paddleocr": paddleocr.__version__, "paddlepaddle": paddle.__version__, "lines": lines})


def crop_tensor(image, box):
    import numpy as np
    import torch
    from PIL import Image
    # Place the full annotated crop in a white 64px canvas, max side 48px.
    glyph = image.convert("L").crop(box)
    glyph.thumbnail((48, 48), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 255)
    canvas.paste(glyph, ((64 - glyph.width) // 2, (64 - glyph.height) // 2))
    return torch.from_numpy(np.asarray(canvas).copy()).unsqueeze(0).float().div_(127.5).sub_(1)


def candidates(manifest, baseline, dataset_dir, checkpoint, output):
    import torch
    import torchvision
    import numpy
    import PIL
    from PIL import Image
    from training.train import model_for
    path = Path(manifest).resolve()
    data, base = read(path), read(baseline)
    if base["manifest_sha256"] != digest(path) or base["materials_sha256"] != materials(path):
        raise ValueError("Baseline belongs to a different manifest")
    dataset_dir = Path(dataset_dir)
    dataset_manifest = read(dataset_dir / "manifest.json")
    labels_path = dataset_dir / "labels.json"
    if digest(labels_path) != dataset_manifest["files"]["labels.json"]:
        raise ValueError("Label file does not match dataset manifest")
    if digest(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint does not match the archived epoch-38 baseline")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_data["config"]["mode"] != "baseline" or checkpoint_data["epoch"] != 38:
        raise ValueError("Expected the baseline epoch-38 checkpoint")
    if checkpoint_data["config"]["data_sha256"] != digest(dataset_dir / "manifest.json"):
        raise ValueError("Checkpoint and dataset manifest do not match")
    table = read(labels_path)
    labels = [x["char"] for x in table["characters"]]
    if [x["class_id"] for x in table["characters"]] != list(range(len(labels))):
        raise ValueError("Unexpected label order")
    pairs = {frozenset((x["a"], x["b"])) for x in table["pairs"]}
    model = model_for(len(labels)).eval()
    model.load_state_dict(checkpoint_data["model"])
    result = []
    coverage = {"target_positions": 0, "aligned": 0, "missing_box": 0, "evaluated": 0}
    for page in data["pages"]:
        for line in page["lines"]:
            key = f"{page['id']}/{line['id']}"
            pred = base["lines"][key]["text"]
            _, mapping = alignment(line["truth"], pred)
            coverage["target_positions"] += sum(ch in labels for ch in pred)
            with Image.open(path.parent / line["image"]) as image:
                for p_index, t_index in mapping.items():
                    box = line["boxes"][t_index]
                    if pred[p_index] not in labels:
                        continue
                    coverage["aligned"] += 1
                    if box is None:
                        coverage["missing_box"] += 1
                        continue
                    coverage["evaluated"] += 1
                    with torch.inference_mode():
                        probabilities = model(crop_tensor(image, box).unsqueeze(0)).softmax(1)[0]
                    confidence, index = probabilities.max(0)
                    proposed = labels[int(index)]
                    if proposed != pred[p_index] and frozenset((proposed, pred[p_index])) in pairs:
                        result.append({"line": key, "index": p_index, "truth_index": t_index,
                                       "original": pred[p_index], "proposed": proposed,
                                       "truth": line["truth"][t_index], "score": float(confidence)})
    save(output, {"manifest_sha256": digest(path), "materials_sha256": materials(path), "baseline_sha256": digest(baseline),
                  "checkpoint_sha256": digest(checkpoint), "labels_sha256": digest(labels_path),
                  "epoch": checkpoint_data["epoch"], "torch": str(torch.__version__),
                  "torchvision": str(torchvision.__version__), "numpy": numpy.__version__, "pillow": PIL.__version__,
                  "labels": labels,
                  "coverage": coverage, "candidates": result})


def verify_inputs(manifest, baseline, candidate_data):
    if baseline["manifest_sha256"] != digest(manifest) or candidate_data["manifest_sha256"] != digest(manifest):
        raise ValueError("Manifest mismatch")
    if baseline["materials_sha256"] != materials(manifest) or candidate_data["materials_sha256"] != materials(manifest):
        raise ValueError("Images or Edge text changed since inference")


def apply(text, entries, threshold):
    chars = list(text)
    changed = []
    if threshold is not None:
        for c in entries:
            if c["score"] >= threshold:
                if chars[c["index"]] != c["original"]:
                    raise ValueError("Candidate does not match baseline")
                chars[c["index"]] = c["proposed"]
                changed.append(c)
    return "".join(chars), changed


def page_results(data, base, candidate_data, threshold):
    entries = {}
    for c in candidate_data["candidates"]:
        entries.setdefault(c["line"], []).append(c)
    pages, changes = [], []
    for page in data["pages"]:
        truth, original, revised = [], [], []
        for line in page["lines"]:
            key = f"{page['id']}/{line['id']}"
            text = base["lines"][key]["text"]
            replacement, made = apply(text, entries.get(key, []), threshold)
            truth.append(line["truth"])
            original.append(text)
            revised.append(replacement)
            changes.extend(made)
        t = "".join(truth)
        pages.append({"id": page["id"], "source_id": page["source_id"], "characters": len(t),
                      "A_errors": alignment(t, "".join(original))[0],
                      "B_errors": alignment(t, "".join(revised))[0]})
    return pages, changes


def reference_group_errors(truth, pred, labels):
    _, mapping = alignment(truth, pred)
    prediction_for_truth = {t: pred[p] for p, t in mapping.items()}
    result = {"target": {"characters": 0, "errors": 0}, "outside": {"characters": 0, "errors": 0}}
    for index, char in enumerate(truth):
        group = result["target" if char in labels else "outside"]
        group["characters"] += 1
        group["errors"] += prediction_for_truth.get(index) != char
    return result


def tune(manifest, baseline, candidate_file, output):
    data, base, cs = read(manifest), read(baseline), read(candidate_file)
    if data["split"] != "dev":
        raise ValueError("Threshold selection requires the development split")
    verify_inputs(manifest, base, cs)
    if cs["baseline_sha256"] != digest(baseline):
        raise ValueError("Candidate baseline mismatch")
    # Prefer no replacement on ties, then the stricter threshold.
    grid = [None, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
    trials = []
    for threshold in grid:
        pages, changes = page_results(data, base, cs, threshold)
        trials.append({"threshold": threshold, "errors": sum(x["B_errors"] for x in pages),
                       "changes": len(changes)})
    chosen = min(trials, key=lambda x: (x["errors"], x["changes"]))
    save(output, {"threshold": chosen["threshold"], "dev_manifest_sha256": digest(manifest),
                  "dev_baseline_sha256": digest(baseline), "dev_candidates_sha256": digest(candidate_file),
                  "checkpoint_sha256": cs["checkpoint_sha256"], "labels_sha256": cs["labels_sha256"],
                  "paddle_model": base["model"], "paddleocr": base["paddleocr"],
                  "paddlepaddle": base["paddlepaddle"], "model_sha256": base["model_sha256"],
                  "device": base["device"], "torch": cs["torch"], "torchvision": cs["torchvision"],
                  "numpy": cs["numpy"], "pillow": cs["pillow"], "trials": trials})


def report(manifest, baseline, candidate_file, policy, output):
    data, base, cs, policy_data = read(manifest), read(baseline), read(candidate_file), read(policy)
    verify_inputs(manifest, base, cs)
    if cs["baseline_sha256"] != digest(baseline):
        raise ValueError("Candidate baseline mismatch")
    for key in ("checkpoint_sha256", "labels_sha256", "torch", "torchvision", "numpy", "pillow"):
        if cs[key] != policy_data[key]:
            raise ValueError(f"Policy mismatch: {key}")
    for key in ("paddle_model", "paddleocr", "paddlepaddle", "model_sha256", "device"):
        actual = base["model"] if key == "paddle_model" else base[key]
        if actual != policy_data[key]:
            raise ValueError(f"Paddle configuration differs from development: {key}")
    pages, changes = page_results(data, base, cs, policy_data["threshold"])
    root = Path(manifest).resolve().parent
    entries = {}
    for c in cs["candidates"]:
        entries.setdefault(c["line"], []).append(c)
    grouped = {group: {category: {"characters": 0, "errors": 0} for category in ("target", "outside")}
               for group in ("A", "B", "D")}
    for p, source in zip(pages, data["pages"]):
        edge = norm((root / source["edge_text"]).read_text(encoding="utf-8-sig"))
        truth = "".join(line["truth"] for line in source["lines"])
        original, revised = [], []
        for line in source["lines"]:
            key = f"{source['id']}/{line['id']}"
            text = base["lines"][key]["text"]
            original.append(text)
            revised.append(apply(text, entries.get(key, []), policy_data["threshold"])[0])
        for group, text in (("A", "".join(original)), ("B", "".join(revised)), ("D", edge)):
            summary = reference_group_errors(truth, text, set(cs["labels"]))
            for category in ("target", "outside"):
                for field in ("characters", "errors"):
                    grouped[group][category][field] += summary[category][field]
        p["D_errors"] = alignment(truth, edge)[0]
        p["edge_status"] = source["edge_status"]
    n = sum(p["characters"] for p in pages)
    totals = {group: sum(p[f"{group}_errors"] for p in pages) / n for group in ("A", "B", "D")}
    by_source = {}
    for page in pages:
        by_source.setdefault(page["source_id"], []).append(page)
    groups = list(by_source.values())
    rng = random.Random(20261027)
    intervals = None
    if len(groups) >= 2:
        boot = {"B_minus_A": [], "D_minus_A": [], "D_minus_B": []}
        for _ in range(2000):
            sample = [p for _ in groups for p in rng.choice(groups)]
            size = sum(p["characters"] for p in sample)
            for left, right in (("B", "A"), ("D", "A"), ("D", "B")):
                boot[f"{left}_minus_{right}"].append(sum(p[f"{left}_errors"] - p[f"{right}_errors"] for p in sample) / size)
        intervals = {}
        for key, values in boot.items():
            values.sort()
            intervals[key] = [values[49], values[1949]]
    counts = {"fixed": 0, "broke": 0, "wrong_to_wrong": 0, "truth_outside_labels": 0}
    for c in changes:
        counts["truth_outside_labels"] += c["truth"] not in cs["labels"]
        if c["original"] != c["truth"] and c["proposed"] == c["truth"]:
            counts["fixed"] += 1
        elif c["original"] == c["truth"] and c["proposed"] != c["truth"]:
            counts["broke"] += 1
        elif c["original"] != c["truth"] and c["proposed"] != c["truth"]:
            counts["wrong_to_wrong"] += 1
    save(output, {"split": data["split"], "manifest_sha256": digest(manifest),
                  "baseline_sha256": digest(baseline), "policy_sha256": digest(policy),
                  "characters": n, "sources": len(groups), "pages": pages, "cer": totals,
                  "B_minus_A": totals["B"] - totals["A"],
                  "D_minus_A": totals["D"] - totals["A"], "D_minus_B": totals["D"] - totals["B"],
                  "paired_bootstrap_95": intervals,
                  "coverage": cs["coverage"], "reference_group_errors_excluding_insertions": grouped,
                  "replacement_counts_oracle_aligned": counts, "changes": changes,
                  "caveat": "B uses ground-truth alignment and manual boxes; D sees the PDF page, A/B see manual line crops."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare-pdf"); q.add_argument("pdf"); q.add_argument("output_dir")
    q.add_argument("--dpi", type=int, default=200)
    a = sub.add_parser("assemble"); a.add_argument("draft"); a.add_argument("output")
    v = sub.add_parser("validate"); v.add_argument("manifests", nargs="+")
    p = sub.add_parser("paddle"); p.add_argument("manifest"); p.add_argument("output")
    p.add_argument("--model", default="PP-OCRv5_server_rec"); p.add_argument("--model-dir", required=True)
    p.add_argument("--device", default="cpu")
    c = sub.add_parser("candidates"); c.add_argument("manifest"); c.add_argument("baseline")
    c.add_argument("dataset_dir"); c.add_argument("checkpoint"); c.add_argument("output")
    t = sub.add_parser("tune"); t.add_argument("manifest"); t.add_argument("baseline")
    t.add_argument("candidates"); t.add_argument("output")
    r = sub.add_parser("report"); r.add_argument("manifest"); r.add_argument("baseline")
    r.add_argument("candidates"); r.add_argument("policy"); r.add_argument("output")
    a = parser.parse_args()
    if a.command == "prepare-pdf": prepare_pdf(a.pdf, a.output_dir, a.dpi)
    elif a.command == "assemble": assemble(a.draft, a.output)
    elif a.command == "validate": print(json.dumps(validate(a.manifests), ensure_ascii=False))
    elif a.command == "paddle": run_paddle(a.manifest, a.output, a.model, a.model_dir, a.device)
    elif a.command == "candidates": candidates(a.manifest, a.baseline, a.dataset_dir, a.checkpoint, a.output)
    elif a.command == "tune": tune(a.manifest, a.baseline, a.candidates, a.output)
    elif a.command == "report": report(a.manifest, a.baseline, a.candidates, a.policy, a.output)


if __name__ == "__main__":
    main()
