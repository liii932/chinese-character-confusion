"""Controlled 81-class sample-size curve; prepare/train/summarize in Colab.

Each of ten blocks contains 15,552 draws (243 batches of 64). Subset rows
are repeated equally and shuffled, so every arm has the same batch budget.
The original trainer and original baseline run are never modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

SIZES = (24, 48, 96, 192)
TRAIN_SEED = 42
SUBSET_SEED = 20261012
BLOCKS = 10
BATCH_SIZE = 64
BLOCK_DRAWS = 15552
SUITE_NAME = "steps2430_s42_v1"


def json_hash(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_subsets(train_conditions):
    """Choose nested conditions using geometry only, before reading predictions."""
    groups, attempts = {}, {}
    for sigma in (0.0, 0.5, 1.0):
        ids = [i for i, c in enumerate(train_conditions) if c["sigma"] == sigma]
        if len(ids) != 64:
            raise ValueError("Expected 64 training conditions per sigma")
        rng = random.Random(SUBSET_SEED + int(1000 * sigma))
        for attempt in range(1, 100001):
            candidate = ids.copy()
            rng.shuffle(candidate)
            acceptable = True
            for count in (8, 16, 32, 64):
                for axis in ("dx", "dy"):
                    values = [train_conditions[i][axis] for i in candidate[:count]]
                    if abs(statistics.mean(values)) > 0.5 or min(values) > -2 or max(values) < 2:
                        acceptable = False
            if acceptable:
                groups[str(sigma)] = candidate
                attempts[str(sigma)] = attempt
                break
        else:
            raise RuntimeError("No geometrically balanced nested subset found; do not relax automatically")
    result = {}
    for size in SIZES:
        ids = sorted(i for group in groups.values() for i in group[:size // 3])
        geometry = {}
        for sigma in (0.0, 0.5, 1.0):
            selected = [train_conditions[i] for i in ids if train_conditions[i]["sigma"] == sigma]
            geometry[str(sigma)] = {"conditions": len(selected)}
            for axis in ("dx", "dy"):
                values = [c[axis] for c in selected]
                geometry[str(sigma)][axis] = {
                    "mean": statistics.mean(values), "min": min(values), "max": max(values)}
        result[str(size)] = {"condition_ids": ids, "geometry": geometry}
    return {"subset_seed": SUBSET_SEED, "attempts": attempts, "subsets": result}


class RepeatedRows:
    """Map a balanced multiset to original train rows; return local sample indexes."""
    def __init__(self, source, rows):
        self.source, self.rows = source, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, label, _ = self.source[int(self.rows[index])]
        return image, label, index


def build_rows(dataset, subset, n_classes=81, block_draws=BLOCK_DRAWS):
    import numpy as np
    rows = np.flatnonzero(np.isin(dataset.condition_ids, subset["condition_ids"]))
    size = len(subset["condition_ids"])
    assert len(rows) == size * n_classes
    np.testing.assert_array_equal(np.bincount(dataset.labels[rows], minlength=n_classes),
                                  np.full(n_classes, size))
    for class_id in range(n_classes):
        actual = sorted(dataset.condition_ids[rows][dataset.labels[rows] == class_id].tolist())
        assert actual == subset["condition_ids"]
    assert block_draws % len(rows) == 0
    return rows, np.tile(rows, block_draws // len(rows))


def run_arm(T, torch, train_data, validation, val_sigma, cfg, run_dir, stop_after=None):
    """Save/resume at block boundaries, including optimizer, scaler and RNG state."""
    import numpy as np
    from training.io_utils import write_json
    device = torch.device(cfg["environment"]["device"])
    T.seed_everything(cfg["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    start, history, best_score, best_block = 0, [], (-math.inf, -math.inf), None
    counter = {"updates": 0}
    saved = None
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["config"] != cfg:
            raise ValueError("Resume configuration/code/environment changed; restore the original settings")
        start, history = saved["block"], saved["history"]
        assert [h["block"] for h in history] == list(range(1, start + 1))
        assert best_path.is_file(), "best.pt is missing"
        if start == cfg["blocks"]:
            write_json(run_dir / "history.json", history)
            print(f"Already complete: {run_dir.name}", flush=True)
            return
    elif any(run_dir.iterdir()):
        # A interrupted setup before the first checkpoint is safe to restart only
        # when there are no weight files and the recorded configuration is identical.
        allowed = {"config.json"}
        if {p.name for p in run_dir.iterdir()} - allowed or \
                json.loads((run_dir / "config.json").read_text(encoding="utf-8")) != cfg:
            raise ValueError("Incomplete nonempty run without last.pt; preserve it and report the error")
    write_json(run_dir / "config.json", cfg)
    model = T.model_for(cfg["n_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["blocks"], eta_min=cfg["lr"] / 100)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def count_update(*unused):
        counter["updates"] += 1

    hook = optimizer.register_step_post_hook(count_update)
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best_score, best_block = tuple(saved["best_score"]), saved["best_block"]
        counter["updates"] = saved["optimizer_updates"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    val_loader = T.loader_for(validation, cfg["batch_size"], device)
    stop = min(cfg["blocks"], start + stop_after) if stop_after else cfg["blocks"]
    for block in range(start, stop):
        beginning = time.perf_counter()
        loader = T.loader_for(train_data, cfg["batch_size"], device, seed=cfg["seed"] + block)
        assert len(loader) * cfg["batch_size"] == cfg["block_draws"]
        lr = optimizer.param_groups[0]["lr"]
        updates_before = counter["updates"]
        train_metrics, _, _ = T.epoch_pass(model, loader, device, cfg["n_classes"], optimizer, scaler)
        val_metrics, matrix, prediction = T.epoch_pass(model, val_loader, device, cfg["n_classes"])
        val_metrics["errors"] = int(matrix.sum() - np.trace(matrix))
        val_metrics["worst_class_accuracy"] = float(np.min(np.diag(matrix) / matrix.sum(1)))
        val_metrics["by_sigma"] = {
            str(float(s)): {"samples": int((val_sigma == s).sum()),
                           "accuracy": float(np.mean(prediction[val_sigma == s] == validation.labels[val_sigma == s]))}
            for s in sorted(set(val_sigma.tolist()))}
        scheduler.step()
        record = {
            "block": block + 1,
            "batch_attempts": (block + 1) * len(loader),
            "optimizer_updates": counter["updates"],
            "amp_skipped_updates_this_block": len(loader) - (counter["updates"] - updates_before),
            "effective_passes_over_unique_subset": (block + 1) * cfg["repeat_factor"],
            "lr": lr, "train": train_metrics, "validation_split": "val_select",
            "validation": val_metrics, "seconds_compute": time.perf_counter() - beginning,
        }
        history.append(record)
        score = (val_metrics["macro_accuracy"], -val_metrics["loss"])
        if score > best_score:
            best_score, best_block = score, block + 1
            T.save_checkpoint(best_path, {"config": cfg, "model": model.state_dict(),
                                           "block": best_block, "metrics": val_metrics})
        T.save_checkpoint(last_path, {
            "config": cfg, "block": block + 1, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_score": best_score, "best_block": best_block,
            "optimizer_updates": counter["updates"], "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })
        write_json(run_dir / "history.json", history)
        print(json.dumps({"per_class": cfg["unique_per_class"], "block": block + 1,
                          "of": cfg["blocks"], "val_accuracy": val_metrics["accuracy"],
                          "val_loss": val_metrics["loss"],
                          "amp_skipped_updates": record["amp_skipped_updates_this_block"],
                          "seconds_including_save": round(time.perf_counter() - beginning, 2)}), flush=True)
    hook.remove()


def summarize(suite, plan):
    from training.io_utils import write_json
    rows = []
    for size in SIZES:
        folder = suite / f"n{size:03d}_s42"
        if not (folder / "history.json").is_file():
            rows.append({"per_class": size, "completed_blocks": 0, "complete": False})
            continue
        history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
        best = max(history, key=lambda r: (r["validation"]["macro_accuracy"],
                                           -r["validation"]["loss"], -r["block"]))
        def first_at(threshold):
            return next((r["batch_attempts"] for r in history
                         if r["validation"]["macro_accuracy"] >= threshold), None)
        rows.append({
            "per_class": size, "unique_training_images": size * 81,
            "completed_blocks": len(history), "complete": len(history) == plan["blocks"],
            "last_completed_batch_attempts": history[-1]["batch_attempts"],
            "optimizer_updates": history[-1]["optimizer_updates"],
            "amp_skipped_updates": sum(r["amp_skipped_updates_this_block"] for r in history),
            "first_observed_99_9pct_at_batch": first_at(0.999),
            "first_observed_100pct_at_batch": first_at(1.0),
            "last_completed_validation": history[-1]["validation"],
            "best_block": best["block"], "best_validation": best["validation"],
            "compute_minutes": sum(r["seconds_compute"] for r in history) / 60,
        })
    summary = {"suite": str(suite), "purpose": plan["purpose"],
               "all_complete": all(r["complete"] for r in rows),
               "planned_batch_attempts_per_arm": plan["blocks"] * plan["block_draws"] // plan["batch_size"],
               "comparison": "Primary: last block at fixed budget. Secondary: best val_select checkpoint.",
               "single_seed_exploratory": True, "results": rows}
    write_json(suite / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "summarize"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, choices=SIZES, nargs="+", default=[192, 96, 48, 24])
    parser.add_argument("--stop-after-blocks", type=int)
    args = parser.parse_args()
    if args.stop_after_blocks is not None and args.stop_after_blocks <= 0:
        parser.error("stop-after-blocks must be positive")
    work, data, root = args.work.resolve(), args.data.resolve(), args.root.resolve()
    assert data.is_relative_to(work) and root.is_dir()
    sys.path.insert(0, str(work))
    T = importlib.import_module("training.train")
    import numpy as np
    import torch
    from training.io_utils import digest, write_json
    assert Path(T.__file__).resolve() == work / "training/train.py"
    assert torch.cuda.is_available(), "Connect the original GPU runtime"
    device = torch.device("cuda")
    reference_cfg = json.loads((root / "runs/baseline81_s42_v1/config.json").read_text(encoding="utf-8"))
    assert reference_cfg["mode"] == "baseline" and reference_cfg["seed"] == TRAIN_SEED
    assert reference_cfg["batch_size"] == BATCH_SIZE
    assert reference_cfg["environment"] == T.environment(device), "Restore original environment"
    assert digest(T.__file__) == reference_cfg["trainer_sha256"]
    assert digest(Path(T.__file__).with_name("io_utils.py")) == reference_cfg["io_sha256"]
    assert digest(data / "manifest.json") == reference_cfg["data_sha256"]
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    # Only the training and selection split are decoded; graph/test are not evaluated.
    for name in ("train.npz", "val_select.npz", "conditions.json", "labels.json"):
        assert digest(data / name) == manifest["files"][name], name
    conditions = json.loads((data / "conditions.json").read_text(encoding="utf-8"))
    labels = json.loads((data / "labels.json").read_text(encoding="utf-8"))["characters"]
    assert len(labels) == 81 and [c["class_id"] for c in labels] == list(range(81))
    raw_train = T.GlyphDataset(data, "train")
    assert len(raw_train) == BLOCK_DRAWS
    selections = choose_subsets(conditions["train"])
    row_maps = {}
    for size in SIZES:
        subset = selections["subsets"][str(size)]
        unique, repeated = build_rows(raw_train, subset)
        subset["unique_rows_sha256"] = json_hash(unique.tolist())
        subset["repeated_rows_sha256"] = json_hash(repeated.tolist())
        subset["unique_images"] = len(unique)
        row_maps[size] = repeated
    for small, large in zip(SIZES, SIZES[1:]):
        assert set(selections["subsets"][str(small)]["condition_ids"]) < \
               set(selections["subsets"][str(large)]["condition_ids"])
    np.testing.assert_array_equal(row_maps[192], np.arange(BLOCK_DRAWS))
    plan = {
        "purpose": "single_seed_81_class_sample_size_curve_fixed_batch_budget",
        "sizes": list(SIZES), "seed": TRAIN_SEED, "n_classes": 81,
        "blocks": BLOCKS, "block_draws": BLOCK_DRAWS, "batch_size": BATCH_SIZE,
        "lr": reference_cfg["lr"], "weight_decay": reference_cfg["weight_decay"],
        "model": reference_cfg["model"], "loss": "cross_entropy",
        "scheduler": "CosineAnnealingLR per 243-batch block; T_max=10; eta_min=lr/100",
        "sampling": "Repeat each unique row 192/n times, then shuffle all 15552 draws per block",
        "selection": "val_select macro accuracy, then lower loss, then earlier block",
        "primary_endpoint": "Last block at batch attempt 2430; selected-best is secondary",
        "amp": True, "track_actual_optimizer_updates": True,
        "environment": T.environment(device), "data_sha256": reference_cfg["data_sha256"],
        "trainer_sha256": reference_cfg["trainer_sha256"], "io_sha256": reference_cfg["io_sha256"],
        "curve_script_sha256": digest(Path(__file__)), "subsets": selections,
        "notes": ["No val_graph/test evaluation; not a final generalization claim.",
                  "Budget matches batch attempts and sample draws, not unique-data epochs.",
                  "Counts are nested and geometry-balanced; only one subset seed and one training seed.",
                  "The old 40-epoch baseline is not the 192-image arm of this new schedule."],
    }
    suite = root / "analyses/baseline81_s42_v1/03_learning_curve" / SUITE_NAME
    suite.mkdir(parents=True, exist_ok=True)
    plan_path = suite / "plan.json"
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Frozen experiment plan changed; restore its original script/data/environment")
    elif any(suite.iterdir()):
        raise ValueError("Suite exists without plan.json; preserve it and report the error")
    else:
        write_json(plan_path, plan)
    # A matching plan authorizes restoring missing descriptive files after setup interruption.
    (suite / "learning_curve.py").write_bytes(Path(__file__).read_bytes())
    write_json(suite / "subsets.json", selections)
    print("Suite:", suite, flush=True)
    if args.command == "prepare":
        print(json.dumps({"prepared": True, "batch_attempts_per_arm": 2430,
                          "arms": [{"per_class": n, "unique_images": n * 81,
                                    "per_sigma_per_class": n // 3,
                                    "effective_subset_passes": BLOCKS * 192 // n} for n in SIZES],
                          "subsets_file": str(suite / "subsets.json")}, indent=2), flush=True)
        return
    if args.command == "train":
        validation = T.GlyphDataset(data, "val_select")
        assert len(validation) == 7776
        val_sigma = np.array([conditions["val_select"][int(i)]["sigma"] for i in validation.condition_ids])
        for size in args.sizes:
            subset = selections["subsets"][str(size)]
            cfg = {k: v for k, v in plan.items() if k not in ("sizes", "subsets", "notes")}
            cfg.update({"suite_plan_sha256": digest(plan_path), "unique_per_class": size,
                        "unique_training_images": 81 * size, "repeat_factor": 192 // size,
                        "subset": subset})
            view = RepeatedRows(raw_train, row_maps[size])
            run_arm(T, torch, view, validation, val_sigma, cfg,
                    suite / f"n{size:03d}_s42", args.stop_after_blocks)
            gc.collect()
            torch.cuda.empty_cache()
    summarize(suite, plan)


if __name__ == "__main__":
    main()
