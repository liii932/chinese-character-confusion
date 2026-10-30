"""Profile three fresh baseline epochs in Colab without changing existing runs.

Run with the original baseline notebook's WORK, DATA and DRIVE_ROOT paths.
This is a timing experiment, not an additional scientific baseline.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import uuid


def summarize_epochs(passes, saves, wall_times):
    rows = []
    for epoch in range(1, 4):
        stages = {p["phase"]: p for p in passes if p["epoch"] == epoch}
        assert set(stages) == {"train", "val_select"}
        train, val = stages["train"], stages["val_select"]
        epoch_saves = [s for s in saves if s["epoch"] == epoch]
        rows.append({
            "epoch": epoch,
            "train_seconds": train["seconds"],
            "train_images_per_second": train["samples"] / train["seconds"],
            "validation_seconds": val["seconds"],
            "validation_images_per_second": val["samples"] / val["seconds"],
            "checkpoint_save_seconds": sum(s["seconds"] for s in epoch_saves),
            "saved_files": [s["file"] for s in epoch_saves],
            "epoch_wall_seconds_including_saves": wall_times[epoch - 1],
            "peak_allocated_gib": max(p["peak_allocated_gib"] for p in stages.values()),
            "peak_reserved_gib": max(p["peak_reserved_gib"] for p in stages.values()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    work, data, root = args.work.resolve(), args.data.resolve(), args.root.resolve()
    assert data.is_relative_to(work), "Use the local extracted dataset under WORK."
    assert root.is_dir(), "Mount Drive and check DRIVE_ROOT first."
    assert (work / "training/train.py").is_file(), "Run the original notebook setup first."
    sys.path.insert(0, str(work))

    # A separate process keeps notebook tensors out of this process's memory statistics.
    T = importlib.import_module("training.train")
    import torch
    from training.io_utils import digest, verify_dataset, write_json

    assert Path(T.__file__).resolve() == work / "training/train.py"
    assert torch.cuda.is_available(), "Connect a GPU runtime."
    device = torch.device("cuda")
    reference = root / "runs/baseline81_s42_v1"
    config_path = reference / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["mode"] == "baseline" and cfg["epochs"] == 40
    assert cfg["per_class"] is None and cfg["batch_size"] == 64 and cfg["seed"] == 42
    assert cfg["data_sha256"] == digest(data / "manifest.json")
    assert cfg["trainer_sha256"] == digest(T.__file__)
    assert cfg["io_sha256"] == digest(Path(T.__file__).with_name("io_utils.py"))
    assert cfg["environment"] == T.environment(device), "Restore the original runtime environment."
    manifest = verify_dataset(data)
    label_data = json.loads((data / "labels.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["images"] == 15552
    assert manifest["splits"]["val_select"]["images"] == 7776

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out = root / "analyses/baseline81_s42_v1/06_profile" / stamp
    out.mkdir(parents=True, exist_ok=False)
    run_dir = out / "benchmark_run"
    source = Path(__file__).read_bytes()
    (out / "profile_runtime.py").write_bytes(source)
    write_json(out / "reference_config.json", cfg)
    protocol = {
        "purpose": "runtime_profile_only_not_scientific_results",
        "initialization": "fresh_random_seed_42_no_baseline_weights_loaded",
        "planned_scheduler_epochs": cfg["epochs"],
        "measured_epochs": 3,
        "warmup_epoch": 1,
        "steady_estimate_epochs": [2, 3],
        "data_sha256": cfg["data_sha256"],
        "trainer_sha256": cfg["trainer_sha256"],
        "profiler_sha256": digest(out / "profile_runtime.py"),
        "reference_config_sha256": digest(config_path),
        "environment": T.environment(device),
        "notes": [
            "No baseline run is modified; no test evaluation is performed.",
            "Dataset load times describe this runtime; filesystem cache is uncontrolled.",
            "PyTorch allocated/reserved memory belongs to this child process, not the whole GPU.",
            "Save time includes serialization and the Drive mount write/rename; it does not prove remote upload durability.",
            "Epoch wall times come from the original trainer and are rounded to 0.1 seconds.",
            "Initialization, dataset verification and process startup are outside per-epoch wall times.",
        ],
    }
    write_json(out / "protocol.json", protocol)
    print("Profile output:", out, flush=True)
    print("Running three fresh timing epochs; the original baseline is unchanged.", flush=True)

    T.seed_everything(cfg["seed"])
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    memory_before = {
        "device_free_gib": free_bytes / 2**30,
        "device_total_gib": total_bytes / 2**30,
        "this_process_allocated_gib": torch.cuda.memory_allocated() / 2**30,
    }
    loads, passes, saves = [], [], []
    current_epoch = 0
    original_dataset, original_pass, original_save = T.GlyphDataset, T.epoch_pass, T.save_checkpoint

    def measured_dataset(directory, split, per_class=None):
        beginning = time.perf_counter()
        dataset = original_dataset(directory, split, per_class)
        loads.append({"split": split, "samples": len(dataset),
                      "seconds": time.perf_counter() - beginning})
        return dataset

    def measured_pass(model, loader, device, n_classes, optimizer=None, scaler=None):
        nonlocal current_epoch
        training = optimizer is not None
        if training:
            current_epoch += 1
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        beginning = time.perf_counter()
        result = original_pass(model, loader, device, n_classes, optimizer, scaler)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - beginning
        passes.append({
            "epoch": current_epoch, "phase": "train" if training else "val_select",
            "seconds": seconds, "samples": result[0]["samples"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
        return result

    def measured_save(path, state):
        torch.cuda.synchronize()
        beginning = time.perf_counter()
        original_save(path, state)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - beginning
        saves.append({"epoch": state["epoch"], "file": path.name,
                      "seconds": seconds, "size_mib": path.stat().st_size / 2**20})

    train_args = SimpleNamespace(
        data=data, run_dir=run_dir, mode="baseline", seed=cfg["seed"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], per_class=None, resume=False, stop_after=3,
    )
    terminal = sys.stdout
    with (out / "training_stdout.txt").open("w", encoding="utf-8") as log:
        class Tee:
            def write(self, value):
                log.write(value)
                log.flush()
                return terminal.write(value)

            def flush(self):
                log.flush()
                terminal.flush()

        beginning = time.perf_counter()
        with patch.object(T, "GlyphDataset", measured_dataset), \
             patch.object(T, "epoch_pass", measured_pass), \
             patch.object(T, "save_checkpoint", measured_save), \
             contextlib.redirect_stdout(Tee()):
            T.train(train_args, manifest, label_data["characters"], device)
        torch.cuda.synchronize()
        total_train_call_seconds = time.perf_counter() - beginning

    transcript = (out / "training_stdout.txt").read_text(encoding="utf-8")
    wall_times = [float(x) for x in re.findall(r"epoch wall time incl\. saves: ([0-9.]+)s", transcript)]
    assert len(wall_times) == 3 and len(passes) == 6
    rows = summarize_epochs(passes, saves, wall_times)
    steady_keys = ["train_seconds", "train_images_per_second", "validation_seconds",
                   "validation_images_per_second", "checkpoint_save_seconds",
                   "epoch_wall_seconds_including_saves"]
    summary = {
        "purpose": protocol["purpose"], "output_directory": str(out),
        "environment": protocol["environment"], "batch_size": cfg["batch_size"],
        "memory_before": memory_before, "dataset_loads": loads,
        "epochs": rows,
        "median_of_epochs_2_and_3": {
            key: statistics.median(row[key] for row in rows[1:]) for key in steady_keys
        },
        "maximum_allocated_gib": max(row["peak_allocated_gib"] for row in rows),
        "maximum_reserved_gib": max(row["peak_reserved_gib"] for row in rows),
        "three_epoch_train_call_seconds_including_setup": total_train_call_seconds,
        "checkpoint_saves": saves,
    }
    write_json(out / "details.json", {"protocol": protocol, "passes": passes, "summary": summary})
    write_json(out / "summary.json", summary)
    print("\nPROFILE SUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Saved:", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
