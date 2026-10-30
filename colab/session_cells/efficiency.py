"""Analyse saved three-epoch T4 measurements using only the Python standard library.

The notebook generator embeds this file into cell 3; it does not run profiling.
"""
import math
import statistics


def efficiency_positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"计时报告字段无效：{name}")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"计时报告字段应为正数：{name}")
    return float(value)


def efficiency_equal(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"计时报告内部不一致：{name}")


def analyze_efficiency(profile, baseline=None, reference_environment=None):
    """Validate measurements, derive ratios, and label assumptions explicitly."""
    if profile["purpose"] != "runtime_profile_only_not_scientific_results":
        raise ValueError("不是原独立计时实验报告")
    if profile["batch_size"] != 64 or profile["environment"]["gpu"] != "Tesla T4":
        raise ValueError("当前分析对应原 T4 / batch=64 计时实验")
    if reference_environment is not None and profile["environment"] != reference_environment:
        raise ValueError("计时报告环境与基线配置不一致")
    epochs = profile["epochs"]
    if [r["epoch"] for r in epochs] != [1, 2, 3]:
        raise ValueError("需要完整的三轮计时报告；不将部分测量标记完成")
    loads = {r["split"]: r for r in profile["dataset_loads"]}
    if loads["train"]["samples"] != 15552 or loads["val_select"]["samples"] != 7776:
        raise ValueError("计时报告数据规模与 81 字基线不一致")
    for value in loads.values():
        efficiency_positive(value["seconds"], "dataset_load_seconds", allow_zero=True)
    timing_keys = ("train_seconds", "validation_seconds", "checkpoint_save_seconds",
                   "epoch_wall_seconds_including_saves")
    rate_keys = ("train_images_per_second", "validation_images_per_second")
    total_memory = efficiency_positive(profile["memory_before"]["device_total_gib"], "device_total_gib")
    for row in epochs:
        for key in timing_keys + rate_keys + ("peak_allocated_gib", "peak_reserved_gib"):
            efficiency_positive(row[key], key)
        efficiency_equal(row["train_images_per_second"], 15552 / row["train_seconds"], "train throughput")
        efficiency_equal(row["validation_images_per_second"], 7776 / row["validation_seconds"], "validation throughput")
        if not row["peak_allocated_gib"] <= row["peak_reserved_gib"] <= total_memory:
            raise ValueError("显存统计关系异常")
        stages = row["train_seconds"] + row["validation_seconds"] + row["checkpoint_save_seconds"]
        if stages > row["epoch_wall_seconds_including_saves"] + .1:
            raise ValueError("阶段耗时超过整轮 wall time")
        saves = [s for s in profile["checkpoint_saves"] if s["epoch"] == row["epoch"]]
        if sorted(s["file"] for s in saves) != sorted(row["saved_files"]) or "last.pt" not in row["saved_files"]:
            raise ValueError("检查点保存记录不完整")
        for save in saves:
            efficiency_positive(save["size_mib"], "checkpoint size")
            efficiency_positive(save["seconds"], "checkpoint time", allow_zero=True)
        efficiency_equal(sum(s["seconds"] for s in saves), row["checkpoint_save_seconds"], "checkpoint total")
    steady = {key: statistics.median(row[key] for row in epochs[1:]) for key in timing_keys + rate_keys}
    for key, value in steady.items():
        efficiency_equal(profile["median_of_epochs_2_and_3"][key], value, key)
    allocated = max(row["peak_allocated_gib"] for row in epochs)
    reserved = max(row["peak_reserved_gib"] for row in epochs)
    efficiency_equal(profile["maximum_allocated_gib"], allocated, "maximum allocated")
    efficiency_equal(profile["maximum_reserved_gib"], reserved, "maximum reserved")
    call_seconds = efficiency_positive(profile["three_epoch_train_call_seconds_including_setup"], "three_epoch_call")
    if call_seconds + .2 < sum(r["epoch_wall_seconds_including_saves"] for r in epochs):
        raise ValueError("三轮总调用时长小于逐轮 wall time 之和")
    wall = steady["epoch_wall_seconds_including_saves"]
    phases = {"train": steady["train_seconds"], "validation": steady["validation_seconds"],
              "checkpoint_save": steady["checkpoint_save_seconds"]}
    phases["other_and_rounding"] = wall - sum(phases.values())
    phase_percent = {k: value / wall * 100 for k, value in phases.items()}
    checkpoint_sizes = {name: max(s["size_mib"] for s in profile["checkpoint_saves"] if s["file"] == name)
                        for name in ("best.pt", "last.pt")}
    ratio = 1140 / 81
    measured_compute = steady["train_seconds"] + steady["validation_seconds"]
    observed_baseline = None
    if baseline and "completed_epochs" in baseline:
        observed_baseline = {k: baseline[k] for k in (
            "completed_epochs", "planned_epochs", "compute_minutes_excluding_saves",
            "steady_median_seconds", "thresholds") if k in baseline}
    return {
        "schema": 1, "status": "complete_saved_measurement_analysis",
        "scope": "81-class T4 batch64 runtime analysis; no new training or model evaluation",
        "environment": profile["environment"], "batch_size": 64,
        "measured_epochs": 3, "steady_epochs": [2, 3],
        "samples_per_epoch": {"train": 15552, "validation": 7776},
        "steady": steady,
        "steady_observed_epoch_wall_range_seconds": {
            "min": min(r["epoch_wall_seconds_including_saves"] for r in epochs[1:]),
            "max": max(r["epoch_wall_seconds_including_saves"] for r in epochs[1:])},
        "phase_seconds": phases,
        "phase_percent_of_steady_wall": phase_percent,
        "memory": {"device_total_gib": total_memory, "peak_allocated_gib": allocated,
                   "peak_reserved_gib": reserved, "allocated_percent_of_capacity": allocated / total_memory * 100,
                   "reserved_percent_of_capacity": reserved / total_memory * 100,
                   "reserved_includes_allocated": True, "gpu_utilization_percent": None},
        "storage": {"checkpoint_sizes_mib": checkpoint_sizes,
                    "best_plus_last_mib": sum(checkpoint_sizes.values()),
                    "four_pairs_estimate_mib": 4 * sum(checkpoint_sizes.values()),
                    "notes": "Four-pair estimate excludes data, logs, extra runs and temporary checkpoint files."},
        "measured_setup": {"dataset_load_seconds": sum(r["seconds"] for r in loads.values()),
                           "three_epoch_train_call_seconds_including_setup": call_seconds},
        "baseline_history_observation": observed_baseline,
        "planning_estimates_not_measurements": {
            "same_81_class_40_epochs_minutes_including_saves": wall * 40 / 60,
            "one_learning_curve_arm_10_blocks_minutes": wall * 10 / 60,
            "four_learning_curve_arms_40_blocks_minutes": wall * 40 / 60,
            "learning_curve_note": "Same 15552 train + 7776 validation draws per block; extra curve metrics/setup overhead unmeasured.",
            "hypothetical_1140_classes": {
                "train_images": 1140 * 192, "validation_images": 1140 * 96,
                "sample_count_ratio": ratio,
                "train_and_validation_seconds_per_epoch": measured_compute * ratio,
                "train_and_validation_hours_for_40_epochs": measured_compute * ratio * 40 / 3600,
                "assumptions": "Keep 192 train/96 validation per class, 64x64 input, batch64 and T4; assume time proportional to sample count. Larger classifier, data pipeline, saves, setup and convergence are unmeasured. This does not authorize expansion or set its epoch budget."}},
        "conclusions": [
            f"当前测量下训练占整轮 {phase_percent['train']:.2f}%，验证占 {phase_percent['validation']:.2f}%，检查点保存占 {phase_percent['checkpoint_save']:.2f}%。",
            ("保存占比较低，现有数据不支持为省时降低断点保存频率。" if phase_percent['checkpoint_save'] < 5
             else "保存开销值得单独排查；调整保存频率前仍需考虑中断损失。"),
            "显存容量占比不等于 GPU 计算利用率，也不直接证明增大 batch 会提速。",
            "保持当前实验 batch=64；后续吞吐优化若改变 batch、精度或数据流程，应另做独立基准。",
            "81 字开发准确率的收敛观察不能直接用来确定 1140 字的训练轮数。"],
        "not_measured": {"compute_units_consumed": None, "colab_cost": None, "gpu_utilization": None,
                         "cpu_utilization": None, "system_ram_peak": None,
                         "original_40_epoch_wall_time_including_saves": None},
        "limitations": [
            "独立临时模型的三轮计时；稳态仅用第 2、3 轮，不给统计置信区间。",
            "显存是计时子进程的 PyTorch allocated/reserved，reserved 已含 allocated，不可相加。",
            "阶段占比使用同一整轮 wall time 分母；余项包含其他开销和整轮 0.1 秒舍入。",
            "本地数据载入时间受缓存影响；不是 Drive 下载速度或完整初始化耗时。",
            "Drive 保存计时到挂载文件系统调用返回，不证明远端上传已完成。",
            "样本量外推只是排期估算；不能替代扩类后实测或推断泛化表现。"]}


def print_efficiency(analysis):
    steady, mem = analysis["steady"], analysis["memory"]
    print("\n第 6 项：训练效率与资源分析（复用已有测量，无新增 GPU 训练）")
    print(f"整轮含保存：{steady['epoch_wall_seconds_including_saves']:.3f} s")
    print(f"训练吞吐：{steady['train_images_per_second']:.2f} 图/s；验证吞吐：{steady['validation_images_per_second']:.2f} 图/s")
    names = {"train": "训练", "validation": "验证", "checkpoint_save": "检查点保存", "other_and_rounding": "其他及舍入"}
    for key, seconds in analysis["phase_seconds"].items():
        print(f"  {names[key]}：{seconds:.3f} s / {analysis['phase_percent_of_steady_wall'][key]:.2f}%")
    print(f"峰值张量占用：{mem['peak_allocated_gib']:.3f} GiB ({mem['allocated_percent_of_capacity']:.2f}% 容量)")
    print(f"峰值预留：{mem['peak_reserved_gib']:.3f} GiB ({mem['reserved_percent_of_capacity']:.2f}% 容量，已含张量占用)")
    print(f"best + last：{analysis['storage']['best_plus_last_mib']:.2f} MiB；不含临时文件等空间。")
    estimate = analysis["planning_estimates_not_measurements"]
    print(f"81 字 40 轮含保存排期估算：{estimate['same_81_class_40_epochs_minutes_including_saves']:.2f} 分钟（非原训练实测 wall time）。")
    full = estimate["hypothetical_1140_classes"]
    print(f"假设 1140 字保持每类 192/96 图，40 轮训练+验证线性估算：{full['train_and_validation_hours_for_40_epochs']:.2f} 小时；不含保存等开销，扩类后需实测。")
    print("实际 compute units、费用、GPU 利用率、CPU/RAM 峰值：未测量。")
    for text in analysis["conclusions"]:
        print("-", text)


def plot_efficiency(analysis, out=None, show=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    phases = analysis["phase_seconds"]
    axes[0].barh(["Train", "Validation", "Checkpoint", "Other + rounding"], list(phases.values()),
                 color=["#3676b6", "#e38d36", "#4d9971", "#87929c"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Seconds per epoch (median of epochs 2 and 3)")
    mem = analysis["memory"]
    axes[1].barh(["Peak allocated", "Peak reserved (includes allocated)", "Device capacity"],
                 [mem["peak_allocated_gib"], mem["peak_reserved_gib"], mem["device_total_gib"]],
                 color=["#3676b6", "#7398bd", "#c4cbd1"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("GiB (memory capacity, not GPU utilization)")
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    if out:
        fig.savefig(out / "efficiency_resources.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
