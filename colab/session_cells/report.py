#@title ③ 结果检查与汇总（可单独运行，无需 GPU）
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics
import uuid
from colab.session_cells.efficiency import analyze_efficiency, print_efficiency, plot_efficiency

SAVE_REPORT = True #@param {type:"boolean"}
DRIVE_ROOT = Path("/content/drive/MyDrive/hanzi81")


def report_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def report_read(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"报告无法读取；未自动覆盖或重算：{path}") from exc


def report_files(folder, names):
    return {name: {"exists": (folder / name).is_file(),
                   "bytes": (folder / name).stat().st_size if (folder / name).is_file() else None}
            for name in names}


def report_efficiency(profile_path, baseline, run):
    if profile_path is None:
        return {"status": "incomplete_missing_measurement", "message": "未找到三轮计时报告；未自动重跑。"}
    try:
        profile = report_read(profile_path)
        reference = report_read(run / "config.json")
        reference_environment = reference["environment"] if reference is not None else None
        analysis = analyze_efficiency(profile, baseline, reference_environment)
        protocol = report_read(profile_path.parent / "protocol.json")
        frozen_script = profile_path.parent / "profile_runtime.py"
        expected_script = "131d07fadea99eece5d0c1c4179e454b8149f8ec960a3c6ba2229f40886052bc"
        if frozen_script.is_file() and report_digest(frozen_script) != expected_script:
            raise ValueError("计时脚本哈希与原版本不同")
        if protocol is not None:
            if protocol["profiler_sha256"] != expected_script or protocol["measured_epochs"] != 3:
                raise ValueError("计时协议与原版本不同")
            if reference is not None:
                for key in ("data_sha256", "trainer_sha256", "environment"):
                    if protocol[key] != reference[key]:
                        raise ValueError(f"计时协议与原基线不匹配：{key}")
        analysis["source"] = {"summary_path": str(profile_path), "summary_sha256": report_digest(profile_path),
                              "baseline_config_checked": reference is not None,
                              "protocol_checked": protocol is not None,
                              "frozen_profiler_checked": frozen_script.is_file(),
                              "new_gpu_measurement": False}
        return analysis
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return {"status": "incomplete_invalid_measurement", "message": str(exc),
                "source_path": str(profile_path), "new_gpu_measurement": False}


def report_learning_progress(suite):
    plan = report_read(suite / "plan.json")
    if plan is not None and (plan["blocks"], plan["block_draws"], plan["batch_size"], plan["sizes"]) != (
            10, 15552, 64, [24, 48, 96, 192]):
        raise ValueError("学习曲线协议与当前 Notebook 不同；不混合比较。")
    rows, histories = [], {}
    for size in (24, 48, 96, 192):
        folder = suite / f"n{size:03d}_s42"
        history = report_read(folder / "history.json")
        files = report_files(folder, ("config.json", "history.json", "best.pt", "last.pt"))
        artifacts_exist = all(value["exists"] and value["bytes"] > 0 for value in files.values())
        row = {"per_class": size, "unique_training_images": size * 81,
               "logged_blocks": None, "planned_blocks": 10,
               "complete_by_history_and_file_presence": False, "files": files,
               "status": "未找到训练日志；不据此断言从未训练"}
        if history is not None:
            if [r["block"] for r in history] != list(range(1, len(history) + 1)) or len(history) > 10:
                raise ValueError(f"{folder.name} 日志块编号异常")
            for record in history:
                if record["batch_attempts"] != record["block"] * 243 or record["validation_split"] != "val_select":
                    raise ValueError(f"{folder.name} 日志与固定批次预算不同")
            row.update(logged_blocks=len(history),
                       complete_by_history_and_file_presence=len(history) == 10 and artifacts_exist,
                       history_sha256=report_digest(folder / "history.json"))
            if history:
                cfg = report_read(folder / "config.json")
                if plan is None or cfg is None or cfg["suite_plan_sha256"] != report_digest(suite / "plan.json"):
                    raise ValueError(f"{folder.name} 缺少匹配的冻结协议或配置")
                if cfg["unique_per_class"] != size or cfg["blocks"] != 10 or cfg["seed"] != 42:
                    raise ValueError(f"{folder.name} 配置不匹配")
                best = max(history, key=lambda r: (r["validation"]["macro_accuracy"],
                                                    -r["validation"]["loss"], -r["block"]))
                row.update(last_completed_validation=history[-1]["validation"],
                           best_validation=best["validation"], best_block=best["block"],
                           optimizer_updates=history[-1]["optimizer_updates"],
                           amp_skipped_updates=sum(r["amp_skipped_updates_this_block"] for r in history))
                row["status"] = ("已记录完整预算" if len(history) == 10 else "中间进度，待续训") if artifacts_exist else "文件不完整，需检查"
            else:
                row["status"] = "空日志，尚无已记录完整块"
            histories[size] = history
        rows.append(row)
    return {"plan_found": plan is not None, "results": rows,
            "all_complete_by_history_and_file_presence": all(r["complete_by_history_and_file_presence"] for r in rows),
            "checkpoint_contents_loaded": False,
            "note": "这里只检查日志与模型文件存在性；未反序列化权重。实际续训由原脚本验证检查点。"}, histories


def report_baseline_history(run):
    history = report_read(run / "history.json")
    if history is None:
        return {"status": "未找到基线 history.json"}, []
    if not history or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("基线日志轮次不连续或为空")
    if len(history) > 40 or any(r["validation_split"] != "val_select" for r in history):
        raise ValueError("基线日志协议不匹配")
    selected = max(history, key=lambda r: (r["validation"]["macro_accuracy"],
                                           -r["validation"]["loss"], -r["epoch"]))
    thresholds = {}
    for name, threshold in [("99%", .99), ("99.9%", .999), ("100%", 1.0)]:
        passed = [r["validation"]["macro_accuracy"] >= threshold for r in history]
        thresholds[name] = {
            "first_epoch": next((history[i]["epoch"] for i in range(len(history)) if passed[i]), None),
            "sustained_through_last_logged_epoch_from": next(
                (history[i]["epoch"] for i in range(len(history)) if all(passed[i:])), None)}
    times = [r["seconds_compute"] for r in history]
    return {"completed_epochs": len(history), "planned_epochs": 40,
            "history_sha256": report_digest(run / "history.json"),
            "selected_epoch_by_original_rule": selected["epoch"],
            "selected_validation": selected["validation"], "last_validation": history[-1]["validation"],
            "thresholds": thresholds, "compute_minutes_excluding_saves": sum(times) / 60,
            "steady_median_seconds": statistics.median(times[1:]) if len(times) > 1 else None,
            "files": report_files(run, ("config.json", "history.json", "best.pt", "last.pt"))}, history


def report_plots(baseline, progress, histories, out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("当前环境没有 matplotlib；JSON 汇总仍可使用，不自动安装依赖。")
        return
    if baseline:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        epochs = [r["epoch"] for r in baseline]
        for key, label in [("train", "train"), ("validation", "val_select")]:
            axes[0].plot(epochs, [max(r[key]["loss"], 1e-12) for r in baseline], label=label)
            axes[1].plot(epochs, [100 * r[key]["accuracy"] for r in baseline], label=label)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Cross entropy")
        axes[1].set_ylabel("Accuracy (%)")
        axes[2].plot(epochs, [r["seconds_compute"] for r in baseline])
        axes[2].set_ylabel("Train + validation (s)")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.3)
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        if out:
            fig.savefig(out / "baseline_history.png", dpi=160)
        plt.show()
        plt.close(fig)
    if any(histories.values()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for size, history in histories.items():
            if history:
                batches = [r["batch_attempts"] for r in history]
                axes[0].plot(batches, [100 * r["validation"]["macro_accuracy"] for r in history], label=f"n={size}")
                axes[1].plot(batches, [max(r["validation"]["loss"], 1e-12) for r in history], label=f"n={size}")
        axes[0].set_ylabel("val_select macro accuracy (%)")
        axes[1].set_ylabel("val_select cross entropy")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("Completed batch attempts")
            ax.grid(alpha=.3)
            ax.legend()
        fig.suptitle("Learning-curve progress (partial budgets may differ)")
        fig.tight_layout()
        if out:
            fig.savefig(out / "learning_curve_progress.png", dpi=160)
        plt.show()
        plt.close(fig)
    if progress["all_complete_by_history_and_file_presence"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        rows = progress["results"]
        for key, label in [("last_completed_validation", "Final: fixed budget"),
                           ("best_validation", "Selected best: secondary")]:
            x = [r["per_class"] for r in rows]
            axes[0].plot(x, [100 * r[key]["macro_accuracy"] for r in rows], "o-", label=label)
            axes[1].plot(x, [max(r[key]["loss"], 1e-12) for r in rows], "o-", label=label)
        axes[0].set_ylabel("val_select macro accuracy (%)")
        axes[1].set_ylabel("val_select cross entropy")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("Unique training images per class")
            ax.set_xticks([24, 48, 96, 192])
            ax.grid(alpha=.3)
            ax.legend()
        fig.tight_layout()
        if out:
            fig.savefig(out / "learning_curve_fixed_budget.png", dpi=160)
        plt.show()
        plt.close(fig)
    else:
        print("四档尚无完整预算结果：仅展示进度，不生成最终固定预算比较图。")


def inspect_session(root, save_report):
    # 本格可以在新 CPU 会话直接执行，不依赖第①/②格、torch 或 /content 中的数据。
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    if not root.is_dir():
        raise FileNotFoundError(f"请检查 Drive 项目路径：{root}")
    run = root / "runs/baseline81_s42_v1"
    analyses = root / "analyses/baseline81_s42_v1"
    baseline, history = report_baseline_history(run)
    progress, histories = report_learning_progress(analyses / "03_learning_curve/steps2430_s42_v1")
    existing = {}
    paths = {"01_history": analyses / "01_history/summary.json",
             "02_margins": analyses / "02_margins/summary.json",
             "val_select": run / "evaluation_val_select/metrics.json",
             "val_graph": run / "evaluation_val_graph/metrics.json"}
    profiles = sorted((analyses / "06_profile").glob("*/summary.json"))
    if profiles:
        paths["06_profile_saved_measurement"] = profiles[-1]
    for key, path in paths.items():
        value = report_read(path)
        existing[key] = {"path": str(path), "found": value is not None,
                         "sha256": report_digest(path) if value is not None else None, "report": value}
    efficiency = report_efficiency(profiles[-1] if profiles else None, baseline, run)
    snapshot = {
        "purpose": "read_saved_files_only_no_model_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline, "learning_curve": progress, "existing_reports": existing,
        "task_06_status": efficiency["status"], "efficiency_resources": efficiency,
        "notes": ["第 6 项根据已保存测量的实际校验与分析结果更新；不新增 GPU 训练。",
                  "本单元未启动训练、模型推理或 test 评估，也未反序列化检查点。",
                  "Drive 挂载读取成功不等于已独立验证远端上传持久性。"]}
    print("基线日志：", json.dumps(baseline, ensure_ascii=False, indent=2))
    print("\n学习曲线：每字图数 | 日志完整块/10 | 状态")
    for row in progress["results"]:
        count = row["logged_blocks"] if row["logged_blocks"] is not None else "未知"
        print(f"{row['per_class']:>3} | {count}/10 | {row['status']}")
        if "last_completed_validation" in row:
            print("  最近验证：", row["last_completed_validation"])
    print("\n已有报告：")
    for key, value in existing.items():
        print(key, "已找到" if value["found"] else "未找到", value["path"])
    if efficiency["status"] == "complete_saved_measurement_analysis":
        print_efficiency(efficiency)
    else:
        print("第 6 项仍未完成：", efficiency["message"])
    out = None
    if save_report:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        out = analyses / "session_reports" / stamp
        out.mkdir(parents=True, exist_ok=False)
        (out / "session_summary.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("本次检查报告：", out / "session_summary.json")
        (out / "efficiency_resources.json").write_text(
            json.dumps(efficiency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("第 6 项分析文件：", out / "efficiency_resources.json")
    report_plots(history, progress, histories, out)
    if efficiency["status"] == "complete_saved_measurement_analysis":
        plot_efficiency(efficiency, out)
    print("模型权重未在本格加载；续训时仍需原脚本验证。确认文件保存后，可释放 GPU 运行时。")
    return snapshot


if __name__ == "__main__":
    inspect_session(DRIVE_ROOT, SAVE_REPORT)
