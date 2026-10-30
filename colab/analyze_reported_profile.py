"""Archive analysis of the user's saved notebook output, without accessing Colab."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from colab.session_cells.efficiency import analyze_efficiency, print_efficiency


def load_reported_profile():
    original = HERE / "hanzi81_baseline_new.before_session_refactor.ipynb"
    content = original.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != "6fca42f0524bd8d3568f68d759750a6e910229d0e43ecf1baeb62977b2cdf190":
        raise ValueError("用户原 Notebook 导出副本哈希不一致")
    notebook = json.loads(content)
    outputs = notebook["cells"][25]["outputs"]
    text = "".join("".join(item.get("text", [])) for item in outputs if item["output_type"] == "stream")
    profile = json.loads(text[text.index("{"):])
    return profile, {"type": "user_exported_notebook_saved_stdout", "notebook": original.name,
                     "notebook_sha256": digest, "zero_based_cell_index": 25,
                     "live_drive_read": False, "new_gpu_measurement": False,
                     "colab_cell_3_execution_verified": False}


def main():
    profile, source = load_reported_profile()
    folder = HERE / "experiments/baseline81_s42_v1"
    previous = json.loads((folder / "development_progress_2026-10-13.json").read_text(encoding="utf-8"))
    report = previous["history_analysis_reported"]
    timing = report["计算耗时_含训练验证_不含检查点保存"]
    baseline = {"completed_epochs": report["完成轮数"], "planned_epochs": 40,
                "compute_minutes_excluding_saves": timing["40轮合计分钟"],
                "steady_median_seconds": timing["第2至40轮中位数秒"],
                "thresholds": report["阈值统计_val_select_macro_accuracy"]}
    analysis = analyze_efficiency(profile, baseline)
    analysis["source"] = source
    analysis["source"]["original_profile_path_reported"] = profile["output_directory"] + "/summary.json"
    analysis["task_status_note"] = "首轮效率分析已基于用户导出中的真实测量完成；新版第③格接入同一算法，尚未确认在用户 Colab 执行。未重跑 GPU 计时。"
    measurement_path = folder / "provenance/reported_profile_summary.json"
    content = (json.dumps(profile, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if measurement_path.exists() and measurement_path.read_bytes() != content:
        raise ValueError("既有回传计时摘要不同，不能覆盖")
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.write_bytes(content)
    analysis["source"]["local_reported_summary_sha256"] = hashlib.sha256(content).hexdigest()
    output = folder / "efficiency_analysis_2026-10-13.json"
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_efficiency(analysis)
    print("Local analysis:", output)
    if "--plot" in sys.argv:
        try:
            import matplotlib
        except ImportError:
            print("当前环境没有 matplotlib；JSON 已保存。Colab 第③格可使用原有 matplotlib 绘图。")
            return
        matplotlib.use("Agg")
        from colab.session_cells.efficiency import plot_efficiency
        plot_efficiency(analysis, folder, show=False)
        print("Local figure:", folder / "efficiency_resources.png")


if __name__ == "__main__":
    main()
