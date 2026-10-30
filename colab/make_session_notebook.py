"""Build the three-cell continuation notebook; preserve the original export once.

This only writes local notebook files. It never mounts Drive or runs training.
The source files under session_cells are embedded, so Colab needs only the ipynb.
"""
from pathlib import Path
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "hanzi81_baseline_new.ipynb"
BACKUP = HERE / "hanzi81_baseline_new.before_session_refactor.ipynb"
ORIGINAL_SHA256 = "6fca42f0524bd8d3568f68d759750a6e910229d0e43ecf1baeb62977b2cdf190"

INTRO = """# 81 字实验 · 会话恢复与分段续训

本文件根据你从 Colab 导出的当前状态整理，更新于 **2026-10-13**。
日常只需下方 **3 个代码单元**；完整代码已嵌入，无需上传 `session_cells` 目录。

| 已知实验状态 | 当前处理 |
|---|---|
| smoke、小集拟合、40 轮基线 | 已完成，不在初始化时重跑 |
| 第 1 项：学习过程复盘 | 已有结果；第③格读取并复核日志 |
| 第 2 项：分类间隔 | 已有结果；第③格读取保存的报告 |
| 第 3 项：学习曲线 | 已确认四档准备成功；训练进度以 Drive 实际文件为准 |
| 第 6 项：训练效率与资源分析 | 第③格复用已有三轮计时完成分析；只有实际校验成功才标记完成 |

**这三个单元的编号与上述分析项目编号是两套编号。**

原来的 29 个单元、图片、输出及历史修复代码，逐字节保存在同目录的
`hanzi81_baseline_new.before_session_refactor.ipynb`。该副本供追溯，不应“全部运行”。
当前文件不携带旧输出，以免把历史结果误认成本次执行结果。
原基线 `train.py`、学习曲线脚本、数据和 ZIP 均保持原版本。

**仅查看结果：** 使用 CPU 运行时，直接运行第③格；它会自行挂载 Drive。

**完成第 6 项：** 同样直接运行第③格。它会校验已有计时报告，计算训练/验证吞吐、
阶段耗时占比、显存容量占比、检查点大小和训练时长估算，并保存 `efficiency_resources.json`
与图表。无需重新进行三轮 GPU 计时；缺失或损坏的报告会保留未完成状态。

**继续学习曲线：** 连接与原实验匹配的 GPU → 运行第①格 → 在第②格显式开启训练，
选择一个档位和本次块数 → 运行第③格检查保存。
默认训练关闭；默认每次只给一个档位追加 3 块。分段不改变每档 10 块 / 2430 批次的总预算。

Drive 项目仍为 `/content/drive/MyDrive/hanzi81`。如确需换目录，要同步修改相关单元顶部的
`DRIVE_ROOT`。本文件面向已建立的实验；不会自动重建缺失的实验协议或放宽环境检查。
"""

SETUP = """## ① 初始化与恢复

合并 Drive 挂载、路径、固定 ZIP 恢复、文件校验、学习曲线脚本恢复和 GPU 环境检查。
重复运行时复用已校验的本地文件；不用重新上传辅助脚本，也不升级 torch/torchvision。
代码与数据位于 `/content`，检查点继续保存在 Drive。

原环境：Python 3.13.15、torch 2.11.0+cu130、torchvision 0.26.0+cu130、NumPy 2.1.3、Tesla T4。
版本或 GPU 改变时严格续训会停止并显示差异；不要删除检查来强行续训。
`PREPARE_TRAINING=False` 时只挂载 Drive；看报告也可以直接跳到第③格。
"""

TRAIN = """## ② 开始或继续一个学习曲线档位

只有 `RUN_TRAINING=True` 才会启动训练。每字图数 `PER_CLASS` 可选 192、96、48、24；
建议按这个顺序完成，但各档互相独立。`BLOCKS_THIS_SESSION` 是**本次追加块数**，不是总预算。
例如已保存 3 块、再运行 3 块，就续至第 6 块；已完成 10 块的档位由原脚本校验后跳过。

每块 243 批次，保存模型、优化器、学习率计划、AMP scaler 和随机状态。
中断后恢复最近完整块；未保存的半块需要重跑。一次只运行一个写入该档位的会话。
输出实时显示，并额外保存在实验目录的 `session_logs` 中。

本格拥有自己的导入和路径，不再依赖 `run_learning_curve` 等上一格定义的函数。
缺少本地数据时会明确提示回第①格恢复。四档尚未全部完成时，正常分段结束不会报错。
基线旧的 40 轮结果不充当学习曲线 192 图档位；原实验参数和总训练量均保持不变。
"""

REPORT = """## ③ 检查保存结果与已有报告

可在全新 CPU 会话中单独运行：只挂载 Drive、读 JSON、检查文件存在性并绘图；
不导入 torch、不读取训练图像、不执行模型推理。没有第①/②格的变量也可以运行。

第 1/2 项和开发集指标从已保存文件读取。第 3 项逐档显示实际日志进度，
尚未完成时只画进度图；四档都有完整预算记录及检查点文件时才画最终比较图。
第 6 项读取已有三轮计时并检验内部一致性；成功时完成首轮效率分析。
测量值与估算分别列示：实际 compute units、费用、GPU 利用率、CPU/RAM 峰值没有测量，
保持为空；显存 allocated/reserved 不相加，也不当作 GPU 计算利用率。
1140 字时长只是保持每类样本配比的线性排期估算，不代表已扩类训练或确定训练轮数。

默认在 `analyses/baseline81_s42_v1/session_reports/<时间戳>/` 新建本次报告，
不覆盖原有分析。这里不加载模型权重，检查点内部状态仍由原续训脚本验证；
挂载读取成功不等于独立验证云端上传已完成。

确认日志和 Drive 文件保存后，选择 **断开连接并删除运行时**。
下次仅读报告不需要 GPU；额度耗尽也不必重跑已经完成的实验。
"""


def md(text, cell_id):
    return {"cell_type": "markdown", "id": cell_id,
            "metadata": {"id": cell_id}, "source": text.splitlines(keepends=True)}


def code(filename, cell_id):
    source = (HERE / "session_cells" / filename).read_text(encoding="utf-8")
    if filename == "report.py":
        imported = "from colab.session_cells.efficiency import analyze_efficiency, print_efficiency, plot_efficiency"
        if source.count(imported) != 1:
            raise ValueError("第③格的分析嵌入入口已改变")
        source = source.replace(imported, (HERE / "session_cells/efficiency.py").read_text(encoding="utf-8"))
    compile(source, filename, "exec")
    return {"cell_type": "code", "id": cell_id,
            "metadata": {"id": cell_id, "cellView": "form"},
            "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)}


def build(original):
    metadata = copy.deepcopy(original["metadata"])
    # Default to CPU when a fresh runtime is selected; training explicitly needs the original GPU.
    metadata.pop("accelerator", None)
    metadata.get("colab", {}).pop("gpuType", None)
    metadata["hanzi81_session_refactor"] = {
        "version": 2, "updated": "2026-10-13", "original_filename": BACKUP.name,
        "original_sha256": ORIGINAL_SHA256, "original_cells": len(original["cells"]),
        "task_06_status": "determined_from_saved_measurement_when_cell_3_runs"}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": metadata,
            "cells": [md(INTRO, "session-intro"), md(SETUP, "setup-notes"), code("initialize.py", "setup-code"),
                      md(TRAIN, "train-notes"), code("train.py", "train-code"),
                      md(REPORT, "report-notes"), code("report.py", "report-code")]}


def main():
    if not BACKUP.exists():
        content = NOTEBOOK.read_bytes()
        if hashlib.sha256(content).hexdigest() != ORIGINAL_SHA256:
            raise ValueError("当前 Notebook 已改变，先核对最新导出，不能覆盖未知改动。")
        with BACKUP.open("xb") as stream:
            stream.write(content)
    content = BACKUP.read_bytes()
    if hashlib.sha256(content).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("原始副本哈希不匹配，停止生成。")
    current = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if "hanzi81_session_refactor" in current.get("metadata", {}) and any(
            c.get("outputs") or c.get("execution_count") is not None for c in current["cells"] if c["cell_type"] == "code"):
        raise ValueError("会话版已含运行输出；请另存此导出再生成，避免丢失新结果。")
    if hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() != ORIGINAL_SHA256 and \
            "hanzi81_session_refactor" not in current.get("metadata", {}):
        raise ValueError("目标文件不是已识别的原文件或会话版，停止覆盖。")
    notebook = build(json.loads(content))
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    temporary.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(NOTEBOOK)
    print(json.dumps({"notebook": str(NOTEBOOK), "backup": str(BACKUP),
                      "original_sha256": ORIGINAL_SHA256, "code_cells": 3}, ensure_ascii=False))


if __name__ == "__main__":
    main()
