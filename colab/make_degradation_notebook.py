"""Build a self-contained two-cell Colab entry point for the phase-1 sweep."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/data/degradation81-calibration-v1"
DESTINATION = ROOT / "colab/hanzi81_degradation_phase1.ipynb"

INTRO = """# 81 字 · 第一阶段退化标定

本阶段：使用**原第 38 轮 best.pt** 对 25 档条件进行推理。共 **32,400 张**，来自原
`val_graph` 的 81 字 × 16 个共享清晰位移。σ=[0,1,1.5,2,2.5]，R=[64,48,32,24,16]，
始终输出 64×64。此阶段不更新模型、不读取原 test 图像，不决定正式训练难度。

**只需上传这个 Notebook 到 Colab，按顺序运行下面两格。** 源代码已嵌入，沿用原 Drive
中的数据 ZIP 和 `runs/baseline81_s42_v1/best.pt`。建议选择 GPU；CPU 也能推理，耗时更长。
不需要重新上传旧数据、模型或本地生成的标定集，也不需要运行旧 Notebook。

第①格固定 Pillow=12.3.0，生成与本地像素指纹一致的标定集。保留 Colab 已有 torch/torchvision；
推理环境会记录在新报告中，不假装与原训练环境完全相同。新代码和数据使用独立 `/content` 目录。
第②格先在本地完成推理和报告，再保存到 Drive 的新时间戳目录。原基线及其记录不被覆盖。

运行完成后，将第②格下载的 `hanzi81_phase1_results.zip` 回传，用于确定下一步人工盲认的档位。
旧模型掉分只说明对新退化的适应情况；仍需人工可读性标定与同分布基线重训。
"""

PREPARE = '''#@title ① 准备固定标定集（不会训练）
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from google.colab import drive

DRIVE_ROOT = Path("/content/drive/MyDrive/hanzi81")
WORK = Path("/content/hanzi81_degradation_phase1_v1")
EXPECTED_PIXELS = __PIXELS__
SOURCES = __SOURCES__

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")
checkpoint = DRIVE_ROOT / "runs/baseline81_s42_v1/best.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(f"缺少原模型：{checkpoint}")
if importlib.metadata.version("Pillow") != "12.3.0":
    subprocess.run([sys.executable, "-m", "pip", "install", "Pillow==12.3.0"], check=True)

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if file_digest(checkpoint) != "8d65412bf6065253ebce3ca5b0c542788b71e0da391cc335a815f980d828a106":
    raise ValueError("best.pt 与归档的第 38 轮模型不一致")
WORK.mkdir(parents=True, exist_ok=True)
for name, source in SOURCES.items():
    target = WORK / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")

bundle_hash = "e477bdd4894e6b585a2149d6f22db3948e5d6e20075d532fded5171d1b57dd35"
candidates = [DRIVE_ROOT / "hanzi81_colab_bundle.zip", DRIVE_ROOT / "packages" / (bundle_hash + ".zip")]
archive_path = next((p for p in candidates if p.is_file()), None)
if archive_path is None:
    raise FileNotFoundError(f"找不到原始数据包，请检查：{candidates}")
source = WORK / "training/data/baseline81-v2"
source.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive_path) as archive:
    # Only extract development data and metadata. train/test arrays stay in the ZIP.
    for name in ("manifest.json", "labels.json", "conditions.json", "val_graph.npz"):
        (source / name).write_bytes(archive.read("training/data/baseline81-v2/" + name))
calibration = WORK / "training/data/degradation81-calibration-v1"
if not calibration.exists():
    subprocess.run([sys.executable, "-u", "-m", "training.degradation", "generate",
                    "--data", str(source), "--output", str(calibration)], cwd=WORK, check=True)
manifest = json.loads((calibration / "manifest.json").read_text(encoding="utf-8"))
if manifest["pixel_sha256"] != EXPECTED_PIXELS:
    raise ValueError("标定像素与本地已检查版本不同，停止推理并检查 Pillow / 输入文件")
for name, expected in manifest["files"].items():
    if file_digest(calibration / name) != expected:
        raise ValueError(f"标定数据文件损坏：{name}")
print("准备完成：", manifest["samples"], "张 /", len(manifest["grids"]), "档；原模型未更新。")
'''

EVALUATE = '''#@title ② 执行 25 档推理并保存新报告
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import shutil
import subprocess
import sys
from IPython.display import Image, display
from google.colab import files

DEVICE = "auto" #@param ["auto", "cuda", "cpu"]
BATCH_SIZE = 128 #@param {type:"integer"}
DRIVE_ROOT = Path("/content/drive/MyDrive/hanzi81")
WORK = Path("/content/hanzi81_degradation_phase1_v1")
if not (WORK / "training/data/degradation81-calibration-v1/manifest.json").is_file():
    raise FileNotFoundError("请先运行第①格准备标定集")
stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S_%f")
local_report = WORK / "reports" / stamp
subprocess.run([sys.executable, "-u", "-m", "training.degradation", "evaluate",
    "--data", str(WORK / "training/data/degradation81-calibration-v1"),
    "--checkpoint", str(DRIVE_ROOT / "runs/baseline81_s42_v1/best.pt"),
    "--output", str(local_report), "--batch-size", str(BATCH_SIZE), "--device", DEVICE], cwd=WORK, check=True)
summary = json.loads((local_report / "summary.json").read_text(encoding="utf-8"))
if summary["status"] != "completed_frozen_model_sweep" or summary["samples"] != 32400 or summary["grids"] != 25:
    raise ValueError("评估未完成全部 25 档，未标记阶段完成")
shutil.copy2(WORK / "training/data/degradation81-calibration-v1/manifest.json", local_report / "calibration_manifest.json")
shutil.copy2(WORK / "training/data/degradation81-calibration-v1/audit.json", local_report / "calibration_audit.json")
shutil.copy2(WORK / "training/degradation.py", local_report / "degradation.py")
drive_report = DRIVE_ROOT / "analyses/baseline81_s42_v1/07_degradation_phase1" / stamp
shutil.copytree(local_report, drive_report)
archive = shutil.make_archive(str(WORK / "hanzi81_phase1_results"), "zip", local_report)
shutil.copy2(archive, drive_report / "hanzi81_phase1_results.zip")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("完整报告已保存：", drive_report)
display(Image(filename=str(local_report / "accuracy_curves.png")))
files.download(archive)
'''


def build():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    names = ["training/__init__.py", "training/io_utils.py", "training/train.py", "training/degradation.py"]
    # Preserve exact archived source bytes: read_text normalizes CRLF. Model identity uses file bytes.
    sources = {name: (ROOT / name).read_bytes().decode("utf-8") for name in names}
    archived = json.loads((ROOT / "colab/experiments/baseline81_s42_v1/provenance/bundle_manifest.json").read_text(encoding="utf-8"))
    for name in ("training/train.py", "training/io_utils.py"):
        if hashlib.sha256(sources[name].encode("utf-8")).hexdigest() != archived["files"][name]:
            raise ValueError(f"Restore the archived baseline source before packaging: {name}")
    prepare = PREPARE.replace("__PIXELS__", repr(manifest["pixel_sha256"])).replace("__SOURCES__", repr(sources))
    cells = [{"cell_type": "markdown", "metadata": {}, "id": "phase1-intro", "source": INTRO.splitlines(keepends=True)}]
    for name, code in (("phase1-prepare", prepare), ("phase1-evaluate", EVALUATE)):
        compile(code, name, "exec")
        cells.append({"cell_type": "code", "metadata": {"cellView": "form"}, "id": name,
                      "source": code.splitlines(keepends=True), "execution_count": None, "outputs": []})
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"name": DESTINATION.name},
        "phase1": {"pixel_sha256": manifest["pixel_sha256"], "model_evaluation_status": "pending_colab_execution"}}, "cells": cells}
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return notebook


if __name__ == "__main__":
    build()
    print(DESTINATION)
