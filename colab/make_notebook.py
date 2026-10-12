"""Generate the portable Colab notebook using only the standard library."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n",
                  "execution_count": None, "outputs": []})


md("""
# 81 字形近汉字基线 · Colab

先完成 GPU 检查、数据检查和短跑，再启动 40 epoch 基线。所有训练输出写入 Google Drive；
四份数据复制到 `/content` 后使用。当前研究边界是固定思源黑体的合成单字识别。

**首次使用**
1. 在 Google Drive 的“我的云端硬盘”下建立 `hanzi81` 文件夹，把 `hanzi81_colab_bundle.zip` 上传进去。
2. 在 Colab 上传此 Notebook，并“在云端硬盘中保存副本”。
3. 菜单“运行时／代码执行程序 → 更改运行时类型”选择 Python 3、GPU。先选可用的 T4；无需为本轮优先选最昂贵的 GPU。
4. 从上到下逐格执行。遇到红色错误就停在该格。完整训练默认关闭，短跑正常后按第 7 步开启。

已有 Pro／算力额度不代表固定 GPU 或无限运行时。断线后重连并重跑前面的准备格，再运行相同的训练格，会从最近完成的 epoch 恢复。
更换 GPU 型号、软件版本、数据或超参数时不能声称逐步一致续训，脚本会要求新建 run。
参考：[Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html)。
""")
md("""
## 1. 确认 GPU 与预装 PyTorch

使用 Colab 预装的 torch/torchvision 配套组合，不运行 `pip install -U torch`。
上传包已包含生成好的数据，不需要在 GPU 会话中重新渲染。版本和设备信息会写入每次实验配置。
""")
code("""
import os, sys, json, subprocess, hashlib, shutil, zipfile
from pathlib import Path
import torch, torchvision, numpy as np
print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'Torchvision:', torchvision.__version__)
assert torch.cuda.is_available(), '未分配到 GPU：更改运行时类型为 GPU，再重新连接。'
print('GPU:', torch.cuda.get_device_name(0))
print('显存 GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
subprocess.run(['nvidia-smi'], check=True)
with torch.no_grad():
    probe = torch.nn.Conv2d(1, 8, 3).cuda()
    print('GPU 前向检查:', tuple(probe(torch.zeros(2, 1, 64, 64, device='cuda')).shape))
del probe
torch.cuda.empty_cache()
""")
md("""
## 2. 连接 Drive

在弹出的 Google 授权窗口完成登录与 Drive 挂载。项目文件夹默认是 `MyDrive/hanzi81`；
若你使用其他名字，只改 `DRIVE_ROOT`。不要将项目放在“与我共享”的快捷方式里。
""")
code("""
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/hanzi81')
BUNDLE = DRIVE_ROOT / 'hanzi81_colab_bundle.zip'
assert BUNDLE.is_file(), f'请先将上传包放到 {BUNDLE}'
RUNS = DRIVE_ROOT / 'runs'
RUNS.mkdir(parents=True, exist_ok=True)
print('上传包 MiB:', round(BUNDLE.stat().st_size / 2**20, 1))
print('结果保存到:', RUNS)
""")
md("""
## 3. 复制代码与数据到本次运行时，并校验文件

避免训练时从 Drive 逐张读取图片。校验完成后，代码与数据位于 `/content`，检查点仍直接写入 Drive。
源文件 ZIP 的固定副本保留在 Drive 的 `packages` 中，便于追溯。校验失败时重新上传原始 ZIP。
""")
code("""
def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

local_zip = Path('/content/hanzi81_colab_bundle.zip')
shutil.copy2(BUNDLE, local_zip)
bundle_hash = file_sha256(local_zip)
WORK = Path('/content') / ('hanzi81-' + bundle_hash[:12])
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(local_zip) as archive:
    for entry in archive.infolist():
        destination = (WORK / entry.filename).resolve()
        assert destination.is_relative_to(WORK.resolve()), '非法 ZIP 路径'
    archive.extractall(WORK)
payload = json.loads((WORK / 'bundle_manifest.json').read_text(encoding='utf-8'))
for name, expected in payload['files'].items():
    assert file_sha256(WORK / name) == expected, f'文件校验失败: {name}'
os.chdir(WORK)
sys.path.insert(0, str(WORK))
from training.io_utils import verify_dataset
DATA = WORK / 'training/data/baseline81-v2'
manifest = verify_dataset(DATA)
assert {k: v['images'] for k, v in manifest['splits'].items()} == {
    'train': 15552, 'val_graph': 7776, 'val_select': 7776, 'test': 15552}
print(json.dumps(manifest['splits'], ensure_ascii=False, indent=2))
print('独立像素数:', manifest['audit']['unique_pixel_hashes'])
packages = DRIVE_ROOT / 'packages'
packages.mkdir(exist_ok=True)
snapshot = packages / (bundle_hash + '.zip')
if not snapshot.exists():
    temp_snapshot = snapshot.with_suffix('.zip.tmp')
    shutil.copy2(local_zip, temp_snapshot)
    assert file_sha256(temp_snapshot) == bundle_hash
    temp_snapshot.replace(snapshot)
print('数据与代码校验通过:', WORK)
""")
md("""
## 4. 查看训练样例（不读取测试集）

每张图是 64×64 灰度。图上方显示类别 ID；字符见输出的标签对照。后续比较使用同一份固定数据。
""")
code("""
import matplotlib.pyplot as plt
label_data = json.loads((DATA / 'labels.json').read_text(encoding='utf-8'))
with np.load(DATA / 'train.npz', allow_pickle=False) as samples:
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    for index, ax in enumerate(axes.flat):
        ax.imshow(samples['images'][index], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        class_id = int(samples['labels'][index])
        ax.set_title(f'ID {class_id}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
print(' / '.join(f"{c['class_id']}: {c['char']}" for c in label_data['characters'][:16]))
""")
md("""
## 5. 一轮短跑：检查训练、验证和 Drive 保存

仅每类 2 张训练图和 2 张验证图，共 162 张，运行 1 epoch。这里只检查整个流程，不判断模型好坏。
完成后应有 `config.json`、`history.json`、`best.pt`、`last.pt`。再次运行会复用已完成的短跑。
""")
code("""
def run_training(name, mode, epochs, per_class=None, stop_after=None):
    folder = RUNS / name
    cmd = [sys.executable, '-u', '-m', 'training.train', 'train', '--data', str(DATA),
           '--run-dir', str(folder), '--mode', mode, '--epochs', str(epochs),
           '--batch-size', '64', '--seed', '42']
    if per_class is not None:
        cmd += ['--per-class', str(per_class)]
    if stop_after is not None:
        cmd += ['--stop-after', str(stop_after)]
    if (folder / 'last.pt').exists():
        cmd += ['--resume']
    subprocess.run(cmd, cwd=WORK, check=True)
    for filename in ['config.json', 'history.json', 'best.pt', 'last.pt']:
        assert (folder / filename).is_file(), f'未成功保存 {filename}'
    (folder / 'source_bundle_sha256.txt').write_text(bundle_hash + '\\n', encoding='utf-8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (folder / 'environment-pip-freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    print('已保存并检查:', folder)
    return folder

smoke_dir = run_training('smoke_s42_v1', 'smoke', 1, per_class=2)
""")
md("""
## 6. 小训练集拟合检查

仍使用每类 2 张训练图，训练 20 epoch，验证也在这同一小训练集上，结果只用于排查标签、模型和优化流程。
应看到 loss 明显下降、训练集准确率显著提高并趋向拟合；不把这里的高准确率当成泛化结果。
若仍难以拟合，请先检查曲线和错误，暂不开启完整训练。该 run 与正式基线完全分开。
""")
code("""
overfit_dir = run_training('overfit_s42_v1', 'overfit', 20, per_class=2)
overfit_history = json.loads((overfit_dir / 'history.json').read_text(encoding='utf-8'))
print('首轮:', overfit_history[0]['validation'])
print('末轮:', overfit_history[-1]['validation'])
plt.plot([r['epoch'] for r in overfit_history], [r['validation']['accuracy'] for r in overfit_history])
plt.xlabel('epoch'); plt.ylabel('training subset accuracy'); plt.grid(); plt.show()
""")
md("""
## 7. 启动 81 字完整基线

前两项检查正常后，把 `START_BASELINE` 改为 `True`。默认先完成 **2/40 epoch**，
这是同一次 40 epoch 计划的前两轮，用于检查耗时和算力消耗，不改变余弦学习率预算。
确认运行正常后，把 `EPOCHS_THIS_SESSION` 改为 `None` 并重新运行本格，即继续完成剩余轮次。

设置：ResNet18 灰度 3×3/stride1 输入、无首层最大池化、随机初始化、交叉熵、AdamW、
batch=64、lr=0.001、weight_decay=0.0001、最多 40 epoch、seed=42。
按 `val_select` macro accuracy 最高选择 `best.pt`，并列取 CE 更低，再并列取更早的 epoch。
40 epoch 是开发预算，是否收敛仍需看结果。

每完成一轮就保存模型、优化器、学习率计划、AMP scaler 和随机状态到 `last.pt`。
运行时被回收后只会从最近完整 epoch 恢复，本轮未完成部分需重跑。不要同时在两个会话写同一个 run。
""")
code("""
START_BASELINE = False  # 前面的检查正常后改为 True
EPOCHS_THIS_SESSION = 2  # 先计时两轮；继续完整训练时改为 None
BASELINE_NAME = 'baseline81_s42_v1'
baseline_dir = RUNS / BASELINE_NAME
if START_BASELINE:
    baseline_dir = run_training(BASELINE_NAME, 'baseline', 40, stop_after=EPOCHS_THIS_SESSION)
else:
    print('基线尚未启动。短跑与拟合检查正常后，将 START_BASELINE 改为 True。')
""")
md("""
## 8. 查看曲线与开发集报告

默认仅评估 `val_select`；此开发集用于选基线检查点，不能代替独立测试证据。
当前只完成两轮时，以下曲线也是中间结果。`val_graph` 留待基线确定后做一次混淆统计，
测试集 `test` 留待协议冻结后最终评价，本 Notebook 不自动评估二者。
""")
code("""
if (baseline_dir / 'history.json').is_file():
    history = json.loads((baseline_dir / 'history.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for key, caption in [('train', 'train'), ('validation', 'val_select')]:
        axes[0].plot([r['epoch'] for r in history], [r[key]['loss'] for r in history], label=caption)
        axes[1].plot([r['epoch'] for r in history], [r[key]['accuracy'] for r in history], label=caption)
    axes[0].set_ylabel('cross entropy'); axes[1].set_ylabel('accuracy')
    for ax in axes:
        ax.set_xlabel('epoch'); ax.grid(); ax.legend()
    plt.tight_layout(); plt.show()
    subprocess.run([sys.executable, '-u', '-m', 'training.train', 'evaluate', '--data', str(DATA),
                    '--run-dir', str(baseline_dir), '--split', 'val_select'], cwd=WORK, check=True)
    report = json.loads((baseline_dir / 'evaluation_val_select/metrics.json').read_text(encoding='utf-8'))
    print(json.dumps(report, ensure_ascii=False, indent=2))
else:
    print('基线尚未启动。')
""")
md("""
## 9. 保存后的下一步

Drive 路径：`hanzi81/runs/baseline81_s42_v1/`。

- `best.pt`：开发集选出的模型；`last.pt`：完整续训状态。
- `history.json`：每轮损失、准确率和计算时间；`config.json`：超参数、版本、GPU 和数据哈希。
- `evaluation_val_select/`：总体／模糊分项指标、混淆矩阵、54 个预设字对误认、逐图预测。

首次 2 轮后，把 GPU 型号、epoch 耗时、train/val 指标及任何报错反馈给我，再决定后续预算。
完成训练并确认 Drive 中保存成功后，菜单选择“断开连接并删除运行时”，停止占用算力。
不要仅靠关闭浏览器标签页停止计费。Notebook 分享时输出可能随之分享，分享前检查输出。

若显存不足：降低 batch size 后换一个 run 名重新开始，不在同一正式 run 中途改变配置。
若 Colab 更换软件／GPU 导致恢复检查失败：使用原运行时配置恢复，或建立新 run，不能直接删除检查。
更多步骤、故障处理和最终测试命令见上传包内 `colab/README.md`。
""")

notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": "hanzi81_baseline.ipynb", "provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "cells": cells}
for index, cell in enumerate(cells):
    cell["id"] = f"hanzi81-{index:02d}"
    if cell["cell_type"] == "code":
        compile(cell["source"], f"cell-{index}", "exec")
destination = ROOT / "hanzi81_baseline.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
