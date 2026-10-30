"""Generate the self-contained 1,140-class, ten-epoch Colab notebook."""
import hashlib
import json
from pathlib import Path
import textwrap

from training.io_utils import digest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'colab'
RECORD = HERE / 'experiments/round2_1140_v1'


def generate():
    delivery = json.loads((RECORD / 'delivery.json').read_text(encoding='utf-8'))
    sources = {name: (ROOT / name).read_text(encoding='utf-8') for name in (
        'training/__init__.py', 'training/io_utils.py', 'training/train.py')}
    source_id = hashlib.sha256(json.dumps(sources, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest_hash = digest(ROOT / 'training/data/round2-1140-v1/manifest.json')
    cells = []

    def md(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n'})

    def code(source):
        source = textwrap.dedent(source).strip() + '\n'
        compile(source, f'round2-cell-{len(cells)}', 'exec')
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': source,
                      'execution_count': None, 'outputs': []})

    md('''
    # 第二轮：1,140 字完整基线 · 训练 10 轮

    “10层”在此按 **10 个 epoch（训练轮次）**设置；网络沿用首轮 ResNet18。
    1,140 类随机初始化，使用已生成的完整训练集 218,880 张与 val_select 109,440 张。

    **上传两个文件即可开始：**
    1. 在“我的云端硬盘”中新建 `hanzi1140_round2`，上传原有 `hanzi1140_round2_data.zip`，保留文件名。
    2. 在 Colab 上传本 Notebook，并保存到 Drive。选择 Python 3 和 GPU，先使用可用的 T4。
    3. 从上到下运行六个代码单元：GPU → 数据准备 → 样例 → 小集短跑 → 完整 10 轮 → 曲线与开发报告。

    训练代码已内嵌，无需另传脚本、字体或首轮模型。完整训练格默认一次完成 10 轮。
    每完成一轮自动保存到 Drive；断线后重跑前面的准备格，再运行同一训练格即可恢复。
    ''')
    md('''
    ## ① 检查 GPU

    使用 Colab 预装的 torch/torchvision，不升级这两个包。实际前向检查在子进程中完成。
    ''')
    probe = '''import sys
import torch, torchvision, numpy as np
print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'Torchvision:', torchvision.__version__, 'NumPy:', np.__version__)
if not torch.cuda.is_available():
    raise RuntimeError('未分配 GPU：请更改运行时类型为 GPU，再重新连接。')
print('GPU:', torch.cuda.get_device_name(0))
print('显存 GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
with torch.inference_mode():
    model = torch.nn.Conv2d(1, 8, 3).cuda()
    print('GPU 前向检查:', tuple(model(torch.zeros(2, 1, 64, 64, device='cuda')).shape))
'''
    code('import subprocess, sys\nsubprocess.run([sys.executable, "-u", "-c", ' + repr(probe) + '], check=True)')
    md('''
    ## ② 连接 Drive，恢复固定代码与数据

    在授权窗口完成 Drive 挂载。若文件夹名不同，修改本格中的 `DRIVE_ROOT`。
    首次复制约 732 MiB 数据包到 `/content`，只解出训练所需的八个文件，校验后使用。
    四集合的像素审计已在本机完成；此处检查文件指纹，不重新渲染或扫描全部像素。
    源码快照保存到 `hanzi1140_round2/sources/`，代码和数据的标识随 run 保存。
    ''')
    prefix = ("from pathlib import Path\nDRIVE_ROOT = Path('/content/drive/MyDrive/hanzi1140_round2')\n\n"
              'DATA_ARCHIVE_SHA256 = ' + repr(delivery['sha256']) + '\n'
              'DATA_MANIFEST_SHA256 = ' + repr(manifest_hash) + '\n'
              'TRAINING_SOURCES = ' + repr(sources) + '\n\n')
    code(prefix + (HERE / 'round2_cells/prepare.py').read_text(encoding='utf-8'))
    md('''
    ## ③ 检查训练图片

    从全字表均匀选择 16 个类别，显示实际训练图。字符对照在图下方输出。
    ''')
    code('''
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    labels = json.loads((DATA / 'labels.json').read_text(encoding='utf-8'))['characters']
    indexes = np.linspace(0, 1139, 16, dtype=int)
    with np.load(DATA / 'train.npz', allow_pickle=False) as saved:
        # Decode the large image array once, then retain only the 16 preview tiles.
        tiles = saved['images'][indexes]
        targets = saved['labels'][indexes]
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    for image, target, ax in zip(tiles, targets, axes.flat):
        ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(f'ID {int(target)}')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(DRIVE_ROOT / 'sample_preview.png', dpi=150)
    plt.show()
    plt.close(fig)
    print(' / '.join(f"{int(i)}: {labels[int(i)]['char']}" for i in targets))
    del tiles, targets
    ''')
    md('''
    ## ④ 一轮小集短跑

    每类选 2 张训练图和 2 张 val_select 图，检查 1,140 类输出、梯度与 Drive 保存。
    这个独立 run 训练 1 轮，检查通过后下一格新建完整基线；完整基线的预算为 10 轮。
    再次运行会复用已完成的短跑检查点。此处准确率只用于流程检查。
    ''')
    code((HERE / 'round2_cells/train.py').read_text(encoding='utf-8') + '''

smoke_dir = run_round2_training(WORK, DATA, DRIVE_ROOT, RUN_NAME + '_smoke', 'smoke', 1, per_class=2)
''')
    md('''
    ## ⑤ 完整训练 10 轮

    运行本格会开始训练。模型、损失与优化设置沿用首轮：灰度 ResNet18、3×3 stride=1 输入、
    无首层 maxpool、随机初始化、交叉熵、AdamW、batch=64、lr=0.001、weight_decay=0.0001、seed=42。
    余弦学习率按 **总计 10 轮**下降到初始值的 1/100；best.pt 按 val_select macro accuracy 最高选择，
    并列取交叉熵更低者。训练采用 CUDA AMP，验证采用 fp32。

    `EPOCHS_THIS_SESSION=None` 一次完成全部轮次；设为 2 时，本次最多追加两轮，总预算仍为 10。
    断线后重新准备数据并运行此格，会从 `last.pt` 的最近完整轮次恢复。
    保持原 GPU 型号、软件版本和超参数；需要改 batch 等设置时，改第④格的 `RUN_NAME` 新建 run。
    同一 run 只在一个 Colab 会话中写入。
    ''')
    code('''
    EPOCHS_THIS_SESSION = None  # None=本次完成剩余全部轮次；2=本次最多追加两轮
    baseline_dir = run_round2_training(
        WORK, DATA, DRIVE_ROOT, RUN_NAME, 'baseline', TOTAL_EPOCHS,
        stop_after=EPOCHS_THIS_SESSION)
    ''')
    md('''
    ## ⑥ 曲线与完整开发报告

    先保存损失／准确率曲线；完成 10 轮后，用 best.pt 分别评估 val_select 与 val_graph。
    输出总体和 sigma 分项准确率、1,140×1,140 混淆矩阵、首轮保留的 54 对误认及逐图预测。
    新增字组保存在标签表中，多字组未自动展开为全部两两关系。
    报告会列出错误最多的类别，便于下一步判断扩类后的错误空间；10 轮完成本身不代表已经收敛。

    结果目录：`MyDrive/hanzi1140_round2/runs/baseline1140_e10_s42_v1/`。
    `best.pt` 是所选模型，`last.pt` 保存完整恢复状态，`history.json` 和 `config.json` 保存训练记录。
    曲线为 `learning_curves.png`，报告在 `evaluation_val_select/` 与 `evaluation_val_graph/`，
    汇总为 `notebook_summary.json`。同一检查点的完整开发报告会复用。
    test 留待最终评价，本 Notebook 仅执行两个开发集推理。
    确认 Drive 中保存完成后，在菜单中断开连接并删除运行时。
    ''')
    code((HERE / 'round2_cells/report.py').read_text(encoding='utf-8'))
    for i, cell in enumerate(cells):
        cell['id'] = f'round2-1140-{i:02d}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'colab': {'name': 'hanzi1140_round2_e10.ipynb', 'provenance': []},
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'}, 'accelerator': 'GPU',
            'round2_protocol': {'classes': 1140, 'epochs': 10, 'seed': 42, 'batch_size': 64,
                'model': 'resnet18-gray3x3-s1-no-maxpool-random',
                'data_archive_sha256': delivery['sha256'], 'training_source_id': source_id}}, 'cells': cells}
    destination = HERE / 'hanzi1140_round2_e10.ipynb'
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record = {'notebook': destination.relative_to(ROOT).as_posix(), 'sha256': digest(destination),
        'classes': 1140, 'baseline_epochs': 10, 'separate_smoke_epochs': 1,
        'data_archive_sha256': delivery['sha256'], 'data_manifest_sha256': manifest_hash,
        'training_source_id': source_id,
        'embedded_source_sha256': {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        'status': 'notebook_prepared_no_colab_training_run'}
    (RECORD / 'notebook_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(destination)
    return notebook


if __name__ == '__main__':
    generate()
