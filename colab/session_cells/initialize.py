#@title ① 初始化与恢复（不会启动训练）
from pathlib import Path
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import zipfile

PREPARE_TRAINING = True #@param {type:"boolean"}
DRIVE_ROOT = Path("/content/drive/MyDrive/hanzi81")
LOCAL_ROOT = Path("/content")

BUNDLE_SHA256 = "e477bdd4894e6b585a2149d6f22db3948e5d6e20075d532fded5171d1b57dd35"
BUNDLE_MANIFEST_SHA256 = "42004e354a1e70a7325f82b7c5752842698caf6476543eaa149f494db6113ec5"
CURVE_SHA256 = "2b098c465e94fd1909c159c1a64b59745e914c13023e1a23d8a3445e2231272d"
SUITE_RELATIVE = "analyses/baseline81_s42_v1/03_learning_curve/steps2430_s42_v1"


def setup_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_mount(root):
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    if not root.is_dir():
        raise FileNotFoundError(f"项目目录不存在，请检查 Drive 挂载和 DRIVE_ROOT：{root}")


def setup_verify_work(work):
    manifest_path = work / "bundle_manifest.json"
    if not manifest_path.is_file():
        return False
    if setup_digest(manifest_path) != BUNDLE_MANIFEST_SHA256:
        raise ValueError("本地包清单与原实验不同；保留现场，恢复原始包，不绕过校验。")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = False
    for name, expected in payload["files"].items():
        path = (work / name).resolve()
        if not path.is_relative_to(work.resolve()):
            raise ValueError("包清单包含非法路径")
        if not path.is_file():
            missing = True
        elif setup_digest(path) != expected:
            raise ValueError(f"本地文件校验失败：{name}；未覆盖此文件，请先检查原因。")
    return not missing


def setup_verified_copy(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    shutil.copyfile(source, temporary)
    if setup_digest(temporary) != expected:
        raise ValueError(f"复制内容校验失败，未替换目标：{target}")
    temporary.replace(target)


def setup_restore_bundle(root, local_root):
    work = local_root / ("hanzi81-" + BUNDLE_SHA256[:12])
    if setup_verify_work(work):
        print("复用已校验的本地代码与数据：", work)
        return work
    local_zip = local_root / "hanzi81_colab_bundle.zip"
    archive_snapshot = root / "packages" / (BUNDLE_SHA256 + ".zip")
    if not local_zip.is_file() or setup_digest(local_zip) != BUNDLE_SHA256:
        candidates = [root / "hanzi81_colab_bundle.zip", archive_snapshot]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"请恢复原始上传包：{candidates[0]}")
        setup_verified_copy(source, local_zip, BUNDLE_SHA256)
    with zipfile.ZipFile(local_zip) as archive:
        for entry in archive.infolist():
            destination = (work / entry.filename).resolve()
            if not destination.is_relative_to(work.resolve()) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("ZIP 包含非法路径或符号链接")
        # ZIP 哈希与固定原包一致，缺失的运行时文件可以从它恢复。
        work.mkdir(parents=True, exist_ok=True)
        archive.extractall(work)
    if not setup_verify_work(work):
        raise ValueError("解包后仍有文件缺失")
    if not archive_snapshot.exists():
        setup_verified_copy(local_zip, archive_snapshot, BUNDLE_SHA256)
    print("已从固定原包恢复：", work)
    return work


def setup_restore_curve(root, local_root):
    suite = root / SUITE_RELATIVE
    plan_path, source = suite / "plan.json", suite / "learning_curve.py"
    if not plan_path.is_file() or not source.is_file():
        raise FileNotFoundError(
            f"找不到已准备的学习曲线协议/脚本：{suite}。请检查 Drive 路径；本单元不会另建实验。")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["curve_script_sha256"] != CURVE_SHA256 or setup_digest(source) != CURVE_SHA256:
        raise ValueError("Drive 中的学习曲线脚本与冻结版本不一致")
    if (plan["sizes"], plan["blocks"], plan["block_draws"], plan["batch_size"], plan["seed"]) != (
            [24, 48, 96, 192], 10, 15552, 64, 42):
        raise ValueError("学习曲线预算或种子与当前实验不同")
    target = local_root / "hanzi81_learning_curve.py"
    if not target.is_file() or setup_digest(target) != CURVE_SHA256:
        setup_verified_copy(source, target, CURVE_SHA256)
    print("学习曲线脚本已恢复；每档总预算仍为 10 块 / 2430 批次。")
    return target


def setup_probe(work, reference):
    # 在子进程中检查 GPU；不向 Notebook 主进程长期保留 CUDA 上下文或模型。
    code = """
import json
from pathlib import Path
import torch
from training import train as T
cfg = json.loads(Path(__import__('sys').argv[1]).read_text(encoding='utf-8'))
if not torch.cuda.is_available():
    raise RuntimeError('未分配 GPU；只读报告可直接运行第③格，无需 GPU。')
actual = T.environment(torch.device('cuda'))
print(json.dumps(actual, ensure_ascii=False, indent=2), flush=True)
if actual != cfg['environment']:
    difference = {k: {'saved': cfg['environment'].get(k), 'current': actual.get(k)}
                  for k in set(actual) | set(cfg['environment'])
                  if actual.get(k) != cfg['environment'].get(k)}
    raise RuntimeError('环境与原实验不同，禁止直接续训：' + json.dumps(difference, ensure_ascii=False))
with torch.inference_mode():
    probe = torch.nn.Conv2d(1, 8, 3).cuda()
    print('GPU 前向检查:', tuple(probe(torch.zeros(2, 1, 64, 64, device='cuda')).shape))
"""
    subprocess.run([sys.executable, "-u", "-c", code, str(reference)], cwd=work, check=True)


def initialize_session(root, local_root, prepare_training):
    setup_mount(root)
    if not prepare_training:
        print("Drive 已连接；可直接运行第③格。未导入 torch、复制数据或执行 GPU 检查。")
        return
    work = setup_restore_bundle(root, local_root)
    data = work / "training/data/baseline81-v2"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    counts = {k: v["images"] for k, v in manifest["splits"].items()}
    if counts != {"train": 15552, "val_select": 7776, "val_graph": 7776, "test": 15552}:
        raise ValueError("数据规模与原实验不同")
    print("四份文件数量校验：", counts, "（未执行测试集推理）")
    setup_restore_curve(root, local_root)
    reference = root / "runs/baseline81_s42_v1/config.json"
    if not reference.is_file():
        raise FileNotFoundError(f"缺少原基线配置：{reference}")
    cfg = json.loads(reference.read_text(encoding="utf-8"))
    for key, path in [("trainer_sha256", work / "training/train.py"),
                      ("io_sha256", work / "training/io_utils.py"),
                      ("data_sha256", data / "manifest.json")]:
        if cfg[key] != setup_digest(path):
            raise ValueError(f"原基线配置与源文件不同：{key}")
    setup_probe(work, reference)
    print("初始化完成。第②格默认关闭；需续训时显式开启。")
    print("第 6 项可在第③格复用已有报告完成分析；本次初始化不会重跑计时。")


if __name__ == "__main__":
    initialize_session(DRIVE_ROOT, LOCAL_ROOT, PREPARE_TRAINING)
