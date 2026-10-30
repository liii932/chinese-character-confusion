#@title ② 分段续训学习曲线（默认关闭，一次一个档位）
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import uuid

RUN_TRAINING = False #@param {type:"boolean"}
PER_CLASS = 192 #@param [192, 96, 48, 24] {type:"raw"}
BLOCKS_THIS_SESSION = 3 #@param {type:"integer"}
DRIVE_ROOT = Path("/content/drive/MyDrive/hanzi81")
LOCAL_ROOT = Path("/content")

TRAIN_CURVE_SHA256 = "2b098c465e94fd1909c159c1a64b59745e914c13023e1a23d8a3445e2231272d"
TRAIN_SOURCE_SHA256 = "f31aece79c2322772139f2f299aa88dc272ec7d02ef3308bcaeeeb48e8cd4197"
TRAIN_IO_SHA256 = "6e0effced2c2a35bd474fba9925404bf84182d8c39348e365052bc7d0d9cb183"
TRAIN_DATA_SHA256 = "6a2d237ae03ed219f76c2d4cf827eeb6b9c852c88c54a5a77d6ca34706d84f86"


def training_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train_session(root, local_root, enabled, per_class, extra_blocks):
    if not enabled:
        print("续训关闭。仅查看已有结果可直接运行第③格。")
        return
    if type(per_class) is not int or per_class not in (24, 48, 96, 192):
        raise ValueError("PER_CLASS 只能为 24、48、96 或 192")
    if type(extra_blocks) is not int or not 1 <= extra_blocks <= 10:
        raise ValueError("BLOCKS_THIS_SESSION 必须是 1～10 的整数；它不改变每档总预算。")
    work = local_root / "hanzi81-e477bdd4894e"
    data = work / "training/data/baseline81-v2"
    suite = root / "analyses/baseline81_s42_v1/03_learning_curve/steps2430_s42_v1"
    plan_path = suite / "plan.json"
    for path in [plan_path, data / "train.npz", data / "val_select.npz"]:
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}。请先运行第①格恢复 Drive、代码和数据。")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan["curve_script_sha256"], plan["blocks"], plan["block_draws"],
            plan["batch_size"], plan["seed"], plan["sizes"]) != (
            TRAIN_CURVE_SHA256, 10, 15552, 64, 42, [24, 48, 96, 192]):
        raise ValueError("冻结实验协议不同；未启动训练。")
    # 本格拥有全部导入与路径，不调用其他单元临时定义的辅助函数。
    source = suite / "learning_curve.py"
    expected_files = {source: TRAIN_CURVE_SHA256,
                      work / "training/train.py": TRAIN_SOURCE_SHA256,
                      work / "training/io_utils.py": TRAIN_IO_SHA256,
                      data / "manifest.json": TRAIN_DATA_SHA256}
    for path, expected in expected_files.items():
        if not path.is_file() or training_digest(path) != expected:
            raise ValueError(f"缺失或校验失败：{path}。请运行第①格检查。")
    script = local_root / "hanzi81_learning_curve.py"
    if not script.is_file() or training_digest(script) != TRAIN_CURVE_SHA256:
        script.write_bytes(source.read_bytes())
    if training_digest(script) != TRAIN_CURVE_SHA256:
        raise ValueError("恢复学习曲线脚本失败")
    command = [sys.executable, "-u", str(script), "train", "--work", str(work),
               "--data", str(data), "--root", str(root), "--sizes", str(per_class),
               "--stop-after-blocks", str(extra_blocks)]
    log_dir = suite / "session_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    log_path = log_dir / f"n{per_class:03d}_{stamp}.txt"
    print(f"本次：每字 {per_class} 图，最多追加 {extra_blocks} 块；总预算仍是 10 块。", flush=True)
    print("已完成档位由原脚本校验后跳过。日志：", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "plan_sha256": training_digest(plan_path)},
                             ensure_ascii=False) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                print("进程已停止；下次从最近成功保存的完整块恢复。请运行第③格检查。")
                raise
    if returncode:
        raise RuntimeError(f"训练退出码 {returncode}；请保留并反馈日志：{log_path}")
    arm = suite / f"n{per_class:03d}_s42"
    required = [arm / name for name in ("config.json", "history.json", "best.pt", "last.pt")]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("本次进程已退出，但检查点文件不完整；请先检查保存情况。")
    history = json.loads((arm / "history.json").read_text(encoding="utf-8"))
    if not history or [r["block"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("训练日志不是连续完整块，请保留现场检查。")
    print(f"本档已记录 {len(history)}/10 个完整块；本次分段正常结束。")
    print("运行第③格查看四档进度。未完成全部档位属于正常状态。")
    print("确认 Drive 文件保存后，可断开连接并删除运行时；不要同时在两个会话续训同一档位。")


if __name__ == "__main__":
    train_session(DRIVE_ROOT, LOCAL_ROOT, RUN_TRAINING, PER_CLASS, BLOCKS_THIS_SESSION)
