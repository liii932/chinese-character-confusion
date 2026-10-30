"""Shared notebook helpers using the unchanged first-round training entry point."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

TOTAL_EPOCHS = 10
BATCH_SIZE = 64
SEED = 42
RUN_NAME = 'baseline1140_e10_s42_v1'


def checkpoint_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_round2_training(work, data, root, name, mode, epochs, per_class=None, stop_after=None):
    folder = Path(root) / 'runs' / name
    command = [sys.executable, '-u', '-m', 'training.train', 'train',
        '--data', str(data), '--run-dir', str(folder), '--mode', mode,
        '--epochs', str(epochs), '--batch-size', str(BATCH_SIZE), '--seed', str(SEED),
        '--lr', '0.001', '--weight-decay', '0.0001']
    if per_class is not None:
        command += ['--per-class', str(per_class)]
    if stop_after is not None:
        command += ['--stop-after', str(stop_after)]
    if (folder / 'last.pt').exists():
        command += ['--resume']
    subprocess.run(command, cwd=work, check=True)
    for name in ('config.json', 'history.json', 'best.pt', 'last.pt'):
        if not (folder / name).is_file():
            raise RuntimeError(f'训练输出缺失：{folder / name}')
    provenance = {'data_archive_sha256': DATA_ARCHIVE_SHA256, 'training_source_id': SOURCE_ID}
    (folder / 'notebook_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    environment = folder / 'environment-pip-freeze.txt'
    if not environment.exists():
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True)
        environment.write_text(freeze.stdout, encoding='utf-8')
    print('已保存：', folder)
    return folder


def evaluate_completed_round2(work, data, folder):
    folder = Path(folder)
    history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
    if [r['epoch'] for r in history] != list(range(1, TOTAL_EPOCHS + 1)):
        print(f'当前 {len(history)}/{TOTAL_EPOCHS} 轮；完整训练后才生成开发集混淆报告。')
        return []
    best_hash = checkpoint_digest(folder / 'best.pt')
    marker = folder / 'development_complete.json'
    splits = ('val_select', 'val_graph')
    if marker.exists() and json.loads(marker.read_text())['checkpoint_sha256'] == best_hash:
        required = [folder / f'evaluation_{split}' / name for split in splits
                    for name in ('metrics.json', 'confusion.npy', 'pairs.json', 'predictions.csv')]
        if all(path.is_file() for path in required):
            print('复用已保存的两个开发集报告。')
            return list(splits)
    for split in splits:
        subprocess.run([sys.executable, '-u', '-m', 'training.train', 'evaluate',
            '--data', str(data), '--run-dir', str(folder), '--split', split,
            '--batch-size', str(BATCH_SIZE)], cwd=work, check=True)
    temporary = marker.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'checkpoint_sha256': best_hash, 'splits': list(splits)}, indent=2) + '\n')
    temporary.replace(marker)
    return list(splits)
