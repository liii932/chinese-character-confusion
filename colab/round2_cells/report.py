"""Plot saved history and generate the two development reports after all ten epochs."""
import json
import numpy as np
import matplotlib.pyplot as plt

baseline_dir = DRIVE_ROOT / 'runs' / RUN_NAME
history = json.loads((baseline_dir / 'history.json').read_text(encoding='utf-8'))
if not history:
    raise ValueError('尚无已完成的训练轮次。请先运行训练格。')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for key, caption in (('train', 'train'), ('validation', 'val_select')):
    epochs = [r['epoch'] for r in history]
    axes[0].plot(epochs, [r[key]['loss'] for r in history], marker='.', label=caption)
    axes[1].plot(epochs, [r[key]['accuracy'] for r in history], marker='.', label=caption)
axes[0].set_ylabel('cross entropy')
axes[1].set_ylabel('accuracy')
for ax in axes:
    ax.set_xlabel('epoch')
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(baseline_dir / 'learning_curves.png', dpi=150)
plt.show()
plt.close(fig)

completed_splits = evaluate_completed_round2(WORK, DATA, baseline_dir)
summary = {'completed_epochs': len(history), 'planned_epochs': TOTAL_EPOCHS,
           'run': RUN_NAME, 'data_archive_sha256': DATA_ARCHIVE_SHA256,
           'training_source_id': SOURCE_ID,
           'seconds_compute_train_and_validation': sum(r['seconds_compute'] for r in history),
           'development_reports': {}}
label_rows = json.loads((DATA / 'labels.json').read_text(encoding='utf-8'))['characters']
for split in completed_splits:
    directory = baseline_dir / f'evaluation_{split}'
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    summary['development_reports'][split] = metrics
    print(split, json.dumps(metrics, ensure_ascii=False, indent=2))
    matrix = np.load(directory / 'confusion.npy', allow_pickle=False)
    if matrix.shape != (1140, 1140) or int(matrix.sum()) != 109440:
        raise ValueError(f'{split} 混淆矩阵规模不正确。')
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    errors = off_diagonal.sum(axis=1)
    indexes = np.argsort(-errors)[:20]
    print('错误最多的类别（真实字 / 错误数 / 最常误认字 / 该方向次数）：')
    for source in indexes:
        if not errors[source]:
            break
        target = int(off_diagonal[source].argmax())
        print(label_rows[source]['char'], int(errors[source]), label_rows[target]['char'], int(off_diagonal[source, target]))
    summary['development_reports'][split]['sources_with_top_confusion_ge5'] = int((off_diagonal.max(axis=1) >= 5).sum())
(baseline_dir / 'notebook_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'已完成 {len(history)}/{TOTAL_EPOCHS} 轮。完整曲线与报告：{baseline_dir}')
print('确认 Drive 中保存完成后，断开连接并删除运行时。')
