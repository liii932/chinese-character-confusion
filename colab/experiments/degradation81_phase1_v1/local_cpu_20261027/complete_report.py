import csv
import json
from pathlib import Path

import numpy as np
import torch

from training import degradation as D
from training import train as T
from training.io_utils import digest, write_json

output = Path(__file__).resolve().parent
data = D.ROOT / 'training/data/degradation81-calibration-v1'
checkpoint_path = D.ROOT / 'colab/best.pt'
rows = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
assert len(rows) == 25 and all(row['samples'] == 1296 for row in rows)
with np.load(output / 'confusion.npz') as saved:
    matrices = saved['matrices']
    assert matrices.shape == (25, 81, 81) and np.all(matrices.sum(axis=(1, 2)) == 1296)
with np.load(data / 'sweep.npz') as saved:
    images, source_indexes = saved['images'], saved['source_indexes']

wrong_examples = []
counts = [0] * 25
wrong_counts = [0] * 25
with (output / 'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
    for prediction in csv.DictReader(stream):
        grid = int(prediction['grid_id'])
        index = counts[grid]
        assert int(prediction['source_index']) == int(source_indexes[index])
        counts[grid] += 1
        if prediction['true_id'] != prediction['predicted_id']:
            if wrong_counts[grid] < 4:
                wrong_examples.append({'pixels': images[grid, index], 'true': prediction['true_char'],
                                       'predicted': prediction['predicted_char'], 'sigma': float(prediction['sigma']),
                                       'resolution': int(prediction['resolution']),
                                       'source_index': int(prediction['source_index'])})
            wrong_counts[grid] += 1
assert counts == [1296] * 25
assert wrong_counts == [row['errors'] for row in rows]
assert digest(checkpoint_path) == D.CHECKPOINT_SHA256
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
D.write_report(output, rows, wrong_examples)
summary = {'status': 'completed_frozen_model_sweep', 'samples': sum(counts), 'grids': len(rows),
           'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_sha256': D.CHECKPOINT_SHA256,
           'calibration_manifest_sha256': digest(data / 'manifest.json'), 'pixel_sha256': manifest['pixel_sha256'],
           'environment': T.environment(torch.device('cpu')),
           'original_training_environment': checkpoint['config']['environment'], 'precision': 'fp32_no_autocast',
           'clean_control_accuracy': rows[0]['macro_accuracy'], 'clean_control_matches_archived': rows[0]['errors'] == 0,
           'evaluator_sha256': digest(D.__file__), 'human_readability': 'not_measured',
           'retrained_baseline': 'not_run', 'test_evaluated': False,
           'interpretation': 'New-corruption evaluation of the old model; not final-test or matched-training evidence.',
           'report_recovered_from_saved_predictions': True}
write_json(output / 'summary.json', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
