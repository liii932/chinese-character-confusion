"""Evaluate a cached PaddleOCR recognizer on the fixed development calibration grid."""
import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import paddle
import paddleocr
from paddleocr import TextRecognition


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'training/data/degradation81-calibration-v1'
BASELINE = ROOT / 'colab/experiments/degradation81_phase1_v1/local_cpu_20261027/summary.csv'
MODELS = Path(r'C:\Users\heshi\.paddlex\official_models')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model', choices=['PP-OCRv5_server_rec', 'PP-OCRv5_mobile_rec'], default='PP-OCRv5_server_rec')
    args = parser.parse_args()
    model_path = MODELS / args.model
    output = ROOT / 'colab/experiments/degradation81_phase1_v1' / f'paddleocr_{args.model}_cpu_20261027'
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if sha256(DATA / name) != expected:
            raise ValueError(f'Dataset hash mismatch: {name}')
    labels = json.loads((DATA / 'labels.json').read_text(encoding='utf-8'))
    chars = [x['char'] for x in labels['characters']]
    with np.load(DATA / 'sweep.npz', allow_pickle=False) as data:
        images, truth = data['images'], data['labels']
        source_indexes, source_ids = data['source_indexes'], data['source_condition_ids']
    model = TextRecognition(model_name=args.model, model_dir=str(model_path), device='cpu',
                            enable_mkldnn=True, cpu_threads=4)
    if args.smoke:
        selected = images[0, :81]
        inputs = [np.repeat(image[:, :, None], 3, axis=2) for image in selected]
        start = time.perf_counter()
        result = model.predict(inputs, batch_size=args.batch_size)
        print(json.dumps({'samples': len(result), 'correct': sum(x['rec_text'] == chars[int(y)]
              for x, y in zip(result, truth[:81])), 'seconds': time.perf_counter() - start,
              'examples': [{'true': chars[int(y)], 'prediction': x['rec_text']}
                           for x, y in zip(result[:8], truth[:8])]}, ensure_ascii=False), flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    fields = ['grid_id', 'sigma', 'resolution', 'source_index', 'source_condition_id',
              'true_id', 'true_char', 'predicted_text', 'confidence']
    run_config = {'model': args.model, 'model_sha256': {name: sha256(model_path / name)
                  for name in ['inference.json', 'inference.pdiparams', 'inference.yml']},
                  'dataset_manifest_sha256': sha256(DATA / 'manifest.json'),
                  'script_sha256': sha256(Path(__file__)), 'batch_size': args.batch_size,
                  'paddle': paddle.__version__, 'paddleocr': paddleocr.__version__,
                  'cpu_threads': 4, 'enable_mkldnn': True}
    config_path = output / 'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != run_config:
            raise ValueError('Cannot resume: model, data, code or environment changed')
    else:
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    started = time.perf_counter()
    for grid in manifest['grids']:
        grid_id = grid['grid_id']
        target = output / f'predictions_{grid_id:02d}.csv'
        if target.exists():
            with target.open(encoding='utf-8-sig', newline='') as stream:
                existing = list(csv.DictReader(stream))
            if len(existing) != len(truth) or any(int(row['grid_id']) != grid_id or
                int(row['source_index']) != int(source_indexes[i]) or
                int(row['true_id']) != int(truth[i]) for i, row in enumerate(existing)):
                raise ValueError(f'Invalid existing grid: {target}')
            print(f'[{grid_id + 1:02d}/25] restored completed grid', flush=True)
            continue
        records = []
        for start in range(0, len(truth), args.batch_size):
            batch = images[grid_id, start:start + args.batch_size]
            inputs = [np.repeat(image[:, :, None], 3, axis=2) for image in batch]
            results = model.predict(inputs, batch_size=args.batch_size)
            if len(results) != len(batch):
                raise ValueError('Paddle returned a different number of predictions')
            for offset, result in enumerate(results):
                i = start + offset
                records.append([grid_id, grid['sigma'], grid['resolution'], int(source_indexes[i]),
                                int(source_ids[i]), int(truth[i]), chars[int(truth[i])],
                                result['rec_text'], float(result['rec_score'])])
        temporary = target.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(records)
        temporary.replace(target)
        correct = sum(record[6] == record[7] for record in records)
        print(f'[{grid_id + 1:02d}/25] sigma={grid["sigma"]} R={grid["resolution"]}: '
              f'{correct}/{len(records)} = {correct / len(records):.2%}', flush=True)

    baseline = list(csv.DictReader(BASELINE.open(encoding='utf-8-sig', newline='')))
    summaries = []
    pairs = []
    with (output / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for grid in manifest['grids']:
            grid_id = grid['grid_id']
            with (output / f'predictions_{grid_id:02d}.csv').open(encoding='utf-8-sig', newline='') as source:
                records = list(csv.DictReader(source))
            if len(records) != 1296:
                raise ValueError(f'Incomplete grid {grid_id}')
            writer.writerows(records)
            errors = sum(r['predicted_text'] != r['true_char'] for r in records)
            outside = sum(r['predicted_text'] not in chars for r in records)
            empty = sum(not r['predicted_text'] for r in records)
            multi = sum(len(r['predicted_text']) > 1 for r in records)
            row = {'grid_id': grid_id, 'sigma': grid['sigma'], 'resolution': grid['resolution'],
                   'samples': len(records), 'errors': errors, 'accuracy': 1 - errors / len(records),
                   'outside_81': outside, 'empty': empty, 'multi_character': multi,
                   'baseline_accuracy': float(baseline[grid_id]['macro_accuracy'])}
            summaries.append(row)
            counts = Counter((r['true_char'], r['predicted_text']) for r in records)
            for p in labels['pairs']:
                pairs.append({'grid_id': grid_id, 'sigma': grid['sigma'], 'resolution': grid['resolution'],
                              'pair_id': p['pair_id'], 'a': p['a'], 'b': p['b'],
                              'a_to_b': counts[p['a'], p['b']], 'b_to_a': counts[p['b'], p['a']]})
    for name, rows in [('summary.csv', summaries), ('pairs.csv', pairs)]:
        with (output / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    info = {'status': 'complete', 'model': args.model, 'model_dir': str(model_path),
            'model_sha256': run_config['model_sha256'],
            'python': platform.python_version(), 'paddle': paddle.__version__, 'paddleocr': paddleocr.__version__,
            'device': 'cpu', 'cpu_threads': 4, 'enable_mkldnn': True, 'batch_size': args.batch_size,
            'dataset_manifest_sha256': sha256(DATA / 'manifest.json'), 'dataset_pixel_sha256': manifest['pixel_sha256'],
            'samples': sum(x['samples'] for x in summaries), 'grids': len(summaries),
            'elapsed_seconds_this_run': time.perf_counter() - started,
            'metric': 'exact raw recognized string equals ground-truth single character',
            'data_scope': 'development calibration derived from val_graph, not the held-out test split',
            'input': 'unchanged 64x64 grayscale pixels repeated into 3 channels; model default preprocessing',
            'training_scope': 'general pretrained recognizer versus custom 81-class baseline; not matched training'}
    (output / 'summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
