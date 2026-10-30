"""Offline checks of notebook setup and command routing; no GPU or model training."""
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode('utf-8')


class Round2NotebookTests(unittest.TestCase):
    def setUp(self):
        self.setup = module('round2_setup', HERE / 'round2_cells/prepare.py')
        self.train = module('round2_train', HERE / 'round2_cells/train.py')
        self.train.DATA_ARCHIVE_SHA256 = 'fixture-archive'
        self.train.SOURCE_ID = 'fixture-source'
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'drive/project'
        self.local = Path(self.temporary.name) / 'content'
        self.root.mkdir(parents=True)
        self.local.mkdir()

    def archive(self):
        # Metadata-only stand-ins test ZIP selection and hashes, never pixel/model behavior.
        payload = {name: b'fixture-not-a-training-array' for name in
                   ('train.npz', 'val_graph.npz', 'val_select.npz', 'test.npz')}
        payload['labels.json'] = encode({'characters': [{'class_id': i} for i in range(1140)]})
        payload['conditions.json'] = encode({})
        manifest = {'splits': {key: {'images': n} for key, n in
                    {'train': 218880, 'val_graph': 109440, 'val_select': 109440, 'test': 218880}.items()},
                    'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()}}
        payload['manifest.json'] = encode(manifest)
        payload['audit.json'] = encode({'automatic_checks_passed': True, 'unique_images': 656640})
        target = self.root / self.setup.ARCHIVE_NAME
        with zipfile.ZipFile(target, 'w') as bundle:
            for name, raw in payload.items():
                bundle.writestr('round2-1140-v1/' + name, raw)
            bundle.writestr('../should-not-be-extracted.py', 'raise RuntimeError()')
        return self.setup.file_sha256(target), hashlib.sha256(payload['manifest.json']).hexdigest()

    def test_prepare_only_required_data_and_freeze_sources(self):
        archive_hash, manifest_hash = self.archive()
        sources = {'training/__init__.py': '', 'training/train.py': 'epochs = 10\n'}
        with redirect_stdout(io.StringIO()):
            work, data, source_id = self.setup.prepare_round2(
                self.root, self.local, archive_hash, manifest_hash, sources)
        self.assertEqual({p.name for p in data.iterdir()}, set(self.setup.DATA_MEMBERS))
        self.assertFalse((self.local / 'should-not-be-extracted.py').exists())
        self.assertEqual((work / 'training/train.py').read_text(), sources['training/train.py'])
        snapshot = self.root / 'sources' / source_id / 'training/train.py'
        self.assertEqual(snapshot.read_text(), sources['training/train.py'])
        snapshot.write_text('changed = True\n')
        with self.assertRaisesRegex(ValueError, '源码快照已变化'), redirect_stdout(io.StringIO()):
            self.setup.prepare_round2(self.root, self.local, archive_hash, manifest_hash, sources)

    def test_corrupt_upload_rejected_before_extraction(self):
        archive_hash, manifest_hash = self.archive()
        with (self.root / self.setup.ARCHIVE_NAME).open('ab') as stream:
            stream.write(b'changed-upload')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.setup.prepare_round2(self.root, self.local, archive_hash, manifest_hash, {})
        self.assertFalse(any(self.local.rglob('train.npz')))

    def test_partial_then_resume_keeps_total_ten_epochs(self):
        def launch(command, **kwargs):
            if 'training.train' not in command:
                return types.SimpleNamespace(stdout='offline-test-fixture\n')
            folder = Path(command[command.index('--run-dir') + 1])
            folder.mkdir(parents=True, exist_ok=True)
            for name in ('best.pt', 'last.pt'):
                (folder / name).write_bytes(b'file-presence-fixture')
            (folder / 'config.json').write_text('{}')
            (folder / 'history.json').write_text('[]')
            return types.SimpleNamespace(returncode=0)

        with patch.object(self.train.subprocess, 'run', side_effect=launch) as run, redirect_stdout(io.StringIO()):
            self.train.run_round2_training(self.local, self.local / 'data', self.root,
                                          self.train.RUN_NAME, 'baseline', 10, stop_after=2)
            self.train.run_round2_training(self.local, self.local / 'data', self.root,
                                          self.train.RUN_NAME, 'baseline', 10)
        calls = [c.args[0] for c in run.call_args_list if 'training.train' in c.args[0]]
        self.assertEqual(len(calls), 2)
        for command in calls:
            self.assertEqual(command[command.index('--epochs') + 1], '10')
            self.assertNotIn('--per-class', command)
        self.assertEqual(calls[0][calls[0].index('--stop-after') + 1], '2')
        self.assertNotIn('--resume', calls[0])
        self.assertIn('--resume', calls[1])
        self.assertNotIn('--stop-after', calls[1])

    def test_complete_gate_and_checkpoint_report_reuse(self):
        folder = self.root / 'runs' / self.train.RUN_NAME
        folder.mkdir(parents=True)
        history = folder / 'history.json'
        history.write_text(json.dumps([{'epoch': 1}, {'epoch': 2}]))
        (folder / 'best.pt').write_bytes(b'checkpoint-fixture-1')
        with patch.object(self.train.subprocess, 'run') as run, redirect_stdout(io.StringIO()):
            self.assertEqual(self.train.evaluate_completed_round2(self.local, self.local, folder), [])
            run.assert_not_called()
        history.write_text(json.dumps([{'epoch': i} for i in range(1, 11)]))

        def launch(command, **kwargs):
            split = command[command.index('--split') + 1]
            destination = folder / f'evaluation_{split}'
            destination.mkdir(exist_ok=True)
            for name in ('metrics.json', 'confusion.npy', 'pairs.json', 'predictions.csv'):
                (destination / name).write_bytes(b'presence-only-fixture')

        with patch.object(self.train.subprocess, 'run', side_effect=launch) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(self.train.evaluate_completed_round2(self.local, self.local, folder), ['val_select', 'val_graph'])
            self.train.evaluate_completed_round2(self.local, self.local, folder)
            self.assertEqual(run.call_count, 2)
            splits = [c.args[0][c.args[0].index('--split') + 1] for c in run.call_args_list]
            self.assertEqual(splits, ['val_select', 'val_graph'])
            self.assertTrue(all('--final-test' not in c.args[0] for c in run.call_args_list))
            (folder / 'best.pt').write_bytes(b'checkpoint-fixture-2')
            self.train.evaluate_completed_round2(self.local, self.local, folder)
            self.assertEqual(run.call_count, 4)

    def test_generated_notebook_frozen_code_and_clean_cells(self):
        notebook = json.loads((HERE / 'hanzi1140_round2_e10.ipynb').read_text(encoding='utf-8'))
        protocol = notebook['metadata']['round2_protocol']
        self.assertEqual((protocol['classes'], protocol['epochs'], protocol['batch_size']), (1140, 10, 64))
        cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 6)
        for cell in cells:
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])
            compile(cell['source'], cell['id'], 'exec')
        assignments = ast.parse(cells[1]['source']).body
        embedded = next(ast.literal_eval(node.value) for node in assignments if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'TRAINING_SOURCES' for t in node.targets))
        for name, source in embedded.items():
            self.assertEqual(source, (HERE.parent / name).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
