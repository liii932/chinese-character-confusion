"""Offline checks for Colab recovery and reporting; no torch/GPU/training used."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
import zipfile

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    return {"sizes": [24, 48, 96, 192], "blocks": 10, "block_draws": 15552,
            "batch_size": 64, "seed": 42,
            "curve_script_sha256": "2b098c465e94fd1909c159c1a64b59745e914c13023e1a23d8a3445e2231272d"}


def history(blocks):
    return [{"block": i, "batch_attempts": i * 243, "validation_split": "val_select",
             "optimizer_updates": i * 243, "amp_skipped_updates_this_block": 0,
             "validation": {"macro_accuracy": .99, "accuracy": .99, "loss": .01 / i}}
            for i in range(1, blocks + 1)]


class SessionNotebookTests(unittest.TestCase):
    def setUp(self):
        self.setup = module("setup_cell", HERE / "session_cells/initialize.py")
        self.train = module("train_cell", HERE / "session_cells/train.py")
        self.report = module("report_cell", HERE / "session_cells/report.py")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "drive/hanzi81"
        self.local = Path(self.tmp.name) / "content"
        self.root.mkdir(parents=True)
        self.local.mkdir()
        self.suite = self.root / "analyses/baseline81_s42_v1/03_learning_curve/steps2430_s42_v1"

    def arm(self, size, blocks):
        write_json(self.suite / "plan.json", plan())
        folder = self.suite / f"n{size:03d}_s42"
        write_json(folder / "history.json", history(blocks))
        write_json(folder / "config.json", {"suite_plan_sha256": sha(self.suite / "plan.json"),
                                            "unique_per_class": size, "blocks": 10, "seed": 42})
        (folder / "best.pt").write_bytes(b"file-presence-only")
        (folder / "last.pt").write_bytes(b"file-presence-only")
        return folder

    def test_disabled_training_runs_in_fresh_namespace_without_drive_or_gpu(self):
        source = (HERE / "session_cells/train.py").read_text(encoding="utf-8")
        with patch("subprocess.Popen") as launch, redirect_stdout(io.StringIO()) as output:
            exec(compile(source, "train-cell", "exec"), {"__name__": "__main__"})
        launch.assert_not_called()
        self.assertIn("续训关闭", output.getvalue())

    def test_read_only_setup_does_not_restore_assets_or_probe_gpu(self):
        with patch.object(self.setup, "setup_mount") as mount, \
                patch.object(self.setup, "setup_restore_bundle") as restore, \
                patch.object(self.setup, "setup_probe") as probe, redirect_stdout(io.StringIO()):
            self.setup.initialize_session(self.root, self.local, False)
        mount.assert_called_once_with(self.root)
        restore.assert_not_called()
        probe.assert_not_called()

    def test_cold_restore_reuses_valid_files_and_rejects_corruption(self):
        contents = {"training/train.py": b"# frozen trainer\n", "data.npz": b"frozen data"}
        manifest = json.dumps({"files": {name: hashlib.sha256(body).hexdigest()
                                         for name, body in contents.items()}}).encode()
        bundle = self.root / "hanzi81_colab_bundle.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("bundle_manifest.json", manifest)
            for name, body in contents.items():
                archive.writestr(name, body)
        self.setup.BUNDLE_SHA256 = sha(bundle)
        self.setup.BUNDLE_MANIFEST_SHA256 = hashlib.sha256(manifest).hexdigest()
        with redirect_stdout(io.StringIO()):
            work = self.setup.setup_restore_bundle(self.root, self.local)
            with patch.object(self.setup.shutil, "copyfile", side_effect=AssertionError("unexpected copy")), \
                    patch.object(self.setup.zipfile.ZipFile, "extractall", side_effect=AssertionError("unexpected extract")):
                self.assertEqual(self.setup.setup_restore_bundle(self.root, self.local), work)
            (work / "data.npz").unlink()
            self.setup.setup_restore_bundle(self.root, self.local)
            self.assertEqual((work / "data.npz").read_bytes(), contents["data.npz"])
            (work / "training/train.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "校验失败"):
                self.setup.setup_restore_bundle(self.root, self.local)
            self.assertEqual((work / "training/train.py").read_bytes(), b"changed")

    def test_restore_frozen_curve_without_upload_or_prepare(self):
        write_json(self.suite / "plan.json", plan())
        (self.suite / "learning_curve.py").write_bytes((HERE / "learning_curve.py").read_bytes())
        with redirect_stdout(io.StringIO()):
            script = self.setup.setup_restore_curve(self.root, self.local)
        self.assertEqual(sha(script), self.setup.CURVE_SHA256)
        (self.suite / "learning_curve.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.setup.setup_restore_curve(self.root, self.local)

    def test_train_calls_one_arm_and_accepts_partial_completion(self):
        self.arm(192, 3)
        (self.suite / "learning_curve.py").write_bytes((HERE / "learning_curve.py").read_bytes())
        work = self.local / "hanzi81-e477bdd4894e"
        data = work / "training/data/baseline81-v2"
        data.mkdir(parents=True)
        for name in ("train.npz", "val_select.npz"):
            (data / name).write_bytes(b"fixture only")
        for name, field in [("training/train.py", "TRAIN_SOURCE_SHA256"),
                            ("training/io_utils.py", "TRAIN_IO_SHA256"),
                            ("training/data/baseline81-v2/manifest.json", "TRAIN_DATA_SHA256")]:
            (work / name).write_bytes(b"fixture frozen source")
            setattr(self.train, field, sha(work / name))
        process = Mock()
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        process.stdout = iter(["saved block 3\n"])
        process.wait.return_value = 0
        with patch.object(self.train.subprocess, "Popen", return_value=process) as launch, \
                redirect_stdout(io.StringIO()) as output:
            self.train.train_session(self.root, self.local, True, 192, 3)
        command = launch.call_args.args[0]
        self.assertEqual(command[-4:], ["--sizes", "192", "--stop-after-blocks", "3"])
        self.assertIn("3/10", output.getvalue())
        self.assertEqual(len(list((self.suite / "session_logs").glob("*.txt"))), 1)

    def test_missing_runtime_has_actionable_error_before_launch(self):
        with patch.object(self.train.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(FileNotFoundError, "第①格"):
                self.train.train_session(self.root, self.local, True, 192, 3)
        launch.assert_not_called()

    def test_prepared_only_and_partial_progress_are_not_final_results(self):
        write_json(self.suite / "plan.json", plan())
        progress, _ = self.report.report_learning_progress(self.suite)
        self.assertFalse(progress["all_complete_by_history_and_file_presence"])
        self.assertIsNone(progress["results"][0]["logged_blocks"])
        self.arm(192, 3)
        progress, _ = self.report.report_learning_progress(self.suite)
        self.assertEqual(progress["results"][-1]["logged_blocks"], 3)
        self.assertFalse(progress["all_complete_by_history_and_file_presence"])
        for size in (24, 48, 96, 192):
            self.arm(size, 10)
        progress, _ = self.report.report_learning_progress(self.suite)
        self.assertTrue(progress["all_complete_by_history_and_file_presence"])
        (self.suite / "n024_s42/last.pt").unlink()
        progress, _ = self.report.report_learning_progress(self.suite)
        self.assertFalse(progress["all_complete_by_history_and_file_presence"])

    def test_report_on_fresh_cpu_session_rejects_incomplete_profile_and_preserves_files(self):
        analyses = self.root / "analyses/baseline81_s42_v1"
        measurement = analyses / "06_profile/20261012T150314Z_396b4588/summary.json"
        write_json(measurement, {"purpose": "runtime_profile_only_not_scientific_results"})
        original = measurement.read_bytes()
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=Mock())
        google.colab = colab
        with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                patch.object(self.report, "report_plots"), redirect_stdout(io.StringIO()):
            snapshot = self.report.inspect_session(self.root, True)
        self.assertEqual(snapshot["task_06_status"], "incomplete_invalid_measurement")
        self.assertTrue(snapshot["existing_reports"]["06_profile_saved_measurement"]["found"])
        self.assertEqual(measurement.read_bytes(), original)
        self.assertFalse(snapshot["learning_curve"]["checkpoint_contents_loaded"])
        self.assertEqual(len(list((analyses / "session_reports").glob("*/session_summary.json"))), 1)

    def test_generated_notebook_is_self_contained_and_backup_is_exact(self):
        generator = module("session_generator", HERE / "make_session_notebook.py")
        original = generator.BACKUP.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), generator.ORIGINAL_SHA256)
        notebook = json.loads(generator.NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook, generator.build(json.loads(original)))
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 3)
        self.assertEqual(len({cell["id"] for cell in notebook["cells"]}), len(notebook["cells"]))
        for cell in code:
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            compile("".join(cell["source"]), cell["id"], "exec")
        # The fixed bundle and frozen scripts are compatibility inputs, never rebuilt here.
        bundle = HERE / "dist/hanzi81_colab_bundle.zip"
        self.assertEqual(sha(bundle), self.setup.BUNDLE_SHA256)
        with zipfile.ZipFile(bundle) as archive:
            self.assertEqual(hashlib.sha256(archive.read("bundle_manifest.json")).hexdigest(),
                             self.setup.BUNDLE_MANIFEST_SHA256)
        self.assertEqual(sha(HERE / "learning_curve.py"), self.train.TRAIN_CURVE_SHA256)
        self.assertEqual(sha(HERE.parent / "training/train.py"), self.train.TRAIN_SOURCE_SHA256)
        self.assertEqual(sha(HERE.parent / "training/io_utils.py"), self.train.TRAIN_IO_SHA256)


if __name__ == "__main__":
    unittest.main()
