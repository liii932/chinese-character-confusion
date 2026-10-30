"""Offline verification with the user's real saved timing output; never trains."""
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch, Mock
from contextlib import redirect_stdout

from colab.analyze_reported_profile import load_reported_profile
from colab.session_cells.efficiency import analyze_efficiency
from colab.session_cells.report import report_efficiency
from colab.make_session_notebook import NOTEBOOK


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.profile, _ = load_reported_profile()

    def test_real_report_ratios_memory_storage_and_scaling(self):
        result = analyze_efficiency(self.profile)
        self.assertEqual(result["status"], "complete_saved_measurement_analysis")
        self.assertAlmostEqual(result["steady"]["epoch_wall_seconds_including_saves"], 32.2)
        self.assertAlmostEqual(result["phase_percent_of_steady_wall"]["checkpoint_save"], 1.8320717096274994)
        self.assertAlmostEqual(sum(result["phase_percent_of_steady_wall"].values()), 100)
        self.assertAlmostEqual(result["memory"]["allocated_percent_of_capacity"], 5.475747109214888)
        self.assertAlmostEqual(result["memory"]["reserved_percent_of_capacity"], 10.367011311676718)
        self.assertAlmostEqual(result["storage"]["best_plus_last_mib"], 171.24801445007324)
        full = result["planning_estimates_not_measurements"]["hypothetical_1140_classes"]
        self.assertEqual((full["train_images"], full["validation_images"]), (218880, 109440))
        self.assertAlmostEqual(full["train_and_validation_hours_for_40_epochs"], 4.859659185876537)
        self.assertIsNone(result["not_measured"]["compute_units_consumed"])
        self.assertIsNone(result["memory"]["gpu_utilization_percent"])

    def test_rejects_missing_epoch_wrong_throughput_memory_or_save_total(self):
        examples = []
        item = copy.deepcopy(self.profile)
        item["epochs"].pop()
        examples.append(item)
        item = copy.deepcopy(self.profile)
        item["epochs"][0]["train_images_per_second"] *= 2
        examples.append(item)
        item = copy.deepcopy(self.profile)
        item["epochs"][0]["peak_allocated_gib"] = 12
        examples.append(item)
        item = copy.deepcopy(self.profile)
        item["checkpoint_saves"][0]["seconds"] += 1
        examples.append(item)
        for item in examples:
            with self.subTest(example=examples.index(item)):
                with self.assertRaises(ValueError):
                    analyze_efficiency(item)

    def test_environment_mismatch_and_missing_measurement_stay_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertEqual(report_efficiency(None, {}, run)["status"], "incomplete_missing_measurement")
            source = run / "summary.json"
            source.write_text(json.dumps(self.profile), encoding="utf-8")
            (run / "config.json").write_text(json.dumps({"environment": {"gpu": "other"}}), encoding="utf-8")
            self.assertEqual(report_efficiency(source, {}, run)["status"], "incomplete_invalid_measurement")

    def test_embedded_third_cell_runs_alone_and_writes_analysis_on_cpu(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if c["id"] == "report-code")
        source = "".join(cell["source"])
        self.assertNotIn("from colab.session_cells.efficiency import", source)
        namespace = {"__name__": "offline_cell_check"}
        exec(compile(source, "embedded-third-cell", "exec"), namespace)
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=Mock())
        google.colab = colab
        namespace["report_plots"] = Mock()
        namespace["plot_efficiency"] = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "analyses/baseline81_s42_v1/06_profile/20261012T150314Z_396b4588/summary.json"
            profile_path.parent.mkdir(parents=True)
            profile_path.write_text(json.dumps(self.profile), encoding="utf-8")
            before = profile_path.read_bytes()
            with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                    patch("subprocess.Popen", side_effect=AssertionError("must not start a process")), \
                    redirect_stdout(io.StringIO()):
                snapshot = namespace["inspect_session"](root, True)
            self.assertEqual(snapshot["task_06_status"], "complete_saved_measurement_analysis")
            reports = list(root.glob("analyses/baseline81_s42_v1/session_reports/*/efficiency_resources.json"))
            self.assertEqual(len(reports), 1)
            saved = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "complete_saved_measurement_analysis")
            self.assertFalse(saved["source"]["new_gpu_measurement"])
            self.assertEqual(profile_path.read_bytes(), before)
            self.assertFalse(snapshot["learning_curve"]["all_complete_by_history_and_file_presence"])


if __name__ == "__main__":
    unittest.main()
