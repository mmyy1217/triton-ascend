import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from triton.backends.ascend import scope_profile


class ScopeProfileRunnerTest(unittest.TestCase):
    def test_apply_one_records_compile_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory)
            report_dir = dump / "kernel"
            report_dir.mkdir()
            with patch.object(scope_profile, "_compile", side_effect=RuntimeError("compile failed")):
                result = scope_profile._apply_one(
                    object(), (), (1, ), {}, dump, report_dir, 4,
                    None, None, 25, 100)

            self.assertEqual(result["failed_phase"], "compile")
            self.assertEqual(result["error"]["message"], "compile failed")
            saved = json.loads((report_dir / "plans" / "4" / "result.json").read_text())
            self.assertEqual(saved["failed_phase"], "compile")

    def test_apply_one_records_runtime_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory)
            report_dir = dump / "kernel"
            report_dir.mkdir()
            metadata = type("Metadata", (), {"scope_profile_report_dir": str(report_dir)})()
            compiled = type("Compiled", (), {"metadata": metadata})()
            with patch.object(scope_profile, "_compile", return_value=(compiled, 1.0)), \
                 patch.object(scope_profile, "_run_correctness", return_value=None), \
                 patch.object(scope_profile, "_benchmark", side_effect=RuntimeError("launch failed")):
                result = scope_profile._apply_one(
                    object(), (), (1, ), {}, dump, report_dir, 5,
                    None, lambda: None, 25, 100)

            self.assertEqual(result["failed_phase"], "runtime")
            self.assertEqual(result["error"]["message"], "launch failed")
            saved = json.loads((report_dir / "plans" / "5" / "result.json").read_text())
            self.assertEqual(saved["failed_phase"], "runtime")

    def test_tune_keeps_running_after_one_plan_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory)
            report_dir = dump / "kernel"
            report_dir.mkdir()
            manifest = {
                "search_model": "single_contiguous_scope_v1",
                "plans": [{"id": 0}, {"id": 1}, {"id": 2}],
            }
            results = {
                0: {"status": "valid", "timing": {"median_ms": 2.0}},
                1: {"status": "invalid", "failed_phase": "compile"},
                2: {"status": "valid", "timing": {"median_ms": 1.0}},
            }

            with patch.object(scope_profile, "_discover", return_value=(report_dir, manifest)), \
                 patch.object(scope_profile, "_apply_one", side_effect=lambda *args: results[args[6]]):
                output = scope_profile.run(object(), grid=(1, ), mode="tune", dump=dump, check=lambda: None)

            summary = output["summary"]
            self.assertTrue(summary["search_complete"])
            self.assertEqual(summary["attempted_count"], 3)
            self.assertEqual(summary["failure_counts"]["compile"], 1)
            self.assertEqual(summary["best_plan_id"], 2)
            self.assertEqual(summary["speedup_over_empty"], 2.0)
            self.assertEqual(json.loads((report_dir / "summary.json").read_text()), summary)

    def test_apply_uses_existing_manifest_without_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory)
            expected = {"status": "valid"}
            with patch.object(scope_profile, "_discover") as discover, \
                 patch.object(scope_profile, "_apply_existing", return_value=(dump, expected)):
                output = scope_profile.run(object(), grid=(1, ), mode="apply", dump=dump, plan_id=7)
            discover.assert_not_called()
            self.assertEqual(output["plan_id"], 7)
            self.assertEqual(output["result"], expected)

    def test_tune_requires_correctness_callback(self):
        with tempfile.TemporaryDirectory() as directory, \
             self.assertRaisesRegex(ValueError, "correctness callback"):
            scope_profile.run(object(), grid=(1, ), mode="tune", dump=directory)

    def test_apply_rejects_negative_plan_id(self):
        with tempfile.TemporaryDirectory() as directory, \
             self.assertRaisesRegex(ValueError, "non-negative"):
            scope_profile.run(object(), grid=(1, ), mode="apply", dump=directory, plan_id=-1)


if __name__ == "__main__":
    unittest.main()
