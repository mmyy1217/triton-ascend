import json
from pathlib import Path
from unittest.mock import patch

import pytest

from triton.backends.ascend import scope_profile


def test_tune_keeps_running_after_one_plan_fails(tmp_path):
    report_dir = tmp_path / "kernel"
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
        output = scope_profile.run(object(), grid=(1, ), mode="tune", dump=tmp_path, check=lambda: None)

    summary = output["summary"]
    assert summary["search_complete"]
    assert summary["attempted_count"] == 3
    assert summary["failure_counts"]["compile"] == 1
    assert summary["best_plan_id"] == 2
    assert summary["speedup_over_empty"] == 2.0
    assert json.loads((report_dir / "summary.json").read_text()) == summary


def test_apply_uses_existing_manifest_without_discovery(tmp_path):
    expected = {"status": "valid"}
    with patch.object(scope_profile, "_discover") as discover, \
         patch.object(scope_profile, "_apply_existing", return_value=(Path(tmp_path), expected)):
        output = scope_profile.run(object(), grid=(1, ), mode="apply", dump=tmp_path, plan_id=7)
    discover.assert_not_called()
    assert output["plan_id"] == 7
    assert output["result"] == expected


def test_tune_requires_correctness_callback(tmp_path):
    with pytest.raises(ValueError, match="correctness callback"):
        scope_profile.run(object(), grid=(1, ), mode="tune", dump=tmp_path)


def test_apply_rejects_negative_plan_id(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        scope_profile.run(object(), grid=(1, ), mode="apply", dump=tmp_path, plan_id=-1)
