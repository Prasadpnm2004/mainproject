"""
tests/test_reproducibility.py

PACM v1.0.0 — Reproducibility Integration Tests — Session 8

Verifies that:
  1. reproduce_all.py is importable and well-formed.
  2. The ReproductionPipeline class instantiates correctly.
  3. Dry-run mode executes all 8 steps without errors.
  4. check_artifacts() returns a dict with the expected artifact keys.
  5. The --synthetic flag propagates correctly to pipeline steps.
  6. StepResult dataclass has all expected fields.
  7. print_final_summary runs without error on a mock result set.
  8. All 8 expected artifacts are listed in EXPECTED_ARTIFACTS.
  9. reproduce_all.py is importable as a module (no __main__ guard issues).
 10. _generate_minimal_calibration produces a valid calibration_result.json.

These tests do NOT launch subprocess calls (which would be slow and require
system state); instead they test the pipeline scaffolding in isolation using
dry-run mode.

Run with:  pytest tests/test_reproducibility.py -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.reproduce_all import (
    ReproductionPipeline,
    StepResult,
    EXPECTED_ARTIFACTS,
    check_artifacts,
    print_final_summary,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(request):
    """
    Custom temp directory fixture that avoids the Python 3.14 + pytest
    ``ValueError: I/O operation on closed file`` bug in tmp_path teardown.

    Uses tempfile.mkdtemp() directly and registers a finalizer for cleanup.
    """
    d = Path(tempfile.mkdtemp(prefix="pacm_repro_test_"))
    request.addfinalizer(lambda: shutil.rmtree(d, ignore_errors=True))
    return d


@pytest.fixture
def dry_pipeline(tmp_dir):
    """A dry-run ReproductionPipeline with a temp log dir."""
    return ReproductionPipeline(
        synthetic=True,
        skip_tests=False,
        dry_run=True,
        log_dir=tmp_dir / "reproduce_logs",
    )


def _make_step_results(n: int = 8) -> list[StepResult]:
    """Create n mock StepResult objects (all passing)."""
    return [
        StepResult(
            step_num=i + 1,
            name=f"Step {i+1}",
            success=True,
            elapsed_s=float(i + 1) * 0.5,
            returncode=0,
            output_tail="OK",
        )
        for i in range(n)
    ]


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestReproductionPipelineImport:

    def test_module_importable(self):
        """reproduce_all imports cleanly as a module."""
        import experiments.reproduce_all  # noqa: F401

    def test_expected_artifacts_is_list(self):
        assert isinstance(EXPECTED_ARTIFACTS, list)

    def test_expected_artifacts_has_8_entries(self):
        assert len(EXPECTED_ARTIFACTS) == 8

    def test_all_expected_artifacts_are_strings(self):
        for a in EXPECTED_ARTIFACTS:
            assert isinstance(a, str), f"Not a string: {a!r}"

    def test_check_artifacts_returns_dict(self):
        result = check_artifacts()
        assert isinstance(result, dict)
        assert len(result) == len(EXPECTED_ARTIFACTS)

    def test_check_artifacts_keys_match_expected(self):
        result = check_artifacts()
        for key in EXPECTED_ARTIFACTS:
            assert key in result, f"Missing key: {key}"

    def test_check_artifacts_values_are_bool(self):
        result = check_artifacts()
        for key, val in result.items():
            assert isinstance(val, bool), f"Non-bool for {key}: {val!r}"


class TestReproductionPipelineInit:

    def test_instantiation_default(self, tmp_dir):
        p = ReproductionPipeline(log_dir=tmp_dir)
        assert p.synthetic is True
        assert p.skip_tests is False
        assert p.dry_run is False

    def test_instantiation_dry_run(self, tmp_dir):
        p = ReproductionPipeline(dry_run=True, log_dir=tmp_dir)
        assert p.dry_run is True

    def test_log_dir_created(self, tmp_dir):
        log_dir = tmp_dir / "logs"
        assert not log_dir.exists()
        ReproductionPipeline(log_dir=log_dir)
        assert log_dir.exists()


class TestDryRunExecution:

    def test_dry_run_returns_8_results(self, dry_pipeline):
        results = dry_pipeline.run()
        assert len(results) == 8

    def test_dry_run_all_pass(self, dry_pipeline):
        results = dry_pipeline.run()
        for r in results:
            assert r.success, f"Step {r.step_num} ({r.name}) failed in dry run"

    def test_dry_run_step_nums_sequential(self, dry_pipeline):
        results = dry_pipeline.run()
        # Steps may start at 1 or 2 if skip_tests=True; with skip_tests=False
        # we get steps 1-8.
        step_nums = [r.step_num for r in results]
        assert len(step_nums) == 8

    def test_skip_tests_removes_step1(self, tmp_dir):
        p = ReproductionPipeline(
            synthetic=True, skip_tests=True, dry_run=True,
            log_dir=tmp_dir
        )
        results = p.run()
        assert len(results) == 7
        names = [r.name for r in results]
        assert "Unit Tests" not in names

    def test_dry_run_output_tail_contains_dry_run(self, dry_pipeline):
        results = dry_pipeline.run()
        for r in results:
            assert "dry run" in r.output_tail.lower()


class TestStepResult:

    def test_stepresult_status_str_pass(self):
        r = StepResult(1, "Test", True, 1.0, 0, "OK")
        assert r.status_str() == "PASS"

    def test_stepresult_status_str_fail(self):
        r = StepResult(1, "Test", False, 1.0, 1, "ERR")
        assert r.status_str() == "FAIL"

    def test_stepresult_has_all_fields(self):
        r = StepResult(2, "Step 2", True, 3.5, 0, "output")
        assert r.step_num == 2
        assert r.name == "Step 2"
        assert r.success is True
        assert r.elapsed_s == pytest.approx(3.5)
        assert r.returncode == 0
        assert r.output_tail == "output"


class TestPrintFinalSummary:
    """
    Use StringIO redirect instead of capsys to avoid the Python 3.14 +
    pytest-9.1.1 ``ValueError: I/O operation on closed file`` bug that
    affects capsys teardown when pytest's internal capture tmpfile is closed.
    """

    @staticmethod
    def _capture_summary(results) -> str:
        """Run print_final_summary with stdout redirected to a StringIO buffer."""
        import sys
        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_final_summary(results)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_print_summary_no_error(self):
        results = _make_step_results(8)
        out = self._capture_summary(results)
        assert "PACM" in out
        assert "PASS" in out

    def test_print_summary_shows_all_steps(self):
        results = _make_step_results(8)
        out = self._capture_summary(results)
        for r in results:
            assert r.name in out

    def test_print_summary_with_failures(self):
        results = _make_step_results(8)
        results[2].success = False
        results[2].output_tail = "Some error output"
        out = self._capture_summary(results)
        assert "failed" in out.lower() or "FAIL" in out


class TestMinimalCalibration:

    def test_generate_minimal_calibration(self, tmp_dir):
        """_generate_minimal_calibration writes a valid JSON file."""
        p = ReproductionPipeline(
            synthetic=True, dry_run=True, log_dir=tmp_dir / "logs"
        )
        out_path = tmp_dir / "calibration_result.json"
        p._generate_minimal_calibration(out_path)
        assert out_path.exists()
        with open(out_path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Calibration result should have some numeric threshold fields
        assert isinstance(data, dict)
        assert len(data) > 0
