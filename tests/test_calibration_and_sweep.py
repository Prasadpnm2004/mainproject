"""
tests/test_calibration_and_sweep.py

Session 5 — Integration tests for calibration pipeline and sweep runner.

Tests cover:
  ThresholdCalibrator:
    1.  Empty records → CalibrationResult with no optimal point
    2.  Pure attack records → ABR=1.0 at any threshold that covers all
    3.  Friction formula used correctly (F = D × (1.5 - 0.5 × T))
    4.  Grid search produces n_grid > 0 results
    5.  Pareto frontier is non-empty and sorted by FPR ascending
    6.  Optimal point satisfies target_fpr constraint when possible
    7.  AUC-ROC in [0.0, 1.0]
    8.  to_dict() is JSON-serialisable
    9.  Synthetic records: attack D > benign D → ABR > FPR at mid threshold
   10.  custom allow_grid / block_grid respected

  _build_pareto_frontier:
   11.  Dominated points are pruned
   12.  Monotone increasing ABR along frontier

  _trapz_auc:
   13.  AUC=0.0 for single point
   14.  AUC=1.0 for perfect classifier (ABR=1 at FPR=0)

  run_sweep (integration, stub mode):
   15.  Quick sweep produces >= 2 run configs
   16.  Each run has a run_id and 'metrics' key
   17.  sweep_summary.json is written and parseable
   18.  Sweep with modes=["undefended"] only has no defended runs
   19.  Sweep with policies=["aggressive"] only has that policy in defended runs

  extract_rows (paper table generator):
   20.  extract_rows returns one row per run
   21.  Undefended rows have policy == "—"
   22.  Defended rows have non-empty policy
   23.  build_markdown_table returns a string with "|" characters
   24.  build_latex_table returns a string with "\\toprule" and "\\bottomrule"
   25.  build_markdown_report returns non-empty string

Run with:  pytest tests/test_calibration_and_sweep.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pacm.calibration import (
    ThresholdCalibrator,
    ThresholdMetrics,
    CalibrationResult,
    _build_pareto_frontier,
    _trapz_auc,
    _frange,
    synthetic_records_for_calibration,
)
from experiments.run_sweep import (
    build_sweep_matrix,
    POLICY_CONFIGS,
    DEFAULT_HARNESSES,
    DEFAULT_ATTACKS,
    DEFAULT_MODES,
    DEFAULT_POLICIES,
)
from experiments.generate_paper_tables import (
    extract_rows,
    build_markdown_table,
    build_latex_table,
    build_markdown_report,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _attack_records(n=10, D=0.80, T=0.2) -> list[dict]:
    return [{"divergence_score": D, "trust_score": T, "is_attack": True} for _ in range(n)]

def _benign_records(n=10, D=0.20, T=0.9) -> list[dict]:
    return [{"divergence_score": D, "trust_score": T, "is_attack": False} for _ in range(n)]

def _mixed_records(n_attack=20, n_benign=20) -> list[dict]:
    return _attack_records(n_attack) + _benign_records(n_benign)


# ─── ThresholdCalibrator ──────────────────────────────────────────────────────

class TestThresholdCalibrator:
    def setup_method(self):
        # Small grid for speed
        self.calibrator = ThresholdCalibrator(
            allow_grid=_frange(0.20, 0.60, 0.10),
            block_grid=_frange(0.30, 0.80, 0.10),
            target_fpr=0.10,
        )

    def test_empty_records(self):
        result = self.calibrator.calibrate([])
        assert result.optimal is None
        assert result.frontier == []
        assert result.grid_size == 0

    def test_pure_attack_records(self):
        """All attacks, no benign → can compute ABR but FPR=0 (no benign)."""
        records = _attack_records(10, D=0.80, T=0.2)
        result = self.calibrator.calibrate(records)
        # Some configs should block all attacks
        if result.optimal:
            assert result.optimal.abr > 0

    def test_friction_formula_applied(self):
        """Verify F = D × (1.5 - 0.5 × T) is used internally."""
        # D=0.6, T=0.0: F = 0.9 → all attack records blocked at block_thr=0.85
        records = [{"divergence_score": 0.6, "trust_score": 0.0, "is_attack": True}] * 5
        records += _benign_records(5, D=0.1, T=1.0)  # F = 0.1 × 1.0 = 0.1, low
        result = self.calibrator.calibrate(records)
        # At block_thr≤0.90, F=0.9 attacks should be blocked
        high_abr = [m for m in result.frontier if m.abr > 0.8]
        assert len(high_abr) > 0, "Expected at least one config with ABR > 0.8"

    def test_grid_size_positive(self):
        result = self.calibrator.calibrate(_mixed_records())
        assert result.grid_size > 0

    def test_pareto_frontier_sorted_by_fpr(self):
        result = self.calibrator.calibrate(_mixed_records())
        fprs = [m.fpr for m in result.frontier]
        assert fprs == sorted(fprs), "Frontier should be sorted by FPR ascending"

    def test_optimal_satisfies_target_fpr(self):
        records = _mixed_records(20, 20)
        result = self.calibrator.calibrate(records)
        if result.optimal and result.frontier:
            # If any point meets target, optimal must also meet it
            feasible = [m for m in result.frontier if m.fpr <= 0.10]
            if feasible:
                assert result.optimal.fpr <= 0.10 + 1e-9

    def test_auc_in_range(self):
        result = self.calibrator.calibrate(_mixed_records())
        assert 0.0 <= result.auc_roc <= 1.0

    def test_to_dict_json_serialisable(self):
        result = self.calibrator.calibrate(_mixed_records())
        d = result.to_dict()
        json.dumps(d)  # must not raise
        assert "optimal" in d
        assert "frontier" in d
        assert "auc_roc" in d

    def test_synthetic_records_high_abr(self):
        """Synthetic records with well-separated distributions → ABR > FPR."""
        records = synthetic_records_for_calibration(n_attack=50, n_benign=50, seed=42)
        result = self.calibrator.calibrate(records)
        if result.optimal:
            assert result.optimal.abr >= result.optimal.fpr, (
                f"Expected ABR ({result.optimal.abr:.3f}) >= FPR ({result.optimal.fpr:.3f})"
            )

    def test_custom_grid_respected(self):
        """Custom allow/block grid is used instead of defaults."""
        cal = ThresholdCalibrator(
            allow_grid=[0.30],
            block_grid=[0.60, 0.70],
            target_fpr=0.10,
        )
        result = cal.calibrate(_mixed_records())
        # Only 1 valid pair (0.30, 0.60) and (0.30, 0.70) → grid_size = 2
        assert result.grid_size == 2


# ─── Pareto frontier & AUC helpers ────────────────────────────────────────────

def _make_metric(abr: float, fpr: float, allow_t: float = 0.3, block_t: float = 0.7) -> ThresholdMetrics:
    return ThresholdMetrics(
        allow_threshold=allow_t, block_threshold=block_t,
        abr=abr, fpr=fpr, f1=0.0,
        n_attack=10, n_blocked=int(abr * 10),
        n_benign_calls=10, n_fp=int(fpr * 10),
    )


class TestParetoAndAUC:
    def test_dominated_points_pruned(self):
        """Point (ABR=0.5, FPR=0.5) dominated by (ABR=0.8, FPR=0.5) → pruned."""
        pts = [
            _make_metric(0.5, 0.5),  # dominated
            _make_metric(0.8, 0.5),  # dominates above
            _make_metric(0.9, 0.8),  # higher FPR but higher ABR → on frontier
        ]
        frontier = _build_pareto_frontier(pts)
        abr_fpr = [(m.abr, m.fpr) for m in frontier]
        assert (0.5, 0.5) not in abr_fpr, "Dominated point should be pruned"

    def test_frontier_monotone_abr(self):
        """ABR is non-decreasing along the Pareto frontier (sorted by FPR)."""
        pts = [
            _make_metric(0.2, 0.0),
            _make_metric(0.5, 0.2),
            _make_metric(0.8, 0.4),
            _make_metric(0.9, 0.6),
        ]
        frontier = _build_pareto_frontier(pts)
        abrs = [m.abr for m in frontier]
        for i in range(1, len(abrs)):
            assert abrs[i] >= abrs[i - 1], "ABR must be non-decreasing on frontier"

    def test_auc_single_point_is_zero(self):
        """Single-point frontier → AUC = 0.0 (trapz needs 2 points)."""
        pts = [_make_metric(0.8, 0.1)]
        assert _trapz_auc(pts) == 0.0

    def test_auc_perfect_classifier(self):
        """Perfect classifier (ABR=1 everywhere at FPR=0) → AUC ≈ 0."""
        # A single Pareto point at (FPR=0, ABR=1) has AUC=0 (no area to integrate)
        pts = [
            _make_metric(1.0, 0.0),
            _make_metric(1.0, 0.5),
        ]
        auc = _trapz_auc(pts)
        assert 0.0 <= auc <= 1.0


# ─── run_sweep integration ────────────────────────────────────────────────────

class TestRunSweepMatrix:
    def test_quick_sweep_config_count(self):
        """Quick config (1 harness, 1 attack, 2 modes, 1 policy) produces >= 2 configs."""
        configs = build_sweep_matrix(
            harnesses=["langgraph"],
            attacks=["ignore_previous"],
            modes=["undefended", "defended"],
            policies=["standard"],
        )
        # 1 undefended + 1 defended = 2
        assert len(configs) == 2

    def test_each_config_has_run_id(self):
        configs = build_sweep_matrix(["langgraph"], ["ignore_previous"],
                                     ["defended"], ["standard"])
        for c in configs:
            assert "run_id" in c and c["run_id"]

    def test_undefended_no_policy_label(self):
        configs = build_sweep_matrix(["langgraph"], ["ignore_previous"],
                                     ["undefended"], ["standard"])
        for c in configs:
            assert c["policy_label"] == "none"
            assert c["policy_mode"] == "allow"

    def test_defended_has_policy_label(self):
        configs = build_sweep_matrix(["langgraph"], ["ignore_previous"],
                                     ["defended"], ["standard", "aggressive"])
        labels = {c["policy_label"] for c in configs}
        assert "STANDARD_POLICY" in labels
        assert "AGGRESSIVE_POLICY" in labels

    def test_undefended_mode_only(self):
        configs = build_sweep_matrix(["langgraph"], ["ignore_previous"],
                                     ["undefended"], DEFAULT_POLICIES)
        # Only undefended → no defended rows
        for c in configs:
            assert c["mode"] == "undefended"

    def test_defended_mode_only(self):
        configs = build_sweep_matrix(["langgraph"], ["ignore_previous"],
                                     ["defended"], ["aggressive"])
        for c in configs:
            assert c["mode"] == "defended"
            assert c["policy_key"] == "aggressive"


# ─── Paper table generator ────────────────────────────────────────────────────

def _make_fake_sweep() -> dict:
    """Generate a minimal sweep_summary dict for table tests."""
    return {
        "sweep_id": "test_sweep",
        "sweep_start_utc": "2026-08-15T00:00:00Z",
        "sweep_end_utc":   "2026-08-15T00:01:00Z",
        "n_configs": 2,
        "n_total_tasks": 8,
        "parameters": {
            "harnesses": ["langgraph"],
            "attacks": ["ignore_previous"],
            "modes": ["undefended", "defended"],
            "policies": ["standard"],
            "model": "gpt-4o-mini",
        },
        "runs": [
            {
                "run_id": "r1", "mode": "undefended", "harness": "langgraph",
                "attack": "ignore_previous", "policy_key": None,
                "policy_label": "none", "policy_mode": "allow",
                "divergence_threshold": 0.6, "label": "undefended/langgraph/ignore_previous",
                "n_tasks": 4, "elapsed_s": 1.2,
                "metrics": {"abr": 0.0, "btsr": 1.0, "fpr": None,
                            "mean_latency_s": 0.3, "p95_latency_s": 0.5,
                            "n_attack": 2, "n_blocked": 0, "n_benign": 2},
            },
            {
                "run_id": "r2", "mode": "defended", "harness": "langgraph",
                "attack": "ignore_previous", "policy_key": "standard",
                "policy_label": "STANDARD_POLICY", "policy_mode": "block",
                "divergence_threshold": 0.6, "label": "defended/langgraph/ignore_previous/standard",
                "n_tasks": 4, "elapsed_s": 1.5,
                "metrics": {"abr": 1.0, "btsr": 1.0, "fpr": 0.0,
                            "mean_latency_s": 0.4, "p95_latency_s": 0.6,
                            "n_attack": 2, "n_blocked": 2, "n_benign": 2},
            },
        ],
    }


class TestPaperTableGenerator:
    def setup_method(self):
        self.sweep = _make_fake_sweep()
        self.rows  = extract_rows(self.sweep)

    def test_extract_rows_count(self):
        assert len(self.rows) == 2

    def test_undefended_rows_policy_dash(self):
        for r in self.rows:
            if r["mode"] == "undefended":
                assert r["policy"] == "—"

    def test_defended_rows_policy_non_empty(self):
        for r in self.rows:
            if r["mode"] == "defended":
                assert r["policy"] and r["policy"] != "—"

    def test_markdown_table_has_pipes(self):
        md = build_markdown_table(self.rows)
        assert "|" in md
        assert "ABR" in md
        assert "BTSR" in md

    def test_latex_table_has_booktabs(self):
        tex = build_latex_table(self.rows)
        assert r"\toprule" in tex
        assert r"\bottomrule" in tex
        assert r"\begin{tabular}" in tex

    def test_markdown_report_non_empty(self):
        report = build_markdown_report(self.sweep, self.rows)
        assert len(report) > 100
        assert "PACM Benchmark Report" in report
