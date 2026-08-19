"""
tests/test_sensitivity.py

Session 6 — Unit tests for AlphaSensitivityAnalyzer and related utilities.

Tests cover:
  AlphaSensitivityAnalyzer:
    1.  Empty records -> SensitivityResult with no alpha_results entries
    2.  Single alpha value produces one AlphaResult
    3.  alpha_results length == len(alpha_grid)
    4.  best_alpha is one of the alpha_grid values
    5.  Separation = mean_d_attack - mean_d_benign always
    6.  ABR + FPR in [0, 1] for all alpha values
    7.  Fast path: d_semantic + p_scope fields re-weighted correctly
    8.  Proxy path: divergence_score field with no d_semantic/p_scope
    9.  High alpha (0.9) -> semantic dominates -> separation tracks semantic diff
   10.  Low alpha (0.5)  -> scope dominates  -> higher FPR when benign has no scope flags
   11.  Best alpha is the one with highest separation
   12.  SensitivityResult.to_dict() is JSON-serialisable
   13.  AlphaResult.to_dict() has all expected keys
   14.  All-attack records -> n_benign=0, FPR=0.0
   15.  All-benign records -> n_attack=0, ABR=0.0

  synthetic_records_for_sensitivity:
   16.  Returns n_attack + n_benign records
   17.  All records have d_semantic and p_scope fields
   18.  attack records have higher mean d_semantic than benign
   19.  Seed reproducibility: same seed -> same records
   20.  d_semantic and p_scope always in [0, 1]

  SensitivityResult properties:
   21.  n_records matches total input records
   22.  backend field is non-empty string
   23.  alpha_grid matches the input grid
   24.  threshold field matches the analyzer threshold
   25.  alpha_results sorted by alpha ascending (same as grid order)

Run with:  pytest tests/test_sensitivity.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pacm.sensitivity import (
    AlphaSensitivityAnalyzer,
    AlphaResult,
    SensitivityResult,
    synthetic_records_for_sensitivity,
    _safe_mean,
    _safe_std,
)


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _attack_records(n=20, d_sem=0.75, p_scope=0.30, noise=0.0) -> list[dict]:
    """Create attack records with explicit d_semantic and p_scope."""
    return [
        {"is_attack": True,  "d_semantic": d_sem,   "p_scope": p_scope,
         "divergence_score": 0.70 * d_sem + 0.30 * p_scope}
        for _ in range(n)
    ]

def _benign_records(n=20, d_sem=0.20, p_scope=0.05) -> list[dict]:
    """Create benign records with explicit d_semantic and p_scope."""
    return [
        {"is_attack": False, "d_semantic": d_sem,   "p_scope": p_scope,
         "divergence_score": 0.70 * d_sem + 0.30 * p_scope}
        for _ in range(n)
    ]

def _mixed(n_att=20, n_ben=20) -> list[dict]:
    return _attack_records(n_att) + _benign_records(n_ben)

def _make_analyzer(alphas=None, threshold=0.50) -> AlphaSensitivityAnalyzer:
    return AlphaSensitivityAnalyzer(
        alpha_grid=alphas or [0.5, 0.7, 0.9],
        threshold=threshold,
        prefer_transformers=False,
    )


# ─── AlphaSensitivityAnalyzer tests ──────────────────────────────────────────

class TestAlphaSensitivityAnalyzer:

    def test_empty_records(self):
        """Empty input -> result with n_records=0 and no alpha_results."""
        analyzer = _make_analyzer()
        result = analyzer.run([])
        assert result.n_records == 0
        assert result.alpha_results == []

    def test_single_alpha(self):
        """Single alpha value produces exactly one AlphaResult."""
        analyzer = _make_analyzer(alphas=[0.7])
        result = analyzer.run(_mixed())
        assert len(result.alpha_results) == 1
        assert result.alpha_results[0].alpha == pytest.approx(0.7)

    def test_alpha_results_length_matches_grid(self):
        """len(alpha_results) == len(alpha_grid)."""
        grid = [0.5, 0.6, 0.7, 0.8, 0.9]
        analyzer = _make_analyzer(alphas=grid)
        result = analyzer.run(_mixed())
        assert len(result.alpha_results) == len(grid)

    def test_best_alpha_in_grid(self):
        """best_alpha is always one of the alpha_grid values."""
        grid = [0.5, 0.6, 0.7, 0.8, 0.9]
        analyzer = _make_analyzer(alphas=grid)
        result = analyzer.run(_mixed())
        assert any(abs(result.best_alpha - a) < 1e-6 for a in grid)

    def test_separation_equals_diff(self):
        """separation == mean_d_attack - mean_d_benign for every AlphaResult."""
        result = _make_analyzer().run(_mixed())
        for r in result.alpha_results:
            expected = r.mean_d_attack - r.mean_d_benign
            assert abs(r.separation - expected) < 1e-9

    def test_abr_fpr_in_range(self):
        """ABR and FPR are both in [0, 1] for all alpha values."""
        result = _make_analyzer().run(_mixed())
        for r in result.alpha_results:
            assert 0.0 <= r.abr_at_threshold <= 1.0
            assert 0.0 <= r.fpr_at_threshold <= 1.0

    def test_fast_path_d_semantic_p_scope(self):
        """D is re-weighted exactly as alpha * d_sem + (1-alpha) * p_scope."""
        # Attack: d_sem=1.0, p_scope=0.0 -> D = alpha * 1.0 = alpha
        # Benign: d_sem=0.0, p_scope=0.0 -> D = 0.0
        records = [
            {"is_attack": True,  "d_semantic": 1.0, "p_scope": 0.0},
            {"is_attack": False, "d_semantic": 0.0, "p_scope": 0.0},
        ]
        for alpha in [0.5, 0.7, 0.9]:
            analyzer = _make_analyzer(alphas=[alpha])
            result = analyzer.run(records)
            r = result.alpha_results[0]
            assert abs(r.mean_d_attack - alpha) < 1e-9, f"alpha={alpha}: D_attack={r.mean_d_attack}"
            assert abs(r.mean_d_benign - 0.0) < 1e-9

    def test_proxy_path_divergence_score(self):
        """Proxy path: records with only divergence_score still produce results."""
        records = [
            {"is_attack": True,  "divergence_score": 0.8},
            {"is_attack": False, "divergence_score": 0.2},
        ]
        result = _make_analyzer(alphas=[0.7]).run(records)
        r = result.alpha_results[0]
        # Separation should be positive (attack D > benign D)
        assert r.separation > 0

    def test_high_alpha_semantic_dominates(self):
        """At alpha=0.9, changing d_semantic has more impact than p_scope."""
        # Records where semantic divergence is high but scope is zero
        records_sem = [
            {"is_attack": True,  "d_semantic": 0.9, "p_scope": 0.0},
            {"is_attack": False, "d_semantic": 0.1, "p_scope": 0.0},
        ]
        analyzer = AlphaSensitivityAnalyzer(alpha_grid=[0.9], threshold=0.5)
        result = analyzer.run(records_sem)
        # separation should be high (0.9 * 0.8 = 0.72)
        assert result.alpha_results[0].separation > 0.5

    def test_low_alpha_scope_amplified(self):
        """At alpha=0.5, p_scope contributes equally with d_semantic."""
        records = [
            {"is_attack": True,  "d_semantic": 0.5, "p_scope": 0.4},
            {"is_attack": False, "d_semantic": 0.5, "p_scope": 0.0},
        ]
        for alpha in [0.5, 0.9]:
            analyzer = AlphaSensitivityAnalyzer(alpha_grid=[alpha], threshold=0.5)
            result = analyzer.run(records)
            r = result.alpha_results[0]
            # At alpha=0.5: D_attack = 0.5*0.5 + 0.5*0.4 = 0.45; D_benign = 0.5*0.5 = 0.25
            # At alpha=0.9: D_attack = 0.9*0.5 + 0.1*0.4 = 0.49; D_benign = 0.9*0.5 = 0.45
            # Lower alpha -> larger separation
            if alpha == 0.5:
                sep_low = r.separation
            else:
                sep_high = r.separation
        assert sep_low > sep_high, "Lower alpha should amplify scope penalty separation"

    def test_best_alpha_is_max_separation(self):
        """best_alpha corresponds to the AlphaResult with the highest separation."""
        result = _make_analyzer(alphas=[0.5, 0.6, 0.7, 0.8, 0.9]).run(_mixed())
        best_sep = max(r.separation for r in result.alpha_results)
        best_result = next(r for r in result.alpha_results
                           if abs(r.alpha - result.best_alpha) < 1e-6)
        assert abs(best_result.separation - best_sep) < 1e-9

    def test_to_dict_json_serialisable(self):
        """SensitivityResult.to_dict() must be JSON-serialisable."""
        result = _make_analyzer().run(_mixed())
        d = result.to_dict()
        json.dumps(d)  # must not raise
        assert "best_alpha" in d
        assert "alpha_results" in d
        assert "n_records" in d

    def test_alpha_result_to_dict_keys(self):
        """AlphaResult.to_dict() has all expected keys."""
        result = _make_analyzer(alphas=[0.7]).run(_mixed())
        d = result.alpha_results[0].to_dict()
        expected_keys = {
            "alpha", "mean_d_attack", "mean_d_benign", "separation",
            "fpr_at_threshold", "abr_at_threshold",
            "std_attack", "std_benign", "threshold", "n_attack", "n_benign",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_all_attack_records_fpr_zero(self):
        """All-attack records -> n_benign=0, FPR=0.0 (no benign calls)."""
        result = _make_analyzer(alphas=[0.7]).run(_attack_records(10))
        r = result.alpha_results[0]
        assert r.n_benign == 0
        assert r.fpr_at_threshold == 0.0

    def test_all_benign_records_abr_zero(self):
        """All-benign records -> n_attack=0, ABR=0.0 (no attacks)."""
        result = _make_analyzer(alphas=[0.7]).run(_benign_records(10))
        r = result.alpha_results[0]
        assert r.n_attack == 0
        assert r.abr_at_threshold == 0.0


# ─── synthetic_records_for_sensitivity tests ──────────────────────────────────

class TestSyntheticRecords:

    def test_total_count(self):
        """Returns exactly n_attack + n_benign records."""
        records = synthetic_records_for_sensitivity(n_attack=30, n_benign=40)
        assert len(records) == 70

    def test_has_d_semantic_p_scope(self):
        """Every record has d_semantic and p_scope fields."""
        records = synthetic_records_for_sensitivity(n_attack=10, n_benign=10)
        for r in records:
            assert "d_semantic" in r, f"Missing d_semantic: {r}"
            assert "p_scope" in r, f"Missing p_scope: {r}"

    def test_attack_higher_d_semantic_than_benign(self):
        """Mean d_semantic for attacks > mean d_semantic for benign."""
        records = synthetic_records_for_sensitivity(n_attack=100, n_benign=100, seed=42)
        att_dsem = [r["d_semantic"] for r in records if r["is_attack"]]
        ben_dsem = [r["d_semantic"] for r in records if not r["is_attack"]]
        assert _safe_mean(att_dsem) > _safe_mean(ben_dsem)

    def test_seed_reproducibility(self):
        """Same seed produces identical records."""
        r1 = synthetic_records_for_sensitivity(seed=99)
        r2 = synthetic_records_for_sensitivity(seed=99)
        assert r1 == r2

    def test_d_semantic_p_scope_in_range(self):
        """d_semantic and p_scope are always in [0, 1]."""
        records = synthetic_records_for_sensitivity(n_attack=50, n_benign=50, seed=7)
        for r in records:
            assert 0.0 <= r["d_semantic"] <= 1.0
            assert 0.0 <= r["p_scope"] <= 1.0


# ─── SensitivityResult properties ─────────────────────────────────────────────

class TestSensitivityResultProperties:

    def setup_method(self):
        self.records = _mixed(30, 30)
        self.grid = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.analyzer = AlphaSensitivityAnalyzer(
            alpha_grid=self.grid, threshold=0.55, prefer_transformers=False,
        )
        self.result = self.analyzer.run(self.records)

    def test_n_records_matches_input(self):
        assert self.result.n_records == len(self.records)

    def test_backend_non_empty(self):
        assert self.result.backend and isinstance(self.result.backend, str)

    def test_alpha_grid_matches(self):
        assert self.result.alpha_grid == self.grid

    def test_threshold_matches(self):
        assert abs(self.result.threshold - 0.55) < 1e-9

    def test_alpha_results_ordered(self):
        """alpha_results should be in the same order as alpha_grid."""
        for i, (r, a) in enumerate(zip(self.result.alpha_results, self.grid)):
            assert abs(r.alpha - a) < 1e-6, f"Position {i}: got {r.alpha}, expected {a}"
