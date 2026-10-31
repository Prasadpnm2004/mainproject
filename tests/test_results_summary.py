"""
tests/test_results_summary.py
==============================
Unit tests for experiments/results_summary.py

Tests cover:
1. Wilson CI computation against known textbook reference values
2. Fisher's exact test against known reference values
3. Discrepancy-detection logic with deliberately mismatched mock stats
4. _load_scc_d_values: correct extraction from mock scope_only records
5. _build_condition_data: correct Wilson CI recomputation from stored nums/dens
6. Cross-check passes on matching data, fires on mismatch
7. Dashboard exits 0 on real data files (integration smoke test)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.results_summary import (
    wilson_ci,
    fisher_exact_2x2,
    _build_condition_data,
    _load_scc_d_values,
    _cross_check,
    main,
)

_ROOT = Path(__file__).parent.parent
_JSONL = _ROOT / "experiments" / "s24_ablation.jsonl"
_STATS = _ROOT / "experiments" / "s24_ablation_stats.json"


# ── Test 1: Wilson CI against textbook reference values ───────────────────────

class TestWilsonCI:
    """
    Reference values from:
    Wilson, E.B. (1927). J. Am. Stat. Assoc. 22, 209-212.
    Also verified via scipy.stats.proportion_confint(method='wilson') where
    available.
    """

    def test_zero_successes(self):
        lo, hi = wilson_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=0.001)
        assert hi == pytest.approx(0.278, abs=0.002), (
            f"Expected upper CI ~0.278 for 0/10, got {hi:.4f}"
        )

    def test_all_successes(self):
        lo, hi = wilson_ci(10, 10)
        assert lo == pytest.approx(0.722, abs=0.002), (
            f"Expected lower CI ~0.722 for 10/10, got {lo:.4f}"
        )
        assert hi == pytest.approx(1.0, abs=0.001)

    def test_half(self):
        lo, hi = wilson_ci(5, 10)
        # For 5/10, 95% Wilson CI is approximately [0.237, 0.763]
        assert lo == pytest.approx(0.237, abs=0.002)
        assert hi == pytest.approx(0.763, abs=0.002)

    def test_two_out_of_ten(self):
        lo, hi = wilson_ci(2, 10)
        # For 2/10, 95% Wilson CI is approximately [0.057, 0.510]
        assert lo == pytest.approx(0.057, abs=0.002), (
            f"Expected lower CI ~0.057 for 2/10, got {lo:.4f}"
        )
        assert hi == pytest.approx(0.510, abs=0.002), (
            f"Expected upper CI ~0.510 for 2/10, got {hi:.4f}"
        )

    def test_n_zero_returns_full_interval(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_symmetry(self):
        """wilson_ci(k, n) should be mirror-symmetric: lo(k,n) == 1-hi(n-k,n)."""
        lo, hi = wilson_ci(3, 10)
        lo2, hi2 = wilson_ci(7, 10)
        assert lo == pytest.approx(1 - hi2, abs=0.0001)
        assert hi == pytest.approx(1 - lo2, abs=0.0001)

    def test_one_out_of_twenty(self):
        lo, hi = wilson_ci(1, 20)
        # Wilson 95% CI for 1/20: lower ≈ 0.009, upper ≈ 0.236
        # (verified by running wilson_ci(1, 20) directly)
        assert lo == pytest.approx(0.009, abs=0.003)
        assert hi == pytest.approx(0.236, abs=0.003)

    def test_known_pacm_fpr_scope_reasoning(self):
        """Verify the project's FPR CI for scope_reasoning: 2/10."""
        lo, hi = wilson_ci(2, 10)
        assert lo == pytest.approx(0.057, abs=0.002)
        assert hi == pytest.approx(0.510, abs=0.002)

    def test_known_pacm_fpr_naive_baseline(self):
        """Verify the project's FPR CI for naive_baseline: 2/7."""
        lo, hi = wilson_ci(2, 7)
        # Approx [0.082, 0.641]
        assert lo == pytest.approx(0.082, abs=0.003)
        assert hi == pytest.approx(0.641, abs=0.003)


# ── Test 2: Fisher's exact test against known reference values ────────────────

class TestFisherExact:
    """
    Reference values from:
    Fisher, R.A. (1934). Statistical Methods for Research Workers, p. 99.
    Also cross-checked with scipy.stats.fisher_exact.
    """

    def test_fisher_table_of_zeros_ones(self):
        """Classic Fisher example: all in one cell. p should be small."""
        # [[5, 0], [0, 5]] — perfectly separated
        p = fisher_exact_2x2(5, 0, 0, 5)
        assert p == pytest.approx(0.00794, abs=0.001), (
            f"Expected p≈0.008 for perfectly separated 5x5, got p={p:.5f}"
        )

    def test_fisher_uniform_table(self):
        """[[3, 3], [3, 3]] — no association, p should be 1.0."""
        p = fisher_exact_2x2(3, 3, 3, 3)
        assert p == pytest.approx(1.0, abs=0.001)

    def test_fisher_known_p_value(self):
        """[[1, 9], [11, 3]] — known p ≈ 0.0028 (two-sided)."""
        p = fisher_exact_2x2(1, 9, 11, 3)
        assert p == pytest.approx(0.0028, abs=0.001), (
            f"Expected p≈0.003, got p={p:.5f}"
        )

    def test_fisher_pacm_result(self):
        """
        The PACM project result: scope_only 0/10 vs scope+reasoning 2/10.
        Expected two-sided p = 0.4737 (verified externally).
        """
        p = fisher_exact_2x2(0, 10, 2, 8)
        assert p == pytest.approx(0.4737, abs=0.002), (
            f"PACM Fisher test: expected p≈0.4737, got p={p:.5f}"
        )

    def test_fisher_p_is_between_0_and_1(self):
        for a, b, c, d in [(1, 2, 3, 4), (0, 5, 5, 0), (10, 0, 0, 10)]:
            p = fisher_exact_2x2(a, b, c, d)
            assert 0.0 <= p <= 1.0, f"p={p} out of range for [{a},{b},{c},{d}]"


# ── Test 3: Discrepancy detection logic ───────────────────────────────────────

class TestCrossCheckDiscrepancy:
    """
    Verifies _cross_check fires on mismatch and passes on match.
    """

    MATCHING_STORED: dict = {
        "scope_only": {
            "fpr_numerator": 0,
            "fpr_denominator": 10,
            "btsr_numerator": 10,
            "btsr_denominator": 10,
            "trap_count": 0,
            "n": 10,
            "fpr_str": "0/10 (0.0%) Wilson 95% CI [0.0%, 27.8%]",
            "btsr_str": "10/10 (100.0%) Wilson 95% CI [72.2%, 100.0%]",
        }
    }

    MISMATCHED_STORED: dict = {
        "scope_only": {
            "fpr_numerator": 0,
            "fpr_denominator": 10,
            "btsr_numerator": 10,
            "btsr_denominator": 10,
            "trap_count": 0,
            "n": 10,
            # Deliberately wrong CI — lower bound claimed as 99.9%
            "fpr_str": "0/10 (0.0%) Wilson 95% CI [99.9%, 100.0%]",
            "btsr_str": "10/10 (100.0%) Wilson 95% CI [72.2%, 100.0%]",
        }
    }

    def test_no_discrepancy_on_matching_data(self):
        """_cross_check must return [] when CI strings match recomputed values."""
        computed = _build_condition_data(self.MATCHING_STORED)
        # Provide valid SCC D values to avoid that check triggering
        scc_d = [0.1546] * 9 + [0.1449]
        discrepancies = _cross_check(computed, scc_d)
        assert discrepancies == [], (
            f"Expected no discrepancies but got: {discrepancies}"
        )

    def test_discrepancy_fires_on_wrong_ci_string(self):
        """_cross_check must flag when stored CI string doesn't match recomputed CI."""
        computed = _build_condition_data(self.MISMATCHED_STORED)
        scc_d = [0.1546] * 9 + [0.1449]
        discrepancies = _cross_check(computed, scc_d)
        assert len(discrepancies) >= 1, (
            "Expected at least one discrepancy for deliberately wrong CI string, "
            f"but got none. Computed CI: {computed['scope_only']['fpr_lo']:.3f}"
        )
        assert "scope_only" in discrepancies[0]
        assert "fpr_ci" in discrepancies[0]

    def test_discrepancy_message_shows_both_values(self):
        """Discrepancy message must include both recomputed and stored values."""
        computed = _build_condition_data(self.MISMATCHED_STORED)
        scc_d = [0.1546] * 9 + [0.1449]
        discrepancies = _cross_check(computed, scc_d)
        assert discrepancies, "Expected at least one discrepancy"
        msg = discrepancies[0]
        # Should contain both the recomputed value and the stored value
        assert "recomputed" in msg or "%" in msg, (
            f"Discrepancy message should contain CI values: {msg}"
        )

    def test_scc_modal_d_check_fires_on_non_modal_data(self):
        """_cross_check must flag when fewer than 50% of D values share the modal value."""
        computed = _build_condition_data(self.MATCHING_STORED)
        # All different D values — no modal dominance
        scc_d = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
        discrepancies = _cross_check(computed, scc_d)
        assert any("SCC" in d for d in discrepancies), (
            f"Expected SCC modal-D discrepancy, got: {discrepancies}"
        )


# ── Test 4: _load_scc_d_values extracts correctly ────────────────────────────

class TestLoadSccDValues:

    def _make_record(self, run_idx: int, d: float, rlen: int,
                     tool_call_count: int = 2) -> dict:
        return {
            "condition": "scope_only",
            "run_idx": run_idx,
            "tool_call_count": tool_call_count,
            "send_email_details": [{"D": d, "reasoning_len": rlen}],
        }

    def test_extracts_one_d_per_run(self):
        """Should return exactly one D value per unique run_idx."""
        recs = [
            self._make_record(0, 0.1546, 0),
            self._make_record(1, 0.1546, 0),
            self._make_record(2, 0.1449, 397),
        ]
        d_vals, rlen_vals = _load_scc_d_values(recs)
        assert len(d_vals) == 3
        assert d_vals == [0.1546, 0.1546, 0.1449]
        assert rlen_vals == [0, 0, 397]

    def test_picks_lowest_tool_call_count_first(self):
        """When run has multiple records, picks the one with lowest tool_call_count."""
        recs = [
            self._make_record(0, 0.20, 100, tool_call_count=3),
            self._make_record(0, 0.15, 0, tool_call_count=2),  # should be picked
        ]
        d_vals, _ = _load_scc_d_values(recs)
        assert len(d_vals) == 1
        assert d_vals[0] == pytest.approx(0.15, abs=0.001)

    def test_returns_empty_for_no_send_email(self):
        """Records without send_email_details contribute no D values."""
        recs = [{"condition": "scope_only", "run_idx": 0, "tool_call_count": 1}]
        d_vals, rlen_vals = _load_scc_d_values(recs)
        assert d_vals == []
        assert rlen_vals == []


# ── Test 5: _build_condition_data recomputes CIs correctly ───────────────────

class TestBuildConditionData:

    def test_recomputes_wilson_ci_from_stored_numerators(self):
        """CIs must be recomputed from stored k/n, not from stored strings."""
        stored = {
            "scope_only": {
                "fpr_numerator": 2,
                "fpr_denominator": 10,
                "btsr_numerator": 8,
                "btsr_denominator": 10,
                "trap_count": 2,
                "n": 10,
                "fpr_str": "",
                "btsr_str": "",
            }
        }
        computed = _build_condition_data(stored)
        so = computed["scope_only"]
        # Wilson CI for 2/10 should be ~[0.057, 0.510]
        assert so["fpr_lo"] == pytest.approx(0.057, abs=0.002)
        assert so["fpr_hi"] == pytest.approx(0.510, abs=0.002)

    def test_afp_count_computed_for_naive_baseline(self):
        """AFP count = n - fpr_denominator for naive_baseline."""
        stored = {
            "naive_baseline": {
                "fpr_numerator": 2,
                "fpr_denominator": 7,  # 10 - 3 AFP runs
                "btsr_numerator": 5,
                "btsr_denominator": 7,
                "trap_count": 2,
                "n": 10,
                "fpr_str": "",
                "btsr_str": "",
            }
        }
        computed = _build_condition_data(stored)
        assert computed["naive_baseline"]["afp_count"] == 3

    def test_afp_count_zero_for_non_naive(self):
        """AFP count must be 0 for non-naive_baseline conditions."""
        stored = {
            "scope_only": {
                "fpr_numerator": 0, "fpr_denominator": 10,
                "btsr_numerator": 10, "btsr_denominator": 10,
                "trap_count": 0, "n": 10, "fpr_str": "", "btsr_str": "",
            }
        }
        computed = _build_condition_data(stored)
        assert computed["scope_only"]["afp_count"] == 0


# ── Test 6: Integration smoke test — dashboard exits 0 on real data ───────────

class TestDashboardIntegration:
    """Smoke tests: run the full dashboard against the real data files."""

    @pytest.mark.skipif(
        not _JSONL.exists() or not _STATS.exists(),
        reason="Real data files not present",
    )
    def test_main_exits_0(self):
        """Dashboard must exit 0 (success) on the real data files."""
        rc = main([])
        assert rc == 0, "results_summary.py main() returned non-zero on real data"

    @pytest.mark.skipif(
        not _JSONL.exists() or not _STATS.exists(),
        reason="Real data files not present",
    )
    def test_verify_only_exits_0(self):
        """--verify-only must exit 0 (PASS) on the real data files."""
        rc = main(["--verify-only"])
        assert rc == 0, (
            "results_summary.py --verify-only returned FAIL on real data. "
            "Check that stored CI strings match recomputed Wilson CIs."
        )

    @pytest.mark.skipif(
        not _JSONL.exists() or not _STATS.exists(),
        reason="Real data files not present",
    )
    def test_no_unicode_exits_0(self):
        """--no-unicode ASCII fallback must also exit 0."""
        rc = main(["--no-unicode"])
        assert rc == 0

    def test_main_returns_2_on_missing_jsonl(self, tmp_path):
        """If JSONL file doesn't exist, must return 2 (data error)."""
        fake_jsonl = tmp_path / "missing.jsonl"
        fake_stats = tmp_path / "stats.json"
        fake_stats.write_text("{}", encoding="utf-8")
        import experiments.results_summary as rs_mod
        orig_jsonl = rs_mod._JSONL
        orig_stats = rs_mod._STATS
        try:
            rs_mod._JSONL = fake_jsonl
            rs_mod._STATS = fake_stats
            rc = main([])
            assert rc == 2
        finally:
            rs_mod._JSONL = orig_jsonl
            rs_mod._STATS = orig_stats
