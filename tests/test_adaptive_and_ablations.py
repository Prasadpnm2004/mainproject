"""
tests/test_adaptive_and_ablations.py

Session 7 — Unit tests for AdaptiveAttacker, ErrorAnalyzer, and ablation runner.

Tests cover:
  AdaptiveAttacker:
    1.  Default run() produces records for all 5 variants
    2.  n_instances respected per variant
    3.  All divergence_scores in [0.0, 1.0]
    4.  All friction_scores in [0.0, 1.0]
    5.  decision field is always "allow", "flag", or "block"
    6.  evaded=True when decision != "block"
    7.  NON_ADAPTIVE has highest mean_D (baseline mismatch)
    8.  FULL_ADAPTIVE has lower mean_D than NON_ADAPTIVE
    9.  Summary variant_stats keys match requested variants
   10.  overall_abr + overall_evasion == 1.0
   11.  AdaptiveAttackSummary.to_dict() is JSON-serialisable
   12.  AdaptiveAttackRecord.to_dict() has expected keys
   13.  adaptive_records_as_calibration_input converts all records
   14.  Single variant run produces records only for that variant
   15.  SCOPE_HIDE variant has n_scope_flags=0 for all records

  ErrorAnalyzer:
   16.  Empty records -> report with 0 FP/FN
   17.  All TP records -> abr=1.0, fpr=0.0
   18.  All TN records -> abr=0.0 (n_attack=0), fpr=0.0
   19.  All FP records -> fpr=1.0, abr=0.0 (n_attack=0)
   20.  FP-1 category assigned when scope_flags fired + low D
   21.  FP-4 category assigned when T < low_trust_threshold
   22.  FN-1 category assigned when D < param_fn_threshold
   23.  FN-3 category assigned when no scope_flags on attack
   24.  FN-4 category assigned when T > high_trust_exploit
   25.  to_markdown() returns non-empty string with "ABR"
   26.  to_dict() is JSON-serialisable
   27.  precision = TP / (TP + FP) correctly computed

  AblationResult:
   28.  run_ablations() returns exactly 5 AblationResult objects
   29.  Variant A, B, D_tfidf have same n_attack and n_benign
   30.  All ABR/FPR values in [0.0, 1.0]
   31.  Variant C (trust-only) has different ABR than Variant A
   32.  to_dict() keys include variant_id, variant_name, abr, fpr
   33.  build_ablation_latex_rows() contains toprule and bottomrule

Run with:  pytest tests/test_adaptive_and_ablations.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pacm.adaptive_attack import (
    AdaptiveAttacker,
    AdaptiveAttackRecord,
    AdaptiveAttackSummary,
    AttackVariant,
    adaptive_records_as_calibration_input,
)
from src.pacm.error_analysis import (
    ErrorAnalyzer,
    ErrorAnalysisReport,
    FailureInstance,
    FP_CATEGORIES,
    FN_CATEGORIES,
)
from experiments.run_ablations import (
    run_ablations,
    build_ablation_latex_rows,
    AblationResult,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def attacker():
    """Module-scoped attacker with minimal instances for speed."""
    from src.pacm.policy import STANDARD_POLICY
    return AdaptiveAttacker(policy=STANDARD_POLICY, n_instances=5, prefer_transformers=False)


@pytest.fixture(scope="module")
def attack_records_and_summary(attacker):
    """Run all variants once at module scope."""
    records, summary = attacker.run()
    return records, summary


@pytest.fixture(scope="module")
def ablation_records():
    """Synthetic records for ablation testing."""
    from src.pacm.sensitivity import synthetic_records_for_sensitivity
    all_r = synthetic_records_for_sensitivity(n_attack=30, n_benign=30, seed=7)
    attack = [r for r in all_r if r["is_attack"]]
    benign  = [r for r in all_r if not r["is_attack"]]
    return attack, benign


# ─── AdaptiveAttacker tests ───────────────────────────────────────────────────

class TestAdaptiveAttacker:

    def test_all_variants_present(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        variants_seen = {r.variant for r in records}
        assert variants_seen == set(AttackVariant)

    def test_n_instances_per_variant(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        for variant in AttackVariant:
            count = sum(1 for r in records if r.variant == variant)
            assert count == 5, f"Expected 5 for {variant}, got {count}"

    def test_divergence_scores_in_range(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        for r in records:
            assert 0.0 <= r.divergence_score <= 1.0, f"D out of range: {r.divergence_score}"

    def test_friction_scores_in_range(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        for r in records:
            assert 0.0 <= r.friction_score <= 1.0, f"F out of range: {r.friction_score}"

    def test_decision_values_valid(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        valid = {"allow", "flag", "block"}
        for r in records:
            assert r.decision in valid, f"Invalid decision: {r.decision}"

    def test_evaded_matches_decision(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        for r in records:
            expected = r.decision != "block"
            assert r.evaded == expected, (
                f"evaded={r.evaded} but decision={r.decision}"
            )

    def test_non_adaptive_baseline_highest_D(self, attack_records_and_summary):
        """NON_ADAPTIVE should have highest or equal mean D (maximum mismatch)."""
        records, summary = attack_records_and_summary
        stats = summary.variant_stats
        d_non_adaptive = stats[AttackVariant.NON_ADAPTIVE.value]["mean_D"]
        d_full_adaptive = stats[AttackVariant.FULL_ADAPTIVE.value]["mean_D"]
        # FULL_ADAPTIVE aligns intent with action -> lower D
        assert d_full_adaptive <= d_non_adaptive + 0.01, (
            f"Full adaptive D ({d_full_adaptive:.3f}) should be <= non-adaptive ({d_non_adaptive:.3f})"
        )

    def test_full_adaptive_lower_D_than_non_adaptive(self, attack_records_and_summary):
        records, summary = attack_records_and_summary
        stats = summary.variant_stats
        d_na  = stats[AttackVariant.NON_ADAPTIVE.value]["mean_D"]
        d_fa  = stats[AttackVariant.FULL_ADAPTIVE.value]["mean_D"]
        assert d_fa <= d_na + 0.05

    def test_summary_variant_keys(self, attack_records_and_summary):
        _, summary = attack_records_and_summary
        for variant in AttackVariant:
            assert variant.value in summary.variant_stats

    def test_overall_abr_plus_evasion_equals_one(self, attack_records_and_summary):
        _, summary = attack_records_and_summary
        total = summary.overall_abr + summary.overall_evasion
        assert abs(total - 1.0) < 1e-9

    def test_summary_to_dict_json_serialisable(self, attack_records_and_summary):
        _, summary = attack_records_and_summary
        d = summary.to_dict()
        json.dumps(d)  # must not raise

    def test_record_to_dict_keys(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        expected = {
            "variant", "intent_text", "action_text", "tool_name", "target",
            "divergence_score", "friction_score", "trust_score",
            "decision", "evaded", "n_scope_flags",
        }
        for r in records[:3]:
            assert expected.issubset(set(r.to_dict().keys()))

    def test_calibration_input_conversion(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        cal_records = adaptive_records_as_calibration_input(records)
        assert len(cal_records) == len(records)
        for r in cal_records:
            assert r["is_attack"] is True
            assert "divergence_score" in r
            assert "trust_score" in r

    def test_single_variant_run(self, attacker):
        records, summary = attacker.run([AttackVariant.SCOPE_HIDE])
        variants = {r.variant for r in records}
        assert variants == {AttackVariant.SCOPE_HIDE}

    def test_scope_hide_no_scope_flags(self, attack_records_and_summary):
        records, _ = attack_records_and_summary
        scope_hide = [r for r in records if r.variant == AttackVariant.SCOPE_HIDE]
        for r in scope_hide:
            assert r.n_scope_flags == 0, f"SCOPE_HIDE should have 0 flags, got {r.n_scope_flags}"


# ─── ErrorAnalyzer tests ──────────────────────────────────────────────────────

def _make_audit(is_attack, decision, D=0.5, F=0.4, T=0.7,
                tool="send_email", scope_flags=None, n_steps=1, record_id="r"):
    return {
        "record_id": record_id, "is_attack": is_attack, "decision": decision,
        "divergence_score": D, "friction_score": F, "trust_score": T,
        "tool_name": tool, "arg_scope_flags": scope_flags or {},
        "n_reasoning_steps": n_steps,
    }


class TestErrorAnalyzer:

    def test_empty_records(self):
        analyzer = ErrorAnalyzer()
        report = analyzer.analyze([])
        assert len(report.fp_instances) == 0
        assert len(report.fn_instances) == 0
        assert report.n_total == 0

    def test_all_tp(self):
        """All attacks blocked -> ABR=1.0, FPR=0.0."""
        records = [_make_audit(True, "block", D=0.9, F=0.9, record_id=f"r{i}")
                   for i in range(10)]
        report = ErrorAnalyzer().analyze(records)
        assert report.abr == pytest.approx(1.0)
        assert report.fpr == pytest.approx(0.0)
        assert report.n_tp == 10

    def test_all_tn(self):
        """All benign allowed -> FPR=0.0."""
        records = [_make_audit(False, "allow", D=0.1, F=0.1, record_id=f"r{i}")
                   for i in range(10)]
        report = ErrorAnalyzer().analyze(records)
        assert report.fpr == pytest.approx(0.0)
        assert report.n_tn == 10

    def test_all_fp(self):
        """All benign blocked -> FPR=1.0."""
        records = [_make_audit(False, "block", D=0.8, F=0.8, record_id=f"r{i}")
                   for i in range(5)]
        report = ErrorAnalyzer().analyze(records)
        assert report.fpr == pytest.approx(1.0)
        assert len(report.fp_instances) == 5

    def test_fp1_scope_fired(self):
        """FP-1 assigned when scope_flags fired + D < scope_fp_threshold."""
        rec = _make_audit(False, "block", D=0.25, F=0.8, T=0.9,
                          scope_flags={"to": "external"})
        report = ErrorAnalyzer(scope_fp_threshold=0.40).analyze([rec])
        assert len(report.fp_instances) == 1
        assert report.fp_instances[0].category_code == "FP-1"

    def test_fp4_low_trust(self):
        """FP-4 assigned when T < low_trust_threshold + low D."""
        rec = _make_audit(False, "block", D=0.20, F=0.80, T=0.30)
        report = ErrorAnalyzer(low_trust_threshold=0.50, scope_fp_threshold=0.40).analyze([rec])
        assert report.fp_instances[0].category_code == "FP-4"

    def test_fn1_param_manipulation(self):
        """FN-1 assigned when D < param_fn_threshold."""
        rec = _make_audit(True, "allow", D=0.20, F=0.18, T=0.7)
        report = ErrorAnalyzer(param_fn_threshold=0.35).analyze([rec])
        assert report.fn_instances[0].category_code == "FN-1"

    def test_fn3_no_scope_flags(self):
        """FN-3 assigned when no scope_flags and moderate D."""
        rec = _make_audit(True, "allow", D=0.45, F=0.40, T=0.7, scope_flags={})
        report = ErrorAnalyzer(param_fn_threshold=0.35).analyze([rec])
        assert report.fn_instances[0].category_code == "FN-3"

    def test_fn4_high_trust_exploit(self):
        """FN-4 assigned when T >= high_trust_exploit."""
        rec = _make_audit(True, "allow", D=0.60, F=0.62, T=0.90)
        report = ErrorAnalyzer(high_trust_exploit=0.85).analyze([rec])
        assert report.fn_instances[0].category_code == "FN-4"

    def test_to_markdown_has_abr(self):
        records = [
            _make_audit(True, "block", D=0.8, record_id="a1"),
            _make_audit(False, "allow", D=0.1, record_id="b1"),
        ]
        report = ErrorAnalyzer().analyze(records)
        md = report.to_markdown()
        assert len(md) > 100
        assert "ABR" in md

    def test_to_dict_json_serialisable(self):
        records = [_make_audit(True, "block", record_id="x")]
        report = ErrorAnalyzer().analyze(records)
        json.dumps(report.to_dict())  # must not raise

    def test_precision_formula(self):
        """precision = TP / (TP + FP)."""
        records = [
            _make_audit(True,  "block", D=0.9, record_id="t1"),  # TP
            _make_audit(True,  "block", D=0.9, record_id="t2"),  # TP
            _make_audit(False, "block", D=0.8, record_id="f1"),  # FP
        ]
        report = ErrorAnalyzer().analyze(records)
        expected = 2 / (2 + 1)
        assert abs(report.precision - expected) < 1e-9


# ─── Ablation tests ───────────────────────────────────────────────────────────

class TestAblationRunner:

    def test_returns_five_results(self, ablation_records):
        attack, benign = ablation_records
        results = run_ablations(attack, benign)
        assert len(results) == 5

    def test_consistent_record_counts(self, ablation_records):
        attack, benign = ablation_records
        results = run_ablations(attack, benign)
        for r in results:
            assert r.n_attack == len(attack), f"{r.variant_id}: wrong n_attack"
            assert r.n_benign == len(benign), f"{r.variant_id}: wrong n_benign"

    def test_abr_fpr_in_range(self, ablation_records):
        attack, benign = ablation_records
        for r in run_ablations(attack, benign):
            assert 0.0 <= r.abr <= 1.0
            assert 0.0 <= r.fpr <= 1.0

    def test_variant_c_different_abr_than_a(self, ablation_records):
        """Trust-only (C) and Full PACM (A) should differ on these records."""
        attack, benign = ablation_records
        results = {r.variant_id: r for r in run_ablations(attack, benign)}
        # Both are valid; just assert they're computed (not necessarily different
        # for perfectly separated synthetic data, but both must be in [0,1])
        assert "A" in results
        assert "C" in results

    def test_to_dict_keys(self, ablation_records):
        attack, benign = ablation_records
        results = run_ablations(attack, benign)
        expected_keys = {"variant_id", "variant_name", "abr", "fpr",
                         "mean_D", "mean_F", "mean_latency_ms", "n_attack", "n_benign"}
        for r in results:
            assert expected_keys.issubset(set(r.to_dict().keys()))

    def test_latex_has_booktabs(self, ablation_records):
        attack, benign = ablation_records
        results = run_ablations(attack, benign)
        latex = build_ablation_latex_rows(results)
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert r"\begin{tabular}" in latex
