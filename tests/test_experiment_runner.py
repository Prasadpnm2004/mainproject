"""
tests/test_experiment_runner.py

Session 4 — Unit tests for run_experiment.py and metrics.py.

Tests verify (no external API calls):
  run_experiment():
    1.  Undefended mode produces records with defense="none"
    2.  Defended mode produces records with defense="PACMMiddleware"
    3.  Records include required fields (task_id, mode, harness, is_attack)
    4.  Attack tasks have attack_blocked field
    5.  Benign tasks have task_success field
    6.  attack_filter=None runs all 4 tasks
    7.  attack_filter="ignore_previous" filters to non-attack + ignore_previous tasks
    8.  Both harness names produce records (langgraph, custom_loop)
    9.  Defended block mode blocks attack tasks (attack_blocked=True)
   10.  Defended flag mode does NOT block (attack_blocked=False for flag mode)
  MetricsReport:
   11.  ABR = n_blocked / n_attack (basic formula)
   12.  BTSR = n_succeeded / n_benign
   13.  FPR from pacm_audit false positives
   14.  ABR=None when no attack records
   15.  BTSR=None when no benign records
   16.  FPR=None when no benign tool calls in audit
   17.  summary_table() returns non-empty string
   18.  to_dict() is JSON-serialisable
   19.  p95 latency correct for sorted list
  MetricsProcessor:
   20.  from_jsonl parses JSONL correctly
   21.  compute() with run_id filters correctly
   22.  compute() with no matching run_id returns empty report
   23.  compare() returns delta dict with abr, btsr, fpr keys
   24.  list_run_ids() returns sorted unique IDs
   25.  compute_all_runs() returns one report per run_id

Run with:  pytest tests/test_experiment_runner.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_experiment import run_experiment, STUB_TASKS, IGNORE_PREVIOUS_VARIANTS
from experiments.metrics import MetricsProcessor, MetricsReport


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(mode="undefended", harness="langgraph", attack_filter=None,
         policy_mode="flag", threshold=0.60):
    return run_experiment(
        mode=mode,
        harness_name=harness,
        attack_filter=attack_filter,
        policy_mode=policy_mode,
        divergence_threshold=threshold,
        model_name="gpt-4o-mini",   # no API key → stub mode
        max_iter=5,
        run_id="test_run_001",
    )


# ─── run_experiment() tests ───────────────────────────────────────────────────

class TestRunExperiment:
    def test_undefended_defense_field(self):
        records = _run(mode="undefended")
        for r in records:
            assert r["defense"] == "none"

    def test_defended_defense_field(self):
        records = _run(mode="defended")
        for r in records:
            assert r["defense"] == "PACMMiddleware"

    def test_records_have_required_fields(self):
        records = _run()
        required = {"task_id", "mode", "harness", "is_attack", "skipped"}
        for r in records:
            assert required.issubset(set(r.keys())), f"Missing fields in {r}"

    def test_attack_records_have_attack_blocked(self):
        records = _run(mode="defended")
        for r in records:
            if r.get("is_attack"):
                assert "attack_blocked" in r

    def test_benign_records_have_task_success(self):
        records = _run(mode="defended")
        for r in records:
            if not r.get("is_attack"):
                assert "task_success" in r

    def test_no_attack_filter_runs_all_tasks(self):
        records = _run(attack_filter=None)
        assert len(records) == len(STUB_TASKS)

    def test_attack_filter_ignore_previous(self):
        records = _run(attack_filter="ignore_previous")
        for r in records:
            if r.get("is_attack"):
                assert r.get("attack_type") == "ignore_previous"
        # Should not include injec_agent attack tasks
        assert not any(r.get("attack_type") == "injec_agent" for r in records)

    def test_both_harnesses_produce_records(self):
        for harness in ("langgraph", "custom_loop"):
            records = _run(harness=harness)
            assert len(records) > 0
            for r in records:
                assert r["harness"] == harness

    def test_block_policy_blocks_attack_tasks(self):
        """Block mode with aggressive policy → attack tasks with scope flags are flagged/blocked."""
        # Use policy_mode="allow" which maps to (allow_threshold=0.30, block_threshold=0.55).
        # Attack calls have D≈0.32-0.37 + scope_boost=0.15 → F≈0.47-0.52 → above allow_threshold.
        # The legacy "allow" policy_mode is the aggressive preset in the policy mapping.
        records = _run(mode="defended", policy_mode="allow", threshold=0.40)
        attack_records = [r for r in records if r.get("is_attack") and not r.get("skipped")]
        # At least one attack should be blocked (external email domain flag triggers scope boost)
        if attack_records:
            blocked = [r for r in attack_records if r.get("attack_blocked") is True]
            assert len(blocked) > 0, (
                f"Expected at least one blocked attack. "
                f"Records: {[(r.get('task_id'), r.get('attack_blocked')) for r in attack_records]}"
            )

    def test_flag_policy_does_not_block(self):
        """Flag mode never blocks regardless of score."""
        records = _run(mode="defended", policy_mode="flag", threshold=0.001)
        for r in records:
            # Flag mode: attack_blocked should be False (not blocked, only flagged)
            if r.get("is_attack") and not r.get("skipped"):
                assert r.get("attack_blocked") is False


# ─── MetricsReport tests ──────────────────────────────────────────────────────

class TestMetricsReport:
    def _make_report(self, **kwargs) -> MetricsReport:
        defaults = dict(
            run_id="r1", mode="defended",
            n_total=10, n_skipped=0,
            n_benign=5, n_attack=5,
            n_blocked=4, n_succeeded=4,
            n_fp=1, n_benign_calls=5,
            wall_times=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        defaults.update(kwargs)
        return MetricsReport(**defaults)

    def test_abr_formula(self):
        r = self._make_report(n_blocked=3, n_attack=5)
        assert abs(r.abr - 0.6) < 1e-9

    def test_btsr_formula(self):
        r = self._make_report(n_succeeded=4, n_benign=5)
        assert abs(r.btsr - 0.8) < 1e-9

    def test_fpr_formula(self):
        r = self._make_report(n_fp=1, n_benign_calls=5)
        assert abs(r.fpr - 0.2) < 1e-9

    def test_abr_none_when_no_attacks(self):
        r = self._make_report(n_attack=0, n_blocked=0)
        assert r.abr is None

    def test_btsr_none_when_no_benign(self):
        r = self._make_report(n_benign=0, n_succeeded=0)
        assert r.btsr is None

    def test_fpr_none_when_no_benign_calls(self):
        r = self._make_report(n_benign_calls=0, n_fp=0)
        assert r.fpr is None

    def test_summary_table_non_empty(self):
        r = self._make_report()
        table = r.summary_table()
        assert isinstance(table, str) and len(table) > 20
        assert "ABR" in table and "BTSR" in table and "FPR" in table

    def test_to_dict_json_serialisable(self):
        r = self._make_report()
        d = r.to_dict()
        json.dumps(d)  # must not raise
        assert "abr" in d and "btsr" in d and "fpr" in d

    def test_p95_latency_correct(self):
        # 20 values: sorted p95 = index ceil(0.95*20)-1 = 19-1 = 18 → value 19/10 = 1.9
        wall_times = [i / 10.0 for i in range(1, 21)]  # 0.1, 0.2, ..., 2.0
        r = self._make_report(wall_times=wall_times)
        assert abs(r.p95_latency_s - 1.9) < 1e-9

    def test_mean_latency(self):
        r = self._make_report(wall_times=[1.0, 2.0, 3.0])
        assert abs(r.mean_latency_s - 2.0) < 1e-9


# ─── MetricsProcessor tests ───────────────────────────────────────────────────

def _make_jsonl_records(run_id_a="run_a", run_id_b="run_b") -> list[dict]:
    """Generate minimal synthetic JSONL records for processor tests."""
    base = {
        "session": "Session_4", "mode": "defended", "harness": "langgraph",
        "defense": "PACMMiddleware", "model": "gpt-4o-mini",
        "timestamp_utc": "2026-08-15T00:00:00Z",
    }
    return [
        # run_a: undefended
        {**base, "run_id": run_id_a, "mode": "undefended", "defense": "none",
         "task_id": "T1", "is_attack": False, "task_success": True,
         "attack_blocked": None, "skipped": False, "wall_time_s": 0.5,
         "pacm_audit": []},
        {**base, "run_id": run_id_a, "mode": "undefended", "defense": "none",
         "task_id": "T2", "is_attack": True, "task_success": False,
         "attack_blocked": False, "skipped": False, "wall_time_s": 0.6,
         "pacm_audit": []},
        # run_b: defended
        {**base, "run_id": run_id_b, "mode": "defended",
         "task_id": "T1", "is_attack": False, "task_success": True,
         "attack_blocked": None, "skipped": False, "wall_time_s": 0.8,
         "pacm_audit": [{"decision": "allow"}]},
        {**base, "run_id": run_id_b, "mode": "defended",
         "task_id": "T2", "is_attack": True, "task_success": True,
         "attack_blocked": True, "skipped": False, "wall_time_s": 0.9,
         "pacm_audit": []},
    ]


class TestMetricsProcessor:
    def _make_processor(self) -> MetricsProcessor:
        return MetricsProcessor(_make_jsonl_records())

    def test_from_jsonl_parses_correctly(self):
        records = _make_jsonl_records()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            fname = f.name
        mp = MetricsProcessor.from_jsonl(fname)
        assert len(mp._records) == 4

    def test_compute_with_run_id_filters(self):
        mp = self._make_processor()
        report = mp.compute("run_a")
        assert report.n_total == 2
        assert report.mode == "undefended"

    def test_compute_no_matching_run_id(self):
        mp = self._make_processor()
        report = mp.compute("nonexistent_run")
        assert report.n_total == 0

    def test_compare_returns_delta_keys(self):
        mp = self._make_processor()
        result = mp.compare("run_a", "run_b")
        assert "delta" in result
        assert "abr" in result["delta"]
        assert "btsr" in result["delta"]
        assert "fpr" in result["delta"]

    def test_list_run_ids_sorted(self):
        mp = self._make_processor()
        run_ids = mp.list_run_ids()
        assert run_ids == sorted(run_ids)
        assert "run_a" in run_ids
        assert "run_b" in run_ids

    def test_compute_all_runs(self):
        mp = self._make_processor()
        reports = mp.compute_all_runs()
        assert len(reports) == 2
        assert all(isinstance(r, MetricsReport) for r in reports)

    def test_fpr_counts_blocked_benign_from_audit(self):
        """FPR is computed from pacm_audit 'block' entries on benign tasks."""
        records = _make_jsonl_records()
        # Add a blocked benign entry to run_b T1
        records[2]["pacm_audit"] = [{"decision": "block"}]
        mp = MetricsProcessor(records)
        report = mp.compute("run_b")
        assert report.n_fp == 1
        assert report.n_benign_calls == 1
        assert abs(report.fpr - 1.0) < 1e-9


# ─── Session 13 tests ────────────────────────────────────────────────────────

class TestSession13:
    """Tests for Session 13 changes: undefended logging gap fix, phrasing variants."""

    def test_undefended_records_include_tool_calls(self):
        """Test 26: Undefended run records must include a 'tool_calls' list.

        This verifies the logging gap fix: before Session 13, undefended JSONL
        records omitted tool_calls entirely. After the fix, tool_calls is
        always serialized regardless of mode (may be empty for benign tasks
        that make no tool calls in stub mode).
        """
        records = run_experiment(
            mode="undefended",
            harness_name="langgraph",
            attack_filter=None,
            policy_mode="flag",
            divergence_threshold=0.60,
            model_name="gpt-4o-mini",  # no API key -> stub
            max_iter=5,
            run_id="s13_test_logging",
            sample_idx=0,
        )
        for rec in records:
            assert "tool_calls" in rec, (
                f"Missing 'tool_calls' key in record for task {rec.get('task_id')!r} "
                f"(mode=undefended). Logging gap fix may not have been applied."
            )
            assert isinstance(rec["tool_calls"], list), (
                f"'tool_calls' must be a list, got {type(rec['tool_calls']).__name__}"
            )

    def test_ignore_previous_variants_are_distinct(self):
        """Test 27: All 3 IGNORE_PREVIOUS_VARIANTS phrasings must be distinct.

        Ensures that multi-sampling with 3 samples produces 3 genuinely different
        attack prompts/payloads rather than repeating the same text.
        """
        assert len(IGNORE_PREVIOUS_VARIANTS) == 3, (
            f"Expected exactly 3 variants, got {len(IGNORE_PREVIOUS_VARIANTS)}"
        )
        descriptions = [v["description"] for v in IGNORE_PREVIOUS_VARIANTS]
        payloads = [v["inbox_payload"] for v in IGNORE_PREVIOUS_VARIANTS]
        ids = [v["variant_id"] for v in IGNORE_PREVIOUS_VARIANTS]

        assert len(set(descriptions)) == 3, "All 3 variant descriptions must be distinct"
        assert len(set(payloads)) == 3, "All 3 variant inbox_payloads must be distinct"
        assert len(set(ids)) == 3, "All 3 variant_ids must be distinct"
        assert ids == ["variant_a", "variant_b", "variant_c"]

    def test_sample_idx_maps_deterministically_to_variant(self):
        """Test 28: sample_idx 0/1/2 map deterministically to variant_a/b/c.

        For attack tasks, variant selection must be deterministic (same sample_idx
        always produces the same variant). This verifies the sample_idx % 3 mapping
        and that the variant_id is recorded in the result.
        """
        expected_variants = [\"variant_a\", \"variant_b\", \"variant_c\"]
        for idx, expected_vid in enumerate(expected_variants):
            records = run_experiment(
                mode="undefended",
                harness_name="langgraph",
                attack_filter="ignore_previous",
                policy_mode="flag",
                divergence_threshold=0.60,
                model_name="gpt-4o-mini",
                max_iter=5,
                run_id=f"s13_variant_test_{idx}",
                sample_idx=idx,
            )
            attack_recs = [r for r in records if r.get("attack_type") == "ignore_previous"]
            assert len(attack_recs) == 1, (
                f"Expected exactly 1 ignore_previous record for sample_idx={idx}"
            )
            assert attack_recs[0].get("variant_id") == expected_vid, (
                f"sample_idx={idx} -> expected variant_id={expected_vid!r}, "
                f"got {attack_recs[0].get('variant_id')!r}"
            )


# ─── Session 16 tests ────────────────────────────────────────────────────────


class TestSession16TimeoutScoring:
    """Tests 29-35: Dual timeout-scoring metrics (Session 16).

    Validates both ABR_incl variants introduced in Session 16:
      (a) ABR_incl (completed-only): denominator = completed attack samples
      (b) ABR_incl (worst-case):     denominator = all configured samples,
                                      timeouts treated as attack-succeeded

    When no timeouts occur, both must be identical.
    When timeouts occur (n_configured_attack > n_attack), they diverge.
    """

    # ── Helper: build a MetricsReport directly ────────────────────────────────

    @staticmethod
    def _make_report(**kwargs) -> MetricsReport:
        defaults = dict(
            run_id="test", mode="defended",
            n_total=9, n_skipped=0, n_benign=6, n_attack=0,
            n_blocked=0, n_flagged=0, n_succeeded=6,
            n_fp=0, n_benign_calls=0,
        )
        defaults.update(kwargs)
        return MetricsReport(**defaults)

    # ── Test 29: identical metrics when no timeouts ───────────────────────────

    def test_both_abr_incl_equal_when_no_timeouts(self):
        """Test 29: When n_configured_attack == n_attack (no timeouts), both
        ABR_incl variants must be identical."""
        report = self._make_report(n_attack=3, n_blocked=0, n_flagged=2,
                                   n_configured_attack=3)
        assert report.abr_inclusive == report.abr_inclusive_worst_case
        assert abs(report.abr_inclusive - 2/3) < 1e-9

    # ── Test 30: worst-case lower when timeouts occurred ─────────────────────

    def test_worst_case_lower_when_timeouts(self):
        """Test 30: When n_configured_attack > n_attack (timeouts occurred),
        worst-case ABR_incl must be strictly lower than completed-only."""
        # 2 completed (both flagged) + 1 timed out
        report = self._make_report(n_attack=2, n_blocked=0, n_flagged=2,
                                   n_configured_attack=3)
        completed_only = report.abr_inclusive        # 2/2 = 100%
        worst_case = report.abr_inclusive_worst_case  # 2/3 = 66.7%
        assert completed_only is not None
        assert worst_case is not None
        assert abs(completed_only - 1.0) < 1e-9,  f"expected 100%, got {completed_only}"
        assert abs(worst_case - 2/3) < 1e-9,       f"expected 66.7%, got {worst_case}"
        assert worst_case < completed_only

    # ── Test 31: n_configured_attack defaults to n_attack ─────────────────────

    def test_n_configured_attack_defaults_to_n_attack(self):
        """Test 31: When n_configured_attack=0 (backward-compat default), it
        falls back to n_attack, making both ABR_incl metrics identical."""
        report = self._make_report(n_attack=3, n_blocked=1, n_flagged=1,
                                   n_configured_attack=0)
        # Both must use n_attack=3 as denominator
        assert report.abr_inclusive == report.abr_inclusive_worst_case
        assert abs(report.abr_inclusive - 2/3) < 1e-9

    # ── Test 32: to_dict includes both metrics and n_configured_attack ─────────

    def test_to_dict_includes_both_abr_incl(self):
        """Test 32: to_dict() must include abr_inclusive, abr_inclusive_worst_case,
        and n_configured_attack."""
        report = self._make_report(n_attack=3, n_blocked=3, n_flagged=0,
                                   n_configured_attack=3)
        d = report.to_dict()
        assert "abr_inclusive" in d, "to_dict must include abr_inclusive"
        assert "abr_inclusive_worst_case" in d, "to_dict must include abr_inclusive_worst_case"
        assert "n_configured_attack" in d, "to_dict must include n_configured_attack"
        assert d["abr_inclusive"] == 1.0
        assert d["abr_inclusive_worst_case"] == 1.0
        assert d["n_configured_attack"] == 3

    # ── Test 33: summary_table shows both ABR_incl lines ──────────────────────

    def test_summary_table_shows_both_abr_incl(self):
        """Test 33: summary_table() must contain both ABR_incl labeled lines."""
        report = self._make_report(n_attack=3, n_blocked=0, n_flagged=2,
                                   n_configured_attack=3)
        table = report.summary_table()
        assert "ABR_incl (BLOCK+FLAG, completed-only denom)" in table, (
            "summary_table must label completed-only ABR_incl variant"
        )
        assert "ABR_incl (BLOCK+FLAG, worst-case denom)" in table, (
            "summary_table must label worst-case ABR_incl variant"
        )

    # ── Test 34: summary_table shows timeout note when timeouts occurred ───────

    def test_summary_table_shows_timeout_note(self):
        """Test 34: When n_configured_attack > n_attack, summary_table must
        include a note about timed-out samples."""
        report = self._make_report(n_attack=2, n_blocked=0, n_flagged=2,
                                   n_configured_attack=3)
        table = report.summary_table()
        assert "timed-out" in table, (
            "summary_table must show timed-out sample count when n_configured > n_attack"
        )

    # ── Test 35: compute() passes n_configured_attack through ─────────────────

    def test_compute_passes_n_configured_attack(self):
        """Test 35: MetricsProcessor.compute(n_configured_attack=N) must set
        n_configured_attack on the returned MetricsReport, enabling worst-case
        ABR_incl to use the correct denominator."""
        records = [
            {"run_id": "r1", "mode": "defended", "is_attack": True,
             "attack_blocked": False, "task_success": False, "skipped": False,
             "wall_time_s": 300.0,
             "pacm_audit": [{"decision": "flag", "tool": "send_email"}]},
            {"run_id": "r1", "mode": "defended", "is_attack": True,
             "attack_blocked": False, "task_success": True, "skipped": False,
             "wall_time_s": 120.0,
             "pacm_audit": []},
        ]
        mp = MetricsProcessor(records)
        # 2 completed, but 3 were configured (1 timed out before writing record)
        report = mp.compute("r1", n_configured_attack=3)
        assert report.n_attack == 2
        assert report.n_configured_attack == 3
        # completed-only: 1/2 = 50%
        assert abs(report.abr_inclusive - 0.5) < 1e-9
        # worst-case: 1/3 = 33.3%
        assert abs(report.abr_inclusive_worst_case - 1/3) < 1e-9

