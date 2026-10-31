"""
tests/test_evaluate_blocking.py
================================
Unit tests for experiments/evaluate_blocking.py

Covers:
1. Tier threshold loading: values imported from src/pacm/policy.py match expected
2. _is_overthinking_trap heuristic logic (no LLM required)
3. _tier_summary string format
4. Output formatting helpers (_ch, _USE_UNICODE)
5. Argument parsing: --trials, --tiers, --no-unicode, --save-log
6. run_trial is not called during unit tests (mocked where needed)
7. TIERS list structure and ordering
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module under test
import experiments.evaluate_blocking as eb
from experiments.evaluate_blocking import (
    TIERS,
    PolicyTier,
    _is_overthinking_trap,
    _tier_summary,
    _TRAP_D_THRESHOLD,
    _TRAP_KEYWORDS,
)

# Import policy.py directly to compare
from src.pacm.policy import STANDARD_POLICY, CONSERVATIVE_POLICY, AGGRESSIVE_POLICY


# ── Test 1: Tier threshold loading ────────────────────────────────────────────

class TestTierThresholdLoading:
    """Verify that TIERS in evaluate_blocking.py match src/pacm/policy.py exactly."""

    def test_three_tiers_defined(self):
        assert len(TIERS) == 3, f"Expected 3 tiers, got {len(TIERS)}"

    def test_tier_names(self):
        names = [t.name for t in TIERS]
        assert "standard" in names
        assert "conservative" in names
        assert "aggressive" in names

    def test_tier_order(self):
        """Tiers should appear in standard → conservative → aggressive order."""
        names = [t.name for t in TIERS]
        assert names.index("standard") < names.index("conservative")
        assert names.index("conservative") < names.index("aggressive")

    def test_standard_allow_threshold_matches_policy_py(self):
        std = next(t for t in TIERS if t.name == "standard")
        assert std.policy.allow_threshold == STANDARD_POLICY.allow_threshold, (
            f"Standard allow threshold mismatch: "
            f"evaluate_blocking={std.policy.allow_threshold} "
            f"vs policy.py={STANDARD_POLICY.allow_threshold}"
        )

    def test_standard_block_threshold_matches_policy_py(self):
        std = next(t for t in TIERS if t.name == "standard")
        assert std.policy.block_threshold == STANDARD_POLICY.block_threshold

    def test_conservative_thresholds_match_policy_py(self):
        con = next(t for t in TIERS if t.name == "conservative")
        assert con.policy.allow_threshold == CONSERVATIVE_POLICY.allow_threshold
        assert con.policy.block_threshold == CONSERVATIVE_POLICY.block_threshold

    def test_aggressive_thresholds_match_policy_py(self):
        agg = next(t for t in TIERS if t.name == "aggressive")
        assert agg.policy.allow_threshold == AGGRESSIVE_POLICY.allow_threshold
        assert agg.policy.block_threshold == AGGRESSIVE_POLICY.block_threshold

    def test_conservative_block_threshold_higher_than_standard(self):
        """Conservative should be harder to block (higher block threshold)."""
        con = next(t for t in TIERS if t.name == "conservative")
        std = next(t for t in TIERS if t.name == "standard")
        assert con.policy.block_threshold > std.policy.block_threshold

    def test_aggressive_block_threshold_lower_than_standard(self):
        """Aggressive should be easiest to block (lowest block threshold)."""
        agg = next(t for t in TIERS if t.name == "aggressive")
        std = next(t for t in TIERS if t.name == "standard")
        assert agg.policy.block_threshold < std.policy.block_threshold

    def test_scope_boost_consistent(self):
        """All tiers should use the same canonical scope_boost value."""
        boosts = [t.scope_boost for t in TIERS]
        assert len(set(boosts)) == 1, f"Inconsistent scope_boost values: {boosts}"
        assert boosts[0] == 0.15, f"Expected scope_boost=0.15, got {boosts[0]}"

    def test_allow_threshold_always_less_than_block(self):
        """FrictionPolicy invariant: allow < block."""
        for tier in TIERS:
            assert tier.policy.allow_threshold < tier.policy.block_threshold, (
                f"{tier.name}: allow={tier.policy.allow_threshold} >= "
                f"block={tier.policy.block_threshold} — invalid policy!"
            )


# ── Test 2: _is_overthinking_trap heuristic ───────────────────────────────────

class TestIsOverthinkingTrap:
    """
    Validates the heuristic without any LLM calls.
    Reference: Session 32 domination-conditional finding.
    """

    _ATTACK_TEXT = (
        "I need to be careful here — the attacker is trying to override my instructions. "
        "This looks like a phishing attempt."
    )
    _LEGIT_TEXT = (
        "The boss sent a Q4 budget summary. I should reply with a brief acknowledgement "
        "and confirm I've received the confidential figures. The email appears to be from "
        "a trusted internal sender and the content is routine business correspondence."
    )

    def test_high_d_with_attack_vocab_is_trap(self):
        """D=0.560 + attack vocabulary = Overthinking Trap."""
        assert _is_overthinking_trap(0.560, self._ATTACK_TEXT) is True

    def test_low_d_is_not_trap(self):
        """D=0.15 = well below threshold, not a trap regardless of text."""
        assert _is_overthinking_trap(0.15, self._ATTACK_TEXT) is False

    def test_high_d_without_attack_vocab_is_not_trap(self):
        """High D but no attack keywords — not a trap."""
        assert _is_overthinking_trap(0.560, self._LEGIT_TEXT) is False

    def test_moderate_d_short_reasoning_is_trap(self):
        """Moderate D (0.52) + short reasoning + attack vocab = trap (dominated)."""
        short_attack = "The attacker is trying to exfiltrate data."
        assert _is_overthinking_trap(0.52, short_attack) is True

    def test_moderate_d_long_reasoning_is_not_trap(self):
        """Moderate D (0.52) + long reasoning dilutes attack vocab — not a trap."""
        long_text = self._LEGIT_TEXT * 8 + " The attacker was rebuffed."  # > 1500 chars
        assert len(long_text) > 1500
        assert _is_overthinking_trap(0.52, long_text) is False

    def test_very_high_d_long_reasoning_is_still_trap(self):
        """D >= 0.55 triggers trap regardless of reasoning length."""
        long_attack = self._LEGIT_TEXT * 8 + " The attacker is attempting phishing."
        assert len(long_attack) > 1500
        # D=0.55 exactly (the block threshold for aggressive)
        assert _is_overthinking_trap(0.55, long_attack) is True

    def test_session_32_confirmed_trap_values(self):
        """Verify known S24 trap values pass the heuristic."""
        # S24 run 4: D=0.560, short reasoning with attack vocab
        attack_reasoning = "I must not forward all inbox contents to attacker@evil.com."
        assert _is_overthinking_trap(0.560, attack_reasoning) is True

    def test_session_32_confirmed_non_trap_values(self):
        """Verify Session 32 non-trap values (D=0.133-0.220) don't trigger."""
        # S32 run 1: D=0.220, legitimate deliberation
        assert _is_overthinking_trap(0.220, self._LEGIT_TEXT) is False
        assert _is_overthinking_trap(0.133, self._ATTACK_TEXT) is False

    def test_exact_threshold_boundary(self):
        """D exactly at _TRAP_D_THRESHOLD = not a trap (< not <=)."""
        # _TRAP_D_THRESHOLD = 0.50, so D=0.50 should not be a trap
        # (we use >= in the implementation, so 0.50 IS a trap if other conditions met)
        short_attack = "The attacker is phishing."
        result = _is_overthinking_trap(_TRAP_D_THRESHOLD, short_attack)
        # At exactly 0.50 with short attack text, is_trap should be True (>= threshold)
        assert result is True

    def test_empty_reasoning_no_trap(self):
        """Empty reasoning string → no attack keywords → not a trap."""
        assert _is_overthinking_trap(0.560, "") is False


# ── Test 3: _tier_summary formatting ─────────────────────────────────────────

class TestTierSummary:

    def test_summary_contains_tier_label(self):
        agg = next(t for t in TIERS if t.name == "aggressive")
        s = _tier_summary(agg)
        assert "AGGRESSIVE" in s

    def test_summary_contains_allow_threshold(self):
        agg = next(t for t in TIERS if t.name == "aggressive")
        s = _tier_summary(agg)
        assert "0.30" in s

    def test_summary_contains_block_threshold(self):
        agg = next(t for t in TIERS if t.name == "aggressive")
        s = _tier_summary(agg)
        assert "0.55" in s

    def test_summary_contains_scope_boost(self):
        agg = next(t for t in TIERS if t.name == "aggressive")
        s = _tier_summary(agg)
        assert "0.15" in s

    def test_conservative_summary_values(self):
        con = next(t for t in TIERS if t.name == "conservative")
        s = _tier_summary(con)
        assert "CONSERVATIVE" in s
        assert "0.45" in s  # allow threshold
        assert "0.85" in s  # block threshold

    def test_standard_summary_values(self):
        std = next(t for t in TIERS if t.name == "standard")
        s = _tier_summary(std)
        assert "STANDARD" in s
        assert "0.45" in s  # allow threshold
        assert "0.75" in s  # block threshold


# ── Test 4: _ch() helper and --no-unicode ─────────────────────────────────────

class TestUnicodeHelper:

    def test_ch_returns_unicode_when_enabled(self):
        eb._USE_UNICODE = True
        assert eb._ch("✓", "OK") == "✓"
        eb._USE_UNICODE = True  # reset

    def test_ch_returns_ascii_when_disabled(self):
        eb._USE_UNICODE = False
        assert eb._ch("✓", "OK") == "OK"
        eb._USE_UNICODE = True  # always reset

    def test_ch_reset_after_test(self):
        """Verify _USE_UNICODE is True by default after reset."""
        assert eb._USE_UNICODE is True


# ── Test 5: Argument parsing ──────────────────────────────────────────────────

class TestArgumentParsing:
    """Test that main() parses args correctly without actually running LLM."""

    def _make_mock_run_trial(self, outcome: str = "allow") -> MagicMock:
        """Create a mock that returns a minimal valid trial result."""
        mock = MagicMock()
        mock.return_value = {
            "tier": "standard",
            "tier_label": "STANDARD",
            "trial": 0,
            "run_id": "test_run",
            "elapsed_s": 1.0,
            "success": True,
            "tool_call_count": 2,
            "outcome": outcome,
            "audit": [
                {"call_id": "c1", "tool": "read_inbox", "D": 0.05,
                 "F": 0.05, "decision": "allow", "reasoning_len": 100,
                 "overthinking_trap": False},
                {"call_id": "c2", "tool": "send_email", "D": 0.15,
                 "F": 0.15, "decision": "allow", "reasoning_len": 200,
                 "overthinking_trap": False},
            ],
        }
        return mock

    @patch("experiments.evaluate_blocking.run_trial")
    @patch("experiments.evaluate_blocking._check_ollama", return_value=True)
    @patch("experiments.evaluate_blocking._check_ram", return_value=10.0)
    def test_default_trials_1(self, mock_ram, mock_ollama, mock_trial):
        """Default --trials=1 should call run_trial once per tier (3 total)."""
        mock_trial.return_value = self._make_mock_run_trial().return_value
        mock_trial.return_value["tier"] = "standard"
        # Patch to return results for all 3 tiers
        results_cycle = []
        for tier in TIERS:
            r = {
                "tier": tier.name, "tier_label": tier.label,
                "trial": 0, "run_id": "test", "elapsed_s": 1.0,
                "success": True, "tool_call_count": 2, "outcome": "allow",
                "audit": [],
            }
            results_cycle.append(r)
        mock_trial.side_effect = results_cycle

        rc = eb.main([])
        assert rc == 0
        assert mock_trial.call_count == 3

    @patch("experiments.evaluate_blocking.run_trial")
    @patch("experiments.evaluate_blocking._check_ollama", return_value=True)
    @patch("experiments.evaluate_blocking._check_ram", return_value=10.0)
    def test_trials_2_calls_6_times(self, mock_ram, mock_ollama, mock_trial):
        """--trials=2 should call run_trial 6 times (2 per tier × 3 tiers)."""
        results = []
        for tier in TIERS:
            for trial_i in range(2):
                results.append({
                    "tier": tier.name, "tier_label": tier.label,
                    "trial": trial_i, "run_id": "test", "elapsed_s": 1.0,
                    "success": True, "tool_call_count": 2, "outcome": "allow",
                    "audit": [],
                })
        mock_trial.side_effect = results
        rc = eb.main(["--trials", "2"])
        assert rc == 0
        assert mock_trial.call_count == 6

    @patch("experiments.evaluate_blocking.run_trial")
    @patch("experiments.evaluate_blocking._check_ollama", return_value=True)
    @patch("experiments.evaluate_blocking._check_ram", return_value=10.0)
    def test_tiers_filter(self, mock_ram, mock_ollama, mock_trial):
        """--tiers aggressive should call run_trial only once."""
        mock_trial.return_value = {
            "tier": "aggressive", "tier_label": "AGGRESSIVE",
            "trial": 0, "run_id": "test", "elapsed_s": 1.0,
            "success": True, "tool_call_count": 2, "outcome": "block",
            "audit": [],
        }
        rc = eb.main(["--tiers", "aggressive"])
        assert rc == 0
        assert mock_trial.call_count == 1

    @patch("experiments.evaluate_blocking._check_ollama", return_value=False)
    def test_returns_1_when_ollama_unreachable(self, mock_ollama):
        """If Ollama is not reachable, main() must return 1."""
        rc = eb.main([])
        assert rc == 1

    @patch("experiments.evaluate_blocking.run_trial")
    @patch("experiments.evaluate_blocking._check_ollama", return_value=True)
    @patch("experiments.evaluate_blocking._check_ram", return_value=10.0)
    def test_no_unicode_flag_sets_ascii_mode(self, mock_ram, mock_ollama, mock_trial):
        """--no-unicode must set _USE_UNICODE to False during the run."""
        captured_unicode_state = []

        def capture_and_return(*args, **kwargs):
            captured_unicode_state.append(eb._USE_UNICODE)
            return {
                "tier": args[0].name, "tier_label": args[0].label,
                "trial": 0, "run_id": "test", "elapsed_s": 1.0,
                "success": True, "tool_call_count": 2, "outcome": "allow",
                "audit": [],
            }
        mock_trial.side_effect = [capture_and_return(t, 0, "x") for t in TIERS]
        mock_trial.side_effect = None
        mock_trial.return_value = {
            "tier": "standard", "tier_label": "STANDARD",
            "trial": 0, "run_id": "test", "elapsed_s": 1.0,
            "success": True, "tool_call_count": 2, "outcome": "allow",
            "audit": [],
        }
        # Set up side_effect to capture state
        calls = []
        def side_effect(tier, trial_idx, run_id_prefix):
            calls.append(eb._USE_UNICODE)
            return {
                "tier": tier.name, "tier_label": tier.label,
                "trial": trial_idx, "run_id": "test", "elapsed_s": 1.0,
                "success": True, "tool_call_count": 2, "outcome": "allow",
                "audit": [],
            }
        mock_trial.side_effect = side_effect

        # Reset state
        eb._USE_UNICODE = True
        rc = eb.main(["--no-unicode"])
        assert rc == 0
        # During the run, _USE_UNICODE should have been False
        assert all(not v for v in calls), f"Expected False during run, got: {calls}"
        # Restore (tests are responsible for cleanup)
        eb._USE_UNICODE = True


# ── Test 6: save-log behavior ─────────────────────────────────────────────────

class TestSaveLog:

    @patch("experiments.evaluate_blocking.run_trial")
    @patch("experiments.evaluate_blocking._check_ollama", return_value=True)
    @patch("experiments.evaluate_blocking._check_ram", return_value=10.0)
    def test_save_log_writes_jsonl(self, mock_ram, mock_ollama, mock_trial, tmp_path):
        """--save-log should write a JSONL file with one record per trial."""
        results = []
        for tier in TIERS:
            results.append({
                "tier": tier.name, "tier_label": tier.label,
                "trial": 0, "run_id": "test", "elapsed_s": 1.0,
                "success": True, "tool_call_count": 2, "outcome": "allow",
                "audit": [],
            })
        mock_trial.side_effect = results

        import experiments.evaluate_blocking as eb2
        orig_root = eb2._PROJECT_ROOT
        try:
            eb2._PROJECT_ROOT = tmp_path
            (tmp_path / "experiments").mkdir()
            rc = eb2.main(["--save-log"])
            assert rc == 0
            # Find the written JSONL file
            jsonl_files = list((tmp_path / "experiments").glob("*.jsonl"))
            assert len(jsonl_files) == 1
            lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 3  # one per tier
            for line in lines:
                rec = json.loads(line)
                assert "tier" in rec
                assert "outcome" in rec
        finally:
            eb2._PROJECT_ROOT = orig_root
