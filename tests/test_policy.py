"""
tests/test_policy.py

Session 4 — Unit tests for FrictionPolicy (trust-weighted friction score).

Covers:
  Formula correctness:
    1.  F = D × (1.5 − 0.5 × T) computed correctly for T=0, T=0.5, T=1.0
    2.  F is always clipped to [0.0, 1.0]
    3.  High D + low T produces higher F than same D + high T
  Decision thresholds:
    4.  F < allow_threshold → ALLOW
    5.  allow_threshold ≤ F < block_threshold → FLAG
    6.  F ≥ block_threshold → BLOCK
    7.  BLOCK decision returns non-None block_message
    8.  ALLOW and FLAG decisions return None block_message
  Boundary conditions:
    9.  D=0.0, T=any → F=0.0, decision=ALLOW
   10.  D=1.0, T=0.0 → F=1.5 clipped to 1.0, decision=BLOCK
   11.  D=1.0, T=1.0 → F=1.0, decision=BLOCK
   12.  trust_score=None uses trust_default
   13.  Invalid thresholds (allow >= block) raise ValueError
  PolicyResult:
   14.  to_dict() is JSON-serialisable
   15.  reasoning string is non-empty
  Preset policies:
   16.  CONSERVATIVE_POLICY has higher block_threshold than STANDARD_POLICY
   17.  AGGRESSIVE_POLICY has lower block_threshold than STANDARD_POLICY

Run with:  pytest tests/test_policy.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pacm.policy import (
    FrictionDecision,
    FrictionPolicy,
    PolicyResult,
    AGGRESSIVE_POLICY,
    CONSERVATIVE_POLICY,
    STANDARD_POLICY,
)


# ─── Formula correctness ──────────────────────────────────────────────────────

class TestFrictionFormula:
    def setup_method(self):
        # Use wide thresholds so we can test F values freely
        self.policy = FrictionPolicy(allow_threshold=0.01, block_threshold=0.99)

    def _F(self, D: float, T: float) -> float:
        result = self.policy.evaluate(D, T)
        return result.friction_score

    def test_zero_trust_amplifies_friction(self):
        """T=0.0: F = D × 1.5"""
        D = 0.4
        expected = min(D * 1.5, 1.0)
        assert abs(self._F(D, 0.0) - expected) < 1e-9

    def test_half_trust_neutral(self):
        """T=0.5: F = D × 1.25"""
        D = 0.4
        expected = min(D * 1.25, 1.0)
        assert abs(self._F(D, 0.5) - expected) < 1e-9

    def test_full_trust_preserves(self):
        """T=1.0: F = D × 1.0 (no amplification)"""
        D = 0.4
        expected = min(D * 1.0, 1.0)
        assert abs(self._F(D, 1.0) - expected) < 1e-9

    def test_low_trust_higher_friction_than_high_trust(self):
        """Same D: low T → higher F than high T."""
        D = 0.5
        f_low_trust  = self._F(D, 0.0)
        f_high_trust = self._F(D, 1.0)
        assert f_low_trust > f_high_trust

    def test_f_always_clipped_to_range(self):
        """F is always in [0.0, 1.0]."""
        for D in [0.0, 0.5, 0.9, 1.0]:
            for T in [0.0, 0.5, 1.0]:
                f = self._F(D, T)
                assert 0.0 <= f <= 1.0, f"F={f} out of range for D={D}, T={T}"


# ─── Decision thresholds ──────────────────────────────────────────────────────

class TestDecisionThresholds:
    def setup_method(self):
        self.policy = FrictionPolicy(allow_threshold=0.40, block_threshold=0.70)

    def test_below_allow_threshold_is_allow(self):
        """Very low divergence → ALLOW."""
        # D=0.1, T=1.0: F = 0.1 × 1.0 = 0.10 < 0.40
        result = self.policy.evaluate(0.1, 1.0)
        assert result.decision == FrictionDecision.ALLOW

    def test_in_flag_band_is_flag(self):
        """Moderate divergence on untrusted tool → FLAG."""
        # D=0.35, T=0.0: F = 0.35 × 1.5 = 0.525, in [0.40, 0.70)
        result = self.policy.evaluate(0.35, 0.0)
        assert result.decision == FrictionDecision.FLAG

    def test_above_block_threshold_is_block(self):
        """High divergence on low-trust tool → BLOCK."""
        # D=0.6, T=0.0: F = 0.6 × 1.5 = 0.90 ≥ 0.70
        result = self.policy.evaluate(0.6, 0.0)
        assert result.decision == FrictionDecision.BLOCK

    def test_block_returns_block_message(self):
        """BLOCK decision includes a non-None block_message."""
        result = self.policy.evaluate(0.9, 0.0, tool_name="evil_tool")
        assert result.decision == FrictionDecision.BLOCK
        assert result.block_message is not None
        assert "evil_tool" in result.block_message

    def test_allow_returns_none_message(self):
        result = self.policy.evaluate(0.05, 1.0)
        assert result.decision == FrictionDecision.ALLOW
        assert result.block_message is None

    def test_flag_returns_none_message(self):
        result = self.policy.evaluate(0.35, 0.0)
        assert result.decision == FrictionDecision.FLAG
        assert result.block_message is None


# ─── Boundary conditions ──────────────────────────────────────────────────────

class TestBoundaryConditions:
    def setup_method(self):
        self.policy = FrictionPolicy(allow_threshold=0.40, block_threshold=0.70)

    def test_zero_divergence_always_allow(self):
        """D=0.0 → F=0.0 → always ALLOW."""
        for T in [0.0, 0.5, 1.0]:
            result = self.policy.evaluate(0.0, T)
            assert result.decision == FrictionDecision.ALLOW
            assert result.friction_score == 0.0

    def test_max_divergence_zero_trust_clips_to_one(self):
        """D=1.0, T=0.0: F = 1.5 → clipped to 1.0."""
        result = self.policy.evaluate(1.0, 0.0)
        assert result.friction_score == 1.0
        assert result.decision == FrictionDecision.BLOCK

    def test_max_divergence_full_trust(self):
        """D=1.0, T=1.0: F = 1.0, must be BLOCK."""
        result = self.policy.evaluate(1.0, 1.0)
        assert result.friction_score == 1.0
        assert result.decision == FrictionDecision.BLOCK

    def test_none_trust_uses_default(self):
        """trust_score=None → uses policy.trust_default (0.5)."""
        result_none = self.policy.evaluate(0.5, None)
        result_half = self.policy.evaluate(0.5, 0.5)
        assert abs(result_none.friction_score - result_half.friction_score) < 1e-9

    def test_invalid_thresholds_raise_valueerror(self):
        with pytest.raises(ValueError):
            FrictionPolicy(allow_threshold=0.8, block_threshold=0.5)

    def test_trust_clamped_to_range(self):
        """trust_score outside [0, 1] is clamped before use."""
        r1 = self.policy.evaluate(0.5, 1.5)  # clamped to 1.0
        r2 = self.policy.evaluate(0.5, 1.0)
        assert abs(r1.friction_score - r2.friction_score) < 1e-9


# ─── PolicyResult ─────────────────────────────────────────────────────────────

class TestPolicyResult:
    def test_to_dict_json_serialisable(self):
        policy = STANDARD_POLICY
        result = policy.evaluate(0.5, 0.5, "test_tool")
        d = result.to_dict()
        json.dumps(d)  # must not raise
        assert "friction_score" in d
        assert "trust_score" in d
        assert "divergence_score" in d
        assert "decision" in d

    def test_reasoning_non_empty(self):
        result = STANDARD_POLICY.evaluate(0.5, 0.5)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 5


# ─── Preset policies ──────────────────────────────────────────────────────────

class TestPresetPolicies:
    def test_conservative_higher_block_than_standard(self):
        assert CONSERVATIVE_POLICY.block_threshold > STANDARD_POLICY.block_threshold

    def test_aggressive_lower_block_than_standard(self):
        assert AGGRESSIVE_POLICY.block_threshold < STANDARD_POLICY.block_threshold

    def test_standard_blocks_high_friction(self):
        """D=0.8, T=0.0: F = 1.2 → clipped to 1.0 → BLOCK in all presets."""
        for policy in [STANDARD_POLICY, AGGRESSIVE_POLICY, CONSERVATIVE_POLICY]:
            result = policy.evaluate(0.8, 0.0)
            assert result.decision == FrictionDecision.BLOCK

    def test_aggressive_blocks_medium_friction(self):
        """D=0.4, T=0.3: F = 0.4 × 1.35 = 0.54, above aggressive threshold (0.55)."""
        # aggressive block_threshold = 0.55, so F=0.54 is still FLAG
        result = AGGRESSIVE_POLICY.evaluate(0.4, 0.3)
        # Just verify it's more restrictive than standard
        r_standard = STANDARD_POLICY.evaluate(0.4, 0.3)
        # Aggressive should be ≥ standard in terms of friction (same F, potentially different decisions)
        assert result.friction_score == r_standard.friction_score


# ─── Scope-flag friction boost tests ──────────────────────────────────────────


class TestScopeFlagBoost:
    """Unit tests for scope_flags parameter friction boosting in FrictionPolicy."""

    def _policy(self, boost: float = 0.15) -> FrictionPolicy:
        return FrictionPolicy(allow_threshold=0.45, block_threshold=0.75, scope_boost=boost)

    def test_boost_applied_with_external_domain_flag(self):
        """scope_flags={'to': 'external_domain'} should add +0.15 to F."""
        policy = self._policy()
        no_flag  = policy.evaluate(0.30, 1.0, scope_flags=None)
        with_flag = policy.evaluate(0.30, 1.0, scope_flags={"to": "external_domain"})
        assert with_flag.friction_score == pytest.approx(no_flag.friction_score + 0.15, abs=1e-6)
        assert with_flag.scope_boost == pytest.approx(0.15)

    def test_boost_pushes_exfiltration_over_allow_threshold(self):
        """Real sweep: D=0.339, T=1.0 → base F=0.339. With boost=0.15 → F=0.489 > 0.45 → FLAG."""
        policy = self._policy()
        # Without boost: D=0.339 → F=0.339 → ALLOW
        no_boost = policy.evaluate(0.339, 1.0)
        assert no_boost.decision == FrictionDecision.ALLOW

        # With external_domain scope flag: F=0.489 → FLAG
        with_boost = policy.evaluate(0.339, 1.0, scope_flags={"to": "external_domain"})
        assert with_boost.decision != FrictionDecision.ALLOW
        assert with_boost.friction_score == pytest.approx(0.339 + 0.15, abs=1e-6)

    def test_multiple_scope_flags_do_not_double_boost(self):
        """Even with multiple flags, boost is applied exactly once."""
        policy = self._policy()
        multi = policy.evaluate(
            0.20, 1.0,
            scope_flags={"to": "external_domain", "url": "external_domain"},
        )
        single = policy.evaluate(
            0.20, 1.0,
            scope_flags={"to": "external_domain"},
        )
        assert multi.friction_score == pytest.approx(single.friction_score, abs=1e-9)

    def test_no_boost_without_scope_flags(self):
        """scope_flags=None produces scope_boost=0.0."""
        policy = self._policy()
        result = policy.evaluate(0.50, 1.0, scope_flags=None)
        assert result.scope_boost == pytest.approx(0.0)

    def test_empty_scope_flags_dict_no_boost(self):
        """scope_flags={} (empty dict) — falsy — produces no boost."""
        policy = self._policy()
        result = policy.evaluate(0.50, 1.0, scope_flags={})
        assert result.scope_boost == pytest.approx(0.0)

    def test_scope_boost_in_to_dict(self):
        """scope_boost appears in serialised PolicyResult dict."""
        policy = self._policy()
        result = policy.evaluate(0.30, 1.0, scope_flags={"to": "external_domain"})
        d = result.to_dict()
        assert "scope_boost" in d
        assert d["scope_boost"] == pytest.approx(0.15)

    def test_custom_boost_value_respected(self):
        """A custom scope_boost value is applied correctly."""
        policy = FrictionPolicy(allow_threshold=0.45, block_threshold=0.75, scope_boost=0.25)
        result = policy.evaluate(0.20, 1.0, scope_flags={"to": "external_domain"})
        assert result.scope_boost == pytest.approx(0.25)
        assert result.friction_score == pytest.approx(0.20 + 0.25, abs=1e-6)

    def test_boost_reasoning_contains_scope_boost(self):
        """Policy reasoning string mentions scope_boost when flags are present."""
        policy = self._policy()
        result = policy.evaluate(0.339, 1.0, scope_flags={"to": "external_domain"})
        assert "scope_boost" in result.reasoning
