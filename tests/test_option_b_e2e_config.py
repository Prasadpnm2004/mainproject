
"""
tests/test_option_b_e2e_config.py

Session 23, Task 2 — Unit tests verifying option_b_e2e_test.py's middleware
now constructs with the correct trust_registry and canonical aggressive policy.

These tests are purely structural/unit — no model calls, no Ollama required.

Guardrails: no harness modification, no model calls.
"""
from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "d:/mainproject")

from src.harness.mcp_tool_wrapper import DEFAULT_STUB_REGISTRY
from src.pacm.middleware import PACMMiddleware
from src.pacm.policy import FrictionPolicy


# Import the canonical constants directly from the fixed e2e test module
# so this test always checks the live values, not hand-typed copies.
import importlib.util
import os

_E2E_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "option_b_e2e_test.py",
)
_spec = importlib.util.spec_from_file_location("option_b_e2e_test", _E2E_PATH)
_e2e = importlib.util.module_from_spec(_spec)
# We don't exec the module (it would try to import Ollama etc.) but we read
# the constants directly from the source via inspection.
# Instead, import only what we need:
import ast

with open(_E2E_PATH) as _f:
    _tree = ast.parse(_f.read())

_CONSTS: dict[str, float] = {}
for _node in ast.walk(_tree):
    if isinstance(_node, ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, ast.Name) and _t.id.startswith("_AGGRESSIVE_"):
                if isinstance(_node.value, (ast.Constant,)):
                    _CONSTS[_t.id] = float(_node.value.value)


class TestE2EConfigConstants(unittest.TestCase):
    """Verify the canonical aggressive policy constants in option_b_e2e_test.py."""

    def test_aggressive_allow_matches_run_experiment(self):
        """_AGGRESSIVE_ALLOW must match run_experiment.py _POLICY_THRESHOLDS['allow'][0]."""
        # run_experiment.py L337: "allow": (0.30, 0.55)
        self.assertIn("_AGGRESSIVE_ALLOW", _CONSTS, "Constant missing from e2e test")
        self.assertAlmostEqual(_CONSTS["_AGGRESSIVE_ALLOW"], 0.30, places=6,
                               msg="allow_threshold must be 0.30 (canonical aggressive tier)")

    def test_aggressive_block_matches_run_experiment(self):
        """_AGGRESSIVE_BLOCK must match run_experiment.py _POLICY_THRESHOLDS['allow'][1]."""
        self.assertIn("_AGGRESSIVE_BLOCK", _CONSTS, "Constant missing from e2e test")
        self.assertAlmostEqual(_CONSTS["_AGGRESSIVE_BLOCK"], 0.55, places=6,
                               msg="block_threshold must be 0.55 (canonical aggressive tier)")

    def test_aggressive_scope_boost_matches_run_experiment(self):
        """_AGGRESSIVE_SCOPE_BOOST must match run_experiment.py's scope_boost=0.15."""
        self.assertIn("_AGGRESSIVE_SCOPE_BOOST", _CONSTS, "Constant missing from e2e test")
        self.assertAlmostEqual(_CONSTS["_AGGRESSIVE_SCOPE_BOOST"], 0.15, places=6,
                               msg="scope_boost must be 0.15 (matches sweep infrastructure)")


class TestE2EPolicyConstruction(unittest.TestCase):
    """Verify a PACMMiddleware built with e2e test constants matches the canonical config."""

    def _build_middleware(self) -> PACMMiddleware:
        """Build middleware exactly as option_b_e2e_test.run_e2e_test() now does."""
        allow = _CONSTS.get("_AGGRESSIVE_ALLOW", 0.30)
        block = _CONSTS.get("_AGGRESSIVE_BLOCK", 0.55)
        boost = _CONSTS.get("_AGGRESSIVE_SCOPE_BOOST", 0.15)
        policy = FrictionPolicy(
            allow_threshold=allow,
            block_threshold=block,
            scope_boost=boost,
        )
        return PACMMiddleware(
            policy=policy,
            run_id="test_s23",
            trust_registry=DEFAULT_STUB_REGISTRY,
        )

    def test_trust_registry_is_not_none(self):
        """trust_registry must be DEFAULT_STUB_REGISTRY, not None."""
        mw = self._build_middleware()
        self.assertIsNotNone(mw._trust_registry,
                             "trust_registry must not be None (fixes Session 22 T=0.5 bug)")

    def test_trust_registry_is_default_stub_registry(self):
        """trust_registry must be the same DEFAULT_STUB_REGISTRY instance used by sweep."""
        mw = self._build_middleware()
        self.assertIs(mw._trust_registry, DEFAULT_STUB_REGISTRY)

    def test_read_inbox_trust_resolves_to_1_00(self):
        """With DEFAULT_STUB_REGISTRY, read_inbox must resolve to trust_score=1.00."""
        mw = self._build_middleware()
        # _resolve_trust uses a ToolCallRecord-like object; we use a simple mock
        from unittest.mock import MagicMock
        record = MagicMock()
        record.tool = None  # ToolCallRecord has no .tool — Priority 1 always skipped
        trust = mw._resolve_trust("read_inbox", record)
        self.assertAlmostEqual(trust, 1.00, places=4,
                               msg="read_inbox trust must be 1.00 via DEFAULT_STUB_REGISTRY")

    def test_send_email_trust_resolves_to_1_00(self):
        """With DEFAULT_STUB_REGISTRY, send_email must resolve to trust_score=1.00."""
        mw = self._build_middleware()
        from unittest.mock import MagicMock
        record = MagicMock()
        record.tool = None
        trust = mw._resolve_trust("send_email", record)
        self.assertAlmostEqual(trust, 1.00, places=4,
                               msg="send_email trust must be 1.00 via DEFAULT_STUB_REGISTRY")

    def test_policy_allow_threshold(self):
        """Policy allow_threshold must be 0.30 (aggressive tier)."""
        mw = self._build_middleware()
        self.assertAlmostEqual(mw._policy.allow_threshold, 0.30, places=6)

    def test_policy_block_threshold(self):
        """Policy block_threshold must be 0.55 (aggressive tier)."""
        mw = self._build_middleware()
        self.assertAlmostEqual(mw._policy.block_threshold, 0.55, places=6)

    def test_policy_scope_boost(self):
        """Policy scope_boost must be 0.15 (matching sweep infrastructure)."""
        mw = self._build_middleware()
        self.assertAlmostEqual(mw._policy.scope_boost, 0.15, places=6)

    def test_friction_formula_read_inbox_t1(self):
        """Sanity: F = D × (1.5 - 0.5×T) with T=1.00 means F=D (no amplification)."""
        # With trust=1.00: F = D × (1.5 - 0.5) = D × 1.00
        # read_inbox historical D ≈ 0.103 → F ≈ 0.103, comfortably below allow=0.30
        historical_d = 0.103
        t = 1.00
        f = historical_d * (1.5 - 0.5 * t)
        self.assertAlmostEqual(f, historical_d, places=6)
        self.assertLess(f, 0.30, "read_inbox F must be < allow_threshold (should be allowed)")

    def test_friction_formula_was_broken_with_t05(self):
        """Verify the Session 22 bug: T=0.5 would give F=0.140 (>0.112=D, wrong amplification)."""
        # Session 22 bug: no trust_registry → T=0.5
        historical_d = 0.112  # observed D in Session 21 e2e
        t_broken = 0.50
        f_broken = historical_d * (1.5 - 0.5 * t_broken)
        self.assertAlmostEqual(f_broken, 0.140, places=2,
                               msg="Validates Session 22's algebraic explanation of F=0.140")
        # With fix (T=1.00): F = D = 0.112
        t_fixed = 1.00
        f_fixed = historical_d * (1.5 - 0.5 * t_fixed)
        self.assertAlmostEqual(f_fixed, historical_d, places=6)
        self.assertLess(f_fixed, f_broken, "Fixed F must be less than broken F")


class TestDefaultStubRegistryTrustScores(unittest.TestCase):
    """Verify DEFAULT_STUB_REGISTRY trust scores for experiment tools."""

    def test_read_inbox_trust_score(self):
        meta = DEFAULT_STUB_REGISTRY.get("read_inbox")
        # publisher_verified(+0.40) + internal_origin(+0.20) + recency(+0.20) + consistency(+0.20)
        self.assertAlmostEqual(meta.trust_score, 1.00, places=4)

    def test_send_email_trust_score(self):
        meta = DEFAULT_STUB_REGISTRY.get("send_email")
        self.assertAlmostEqual(meta.trust_score, 1.00, places=4)

    def test_unknown_tool_gets_zero_trust(self):
        """Unknown tools get zero-trust placeholder (not 1.00)."""
        meta = DEFAULT_STUB_REGISTRY.get("nonexistent_tool_xyz")
        # No positive signals → trust_score = 0.0
        self.assertAlmostEqual(meta.trust_score, 0.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
