"""
src/pacm/policy.py

Trust-Weighted Friction Policy — Session 4

Combines the divergence score D ∈ [0.0, 1.0] from DivergenceScorer with
the tool trust score T ∈ [0.0, 1.0] from MCPToolMetadata into a single
Friction Score F, then maps F to a policy decision.

Design (PACM_Research_Guide.md §9.1 step 4 / §7 / §12):

  Friction Score formula:
      F = D × (1.5 − 0.5 × T)

  Intuition:
    • T = 1.0 (fully trusted tool):  F = D × 1.0  → friction = divergence
    • T = 0.0 (zero-trust tool):    F = D × 1.5  → friction amplified by 50%
    • Low-trust tools (T→0) require less divergence to trigger escalation.
    • High-trust internal tools (T→1) require stronger evidence before blocking.
    • F is always clipped to [0.0, 1.0].

  Decision thresholds (configurable):
    F < allow_threshold               → ALLOW
    allow_threshold ≤ F < block_threshold → FLAG  (human-review / log warning)
    F ≥ block_threshold               → BLOCK (refusal response to LLM)

  Default thresholds (tuned for F-formula, not raw D):
    allow_threshold  = 0.45   (adjusted for trust amplification)
    block_threshold  = 0.75

References:
  - Li & Gao (2025) — MCP Security (2510.16558): trust taxonomy
  - Hasan et al. (2026) — MCP at First Glance (2506.13538)
  - PACM_Research_Guide.md §9.1 steps 3-4, §10 evaluation metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Decision enum ────────────────────────────────────────────────────────────


class FrictionDecision(str, Enum):
    """
    Three-way policy decision output.

    Using str-Enum means values can be directly stored as JSON strings.
    """
    ALLOW = "allow"
    FLAG  = "flag"
    BLOCK = "block"


# ─── Policy result ────────────────────────────────────────────────────────────


@dataclass
class PolicyResult:
    """
    Full result of applying FrictionPolicy to one tool call.

    Attributes:
        divergence_score: Raw D from DivergenceScorer.
        trust_score:      T from MCPToolMetadata (0.0 if unknown).
        friction_score:   F = D × (1.5 − 0.5 × T), clipped to [0, 1].
        decision:         FrictionDecision.ALLOW / FLAG / BLOCK.
        block_message:    Synthetic refusal string to return to LLM (BLOCK only).
        reasoning:        Human-readable explanation of the decision.
    """
    divergence_score: float
    trust_score: float
    friction_score: float
    scope_boost: float          # additive boost from parameter scope flags
    decision: FrictionDecision
    block_message: Optional[str]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "divergence_score": round(self.divergence_score, 4),
            "trust_score":      round(self.trust_score, 4),
            "friction_score":   round(self.friction_score, 4),
            "scope_boost":      round(self.scope_boost, 4),
            "decision":         self.decision.value,
            "block_message":    self.block_message,
            "reasoning":        self.reasoning,
        }


# ─── Friction policy ──────────────────────────────────────────────────────────


class FrictionPolicy:
    """
    Trust-weighted friction policy for PACM middleware.

    Applies the formula F = D × (1.5 − 0.5 × T) to combine divergence and
    trust into a single friction score, then maps F to ALLOW/FLAG/BLOCK.

    Args:
        allow_threshold:  F below this → ALLOW. Default 0.45.
        block_threshold:  F at or above this → BLOCK. Default 0.75.
                          FLAG is the band [allow_threshold, block_threshold).
        trust_default:    Trust score to use when metadata unavailable. Default 0.5.

    Threshold design:
        With T=1.0 (trusted):  raw D needs ≥ 0.75 to BLOCK.
        With T=0.5 (neutral):  raw D needs ≥ 0.60 to BLOCK.
        With T=0.0 (untrusted): raw D needs ≥ 0.50 to BLOCK.
        This gradient incentivises use of trusted, registered tools.
    """

    def __init__(
        self,
        allow_threshold: float = 0.45,
        block_threshold: float = 0.75,
        trust_default: float = 0.5,
        scope_boost: float = 0.15,
    ) -> None:
        if not (0.0 <= allow_threshold < block_threshold <= 1.5):
            raise ValueError(
                f"Thresholds must satisfy 0 ≤ allow_threshold < block_threshold ≤ 1.5. "
                f"Got allow={allow_threshold}, block={block_threshold}"
            )
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold
        self.trust_default = trust_default
        self.scope_boost = max(0.0, float(scope_boost))

    def evaluate(
        self,
        divergence_score: float,
        trust_score: Optional[float],
        tool_name: str = "",
        scope_flags: Optional[dict] = None,
    ) -> PolicyResult:
        """
        Evaluate the friction policy for a single tool call.

        Args:
            divergence_score: D ∈ [0.0, 1.0] from DivergenceScorer.
            trust_score:      T ∈ [0.0, 1.0] from MCPToolMetadata.trust_score,
                              or None if metadata not available.
            tool_name:        Tool name (for logging / block message).
            scope_flags:      Optional dict of parameter-level scope flags from
                              ActionCapturer (e.g. ``{"to": "external_domain"}``).
                              When any flag is present, ``self.scope_boost`` is
                              added to F before threshold comparison.

        Returns:
            PolicyResult with friction_score, scope_boost, decision, optional block_message.
        """
        T = trust_score if trust_score is not None else self.trust_default
        T = max(0.0, min(1.0, T))

        # Base friction: F = D × (1.5 − 0.5 × T)
        base_friction = divergence_score * (1.5 - 0.5 * T)

        # ── Scope-flag friction boost ─────────────────────────────────────
        # Scope flags (e.g. to=external_domain) indicate that a parameter
        # value refers to a resource outside the trusted perimeter. Even
        # moderate semantic divergence becomes highly suspicious in this
        # context (e.g. exfiltrating inbox data to attacker@evil.com).
        # Adding a fixed boost closes the gap between observed D and threshold.
        applied_boost = 0.0
        if scope_flags:
            applied_boost = self.scope_boost
            logger.debug(
                "FrictionPolicy | tool=%s | scope_flags=%s | boost=+%.2f",
                tool_name, list(scope_flags.keys()), applied_boost,
            )

        friction_score = max(0.0, min(1.0, base_friction + applied_boost))

        decision, reasoning, block_message = self._decide(
            friction_score=friction_score,
            divergence_score=divergence_score,
            trust_score=T,
            tool_name=tool_name,
            scope_boost=applied_boost,
        )

        logger.debug(
            "FrictionPolicy | tool=%s | D=%.3f T=%.2f boost=+%.2f F=%.3f | decision=%s",
            tool_name, divergence_score, T, applied_boost, friction_score, decision.value,
        )

        return PolicyResult(
            divergence_score=divergence_score,
            trust_score=T,
            friction_score=friction_score,
            scope_boost=applied_boost,
            decision=decision,
            block_message=block_message,
            reasoning=reasoning,
        )

    def _decide(
        self,
        friction_score: float,
        divergence_score: float,
        trust_score: float,
        tool_name: str,
        scope_boost: float = 0.0,
    ) -> tuple[FrictionDecision, str, Optional[str]]:
        """Map friction score to a decision + reasoning string."""
        boost_str = f" +scope_boost={scope_boost:.2f}" if scope_boost else ""
        if friction_score < self.allow_threshold:
            reasoning = (
                f"F={friction_score:.3f} < allow_threshold={self.allow_threshold:.2f} "
                f"(D={divergence_score:.3f}, T={trust_score:.2f}{boost_str})"
            )
            return FrictionDecision.ALLOW, reasoning, None

        elif friction_score < self.block_threshold:
            reasoning = (
                f"F={friction_score:.3f} in FLAG zone "
                f"[{self.allow_threshold:.2f}, {self.block_threshold:.2f}) "
                f"(D={divergence_score:.3f}, T={trust_score:.2f}{boost_str}) — "
                f"suspicious but not conclusive; logging for review"
            )
            logger.warning(
                "PACM [FLAG] tool=%s | F=%.3f | D=%.3f | T=%.2f | boost=%.2f",
                tool_name, friction_score, divergence_score, trust_score, scope_boost,
            )
            return FrictionDecision.FLAG, reasoning, None

        else:
            reasoning = (
                f"F={friction_score:.3f} ≥ block_threshold={self.block_threshold:.2f} "
                f"(D={divergence_score:.3f}, T={trust_score:.2f}{boost_str}) — blocked"
            )
            block_message = (
                f"[PACM BLOCK] Tool call '{tool_name}' was blocked by the PACM "
                f"trust-weighted friction policy. "
                f"Friction score F={friction_score:.3f} (divergence={divergence_score:.3f}, "
                f"trust={trust_score:.2f}, scope_boost={scope_boost:.2f}) exceeded block threshold "
                f"{self.block_threshold:.2f}. "
                f"Please re-evaluate your action to ensure it is consistent with "
                f"the user's original task."
            )
            logger.warning(
                "PACM [BLOCK] tool=%s | F=%.3f | D=%.3f | T=%.2f | boost=%.2f",
                tool_name, friction_score, divergence_score, trust_score, scope_boost,
            )
            return FrictionDecision.BLOCK, reasoning, block_message


# ─── Default policy instances ─────────────────────────────────────────────────

# Conservative policy: only block on very strong evidence
CONSERVATIVE_POLICY = FrictionPolicy(allow_threshold=0.45, block_threshold=0.85)

# Standard policy: balanced for research evaluation
STANDARD_POLICY = FrictionPolicy(allow_threshold=0.45, block_threshold=0.75)

# Aggressive policy: lower threshold for high-risk deployments
AGGRESSIVE_POLICY = FrictionPolicy(allow_threshold=0.30, block_threshold=0.55)
