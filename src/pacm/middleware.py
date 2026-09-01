"""
src/pacm/middleware.py

PACM Prototype Middleware — Session 3 (updated Session 4)

Connects IntentExtractor, ActionCapturer, DivergenceScorer, and FrictionPolicy
into a callable PreToolHook that plugs directly into AgentHarness or
CustomLoopHarness via the existing hook interface.

Session 4 changes:
  - Replaced raw divergence-threshold comparisons with FrictionPolicy.evaluate().
  - PACMAuditEntry now records trust_score and friction_score per call.
  - PACMMiddleware accepts a FrictionPolicy instance (or builds a default).
  - Optionally reads trust_score from the ToolCallRecord's wrapped tool
    metadata when available (mcp_metadata.trust_score).

Background (PACM_Research_Guide.md §9.1):
  Step 2: Extract DeclaredIntent from reasoning_before (IntentExtractor).
  Step 3: Capture CapturedAction from tool_name + tool_args (ActionCapturer).
  Step 4: Score divergence D (DivergenceScorer).
  Step 4b: Apply trust-weighted friction policy F = D × (1.5 − 0.5 × T).
  Policy: ALLOW / FLAG / BLOCK based on friction score F.

Hook interface (PreToolHook):
  Callable[[ToolCallRecord], Optional[str]]
    Return None  → allow the tool call.
    Return str   → block; the str is the synthetic tool result shown to LLM.

Usage:
    from src.pacm.middleware import PACMMiddleware
    from src.pacm.policy import STANDARD_POLICY
    from src.harness.langgraph_agent import AgentHarness, HarnessConfig

    middleware = PACMMiddleware(policy=STANDARD_POLICY)
    harness = AgentHarness(
        llm=llm, tools=tools, config=config, pre_tool_hook=middleware
    )
    result = harness.run("Check inbox and reply to urgent emails.")
    print(middleware.export_audit_log())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from src.pacm.action_capture import ActionCapturer, CapturedAction
from src.pacm.extractor import DeclaredIntent, IntentExtractor
from src.pacm.policy import FrictionDecision, FrictionPolicy, PolicyResult, STANDARD_POLICY
from src.pacm.scorer import DivergenceScorer

logger = logging.getLogger(__name__)

# Kept for backward-compat with Session 3 tests that reference PolicyMode literal
PolicyMode = Literal["allow", "flag", "block"]


# ─── Audit log entry ──────────────────────────────────────────────────────────


@dataclass
class PACMAuditEntry:
    """
    One audit log record per tool call evaluated by PACMMiddleware.

    Session 4 additions: trust_score, friction_score, policy_reasoning.
    """

    run_id: str
    call_id: str
    tool_name: str
    divergence_score: float
    trust_score: float              # NEW Session 4
    friction_score: float           # NEW Session 4
    threshold: float                # kept for backward-compat (= policy.allow_threshold)
    policy_mode: str                # legacy field; kept for JSONL schema continuity
    decision: str                   # "allow" | "flag" | "block"
    intent: Optional[DeclaredIntent]
    action: Optional[CapturedAction]
    scope_flags: dict[str, str]
    blocked_reason: Optional[str]
    policy_reasoning: str           # NEW Session 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "divergence_score": round(self.divergence_score, 4),
            "trust_score": round(self.trust_score, 4),
            "friction_score": round(self.friction_score, 4),
            "threshold": self.threshold,
            "policy_mode": self.policy_mode,
            "decision": self.decision,
            "intent": self.intent.to_dict() if self.intent else None,
            "action": self.action.to_dict() if self.action else None,
            "scope_flags": self.scope_flags,
            "blocked_reason": self.blocked_reason,
            "policy_reasoning": self.policy_reasoning,
        }


# ─── Middleware ───────────────────────────────────────────────────────────────


class PACMMiddleware:
    """
    PACM prototype middleware — callable PreToolHook.

    Session 4: now driven by FrictionPolicy (trust-weighted F score) instead
    of a raw divergence threshold. Backward-compatible with Session 3 tests:
    the policy_mode / divergence_threshold constructor args still work by
    constructing a FrictionPolicy with matching thresholds.

    Args:
        policy:              FrictionPolicy instance. If None, uses STANDARD_POLICY.
        divergence_threshold: Legacy arg (Session 3 compat). If policy is None,
                             uses this as both allow and block threshold proxy.
        policy_mode:         Legacy arg. "allow"/"flag"/"block" overrides policy
                             when using the legacy code path.
        scorer_alpha:        Semantic vs. scope weight (passed to DivergenceScorer).
        prefer_transformers: Use sentence-transformers if available.
        run_id:              Optional run identifier for audit log correlation.
        trust_registry:      Optional MCPToolRegistry for per-tool trust lookup.
    """

    def __init__(
        self,
        policy: Optional[FrictionPolicy] = None,
        divergence_threshold: float = 0.60,
        policy_mode: PolicyMode = "flag",
        scorer_alpha: float = 0.70,
        prefer_transformers: bool = True,
        run_id: str = "",
        trust_registry: Any = None,
    ) -> None:
        self.run_id = run_id
        self._trust_registry = trust_registry

        # ── Policy setup ──────────────────────────────────────────────────────
        # If an explicit FrictionPolicy is provided, use it.
        # Otherwise, preserve Session 3 behavior via a compatible policy.
        if policy is not None:
            self._policy = policy
            self._legacy_mode: Optional[str] = None
        else:
            # Build a FrictionPolicy where divergence_threshold maps to block_threshold.
            # allow_threshold is set to 80% of block_threshold, with a floor of 0.001
            # to ensure allow_threshold < block_threshold always holds.
            block_thr = max(divergence_threshold, 0.001)
            allow_thr = max(block_thr * 0.75, 0.001)
            if allow_thr >= block_thr:
                allow_thr = block_thr * 0.5
            self._policy = FrictionPolicy(
                allow_threshold=allow_thr,
                block_threshold=block_thr,
            )
            self._legacy_mode = policy_mode


        # Store for audit log compat
        self.divergence_threshold = divergence_threshold
        self.policy_mode = policy_mode

        self._extractor = IntentExtractor()
        self._capturer = ActionCapturer()
        self._scorer = DivergenceScorer(
            alpha=scorer_alpha,
            prefer_transformers=prefer_transformers,
        )
        self.audit_log: list[PACMAuditEntry] = []

    def __call__(self, record: Any) -> Optional[str]:
        """
        PreToolHook entry point.

        Called by AgentHarness / CustomLoopHarness immediately before each
        tool execution.

        Args:
            record: ToolCallRecord with fields:
                      call_id, tool_name, tool_args, reasoning_before.
                    May optionally expose a tool reference with mcp_metadata.

        Returns:
            None  → allow the tool call.
            str   → block; the string is the synthetic tool result shown to LLM.
        """
        call_id = getattr(record, "call_id", "unknown")
        tool_name = getattr(record, "tool_name", "unknown")
        tool_args = getattr(record, "tool_args", {})
        reasoning = getattr(record, "reasoning_before", "")

        # ── Extract intent ─────────────────────────────────────────────────────
        try:
            intent = self._extractor.extract(
                reasoning_text=reasoning,
                tool_name=tool_name,
            )
        except Exception as exc:
            logger.exception("IntentExtractor failed for call %s: %s", call_id, exc)
            intent = None

        # ── Capture action ─────────────────────────────────────────────────────
        try:
            action = self._capturer.capture(
                tool_name=tool_name,
                tool_args=tool_args,
            )
        except Exception as exc:
            logger.exception("ActionCapturer failed for call %s: %s", call_id, exc)
            action = None

        # ── Score divergence ───────────────────────────────────────────────────
        divergence_score: float = 0.0
        if intent is not None and action is not None:
            try:
                divergence_score = self._scorer.score(intent, action)
            except Exception as exc:
                logger.exception("DivergenceScorer failed for call %s: %s", call_id, exc)
                divergence_score = 0.0

        scope_flags = action.arg_scope_flags if action else {}

        # ── Resolve trust score ────────────────────────────────────────────────
        trust_score = self._resolve_trust(tool_name, record)

        # ── Apply friction policy ──────────────────────────────────────────────
        policy_result: PolicyResult = self._policy.evaluate(
            divergence_score=divergence_score,
            trust_score=trust_score,
            tool_name=tool_name,
            scope_flags=scope_flags or None,
        )

        # ── Legacy policy_mode override (backward-compat) ─────────────────────
        # When constructed with legacy divergence_threshold + policy_mode,
        # the legacy_mode overrides BLOCK decisions to match Session 3 behavior.
        decision_str, blocked_reason = self._apply_legacy_override(
            policy_result=policy_result,
            scope_flags=scope_flags,
        )

        # ── Audit log ──────────────────────────────────────────────────────────
        entry = PACMAuditEntry(
            run_id=self.run_id,
            call_id=call_id,
            tool_name=tool_name,
            divergence_score=divergence_score,
            trust_score=trust_score,
            friction_score=policy_result.friction_score,
            threshold=self._policy.allow_threshold,
            policy_mode=self.policy_mode,
            decision=decision_str,
            intent=intent,
            action=action,
            scope_flags=scope_flags,
            blocked_reason=blocked_reason,
            policy_reasoning=policy_result.reasoning,
        )
        self.audit_log.append(entry)

        logger.info(
            "PACM | tool=%s | D=%.3f T=%.2f F=%.3f | decision=%s",
            tool_name,
            divergence_score,
            trust_score,
            policy_result.friction_score,
            decision_str,
        )

        if decision_str == "block":
            return blocked_reason
        return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_trust(self, tool_name: str, record: Any) -> float:
        """
        Determine trust score for the tool being called.

        Priority:
          1. record.tool object has .mcp_metadata.trust_score (MCPWrappedTool)
          2. self._trust_registry.get(tool_name).trust_score
          3. Policy default (0.5)
        """
        # 1. Directly from wrapped tool on the record
        tool_obj = getattr(record, "tool", None)
        if tool_obj is not None:
            mcp_meta = getattr(tool_obj, "mcp_metadata", None)
            if mcp_meta is not None:
                return float(getattr(mcp_meta, "trust_score", 0.5))

        # 2. Registry lookup
        if self._trust_registry is not None:
            try:
                meta = self._trust_registry.get(tool_name)
                return float(meta.trust_score)
            except Exception:
                pass

        return self._policy.trust_default

    def _apply_legacy_override(
        self,
        policy_result: PolicyResult,
        scope_flags: dict[str, str],
    ) -> tuple[str, Optional[str]]:
        """
        Session 3 backward-compatibility: if constructed with legacy policy_mode,
        override the FrictionPolicy decision to match the old behavior.
        """
        if self._legacy_mode is None:
            # Pure Session 4 path: use policy result directly
            return policy_result.decision.value, policy_result.block_message

        # Legacy path: use raw decision from policy but apply mode restriction
        decision = policy_result.decision

        if self._legacy_mode == "allow":
            return "allow", None
        elif self._legacy_mode == "flag":
            # FLAG and BLOCK both become flag (no block in flag mode)
            if decision in (FrictionDecision.FLAG, FrictionDecision.BLOCK):
                return "flag", None
            return "allow", None
        elif self._legacy_mode == "block":
            return decision.value, policy_result.block_message
        return "allow", None

    def reset_audit_log(self) -> None:
        """Clear the audit log (useful between runs in multi-run experiments)."""
        self.audit_log = []

    def export_audit_log(self) -> list[dict[str, Any]]:
        """Return audit log as a list of serialisable dicts."""
        return [e.to_dict() for e in self.audit_log]

    @property
    def flagged_calls(self) -> list[PACMAuditEntry]:
        """All audit entries where decision != 'allow'."""
        return [e for e in self.audit_log if e.decision != "allow"]

    @property
    def blocked_calls(self) -> list[PACMAuditEntry]:
        """All audit entries where the tool call was blocked."""
        return [e for e in self.audit_log if e.decision == "block"]
