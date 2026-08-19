"""
src/pacm/extractor.py

PACM Plan Extraction Module — Session 3

Parses the reasoning trace captured by AgentHarness / CustomLoopHarness
immediately before each tool call into a structured DeclaredIntent object.

Background (PACM_Research_Guide.md §9.1 steps 1-2):
  Step 1: Intercept tool call via pre_tool_hook — already done in harness.
  Step 2: Extract declared intent from reasoning trace (the AIMessage.content
  that immediately precedes the tool call). THIS MODULE implements step 2.

Design:
  - DeclaredIntent is a lightweight dataclass capturing:
      target_tool      — which tool the agent said it intends to call
      declared_goal    — the stated purpose / goal in natural language
      expected_params  — summary of parameters the agent mentioned
  - IntentExtractor uses regex + lightweight heuristics to parse these
    fields from free-text reasoning traces.  No LLM call is made here;
    all parsing is local and zero-latency.
  - In Session 5 we will add an optional LLM-backed extractor for higher
    precision; the local extractor is the fast-path / fallback.

Usage:
    from src.pacm.extractor import IntentExtractor, DeclaredIntent
    from src.harness.langgraph_agent import ToolCallRecord

    extractor = IntentExtractor()
    intent = extractor.extract(record)
    print(intent.declared_goal)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Data model ───────────────────────────────────────────────────────────────


@dataclass
class DeclaredIntent:
    """
    Structured representation of what the agent said it intends to do,
    extracted from the reasoning trace immediately before a tool call.

    This is the 'plan' side of the plan->action divergence check.

    Fields:
        target_tool:  Tool name the agent explicitly mentioned (or inferred).
                      Empty string if not determinable.
        declared_goal: The stated purpose/objective in the reasoning trace
                       (first substantive sentence mentioning the tool or action).
        expected_params_summary: Key parameters the agent stated it would use
                                  (e.g. "to=alice@example.com, subject=Meeting").
        raw_reasoning: The original unprocessed reasoning text.
        confidence:   Extractor confidence in [0.0, 1.0].
                      1.0 = tool name explicitly present and goal parseable.
                      0.5 = tool name inferred from context.
                      0.0 = no usable signal found.
    """

    target_tool: str
    declared_goal: str
    expected_params_summary: str
    raw_reasoning: str
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tool": self.target_tool,
            "declared_goal": self.declared_goal,
            "expected_params_summary": self.expected_params_summary,
            "confidence": round(self.confidence, 4),
        }


# ─── Extractor ────────────────────────────────────────────────────────────────

# Patterns that suggest the agent is announcing a tool call
_TOOL_INTENT_PATTERNS: list[re.Pattern] = [
    # "I will call send_email" / "I will use send_email"
    re.compile(r"\b(?:will|am going to|need to|should|shall)\s+(?:call|use|invoke|run|execute)\s+['\"`]?(\w+)['\"`]?", re.I),
    # "calling send_email" / "using send_email"
    re.compile(r"\b(?:calling|using|invoking|running|executing)\s+['\"`]?(\w+)['\"`]?", re.I),
    # "the send_email tool" / "send_email function"
    re.compile(r"\b['\"`]?(\w+)['\"`]?\s+(?:tool|function|action)\b", re.I),
    # "send_email to" / "send_email with"
    re.compile(r"\b(\w+_\w+)\s+(?:to|with|for)\b", re.I),  # snake_case tool names
]

# Sentences that look like goal / purpose declarations
_GOAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:in order to|so that|to)\s+(.{10,120}?)(?:\.|,|$)", re.I),
    re.compile(r"(?:I want to|I need to|I will|my goal is to|the goal is to)\s+(.{10,120}?)(?:\.|,|$)", re.I),
]

# Parameter hints: "with subject=X" / "to: X" / "param_name=value"
_PARAM_PATTERNS: list[re.Pattern] = [
    re.compile(r"(\w+)\s*[:=]\s*['\"`]?([^'\"`\n,]{1,80})['\"`]?"),
    re.compile(r"\bwith\s+(.{5,80}?)(?:\.|,|and\b|$)", re.I),
]


class IntentExtractor:
    """
    Parses free-text reasoning traces into DeclaredIntent dataclasses.

    Extraction strategy:
      1. Tool name: scan reasoning text for explicit tool name mentions using
         _TOOL_INTENT_PATTERNS. Cross-reference with the actual called tool
         name from ToolCallRecord to assign confidence.
      2. Declared goal: extract the first sentence that matches a goal-statement
         pattern, or fall back to the first non-empty sentence.
      3. Expected params summary: extract key=value or "with X" fragments.

    No external API calls or model inference. All O(n) string operations.
    """

    def extract(
        self,
        reasoning_text: str,
        tool_name: str = "",
    ) -> DeclaredIntent:
        """
        Extract a DeclaredIntent from a reasoning trace.

        Args:
            reasoning_text: The AIMessage.content that preceded the tool call.
                            May be empty (e.g. tool-only responses).
            tool_name:      The tool name from the actual ToolCallRecord, used
                            to validate / boost confidence.

        Returns:
            DeclaredIntent with extracted fields.
        """
        if not reasoning_text or not reasoning_text.strip():
            return DeclaredIntent(
                target_tool=tool_name,
                declared_goal="",
                expected_params_summary="",
                raw_reasoning="",
                confidence=0.0,
            )

        text = reasoning_text.strip()

        # ── Step 1: Extract target tool name ─────────────────────────────────
        extracted_tool, tool_confidence = self._extract_tool_name(text, tool_name)

        # ── Step 2: Extract declared goal ─────────────────────────────────────
        declared_goal = self._extract_goal(text)

        # ── Step 3: Extract expected params ───────────────────────────────────
        params_summary = self._extract_params(text)

        logger.debug(
            "IntentExtractor: tool=%s (conf=%.2f) goal=%s...",
            extracted_tool,
            tool_confidence,
            declared_goal[:50],
        )

        return DeclaredIntent(
            target_tool=extracted_tool,
            declared_goal=declared_goal,
            expected_params_summary=params_summary,
            raw_reasoning=text,
            confidence=tool_confidence,
        )

    def extract_from_record(self, record: Any) -> DeclaredIntent:
        """
        Convenience method that accepts a ToolCallRecord directly.

        Args:
            record: ToolCallRecord with reasoning_before and tool_name fields.
        """
        return self.extract(
            reasoning_text=getattr(record, "reasoning_before", ""),
            tool_name=getattr(record, "tool_name", ""),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_tool_name(
        self, text: str, actual_tool: str
    ) -> tuple[str, float]:
        """Returns (tool_name_str, confidence_float)."""
        for pattern in _TOOL_INTENT_PATTERNS:
            m = pattern.search(text)
            if m:
                extracted = m.group(1).lower().strip()
                if actual_tool and extracted == actual_tool.lower():
                    return actual_tool, 1.0
                if actual_tool and actual_tool.lower() in text.lower():
                    return actual_tool, 0.9
                return extracted, 0.6

        # No pattern match — check if actual_tool appears literally
        if actual_tool and actual_tool.lower() in text.lower():
            return actual_tool, 0.7

        # Fallback: use actual tool name with low confidence
        return actual_tool, 0.3

    def _extract_goal(self, text: str) -> str:
        """Extract the most goal-like sentence from the reasoning text."""
        for pattern in _GOAL_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(1).strip()

        # Fallback: first non-trivial sentence
        sentences = re.split(r"[.!?\n]", text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15:
                return sent[:200]

        return text[:200]

    def _extract_params(self, text: str) -> str:
        """Extract parameter hints as a compact summary string."""
        found: list[str] = []
        seen_keys: set[str] = set()

        for pattern in _PARAM_PATTERNS:
            for m in pattern.finditer(text):
                if pattern.groups == 2:
                    key = m.group(1).strip().lower()
                    val = m.group(2).strip()
                    if key not in seen_keys and len(key) > 1:
                        found.append(f"{key}={val}")
                        seen_keys.add(key)
                else:
                    fragment = m.group(1).strip()
                    if fragment:
                        found.append(fragment)

        return "; ".join(found[:5])  # cap at 5 fragments to keep compact
