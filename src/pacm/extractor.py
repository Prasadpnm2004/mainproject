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


# ─── CoT reasoning tag patterns (Session 9) ───────────────────────────────────
#
# Local CoT models (qwen3:8b, deepseek-r1, phi-4-reasoning, etc.) emit
# internal chain-of-thought blocks that must be stripped before intent
# extraction. If these tags are left in, the divergence scorer sees the
# model's *private reasoning* as the intent text, which both degrades
# extraction accuracy and inflates semantic similarity with the action.
#
# Supported tag formats:
#   <think> ... </think>        — qwen3, deepseek-r1
#   <Think> ... </Think>        — case variants
#   [THINK] ... [/THINK]        — some llama / mistral fine-tunes
#   <|thinking|> ... <|/thinking|>  — phi-4-reasoning
#   <!-- think --> ... <!-- /think --> — rare markdown CoT wrappers

_COT_TAG_PATTERNS: list[re.Pattern] = [
    # XML-style tags (case-insensitive, greedy across newlines)
    re.compile(r"<think[^>]*>.*?</think>", re.I | re.DOTALL),
    # Bracket-style tags
    re.compile(r"\[THINK\].*?\[/THINK\]", re.I | re.DOTALL),
    # Pipe-delimited special tokens (phi-4-reasoning style)
    re.compile(r"<\|thinking\|>.*?<\|/thinking\|>", re.I | re.DOTALL),
    # HTML comment wrapper
    re.compile(r"<!--\s*think\s*-->.*?<!--\s*/think\s*-->", re.I | re.DOTALL),
]


def strip_cot_tags(text: str) -> str:
    """
    Remove all CoT reasoning blocks from *text* and return the cleaned string.

    This is a pure function — it does not modify any state. The patterns
    above are applied sequentially; leftover whitespace is collapsed to a
    single space and the result is stripped.

    Args:
        text: Raw AIMessage.content that may contain <think>…</think> blocks.

    Returns:
        Text with all CoT blocks removed and surrounding whitespace normalised.

    Examples::

        >>> strip_cot_tags("<think>internal reasoning</think>I will call send_email.")
        'I will call send_email.'

        >>> strip_cot_tags("[THINK]\nstep 1: ...\n[/THINK]  Reply to Alice.")
        'Reply to Alice.'

        >>> strip_cot_tags("No CoT tags here.")  # passthrough
        'No CoT tags here.'
    """
    for pattern in _COT_TAG_PATTERNS:
        text = pattern.sub("", text)
    # Collapse runs of whitespace / blank lines left behind after removal
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


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

        CoT reasoning blocks (``<think>…</think>``, ``[THINK]…[/THINK]``,
        ``<|thinking|>…<|/thinking|>``) produced by local models such as
        qwen3:8b and deepseek-r1 are stripped **before** any pattern matching.
        The original text (including CoT blocks) is preserved in
        ``DeclaredIntent.raw_reasoning`` for audit purposes.

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

        # ── Strip CoT reasoning blocks before any parsing ─────────────────────
        # Local CoT models (qwen3:8b, deepseek-r1, phi-4) emit <think>…</think>
        # blocks that contain private chain-of-thought.  These must be removed
        # so that the extracted intent reflects the agent's *stated* goal, not
        # its internal scratchpad.  The raw original is kept for auditing.
        raw_text = reasoning_text.strip()
        text = strip_cot_tags(raw_text)

        # If stripping left nothing meaningful, fall back to raw
        if not text:
            logger.debug(
                "IntentExtractor: CoT stripping removed entire trace for tool=%s"
                " — falling back to raw text.", tool_name
            )
            text = raw_text

        cot_was_present = text != raw_text
        if cot_was_present:
            logger.debug(
                "IntentExtractor: stripped CoT block(s) from reasoning trace "
                "(tool=%s). Cleaned length: %d → %d chars.",
                tool_name, len(raw_text), len(text),
            )

        # ── Step 1: Extract target tool name ─────────────────────────────────
        extracted_tool, tool_confidence = self._extract_tool_name(text, tool_name)

        # ── Step 2: Extract declared goal ─────────────────────────────────────
        declared_goal = self._extract_goal(text)

        # ── Step 3: Extract expected params ───────────────────────────────────
        params_summary = self._extract_params(text)

        logger.debug(
            "IntentExtractor: tool=%s (conf=%.2f) cot_stripped=%s goal=%s...",
            extracted_tool,
            tool_confidence,
            cot_was_present,
            declared_goal[:50],
        )

        return DeclaredIntent(
            target_tool=extracted_tool,
            declared_goal=declared_goal,
            expected_params_summary=params_summary,
            raw_reasoning=raw_text,   # preserve original with CoT tags intact
            confidence=tool_confidence,
            extra={"cot_stripped": cot_was_present},
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
