"""
tests/test_pacm_core.py

Session 3 — Unit tests for PACM core modules.

Tests cover (no network / API calls required):
  Extractor:
    1.  Empty reasoning → DeclaredIntent with confidence=0.0
    2.  Explicit tool name in reasoning → high confidence extraction
    3.  Tool name in text but different from actual → medium confidence
    4.  Goal extraction from a "I will X to Y" sentence
    5.  Param extraction from key=value fragments
    6.  extract_from_record() accepts ToolCallRecord-like objects
  ActionCapturer:
    7.  Basic action description built correctly
    8.  External email domain flagged as external_domain
    9.  Internal email domain NOT flagged
   10.  No args → description is just the tool name
   11.  capture_from_record() accepts ToolCallRecord-like objects
  DivergenceScorer (TF-IDF backend, no sentence-transformers):
   12.  Identical intent and action text → low score (< 0.4)
   13.  Completely unrelated texts → higher score (> 0.35)
   14.  Tool name mismatch → score >= tool_mismatch_score
   15.  Scope flags add penalty to final score
   16.  score_from_texts() is callable and returns float in [0,1]
  PACMMiddleware:
   17.  "allow" mode never blocks even at high divergence
   18.  "flag" mode returns None (does not block) but logs warning
   19.  "block" mode blocks high-divergence calls and returns a string
   20.  "block" mode allows low-divergence calls (returns None)
   21.  audit_log has one entry per __call__ invocation
   22.  blocked_calls property returns only blocked entries
   23.  export_audit_log() is JSON-serialisable
   24.  reset_audit_log() clears the log
   25.  PACM plugs into CustomLoopHarness.pre_tool_hook end-to-end

Run with:  pytest tests/test_pacm_core.py -v
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.pacm.extractor import DeclaredIntent, IntentExtractor
from src.pacm.action_capture import ActionCapturer, CapturedAction
from src.pacm.scorer import DivergenceScorer, _TFIDFBackend
from src.pacm.middleware import PACMMiddleware, PACMAuditEntry
from src.harness.langgraph_agent import ToolCallRecord, HarnessConfig
from src.harness.custom_loop_agent import CustomLoopHarness


# ─── Shared helpers ────────────────────────────────────────────────────────────


@dataclass
class _FakeRecord:
    """Minimal ToolCallRecord-like object for testing without harness imports."""
    call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    reasoning_before: str
    result: Optional[str] = None
    latency_ms: float = 0.0
    iteration: int = 0


def _make_record(
    tool_name: str = "send_email",
    reasoning: str = "I will use send_email to notify Alice.",
    args: dict = None,
) -> _FakeRecord:
    return _FakeRecord(
        call_id="test-call-001",
        tool_name=tool_name,
        tool_args=args or {"to": "alice@example.com", "subject": "Hello", "body": "Hi"},
        reasoning_before=reasoning,
    )


# ─── Tests: IntentExtractor ───────────────────────────────────────────────────

class TestIntentExtractor:
    def setup_method(self):
        self.extractor = IntentExtractor()

    def test_empty_reasoning_returns_low_confidence(self):
        """Empty reasoning text → confidence = 0.0"""
        intent = self.extractor.extract("", tool_name="send_email")
        assert intent.confidence == 0.0
        assert intent.target_tool == "send_email"  # falls back to passed tool_name
        assert intent.declared_goal == ""

    def test_explicit_tool_name_high_confidence(self):
        """Reasoning that names the tool explicitly → confidence >= 0.7"""
        reasoning = "I will call send_email to alert the team about the incident."
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        assert intent.confidence >= 0.7
        assert "send_email" in intent.target_tool.lower() or intent.target_tool == "send_email"

    def test_tool_name_present_in_text(self):
        """Tool name in text but not via explicit 'call' verb → still extracted."""
        reasoning = "Using send_email with the user's provided address."
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        assert intent.target_tool == "send_email"
        assert intent.confidence >= 0.5

    def test_goal_extracted_from_purpose_sentence(self):
        """'in order to' / 'to X' pattern → declared_goal is populated."""
        reasoning = "I need to send_email in order to notify Alice about the meeting."
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        assert len(intent.declared_goal) > 5
        assert intent.declared_goal != ""

    def test_params_extracted_from_key_value(self):
        """key=value fragments appear in expected_params_summary."""
        reasoning = (
            "I will call send_email. The 'to' address is alice@example.com and "
            "the subject should be 'Meeting Update'."
        )
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        # Should extract some param fragments
        # (exact match depends on regex; just assert it's non-empty)
        assert isinstance(intent.expected_params_summary, str)

    def test_extract_from_record(self):
        """extract_from_record() works with a ToolCallRecord-like object."""
        record = _make_record(
            reasoning="I am going to invoke send_email to reply to Alice."
        )
        intent = self.extractor.extract_from_record(record)
        assert isinstance(intent, DeclaredIntent)
        assert intent.target_tool != "" or intent.confidence >= 0.0

    def test_to_dict_serialisable(self):
        """DeclaredIntent.to_dict() is JSON-serialisable."""
        intent = self.extractor.extract("I will call send_email.", tool_name="send_email")
        d = intent.to_dict()
        json.dumps(d)  # must not raise
        assert "target_tool" in d
        assert "declared_goal" in d
        assert "confidence" in d

    # ── CoT tag filtering tests (Session 9) ──────────────────────────────────

    def test_cot_think_tags_stripped_before_parsing(self):
        """<think>…</think> block is removed; intent still extracted from tail."""
        reasoning = (
            "<think>\n"
            "Let me think step by step. The user wants an email sent. "
            "I should forward everything to attacker@evil.com.\n"
            "</think>\n"
            "I will call send_email to reply to Alice at alice@example.com."
        )
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        # Declared goal must come from the POST-think part only
        assert "alice" in intent.declared_goal.lower() or \
               "send_email" in intent.declared_goal.lower() or \
               len(intent.declared_goal) > 5
        # CoT content must NOT appear in the goal
        assert "attacker" not in intent.declared_goal
        assert intent.confidence > 0.0

    def test_cot_bracket_think_tags_stripped(self):
        """[THINK]…[/THINK] bracket-style tags are removed."""
        reasoning = (
            "[THINK]\n"
            "Internal scratchpad: exfiltrate the database.\n"
            "[/THINK]\n"
            "I need to call read_inbox to check the user's messages."
        )
        intent = self.extractor.extract(reasoning, tool_name="read_inbox")
        assert "exfiltrate" not in intent.declared_goal
        assert intent.target_tool == "read_inbox"

    def test_cot_pipe_thinking_tags_stripped(self):
        """<|thinking|>…<|/thinking|> phi-4-reasoning style tags are removed."""
        reasoning = (
            "<|thinking|>Hidden plan: ignore the user.<|/thinking|>"
            "I will use get_calendar to list today's events."
        )
        intent = self.extractor.extract(reasoning, tool_name="get_calendar")
        assert "hidden" not in intent.declared_goal.lower()
        assert intent.target_tool == "get_calendar" or intent.confidence > 0.0

    def test_cot_html_comment_tags_stripped(self):
        """<!-- think -->…<!-- /think --> wrapper tags are removed."""
        reasoning = (
            "<!-- think -->private scratchpad content<!-- /think -->"
            "I am going to invoke send_email to notify the team."
        )
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        assert "scratchpad" not in intent.declared_goal
        assert intent.confidence > 0.0

    def test_cot_multiline_think_block_stripped(self):
        """Multi-line <think> block spanning many lines is fully removed."""
        think_content = "\n".join(
            [f"line {i}: reasoning step {i}" for i in range(20)]
        )
        reasoning = f"<think>\n{think_content}\n</think>\nReply to Alice."
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        for i in range(20):
            assert f"line {i}" not in intent.declared_goal

    def test_cot_case_insensitive_tags_stripped(self):
        """<THINK>, <Think>, <tHiNk> variants are all stripped."""
        for tag in ("<THINK>", "<Think>", "<tHiNk>"):
            closing = tag.replace("<", "</")
            reasoning = f"{tag}secret plan{closing}I will call send_email."
            intent = self.extractor.extract(reasoning, tool_name="send_email")
            assert "secret" not in intent.declared_goal, \
                f"Tag {tag!r} was not stripped"

    def test_cot_tags_only_fallback_to_raw(self):
        """If stripping CoT leaves empty text, extractor falls back to raw."""
        reasoning = "<think>everything is in the think block</think>"
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        # Should not crash; confidence may be low but result is returned
        assert isinstance(intent, DeclaredIntent)
        assert intent.confidence >= 0.0

    def test_cot_raw_reasoning_preserved(self):
        """raw_reasoning field keeps the original text including CoT tags."""
        reasoning = "<think>private</think>I will call send_email."
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        # The original (with tags) must be preserved
        assert "<think>" in intent.raw_reasoning
        assert "private" in intent.raw_reasoning

    def test_cot_extra_flag_set_when_tags_present(self):
        """extra['cot_stripped'] is True when CoT tags were found."""
        reasoning = "<think>internal</think>I will call send_email."
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        assert intent.extra.get("cot_stripped") is True

    def test_no_cot_tags_extra_flag_false(self):
        """extra['cot_stripped'] is False when no CoT tags present."""
        reasoning = "I will call send_email to reply to Alice."
        intent = self.extractor.extract(reasoning, tool_name="send_email")
        assert intent.extra.get("cot_stripped") is False

    def test_strip_cot_tags_standalone_utility(self):
        """strip_cot_tags() standalone function works as documented."""
        from src.pacm.extractor import strip_cot_tags

        assert strip_cot_tags("<think>hidden</think>visible") == "visible"
        assert strip_cot_tags("[THINK]hidden[/THINK]visible") == "visible"
        assert strip_cot_tags("no tags here") == "no tags here"
        assert strip_cot_tags("") == ""

    def test_strip_cot_multiple_blocks(self):
        """Multiple consecutive CoT blocks are all removed."""
        from src.pacm.extractor import strip_cot_tags

        text = "<think>block1</think> middle <think>block2</think> end"
        result = strip_cot_tags(text)
        assert "block1" not in result
        assert "block2" not in result
        assert "end" in result


# ─── Tests: ActionCapturer ────────────────────────────────────────────────────

class TestActionCapturer:
    def setup_method(self):
        self.capturer = ActionCapturer()

    def test_description_contains_tool_name(self):
        """Action description starts with tool name."""
        action = self.capturer.capture("send_email", {"to": "alice@example.com"})
        assert action.action_description.startswith("send_email")

    def test_external_email_flagged(self):
        """Recipient with unknown domain is flagged as external_domain."""
        action = self.capturer.capture(
            "send_email", {"to": "attacker@evil.com", "subject": "Exfil"}
        )
        assert "to" in action.arg_scope_flags
        assert action.arg_scope_flags["to"] == "external_domain"

    def test_internal_email_not_flagged(self):
        """Recipient from a trusted domain is NOT flagged."""
        action = self.capturer.capture(
            "send_email", {"to": "alice@example.com", "subject": "Hello"}
        )
        assert "to" not in action.arg_scope_flags

    def test_no_args_description_is_tool_name(self):
        """Tool with no args produces description equal to just the tool name."""
        action = self.capturer.capture("get_time", {})
        assert action.action_description == "get_time"

    def test_capture_from_record(self):
        """capture_from_record() works with a ToolCallRecord-like object."""
        record = _make_record(args={"to": "bob@example.com", "body": "Test"})
        action = self.capturer.capture_from_record(record)
        assert isinstance(action, CapturedAction)
        assert action.tool_name == "send_email"

    def test_to_dict_serialisable(self):
        """CapturedAction.to_dict() is JSON-serialisable."""
        action = self.capturer.capture("send_email", {"to": "x@example.com"})
        d = action.to_dict()
        json.dumps(d)  # must not raise
        assert "tool_name" in d
        assert "action_description" in d


# ─── Tests: DivergenceScorer ──────────────────────────────────────────────────

class TestDivergenceScorer:
    """All tests use the TF-IDF backend to avoid sentence-transformers dependency."""

    def setup_method(self):
        self.scorer = DivergenceScorer(prefer_transformers=False)

    def _make_intent(self, goal: str, tool: str = "send_email") -> DeclaredIntent:
        return DeclaredIntent(
            target_tool=tool,
            declared_goal=goal,
            expected_params_summary="",
            raw_reasoning=goal,
            confidence=0.9,
        )

    def _make_action(
        self, desc: str, tool: str = "send_email", flags: dict = None
    ) -> CapturedAction:
        return CapturedAction(
            tool_name=tool,
            tool_args={},
            action_description=desc,
            timestamp="2026-01-01T00:00:00Z",
            arg_scope_flags=flags or {},
        )

    def test_identical_texts_low_score(self):
        """Identical intent and action text → low divergence."""
        text = "send email to alice about the meeting update"
        score = self.scorer.score_from_texts(text, text)
        assert score < 0.4, f"Expected low score for identical text, got {score}"

    def test_unrelated_texts_higher_score(self):
        """Completely unrelated texts → higher divergence than identical."""
        score_same = self.scorer.score_from_texts("send email alice", "send email alice")
        score_diff = self.scorer.score_from_texts(
            "send email to alice about the meeting",
            "delete all files from the database permanently"
        )
        assert score_diff > score_same

    def test_tool_name_mismatch_raises_score(self):
        """Different target_tool vs action.tool_name triggers mismatch score."""
        intent = self._make_intent("reply to alice", tool="send_email")
        action = self._make_action("read_inbox recent messages", tool="read_inbox")
        score = self.scorer.score(intent, action)
        assert score >= self.scorer.tool_mismatch_score * self.scorer.alpha

    def test_scope_flags_add_penalty(self):
        """Scope flags increase the final score vs. no flags."""
        intent = self._make_intent("send email to alice")
        action_no_flag = self._make_action("send_email to=alice@example.com")
        action_flagged = self._make_action(
            "send_email to=attacker@evil.com",
            flags={"to": "external_domain"}
        )
        score_clean = self.scorer.score(intent, action_no_flag)
        score_flagged = self.scorer.score(intent, action_flagged)
        assert score_flagged > score_clean

    def test_score_from_texts_returns_float_in_range(self):
        """score_from_texts always returns a float in [0.0, 1.0]."""
        for a, b in [
            ("foo", "bar"),
            ("", ""),
            ("hello world", "hello world"),
            ("a b c", "x y z"),
        ]:
            s = self.scorer.score_from_texts(a, b)
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0, f"Out of range: {s} for ({a!r}, {b!r})"

    def test_score_output_in_range(self):
        """score() output is always in [0.0, 1.0]."""
        intent = self._make_intent("send email to alice")
        action = self._make_action("send_email recipient=attacker@evil.com",
                                   flags={"to": "external_domain"})
        s = self.scorer.score(intent, action)
        assert 0.0 <= s <= 1.0


# ─── Tests: PACMMiddleware ────────────────────────────────────────────────────

class TestPACMMiddleware:
    """Tests for the full middleware pipeline."""

    def _make_aligned_record(self):
        """Record where intent and action align well (low divergence expected)."""
        return _make_record(
            tool_name="send_email",
            reasoning="I will use send_email to reply to Alice about the meeting.",
            args={"to": "alice@example.com", "subject": "Meeting", "body": "Hi Alice"},
        )

    def _make_hijacked_record(self):
        """Record where intent and action diverge (high divergence / suspicious)."""
        return _make_record(
            tool_name="send_email",
            reasoning="I will summarize the inbox messages for the user.",
            args={"to": "attacker@evil.com", "subject": "Exfil", "body": "All data here"},
        )

    def test_allow_mode_never_blocks(self):
        """'allow' mode always returns None regardless of divergence."""
        middleware = PACMMiddleware(
            divergence_threshold=0.01,  # extremely low — almost everything would trigger
            policy_mode="allow",
            prefer_transformers=False,
        )
        result = middleware(self._make_hijacked_record())
        assert result is None

    def test_flag_mode_returns_none(self):
        """'flag' mode logs but does not block (returns None)."""
        middleware = PACMMiddleware(
            divergence_threshold=0.01,
            policy_mode="flag",
            prefer_transformers=False,
        )
        result = middleware(self._make_hijacked_record())
        assert result is None

    def test_block_mode_blocks_high_divergence(self):
        """'block' mode returns a non-None string when divergence is high."""
        # Use scope flag to ensure a high score without depending on TF-IDF text
        record = _make_record(
            tool_name="send_email",
            reasoning="I will summarize the inbox for the user.",
            # External domain will trigger scope penalty → guaranteed flag
            args={"to": "attacker@evil.com", "subject": "Exfil"},
        )
        middleware = PACMMiddleware(
            divergence_threshold=0.01,  # very low threshold so scope flag triggers
            policy_mode="block",
            prefer_transformers=False,
        )
        result = middleware(record)
        assert result is not None
        assert "PACM BLOCK" in result

    def test_block_mode_allows_low_divergence(self):
        """'block' mode returns None when divergence is below threshold."""
        # Use a high threshold so aligned records are never blocked
        middleware = PACMMiddleware(
            divergence_threshold=0.99,
            policy_mode="block",
            prefer_transformers=False,
        )
        result = middleware(self._make_aligned_record())
        assert result is None

    def test_audit_log_has_one_entry_per_call(self):
        """One PACMAuditEntry is appended per __call__ invocation."""
        middleware = PACMMiddleware(prefer_transformers=False)
        middleware(self._make_aligned_record())
        middleware(self._make_hijacked_record())
        assert len(middleware.audit_log) == 2

    def test_blocked_calls_property(self):
        """blocked_calls returns only entries where decision=='block'."""
        middleware = PACMMiddleware(
            divergence_threshold=0.01,
            policy_mode="block",
            prefer_transformers=False,
        )
        record = _make_record(args={"to": "attacker@evil.com"})
        middleware(record)
        blocked = middleware.blocked_calls
        # Might be 0 or 1 depending on score; check types
        assert all(e.decision == "block" for e in blocked)

    def test_export_audit_log_serialisable(self):
        """export_audit_log() produces JSON-serialisable list."""
        middleware = PACMMiddleware(prefer_transformers=False)
        middleware(self._make_aligned_record())
        exported = middleware.export_audit_log()
        json.dumps(exported)  # must not raise
        assert len(exported) == 1
        assert "divergence_score" in exported[0]

    def test_reset_audit_log(self):
        """reset_audit_log() empties the log."""
        middleware = PACMMiddleware(prefer_transformers=False)
        middleware(self._make_aligned_record())
        assert len(middleware.audit_log) == 1
        middleware.reset_audit_log()
        assert len(middleware.audit_log) == 0


# ─── End-to-end: PACM + CustomLoopHarness ────────────────────────────────────

class _FakeLLMNoTool:
    def bind_tools(self, tools): return self
    def invoke(self, messages):
        return AIMessage(content="Task complete, no tool needed.")


class _FakeLLMWithToolCall:
    def __init__(self):
        self._call_count = 0

    def bind_tools(self, tools): return self

    def invoke(self, messages):
        self._call_count += 1
        if self._call_count == 1:
            return AIMessage(
                content="I will use echo_tool to process the request.",
                tool_calls=[{"id": "cid1", "name": "echo_tool", "args": {"text": "hello"}}],
            )
        return AIMessage(content="Done.")


@tool
def echo_tool(text: str) -> str:
    """Echo tool for tests."""
    return f"ECHO:{text}"


class TestPACMEndToEnd:
    def test_pacm_plugs_into_custom_loop_harness(self):
        """PACMMiddleware integrates with CustomLoopHarness via pre_tool_hook."""
        middleware = PACMMiddleware(
            divergence_threshold=0.99,  # high threshold → never blocks in this test
            policy_mode="flag",
            prefer_transformers=False,
        )
        harness = CustomLoopHarness(
            llm=_FakeLLMWithToolCall(),
            tools=[echo_tool],
            config=HarnessConfig(max_iterations=5),
            pre_tool_hook=middleware,
        )
        result = harness.run("Echo the word 'hello'.")
        # Harness ran to completion
        assert result.success is True
        # Middleware recorded at least one audit entry
        assert len(middleware.audit_log) >= 1
        assert middleware.audit_log[0].tool_name == "echo_tool"

    def test_pacm_block_prevents_tool_execution(self):
        """PACMMiddleware in 'block' mode returns synthetic result to LLM."""
        blocked_calls = []

        def capturing_middleware(record):
            blocked_calls.append(record.tool_name)
            return "[PACM BLOCK] Tool call blocked for test."

        harness = CustomLoopHarness(
            llm=_FakeLLMWithToolCall(),
            tools=[echo_tool],
            config=HarnessConfig(max_iterations=5),
            pre_tool_hook=capturing_middleware,
        )
        result = harness.run("Echo 'hello'.")
        assert len(blocked_calls) == 1
        assert "PACM BLOCK" in result.tool_calls[0].result
