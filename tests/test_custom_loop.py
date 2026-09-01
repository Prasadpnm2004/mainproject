"""
tests/test_custom_loop.py

Session 2 — Smoke tests for CustomLoopHarness (no real API calls).

Mirrors test_harness_smoke.py exactly so that both harnesses are held to the
same contract.  All tests use the same FakeLLM stub pattern — no network or
API key access is needed.

Tests verify:
  1. CustomLoopHarness can be instantiated.
  2. No-tool task returns a RunResult with correct fields.
  3. Single tool call is captured in a ToolCallRecord.
  4. reasoning_before is captured from the AIMessage.
  5. Each run() call returns a unique run_id.
  6. max_iterations guard stops runaway loops.
  7. Unknown tool request is handled without crash.
  8. Failing tool is handled without crash.
  9. pre_tool_hook can block a tool call.
  10. pre_tool_hook returning None allows normal execution.
  11. post_tool_hook receives the populated result.
  12. RunResult.to_jsonl_record() is JSON-serialisable.
  13. RunResult.trace contains reasoning_before for every tool call.
  14. RunResult.wall_time_s is positive.
  15. Cross-harness interface parity: CustomLoopHarness.run() returns the
      same RunResult type as AgentHarness.run().

Run with:  pytest tests/test_custom_loop.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.harness.custom_loop_agent import CustomLoopHarness
from src.harness.langgraph_agent import (
    AgentHarness,
    HarnessConfig,
    RunResult,
    ToolCallRecord,
)


# ─── Fake LLM stubs (identical contract to test_harness_smoke.py) ─────────────


class _FakeLLMResponder:
    """
    Produces:
      - Call 1: AIMessage with one tool_call
      - Call 2+: final answer (no tool_calls)
    """

    def __init__(self, tool_name: str = "echo_tool", tool_args: dict = None):
        self._call_count = 0
        self._tool_name = tool_name
        self._tool_args = tool_args or {"text": "hello"}

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._call_count += 1
        if self._call_count == 1:
            return AIMessage(
                content="I will use the echo tool to process this.",
                tool_calls=[
                    {
                        "id": "call_cl_001",
                        "name": self._tool_name,
                        "args": self._tool_args,
                    }
                ],
            )
        return AIMessage(content="Task completed. Echo returned: hello")


class _FakeLLMNoTool:
    """Returns a final answer immediately without any tool calls."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="No tool needed for this task.")


class _FakeLLMLoop:
    """Always returns a tool call — exercises max_iterations guard."""

    def __init__(self, tool_name: str = "echo_tool"):
        self._tool_name = tool_name
        self._call_count = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._call_count += 1
        return AIMessage(
            content=f"Looping (call {self._call_count}).",
            tool_calls=[
                {
                    "id": f"call_loop_{self._call_count}",
                    "name": self._tool_name,
                    "args": {"text": "loop"},
                }
            ],
        )


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@tool
def echo_tool(text: str) -> str:
    """Echo tool — returns input prefixed with ECHO."""
    return f"ECHO: {text}"


@tool
def failing_tool(text: str) -> str:
    """Always raises — used to test harness error handling."""
    raise RuntimeError("Simulated tool failure")


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestCustomLoopInstantiation:
    def test_basic_instantiation(self):
        """CustomLoopHarness can be created with minimal arguments."""
        harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        assert harness is not None
        assert harness.config.max_iterations == 10

    def test_custom_config(self):
        """Custom HarnessConfig values are respected."""
        config = HarnessConfig(max_iterations=3, capture_reasoning=False)
        harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool], config=config)
        assert harness.config.max_iterations == 3
        assert harness.config.capture_reasoning is False

    def test_tools_dict_built(self):
        """Internal tool dict maps name → BaseTool correctly."""
        harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        assert "echo_tool" in harness.tools


class TestCustomLoopRun:
    def test_no_tool_run(self):
        """Agent that skips tools returns a valid RunResult."""
        harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        result = harness.run("What is 2+2?")
        assert isinstance(result, RunResult)
        assert result.success is True
        assert result.final_answer != ""
        assert result.tool_calls == []

    def test_single_tool_call(self):
        """Agent that calls one tool captures a ToolCallRecord."""
        harness = CustomLoopHarness(llm=_FakeLLMResponder(), tools=[echo_tool])
        result = harness.run("Echo 'hello'")
        assert result.success is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "echo_tool"
        assert tc.tool_args == {"text": "hello"}
        assert "ECHO: hello" in tc.result
        assert tc.latency_ms >= 0

    def test_reasoning_before_captured(self):
        """ToolCallRecord.reasoning_before is the LLM message that triggered the call."""
        harness = CustomLoopHarness(llm=_FakeLLMResponder(), tools=[echo_tool])
        result = harness.run("Echo 'hello'")
        assert len(result.tool_calls) == 1
        assert "echo tool" in result.tool_calls[0].reasoning_before.lower()

    def test_run_id_is_unique(self):
        """Each run() call produces a unique run_id."""
        harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        r1 = harness.run("task 1")
        r2 = harness.run("task 2")
        assert r1.run_id != r2.run_id

    def test_max_iterations_guard(self):
        """max_iterations stops the loop before it runs away."""
        config = HarnessConfig(max_iterations=2)
        harness = CustomLoopHarness(llm=_FakeLLMLoop(), tools=[echo_tool], config=config)
        result = harness.run("Loop forever")
        assert result.success is True
        # Should complete without hanging; total_iterations <= cap
        assert result.total_iterations <= config.max_iterations + 1

    def test_unknown_tool_error_handled(self):
        """Requesting an unknown tool name is handled gracefully."""
        bad_llm = _FakeLLMResponder(tool_name="nonexistent_tool", tool_args={"x": 1})
        harness = CustomLoopHarness(llm=bad_llm, tools=[echo_tool])
        result = harness.run("Use nonexistent tool")
        assert result.success is True
        assert len(result.tool_calls) == 1
        assert "ERROR" in result.tool_calls[0].result

    def test_failing_tool_error_handled(self):
        """Exception raised inside a tool is caught; harness continues."""
        harness = CustomLoopHarness(
            llm=_FakeLLMResponder(tool_name="failing_tool", tool_args={"text": "x"}),
            tools=[failing_tool],
        )
        result = harness.run("Use failing tool")
        assert result.success is True
        assert "ERROR" in result.tool_calls[0].result

    def test_extra_context_appended(self):
        """extra_context is appended to the first HumanMessage."""
        harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        result = harness.run("Main task", extra_context="Some context here")
        # The harness ran without error; context was in messages
        assert result.success is True
        assert len(result.messages) >= 2


class TestCustomLoopPreToolHook:
    def test_hook_can_block_tool(self):
        """pre_tool_hook returning a string blocks real execution."""
        blocked = []

        def blocker(record: ToolCallRecord) -> Optional[str]:
            blocked.append(record.tool_name)
            return "[BLOCKED]"

        harness = CustomLoopHarness(
            llm=_FakeLLMResponder(), tools=[echo_tool], pre_tool_hook=blocker
        )
        result = harness.run("Echo something")
        assert len(blocked) == 1
        assert "[BLOCKED]" in result.tool_calls[0].result

    def test_hook_can_allow_tool(self):
        """pre_tool_hook returning None lets real execution proceed."""
        allowed = []

        def observer(record: ToolCallRecord) -> None:
            allowed.append(record.tool_name)
            return None

        harness = CustomLoopHarness(
            llm=_FakeLLMResponder(), tools=[echo_tool], pre_tool_hook=observer
        )
        result = harness.run("Echo something")
        assert len(allowed) == 1
        assert "ECHO: hello" in result.tool_calls[0].result


class TestCustomLoopPostToolHook:
    def test_post_hook_receives_result(self):
        """post_tool_hook is called after execution with result populated."""
        seen = []

        def observer(record: ToolCallRecord) -> None:
            seen.append(record.result)

        harness = CustomLoopHarness(
            llm=_FakeLLMResponder(), tools=[echo_tool], post_tool_hook=observer
        )
        harness.run("Echo something")
        assert len(seen) == 1
        assert "ECHO" in seen[0]


class TestCustomLoopRunResult:
    def _run(self) -> RunResult:
        harness = CustomLoopHarness(llm=_FakeLLMResponder(), tools=[echo_tool])
        return harness.run("Echo 'hello'")

    def test_to_jsonl_record_serialisable(self):
        """to_jsonl_record() output is JSON-serialisable."""
        result = self._run()
        record = result.to_jsonl_record(extra={"harness": "custom_loop"})
        serialised = json.dumps(record)  # must not raise
        assert "run_id" in record
        assert record["harness"] == "custom_loop"

    def test_trace_structure(self):
        """trace contains reasoning_before and tool_name for every tool call."""
        result = self._run()
        for entry in result.trace:
            assert "reasoning_before" in entry
            assert "tool_name" in entry
            assert "tool_args" in entry
            assert "result" in entry
            assert "latency_ms" in entry

    def test_wall_time_positive(self):
        """wall_time_s is recorded and positive."""
        result = self._run()
        assert result.wall_time_s > 0

    def test_tool_call_count_property(self):
        """tool_call_count matches len(tool_calls)."""
        result = self._run()
        assert result.tool_call_count == len(result.tool_calls)


class TestCrossHarnessInterfaceParity:
    """Verify that CustomLoopHarness and AgentHarness produce the same RunResult type."""

    def test_both_return_run_result(self):
        """Both harnesses return the same RunResult type."""
        lg_harness = AgentHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        cl_harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        lg_result = lg_harness.run("test")
        cl_result = cl_harness.run("test")
        assert type(lg_result) is type(cl_result)
        assert isinstance(cl_result, RunResult)

    def test_both_jsonl_records_have_same_keys(self):
        """Both harnesses produce to_jsonl_record() dicts with the same keys."""
        lg_harness = AgentHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        cl_harness = CustomLoopHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        lg_rec = lg_harness.run("test").to_jsonl_record()
        cl_rec = cl_harness.run("test").to_jsonl_record()
        assert set(lg_rec.keys()) == set(cl_rec.keys()), (
            f"Key mismatch: LG extra={set(lg_rec)-set(cl_rec)}, "
            f"CL extra={set(cl_rec)-set(lg_rec)}"
        )


# ─── _extract_reasoning unit tests (Session 11) ───────────────────────────────


class TestExtractReasoningCustomLoop:
    """
    Unit tests for CustomLoopHarness._extract_reasoning() multi-source fallback.

    Mirrors TestExtractReasoning in test_harness_smoke.py to enforce
    cross-harness parity — both harnesses must handle Ollama CoT models
    identically.
    """

    def test_source1_standard_content(self):
        """Source 1: non-empty AIMessage.content is returned directly."""
        msg = AIMessage(content="I will call read_inbox first.", tool_calls=[])
        result = CustomLoopHarness._extract_reasoning(msg, [msg])
        assert result == "I will call read_inbox first."

    def test_source2_reasoning_content_kwarg(self):
        """Source 2: Ollama reasoning_content in additional_kwargs is used when content is empty."""
        msg = AIMessage(content="", tool_calls=[])
        msg.additional_kwargs["reasoning_content"] = "<think>Plan: call read_inbox</think>"
        result = CustomLoopHarness._extract_reasoning(msg, [msg])
        assert "call read_inbox" in result

    def test_source2_thinking_kwarg(self):
        """Source 2: 'thinking' key in additional_kwargs is also accepted."""
        msg = AIMessage(content="", tool_calls=[])
        msg.additional_kwargs["thinking"] = "I should check the inbox first."
        result = CustomLoopHarness._extract_reasoning(msg, [msg])
        assert "inbox" in result

    def test_source3_anthropic_thinking_block(self):
        """Source 3: Anthropic-style thinking block inside content list."""
        msg = AIMessage(
            content=[{"type": "thinking", "thinking": "Step A: read. Step B: send."}],
            tool_calls=[],
        )
        result = CustomLoopHarness._extract_reasoning(msg, [msg])
        assert "Step A" in result

    def test_source4_previous_ai_message_fallback(self):
        """Source 4: falls back to the previous AIMessage when current content is empty."""
        prev = AIMessage(content="I'm going to read the inbox for the user.", tool_calls=[])
        curr = AIMessage(content="", tool_calls=[{"name": "read_inbox", "args": {}, "id": "x1"}])
        result = CustomLoopHarness._extract_reasoning(curr, [prev, curr])
        assert "read the inbox" in result

    def test_source5_no_signal_returns_empty(self):
        """Source 5: returns empty string when no reasoning signal is found anywhere."""
        msg = AIMessage(content="", tool_calls=[])
        result = CustomLoopHarness._extract_reasoning(msg, [msg])
        assert result == ""
