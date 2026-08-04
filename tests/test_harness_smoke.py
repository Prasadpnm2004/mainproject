"""
tests/test_harness_smoke.py

Session 1 — Smoke tests for AgentHarness (no real API calls).

These tests verify that:
  1. AgentHarness can be instantiated without errors.
  2. The graph compiles and responds to a fake LLM stub.
  3. Tool call records are captured correctly.
  4. Pre-tool hooks can block a tool call.
  5. max_iterations guard works.
  6. RunResult.to_jsonl_record() serialises without error.

All tests use a FakeLLM stub so no API keys or network access is needed.
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.harness.langgraph_agent import (
    AgentHarness,
    HarnessConfig,
    RunResult,
    ToolCallRecord,
)


# ─── Fake LLM stub ────────────────────────────────────────────────────────────


class _FakeLLMResponder:
    """
    Minimal stub that mimics a bound ChatModel for use in tests.

    Produces a sequence of canned responses:
      - First call: returns an AIMessage with a tool_call for `echo_tool`
      - Second call: returns a final text answer

    This simulates the agent's "think → call tool → think → answer" loop
    without touching any real LLM API.
    """

    def __init__(self, tool_name: str = "echo_tool", tool_args: dict = None):
        self._call_count = 0
        self._tool_name = tool_name
        self._tool_args = tool_args or {"text": "hello"}

    def bind_tools(self, tools):
        """bind_tools returns self (the stub already knows what tools to call)."""
        return self

    def invoke(self, messages):
        self._call_count += 1
        if self._call_count == 1:
            # First LLM call: decide to use the tool
            ai_msg = AIMessage(
                content="I will use the echo tool to process this.",
                tool_calls=[
                    {
                        "id": "call_fake_001",
                        "name": self._tool_name,
                        "args": self._tool_args,
                    }
                ],
            )
            return ai_msg
        else:
            # Second LLM call: produce final answer
            return AIMessage(content="Task completed. The echo returned: hello")


class _FakeLLMNoTool:
    """Stub that immediately returns a text answer without using any tools."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="I don't need any tools for this.")


class _FakeLLMLoop:
    """Stub that always requests a tool call — exercises max_iterations guard."""

    def __init__(self, tool_name: str = "echo_tool"):
        self._tool_name = tool_name

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(
            content="Calling tool again.",
            tool_calls=[
                {
                    "id": f"call_{len(messages)}",
                    "name": self._tool_name,
                    "args": {"text": "loop"},
                }
            ],
        )


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@tool
def echo_tool(text: str) -> str:
    """A simple echo tool that returns its input. Used for testing only."""
    return f"ECHO: {text}"


@tool
def failing_tool(text: str) -> str:
    """A tool that always raises, to test error handling."""
    raise RuntimeError("Simulated tool failure")


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestHarnessInstantiation:
    def test_basic_instantiation(self):
        """Harness can be created with minimal arguments."""
        harness = AgentHarness(
            llm=_FakeLLMNoTool(),
            tools=[echo_tool],
        )
        assert harness is not None
        assert harness.config.max_iterations == 10

    def test_custom_config(self):
        """Custom HarnessConfig values are respected."""
        config = HarnessConfig(max_iterations=3, capture_reasoning=False)
        harness = AgentHarness(
            llm=_FakeLLMNoTool(),
            tools=[echo_tool],
            config=config,
        )
        assert harness.config.max_iterations == 3
        assert harness.config.capture_reasoning is False

    def test_graph_compiles(self):
        """LangGraph graph compiles without error."""
        harness = AgentHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        # _graph is set in __init__ — if compilation failed it would raise
        assert harness._graph is not None


class TestHarnessRun:
    def test_no_tool_run(self):
        """Agent that doesn't call tools returns a RunResult."""
        harness = AgentHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        result = harness.run("What is 2+2?")
        assert isinstance(result, RunResult)
        assert result.success is True
        assert result.final_answer != ""
        assert result.tool_calls == []

    def test_single_tool_call(self):
        """Agent that calls one tool captures a ToolCallRecord."""
        harness = AgentHarness(llm=_FakeLLMResponder(), tools=[echo_tool])
        result = harness.run("Echo 'hello'")
        assert result.success is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "echo_tool"
        assert tc.tool_args == {"text": "hello"}
        assert "ECHO: hello" in tc.result
        assert tc.latency_ms >= 0

    def test_reasoning_before_captured(self):
        """ToolCallRecord.reasoning_before contains the AIMessage content."""
        harness = AgentHarness(llm=_FakeLLMResponder(), tools=[echo_tool])
        result = harness.run("Echo 'hello'")
        assert len(result.tool_calls) == 1
        assert "echo tool" in result.tool_calls[0].reasoning_before.lower()

    def test_run_id_is_unique(self):
        """Each run() call gets a unique run_id."""
        harness = AgentHarness(llm=_FakeLLMNoTool(), tools=[echo_tool])
        r1 = harness.run("task 1")
        r2 = harness.run("task 2")
        assert r1.run_id != r2.run_id

    def test_max_iterations_guard(self):
        """max_iterations stops the loop before runaway."""
        config = HarnessConfig(max_iterations=2)
        harness = AgentHarness(
            llm=_FakeLLMLoop(),
            tools=[echo_tool],
            config=config,
        )
        result = harness.run("Loop forever")
        # Should complete without hanging; iterations capped
        assert result.total_iterations <= config.max_iterations + 2  # +2 for agent node overhead

    def test_unknown_tool_error_handled(self):
        """Agent requesting an unknown tool gets an error result, not a crash."""
        bad_llm = _FakeLLMResponder(tool_name="nonexistent_tool", tool_args={"x": 1})
        harness = AgentHarness(llm=bad_llm, tools=[echo_tool])
        result = harness.run("Use nonexistent tool")
        assert result.success is True  # harness itself doesn't crash
        assert len(result.tool_calls) == 1
        assert "ERROR" in result.tool_calls[0].result

    def test_failing_tool_error_handled(self):
        """Tool that raises an exception is caught; harness continues."""
        harness = AgentHarness(
            llm=_FakeLLMResponder(tool_name="failing_tool", tool_args={"text": "x"}),
            tools=[failing_tool],
        )
        result = harness.run("Use failing tool")
        assert result.success is True
        assert "ERROR" in result.tool_calls[0].result


class TestPreToolHook:
    def test_hook_can_block_tool(self):
        """pre_tool_hook returning a string blocks execution."""
        blocked_calls = []

        def blocker(record: ToolCallRecord) -> Optional[str]:
            blocked_calls.append(record.tool_name)
            return "[BLOCKED by pre-hook]"

        harness = AgentHarness(
            llm=_FakeLLMResponder(),
            tools=[echo_tool],
            pre_tool_hook=blocker,
        )
        result = harness.run("Echo something")
        assert len(blocked_calls) == 1
        assert "[BLOCKED by pre-hook]" in result.tool_calls[0].result

    def test_hook_can_allow_tool(self):
        """pre_tool_hook returning None allows normal execution."""
        allowed_calls = []

        def observer(record: ToolCallRecord) -> None:
            allowed_calls.append(record.tool_name)
            return None  # allow

        harness = AgentHarness(
            llm=_FakeLLMResponder(),
            tools=[echo_tool],
            pre_tool_hook=observer,
        )
        result = harness.run("Echo something")
        assert len(allowed_calls) == 1
        assert "ECHO: hello" in result.tool_calls[0].result


class TestPostToolHook:
    def test_post_hook_receives_result(self):
        """post_tool_hook is called after execution with result populated."""
        seen_results = []

        def observer(record: ToolCallRecord) -> None:
            seen_results.append(record.result)

        harness = AgentHarness(
            llm=_FakeLLMResponder(),
            tools=[echo_tool],
            post_tool_hook=observer,
        )
        harness.run("Echo something")
        assert len(seen_results) == 1
        assert "ECHO" in seen_results[0]


class TestRunResult:
    def _make_result(self) -> RunResult:
        harness = AgentHarness(llm=_FakeLLMResponder(), tools=[echo_tool])
        return harness.run("Echo 'hello'")

    def test_to_jsonl_record_serialisable(self):
        """RunResult.to_jsonl_record() produces JSON-serialisable output."""
        result = self._make_result()
        record = result.to_jsonl_record(extra={"test_key": "test_value"})
        serialised = json.dumps(record)  # should not raise
        assert "run_id" in record
        assert "test_key" in record
        assert record["test_key"] == "test_value"

    def test_trace_contains_reasoning(self):
        """RunResult.trace includes reasoning_before for every tool call."""
        result = self._make_result()
        for entry in result.trace:
            assert "reasoning_before" in entry
            assert "tool_name" in entry
            assert "tool_args" in entry

    def test_wall_time_positive(self):
        """Run time is recorded and positive."""
        result = self._make_result()
        assert result.wall_time_s > 0
