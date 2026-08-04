"""
src/harness/langgraph_agent.py

LangGraph Tool-Calling Agent Harness for PACM experiments.

This module provides a modular, instrumentable agent host built on LangGraph.
It is designed as the primary execution host for Session 1 baseline runs and
as the main integration point for PACM middleware hooks in later sessions.

Design principles (grounded in PACM_Research_Guide.md §9.3 and §12):
  - Cross-framework uniform interface: `AgentHarness.run()` returns a
    `RunResult` regardless of which framework is underneath.
  - Reasoning-trace capture: every LLM message and every tool call is
    recorded so future PACM modules can inspect the plan→action transition.
  - Hook points: `pre_tool_hook` and `post_tool_hook` are no-ops here but
    will be wired to PACM middleware in Weeks 5-8.
  - Cost guard: `max_iterations` hard-caps the tool-call loop, preventing
    runaway API spend.

Usage:
    from src.harness.langgraph_agent import AgentHarness, HarnessConfig
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    harness = AgentHarness(llm=llm, tools=[...], config=HarnessConfig())
    result = await harness.run(task="Send a summary email to Alice.")
    print(result.trace)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

logger = logging.getLogger(__name__)

# ─── Data models ──────────────────────────────────────────────────────────────


@dataclass
class HarnessConfig:
    """Configuration for AgentHarness.

    Attributes:
        max_iterations: Hard cap on tool-call loop iterations. Prevents
            runaway costs during bulk experiment runs (PACM_Research_Guide §13).
        system_prompt: Optional system message prepended to every run.
        capture_reasoning: If True, persist raw AIMessage content (which may
            include CoT / reasoning traces) in RunResult.trace. Leave True for
            all PACM experiments — the reasoning trace is the primary PACM signal.
        log_level: Python logging level for harness-internal messages.
    """

    max_iterations: int = 10
    system_prompt: str = (
        "You are a helpful assistant with access to tools. "
        "Think step by step before each tool call. "
        "Always explain your reasoning before calling a tool."
    )
    capture_reasoning: bool = True
    log_level: int = logging.INFO


@dataclass
class ToolCallRecord:
    """Captures a single tool invocation for later analysis.

    Fields mirror what PACM will need in Weeks 5-8:
      - reasoning_before: the AIMessage content immediately *before* the call
        (this is the 'plan' side of the plan→action check).
      - tool_name / tool_args: the 'action' side.
      - result: raw string returned by the tool.
    """

    call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    reasoning_before: str  # AIMessage.content that triggered this call
    result: Optional[str] = None
    latency_ms: float = 0.0
    iteration: int = 0


@dataclass
class RunResult:
    """Unified result object returned by every AgentHarness.run() call.

    This is the common interface contract; future harness implementations
    (custom loop, AutoGen) must produce the same type so experiments/ scripts
    don't need framework-specific logic.
    """

    run_id: str
    task: str
    success: bool
    final_answer: str
    tool_calls: list[ToolCallRecord]
    messages: list[BaseMessage]
    total_iterations: int
    wall_time_s: float
    error: Optional[str] = None

    @property
    def tool_call_count(self) -> int:
        """Convenience count of tool calls made in this run."""
        return len(self.tool_calls)

    @property
    def trace(self) -> list[dict[str, Any]]:
        """Serialisable reasoning trace for logging to JSONL."""
        return [
            {
                "iteration": tc.iteration,
                "call_id": tc.call_id,
                "tool_name": tc.tool_name,
                "tool_args": tc.tool_args,
                "reasoning_before": tc.reasoning_before,
                "result": tc.result,
                "latency_ms": tc.latency_ms,
            }
            for tc in self.tool_calls
        ]

    def to_jsonl_record(self, extra: Optional[dict] = None) -> dict[str, Any]:
        """Flatten into a JSONL-serialisable dict for experiments/logs."""
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "task": self.task,
            "success": self.success,
            "final_answer": self.final_answer,
            "total_iterations": self.total_iterations,
            "wall_time_s": round(self.wall_time_s, 3),
            "tool_call_count": len(self.tool_calls),
            "trace": self.trace,
            "error": self.error,
        }
        if extra:
            record.update(extra)
        return record


# ─── LangGraph state ──────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph node graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    tool_call_records: list[ToolCallRecord]


# ─── Hook type aliases ────────────────────────────────────────────────────────

PreToolHook = Callable[[ToolCallRecord], Optional[str]]
"""
Called immediately *before* a tool executes.

Args:
    record: ToolCallRecord with tool_name, tool_args, reasoning_before populated
            (result field is still None at this point).

Returns:
    None  → allow the call to proceed normally.
    str   → block the call; the string is used as the tool's synthetic result
            (so the agent sees a refusal message rather than crashing).

This is the primary PACM injection point (Weeks 5-8).
"""

PostToolHook = Callable[[ToolCallRecord], None]
"""
Called immediately *after* a tool executes.

Args:
    record: ToolCallRecord with result populated.

Returns:
    None (hooks may have side effects but cannot modify the result).
"""


# ─── Harness ──────────────────────────────────────────────────────────────────


class AgentHarness:
    """
    LangGraph-based tool-calling agent host.

    The harness compiles a two-node LangGraph:
      agent_node  → ToolNode  → (loop back or END)

    It intercepts tool execution at the ToolNode level via a custom
    _instrumented_tool_node so that pre_tool_hook / post_tool_hook can be
    injected without modifying AgentDojo's tool implementations.

    Args:
        llm: Any LangChain chat model. Must support tool-calling (bind_tools).
        tools: Sequence of LangChain BaseTool instances.
        config: HarnessConfig controlling limits and logging.
        pre_tool_hook: Optional; called before each tool execution.
        post_tool_hook: Optional; called after each tool execution.
        system_prompt_override: If supplied, overrides config.system_prompt.
    """

    def __init__(
        self,
        llm: Any,
        tools: Sequence[BaseTool],
        config: Optional[HarnessConfig] = None,
        pre_tool_hook: Optional[PreToolHook] = None,
        post_tool_hook: Optional[PostToolHook] = None,
        system_prompt_override: Optional[str] = None,
    ) -> None:
        self.config = config or HarnessConfig()
        self.llm = llm.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        self.pre_tool_hook: PreToolHook = pre_tool_hook or (lambda r: None)
        self.post_tool_hook: PostToolHook = post_tool_hook or (lambda r: None)
        self.system_prompt = system_prompt_override or self.config.system_prompt

        logging.basicConfig(level=self.config.log_level)
        logger.setLevel(self.config.log_level)

        self._graph = self._build_graph()
        self._current_run_records: list[ToolCallRecord] = []

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Compile the LangGraph agent loop."""
        builder = StateGraph(AgentState)

        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._instrumented_tool_node)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        builder.add_edge("tools", "agent")

        return builder.compile()

    # ── Node implementations ──────────────────────────────────────────────────

    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        """Invoke the LLM and append its response to the message list."""
        messages = state["messages"]
        iterations = state.get("iterations", 0)

        if iterations >= self.config.max_iterations:
            # Synthetic stop — inject a stop message so the graph can exit cleanly
            logger.warning(
                "max_iterations=%d reached; forcing stop.", self.config.max_iterations
            )
            stop_msg = AIMessage(
                content=(
                    f"[HARNESS] Maximum iterations ({self.config.max_iterations}) "
                    "reached. Stopping."
                )
            )
            return {"messages": [stop_msg], "iterations": iterations + 1}

        response = self.llm.invoke(messages)
        logger.debug("Agent response (iter %d): %s", iterations, response.content)
        return {"messages": [response], "iterations": iterations + 1}

    def _instrumented_tool_node(self, state: AgentState) -> dict[str, Any]:
        """
        Execute tool calls from the latest AIMessage, applying hooks.

        For each tool_call in the last AIMessage:
          1. Build a ToolCallRecord capturing reasoning_before.
          2. Call pre_tool_hook — if it returns a string, skip real execution.
          3. Execute the tool (or use the hook's synthetic result).
          4. Call post_tool_hook.
          5. Append a ToolMessage to the state.
        """
        messages = state["messages"]
        last_ai_msg: AIMessage = messages[-1]  # type: ignore[assignment]
        reasoning_before = (
            last_ai_msg.content if isinstance(last_ai_msg.content, str) else ""
        )

        tool_messages: list[ToolMessage] = []
        records: list[ToolCallRecord] = list(state.get("tool_call_records", []))

        for tc in last_ai_msg.tool_calls:  # type: ignore[union-attr]
            record = ToolCallRecord(
                call_id=tc.get("id", str(uuid.uuid4())),
                tool_name=tc["name"],
                tool_args=tc.get("args", {}),
                reasoning_before=reasoning_before,
                iteration=state.get("iterations", 0),
            )

            # ── Pre-hook ──────────────────────────────────────────────────────
            blocked_result = self.pre_tool_hook(record)

            t_start = time.perf_counter()
            if blocked_result is not None:
                # Hook blocked the call
                tool_result = blocked_result
                logger.info(
                    "[PRE-HOOK BLOCK] tool=%s args=%s", record.tool_name, record.tool_args
                )
            else:
                # Normal execution
                tool_impl = self.tools.get(record.tool_name)
                if tool_impl is None:
                    tool_result = f"[ERROR] Unknown tool: {record.tool_name}"
                    logger.error("Unknown tool requested: %s", record.tool_name)
                else:
                    try:
                        tool_result = tool_impl.invoke(record.tool_args)
                        if not isinstance(tool_result, str):
                            tool_result = str(tool_result)
                    except Exception as exc:
                        tool_result = f"[ERROR] Tool execution failed: {exc}"
                        logger.exception(
                            "Tool %s raised an exception", record.tool_name
                        )

            record.latency_ms = (time.perf_counter() - t_start) * 1000
            record.result = tool_result
            records.append(record)
            self._current_run_records.append(record)

            # ── Post-hook ─────────────────────────────────────────────────────
            self.post_tool_hook(record)

            tool_messages.append(
                ToolMessage(content=tool_result, tool_call_id=record.call_id)
            )
            logger.debug(
                "Tool %s executed in %.1fms → %s",
                record.tool_name,
                record.latency_ms,
                tool_result[:120],
            )

        return {"messages": tool_messages, "tool_call_records": records}

    # ── Graph routing ─────────────────────────────────────────────────────────

    @staticmethod
    def _should_continue(state: AgentState) -> str:
        """Route: continue to tools if the last message has tool calls, else end."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "continue"
        return "end"

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, task: str, extra_context: Optional[str] = None) -> RunResult:
        """
        Execute the agent on a single task synchronously.

        Args:
            task: The user-level task description string.
            extra_context: Optional additional context to append to the
                           first HumanMessage (e.g., injected tool content
                           for attack simulations).

        Returns:
            RunResult with full trace, messages, and timing.
        """
        run_id = str(uuid.uuid4())
        self._current_run_records = []

        human_content = task
        if extra_context:
            human_content = f"{task}\n\n[Context]\n{extra_context}"

        init_messages: list[BaseMessage] = [
            HumanMessage(content=human_content)
        ]

        # Prepend system prompt as a HumanMessage wrapper if model doesn't
        # support SystemMessage (some older models/providers don't).
        if self.system_prompt:
            from langchain_core.messages import SystemMessage
            init_messages = [SystemMessage(content=self.system_prompt)] + init_messages

        initial_state: AgentState = {
            "messages": init_messages,
            "iterations": 0,
            "tool_call_records": [],
        }

        t0 = time.perf_counter()
        error: Optional[str] = None
        final_state: Optional[AgentState] = None

        try:
            final_state = self._graph.invoke(initial_state)
            success = True
        except Exception as exc:
            logger.exception("AgentHarness.run() caught exception for run_id=%s", run_id)
            success = False
            error = str(exc)

        wall_time = time.perf_counter() - t0

        # Extract final answer from last AIMessage
        final_answer = ""
        if final_state:
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                    # Skip the harness stop message
                    if not msg.content.startswith("[HARNESS]"):
                        final_answer = msg.content
                        break

        return RunResult(
            run_id=run_id,
            task=task,
            success=success,
            final_answer=final_answer,
            tool_calls=list(self._current_run_records),
            messages=final_state["messages"] if final_state else init_messages,
            total_iterations=final_state.get("iterations", 0) if final_state else 0,
            wall_time_s=wall_time,
            error=error,
        )

    async def arun(self, task: str, extra_context: Optional[str] = None) -> RunResult:
        """Async wrapper around run() for use in async test contexts."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run, task, extra_context
        )
