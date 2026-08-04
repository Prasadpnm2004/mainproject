"""
src/harness/custom_loop_agent.py

Framework-Free Custom Agent Loop Harness (Framework #2) for PACM experiments.

This module provides a minimal, dependency-light agent host that executes the
standard think → tool-call → think → … → answer loop using a plain Python
while-loop instead of a compiled graph framework.

Purpose (PACM_Research_Guide.md §9.3 / §12):
  The research guide explicitly requires "2-3 different agent 'hosts'" to
  demonstrate cross-framework generalizability — a core differentiator of
  PACM.  This harness is Framework #2 alongside the LangGraph harness.  It
  exists to confirm that PACM's hook interface is framework-agnostic: the
  same PreToolHook / PostToolHook / ToolCallRecord / RunResult types work
  identically here.

Design choices vs. langgraph_agent.py:
  - No LangGraph import at all — pure Python stdlib + langchain-core.
  - LLM invocation uses langchain_core.messages directly; tool dispatch is a
    dict lookup, identical to the LangGraph harness's _instrumented_tool_node.
  - The public interface (AgentHarness → CustomLoopHarness, same HarnessConfig,
    PreToolHook, PostToolHook, RunResult) is deliberately identical so that
    experiments/ scripts can swap frameworks with one line change.

Usage:
    from src.harness.custom_loop_agent import CustomLoopHarness
    from src.harness.langgraph_agent import HarnessConfig
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    harness = CustomLoopHarness(llm=llm, tools=[...], config=HarnessConfig())
    result = harness.run(task="Reply to Alice's email.")
    print(result.trace)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

# Re-export shared data models from langgraph_agent so experiments/ scripts
# only need to change the harness class, not the import path for the types.
from src.harness.langgraph_agent import (
    HarnessConfig,
    PostToolHook,
    PreToolHook,
    RunResult,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)


class CustomLoopHarness:
    """
    Framework-free while-loop agent host with identical interface to AgentHarness.

    The execution loop:
        1.  Build initial messages (system prompt + user task).
        2.  Call LLM → get AIMessage.
        3.  If AIMessage has tool_calls:
              a. For each tool call: run pre_tool_hook, execute tool, run post_tool_hook.
              b. Append ToolMessage results to conversation.
              c. Increment iteration counter; stop if max_iterations reached.
              d. Go to step 2.
        4.  If AIMessage has no tool_calls: extract final answer and return RunResult.

    Args:
        llm: Any LangChain chat model supporting bind_tools().
        tools: Sequence of LangChain BaseTool instances.
        config: HarnessConfig controlling limits and logging.
        pre_tool_hook: Optional; called before each tool execution.
            Return None to allow; return str to block (str becomes tool result).
        post_tool_hook: Optional; called after each tool execution.
        system_prompt_override: If provided, overrides config.system_prompt.
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
        self.tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self.pre_tool_hook: PreToolHook = pre_tool_hook or (lambda r: None)
        self.post_tool_hook: PostToolHook = post_tool_hook or (lambda r: None)
        self.system_prompt = system_prompt_override or self.config.system_prompt

        logging.basicConfig(level=self.config.log_level)
        logger.setLevel(self.config.log_level)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _execute_tool_call(
        self,
        tc: dict[str, Any],
        reasoning_before: str,
        iteration: int,
    ) -> tuple[ToolCallRecord, str]:
        """
        Execute one tool_call dict from an AIMessage, applying hooks.

        Returns:
            (ToolCallRecord, tool_result_string)
        """
        record = ToolCallRecord(
            call_id=tc.get("id", str(uuid.uuid4())),
            tool_name=tc["name"],
            tool_args=tc.get("args", {}),
            reasoning_before=reasoning_before,
            iteration=iteration,
        )

        # Pre-hook: may block the call
        blocked = self.pre_tool_hook(record)

        t0 = time.perf_counter()
        if blocked is not None:
            tool_result = blocked
            logger.info("[PRE-HOOK BLOCK] tool=%s", record.tool_name)
        else:
            tool_impl = self.tools.get(record.tool_name)
            if tool_impl is None:
                tool_result = f"[ERROR] Unknown tool: {record.tool_name}"
                logger.error("Unknown tool requested: %s", record.tool_name)
            else:
                try:
                    raw = tool_impl.invoke(record.tool_args)
                    tool_result = raw if isinstance(raw, str) else str(raw)
                except Exception as exc:
                    tool_result = f"[ERROR] Tool execution failed: {exc}"
                    logger.exception("Tool %s raised an exception", record.tool_name)

        record.latency_ms = (time.perf_counter() - t0) * 1000
        record.result = tool_result

        # Post-hook
        self.post_tool_hook(record)

        logger.debug(
            "Tool %s → %.1fms | result[:80]=%s",
            record.tool_name,
            record.latency_ms,
            tool_result[:80],
        )
        return record, tool_result

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, task: str, extra_context: Optional[str] = None) -> RunResult:
        """
        Execute the agent on a single task synchronously.

        Args:
            task: User-level task description.
            extra_context: Optional additional context appended to the first
                           HumanMessage (used for attack simulation).

        Returns:
            RunResult — same type as AgentHarness.run().
        """
        run_id = str(uuid.uuid4())
        tool_call_records: list[ToolCallRecord] = []

        human_content = task if not extra_context else f"{task}\n\n[Context]\n{extra_context}"

        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=human_content))

        t0 = time.perf_counter()
        error: Optional[str] = None
        final_answer = ""
        iteration = 0
        success = True

        try:
            while True:
                # ── Iteration cap ─────────────────────────────────────────────
                if iteration >= self.config.max_iterations:
                    logger.warning(
                        "max_iterations=%d reached; stopping run %s.",
                        self.config.max_iterations,
                        run_id,
                    )
                    final_answer = (
                        f"[HARNESS] Maximum iterations ({self.config.max_iterations}) reached."
                    )
                    break

                # ── LLM call ──────────────────────────────────────────────────
                response: AIMessage = self.llm.invoke(messages)  # type: ignore[assignment]
                messages.append(response)
                iteration += 1
                logger.debug("LLM response (iter %d): %s", iteration, response.content)

                # ── Routing: tool calls or final answer ───────────────────────
                if not response.tool_calls:
                    # No tool calls → this is the final answer
                    final_answer = (
                        response.content if isinstance(response.content, str) else ""
                    )
                    break

                # ── Execute all tool calls in this response ───────────────────
                for tc in response.tool_calls:
                    reasoning_before = (
                        response.content if isinstance(response.content, str) else ""
                    )
                    record, tool_result = self._execute_tool_call(
                        tc, reasoning_before, iteration
                    )
                    tool_call_records.append(record)

                    # Append ToolMessage so LLM sees the result
                    from langchain_core.messages import ToolMessage
                    messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=record.call_id,
                        )
                    )

        except Exception as exc:
            logger.exception("CustomLoopHarness.run() caught exception for run %s", run_id)
            success = False
            error = str(exc)

        wall_time = time.perf_counter() - t0

        return RunResult(
            run_id=run_id,
            task=task,
            success=success,
            final_answer=final_answer,
            tool_calls=tool_call_records,
            messages=messages,
            total_iterations=iteration,
            wall_time_s=wall_time,
            error=error,
        )

    async def arun(self, task: str, extra_context: Optional[str] = None) -> RunResult:
        """Async wrapper for use in async test contexts."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run, task, extra_context
        )
