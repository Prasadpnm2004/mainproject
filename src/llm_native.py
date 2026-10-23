"""
src/llm_native.py

Option B: Native Ollama /api/chat LLM wrapper for PACM.

Replaces LangChain's ChatOpenAI-via-Ollama path with a direct HTTP call to
Ollama's native /api/chat endpoint. This is required because:

  - The OpenAI-compat endpoint (/v1/chat/completions) never exposes reasoning
    tokens in tool-calling responses — message.content is always empty and there
    is no thinking field (confirmed in Session 19).
  - The native /api/chat endpoint DOES expose reasoning via message.thinking
    in the same response object as message.tool_calls (confirmed in Session 21:
    1,611-char reasoning captured from Turn 1 with think=True).
  - Option B (same inference call for reasoning + execution) is necessary to
    avoid the non-determinism risk of Option A (two separate calls where the
    reasoning call and execution call may sample different tool choices).

This wrapper:
  1. Accepts LangChain messages (HumanMessage, AIMessage, SystemMessage, ToolMessage).
  2. Converts them to Ollama's native /api/chat messages[] format.
  3. Calls POST /api/chat with think=True and tools=[...].
  4. Wraps the response into a LangChain AIMessage with:
       - content = message.content (empty for tool-call turns, fine)
       - additional_kwargs["thinking"] = message.thinking  ← reasoning!
       - tool_calls = [{"id":..., "name":..., "args":{...}}]  ← LC format
  5. The existing harness's _extract_reasoning() already reads
     additional_kwargs["thinking"] as Source 3 in its fallback chain.
  6. The existing harness's _instrumented_tool_node already reads
     last_ai_msg.tool_calls in LangChain format (name/args/id).

This approach requires ZERO changes to AgentHarness or CustomLoopHarness —
only the LLM object passed into the harness changes.

Usage:
    from src.llm_native import NativeOllamaLLM
    llm = NativeOllamaLLM(model="qwen3:8b", think=True)
    harness = AgentHarness(llm=llm, tools=tools, config=config)
    result = harness.run("Send a summary email.")
    # result.tool_calls[0].reasoning_before will now contain real reasoning!

Compatibility:
    - Works with both AgentHarness (langgraph_agent.py) and CustomLoopHarness.
    - Existing test suite remains unaffected (tests mock the harness at a higher
      level; they don't inspect the LLM object's internal call mechanism).
    - PACM middleware (PACMMiddleware) receives ToolCallRecord with populated
      reasoning_before — the key end-to-end fix.

Limitations / Known Issues (to address in Session 22):
    - Does not implement streaming (stream=False always).
    - Does not support batching (ainvoke batches fall back to sequential).
    - Temperature, max_tokens, and other model options not yet wired through.
    - bind_tools() is a no-op — tools are re-bound at call time via the
      tools= parameter to harness constructor (stored in self._tools).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional, Sequence

import requests
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_TIMEOUT = 300  # seconds — generous for cold load + long thinking


class NativeOllamaLLM:
    """
    Drop-in replacement for ChatOpenAI that calls Ollama's native /api/chat.

    Exposes reasoning content via AIMessage.additional_kwargs["thinking"] so
    that AgentHarness._extract_reasoning() captures it without modification.

    Args:
        model:      Ollama model name (e.g., "qwen3:8b").
        think:      If True, request thinking/CoT tokens (default: True).
        base_url:   Ollama API base URL (default: "http://localhost:11434").
        timeout:    HTTP request timeout in seconds (default: 300).
        keep_alive: Ollama keep_alive value — -1 keeps model resident
                    indefinitely, "5m" is the default. Use -1 for experiments.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        think: bool = True,
        base_url: str = OLLAMA_BASE,
        timeout: int = DEFAULT_TIMEOUT,
        keep_alive: Any = -1,
    ) -> None:
        self.model = model
        self.think = think
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive

        # Populated by bind_tools()
        self._tools_json: list[dict] = []

    # ── LangChain compatibility interface ─────────────────────────────────────

    def bind_tools(self, tools: Sequence[BaseTool]) -> "NativeOllamaLLM":
        """
        Store tool definitions in Ollama's native format.
        Returns self so the pattern `llm.bind_tools(tools)` works identically.
        """
        bound = NativeOllamaLLM(
            model=self.model,
            think=self.think,
            base_url=self.base_url,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
        )
        bound._tools_json = [self._tool_to_ollama(t) for t in tools]
        return bound

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """
        Invoke the model synchronously. Drop-in for ChatOpenAI.invoke().

        Args:
            messages: LangChain message objects (System/Human/AI/Tool).
            **kwargs: Ignored (for compatibility with bind_tools chains).

        Returns:
            AIMessage with:
              - content: str (empty for tool-call turns)
              - additional_kwargs["thinking"]: reasoning text (think=True only)
              - tool_calls: list of {"id", "name", "args"} dicts (LC format)
        """
        native_messages = [self._lc_msg_to_ollama(m) for m in messages]

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": self.think,
            "messages": native_messages,
        }
        if self._tools_json:
            payload["tools"] = self._tools_json

        logger.debug(
            "NativeOllamaLLM.invoke: model=%s think=%s tools=%d messages=%d",
            self.model, self.think, len(self._tools_json), len(native_messages),
        )

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("NativeOllamaLLM.invoke failed: %s", exc)
            raise

        elapsed = time.perf_counter() - t0
        data = resp.json()
        msg = data.get("message", {})

        logger.debug(
            "NativeOllamaLLM.invoke done in %.1fs | keys=%s | tool_calls=%d",
            elapsed,
            list(msg.keys()),
            len(msg.get("tool_calls", []) or []),
        )

        return self._ollama_msg_to_lc(msg)

    # ── Message format converters ──────────────────────────────────────────────

    @staticmethod
    def _tool_to_ollama(tool: BaseTool) -> dict:
        """Convert a LangChain BaseTool to Ollama's native tool definition format."""
        # BaseTool.args_schema is a Pydantic model; get its JSON schema
        try:
            if hasattr(tool, "args_schema") and tool.args_schema is not None:
                schema = tool.args_schema.model_json_schema()
                # Remove $defs and title from top-level for cleaner Ollama format
                schema.pop("title", None)
                schema.pop("$defs", None)
            else:
                schema = {"type": "object", "properties": {}, "required": []}
        except Exception:
            schema = {"type": "object", "properties": {}, "required": []}

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        }

    @staticmethod
    def _lc_msg_to_ollama(msg: BaseMessage) -> dict:
        """Convert a LangChain BaseMessage to Ollama's native /api/chat message format."""
        if isinstance(msg, SystemMessage):
            return {
                "role": "system",
                "content": msg.content if isinstance(msg.content, str) else "",
            }

        if isinstance(msg, HumanMessage):
            return {
                "role": "user",
                "content": msg.content if isinstance(msg.content, str) else "",
            }

        if isinstance(msg, ToolMessage):
            # Ollama native format for tool results: role="tool" with content
            return {
                "role": "tool",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            }

        if isinstance(msg, AIMessage):
            ollama_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content if isinstance(msg.content, str) else "",
            }
            # Re-attach thinking if present (for multi-turn conversations where
            # we pass back the assistant's prior message)
            thinking = (getattr(msg, "additional_kwargs", {}) or {}).get("thinking", "")
            if thinking:
                ollama_msg["thinking"] = thinking

            # Re-attach tool_calls in Ollama native format if present
            lc_tool_calls = getattr(msg, "tool_calls", []) or []
            if lc_tool_calls:
                ollama_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", str(uuid.uuid4())),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": tc.get("args", {}),
                        },
                    }
                    for tc in lc_tool_calls
                ]
            return ollama_msg

        # Fallback: unknown message type — send as user
        logger.warning(
            "NativeOllamaLLM: unknown message type %s — treating as user",
            type(msg).__name__,
        )
        return {
            "role": "user",
            "content": str(msg.content),
        }

    @staticmethod
    def _ollama_msg_to_lc(msg: dict) -> AIMessage:
        """
        Convert an Ollama native /api/chat response message to a LangChain AIMessage.

        Key mappings:
          - msg["content"]   → AIMessage.content
          - msg["thinking"]  → AIMessage.additional_kwargs["thinking"]  ← reasoning!
          - msg["tool_calls"] → AIMessage.tool_calls (LangChain format)
        """
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""
        native_tool_calls = msg.get("tool_calls", []) or []

        # Convert native tool_calls to LangChain format:
        # Native: [{"id": ..., "type": "function", "function": {"name": ..., "arguments": {...}}}]
        # LangChain: [{"id": ..., "name": ..., "args": {...}, "type": "tool_call"}]
        lc_tool_calls = []
        for tc in native_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            # arguments may be a dict (native) or JSON string (some versions)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(
                        "NativeOllamaLLM: could not parse tool_call arguments as JSON: %r",
                        args[:100],
                    )
                    args = {}
            lc_tool_calls.append({
                "id": tc.get("id", str(uuid.uuid4())),
                "name": fn.get("name", ""),
                "args": args,
                "type": "tool_call",
            })

        additional_kwargs: dict[str, Any] = {}
        if thinking:
            additional_kwargs["thinking"] = thinking
            logger.debug(
                "NativeOllamaLLM: captured %d chars of reasoning", len(thinking)
            )

        ai_msg = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=lc_tool_calls,
        )
        return ai_msg
