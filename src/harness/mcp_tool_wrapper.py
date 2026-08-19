"""
src/harness/mcp_tool_wrapper.py

MCP Tool Wrapper Stub — Session 2 (refactored in Session 3)

Provides a standardized, in-process wrapper that formats LangChain tool
inputs/outputs with MCP-compliant JSON-RPC metadata and attaches trust
attribute fields to every tool call record.

Purpose (PACM_Research_Guide.md §9.1 step 4 / §7 / §12):
  PACM's friction policy combines a divergence score with a lightweight
  MCP server/tool trust score derived from registry reputation signals
  (publisher verification, metadata consistency, update recency) — grounded
  in Li & Gao (2510.16558) and Hasan et al. (2506.13538).

  This module is NOT a live MCP server (out of scope until Session 3–4).
  Instead it is an in-process wrapper that:
    1. Attaches MCP-style JSON-RPC metadata to every tool call for logging.
    2. Records trust attributes (publisher, recency, verified) per tool.
    3. Surfaces a trust_score float (0.0–1.0) for use by the future PACM
       friction policy.
    4. Provides a WrappedTool.invoke() shim that can be dropped into
       AgentHarness or CustomLoopHarness as a replacement for a raw BaseTool.

Scope guardrails (Session 2):
  - NO live network MCP servers, NO external API calls.
  - Trust scoring is purely static rule-based from metadata (no ML).
  - PACM divergence calculation is NOT implemented here (Session 3 scope).

References:
  - Li & Gao (2025/2026) — First Look at MCP Ecosystem Security (2510.16558)
  - Hasan et al. (2026) — MCP at First Glance (2506.13538)
  - MCP JSON-RPC specification: https://spec.modelcontextprotocol.io/
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ─── Trust metadata ───────────────────────────────────────────────────────────


@dataclass
class MCPToolMetadata:
    """
    Lightweight trust attribute record for a single MCP tool/server.

    Fields are inspired by the empirical registry features identified in:
      - Li & Gao (2510.16558): publisher_verified, update_recency_days,
        metadata_consistency
      - Hasan et al. (2506.13538): description_matches_behavior, tool_origin

    These fields will feed into PACM's trust-scoring module (Session 5 scope).
    For now they are populated statically from a metadata dict.

    trust_score: float in [0.0, 1.0] computed by compute_trust_score().
      1.0 = fully trusted, 0.0 = not trusted.
    """

    tool_name: str
    server_name: str = "local_stub"
    publisher: str = "unknown"
    publisher_verified: bool = False
    description: str = ""
    # Days since the tool definition was last updated (lower = more recent)
    update_recency_days: int = 0
    # Whether the tool's description textually matches what it actually does
    metadata_consistency: float = 1.0  # 0.0–1.0
    # Whether this tool originates from a vetted internal registry
    tool_origin: str = "internal"  # "internal" | "third_party" | "user_provided"

    # Computed fields — call compute_trust_score() to populate
    trust_score: float = field(default=0.0, init=False)
    trust_reasoning: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.compute_trust_score()

    def compute_trust_score(self) -> float:
        """
        Compute a simple additive trust score from metadata features.

        Scoring rubric (grounded in Li & Gao §3 feature taxonomy):
          +0.40  publisher_verified
          +0.20  tool_origin == "internal"
          +0.20  update_recency_days <= 30  (recently maintained)
          +0.20  metadata_consistency >= 0.8

        Range: 0.0 (no positive signals) – 1.0 (all signals present).

        This is intentionally simple — a full ML-based or registry-crawl-based
        trust scorer is planned for Session 5 / Week 7.
        """
        score = 0.0
        reasons = []

        if self.publisher_verified:
            score += 0.40
            reasons.append("publisher_verified(+0.40)")

        if self.tool_origin == "internal":
            score += 0.20
            reasons.append("internal_origin(+0.20)")

        if self.update_recency_days <= 30:
            score += 0.20
            reasons.append(f"recency<={self.update_recency_days}d(+0.20)")

        if self.metadata_consistency >= 0.8:
            score += 0.20
            reasons.append(f"consistency={self.metadata_consistency:.2f}(+0.20)")

        self.trust_score = min(score, 1.0)
        self.trust_reasoning = " | ".join(reasons) if reasons else "no_positive_signals"
        return self.trust_score

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation for JSONL logging."""
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "publisher": self.publisher,
            "publisher_verified": self.publisher_verified,
            "tool_origin": self.tool_origin,
            "update_recency_days": self.update_recency_days,
            "metadata_consistency": self.metadata_consistency,
            "trust_score": round(self.trust_score, 4),
            "trust_reasoning": self.trust_reasoning,
        }


# ─── JSON-RPC envelope ────────────────────────────────────────────────────────


def make_jsonrpc_request(
    tool_name: str,
    tool_args: dict[str, Any],
    call_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Format a tool call as a MCP-style JSON-RPC 2.0 request envelope.

    This is the wire format that a real MCP client would send to an MCP server.
    We log this alongside every tool call so that future analysis can examine
    the full MCP-layer interaction (including tool poisoning vectors that live
    in the method/params fields).

    Args:
        tool_name: The tool to call (maps to JSON-RPC method).
        tool_args: The argument dict (maps to JSON-RPC params).
        call_id: Optional idempotency key; auto-generated if not supplied.

    Returns:
        JSON-RPC 2.0 request dict.

    Reference: https://spec.modelcontextprotocol.io/specification/server/tools/
    """
    return {
        "jsonrpc": "2.0",
        "id": call_id or str(uuid.uuid4()),
        "method": f"tools/call",
        "params": {
            "name": tool_name,
            "arguments": tool_args,
        },
    }


def make_jsonrpc_response(
    call_id: str,
    result: str,
    is_error: bool = False,
) -> dict[str, Any]:
    """
    Format a tool result as a MCP-style JSON-RPC 2.0 response envelope.

    Args:
        call_id: Must match the id from the corresponding request.
        result: Tool output string.
        is_error: True if the tool raised an exception.

    Returns:
        JSON-RPC 2.0 response dict.
    """
    if is_error:
        return {
            "jsonrpc": "2.0",
            "id": call_id,
            "error": {
                "code": -32000,
                "message": result,
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": call_id,
        "result": {
            "content": [{"type": "text", "text": result}],
            "isError": False,
        },
    }


# ─── Wrapped tool ─────────────────────────────────────────────────────────────


@dataclass
class MCPToolCallLog:
    """
    Full log record for one MCP-wrapped tool invocation.

    Combines:
      - JSON-RPC request/response envelopes (for protocol-level analysis)
      - Trust metadata (for PACM trust-scoring module)
      - Timing data
    """

    call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    jsonrpc_request: dict[str, Any]
    jsonrpc_response: dict[str, Any]
    metadata: MCPToolMetadata
    latency_ms: float
    is_error: bool
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "jsonrpc_request": self.jsonrpc_request,
            "jsonrpc_response": self.jsonrpc_response,
            "trust": self.metadata.to_dict(),
            "latency_ms": round(self.latency_ms, 3),
            "is_error": self.is_error,
            "timestamp_utc": self.timestamp_utc,
        }


class MCPWrappedTool(BaseTool):
    """
    BaseTool subclass that wraps any LangChain BaseTool with MCP-layer logging
    and trust metadata. Session 3: refactored from plain class to proper
    BaseTool subclass for full type safety and native bind_tools() compatibility.

    Every tool call is logged to self.call_log — a list of MCPToolCallLog
    records exportable to JSONL for PACM divergence analysis.

    Args:
        wrapped_tool: The underlying LangChain BaseTool to wrap.
        metadata: MCPToolMetadata with trust attributes for this tool.

    Example::

        from langchain_core.tools import tool

        @tool
        def send_email(to: str, subject: str, body: str) -> str:
            \"\"\"Send an email.\"\"\"
            return f"Sent to {to}"

        wrapped = MCPWrappedTool(
            wrapped_tool=send_email,
            metadata=MCPToolMetadata(
                tool_name="send_email",
                publisher="trusted_corp",
                publisher_verified=True,
                tool_origin="internal",
                update_recency_days=5,
                metadata_consistency=0.95,
            ),
        )
        result = wrapped.invoke({"to": "a@b.com", "subject": "Hi", "body": "Hello"})
    """

    # ── Pydantic fields ───────────────────────────────────────────────────────
    wrapped_tool: Any = Field(..., description="Underlying BaseTool instance.")
    mcp_metadata: Any = Field(..., description="MCPToolMetadata for this tool.")
    call_log: list = Field(default_factory=list, description="MCPToolCallLog records.")

    # name and description are required by BaseTool; populated in model_validator
    name: str = Field(default="")
    description: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _populate_from_wrapped(self) -> "MCPWrappedTool":
        """Copy name and description from the wrapped tool."""
        if not self.name:
            self.name = self.wrapped_tool.name
        if not self.description:
            self.description = self.wrapped_tool.description
        return self

    # ── BaseTool abstract method ──────────────────────────────────────────────

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the wrapped tool. This is the abstract method required by BaseTool.
        MCP logging happens here so it fires on both invoke() and direct _run() calls.
        """
        from datetime import datetime, timezone

        # Normalise args: _run may be called with positional or keyword args
        if args and not kwargs:
            # LangChain sometimes passes a single dict positionally
            tool_args: dict[str, Any] = args[0] if isinstance(args[0], dict) else {"input": args[0]}
        else:
            tool_args = dict(kwargs)

        call_id = str(uuid.uuid4())
        jsonrpc_req = make_jsonrpc_request(self.name, tool_args, call_id)
        is_error = False

        t0 = time.perf_counter()
        try:
            raw_result = self.wrapped_tool.invoke(tool_args)
            result = raw_result if isinstance(raw_result, str) else str(raw_result)
        except Exception as exc:
            result = f"[ERROR] {exc}"
            is_error = True
            logger.exception("MCPWrappedTool %s raised an exception", self.name)
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000

        jsonrpc_resp = make_jsonrpc_response(call_id, result, is_error=is_error)

        log_entry = MCPToolCallLog(
            call_id=call_id,
            tool_name=self.name,
            tool_args=tool_args,
            jsonrpc_request=jsonrpc_req,
            jsonrpc_response=jsonrpc_resp,
            metadata=self.mcp_metadata,
            latency_ms=latency_ms,
            is_error=is_error,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.call_log.append(log_entry)

        logger.debug(
            "MCP tool %s | trust=%.2f | latency=%.1fms | error=%s",
            self.name,
            self.mcp_metadata.trust_score,
            latency_ms,
            is_error,
        )
        return result

    def export_call_log(self) -> list[dict[str, Any]]:
        """Return all call log entries as serialisable dicts."""
        return [entry.to_dict() for entry in self.call_log]



# ─── Registry helpers ─────────────────────────────────────────────────────────


class MCPToolRegistry:
    """
    In-process registry mapping tool names to MCPToolMetadata.

    In a real deployment this would be backed by a live MCP registry crawl.
    For Session 2 it is populated statically from a list of metadata dicts.

    Usage:
        registry = MCPToolRegistry.from_list([
            {"tool_name": "send_email", "publisher_verified": True, ...},
            {"tool_name": "read_inbox", "tool_origin": "third_party", ...},
        ])
        meta = registry.get("send_email")
        print(meta.trust_score)
    """

    def __init__(self) -> None:
        self._store: dict[str, MCPToolMetadata] = {}

    def register(self, metadata: MCPToolMetadata) -> None:
        """Register a tool's trust metadata."""
        self._store[metadata.tool_name] = metadata
        logger.debug(
            "Registered tool %s with trust_score=%.2f", metadata.tool_name, metadata.trust_score
        )

    def get(self, tool_name: str) -> MCPToolMetadata:
        """
        Retrieve metadata for a tool.  Returns a low-trust placeholder if unknown.
        """
        if tool_name not in self._store:
            logger.warning(
                "Tool %s not in registry — assigning zero-trust placeholder.", tool_name
            )
            return MCPToolMetadata(
                tool_name=tool_name,
                publisher="unknown",
                publisher_verified=False,
                tool_origin="user_provided",
                update_recency_days=999,
                metadata_consistency=0.0,
            )
        return self._store[tool_name]

    @classmethod
    def from_list(cls, entries: list[dict[str, Any]]) -> "MCPToolRegistry":
        """
        Convenience constructor from a list of metadata dicts.

        Each dict is passed as kwargs to MCPToolMetadata.  Unknown keys are ignored.
        """
        registry = cls()
        valid_fields = {f.name for f in MCPToolMetadata.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for entry in entries:
            filtered = {k: v for k, v in entry.items() if k in valid_fields}
            registry.register(MCPToolMetadata(**filtered))
        return registry

    def wrap_tools(self, tools: list[BaseTool]) -> list[MCPWrappedTool]:
        """
        Wrap a list of BaseTool instances using their registered metadata.
        Unknown tools get zero-trust placeholder metadata.
        Returns proper BaseTool subclass instances (no type: ignore needed).
        """
        return [
            MCPWrappedTool(wrapped_tool=t, mcp_metadata=self.get(t.name))
            for t in tools
        ]


# ─── Default stub registry ─────────────────────────────────────────────────────
# Used by experiments/ scripts as the default trust configuration.

DEFAULT_STUB_REGISTRY = MCPToolRegistry.from_list([
    {
        "tool_name": "send_email",
        "server_name": "email_server",
        "publisher": "internal_team",
        "publisher_verified": True,
        "tool_origin": "internal",
        "update_recency_days": 7,
        "metadata_consistency": 0.95,
        "description": "Send an email to a recipient.",
    },
    {
        "tool_name": "read_inbox",
        "server_name": "email_server",
        "publisher": "internal_team",
        "publisher_verified": True,
        "tool_origin": "internal",
        "update_recency_days": 7,
        "metadata_consistency": 0.95,
        "description": "Read the most recent messages from the inbox.",
    },
    {
        "tool_name": "get_calendar",
        "server_name": "calendar_server",
        "publisher": "internal_team",
        "publisher_verified": True,
        "tool_origin": "internal",
        "update_recency_days": 14,
        "metadata_consistency": 0.90,
        "description": "Retrieve calendar events for a specific date.",
    },
    # Low-trust external tool (simulates a third-party MCP server found via registry)
    {
        "tool_name": "web_search",
        "server_name": "external_search_mcp",
        "publisher": "unknown_vendor",
        "publisher_verified": False,
        "tool_origin": "third_party",
        "update_recency_days": 180,
        "metadata_consistency": 0.55,
        "description": "Search the web.",
    },
])
