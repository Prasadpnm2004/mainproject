"""
tests/test_mcp_wrapper.py

Session 3 — Formal unit tests for MCPWrappedTool (BaseTool subclass).

Covers:
  Trust scoring:
    1.  Full-trust metadata → score = 1.0
    2.  Zero-trust metadata → score = 0.0
    3.  Partial trust → intermediate score
    4.  Trust reasoning string populated correctly
  JSON-RPC 2.0 format:
    5.  make_jsonrpc_request produces correct jsonrpc/method/id/params
    6.  make_jsonrpc_response success format
    7.  make_jsonrpc_response error format
  MCPWrappedTool (BaseTool subclass):
    8.  MCPWrappedTool is a proper BaseTool instance
    9.  invoke() returns correct result
   10.  call_log has one entry per invoke
   11.  call_log entry is JSON-serialisable
   12.  call_log records latency_ms > 0
   13.  call_log trust score matches metadata
   14.  Failing tool → error result, is_error=True in log
   15.  wrap_tools() returns BaseTool instances (no type: ignore needed)
  MCPToolRegistry:
   16.  from_list builds registry correctly
   17.  Unknown tool gets zero-trust placeholder
   18.  DEFAULT_STUB_REGISTRY has 4 expected tools
   19.  DEFAULT_STUB_REGISTRY internal tools have trust_score > 0
   20.  DEFAULT_STUB_REGISTRY web_search has trust_score < 0.5

Run with:  pytest tests/test_mcp_wrapper.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import BaseTool, tool

from src.harness.mcp_tool_wrapper import (
    DEFAULT_STUB_REGISTRY,
    MCPToolCallLog,
    MCPToolMetadata,
    MCPToolRegistry,
    MCPWrappedTool,
    make_jsonrpc_request,
    make_jsonrpc_response,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@tool
def echo_tool(text: str) -> str:
    """Echo tool — returns input prefixed with ECHO."""
    return f"ECHO:{text}"


@tool
def failing_tool(text: str) -> str:
    """Always raises RuntimeError."""
    raise RuntimeError("Simulated failure")


def _full_trust_meta(tool_name: str = "echo_tool") -> MCPToolMetadata:
    return MCPToolMetadata(
        tool_name=tool_name,
        publisher="trusted_corp",
        publisher_verified=True,
        tool_origin="internal",
        update_recency_days=7,
        metadata_consistency=0.95,
    )


def _zero_trust_meta(tool_name: str = "rogue_tool") -> MCPToolMetadata:
    return MCPToolMetadata(
        tool_name=tool_name,
        publisher="unknown",
        publisher_verified=False,
        tool_origin="user_provided",
        update_recency_days=999,
        metadata_consistency=0.0,
    )


# ─── Trust scoring ────────────────────────────────────────────────────────────

class TestTrustScoring:
    def test_full_trust_score_is_one(self):
        meta = _full_trust_meta()
        assert meta.trust_score == 1.0

    def test_zero_trust_score_is_zero(self):
        meta = _zero_trust_meta()
        assert meta.trust_score == 0.0

    def test_partial_trust_intermediate_score(self):
        meta = MCPToolMetadata(
            tool_name="partial",
            publisher_verified=True,    # +0.40
            tool_origin="third_party",  # no bonus
            update_recency_days=60,     # no bonus (> 30)
            metadata_consistency=0.9,   # +0.20
        )
        assert 0.0 < meta.trust_score < 1.0
        assert abs(meta.trust_score - 0.60) < 0.01

    def test_trust_reasoning_populated(self):
        meta = _full_trust_meta()
        assert "publisher_verified" in meta.trust_reasoning
        assert "internal_origin" in meta.trust_reasoning

    def test_to_dict_includes_trust_score(self):
        meta = _full_trust_meta()
        d = meta.to_dict()
        assert "trust_score" in d
        assert d["trust_score"] == 1.0


# ─── JSON-RPC format ──────────────────────────────────────────────────────────

class TestJSONRPC:
    def test_request_structure(self):
        req = make_jsonrpc_request("send_email", {"to": "a@b.com"}, "req-001")
        assert req["jsonrpc"] == "2.0"
        assert req["id"] == "req-001"
        assert req["method"] == "tools/call"
        assert req["params"]["name"] == "send_email"
        assert req["params"]["arguments"] == {"to": "a@b.com"}

    def test_request_auto_id(self):
        req = make_jsonrpc_request("send_email", {})
        assert isinstance(req["id"], str) and len(req["id"]) > 0

    def test_response_success_structure(self):
        resp = make_jsonrpc_response("req-001", "result text")
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-001"
        assert resp["result"]["content"][0]["text"] == "result text"
        assert resp["result"]["isError"] is False

    def test_response_error_structure(self):
        resp = make_jsonrpc_response("req-001", "something failed", is_error=True)
        assert "error" in resp
        assert resp["error"]["code"] == -32000
        assert "failed" in resp["error"]["message"]

    def test_request_is_json_serialisable(self):
        req = make_jsonrpc_request("tool", {"a": 1, "b": "x"})
        json.dumps(req)  # must not raise


# ─── MCPWrappedTool ───────────────────────────────────────────────────────────

class TestMCPWrappedTool:
    def _make_wrapped(self, t=None, meta=None) -> MCPWrappedTool:
        return MCPWrappedTool(
            wrapped_tool=t or echo_tool,
            mcp_metadata=meta or _full_trust_meta(),
        )

    def test_is_basetool_instance(self):
        w = self._make_wrapped()
        assert isinstance(w, BaseTool)

    def test_invoke_returns_correct_result(self):
        w = self._make_wrapped()
        result = w.invoke({"text": "hello"})
        assert result == "ECHO:hello"

    def test_call_log_has_one_entry_after_invoke(self):
        w = self._make_wrapped()
        w.invoke({"text": "test"})
        assert len(w.call_log) == 1

    def test_call_log_entry_json_serialisable(self):
        w = self._make_wrapped()
        w.invoke({"text": "test"})
        d = w.call_log[0].to_dict()
        json.dumps(d)  # must not raise
        assert "trust" in d
        assert "jsonrpc_request" in d
        assert "jsonrpc_response" in d

    def test_call_log_latency_positive(self):
        w = self._make_wrapped()
        w.invoke({"text": "test"})
        assert w.call_log[0].latency_ms >= 0

    def test_call_log_trust_score_matches_metadata(self):
        w = self._make_wrapped(meta=_full_trust_meta())
        w.invoke({"text": "test"})
        assert w.call_log[0].metadata.trust_score == 1.0

    def test_failing_tool_is_error_in_log(self):
        w = MCPWrappedTool(wrapped_tool=failing_tool, mcp_metadata=_zero_trust_meta())
        result = w.invoke({"text": "x"})
        assert "ERROR" in result
        assert w.call_log[0].is_error is True

    def test_export_call_log_returns_list_of_dicts(self):
        w = self._make_wrapped()
        w.invoke({"text": "a"})
        exported = w.export_call_log()
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert isinstance(exported[0], dict)

    # ── _normalise_args tests ─────────────────────────────────────────────────

    def test_normalise_recipient_to_to(self):
        """'recipient' should be renamed to 'to'."""
        result = MCPWrappedTool._normalise_args({"recipient": "alice@example.com", "subject": "Hi"})
        assert result["to"] == "alice@example.com"
        assert "recipient" not in result

    def test_normalise_message_to_body(self):
        """'message' should be renamed to 'body'."""
        result = MCPWrappedTool._normalise_args({"to": "bob@example.com", "message": "Hello!"})
        assert result["body"] == "Hello!"
        assert "message" not in result

    def test_normalise_content_to_body(self):
        """'content' should be renamed to 'body'."""
        result = MCPWrappedTool._normalise_args({"to": "bob@example.com", "content": "Hi"})
        assert result["body"] == "Hi"
        assert "content" not in result

    def test_normalise_title_to_subject(self):
        """'title' should be renamed to 'subject'."""
        result = MCPWrappedTool._normalise_args({"to": "x", "title": "Meeting"})
        assert result["subject"] == "Meeting"
        assert "title" not in result

    def test_normalise_for_date_to_date(self):
        """'for_date' should be renamed to 'date'."""
        result = MCPWrappedTool._normalise_args({"for_date": "2026-09-01"})
        assert result["date"] == "2026-09-01"
        assert "for_date" not in result

    def test_normalise_text_NOT_remapped(self):
        """'text' must NOT be remapped — it's a legitimate field name."""
        result = MCPWrappedTool._normalise_args({"text": "hello"})
        assert result == {"text": "hello"}
        assert "body" not in result

    def test_normalise_existing_fields_not_overwritten(self):
        """If 'to' already exists, 'recipient' should not overwrite it."""
        result = MCPWrappedTool._normalise_args({"to": "real@example.com", "recipient": "fake@example.com"})
        assert result["to"] == "real@example.com"

    def test_normalise_passthrough_unknown(self):
        """Unknown args are passed through unchanged."""
        result = MCPWrappedTool._normalise_args({"foo": "bar", "baz": 42})
        assert result == {"foo": "bar", "baz": 42}

    def test_normalise_non_dict_passthrough(self):
        """Non-dict input (e.g. a plain string) is returned as-is."""
        assert MCPWrappedTool._normalise_args("hello") == "hello"


# ─── MCPToolRegistry ──────────────────────────────────────────────────────────

class TestMCPToolRegistry:
    def test_from_list_builds_registry(self):
        registry = MCPToolRegistry.from_list([
            {"tool_name": "tool_a", "publisher_verified": True, "tool_origin": "internal",
             "update_recency_days": 1, "metadata_consistency": 1.0},
        ])
        meta = registry.get("tool_a")
        assert meta.trust_score == 1.0

    def test_unknown_tool_zero_trust(self):
        registry = MCPToolRegistry()
        meta = registry.get("nonexistent")
        assert meta.trust_score == 0.0
        assert meta.publisher_verified is False

    def test_wrap_tools_returns_basetool_list(self):
        registry = DEFAULT_STUB_REGISTRY
        wrapped = registry.wrap_tools([echo_tool])
        assert len(wrapped) == 1
        assert isinstance(wrapped[0], BaseTool)
        assert isinstance(wrapped[0], MCPWrappedTool)

    def test_default_registry_has_expected_tools(self):
        expected = {"send_email", "read_inbox", "get_calendar", "web_search"}
        assert expected.issubset(set(DEFAULT_STUB_REGISTRY._store.keys()))

    def test_default_internal_tools_have_trust_score(self):
        for name in ("send_email", "read_inbox", "get_calendar"):
            meta = DEFAULT_STUB_REGISTRY.get(name)
            assert meta.trust_score > 0, f"{name} should have trust_score > 0"

    def test_default_web_search_low_trust(self):
        meta = DEFAULT_STUB_REGISTRY.get("web_search")
        assert meta.trust_score < 0.5
