"""
tests/test_llm_native.py

Unit tests for src/llm_native.py (NativeOllamaLLM).

All tests are purely structural/unit tests — no HTTP calls, no Ollama, no model.
The HTTP call is mocked at the requests.post level.

Tests cover:
  - bind_tools(): tool JSON schema generation from LangChain BaseTool
  - invoke(): message conversion, response parsing, thinking field routing
  - _lc_msg_to_ollama(): all message types (System, Human, AI, Tool)
  - _ollama_msg_to_lc(): reasoning in additional_kwargs["thinking"]
  - _tool_to_ollama(): BaseTool → Ollama tool definition
  - MCP-wrapped tool compatibility (simulated via a BaseTool with args_schema)

Session 21 guardrails: no harness modification, no model calls.
"""
from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "d:/mainproject")

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from src.llm_native import NativeOllamaLLM


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_requests_mock(response_data: dict):
    """Return a mock for requests.post that returns the given response dict."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _make_tool(name: str, description: str, args_schema=None) -> StructuredTool:
    """Create a minimal StructuredTool for testing."""
    if args_schema is None:
        from pydantic import BaseModel
        class EmptySchema(BaseModel):
            pass
        args_schema = EmptySchema

    def _fn(**kwargs):
        return "ok"

    return StructuredTool.from_function(
        func=_fn,
        name=name,
        description=description,
        args_schema=args_schema,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestNativeOllamaLLMInit(unittest.TestCase):
    def test_default_params(self):
        llm = NativeOllamaLLM()
        self.assertEqual(llm.model, "qwen3:8b")
        self.assertTrue(llm.think)
        self.assertEqual(llm.base_url, "http://localhost:11434")
        self.assertEqual(llm.timeout, 300)
        self.assertEqual(llm.keep_alive, -1)
        self.assertEqual(llm._tools_json, [])

    def test_custom_params(self):
        llm = NativeOllamaLLM(
            model="llama3.1:8b",
            think=False,
            base_url="http://remote:11434",
            timeout=60,
            keep_alive="5m",
        )
        self.assertEqual(llm.model, "llama3.1:8b")
        self.assertFalse(llm.think)
        self.assertEqual(llm.base_url, "http://remote:11434")
        self.assertEqual(llm.timeout, 60)
        self.assertEqual(llm.keep_alive, "5m")

    def test_base_url_trailing_slash_stripped(self):
        llm = NativeOllamaLLM(base_url="http://localhost:11434/")
        self.assertEqual(llm.base_url, "http://localhost:11434")


class TestBindTools(unittest.TestCase):
    def test_bind_tools_returns_new_instance(self):
        """bind_tools() must return a new NativeOllamaLLM, not mutate self."""
        llm = NativeOllamaLLM()
        from pydantic import BaseModel
        class NoArgs(BaseModel):
            pass
        tool = _make_tool("read_inbox", "Read inbox", NoArgs)
        bound = llm.bind_tools([tool])
        self.assertIsNot(bound, llm)
        self.assertEqual(llm._tools_json, [], "Original should be unchanged")
        self.assertEqual(len(bound._tools_json), 1)

    def test_bind_tools_format(self):
        """Bound tools should be in Ollama's native function format."""
        from pydantic import BaseModel
        class SendArgs(BaseModel):
            to: str
            subject: str
            body: str

        tool = _make_tool("send_email", "Send an email.", SendArgs)
        llm = NativeOllamaLLM().bind_tools([tool])

        self.assertEqual(len(llm._tools_json), 1)
        t = llm._tools_json[0]
        self.assertEqual(t["type"], "function")
        self.assertEqual(t["function"]["name"], "send_email")
        self.assertEqual(t["function"]["description"], "Send an email.")
        params = t["function"]["parameters"]
        self.assertIn("to", params.get("properties", {}))
        self.assertIn("subject", params.get("properties", {}))
        self.assertIn("body", params.get("properties", {}))

    def test_bind_empty_tools(self):
        llm = NativeOllamaLLM().bind_tools([])
        self.assertEqual(llm._tools_json, [])

    def test_bind_tool_no_args_schema(self):
        """Tool with no args_schema should produce empty parameters dict."""
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does something."
        tool.args_schema = None
        llm = NativeOllamaLLM().bind_tools([tool])
        params = llm._tools_json[0]["function"]["parameters"]
        self.assertEqual(params.get("properties", {}), {})


class TestMessageConversion(unittest.TestCase):
    """Tests for _lc_msg_to_ollama() - LangChain → Ollama native format."""

    def test_system_message(self):
        msg = SystemMessage(content="You are a helpful assistant.")
        out = NativeOllamaLLM._lc_msg_to_ollama(msg)
        self.assertEqual(out["role"], "system")
        self.assertEqual(out["content"], "You are a helpful assistant.")

    def test_human_message(self):
        msg = HumanMessage(content="Hello!")
        out = NativeOllamaLLM._lc_msg_to_ollama(msg)
        self.assertEqual(out["role"], "user")
        self.assertEqual(out["content"], "Hello!")

    def test_tool_message(self):
        msg = ToolMessage(content='{"emails": []}', tool_call_id="tc_001")
        out = NativeOllamaLLM._lc_msg_to_ollama(msg)
        self.assertEqual(out["role"], "tool")
        self.assertEqual(out["content"], '{"emails": []}')

    def test_ai_message_no_tool_calls(self):
        msg = AIMessage(content="Here is my answer.")
        out = NativeOllamaLLM._lc_msg_to_ollama(msg)
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["content"], "Here is my answer.")
        self.assertNotIn("tool_calls", out)

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc_1", "name": "send_email", "args": {"to": "a@b.com"}, "type": "tool_call"}],
        )
        out = NativeOllamaLLM._lc_msg_to_ollama(msg)
        self.assertEqual(out["role"], "assistant")
        self.assertIn("tool_calls", out)
        self.assertEqual(len(out["tool_calls"]), 1)
        tc = out["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "send_email")
        self.assertEqual(tc["function"]["arguments"]["to"], "a@b.com")

    def test_ai_message_with_thinking_preserved(self):
        """thinking from additional_kwargs should be re-attached in multi-turn."""
        msg = AIMessage(
            content="",
            additional_kwargs={"thinking": "I think therefore I am."},
        )
        out = NativeOllamaLLM._lc_msg_to_ollama(msg)
        self.assertEqual(out.get("thinking"), "I think therefore I am.")


class TestOllamaMsgToLC(unittest.TestCase):
    """Tests for _ollama_msg_to_lc() - Ollama response → LangChain AIMessage."""

    def test_empty_response(self):
        msg = {}
        ai = NativeOllamaLLM._ollama_msg_to_lc(msg)
        self.assertIsInstance(ai, AIMessage)
        self.assertEqual(ai.content, "")
        self.assertEqual(ai.tool_calls, [])
        self.assertNotIn("thinking", ai.additional_kwargs)

    def test_text_response(self):
        msg = {"role": "assistant", "content": "Here is the answer."}
        ai = NativeOllamaLLM._ollama_msg_to_lc(msg)
        self.assertEqual(ai.content, "Here is the answer.")
        self.assertEqual(ai.tool_calls, [])

    def test_thinking_field_routed_to_additional_kwargs(self):
        """thinking goes into additional_kwargs["thinking"] — key for _extract_reasoning()."""
        msg = {
            "role": "assistant",
            "content": "",
            "thinking": "Let me reason about this carefully.",
        }
        ai = NativeOllamaLLM._ollama_msg_to_lc(msg)
        self.assertEqual(ai.additional_kwargs.get("thinking"), "Let me reason about this carefully.")

    def test_tool_calls_converted_to_lc_format(self):
        """Native tool_calls list → LangChain format {id, name, args, type}."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_123",
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "arguments": {"to": "boss@company.com", "subject": "Hi", "body": "Hello"},
                    },
                }
            ],
        }
        ai = NativeOllamaLLM._ollama_msg_to_lc(msg)
        self.assertEqual(len(ai.tool_calls), 1)
        tc = ai.tool_calls[0]
        self.assertEqual(tc["id"], "tc_123")
        self.assertEqual(tc["name"], "send_email")
        self.assertEqual(tc["args"]["to"], "boss@company.com")
        self.assertEqual(tc["type"], "tool_call")

    def test_tool_calls_args_as_json_string(self):
        """Some Ollama versions return arguments as a JSON string — should be parsed."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_456",
                    "type": "function",
                    "function": {
                        "name": "read_inbox",
                        "arguments": json.dumps({}),  # string, not dict
                    },
                }
            ],
        }
        ai = NativeOllamaLLM._ollama_msg_to_lc(msg)
        self.assertEqual(ai.tool_calls[0]["args"], {})

    def test_thinking_and_tool_calls_together(self):
        """Both thinking and tool_calls can coexist in the same response."""
        msg = {
            "role": "assistant",
            "content": "",
            "thinking": "I should call read_inbox first.",
            "tool_calls": [
                {
                    "id": "tc_789",
                    "type": "function",
                    "function": {"name": "read_inbox", "arguments": {}},
                }
            ],
        }
        ai = NativeOllamaLLM._ollama_msg_to_lc(msg)
        self.assertEqual(ai.additional_kwargs.get("thinking"), "I should call read_inbox first.")
        self.assertEqual(len(ai.tool_calls), 1)
        self.assertEqual(ai.tool_calls[0]["name"], "read_inbox")


class TestInvoke(unittest.TestCase):
    """Tests for NativeOllamaLLM.invoke() — mocked HTTP."""

    def _invoke_with_response(self, response_msg: dict, think: bool = True) -> AIMessage:
        llm = NativeOllamaLLM(think=think)
        messages = [HumanMessage(content="Read my inbox.")]
        mock_resp = _make_requests_mock({"message": response_msg})
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = llm.invoke(messages)
            # Verify POST was called to the correct endpoint
            call_args = mock_post.call_args
            self.assertIn("/api/chat", call_args[0][0])
            payload = call_args[1]["json"]
            self.assertEqual(payload["think"], think)
            self.assertFalse(payload["stream"])
        return result

    def test_text_response_returned_as_aimessage(self):
        ai = self._invoke_with_response({"role": "assistant", "content": "Done."})
        self.assertIsInstance(ai, AIMessage)
        self.assertEqual(ai.content, "Done.")

    def test_thinking_in_additional_kwargs(self):
        ai = self._invoke_with_response({
            "role": "assistant",
            "content": "",
            "thinking": "Step 1: read inbox.",
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_inbox", "arguments": {}}}],
        })
        self.assertEqual(ai.additional_kwargs.get("thinking"), "Step 1: read inbox.")

    def test_tool_calls_present(self):
        ai = self._invoke_with_response({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_inbox", "arguments": {}}}],
        })
        self.assertEqual(len(ai.tool_calls), 1)
        self.assertEqual(ai.tool_calls[0]["name"], "read_inbox")

    def test_think_false_passed_in_payload(self):
        """think=False should be sent in the payload."""
        ai = self._invoke_with_response({"role": "assistant", "content": "ok"}, think=False)
        # _invoke_with_response already asserts think=False in payload

    def test_tools_included_when_bound(self):
        """tools= payload should contain tool definitions when bind_tools was called."""
        from pydantic import BaseModel
        class ReadInboxArgs(BaseModel):
            pass
        tool = _make_tool("read_inbox", "Read inbox.", ReadInboxArgs)
        llm = NativeOllamaLLM().bind_tools([tool])
        messages = [HumanMessage(content="Read my inbox.")]
        mock_resp = _make_requests_mock({"message": {"role": "assistant", "content": "ok"}})
        with patch("requests.post", return_value=mock_resp) as mock_post:
            llm.invoke(messages)
            payload = mock_post.call_args[1]["json"]
            self.assertIn("tools", payload)
            self.assertEqual(len(payload["tools"]), 1)
            self.assertEqual(payload["tools"][0]["function"]["name"], "read_inbox")

    def test_keep_alive_in_payload(self):
        """keep_alive should be passed in the payload."""
        llm = NativeOllamaLLM(keep_alive="10m")
        messages = [HumanMessage(content="Hi")]
        mock_resp = _make_requests_mock({"message": {"role": "assistant", "content": "ok"}})
        with patch("requests.post", return_value=mock_resp) as mock_post:
            llm.invoke(messages)
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["keep_alive"], "10m")

    def test_http_error_raises(self):
        """HTTP errors from requests should propagate."""
        import requests as req
        llm = NativeOllamaLLM()
        messages = [HumanMessage(content="Hi")]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("400 Bad Request")
        with patch("requests.post", return_value=mock_resp):
            with self.assertRaises(req.HTTPError):
                llm.invoke(messages)


class TestMCPWrappedToolCompatibility(unittest.TestCase):
    """
    Verifies NativeOllamaLLM.bind_tools() works with MCP-wrapped tools
    (DEFAULT_STUB_REGISTRY.wrap_tools output).

    Uses the real DEFAULT_STUB_REGISTRY and stub tools from the experiment runner,
    then checks that bind_tools() produces valid Ollama tool definitions.
    This is a prerequisite test for the Session 22 n=10 sweep.
    """

    def test_bind_mcp_wrapped_stub_tools(self):
        """MCP-wrapped stub tools should produce valid Ollama tool definitions."""
        from src.harness.mcp_tool_wrapper import DEFAULT_STUB_REGISTRY
        from langchain_core.tools import BaseTool

        # Build stub tools the same way the experiment runner does
        raw_tools = _build_stub_tools()
        mcp_tools = DEFAULT_STUB_REGISTRY.wrap_tools(raw_tools)

        llm = NativeOllamaLLM().bind_tools(mcp_tools)

        self.assertGreater(len(llm._tools_json), 0, "No tools bound from MCP-wrapped tools")

        for t in llm._tools_json:
            self.assertEqual(t["type"], "function")
            fn = t["function"]
            self.assertIsInstance(fn["name"], str)
            self.assertGreater(len(fn["name"]), 0)
            self.assertIsInstance(fn.get("description", ""), str)
            params = fn.get("parameters", {})
            self.assertIn("type", params, f"Tool {fn['name']} missing 'type' in parameters")

    def test_mcp_tool_names_preserved(self):
        """Tool names should be preserved through the MCP wrap."""
        from src.harness.mcp_tool_wrapper import DEFAULT_STUB_REGISTRY

        raw_tools = _build_stub_tools()
        mcp_tools = DEFAULT_STUB_REGISTRY.wrap_tools(raw_tools)
        llm = NativeOllamaLLM().bind_tools(mcp_tools)

        tool_names = {t["function"]["name"] for t in llm._tools_json}
        # At minimum, read_inbox and send_email should be present
        self.assertIn("read_inbox", tool_names, f"read_inbox missing from {tool_names}")
        self.assertIn("send_email", tool_names, f"send_email missing from {tool_names}")


def _build_stub_tools():
    """Build stub tools the same way experiments/run_experiment.py does."""
    from langchain_core.tools import tool as lc_tool
    from pydantic import BaseModel

    @lc_tool
    def read_inbox() -> str:
        """Read the user's email inbox and return a list of recent messages."""
        return "[]"

    @lc_tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email on behalf of the user.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Email body text.
        """
        return f"Email sent to {to}."

    return [read_inbox, send_email]


if __name__ == "__main__":
    unittest.main(verbosity=2)
