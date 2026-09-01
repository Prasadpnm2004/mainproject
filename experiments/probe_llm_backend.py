"""
experiments/probe_llm_backend.py

PACM — LLM Backend Capability Probe

A 10-second sanity-check CLI that verifies the currently configured LLM
backend can:
  1. Accept a connection / respond to a trivial chat request.
  2. Return structured tool_calls (not just free text).
  3. Have its tool_calls parsed correctly by PACM's defensive parser.

Run this before every real-model sweep or session to confirm your backend
(OpenAI or Ollama) is working correctly.

Usage:
    # Test whatever is configured in .env / environment:
    python -m experiments.probe_llm_backend

    # Override provider on the fly:
    LLM_PROVIDER=ollama LLM_MODEL=qwen3:8b python -m experiments.probe_llm_backend
    LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini python -m experiments.probe_llm_backend

    # Non-interactive (exit code 0 = pass, 1 = fail):
    python -m experiments.probe_llm_backend --quiet

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path when run as a script ──────────────────
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

# Load .env before importing llm_config so env vars are set (override=False
# means shell environment vars take precedence over .env values).
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell env

from src.llm_config import (
    LLMConfig,
    OLLAMA_TOOL_CALLING_MODELS,
    build_chat_model,
    load_llm_config,
)

logger = logging.getLogger(__name__)

# ── Trivial tool used by the probe ───────────────────────────────────────────

_PROBE_TOOL_NAME = "get_current_time"
_PROBE_TOOL_SCHEMA = {
    "name": _PROBE_TOOL_NAME,
    "description": "Return the current UTC time as an ISO-8601 string.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ── Probe result ─────────────────────────────────────────────────────────────

class ProbeResult:
    def __init__(self) -> None:
        self.connected: bool = False
        self.responded: bool = False
        self.tool_calls_returned: bool = False
        self.tool_calls_parsed: bool = False
        self.latency_s: float = 0.0
        self.raw_response: str = ""
        self.parsed_tool_name: str = ""
        self.parsed_tool_args: dict = {}
        self.error: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.connected
            and self.responded
            and self.tool_calls_returned
            and self.tool_calls_parsed
        )

    def summary_lines(self, config: LLMConfig) -> list[str]:
        lines = [
            "",
            "=" * 60,
            f"  PACM LLM Backend Probe",
            "=" * 60,
            f"  Provider   : {config.provider}",
            f"  Model      : {config.model}",
            f"  Base URL   : {config.base_url or '(library default)'}",
            f"  API key    : {'SET' if config.api_key else 'NOT SET'}",
            "-" * 60,
            f"  [{'OK' if self.connected         else 'FAIL':>4}] Connection / import",
            f"  [{'OK' if self.responded         else 'FAIL':>4}] LLM responded",
            f"  [{'OK' if self.tool_calls_returned else 'FAIL':>4}] Structured tool_calls returned",
            f"  [{'OK' if self.tool_calls_parsed else 'FAIL':>4}] tool_calls parsed correctly",
            f"  Latency    : {self.latency_s:.2f}s",
        ]
        if self.tool_calls_parsed:
            lines.append(f"  Tool name  : {self.parsed_tool_name!r}")
            lines.append(f"  Tool args  : {self.parsed_tool_args}")
        if self.error:
            lines.append(f"  Error      : {self.error}")
        lines.append("-" * 60)
        if self.passed:
            lines.append("  RESULT: PASS — backend is ready for PACM experiments.")
        else:
            lines.append("  RESULT: FAIL — see errors above.")
            if config.is_ollama:
                lines += [
                    "",
                    "  Ollama troubleshooting:",
                    "    1. Is Ollama running?  ollama serve",
                    f"   2. Is the model pulled? ollama pull {config.model}",
                    "    3. Does the model support tool calling?",
                    f"      Confirmed models: {', '.join(sorted(OLLAMA_TOOL_CALLING_MODELS))}",
                    "    4. Try:  LLM_MODEL=qwen3:8b python -m experiments.probe_llm_backend",
                ]
            else:
                lines += [
                    "",
                    "  OpenAI troubleshooting:",
                    "    1. Is OPENAI_API_KEY set in .env?",
                    "    2. Is the model name correct? (e.g. gpt-4o-mini)",
                    "    3. Do you have API quota remaining?",
                ]
        lines.append("=" * 60)
        return lines


# ── Core probe logic ──────────────────────────────────────────────────────────

def _parse_tool_calls_defensively(raw_tool_calls: list) -> tuple[str, dict, str]:
    """
    Use the same defensive parsing logic as custom_loop_agent._execute_tool_call.

    Returns (tool_name, tool_args, error_message).
    error_message is empty string on success.
    """
    if not raw_tool_calls:
        return "", {}, "No tool_calls in response"

    tc = raw_tool_calls[0]

    # Handle both dict-style (LangChain) and object-style
    if hasattr(tc, "name"):
        raw_name = tc.name
        raw_args = tc.args if hasattr(tc, "args") else {}
    elif isinstance(tc, dict):
        raw_name = tc.get("name") or tc.get("function", {}).get("name", "")
        raw_args = tc.get("args") or tc.get("function", {}).get("arguments", {})
    else:
        return "", {}, f"Unexpected tool_call type: {type(tc)}"

    if not raw_name:
        return "", {}, "tool_call missing 'name' field"

    # JSON string args (common in some local models)
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return raw_name, {}, f"Could not parse args as JSON: {e} | raw={raw_args!r}"

    if not isinstance(raw_args, dict):
        raw_args = {}

    return raw_name, raw_args, ""


def run_probe(config: LLMConfig) -> ProbeResult:
    """Execute the probe against the given LLM config."""
    result = ProbeResult()

    # ── Step 1: Build the model ───────────────────────────────────────────────
    try:
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def get_current_time() -> str:
            """Return the current UTC time as an ISO-8601 string."""
            return "2026-08-30T12:00:00Z"

        llm = build_chat_model(config, temperature=0)
        llm_with_tools = llm.bind_tools([get_current_time])
        result.connected = True
    except Exception as e:
        result.error = f"Failed to build LLM model: {e}"
        return result

    # ── Step 2: Send the probe request ───────────────────────────────────────
    from langchain_core.messages import HumanMessage

    t0 = time.perf_counter()
    try:
        response = llm_with_tools.invoke([
            HumanMessage(content=(
                "What is the current UTC time? "
                "Use the get_current_time tool to find out."
            ))
        ])
        result.latency_s = time.perf_counter() - t0
        result.responded = True
        result.raw_response = str(response.content)[:500]
    except Exception as e:
        result.latency_s = time.perf_counter() - t0
        result.error = f"LLM invocation failed: {e}"
        return result

    # ── Step 3: Check for structured tool_calls ───────────────────────────────
    raw_tcs = getattr(response, "tool_calls", None) or []
    if not raw_tcs:
        result.error = (
            "LLM responded but did NOT return structured tool_calls. "
            "Response text: " + result.raw_response[:200]
        )
        return result

    result.tool_calls_returned = True

    # ── Step 4: Parse tool_calls defensively ──────────────────────────────────
    name, args, parse_err = _parse_tool_calls_defensively(raw_tcs)
    if parse_err:
        result.error = f"tool_calls parse failed: {parse_err}"
        return result

    result.tool_calls_parsed = True
    result.parsed_tool_name = name
    result.parsed_tool_args = args

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PACM LLM Backend Capability Probe — 10-second sanity check."
    )
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output; exit code 0=pass, 1=fail.")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output result as JSON.")
    args = parser.parse_args()

    if not args.quiet:
        print("\n  Loading LLM config from environment...", flush=True)

    try:
        config = load_llm_config()
    except ValueError as e:
        print(f"  CONFIG ERROR: {e}")
        sys.exit(1)

    if not args.quiet:
        print(f"  Provider: {config.provider}  Model: {config.model}", flush=True)
        print("  Sending probe request (this may take 5-30s for Ollama)...",
              flush=True)

    result = run_probe(config)

    if args.json_output:
        print(json.dumps({
            "passed": result.passed,
            "connected": result.connected,
            "responded": result.responded,
            "tool_calls_returned": result.tool_calls_returned,
            "tool_calls_parsed": result.tool_calls_parsed,
            "latency_s": round(result.latency_s, 3),
            "parsed_tool_name": result.parsed_tool_name,
            "parsed_tool_args": result.parsed_tool_args,
            "error": result.error,
            "provider": config.provider,
            "model": config.model,
        }, indent=2))
    elif not args.quiet:
        for line in result.summary_lines(config):
            print(line)

    sys.exit(0 if result.passed else 1)
