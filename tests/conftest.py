"""
tests/conftest.py — Shared pytest fixtures for the PACM test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def echo_tool_fixture():
    """Return the echo_tool for use in harness tests."""
    from langchain_core.tools import tool

    @tool
    def echo_tool(text: str) -> str:
        """Returns the input text prefixed with ECHO."""
        return f"ECHO: {text}"

    return echo_tool


@pytest.fixture
def default_config():
    """A HarnessConfig with conservative defaults for testing."""
    from src.harness.langgraph_agent import HarnessConfig

    return HarnessConfig(max_iterations=5, log_level=50)  # CRITICAL — suppress logs
