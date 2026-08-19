"""
tests/conftest.py — Shared pytest fixtures for the PACM test suite.

Session 6: Added session-scoped embedding backend fixtures to reduce
test_scorer_transformers.py execution from ~2 min to ~15 sec by loading
the sentence-transformers model once per test session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Harness fixtures ──────────────────────────────────────────────────────────

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


# ── Session-scoped embedding backend fixtures (Session 6) ─────────────────────


@pytest.fixture(scope="session")
def tfidf_backend():
    """
    Session-scoped TF-IDF backend instance.
    Created once and reused across all tests in the session.
    """
    from src.pacm.scorer import _TFIDFBackend
    return _TFIDFBackend()


@pytest.fixture(scope="session")
def st_backend():
    """
    Session-scoped SentenceTransformersBackend instance.

    Loads the all-MiniLM-L6-v2 model exactly ONCE per pytest session,
    reducing test_scorer_transformers.py runtime from ~2 minutes to ~15 seconds.

    Skips automatically if sentence-transformers is not installed.
    """
    try:
        from src.pacm.scorer import _SentenceTransformersBackend
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    backend = _SentenceTransformersBackend()
    # Pre-warm the model with a dummy call so tests don't pay load cost
    backend.similarity("warmup", "warmup")
    return backend


@pytest.fixture(scope="session")
def st_scorer():
    """
    Session-scoped DivergenceScorer using the ST backend.
    Reused across all DivergenceScorer + ST tests.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    from src.pacm.scorer import DivergenceScorer
    scorer = DivergenceScorer(prefer_transformers=True, alpha=0.70)
    return scorer
