"""
src/llm_config.py

PACM — Provider-Agnostic LLM Configuration Layer

Reads environment variables to select between OpenAI and Ollama (or any
OpenAI-compatible endpoint).  All harnesses and experiment runners import
this module instead of hardcoding ``ChatOpenAI``.

Environment variables
---------------------
LLM_PROVIDER   : "openai" | "ollama"   (default: "openai")
LLM_MODEL      : model name            (default: "gpt-4o-mini" for openai,
                                                  "qwen3:8b" for ollama)
LLM_BASE_URL   : API base URL          (default: inferred from provider)
LLM_API_KEY    : API key               (default: OPENAI_API_KEY for openai,
                                                  "ollama" placeholder for ollama)

Ollama notes
------------
* Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1.
* The API key value is ignored by Ollama but must be non-empty for the
  openai-python client; "ollama" is used as a placeholder.
* Pull the model before running:
      ollama pull qwen3:8b       # recommended — confirmed tool-calling support
      ollama pull llama3.1:8b   # alternative with confirmed tool-calling support
* Smaller or non-instruct models (e.g. tinyllama) may not support structured
  tool_calls reliably — use the probe script to verify first:
      python -m experiments.probe_llm_backend

Quick-switch examples
---------------------
  # Use local Ollama with qwen3:8b
  LLM_PROVIDER=ollama LLM_MODEL=qwen3:8b python -m experiments.run_baseline

  # Use OpenAI (requires OPENAI_API_KEY in .env)
  LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini python -m experiments.run_baseline
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]

try:
    import openai  # type: ignore[import]
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]

# ─── Constants ────────────────────────────────────────────────────────────────

_PROVIDER_OPENAI = "openai"
_PROVIDER_OLLAMA = "ollama"
SUPPORTED_PROVIDERS = (_PROVIDER_OPENAI, _PROVIDER_OLLAMA)

_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_DEFAULT_MODEL = "qwen3:8b"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# Models with confirmed tool-calling support on Ollama (as of Aug 2026).
# Used only for documentation / probe warnings.
OLLAMA_TOOL_CALLING_MODELS = frozenset({
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:4b",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.2:3b",
    "mistral:7b-instruct",
    "mistral-nemo",
})


# ─── Config dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMConfig:
    """
    Resolved, validated LLM provider configuration.

    Attributes
    ----------
    provider   : "openai" or "ollama"
    model      : model name string
    base_url   : API base URL (None → use library default)
    api_key    : API key (may be the "ollama" placeholder)
    is_ollama  : convenience bool
    is_openai  : convenience bool
    """

    provider: str
    model: str
    base_url: str | None
    api_key: str | None

    @property
    def is_ollama(self) -> bool:
        return self.provider == _PROVIDER_OLLAMA

    @property
    def is_openai(self) -> bool:
        return self.provider == _PROVIDER_OPENAI

    def __str__(self) -> str:
        url = self.base_url or "(library default)"
        return f"LLMConfig(provider={self.provider!r}, model={self.model!r}, base_url={url!r})"


# ─── Factory ──────────────────────────────────────────────────────────────────

def load_llm_config() -> LLMConfig:
    """
    Read LLM configuration from environment variables and return an
    ``LLMConfig``.  This is the single authoritative config-load call
    for all harnesses and experiment runners.

    Resolution order:
    1. Explicit ``LLM_PROVIDER`` env-var.
    2. If provider=openai and ``OPENAI_API_KEY`` is set → openai.
    3. Defaults to openai with model gpt-4o-mini if nothing is set.

    Returns
    -------
    LLMConfig
    """
    raw_provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if raw_provider not in (SUPPORTED_PROVIDERS + ("",)):
        raise ValueError(
            f"LLM_PROVIDER={raw_provider!r} is not supported. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    # Auto-detect provider if not explicitly set
    if not raw_provider:
        raw_provider = _PROVIDER_OPENAI  # default

    if raw_provider == _PROVIDER_OLLAMA:
        model = os.getenv("LLM_MODEL", _OLLAMA_DEFAULT_MODEL).strip()
        base_url = os.getenv("LLM_BASE_URL", _OLLAMA_DEFAULT_BASE_URL).strip()
        api_key = os.getenv("LLM_API_KEY", "ollama")  # placeholder; ignored by Ollama
    else:  # openai
        model = os.getenv("LLM_MODEL", "").strip() or os.getenv(
            "PACM_DEFAULT_MODEL", _OPENAI_DEFAULT_MODEL
        ).strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip() or None
        api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv(
            "OPENAI_API_KEY", ""
        ).strip() or None

    return LLMConfig(
        provider=raw_provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


# ─── LangChain chat-model factory ─────────────────────────────────────────────

def build_chat_model(config: LLMConfig | None = None, temperature: float = 0):
    """
    Build and return a LangChain ``BaseChatModel`` from an ``LLMConfig``.

    Both OpenAI and Ollama use ``ChatOpenAI`` under the hood because Ollama
    exposes an OpenAI-compatible endpoint.  The only difference is ``base_url``
    and ``api_key``.

    Parameters
    ----------
    config      : LLMConfig (default: load_llm_config())
    temperature : sampling temperature (default: 0 for reproducibility)

    Returns
    -------
    langchain_openai.ChatOpenAI
    """
    if config is None:
        config = load_llm_config()

    if ChatOpenAI is None:  # pragma: no cover
        raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai")

    kwargs: dict = dict(
        model=config.model,
        temperature=temperature,
    )
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key

    return ChatOpenAI(**kwargs)


# ─── AgentDojo OpenAI-client factory ──────────────────────────────────────────

def build_openai_client(config: LLMConfig | None = None):
    """
    Build and return a raw ``openai.OpenAI`` client configured for the
    current provider.  Used by AgentDojo's ``OpenAILLM`` wrapper.

    For Ollama, ``base_url`` and ``api_key`` are patched so the openai SDK
    talks to the local Ollama endpoint.

    Parameters
    ----------
    config : LLMConfig (default: load_llm_config())

    Returns
    -------
    openai.OpenAI
    """
    if config is None:
        config = load_llm_config()

    if openai is None:  # pragma: no cover
        raise ImportError("openai is not installed. Run: pip install openai")

    kwargs: dict = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key

    return openai.OpenAI(**kwargs)


# ─── Convenience: is the current config live-LLM capable? ────────────────────

def is_real_llm_available(config: LLMConfig | None = None) -> bool:
    """
    Return True if a live LLM is expected to be reachable with the current
    config.

    For OpenAI: checks that OPENAI_API_KEY / LLM_API_KEY is non-empty.
    For Ollama : performs a non-blocking TCP socket check to localhost:11434
                 with a 1-second timeout.  Returns False if Ollama is not
                 reachable (e.g. ``ollama serve`` not running, or in CI).

    Override: set ``PACM_FORCE_STUB=1`` in the environment to unconditionally
    return False (useful for unit tests that must not make real LLM calls).
    """
    # Hard override for test environments
    if os.getenv("PACM_FORCE_STUB", "").strip() == "1":
        return False

    if config is None:
        config = load_llm_config()

    if config.is_ollama:
        # Non-blocking TCP reachability check — avoids hanging on a
        # configured-but-not-running Ollama server.
        import socket
        host = "localhost"
        port = 11434
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (OSError, TimeoutError):
            return False

    # openai path
    return bool(config.api_key)
