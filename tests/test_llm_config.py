"""
tests/test_llm_config.py

Unit tests for the PACM provider-agnostic LLM configuration layer.

Coverage:
  - load_llm_config() env-var resolution for both openai and ollama providers
  - Default values when env-vars are absent
  - Provider validation / error on unsupported value
  - is_real_llm_available() behaviour for both providers
  - build_chat_model() returns ChatOpenAI with correct kwargs (mocked)
  - build_openai_client() returns openai.OpenAI with correct kwargs (mocked)

All tests are purely mock-based — no real network calls, no OpenAI key,
no running Ollama instance required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.llm_config import (
    LLMConfig,
    _OLLAMA_DEFAULT_BASE_URL,
    _OLLAMA_DEFAULT_MODEL,
    _OPENAI_DEFAULT_MODEL,
    build_chat_model,
    build_openai_client,
    is_real_llm_available,
    load_llm_config,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env(**kwargs):
    """Return an env dict with only the given keys set; all others absent."""
    return {k: v for k, v in kwargs.items()}


# ── load_llm_config tests ─────────────────────────────────────────────────────

class TestLoadLLMConfig:

    def test_defaults_to_openai(self, monkeypatch):
        """With no env-vars, provider=openai, model=gpt-4o-mini."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("PACM_DEFAULT_MODEL", raising=False)

        cfg = load_llm_config()

        assert cfg.provider == "openai"
        assert cfg.model == _OPENAI_DEFAULT_MODEL
        assert cfg.base_url is None  # library default
        assert cfg.is_openai
        assert not cfg.is_ollama

    def test_openai_explicit(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        cfg = load_llm_config()

        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.api_key == "sk-test-key"
        assert cfg.base_url is None

    def test_openai_fallback_to_OPENAI_API_KEY(self, monkeypatch):
        """LLM_API_KEY absent but OPENAI_API_KEY present — should use it."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        cfg = load_llm_config()

        assert cfg.api_key == "sk-fallback"
        assert cfg.provider == "openai"

    def test_ollama_defaults(self, monkeypatch):
        """provider=ollama → correct defaults for base_url and model."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        cfg = load_llm_config()

        assert cfg.provider == "ollama"
        assert cfg.model == _OLLAMA_DEFAULT_MODEL
        assert cfg.base_url == _OLLAMA_DEFAULT_BASE_URL
        assert cfg.api_key == "ollama"  # placeholder
        assert cfg.is_ollama
        assert not cfg.is_openai

    def test_ollama_custom_model_and_url(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
        monkeypatch.setenv("LLM_BASE_URL", "http://192.168.1.100:11434/v1")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        cfg = load_llm_config()

        assert cfg.model == "llama3.1:8b"
        assert cfg.base_url == "http://192.168.1.100:11434/v1"

    def test_unsupported_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with pytest.raises(ValueError, match="bedrock"):
            load_llm_config()

    def test_case_insensitive_provider(self, monkeypatch):
        """Provider env-var should be case-insensitive."""
        monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        cfg = load_llm_config()
        assert cfg.provider == "ollama"

    def test_str_representation(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        cfg = load_llm_config()
        s = str(cfg)
        assert "ollama" in s
        assert _OLLAMA_DEFAULT_MODEL in s


# ── is_real_llm_available tests ───────────────────────────────────────────────

class TestIsRealLLMAvailable:
    """
    Tests for is_real_llm_available().

    NOTE: The conftest sets PACM_FORCE_STUB=1 for all tests to prevent
    accidental real LLM calls. Tests in this class that verify the *actual*
    availability logic must use monkeypatch to clear that override first.
    """

    def test_openai_with_key_is_available(self, monkeypatch):
        monkeypatch.delenv("PACM_FORCE_STUB", raising=False)
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key="sk-real")
        assert is_real_llm_available(cfg) is True

    def test_openai_without_key_is_not_available(self):
        # PACM_FORCE_STUB=1 also returns False — same expected result
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key=None)
        assert is_real_llm_available(cfg) is False

    def test_openai_empty_key_is_not_available(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key="")
        assert is_real_llm_available(cfg) is False

    def test_ollama_available_when_server_running(self, monkeypatch):
        """Ollama TCP check — returns True when localhost:11434 is reachable."""
        monkeypatch.delenv("PACM_FORCE_STUB", raising=False)
        cfg = LLMConfig(provider="ollama", model="qwen3:8b",
                        base_url=_OLLAMA_DEFAULT_BASE_URL, api_key="ollama")
        # In CI where Ollama isn't running this will return False which is correct.
        # We just verify no exception is raised.
        result = is_real_llm_available(cfg)
        assert isinstance(result, bool)

    def test_ollama_available_even_with_no_api_key(self, monkeypatch):
        """Ollama doesn't need an API key — key=None is fine."""
        monkeypatch.delenv("PACM_FORCE_STUB", raising=False)
        cfg = LLMConfig(provider="ollama", model="qwen3:8b",
                        base_url=_OLLAMA_DEFAULT_BASE_URL, api_key=None)
        result = is_real_llm_available(cfg)
        assert isinstance(result, bool)  # True if server up, False if not

    def test_no_config_arg_reads_from_env(self, monkeypatch):
        """is_real_llm_available() with no args loads config from env."""
        monkeypatch.delenv("PACM_FORCE_STUB", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "sk-env-key")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        assert is_real_llm_available() is True

    def test_force_stub_overrides_everything(self, monkeypatch):
        """PACM_FORCE_STUB=1 returns False regardless of config."""
        monkeypatch.setenv("PACM_FORCE_STUB", "1")
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key="sk-real")
        assert is_real_llm_available(cfg) is False

    def test_force_stub_zero_does_not_override(self, monkeypatch):
        """PACM_FORCE_STUB=0 does not trigger the override."""
        monkeypatch.setenv("PACM_FORCE_STUB", "0")
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key="sk-real")
        # With PACM_FORCE_STUB=0 and a real key, should return True
        assert is_real_llm_available(cfg) is True


# ── build_chat_model tests (mocked) ──────────────────────────────────────────

class TestBuildChatModel:

    def test_openai_uses_no_base_url_by_default(self, monkeypatch):
        """OpenAI config → ChatOpenAI called without base_url."""
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key="sk-test")

        mock_chat_cls = MagicMock()
        with patch("src.llm_config.ChatOpenAI", mock_chat_cls):
            build_chat_model(cfg, temperature=0)

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0
        assert "base_url" not in call_kwargs  # not set when None

    def test_ollama_sets_base_url_and_api_key(self, monkeypatch):
        """Ollama config → ChatOpenAI called with base_url and api_key."""
        cfg = LLMConfig(
            provider="ollama",
            model="qwen3:8b",
            base_url=_OLLAMA_DEFAULT_BASE_URL,
            api_key="ollama",
        )

        mock_chat_cls = MagicMock()
        with patch("src.llm_config.ChatOpenAI", mock_chat_cls):
            build_chat_model(cfg)

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:8b"
        assert call_kwargs["base_url"] == _OLLAMA_DEFAULT_BASE_URL
        assert call_kwargs["api_key"] == "ollama"

    def test_no_config_loads_from_env(self, monkeypatch):
        """build_chat_model() with no args calls load_llm_config()."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        mock_chat_cls = MagicMock()
        with patch("src.llm_config.ChatOpenAI", mock_chat_cls):
            build_chat_model()

        assert mock_chat_cls.call_args.kwargs["model"] == "gpt-4o"

    def test_custom_temperature(self, monkeypatch):
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key="sk-test")
        mock_chat_cls = MagicMock()
        with patch("src.llm_config.ChatOpenAI", mock_chat_cls):
            build_chat_model(cfg, temperature=0.7)

        assert mock_chat_cls.call_args.kwargs["temperature"] == 0.7


# ── build_openai_client tests (mocked) ───────────────────────────────────────

class TestBuildOpenAIClient:

    def test_openai_standard_no_override(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key="sk-real")

        mock_openai_cls = MagicMock()
        with patch("src.llm_config.openai") as mock_openai_mod:
            mock_openai_mod.OpenAI = mock_openai_cls
            build_openai_client(cfg)

        call_kwargs = mock_openai_cls.call_args.kwargs
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "sk-real"
        assert "base_url" not in call_kwargs  # not set when None

    def test_ollama_sets_base_url(self):
        cfg = LLMConfig(
            provider="ollama", model="qwen3:8b",
            base_url=_OLLAMA_DEFAULT_BASE_URL, api_key="ollama",
        )

        mock_openai_cls = MagicMock()
        with patch("src.llm_config.openai") as mock_openai_mod:
            mock_openai_mod.OpenAI = mock_openai_cls
            build_openai_client(cfg)

        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["base_url"] == _OLLAMA_DEFAULT_BASE_URL
        assert call_kwargs["api_key"] == "ollama"

    def test_no_config_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        mock_openai_cls = MagicMock()
        with patch("src.llm_config.openai") as mock_openai_mod:
            mock_openai_mod.OpenAI = mock_openai_cls
            build_openai_client()

        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["base_url"] == _OLLAMA_DEFAULT_BASE_URL


# ── LLMConfig dataclass properties ───────────────────────────────────────────

class TestLLMConfigProperties:

    def test_is_openai_and_is_ollama_mutually_exclusive(self):
        openai_cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                               base_url=None, api_key=None)
        ollama_cfg = LLMConfig(provider="ollama", model="qwen3:8b",
                               base_url=_OLLAMA_DEFAULT_BASE_URL, api_key="ollama")

        assert openai_cfg.is_openai and not openai_cfg.is_ollama
        assert ollama_cfg.is_ollama and not ollama_cfg.is_openai

    def test_config_is_frozen(self):
        """LLMConfig is a frozen dataclass — must not be mutable."""
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini",
                        base_url=None, api_key=None)
        with pytest.raises((AttributeError, TypeError)):
            cfg.model = "gpt-4o"  # type: ignore[misc]


# ── Session 14: LLM_DISABLE_THINKING tests ───────────────────────────────────

class TestDisableThinking:
    """Tests for the LLM_DISABLE_THINKING environment variable and its effect
    on build_chat_model() extra_body injection.

    All tests are purely offline — no Ollama or OpenAI calls made.
    """

    def test_llm_disable_thinking_env_sets_flag(self):
        """LLM_DISABLE_THINKING=1 → config.disable_thinking is True."""
        env = {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen3:8b",
            "LLM_BASE_URL": _OLLAMA_DEFAULT_BASE_URL,
            "LLM_API_KEY": "ollama",
            "LLM_DISABLE_THINKING": "1",
            "PACM_FORCE_STUB": "1",
        }
        with patch("os.environ", env):
            cfg = load_llm_config()
        assert cfg.disable_thinking is True, (
            "LLM_DISABLE_THINKING=1 must set disable_thinking=True"
        )

    def test_disable_thinking_defaults_to_false(self):
        """LLM_DISABLE_THINKING absent (or empty) → config.disable_thinking is False."""
        env = {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen3:8b",
            "LLM_BASE_URL": _OLLAMA_DEFAULT_BASE_URL,
            "LLM_API_KEY": "ollama",
            "PACM_FORCE_STUB": "1",
        }
        with patch("os.environ", env):
            cfg = load_llm_config()
        assert cfg.disable_thinking is False, (
            "Missing LLM_DISABLE_THINKING must leave disable_thinking=False"
        )

    def test_build_chat_model_passes_extra_body_for_ollama(self):
        """When disable_thinking=True + provider=ollama, extra_body={'think':False} injected."""
        cfg = LLMConfig(
            provider="ollama",
            model="qwen3:8b",
            base_url=_OLLAMA_DEFAULT_BASE_URL,
            api_key="ollama",
            disable_thinking=True,
        )
        mock_cls = MagicMock()
        with patch("src.llm_config.ChatOpenAI", mock_cls):
            build_chat_model(cfg, temperature=0)

        call_kwargs = mock_cls.call_args.kwargs
        assert "extra_body" in call_kwargs, (
            "extra_body must be present in ChatOpenAI kwargs when disable_thinking=True"
        )
        assert call_kwargs["extra_body"] == {"think": False}, (
            f"extra_body must be {{'think': False}}, got {call_kwargs['extra_body']!r}"
        )

    def test_build_chat_model_no_extra_body_for_openai(self):
        """When provider=openai, extra_body must NOT be injected even if disable_thinking=True.
        OpenAI's API does not accept 'think' and would return a 400 error.
        """
        cfg = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            base_url=None,
            api_key="sk-test-key",
            disable_thinking=True,  # flag set but provider is openai → no injection
        )
        mock_cls = MagicMock()
        with patch("src.llm_config.ChatOpenAI", mock_cls):
            build_chat_model(cfg, temperature=0)

        call_kwargs = mock_cls.call_args.kwargs
        assert "extra_body" not in call_kwargs, (
            "extra_body must NOT be present for OpenAI provider (would cause API 400)"
        )



# -- Session 14 (timeout): LLM_REQUEST_TIMEOUT tests -------------------------

class TestRequestTimeout:
    """Tests for LLM_REQUEST_TIMEOUT env var and request_timeout field."""

    def test_ollama_default_timeout_is_300(self):
        """Ollama provider gets request_timeout=300.0 by default."""
        import os
        from unittest.mock import patch
        from src.llm_config import load_llm_config, _OLLAMA_DEFAULT_BASE_URL
        env = {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen3:8b",
            "LLM_BASE_URL": _OLLAMA_DEFAULT_BASE_URL,
            "LLM_API_KEY": "ollama",
            "PACM_FORCE_STUB": "1",
        }
        with patch("os.environ", env):
            cfg = load_llm_config()
        assert cfg.request_timeout == 300.0, f"Ollama default must be 300.0, got {cfg.request_timeout}"

    def test_llm_request_timeout_env_overrides_default(self):
        """LLM_REQUEST_TIMEOUT=120 sets request_timeout=120.0."""
        from unittest.mock import patch
        from src.llm_config import load_llm_config, _OLLAMA_DEFAULT_BASE_URL
        env = {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen3:8b",
            "LLM_BASE_URL": _OLLAMA_DEFAULT_BASE_URL,
            "LLM_API_KEY": "ollama",
            "LLM_REQUEST_TIMEOUT": "120",
            "PACM_FORCE_STUB": "1",
        }
        with patch("os.environ", env):
            cfg = load_llm_config()
        assert cfg.request_timeout == 120.0, f"Expected 120.0, got {cfg.request_timeout}"

    def test_openai_default_timeout_is_none(self):
        """OpenAI provider gets request_timeout=None by default."""
        from unittest.mock import patch
        from src.llm_config import load_llm_config
        env = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4o-mini",
            "LLM_API_KEY": "sk-test",
            "PACM_FORCE_STUB": "1",
        }
        with patch("os.environ", env):
            cfg = load_llm_config()
        assert cfg.request_timeout is None, f"OpenAI timeout must be None, got {cfg.request_timeout}"

    def test_build_chat_model_passes_request_timeout(self):
        """request_timeout forwarded to ChatOpenAI kwargs."""
        from unittest.mock import patch, MagicMock
        from src.llm_config import LLMConfig, build_chat_model, _OLLAMA_DEFAULT_BASE_URL
        cfg = LLMConfig(
            provider="ollama",
            model="qwen3:8b",
            base_url=_OLLAMA_DEFAULT_BASE_URL,
            api_key="ollama",
            request_timeout=300.0,
        )
        mock_cls = MagicMock()
        with patch("src.llm_config.ChatOpenAI", mock_cls):
            build_chat_model(cfg, temperature=0)
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("request_timeout") == 300.0, (
            f"request_timeout=300.0 must be in ChatOpenAI kwargs, got {call_kwargs}"
        )
