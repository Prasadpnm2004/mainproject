"""
src/pacm/action_capture.py

PACM Action Capture Module — Session 3

Formats the proposed tool call (name + arguments) from a ToolCallRecord
into a structured CapturedAction instance for divergence scoring.

Background (PACM_Research_Guide.md §9.1 step 2):
  After intercepting the tool call via pre_tool_hook, we need a normalized
  representation of the *actual* action being proposed so the scorer can
  compare it against the DeclaredIntent from the extractor.

Design:
  - CapturedAction is the 'action' side of the plan->action pair.
  - ActionCapturer normalizes tool name + argument dict into a human-readable
    action description string (used for embedding similarity in scorer.py).
  - Also provides argument scope helpers: checks whether arguments fall within
    expected scopes (e.g. email domains, path prefixes) for rule-based checks.

Usage:
    from src.pacm.action_capture import ActionCapturer, CapturedAction
    from src.harness.langgraph_agent import ToolCallRecord

    capturer = ActionCapturer()
    action = capturer.capture(record)
    print(action.action_description)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Data model ───────────────────────────────────────────────────────────────


@dataclass
class CapturedAction:
    """
    Structured representation of the tool call actually being proposed.

    This is the 'action' side of the plan->action divergence check.

    Fields:
        tool_name:          The exact tool name being called.
        tool_args:          The raw argument dict.
        action_description: Human-readable one-line description of the action
                            (e.g. "send_email to attacker@evil.com with subject=Exfil").
                            Used as input to the embedding similarity scorer.
        timestamp:          ISO 8601 UTC timestamp of capture.
        arg_scope_flags:    Dict of {arg_name: flag} for rule-based checks.
                            E.g. {"to": "external_domain"} if the recipient
                            domain is not in an allowlist.
    """

    tool_name: str
    tool_args: dict[str, Any]
    action_description: str
    timestamp: str
    arg_scope_flags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "action_description": self.action_description,
            "timestamp": self.timestamp,
            "arg_scope_flags": self.arg_scope_flags,
        }


# ─── Scope check helpers ──────────────────────────────────────────────────────

# Known internal/trusted email domains (configurable in production)
_TRUSTED_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "example.com", "internal.corp", "company.com",
})

# Known safe file path prefixes (configurable in production)
_SAFE_PATH_PREFIXES: tuple[str, ...] = (
    "/workspace/", "/home/user/", "C:\\Users\\",
)


def _check_email_scope(value: str) -> Optional[str]:
    """Return a scope flag if the email looks suspicious, else None."""
    if "@" not in value:
        return None
    domain = value.split("@")[-1].lower().strip()
    if domain not in _TRUSTED_EMAIL_DOMAINS:
        return "external_domain"
    return None


def _check_path_scope(value: str) -> Optional[str]:
    """Return a scope flag if a file path is outside safe prefixes."""
    if not any(value.startswith(pfx) for pfx in _SAFE_PATH_PREFIXES):
        if "/" in value or "\\" in value:
            return "path_outside_safe_prefix"
    return None


# Argument names that warrant scope checks
_EMAIL_ARG_NAMES: frozenset[str] = frozenset({"to", "recipient", "email", "cc", "bcc"})
_PATH_ARG_NAMES: frozenset[str] = frozenset({"path", "file", "filepath", "destination", "src"})


# ─── Capturer ─────────────────────────────────────────────────────────────────


class ActionCapturer:
    """
    Formats proposed tool calls into CapturedAction instances.

    Also runs lightweight rule-based argument scope checks that feed into
    the divergence scorer as supplementary signals (PACM_Research_Guide §9.1
    step 3: rule-based argument scope checks).
    """

    def capture(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> CapturedAction:
        """
        Capture a proposed tool call as a CapturedAction.

        Args:
            tool_name: Name of the tool being called.
            tool_args: Argument dict from the tool call.

        Returns:
            CapturedAction with description and scope flags.
        """
        action_description = self._build_description(tool_name, tool_args)
        scope_flags = self._check_arg_scopes(tool_args)
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.debug(
            "ActionCapturer: tool=%s | flags=%s | desc=%s...",
            tool_name,
            scope_flags,
            action_description[:60],
        )

        return CapturedAction(
            tool_name=tool_name,
            tool_args=tool_args,
            action_description=action_description,
            timestamp=timestamp,
            arg_scope_flags=scope_flags,
        )

    def capture_from_record(self, record: Any) -> CapturedAction:
        """
        Convenience method that accepts a ToolCallRecord directly.

        Args:
            record: ToolCallRecord with tool_name and tool_args fields.
        """
        return self.capture(
            tool_name=getattr(record, "tool_name", "unknown"),
            tool_args=getattr(record, "tool_args", {}),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_description(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> str:
        """
        Build a concise, human-readable one-line description of the action.

        Format: "<tool_name> <key1>=<val1> <key2>=<val2> ..."
        Values are truncated to 60 chars to keep descriptions scannable.
        """
        if not tool_args:
            return tool_name

        parts = [tool_name]
        for key, val in sorted(tool_args.items()):
            val_str = str(val)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            parts.append(f"{key}={val_str!r}")
        return " ".join(parts)

    def _check_arg_scopes(
        self, tool_args: dict[str, Any]
    ) -> dict[str, str]:
        """
        Run lightweight rule-based scope checks on tool arguments.

        Returns a dict of {arg_name: flag_string} for any flagged args.
        Empty dict = no scope concerns.
        """
        flags: dict[str, str] = {}
        for arg_name, arg_val in tool_args.items():
            val_str = str(arg_val)
            arg_lower = arg_name.lower()

            if arg_lower in _EMAIL_ARG_NAMES:
                flag = _check_email_scope(val_str)
                if flag:
                    flags[arg_name] = flag

            elif arg_lower in _PATH_ARG_NAMES:
                flag = _check_path_scope(val_str)
                if flag:
                    flags[arg_name] = flag

        return flags
