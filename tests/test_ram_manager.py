"""
tests/test_ram_manager.py

Unit tests for experiments/ram_manager.py.

All tests use mocked psutil — no real processes are killed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.ram_manager import (
    HARD_DENYLIST,
    ProcessInfo,
    _get_kill_candidates,
    _is_denylisted,
    _is_allowlisted,
    _kill_process,
    free_ram,
    get_free_ram_gb,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_proc(pid: int, name: str, rss_bytes: int) -> MagicMock:
    """Build a mock psutil.Process object for process_iter."""
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "name": name,
        "memory_info": MagicMock(rss=rss_bytes),
    }
    return proc


SAFE_ALLOWLIST = ["chrome", "msedge", "spotify", "discord"]
MIXED_PROCS = [
    _make_proc(1001, "chrome", 500 * 1024 ** 2),       # allowlisted, killable
    _make_proc(1002, "msedge", 300 * 1024 ** 2),       # allowlisted, killable
    _make_proc(1003, "explorer", 200 * 1024 ** 2),     # DENYLISTED — must never be killed
    _make_proc(1004, "python", 150 * 1024 ** 2),       # DENYLISTED — running interpreter
    _make_proc(1005, "ollama", 400 * 1024 ** 2),       # DENYLISTED — LLM backend
    _make_proc(1006, "spotify", 250 * 1024 ** 2),      # allowlisted, killable
    _make_proc(1007, "dwm", 100 * 1024 ** 2),          # DENYLISTED — compositor
    _make_proc(1008, "code", 800 * 1024 ** 2),         # DENYLISTED — IDE (VS Code)
    _make_proc(1009, "svchost", 50 * 1024 ** 2),       # DENYLISTED — windows services
    _make_proc(1010, "discord", 180 * 1024 ** 2),      # allowlisted, killable
]


# ── Test 1: Hard denylist integrity ──────────────────────────────────────────

class TestHardDenylist:
    """The denylist is non-overridable and covers the required critical processes."""

    REQUIRED_ENTRIES = [
        "explorer", "dwm", "winlogon", "csrss", "services", "svchost",
        "system", "python", "python3", "ollama", "llama-server",
        "msmpeng", "sqlservr", "code", "cursor",
    ]

    def test_required_entries_present(self):
        for name in self.REQUIRED_ENTRIES:
            assert name in HARD_DENYLIST, (
                f"HARD_DENYLIST must contain '{name}' but it is missing. "
                f"This is a safety regression."
            )

    def test_denylist_is_frozenset(self):
        assert isinstance(HARD_DENYLIST, frozenset), (
            "HARD_DENYLIST must be a frozenset to prevent runtime modification."
        )

    def test_denylist_case_insensitive_matching(self):
        """_is_denylisted must match regardless of case."""
        assert _is_denylisted("Explorer")
        assert _is_denylisted("PYTHON")
        assert _is_denylisted("Ollama")
        assert _is_denylisted("DWM")
        assert not _is_denylisted("chrome")

    def test_denylist_wins_over_allowlist(self):
        """
        CRITICAL: Even if 'explorer' or 'python' is in the allowlist file,
        the denylist must prevent it from being selected as a kill candidate.
        """
        malicious_allowlist = ["explorer", "python", "ollama", "dwm", "chrome"]
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = MIXED_PROCS
            candidates = _get_kill_candidates(malicious_allowlist)

        candidate_names = {c.name.lower() for c in candidates}
        # Only chrome should be a candidate (it's on allowlist and NOT on denylist)
        assert "chrome" in candidate_names
        # All denylisted names must be absent — even though they're in the allowlist
        for denied in ["explorer", "python", "ollama", "dwm"]:
            assert denied not in candidate_names, (
                f"SAFETY FAILURE: '{denied}' appeared as a kill candidate despite being "
                f"on the HARD_DENYLIST. The denylist was bypassed."
            )

    def test_denylist_cannot_be_bypassed_via_allowlist_file(self, tmp_path):
        """
        Simulate a user editing ram_allowlist.json to include denylisted names.
        Confirm that no denylisted process is ever selected for killing.
        """
        allowlist_data = {
            "allowlist": ["explorer", "dwm", "svchost", "python", "ollama", "chrome"]
        }
        allowlist_file = tmp_path / "ram_allowlist.json"
        allowlist_file.write_text(json.dumps(allowlist_data), encoding="utf-8")

        with patch("experiments.ram_manager._ALLOWLIST_FILE", allowlist_file):
            from experiments.ram_manager import _load_allowlist
            loaded = _load_allowlist()

        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = MIXED_PROCS
            candidates = _get_kill_candidates(loaded)

        candidate_names = {c.name.lower() for c in candidates}
        for denied in ["explorer", "dwm", "svchost", "python", "ollama"]:
            assert denied not in candidate_names, (
                f"SAFETY FAILURE: '{denied}' bypassed the denylist via allowlist file."
            )
        # chrome is safe to kill and should still be a candidate
        assert "chrome" in candidate_names


# ── Test 1b: .exe suffix matching (Session 32 bug fix) ───────────────────────

class TestExeSuffixMatching:
    """
    On Windows, psutil returns process names with .exe suffix (e.g. 'chrome.exe').
    Allowlist and denylist entries omit the suffix (e.g. 'chrome', 'explorer').
    Both matching functions must normalize by stripping .exe before comparison.
    """

    def test_denylist_matches_exe_suffix(self):
        """'explorer.exe' must be denylisted even though denylist has 'explorer'."""
        assert _is_denylisted("explorer.exe")
        assert _is_denylisted("Explorer.exe")
        assert _is_denylisted("python.exe")
        assert _is_denylisted("ollama.exe")
        assert _is_denylisted("svchost.exe")
        assert _is_denylisted("dwm.exe")

    def test_allowlist_matches_exe_suffix(self):
        """'chrome.exe' must be allowlisted when allowlist has 'chrome'."""
        assert _is_allowlisted("chrome.exe", ["chrome", "spotify"])
        assert _is_allowlisted("Chrome.exe", ["chrome"])
        assert _is_allowlisted("pyrefly.exe", ["pyrefly", "chrome"])
        assert _is_allowlisted("OmenCommandCenterBackground.exe",
                               ["OmenCommandCenterBackground"])

    def test_exe_denylist_still_wins_when_both_match(self):
        """
        If 'python.exe' is in the allowlist and psutil returns 'python.exe',
        the denylist must still block it.
        """
        malicious_allowlist = ["python", "chrome"]
        exe_procs = [
            _make_proc(1, "python.exe", 200 * 1024 ** 2),   # DENYLISTED
            _make_proc(2, "chrome.exe", 300 * 1024 ** 2),   # allowlisted
            _make_proc(3, "explorer.exe", 100 * 1024 ** 2), # DENYLISTED
        ]
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = exe_procs
            candidates = _get_kill_candidates(malicious_allowlist)

        names = {c.name for c in candidates}
        assert "chrome.exe" in names
        assert "python.exe" not in names
        assert "explorer.exe" not in names

    def test_list_candidates_finds_exe_processes(self):
        """
        Regression: _get_kill_candidates must return chrome.exe / pyrefly.exe
        when allowlist has 'chrome' / 'pyrefly' (the original Session 32 bug).
        """
        exe_procs = [
            _make_proc(1, "chrome.exe", 500 * 1024 ** 2),
            _make_proc(2, "pyrefly.exe", 2000 * 1024 ** 2),
            _make_proc(3, "explorer.exe", 400 * 1024 ** 2),  # must be excluded
        ]
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = exe_procs
            candidates = _get_kill_candidates(["chrome", "pyrefly"])

        names = {c.name for c in candidates}
        assert "chrome.exe" in names, "chrome.exe must match allowlist entry 'chrome'"
        assert "pyrefly.exe" in names, "pyrefly.exe must match allowlist entry 'pyrefly'"
        assert "explorer.exe" not in names, "explorer.exe must be blocked by denylist"
        # Also check sort order — pyrefly (2000 MB) should be first
        assert candidates[0].name == "pyrefly.exe"



# ── Test 2: free_ram returns True when RAM already sufficient ─────────────────

class TestFreeRamAlreadySufficient:

    def test_returns_true_immediately_when_ram_sufficient(self):
        """When free RAM >= target, free_ram must return True and kill nothing."""
        with patch("experiments.ram_manager.get_free_ram_gb", return_value=8.0), \
             patch("experiments.ram_manager._kill_process") as mock_kill:
            result = free_ram(target_gb=4.0, interactive=False)

        assert result is True
        mock_kill.assert_not_called()

    def test_returns_true_when_exactly_at_target(self):
        with patch("experiments.ram_manager.get_free_ram_gb", return_value=4.0), \
             patch("experiments.ram_manager._kill_process") as mock_kill:
            result = free_ram(target_gb=4.0, interactive=False)

        assert result is True
        mock_kill.assert_not_called()


# ── Test 3: Interactive mode requires confirmation before any kill ─────────────

class TestInteractiveMode:

    def test_no_kill_when_user_says_no(self):
        """
        In interactive mode, if the user answers anything other than 'yes',
        no process must be killed.
        """
        with patch("experiments.ram_manager.get_free_ram_gb", return_value=1.0), \
             patch("experiments.ram_manager._get_kill_candidates", return_value=[
                 ProcessInfo(pid=1001, name="chrome", ram_mb=500),
             ]), \
             patch("builtins.input", return_value="no"), \
             patch("experiments.ram_manager._kill_process") as mock_kill:
            result = free_ram(target_gb=4.0, interactive=True)

        assert result is False
        mock_kill.assert_not_called()

    def test_no_kill_on_empty_input(self):
        """Empty input also cancels."""
        with patch("experiments.ram_manager.get_free_ram_gb", return_value=1.0), \
             patch("experiments.ram_manager._get_kill_candidates", return_value=[
                 ProcessInfo(pid=1001, name="chrome", ram_mb=500),
             ]), \
             patch("builtins.input", return_value=""), \
             patch("experiments.ram_manager._kill_process") as mock_kill:
            result = free_ram(target_gb=4.0, interactive=True)

        mock_kill.assert_not_called()

    def test_kills_when_user_confirms(self):
        """
        In interactive mode, if user types 'yes', processes should be killed.
        """
        candidate = ProcessInfo(pid=1001, name="chrome", ram_mb=500)
        with patch("experiments.ram_manager.get_free_ram_gb", side_effect=[1.0, 5.0]), \
             patch("experiments.ram_manager._get_kill_candidates", return_value=[candidate]), \
             patch("builtins.input", return_value="yes"), \
             patch("experiments.ram_manager._kill_process", return_value=True) as mock_kill, \
             patch("experiments.ram_manager.time") as mock_time:
            result = free_ram(target_gb=4.0, interactive=True)

        assert result is True
        mock_kill.assert_called_once_with(candidate)

    def test_input_not_called_in_non_interactive_mode(self):
        """Non-interactive mode must never call input()."""
        candidate = ProcessInfo(pid=1001, name="chrome", ram_mb=500)
        with patch("experiments.ram_manager.get_free_ram_gb", side_effect=[1.0, 5.0]), \
             patch("experiments.ram_manager._get_kill_candidates", return_value=[candidate]), \
             patch("builtins.input") as mock_input, \
             patch("experiments.ram_manager._kill_process", return_value=True), \
             patch("experiments.ram_manager.time"):
            free_ram(target_gb=4.0, interactive=False)

        mock_input.assert_not_called()


# ── Test 4: Non-interactive mode kills allowlist only, logs every action ──────

class TestNonInteractiveMode:

    def test_kills_only_allowlisted_candidates(self):
        """Non-interactive mode kills every candidate returned by _get_kill_candidates."""
        candidates = [
            ProcessInfo(pid=1001, name="chrome", ram_mb=500),
            ProcessInfo(pid=1002, name="spotify", ram_mb=250),
        ]
        with patch("experiments.ram_manager.get_free_ram_gb", side_effect=[1.0, 5.0]), \
             patch("experiments.ram_manager._get_kill_candidates", return_value=candidates), \
             patch("experiments.ram_manager._kill_process", return_value=True) as mock_kill, \
             patch("experiments.ram_manager.time"):
            result = free_ram(target_gb=4.0, interactive=False)

        assert result is True
        assert mock_kill.call_count == 2
        killed_names = {c.args[0].name for c in mock_kill.call_args_list}
        assert killed_names == {"chrome", "spotify"}

    def test_returns_false_when_still_insufficient_after_kills(self):
        """If RAM is still short after kills, returns False."""
        candidate = ProcessInfo(pid=1001, name="chrome", ram_mb=200)
        with patch("experiments.ram_manager.get_free_ram_gb", side_effect=[1.0, 1.5]), \
             patch("experiments.ram_manager._get_kill_candidates", return_value=[candidate]), \
             patch("experiments.ram_manager._kill_process", return_value=True), \
             patch("experiments.ram_manager.time"):
            result = free_ram(target_gb=4.0, interactive=False)

        assert result is False

    def test_returns_false_when_no_candidates(self):
        """If no allowlisted processes are running, returns False."""
        with patch("experiments.ram_manager.get_free_ram_gb", return_value=1.0), \
             patch("experiments.ram_manager._get_kill_candidates", return_value=[]):
            result = free_ram(target_gb=4.0, interactive=False)

        assert result is False


# ── Test 5: Kill safety — denylist final gate ─────────────────────────────────

class TestKillSafety:

    def test_kill_process_refuses_denylisted_name(self):
        """_kill_process has its own denylist check as the final gate."""
        denied = ProcessInfo(pid=9999, name="explorer", ram_mb=100)
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            result = _kill_process(denied)
        # Must return False — refused
        assert result is False
        # psutil.Process should never have been constructed
        mock_psutil.Process.assert_not_called()

    def test_kill_process_handles_no_such_process(self):
        """If the process is already gone, _kill_process returns True (graceful)."""
        import psutil as _psutil
        target = ProcessInfo(pid=9998, name="chrome", ram_mb=100)
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_proc = MagicMock()
            mock_proc.terminate.side_effect = _psutil.NoSuchProcess(9998)
            mock_psutil.NoSuchProcess = _psutil.NoSuchProcess
            mock_psutil.AccessDenied = _psutil.AccessDenied
            mock_psutil.Process.return_value = mock_proc
            result = _kill_process(target)
        assert result is True

    def test_kill_process_handles_access_denied(self):
        """AccessDenied returns False without raising."""
        import psutil as _psutil
        target = ProcessInfo(pid=9997, name="chrome", ram_mb=100)
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_proc = MagicMock()
            mock_proc.terminate.side_effect = _psutil.AccessDenied(9997)
            mock_psutil.NoSuchProcess = _psutil.NoSuchProcess
            mock_psutil.AccessDenied = _psutil.AccessDenied
            mock_psutil.Process.return_value = mock_proc
            result = _kill_process(target)
        assert result is False


# ── Test 6: get_free_ram_gb ────────────────────────────────────────────────────

class TestGetFreeRamGb:

    def test_returns_float(self):
        result = get_free_ram_gb()
        assert isinstance(result, float)
        assert result >= 0.0

    def test_returns_99_when_psutil_unavailable(self):
        with patch("experiments.ram_manager._PSUTIL_AVAILABLE", False):
            result = get_free_ram_gb()
        assert result == 99.0


# ── Test 7: _get_kill_candidates ordering and filtering ───────────────────────

class TestGetKillCandidates:

    def test_candidates_sorted_by_ram_descending(self):
        """Candidates should be sorted largest-first for maximum RAM freed."""
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = [
                _make_proc(1, "chrome", 200 * 1024 ** 2),
                _make_proc(2, "spotify", 500 * 1024 ** 2),
                _make_proc(3, "discord", 100 * 1024 ** 2),
            ]
            candidates = _get_kill_candidates(["chrome", "spotify", "discord"])

        assert candidates[0].name == "spotify"
        assert candidates[1].name == "chrome"
        assert candidates[2].name == "discord"

    def test_only_allowlisted_returned(self):
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = [
                _make_proc(1, "chrome", 300 * 1024 ** 2),
                _make_proc(2, "notepad", 50 * 1024 ** 2),  # not allowlisted
            ]
            candidates = _get_kill_candidates(["chrome"])

        names = [c.name for c in candidates]
        assert "chrome" in names
        assert "notepad" not in names

    def test_denylisted_excluded_even_if_in_allowlist(self):
        with patch("experiments.ram_manager.psutil") as mock_psutil:
            mock_psutil.process_iter.return_value = MIXED_PROCS
            candidates = _get_kill_candidates(SAFE_ALLOWLIST + ["explorer", "python"])

        names_lower = {c.name.lower() for c in candidates}
        assert "explorer" not in names_lower
        assert "python" not in names_lower
