"""
experiments/ram_manager.py

PACM — Safe RAM Freeing Utility
================================
Provides get_free_ram_gb() and free_ram() for use before real-model runs.

SAFETY DESIGN
-------------
- ALLOWLIST  (ram_allowlist.json): editable list of processes that MAY be killed.
- HARD_DENYLIST (this file): hardcoded, non-overridable list that can NEVER be
  killed, even if present in the allowlist file. Denylist always wins.
- interactive mode (default): shows candidates and asks for explicit confirmation
  before killing anything.
- non-interactive mode: kills allowlisted candidates without prompting but logs
  every action.

Usage
-----
    # From Python:
    from experiments.ram_manager import free_ram, get_free_ram_gb
    freed = free_ram(target_gb=4.0, interactive=True)

    # From CLI:
    python -m experiments.ram_manager --target 4.0
    python -m experiments.ram_manager --target 4.0 --non-interactive
    python -m experiments.ram_manager --check-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ALLOWLIST_FILE = _HERE / "ram_allowlist.json"
_LOG_FILE = _HERE / "logs" / "ram_manager.log"

# ── Hard denylist — NON-OVERRIDABLE ──────────────────────────────────────────
# These process names can NEVER be killed, regardless of what ram_allowlist.json
# says. Case-insensitive matching is applied at runtime.
#
# RATIONALE: killing any of these would be destructive:
#   - System shell and compositor (explorer, dwm)
#   - Security/auth processes (winlogon, csrss, lsass)
#   - Windows services layer (services, svchost)
#   - The Python interpreter running this script itself
#   - Ollama / llama-server (the LLM backend we're trying to free RAM FOR)
#   - Windows Defender (MsMpEng)
#   - SQL Server (sqlservr — may hold user data)
#   - IDE/editor processes (antigravity, code, cursor, pycharm, notepad, vim)
#
HARD_DENYLIST: frozenset[str] = frozenset({
    # Windows shell and compositor
    "explorer",
    "dwm",
    "taskbar",
    "shellexperiencehost",
    "searchhost",
    "startmenuexperiencehost",
    # Security / auth
    "winlogon",
    "csrss",
    "lsass",
    "smss",
    # Services layer
    "services",
    "svchost",
    "wininit",
    "system",
    "registry",
    "secure system",
    # Python (running this script)
    "python",
    "python3",
    "pythonw",
    # LLM backend (the thing we're freeing RAM FOR — don't kill it)
    "ollama",
    "ollama_llama_server",
    "llama-server",
    "llama_server",
    # Windows Defender
    "msmpeng",
    "nissrv",
    "antimalware service executable",
    # Database
    "sqlservr",
    "mysqld",
    "postgres",
    # IDE / editors — killing these would terminate the agent
    "antigravity",
    "antigravity-ide",
    "code",          # VS Code
    "cursor",        # Cursor IDE
    "pycharm64",
    "pycharm",
    "notepad",
    "notepad++",
    "vim",
    "nvim",
    "atom",
    "sublime_text",
    # Terminal emulators
    "windowsterminal",
    "conhost",
    "cmd",
    "powershell",
    "pwsh",
    # Windows Runtime / UWP
    "runtimebroker",
    "backgroundtaskhost",
    "applicationframehost",
    "wudfhost",
})


# ── Data types ────────────────────────────────────────────────────────────────
class ProcessInfo(NamedTuple):
    pid: int
    name: str
    ram_mb: float


# ── Core functions ─────────────────────────────────────────────────────────────

def get_free_ram_gb() -> float:
    """Return free (available) RAM in GB. Returns 99.0 if psutil unavailable."""
    if not _PSUTIL_AVAILABLE:
        return 99.0
    return psutil.virtual_memory().available / 1024 ** 3


def _load_allowlist() -> list[str]:
    """Load the editable allowlist from ram_allowlist.json. Returns [] on error."""
    if not _ALLOWLIST_FILE.exists():
        logger.warning("ram_allowlist.json not found at %s — using empty allowlist", _ALLOWLIST_FILE)
        return []
    try:
        data = json.loads(_ALLOWLIST_FILE.read_text(encoding="utf-8"))
        raw = data.get("allowlist", [])
        if not isinstance(raw, list):
            logger.error("ram_allowlist.json 'allowlist' key is not a list — using empty allowlist")
            return []
        return [str(name) for name in raw]
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load ram_allowlist.json: %s", exc)
        return []


def _is_denylisted(name: str) -> bool:
    """Return True if the process name matches the hard denylist (case-insensitive).

    On Windows, psutil returns names with .exe suffix (e.g., 'chrome.exe').
    Both the raw name and the .exe-stripped name are checked so that
    allowlist/denylist entries without .exe work correctly on all platforms.
    """
    normalized = name.lower().strip()
    normalized_no_ext = normalized.removesuffix(".exe")
    return normalized in HARD_DENYLIST or normalized_no_ext in HARD_DENYLIST


def _is_allowlisted(name: str, allowlist: list[str]) -> bool:
    """Return True if name matches any entry in the allowlist (case-insensitive).

    Matches both 'chrome' and 'chrome.exe' against allowlist entry 'chrome'.
    """
    normalized = name.lower().strip()
    normalized_no_ext = normalized.removesuffix(".exe")
    allowlist_lower = [a.lower().strip() for a in allowlist]
    return normalized in allowlist_lower or normalized_no_ext in allowlist_lower


def _get_kill_candidates(allowlist: list[str]) -> list[ProcessInfo]:
    """
    Return all running processes that are on the allowlist AND NOT on the
    denylist, sorted by RAM usage descending.

    Conflicts (name in both allowlist and denylist) are logged as warnings.
    """
    if not _PSUTIL_AVAILABLE:
        return []

    # Check for conflicts in the allowlist file
    conflicts = [a for a in allowlist if _is_denylisted(a)]
    for conflict in conflicts:
        logger.warning(
            "CONFLICT: '%s' is in both ram_allowlist.json and HARD_DENYLIST. "
            "Denylist wins — this process will never be killed.",
            conflict,
        )
        print(f"  [WARN] '{conflict}' is in both the allowlist file and the hard denylist. "
              f"Denylist wins — it will not be killed.")

    candidates: list[ProcessInfo] = []
    seen_pids: set[int] = set()

    try:
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                pid = proc.info["pid"]
                name = proc.info["name"] or ""
                mem_info = proc.info["memory_info"]
                if pid in seen_pids or mem_info is None:
                    continue
                seen_pids.add(pid)

                # Safety double-check: never pass through a denylisted process
                if _is_denylisted(name):
                    continue
                if not _is_allowlisted(name, allowlist):
                    continue

                ram_mb = mem_info.rss / 1024 ** 2
                candidates.append(ProcessInfo(pid=pid, name=name, ram_mb=ram_mb))

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as exc:
        logger.error("Error enumerating processes: %s", exc)

    # Sort by RAM usage descending (kill biggest first for maximum effect)
    candidates.sort(key=lambda p: p.ram_mb, reverse=True)
    return candidates


def _log_action(message: str) -> None:
    """Append a timestamped line to the RAM manager log file."""
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except OSError:
        pass  # Non-fatal: logging failure shouldn't block the operation


def _kill_process(proc_info: ProcessInfo) -> bool:
    """
    Attempt to terminate a process. Returns True if killed or already gone.
    Never raises. Logs the action.
    """
    if not _PSUTIL_AVAILABLE:
        return False

    # Final safety check — this is the last gate before any actual kill
    if _is_denylisted(proc_info.name):
        logger.error(
            "SAFETY BLOCK: Attempted to kill denylisted process %s (PID %d). Skipping.",
            proc_info.name, proc_info.pid,
        )
        _log_action(f"SAFETY BLOCK: Refused to kill {proc_info.name} (PID {proc_info.pid}) — on denylist")
        return False

    try:
        proc = psutil.Process(proc_info.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            proc.kill()  # Force kill if terminate didn't work in 5s
        _log_action(f"KILLED {proc_info.name} (PID {proc_info.pid}, {proc_info.ram_mb:.0f} MB)")
        logger.info("Killed %s (PID %d, %.0f MB)", proc_info.name, proc_info.pid, proc_info.ram_mb)
        return True
    except psutil.NoSuchProcess:
        _log_action(f"ALREADY_GONE {proc_info.name} (PID {proc_info.pid})")
        return True  # Already gone — that's fine
    except psutil.AccessDenied as exc:
        _log_action(f"ACCESS_DENIED {proc_info.name} (PID {proc_info.pid}): {exc}")
        logger.warning("AccessDenied killing %s (PID %d): %s", proc_info.name, proc_info.pid, exc)
        return False
    except Exception as exc:
        _log_action(f"ERROR killing {proc_info.name} (PID {proc_info.pid}): {exc}")
        logger.error("Unexpected error killing %s: %s", proc_info.name, exc)
        return False


def free_ram(
    target_gb: float = 4.0,
    interactive: bool = True,
    _allowlist_override: list[str] | None = None,  # for testing
) -> bool:
    """
    Attempt to free RAM to reach target_gb available.

    Parameters
    ----------
    target_gb:   GB of free RAM to target. Returns True immediately if already met.
    interactive: If True (default), print candidates and prompt for confirmation
                 before killing anything. If False, kill allowlisted candidates
                 silently (non-interactive / unattended mode).

    Returns
    -------
    True if target_gb of RAM is free after the function returns.
    False if RAM is still insufficient after all attempts.
    """
    if not _PSUTIL_AVAILABLE:
        print("  [ram_manager] psutil not available — skipping RAM management.")
        return True  # Don't block; let the caller decide

    free_now = get_free_ram_gb()
    print(f"  [ram_manager] Free RAM: {free_now:.2f} GB  (target: {target_gb:.2f} GB)")

    if free_now >= target_gb:
        print(f"  [ram_manager] Already have {free_now:.2f} GB free — no action needed.")
        return True

    needed = target_gb - free_now
    print(f"  [ram_manager] Need to free {needed:.2f} GB more.")

    allowlist = _allowlist_override if _allowlist_override is not None else _load_allowlist()
    candidates = _get_kill_candidates(allowlist)

    if not candidates:
        print("  [ram_manager] No allowlisted processes are currently running.")
        print(f"  [ram_manager] Cannot free RAM automatically. Free RAM manually and retry.")
        _log_action(f"NO_CANDIDATES: need {needed:.2f} GB, allowlist has no running matches")
        return False

    # Show candidates
    print(f"\n  Processes eligible to kill (allowlisted, not denylisted):")
    print(f"  {'PID':>6}  {'Name':<35}  {'RAM (MB)':>10}")
    print(f"  {'-'*6}  {'-'*35}  {'-'*10}")
    for p in candidates:
        print(f"  {p.pid:>6}  {p.name:<35}  {p.ram_mb:>10.0f}")

    estimated_freed = sum(p.ram_mb for p in candidates) / 1024
    print(f"\n  Estimated RAM recoverable: {estimated_freed:.2f} GB")

    if interactive:
        print(f"\n  Kill ALL {len(candidates)} process(es) above to free ~{estimated_freed:.2f} GB?")
        print("  Type 'yes' to confirm, anything else to cancel: ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer != "yes":
            print("  [ram_manager] Cancelled — no processes killed.")
            _log_action("USER_CANCELLED: interactive prompt declined")
            return False

        # User confirmed — kill all candidates
        _log_action(f"USER_CONFIRMED: killing {len(candidates)} processes")
        for p in candidates:
            print(f"  Killing {p.name} (PID {p.pid})... ", end="", flush=True)
            ok = _kill_process(p)
            print("done" if ok else "FAILED")
    else:
        # Non-interactive: kill without prompting, but log everything
        _log_action(f"NON_INTERACTIVE: killing {len(candidates)} allowlisted processes")
        for p in candidates:
            _kill_process(p)

    # Wait a moment for RAM to be released
    time.sleep(1.5)

    # Re-check
    free_after = get_free_ram_gb()
    print(f"\n  [ram_manager] Free RAM after: {free_after:.2f} GB")

    if free_after >= target_gb:
        print(f"  [ram_manager] Target reached ({free_after:.2f} >= {target_gb:.2f} GB). Proceeding.")
        _log_action(f"SUCCESS: {free_after:.2f} GB free (target {target_gb:.2f} GB)")
        return True
    else:
        still_needed = target_gb - free_after
        print(f"  [ram_manager] Still {still_needed:.2f} GB short of target after killing allowlisted processes.")
        print("  [ram_manager] Close additional applications manually and retry.")
        _log_action(f"INSUFFICIENT: {free_after:.2f} GB free, still need {still_needed:.2f} GB more")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PACM RAM Manager — safely free RAM before a real-model run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.ram_manager --check-only
  python -m experiments.ram_manager --target 4.0
  python -m experiments.ram_manager --target 4.0 --non-interactive
  python -m experiments.ram_manager --list-candidates
        """,
    )
    parser.add_argument(
        "--target", type=float, default=4.0,
        help="Target free RAM in GB (default: 4.0)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Print current free RAM and candidates without killing anything.",
    )
    parser.add_argument(
        "--list-candidates", action="store_true",
        help="List allowlisted processes currently running, then exit.",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Kill allowlisted candidates without prompting (unattended mode). "
             "Every kill is logged to experiments/logs/ram_manager.log.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if not _PSUTIL_AVAILABLE:
        print("ERROR: psutil is required. Install with: pip install psutil")
        sys.exit(1)

    if args.check_only:
        free = get_free_ram_gb()
        total = psutil.virtual_memory().total / 1024 ** 3
        used = total - free
        print(f"RAM: {free:.2f} GB free / {total:.2f} GB total ({used:.2f} GB used)")
        sys.exit(0)

    if args.list_candidates:
        allowlist = _load_allowlist()
        candidates = _get_kill_candidates(allowlist)
        if not candidates:
            print("No allowlisted processes currently running.")
        else:
            print(f"{'PID':>6}  {'Name':<35}  {'RAM (MB)':>10}")
            print(f"{'-'*6}  {'-'*35}  {'-'*10}")
            for p in candidates:
                print(f"{p.pid:>6}  {p.name:<35}  {p.ram_mb:>10.0f}")
        sys.exit(0)

    success = free_ram(
        target_gb=args.target,
        interactive=not args.non_interactive,
    )
    sys.exit(0 if success else 1)
