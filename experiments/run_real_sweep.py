"""
experiments/run_real_sweep.py

Session 10 — One-shot orchestrator for the full 14-config real-LLM sweep.

Runs in order:
  1. Probe backend (must PASS)
  2. Full 14-config matrix sweep  -> real_llm_sweep_summary.json
  3. Generate paper tables         -> paper_table.tex / paper_table.md
  4. Generate SVG figures          -> experiments/figures/*.svg

Usage:
    python -m experiments.run_real_sweep
    python -m experiments.run_real_sweep --quick   # 2-config pilot only
    python -m experiments.run_real_sweep --dry-run # skip sweep, regen tables only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
load_dotenv(_repo_root / ".env")

REAL_SWEEP_JSON  = _repo_root / "experiments" / "real_llm_sweep_summary.json"
FIGURES_DIR      = _repo_root / "experiments" / "figures"

# Resolve the Python interpreter to use for subprocesses.
# If the current process is NOT running inside the project's .venv,
# locate the venv Python explicitly so all subprocesses have the correct
# packages (langchain_core, etc.).  This prevents the failure where
# `python` on PATH resolves to the bare system interpreter.
def _resolve_python() -> str:
    venv_root = _repo_root / ".venv"
    candidates = [
        venv_root / "Scripts" / "python.exe",   # Windows
        venv_root / "bin" / "python",            # POSIX
        venv_root / "bin" / "python3",           # POSIX alt
    ]
    current = Path(sys.executable).resolve()
    # If already inside the venv, use as-is
    try:
        current.relative_to(venv_root.resolve())
        return sys.executable
    except ValueError:
        pass
    # Otherwise prefer the first existing venv candidate
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back to sys.executable and let the user fix it
    return sys.executable

PYTHON = _resolve_python()


def _run(cmd: list[str], label: str) -> int:
    """Run a subprocess command, print its output, and return exit code."""
    print(f"\n{'-'*60}")
    print(f"  [{label}]")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'-'*60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - t0
    rc = result.returncode
    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
    print(f"\n  {status}  ({elapsed:.1f}s)")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PACM Session 10 — Real-LLM sweep orchestrator."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 2-config pilot only (langgraph, ignore_previous).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip probe and sweep; just regenerate tables and figures from existing JSON.",
    )
    parser.add_argument(
        "--skip-probe", action="store_true",
        help="Skip the backend probe step (use if already verified).",
    )
    parser.add_argument(
        "--samples-per-config", type=int, default=1, dest="samples_per_config",
        help=(
            "Number of independent task-set runs per config (default: 1). "
            "Set to 3 for a pilot multi-sample run. "
            "ABR_inclusive (BLOCK+FLAG) is reported alongside strict ABR."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  PACM Session 10 — Real-LLM Sweep Orchestrator")
    print(f"  quick={args.quick}  dry_run={args.dry_run}")
    print(f"{'='*60}")

    overall_ok = True

    # ── Step 1: Probe ─────────────────────────────────────────────────────────
    if not args.dry_run and not args.skip_probe:
        rc = _run(
            [PYTHON, "-m", "experiments.probe_llm_backend"],
            "1/4 Backend Probe",
        )
        if rc != 0:
            print("\n  !! Probe FAILED. Fix backend configuration before sweeping.")
            print("     Set LLM_PROVIDER=ollama and ensure 'ollama serve' is running,")
            print("     OR set OPENAI_API_KEY in .env for the OpenAI provider.")
            return 1
    else:
        print("\n  [1/4 Backend Probe] SKIPPED")

    # ── Step 2: Sweep ─────────────────────────────────────────────────────────
    if not args.dry_run:
        sweep_cmd = [
            PYTHON, "-m", "experiments.run_sweep",
            "--output", str(REAL_SWEEP_JSON),
            "--log-level", args.log_level,
        ]
        if args.quick:
            sweep_cmd.append("--quick")
        if args.samples_per_config > 1:
            sweep_cmd.extend(["--samples-per-config", str(args.samples_per_config)])

        rc = _run(sweep_cmd, "2/4 Full 14-Config Sweep")
        if rc != 0:
            print("\n  !! Sweep FAILED. Check logs for details.")
            overall_ok = False
            # Continue to try regenerating tables from whatever JSON exists
    else:
        print("\n  [2/4 Sweep] SKIPPED (--dry-run)")

    if not REAL_SWEEP_JSON.exists():
        print(f"\n  !! {REAL_SWEEP_JSON} not found — cannot regenerate tables/figures.")
        return 1

    # ── Step 3: Paper tables ──────────────────────────────────────────────────
    rc = _run(
        [PYTHON, "-m", "experiments.generate_paper_tables", "--real"],
        "3/4 Paper Tables",
    )
    if rc != 0:
        overall_ok = False

    # ── Step 4: Paper figures ─────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rc = _run(
        [
            PYTHON, "-m", "experiments.generate_paper_figures",
            "--real",
            "--output-dir", str(FIGURES_DIR),
        ],
        "4/4 Paper Figures",
    )
    if rc != 0:
        overall_ok = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if overall_ok:
        print("  ALL STEPS PASSED")
        _print_sweep_highlights()
    else:
        print("  COMPLETED WITH ERRORS — see output above")
    print(f"{'='*60}\n")

    return 0 if overall_ok else 1


def _print_sweep_highlights() -> None:
    """Print a compact summary of the real sweep metrics."""
    if not REAL_SWEEP_JSON.exists():
        return
    try:
        with open(REAL_SWEEP_JSON) as fh:
            data = json.load(fh)
        runs = data.get("runs", [])
        params = data.get("parameters", {})
        print(f"\n  Provider : {params.get('llm_provider', '?')}  "
              f"Model: {params.get('model', '?')}")
        print(f"  Configs  : {len(runs)}  Total tasks: {data.get('n_total_tasks', '?')}")
        print()
        defended = [r for r in runs if r.get("mode") == "defended"]
        if defended:
            abrs = [r["metrics"]["abr"] for r in defended if r["metrics"].get("abr") is not None]
            fprs = [r["metrics"]["fpr"] for r in defended if r["metrics"].get("fpr") is not None]
            if abrs:
                print(f"  Defended ABR  : min={min(abrs)*100:.1f}%  "
                      f"max={max(abrs)*100:.1f}%  "
                      f"mean={sum(abrs)/len(abrs)*100:.1f}%")
            if fprs:
                print(f"  Defended FPR  : min={min(fprs)*100:.1f}%  "
                      f"max={max(fprs)*100:.1f}%  "
                      f"mean={sum(fprs)/len(fprs)*100:.1f}%")
        undefended = [r for r in runs if r.get("mode") == "undefended"]
        if undefended:
            btsrs = [r["metrics"]["btsr"] for r in undefended
                     if r["metrics"].get("btsr") is not None]
            if btsrs:
                print(f"  Undefended BTSR: mean={sum(btsrs)/len(btsrs)*100:.1f}%")
    except Exception as exc:
        print(f"  (could not read sweep highlights: {exc})")


if __name__ == "__main__":
    sys.exit(main())
