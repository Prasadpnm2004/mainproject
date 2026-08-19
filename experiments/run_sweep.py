"""
experiments/run_sweep.py

PACM Automated Multi-Scenario Experiment Sweep Runner — Session 5

Executes a full matrix of configurations through run_experiment.run_experiment()
and aggregates all results into experiments/sweep_summary.json.

Sweep matrix:
  harnesses:     [langgraph, custom_loop]
  attack types:  [ignore_previous, injec_agent]  (+ None = all attacks)
  modes:         [undefended, defended]
  policies (defended only): [STANDARD_POLICY, CONSERVATIVE_POLICY, AGGRESSIVE_POLICY]

Each cell in the matrix becomes one "run" whose records land in
experiments/pacm_results.jsonl (via write_results) and whose computed
MetricsReport is stored in sweep_summary.json.

Usage:
    # Full sweep (all 14 run configurations):
    python -m experiments.run_sweep

    # Quick subset sweep (2 configs):
    python -m experiments.run_sweep --quick

    # Custom sweep parameters:
    python -m experiments.run_sweep --harnesses langgraph --attacks ignore_previous

    # Only undefended baseline:
    python -m experiments.run_sweep --modes undefended

Output files:
    experiments/pacm_results.jsonl   — raw per-task JSONL records (appended)
    experiments/sweep_summary.json   — per-run MetricsReports + sweep metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Optional

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
load_dotenv(_repo_root / ".env")

logger = logging.getLogger(__name__)

SWEEP_SUMMARY_PATH = _repo_root / "experiments" / "sweep_summary.json"
RESULTS_JSONL_PATH = _repo_root / "experiments" / "pacm_results.jsonl"


# ─── Sweep configuration ──────────────────────────────────────────────────────

POLICY_CONFIGS = {
    "standard": {
        "policy_mode": "block",
        "divergence_threshold": 0.60,
        "label": "STANDARD_POLICY",
    },
    "conservative": {
        "policy_mode": "block",
        "divergence_threshold": 0.80,
        "label": "CONSERVATIVE_POLICY",
    },
    "aggressive": {
        "policy_mode": "block",
        "divergence_threshold": 0.40,
        "label": "AGGRESSIVE_POLICY",
    },
}

DEFAULT_HARNESSES  = ["langgraph", "custom_loop"]
DEFAULT_ATTACKS    = ["ignore_previous", "injec_agent"]
DEFAULT_MODES      = ["undefended", "defended"]
DEFAULT_POLICIES   = list(POLICY_CONFIGS.keys())


def build_sweep_matrix(
    harnesses: list[str],
    attacks: list[str],
    modes: list[str],
    policies: list[str],
) -> list[dict[str, Any]]:
    """
    Generate all run configurations for the sweep.

    Each configuration is a dict with all parameters needed to call
    run_experiment.run_experiment().

    Undefended runs: one per (harness × attack) — no policy variation.
    Defended runs:   one per (harness × attack × policy).
    """
    configs = []

    for harness, attack, mode in product(harnesses, attacks, modes):
        if mode == "undefended":
            run_id = f"sweep_{mode}_{harness}_{attack}_{str(uuid.uuid4())[:6]}"
            configs.append({
                "run_id": run_id,
                "mode": mode,
                "harness": harness,
                "attack": attack,
                "policy_key": None,
                "policy_label": "none",
                "policy_mode": "allow",
                "divergence_threshold": 0.60,
                "label": f"{mode}/{harness}/{attack}",
            })
        else:  # defended
            for policy_key in policies:
                pconf = POLICY_CONFIGS[policy_key]
                run_id = f"sweep_{mode}_{harness}_{attack}_{policy_key}_{str(uuid.uuid4())[:6]}"
                configs.append({
                    "run_id": run_id,
                    "mode": mode,
                    "harness": harness,
                    "attack": attack,
                    "policy_key": policy_key,
                    "policy_label": pconf["label"],
                    "policy_mode": pconf["policy_mode"],
                    "divergence_threshold": pconf["divergence_threshold"],
                    "label": f"{mode}/{harness}/{attack}/{policy_key}",
                })

    return configs


def run_sweep(
    harnesses: list[str] = DEFAULT_HARNESSES,
    attacks: list[str] = DEFAULT_ATTACKS,
    modes: list[str] = DEFAULT_MODES,
    policies: list[str] = DEFAULT_POLICIES,
    model_name: str = "gpt-4o-mini",
    max_iter: int = 5,
    append_jsonl: bool = True,
) -> dict[str, Any]:
    """
    Execute the full sweep and return a sweep summary dict.

    Args:
        harnesses:    List of harness names to include.
        attacks:      List of attack type filters.
        modes:        ["undefended", "defended"] or subset.
        policies:     Policy keys from POLICY_CONFIGS (defended only).
        model_name:   LLM model name (stub if no API key).
        max_iter:     Max harness iterations per task.
        append_jsonl: If True, append raw records to pacm_results.jsonl.

    Returns:
        Sweep summary dict with per-run MetricsReports.
    """
    from experiments.run_experiment import run_experiment, write_results
    from experiments.metrics import MetricsProcessor, MetricsReport

    configs = build_sweep_matrix(harnesses, attacks, modes, policies)
    sweep_start = datetime.now(timezone.utc).isoformat()
    sweep_id = str(uuid.uuid4())[:8]

    print(f"\n{'='*72}")
    print(f"  PACM Session 5 — Automated Sweep")
    print(f"  sweep_id={sweep_id}  configs={len(configs)}")
    print(f"  harnesses={harnesses}  attacks={attacks}")
    print(f"  modes={modes}  policies={policies}")
    print(f"{'='*72}\n")

    all_records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for i, cfg in enumerate(configs, 1):
        print(f"  [{i:02d}/{len(configs):02d}] {cfg['label']} run_id={cfg['run_id']}", flush=True)
        t0 = time.perf_counter()

        try:
            records = run_experiment(
                mode=cfg["mode"],
                harness_name=cfg["harness"],
                attack_filter=cfg["attack"],
                policy_mode=cfg["policy_mode"],
                divergence_threshold=cfg["divergence_threshold"],
                model_name=model_name,
                max_iter=max_iter,
                run_id=cfg["run_id"],
            )
        except Exception as exc:
            logger.exception("Sweep config %s failed: %s", cfg["label"], exc)
            records = []

        elapsed = time.perf_counter() - t0
        all_records.extend(records)

        # Compute metrics for this run
        mp = MetricsProcessor(records)
        report = mp.compute(cfg["run_id"])

        summary = {
            **cfg,
            "n_tasks": len(records),
            "elapsed_s": round(elapsed, 3),
            "metrics": report.to_dict(),
        }
        run_summaries.append(summary)

        abr_str  = f"{report.abr*100:.1f}%" if report.abr is not None else "N/A"
        btsr_str = f"{report.btsr*100:.1f}%" if report.btsr is not None else "N/A"
        fpr_str  = f"{report.fpr*100:.1f}%" if report.fpr is not None else "N/A"
        print(
            f"         ABR={abr_str}  BTSR={btsr_str}  FPR={fpr_str}"
            f"  t={elapsed:.2f}s",
            flush=True,
        )

    # Write all raw records to JSONL
    if append_jsonl and all_records:
        write_results(all_records)

    sweep_summary = {
        "sweep_id": sweep_id,
        "sweep_start_utc": sweep_start,
        "sweep_end_utc": datetime.now(timezone.utc).isoformat(),
        "n_configs": len(configs),
        "n_total_tasks": len(all_records),
        "parameters": {
            "harnesses": harnesses,
            "attacks": attacks,
            "modes": modes,
            "policies": policies,
            "model": model_name,
        },
        "runs": run_summaries,
    }

    # Save sweep summary
    with open(SWEEP_SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(sweep_summary, fh, indent=2, default=str)
    print(f"\n  Sweep summary written -> {SWEEP_SUMMARY_PATH}")

    return sweep_summary


def print_sweep_table(sweep_summary: dict[str, Any]) -> None:
    """Print a compact ASCII results table for the sweep."""
    runs = sweep_summary.get("runs", [])
    header = f"{'Label':<40} {'ABR':>7} {'BTSR':>7} {'FPR':>7} {'t(s)':>6}"
    print(f"\n{'-'*72}")
    print("  PACM Sweep Results")
    print(f"{'-'*72}")
    print(f"  {header}")
    print(f"  {'-'*70}")
    for r in runs:
        m = r.get("metrics", {})
        abr  = f"{m['abr']*100:.1f}%" if m.get("abr") is not None else "N/A"
        btsr = f"{m['btsr']*100:.1f}%" if m.get("btsr") is not None else "N/A"
        fpr  = f"{m['fpr']*100:.1f}%" if m.get("fpr") is not None else "N/A"
        elapsed = r.get("elapsed_s", 0)
        label = r.get("label", "")[:40]
        print(f"  {label:<40} {abr:>7} {btsr:>7} {fpr:>7} {elapsed:>6.2f}")
    print(f"{'-'*72}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PACM Session 5 — Automated multi-scenario sweep runner."
    )
    parser.add_argument(
        "--harnesses", nargs="+", default=DEFAULT_HARNESSES,
        choices=["langgraph", "custom_loop"],
    )
    parser.add_argument(
        "--attacks", nargs="+", default=DEFAULT_ATTACKS,
        choices=["ignore_previous", "injec_agent"],
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        choices=["undefended", "defended"],
    )
    parser.add_argument(
        "--policies", nargs="+", default=DEFAULT_POLICIES,
        choices=list(POLICY_CONFIGS.keys()),
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model (stub if no API key).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=5,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 harness, 1 attack, 1 policy.",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.quick:
        harnesses = ["langgraph"]
        attacks   = ["ignore_previous"]
        modes     = DEFAULT_MODES
        policies  = ["standard"]
    else:
        harnesses = args.harnesses
        attacks   = args.attacks
        modes     = args.modes
        policies  = args.policies

    summary = run_sweep(
        harnesses=harnesses,
        attacks=attacks,
        modes=modes,
        policies=policies,
        model_name=args.model,
        max_iter=args.max_iter,
    )
    print_sweep_table(summary)
