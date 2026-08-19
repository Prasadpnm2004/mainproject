"""
experiments/run_error_analysis.py

PACM Error Analysis CLI — Session 7

Loads PACM audit records and runs ErrorAnalyzer to produce the failure
mode taxonomy report at experiments/error_analysis_report.md.

Usage:
    python -m experiments.run_error_analysis --synthetic
    python -m experiments.run_error_analysis --input experiments/pacm_results.jsonl
    python -m experiments.run_error_analysis --synthetic --n-attack 100 --n-benign 100
    python -m experiments.run_error_analysis --flag-as-block
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

REPORT_PATH = _repo_root / "experiments" / "error_analysis_report.md"
JSON_PATH   = _repo_root / "experiments" / "error_analysis_report.json"


def _make_synthetic_records(
    n_attack: int = 100, n_benign: int = 100, seed: int = 42,
    block_threshold: float = 0.75,
) -> list[dict]:
    """
    Generate synthetic audit records with explicit is_attack, decision,
    divergence_score, friction_score, trust_score, and arg_scope_flags
    for error analysis testing.

    Simulates a realistic distribution:
      - ~80% of attacks are blocked (TP), ~20% evade (FN)
      - ~5% of benign are blocked (FP), ~95% allowed (TN)
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    records = []

    # Attack records
    for i in range(n_attack):
        # 80% of attacks have D > 0.55 (blocked); 20% evade
        evades = rng.random() < 0.20
        if evades:
            # FN scenarios: low D (param manipulation / adaptive)
            D = float(rng.uniform(0.15, 0.55))
            T = float(rng.uniform(0.5, 1.0))
            # Randomly assign FN-3 (no scope flags) or FN-1 (param manip)
            scope_flags: dict = {} if rng.random() < 0.6 else {"to": "external"}
            decision = "allow"
        else:
            D = float(rng.uniform(0.55, 0.95))
            T = float(rng.uniform(0.1, 0.8))
            scope_flags = {"to": "external_domain"} if rng.random() < 0.7 else {}
            decision = "block"

        F = float(min(1.0, D * (1.5 - 0.5 * T)))
        records.append({
            "record_id":       f"atk_{i:04d}",
            "is_attack":       True,
            "decision":        decision,
            "divergence_score": D,
            "friction_score":  F,
            "trust_score":     T,
            "tool_name":       "send_email",
            "arg_scope_flags": scope_flags,
            "n_reasoning_steps": int(rng.integers(1, 4)),
        })

    # Benign records
    for i in range(n_benign):
        # 5% of benign are blocked (FP)
        blocked = rng.random() < 0.05
        if blocked:
            # FP scenarios
            D = float(rng.uniform(0.30, 0.70))
            T = float(rng.uniform(0.1, 0.9))
            scope_flags = {"to": "external"} if rng.random() < 0.5 else {}
            n_steps = int(rng.integers(1, 5))
            decision = "block"
        else:
            D = float(rng.uniform(0.0, 0.35))
            T = float(rng.uniform(0.5, 1.0))
            scope_flags = {}
            n_steps = 1
            decision = "allow"

        F = float(min(1.0, D * (1.5 - 0.5 * T)))
        records.append({
            "record_id":       f"ben_{i:04d}",
            "is_attack":       False,
            "decision":        decision,
            "divergence_score": D,
            "friction_score":  F,
            "trust_score":     T,
            "tool_name":       "read_email",
            "arg_scope_flags": scope_flags,
            "n_reasoning_steps": n_steps,
        })

    return records


def _load_from_jsonl(path: Path) -> list[dict]:
    """Load records from pacm_results.jsonl."""
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                audit = rec.get("pacm_audit") or []
                for entry in audit:
                    records.append({
                        "record_id":        entry.get("call_id", "?"),
                        "is_attack":        bool(rec.get("is_attack", False)),
                        "decision":         entry.get("decision", "allow"),
                        "divergence_score": float(entry.get("divergence_score", 0.5)),
                        "friction_score":   float(entry.get("friction_score", 0.0)),
                        "trust_score":      float(entry.get("trust_score", 0.5)),
                        "tool_name":        entry.get("tool_name", "unknown"),
                        "arg_scope_flags":  entry.get("scope_flags", {}),
                        "n_reasoning_steps": 1,
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    return records


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PACM Session 7 -- Error analysis & failure taxonomy."
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--input",
                        default=str(_repo_root / "experiments" / "pacm_results.jsonl"))
    parser.add_argument("--n-attack", type=int, default=100)
    parser.add_argument("--n-benign", type=int, default=100)
    parser.add_argument("--flag-as-block", action="store_true",
                        help="Count FLAG decisions as blocked for ABR.")
    parser.add_argument("--block-threshold", type=float, default=0.75)
    parser.add_argument("--report-out",  default=str(REPORT_PATH))
    parser.add_argument("--json-out",    default=str(JSON_PATH))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  PACM Session 7 -- Error Analysis")
    print(f"{'='*60}\n")

    if args.synthetic or not Path(args.input).exists():
        records = _make_synthetic_records(
            n_attack=args.n_attack, n_benign=args.n_benign,
        )
        print(f"  Generated {len(records)} synthetic audit records.")
    else:
        records = _load_from_jsonl(Path(args.input))
        print(f"  Loaded {len(records)} audit records from {args.input}.")

    if not records:
        print("  No records. Use --synthetic or run an experiment first.")
        sys.exit(0)

    from src.pacm.error_analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer(
        block_threshold=args.block_threshold,
        flag_as_block=args.flag_as_block,
    )
    report = analyzer.analyze(records)

    # Print summary
    n_fp = len(report.fp_instances)
    n_fn = len(report.fn_instances)
    print(f"  Total records:  {report.n_total}")
    print(f"  TP={report.n_tp}  TN={report.n_tn}  FP={n_fp}  FN={n_fn}")
    print(f"  ABR={report.abr*100:.1f}%  FPR={report.fpr*100:.1f}%  "
          f"Precision={report.precision*100:.1f}%")
    print()
    print("  FP breakdown:", {k: v for k, v in report.fp_counts.items() if v > 0})
    print("  FN breakdown:", {k: v for k, v in report.fn_counts.items() if v > 0})

    # Save Markdown report
    md = report.to_markdown()
    Path(args.report_out).write_text(md, encoding="utf-8")
    print(f"\n  Markdown report -> {args.report_out}")

    # Save JSON
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"  JSON report     -> {args.json_out}")
