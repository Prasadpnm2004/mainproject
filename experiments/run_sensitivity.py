"""
experiments/run_sensitivity.py

PACM Alpha Sensitivity Sweep CLI — Session 6

Sweeps alpha in {0.5, 0.6, 0.7, 0.8, 0.9} over labeled task records and
outputs an ASCII sensitivity table plus experiments/alpha_sensitivity.json.

Usage:
    # Synthetic data (no API key / real experiment needed):
    python -m experiments.run_sensitivity --synthetic

    # From real pacm_results.jsonl (with PACM audit entries):
    python -m experiments.run_sensitivity --input experiments/pacm_results.jsonl

    # Custom alpha grid:
    python -m experiments.run_sensitivity --synthetic --alphas 0.5 0.6 0.7 0.8 0.9

    # Custom decision threshold:
    python -m experiments.run_sensitivity --synthetic --threshold 0.55

References:
    - PACM_Research_Guide.md §9.1 (scoring formula)
    - src/pacm/sensitivity.py (AlphaSensitivityAnalyzer)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

OUTPUT_PATH = _repo_root / "experiments" / "alpha_sensitivity.json"


def _load_records_from_jsonl(path: Path, run_id: str | None = None) -> list[dict]:
    """
    Load records from a JSONL file, expanding pacm_audit entries.
    Each audit entry contains divergence_score, trust_score, decision, etc.
    """
    records = []
    if not path.exists():
        print(f"  [WARN] {path} not found — using empty record set.")
        return []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if run_id and rec.get("run_id") != run_id:
                continue

            # Expand pacm_audit call-level records
            audit = rec.get("pacm_audit") or []
            for entry in audit:
                if "divergence_score" in entry and "trust_score" in entry:
                    records.append({
                        "divergence_score": float(entry["divergence_score"]),
                        "d_semantic":       float(entry.get("d_semantic", entry["divergence_score"])),
                        "p_scope":          float(entry.get("p_scope", 0.0)),
                        "trust_score":      float(entry.get("trust_score", 0.5)),
                        "friction_score":   float(entry.get("friction_score", 0.0)),
                        "is_attack":        bool(rec.get("is_attack", False)),
                        "tool_name":        entry.get("tool_name", ""),
                        "run_id":           rec.get("run_id", ""),
                        "arg_scope_flags":  entry.get("scope_flags", {}),
                    })

            # If no audit, use record-level data as fallback
            if not audit and "divergence_score" in rec and "is_attack" in rec:
                records.append({
                    "divergence_score": float(rec["divergence_score"]),
                    "is_attack":        bool(rec["is_attack"]),
                    "trust_score":      0.5,
                    "run_id":           rec.get("run_id", ""),
                })

    return records


def print_sensitivity_table(result_dict: dict) -> None:
    """Print an ASCII sensitivity table from a SensitivityResult dict."""
    rows = result_dict.get("alpha_results", [])
    best_alpha = result_dict.get("best_alpha")
    backend = result_dict.get("backend", "?")
    threshold = result_dict.get("threshold", 0.50)

    print(f"\n{'-'*76}")
    print(f"  PACM Alpha Sensitivity Analysis")
    print(f"  backend={backend}  threshold={threshold:.2f}  "
          f"n_records={result_dict.get('n_records', 0)}")
    print(f"{'-'*76}")
    print(f"  {'alpha':>5}  {'D_attack':>8}  {'D_benign':>8}  "
          f"{'Sep':>6}  {'ABR':>6}  {'FPR':>6}  *")
    print(f"  {'-'*68}")
    for r in rows:
        star = " <-- BEST" if abs(r["alpha"] - best_alpha) < 1e-6 else ""
        print(
            f"  {r['alpha']:>5.1f}  "
            f"{r['mean_d_attack']:>8.4f}  "
            f"{r['mean_d_benign']:>8.4f}  "
            f"{r['separation']:>6.4f}  "
            f"{r['abr_at_threshold']*100:>5.1f}%  "
            f"{r['fpr_at_threshold']*100:>5.1f}%"
            f"{star}"
        )
    print(f"{'-'*76}")
    if best_alpha is not None:
        best_row = next((r for r in rows if abs(r["alpha"] - best_alpha) < 1e-6), None)
        if best_row:
            print(f"\n  Best alpha: {best_alpha:.1f}")
            print(f"    Separation = {best_row['separation']:.4f}")
            print(f"    ABR = {best_row['abr_at_threshold']*100:.1f}%  "
                  f"FPR = {best_row['fpr_at_threshold']*100:.1f}%\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PACM Session 6 -- Alpha sensitivity sweep runner."
    )
    parser.add_argument(
        "--input",
        default=str(_repo_root / "experiments" / "pacm_results.jsonl"),
        help="Path to JSONL results file.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Filter to specific run_id (default: all records).",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Alpha values to sweep (default: 0.5 0.6 0.7 0.8 0.9).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Decision threshold for ABR/FPR (default: 0.50).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic records (no real experiment needed).",
    )
    parser.add_argument(
        "--n-attack", type=int, default=100,
        help="Synthetic attack records (default: 100).",
    )
    parser.add_argument(
        "--n-benign", type=int, default=100,
        help="Synthetic benign records (default: 100).",
    )
    parser.add_argument(
        "--use-transformers", action="store_true",
        help="Use sentence-transformers backend (default: TF-IDF).",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help="Output JSON path (default: experiments/alpha_sensitivity.json).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print full JSON result to stdout.",
    )
    args = parser.parse_args()

    from src.pacm.sensitivity import (
        AlphaSensitivityAnalyzer,
        synthetic_records_for_sensitivity,
    )

    if args.synthetic:
        records = synthetic_records_for_sensitivity(
            n_attack=args.n_attack, n_benign=args.n_benign, seed=42,
        )
        print(f"  Using {len(records)} synthetic records "
              f"({args.n_attack} attack, {args.n_benign} benign).")
    else:
        records = _load_records_from_jsonl(Path(args.input), args.run_id)
        print(f"  Loaded {len(records)} records from {args.input}.")

    if not records:
        print("  No records found. Use --synthetic or run an experiment first.")
        sys.exit(0)

    analyzer = AlphaSensitivityAnalyzer(
        alpha_grid=sorted(set(args.alphas)),
        threshold=args.threshold,
        prefer_transformers=args.use_transformers,
    )
    result = analyzer.run(records)
    result_dict = result.to_dict()

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2)
    print(f"  Sensitivity result saved -> {out_path}")

    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print_sensitivity_table(result_dict)
