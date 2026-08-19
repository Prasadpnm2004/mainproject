"""
experiments/calibrate.py

PACM Threshold Calibration Runner — Session 5

CLI entry point for the calibration pipeline. Reads scored records from
experiments/pacm_results.jsonl (or sweep_summary.json), runs the
ThresholdCalibrator grid search, and outputs:
  - experiments/calibration_result.json  (full CalibrationResult)
  - A human-readable ASCII ROC summary table

Usage:
    # Calibrate over all records in pacm_results.jsonl:
    python -m experiments.calibrate

    # Calibrate over a specific run_id:
    python -m experiments.calibrate --run-id sweep_defended_langgraph_001

    # Custom FPR target and grid step:
    python -m experiments.calibrate --target-fpr 0.05 --step 0.05

    # Use synthetic data (no real experiment required):
    python -m experiments.calibrate --synthetic --n-attack 100 --n-benign 100

    # Output JSON:
    python -m experiments.calibrate --json

References:
    - PACM_Research_Guide.md §10
    - src/pacm/calibration.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

CALIBRATION_OUTPUT = _repo_root / "experiments" / "calibration_result.json"
RESULTS_JSONL      = _repo_root / "experiments" / "pacm_results.jsonl"


def _load_records_from_jsonl(
    path: Path,
    run_id: str | None = None,
) -> list[dict]:
    """
    Load scored records from a JSONL file, optionally filtering by run_id.

    Each record must have:
      - divergence_score  (from PACMAuditEntry or experiment record)
      - trust_score
      - is_attack

    Records without these fields are skipped with a warning.
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

            # Two record shapes:
            # 1. Top-level experiment record (has is_attack + optional pacm_audit)
            # 2. PACMAuditEntry sub-record (has divergence_score, trust_score, decision)

            if "is_attack" in rec:
                # Expand pacm_audit entries as individual call-level records
                audit = rec.get("pacm_audit") or []
                if audit:
                    for entry in audit:
                        if "divergence_score" in entry and "trust_score" in entry:
                            records.append({
                                "divergence_score": entry["divergence_score"],
                                "trust_score": entry["trust_score"],
                                "is_attack": rec["is_attack"],
                                "tool_name": entry.get("tool_name", ""),
                                "run_id": rec.get("run_id", ""),
                                "_from_audit": True,
                            })
                else:
                    # No audit — use top-level friction approximation if available
                    pass  # task-level records without audit are skipped for calibration
            elif "divergence_score" in rec and "trust_score" in rec:
                records.append(rec)

    return records


def _extract_records_from_sweep(
    sweep_path: Path,
    run_id: str | None = None,
) -> list[dict]:
    """
    Extract calibration records from sweep_summary.json metrics data.
    This gives per-config aggregate ThresholdMetrics, which can be used
    to reconstruct a simplified ROC curve without raw per-call data.
    """
    if not sweep_path.exists():
        return []

    with open(sweep_path, encoding="utf-8") as fh:
        sweep = json.load(fh)

    synthetic_records = []
    for run in sweep.get("runs", []):
        if run_id and run.get("run_id") != run_id:
            continue
        m = run.get("metrics", {})
        # Reconstruct as synthetic records (one per task) from aggregate stats
        n_attack = m.get("n_attack", 0)
        n_blocked = m.get("n_blocked", 0)
        n_benign = m.get("n_benign", 0)

        threshold = run.get("divergence_threshold", 0.6)

        # Attack records: assign F at threshold boundary based on ABR
        for i in range(n_attack):
            blocked = i < n_blocked
            synthetic_records.append({
                "divergence_score": threshold + 0.1 if blocked else threshold - 0.1,
                "trust_score": 0.3,
                "is_attack": True,
                "run_id": run.get("run_id", ""),
            })
        for _ in range(n_benign):
            synthetic_records.append({
                "divergence_score": threshold * 0.5,
                "trust_score": 0.9,
                "is_attack": False,
                "run_id": run.get("run_id", ""),
            })

    return synthetic_records


def run_calibration(
    records: list[dict],
    target_fpr: float = 0.10,
    allow_step: float = 0.05,
    block_step: float = 0.05,
    allow_range: tuple[float, float] = (0.10, 0.85),
    block_range: tuple[float, float] = (0.20, 0.95),
) -> dict:
    """
    Run calibration on a list of scored records and return the result dict.

    Args:
        records:     List of dicts with divergence_score, trust_score, is_attack.
        target_fpr:  Maximum acceptable FPR.
        allow_step:  Grid step for allow_threshold.
        block_step:  Grid step for block_threshold.
        allow_range: (min, max) for allow_threshold grid.
        block_range: (min, max) for block_threshold grid.

    Returns:
        CalibrationResult as a dict.
    """
    from src.pacm.calibration import ThresholdCalibrator, _frange

    calibrator = ThresholdCalibrator(
        allow_grid=_frange(allow_range[0], allow_range[1], allow_step),
        block_grid=_frange(block_range[0], block_range[1], block_step),
        target_fpr=target_fpr,
    )
    result = calibrator.calibrate(records)
    return result.to_dict()


def print_roc_table(calibration_dict: dict) -> None:
    """Print a compact ASCII ROC frontier table."""
    frontier = calibration_dict.get("frontier", [])
    optimal  = calibration_dict.get("optimal")
    auc      = calibration_dict.get("auc_roc", 0.0)

    print(f"\n{'-'*72}")
    print("  PACM Threshold Calibration -- ROC Frontier")
    print(f"  target_fpr={calibration_dict.get('target_fpr')}  "
          f"grid_size={calibration_dict.get('grid_size')}  "
          f"AUC-ROC={auc:.4f}")
    print(f"{'-'*72}")
    print(f"  {'allow_thr':>9} {'block_thr':>9} {'ABR':>7} {'FPR':>7} {'F1':>7}  *")
    print(f"  {'-'*60}")
    for pt in frontier:
        star = " <-- OPTIMAL" if (
            optimal and
            abs(pt["allow_threshold"] - optimal["allow_threshold"]) < 1e-6 and
            abs(pt["block_threshold"] - optimal["block_threshold"]) < 1e-6
        ) else ""
        print(
            f"  {pt['allow_threshold']:>9.2f} {pt['block_threshold']:>9.2f}"
            f" {pt['abr']*100:>6.1f}% {pt['fpr']*100:>6.1f}%"
            f" {pt['f1']:>7.4f}{star}"
        )
    print(f"{'-'*72}")
    if optimal:
        print(f"\n  Recommended thresholds:")
        print(f"    allow_threshold = {optimal['allow_threshold']:.2f}")
        print(f"    block_threshold = {optimal['block_threshold']:.2f}")
        print(f"    ABR = {optimal['abr']*100:.1f}%  FPR = {optimal['fpr']*100:.1f}%")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PACM Session 5 -- Threshold calibration runner."
    )
    parser.add_argument(
        "--input", default=str(RESULTS_JSONL),
        help="Path to JSONL results file (default: experiments/pacm_results.jsonl).",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Filter to a specific run_id.",
    )
    parser.add_argument(
        "--target-fpr", type=float, default=0.10,
        help="Maximum acceptable FPR (default: 0.10).",
    )
    parser.add_argument(
        "--step", type=float, default=0.05,
        help="Grid step for both allow and block thresholds (default: 0.05).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of real records.",
    )
    parser.add_argument(
        "--n-attack", type=int, default=50,
        help="Synthetic attack records (default: 50).",
    )
    parser.add_argument(
        "--n-benign", type=int, default=50,
        help="Synthetic benign records (default: 50).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output full JSON result instead of table.",
    )
    parser.add_argument(
        "--output", default=str(CALIBRATION_OUTPUT),
        help="Path for JSON output.",
    )
    args = parser.parse_args()

    if args.synthetic:
        from src.pacm.calibration import synthetic_records_for_calibration
        records = synthetic_records_for_calibration(
            n_attack=args.n_attack, n_benign=args.n_benign,
        )
        print(f"  Using {len(records)} synthetic records "
              f"({args.n_attack} attack, {args.n_benign} benign).")
    else:
        records = _load_records_from_jsonl(Path(args.input), args.run_id)
        if not records:
            # Fallback: try extracting from sweep_summary
            sweep_path = _repo_root / "experiments" / "sweep_summary.json"
            records = _extract_records_from_sweep(sweep_path, args.run_id)
        print(f"  Loaded {len(records)} records for calibration.")

    if not records:
        print("  No records found. Run an experiment first or use --synthetic.")
        sys.exit(0)

    result = run_calibration(
        records=records,
        target_fpr=args.target_fpr,
        allow_step=args.step,
        block_step=args.step,
    )

    # Save JSON output
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Calibration result saved -> {args.output}")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_roc_table(result)
