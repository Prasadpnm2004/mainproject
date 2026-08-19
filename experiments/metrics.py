"""
experiments/metrics.py

PACM Metrics Processing Pipeline — Session 4

Parses JSONL execution logs produced by run_experiment.py and computes:
  - ABR  (Attack Block Rate):        % of injection attacks blocked by PACM.
  - BTSR (Benign Task Success Rate): % of benign tasks completed successfully.
  - FPR  (False Positive Rate):      % of benign tool calls incorrectly blocked.
  - Latency overhead:                mean / p95 wall_time_s for defended vs. undefended.

Usage (library):
    from experiments.metrics import MetricsProcessor
    mp = MetricsProcessor.from_jsonl("experiments/pacm_results.jsonl")
    report = mp.compute(run_id="abc12345")
    print(report.summary_table())

Usage (CLI):
    python -m experiments.metrics --input experiments/pacm_results.jsonl
    python -m experiments.metrics --input experiments/pacm_results.jsonl --run-id abc12345

References:
    - PACM_Research_Guide.md §10 (evaluation metrics)
    - AgentDojo evaluation: Debenedetti et al. (2024), arXiv:2406.13352
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))


# ─── Data models ──────────────────────────────────────────────────────────────


@dataclass
class MetricsReport:
    """
    Computed metrics for one experiment run (or a filtered group of records).

    Attributes:
        run_id:      Experiment run identifier (or "all" for multi-run aggregate).
        mode:        "undefended" | "defended" | "mixed".
        n_total:     Total records in scope.
        n_skipped:   Records with skipped=True (no API key).
        n_benign:    Non-attack task records.
        n_attack:    Attack task records.
        n_blocked:   Attack records where attack_blocked=True.
        n_succeeded: Benign records where task_success=True.
        n_fp:        Benign tool calls incorrectly blocked (from PACM audit).
        n_benign_calls: Total benign tool calls evaluated by PACM (from audit).
        wall_times:  List of wall_time_s for executed (non-skipped) records.
    """

    run_id: str
    mode: str
    n_total: int
    n_skipped: int
    n_benign: int
    n_attack: int
    n_blocked: int
    n_succeeded: int
    n_fp: int
    n_benign_calls: int
    wall_times: list[float] = field(default_factory=list)

    # ── Derived metrics ────────────────────────────────────────────────────────

    @property
    def abr(self) -> Optional[float]:
        """
        Attack Block Rate = n_blocked / n_attack.
        None if no attack tasks were run.
        """
        return self.n_blocked / self.n_attack if self.n_attack > 0 else None

    @property
    def btsr(self) -> Optional[float]:
        """
        Benign Task Success Rate = n_succeeded / n_benign.
        None if no benign tasks were run.
        """
        return self.n_succeeded / self.n_benign if self.n_benign > 0 else None

    @property
    def fpr(self) -> Optional[float]:
        """
        False Positive Rate = n_fp / n_benign_calls.
        None if no benign tool calls were evaluated by PACM.
        """
        return self.n_fp / self.n_benign_calls if self.n_benign_calls > 0 else None

    @property
    def mean_latency_s(self) -> Optional[float]:
        """Mean wall time across executed tasks."""
        if not self.wall_times:
            return None
        return sum(self.wall_times) / len(self.wall_times)

    @property
    def p95_latency_s(self) -> Optional[float]:
        """95th percentile wall time across executed tasks."""
        if not self.wall_times:
            return None
        sorted_times = sorted(self.wall_times)
        idx = max(0, math.ceil(0.95 * len(sorted_times)) - 1)
        return sorted_times[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "n_total": self.n_total,
            "n_skipped": self.n_skipped,
            "n_benign": self.n_benign,
            "n_attack": self.n_attack,
            "n_blocked": self.n_blocked,
            "n_succeeded": self.n_succeeded,
            "n_fp": self.n_fp,
            "n_benign_calls": self.n_benign_calls,
            "abr": round(self.abr, 4) if self.abr is not None else None,
            "btsr": round(self.btsr, 4) if self.btsr is not None else None,
            "fpr": round(self.fpr, 4) if self.fpr is not None else None,
            "mean_latency_s": round(self.mean_latency_s, 4) if self.mean_latency_s else None,
            "p95_latency_s": round(self.p95_latency_s, 4) if self.p95_latency_s else None,
        }

    def summary_table(self) -> str:
        """Return a human-readable summary table string."""
        def _pct(v: Optional[float]) -> str:
            return f"{v*100:.1f}%" if v is not None else "N/A"

        def _lat(v: Optional[float]) -> str:
            return f"{v:.4f}s" if v is not None else "N/A"

        lines = [
            f"{'-'*56}",
            f"  PACM Metrics Report -- run_id={self.run_id}",
            f"  mode={self.mode}",
            f"{'-'*56}",
            f"  Records:  total={self.n_total}  skipped={self.n_skipped}",
            f"  Benign:   n={self.n_benign}  succeeded={self.n_succeeded}",
            f"  Attack:   n={self.n_attack}  blocked={self.n_blocked}",
            f"{'-'*56}",
            f"  ABR  (Attack Block Rate):        {_pct(self.abr)}",
            f"  BTSR (Benign Task Success Rate): {_pct(self.btsr)}",
            f"  FPR  (False Positive Rate):      {_pct(self.fpr)}",
            f"  Latency mean / p95:              {_lat(self.mean_latency_s)} / {_lat(self.p95_latency_s)}",
            f"{'-'*56}",
        ]
        return "\n".join(lines)


# ─── Metrics processor ────────────────────────────────────────────────────────


class MetricsProcessor:
    """
    Parses JSONL experiment logs and computes PACM evaluation metrics.

    Each record in the JSONL is expected to follow the schema produced by
    experiments/run_experiment.py:
      - run_id, mode, is_attack, task_success, attack_blocked, skipped
      - wall_time_s
      - pacm_audit (optional list of PACMAuditEntry.to_dict())

    Usage:
        mp = MetricsProcessor.from_jsonl("experiments/pacm_results.jsonl")
        report = mp.compute()          # all records
        report = mp.compute("abc123")  # specific run_id
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MetricsProcessor":
        """Load and parse a JSONL results file."""
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return cls(records)

    def compute(self, run_id: Optional[str] = None) -> MetricsReport:
        """
        Compute metrics for a specific run_id, or all records if None.

        Args:
            run_id: Filter to records matching this run_id. Pass None for all.

        Returns:
            MetricsReport with ABR, BTSR, FPR, and latency stats.
        """
        records = self._records
        if run_id is not None:
            records = [r for r in records if r.get("run_id") == run_id]

        if not records:
            return MetricsReport(
                run_id=run_id or "all",
                mode="unknown",
                n_total=0, n_skipped=0, n_benign=0, n_attack=0,
                n_blocked=0, n_succeeded=0, n_fp=0, n_benign_calls=0,
            )

        # Determine dominant mode label
        modes = [r.get("mode", "unknown") for r in records]
        mode_label = modes[0] if len(set(modes)) == 1 else "mixed"

        n_total = len(records)
        n_skipped = sum(1 for r in records if r.get("skipped"))
        executed = [r for r in records if not r.get("skipped")]

        benign_records = [r for r in executed if not r.get("is_attack")]
        attack_records = [r for r in executed if r.get("is_attack")]

        n_benign = len(benign_records)
        n_attack = len(attack_records)
        n_blocked = sum(1 for r in attack_records if r.get("attack_blocked") is True)
        n_succeeded = sum(1 for r in benign_records if r.get("task_success") is True)

        # FPR: count PACM audit entries for benign tasks where decision=="block"
        n_fp, n_benign_calls = self._count_false_positives(benign_records)

        wall_times = [r["wall_time_s"] for r in executed if r.get("wall_time_s", 0) > 0]

        return MetricsReport(
            run_id=run_id or "all",
            mode=mode_label,
            n_total=n_total,
            n_skipped=n_skipped,
            n_benign=n_benign,
            n_attack=n_attack,
            n_blocked=n_blocked,
            n_succeeded=n_succeeded,
            n_fp=n_fp,
            n_benign_calls=n_benign_calls,
            wall_times=wall_times,
        )

    def list_run_ids(self) -> list[str]:
        """Return sorted unique run_ids present in the loaded records."""
        seen = set()
        result = []
        for r in self._records:
            rid = r.get("run_id", "")
            if rid and rid not in seen:
                seen.add(rid)
                result.append(rid)
        return sorted(result)

    def compute_all_runs(self) -> list[MetricsReport]:
        """Compute per-run metrics for every unique run_id."""
        return [self.compute(rid) for rid in self.list_run_ids()]

    def compare(
        self,
        undefended_run_id: str,
        defended_run_id: str,
    ) -> dict[str, Any]:
        """
        Compare defended vs. undefended runs for the paper's main result table.

        Returns a dict suitable for pretty-printing or JSON serialisation.
        """
        u = self.compute(undefended_run_id)
        d = self.compute(defended_run_id)

        def _delta(defended: Optional[float], undefended: Optional[float]) -> Optional[float]:
            if defended is None or undefended is None:
                return None
            return round(defended - undefended, 4)

        return {
            "undefended_run_id": undefended_run_id,
            "defended_run_id": defended_run_id,
            "undefended": u.to_dict(),
            "defended": d.to_dict(),
            "delta": {
                "abr":  _delta(d.abr,  u.abr),
                "btsr": _delta(d.btsr, u.btsr),
                "fpr":  _delta(d.fpr,  u.fpr),
                "mean_latency_s": _delta(d.mean_latency_s, u.mean_latency_s),
            },
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _count_false_positives(
        benign_records: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Count PACM false positives from the pacm_audit field of benign records.

        Returns:
            (n_fp, n_benign_calls) where n_fp = blocked benign tool calls.
        """
        n_fp = 0
        n_benign_calls = 0
        for record in benign_records:
            audit = record.get("pacm_audit") or []
            for entry in audit:
                n_benign_calls += 1
                if entry.get("decision") == "block":
                    n_fp += 1
        return n_fp, n_benign_calls


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PACM Session 4 — Metrics processor for experiment JSONL logs."
    )
    parser.add_argument(
        "--input",
        default=str(_repo_root / "experiments" / "pacm_results.jsonl"),
        help="Path to JSONL results file.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Filter to a specific run_id (default: compute over all records).",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("UNDEFENDED_ID", "DEFENDED_ID"),
        help="Compare two run_ids (undefended vs. defended).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable table.",
    )
    args = parser.parse_args()

    mp = MetricsProcessor.from_jsonl(args.input)

    if args.compare:
        result = mp.compare(args.compare[0], args.compare[1])
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            u = MetricsReport(**{k: v for k, v in result["undefended"].items()
                                  if k in MetricsReport.__dataclass_fields__})
            d = MetricsReport(**{k: v for k, v in result["defended"].items()
                                  if k in MetricsReport.__dataclass_fields__})
            print(u.summary_table())
            print(d.summary_table())
            print("\nDelta (defended − undefended):")
            for k, v in result["delta"].items():
                print(f"  {k}: {v}")
    else:
        report = mp.compute(args.run_id)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.summary_table())
