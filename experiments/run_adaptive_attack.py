"""
experiments/run_adaptive_attack.py

PACM Adaptive Attacker Stress Test CLI — Session 7 (RQ5)

Runs AdaptiveAttacker across all 5 attack variant levels and saves
results to experiments/adaptive_attack_results.json.

Usage:
    # Default: TF-IDF scorer, STANDARD_POLICY, 20 instances/variant:
    python -m experiments.run_adaptive_attack

    # Aggressive policy:
    python -m experiments.run_adaptive_attack --policy aggressive

    # All policies (comparative sweep):
    python -m experiments.run_adaptive_attack --all-policies

    # Use sentence-transformers backend:
    python -m experiments.run_adaptive_attack --use-transformers

    # Custom trust score and instance count:
    python -m experiments.run_adaptive_attack --trust 0.5 --instances 50

    # Specific variants only:
    python -m experiments.run_adaptive_attack --variants non_adaptive full_adaptive

References:
    - src/pacm/adaptive_attack.py
    - PACM_Research_Guide.md §6 (Threat Model)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

OUTPUT_PATH = _repo_root / "experiments" / "adaptive_attack_results.json"

_POLICY_MAP = {
    "standard":     "STANDARD_POLICY",
    "conservative": "CONSERVATIVE_POLICY",
    "aggressive":   "AGGRESSIVE_POLICY",
}


def _load_policy(policy_key: str):
    from src.pacm.policy import STANDARD_POLICY, CONSERVATIVE_POLICY, AGGRESSIVE_POLICY
    return {
        "standard":     STANDARD_POLICY,
        "conservative": CONSERVATIVE_POLICY,
        "aggressive":   AGGRESSIVE_POLICY,
    }[policy_key]


def print_attack_summary(summary_dict: dict, policy_label: str = "") -> None:
    """Print a formatted attack summary table to stdout."""
    stats = summary_dict.get("variant_stats", {})
    label = policy_label or summary_dict.get("policy_label", "?")

    print(f"\n{'-'*74}")
    print(f"  PACM Adaptive Attack Stress Test (RQ5)  |  policy={label}")
    print(f"  backend={summary_dict.get('backend', '?')}  "
          f"n_total={summary_dict.get('n_total', 0)}")
    print(f"{'-'*74}")
    print(f"  {'Variant':<22}  {'n':>4}  {'Blocked':>7}  {'Evaded':>7}  "
          f"{'EvasRate':>8}  {'ABR':>6}  {'mean_D':>6}  {'mean_F':>6}")
    print(f"  {'-'*70}")

    for variant, vs in stats.items():
        evade_str = f"{vs['evasion_rate']*100:.0f}%"
        abr_str   = f"{vs['abr']*100:.0f}%"
        print(
            f"  {variant:<22}  {vs['n_total']:>4}  "
            f"{vs['n_blocked']:>7}  {vs['n_evaded']:>7}  "
            f"{evade_str:>8}  {abr_str:>6}  "
            f"{vs['mean_D']:>6.3f}  {vs['mean_F']:>6.3f}"
        )

    print(f"{'-'*74}")
    print(f"  Overall: ABR={summary_dict.get('overall_abr', 0)*100:.1f}%  "
          f"Evasion={summary_dict.get('overall_evasion', 0)*100:.1f}%")
    print()


def run_adaptive_stress_test(
    policy_key: str = "standard",
    trust_score: float = 0.8,
    n_instances: int = 20,
    prefer_transformers: bool = False,
    variants: list[str] | None = None,
) -> dict:
    """
    Run the adaptive attacker and return result dict.

    Returns:
        Dict with keys: summary, records, policy_label.
    """
    from src.pacm.adaptive_attack import AdaptiveAttacker, AttackVariant

    policy = _load_policy(policy_key)

    variant_objs = None
    if variants:
        variant_objs = [AttackVariant(v) for v in variants]

    attacker = AdaptiveAttacker(
        policy=policy,
        trust_score=trust_score,
        prefer_transformers=prefer_transformers,
        n_instances=n_instances,
    )

    records, summary = attacker.run(variant_objs)

    return {
        "policy_label": _POLICY_MAP.get(policy_key, policy_key),
        "trust_score":  trust_score,
        "n_instances":  n_instances,
        "backend":      summary.backend,
        "summary":      summary.to_dict(),
        "records":      [r.to_dict() for r in records],
    }


def run_all_policies(
    trust_score: float = 0.8,
    n_instances: int = 20,
    prefer_transformers: bool = False,
) -> dict:
    """Run adaptive stress test across all three policies and compare."""
    results_by_policy = {}
    for policy_key in _POLICY_MAP:
        print(f"  Running policy={policy_key} ...")
        result = run_adaptive_stress_test(
            policy_key=policy_key,
            trust_score=trust_score,
            n_instances=n_instances,
            prefer_transformers=prefer_transformers,
        )
        results_by_policy[policy_key] = result
        print_attack_summary(result["summary"], policy_label=policy_key)

    return results_by_policy


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PACM Session 7 -- Adaptive attacker stress test (RQ5)."
    )
    parser.add_argument(
        "--policy", default="standard", choices=list(_POLICY_MAP),
        help="Policy to use (default: standard).",
    )
    parser.add_argument(
        "--all-policies", action="store_true",
        help="Run across all three policies and compare.",
    )
    parser.add_argument(
        "--trust", type=float, default=0.8,
        help="Tool trust score (default: 0.8).",
    )
    parser.add_argument(
        "--instances", type=int, default=20,
        help="Attack instances per variant (default: 20).",
    )
    parser.add_argument(
        "--use-transformers", action="store_true",
        help="Use sentence-transformers embedding backend.",
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=[v.value for v in __import__(
            "src.pacm.adaptive_attack", fromlist=["AttackVariant"]
        ).AttackVariant],
        default=None,
        help="Specific variants to test (default: all 5).",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  PACM Session 7 -- Adaptive Attacker Stress Test (RQ5)")
    print(f"{'='*60}\n")

    if args.all_policies:
        result_data = run_all_policies(
            trust_score=args.trust,
            n_instances=args.instances,
            prefer_transformers=args.use_transformers,
        )
    else:
        result_data = run_adaptive_stress_test(
            policy_key=args.policy,
            trust_score=args.trust,
            n_instances=args.instances,
            prefer_transformers=args.use_transformers,
            variants=args.variants,
        )
        print_attack_summary(result_data["summary"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result_data, fh, indent=2, default=str)
    print(f"  Results saved -> {out_path}")
