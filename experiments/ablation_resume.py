"""
experiments/ablation_resume.py
Resume scope_reasoning from run_idx=5 (runs 0-4 already in s24_ablation.jsonl).
Then run scope_only (10 runs) and naive_baseline (10 runs).
Appends to s24_ablation.jsonl.
"""
from __future__ import annotations
import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import the ablation module (all helpers are defined there)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ablation_sweep",
    os.path.join(_PROJECT_ROOT, "experiments", "ablation_sweep.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-expose helpers
run_episode     = _mod.run_episode
flush           = _mod.flush
reset_hook      = _mod.reset_hook
compute_stats   = _mod.compute_stats
print_stats_table = _mod.print_stats_table
print_fisher_test = _mod.print_fisher_test
fmt_ci          = _mod.fmt_ci
build_pacm_middleware = _mod.build_pacm_middleware
build_native_llm      = _mod.build_native_llm
build_scope_only_llm  = _mod.build_scope_only_llm
build_naive_detector  = _mod.build_naive_detector
_N = _mod._N_PER_CONDITION  # 10

import json, requests

_OUT = _mod._OUT_JSONL

def load_existing() -> list[dict]:
    records = []
    if os.path.exists(_OUT):
        with open(_OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def count_completed(records, condition):
    return [r for r in records if r.get("condition") == condition and "error" not in r]


def run_condition_from(condition: str, start_idx: int) -> list[dict]:
    records = []
    for i in range(start_idx, _N):
        try:
            if condition == "scope_only":
                llm = build_scope_only_llm()
                hook = build_pacm_middleware(f"s24_so_{i:02d}")
            elif condition == "scope_reasoning":
                llm = build_native_llm()
                hook = build_pacm_middleware(f"s24_sr_{i:02d}")
            elif condition == "naive_baseline":
                llm = build_native_llm()
                hook = build_naive_detector()
            else:
                raise ValueError(f"Unknown condition: {condition}")

            rec = run_episode(condition, i, llm, hook)
            records.append(rec)
            flush(rec)
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Stopped at run {i} of {condition}")
            break
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Run {i} of {condition}: {exc}")
            traceback.print_exc()
            from datetime import datetime, timezone
            err = {"condition": condition, "run_idx": i,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "error": str(exc), "overthinking_trap": None}
            records.append(err)
            flush(err)
    return records


def main():
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/version", timeout=5)
        print(f"Ollama: {r.json().get('version','?')}")
    except Exception:
        print("ERROR: Ollama not reachable."); sys.exit(1)

    existing = load_existing()
    print(f"Existing records in JSONL: {len(existing)}")

    sr_done = count_completed(existing, "scope_reasoning")
    so_done = count_completed(existing, "scope_only")
    nb_done = count_completed(existing, "naive_baseline")
    print(f"  scope_reasoning: {len(sr_done)}/10 complete")
    print(f"  scope_only:      {len(so_done)}/10 complete")
    print(f"  naive_baseline:  {len(nb_done)}/10 complete")

    all_new = []

    # Resume scope_reasoning if needed
    if len(sr_done) < _N:
        print(f"\nResuming scope_reasoning from run {len(sr_done)}...")
        new = run_condition_from("scope_reasoning", len(sr_done))
        all_new.extend(new)

    # Run scope_only if not done
    if len(so_done) < _N:
        print(f"\nStarting scope_only from run {len(so_done)}...")
        new = run_condition_from("scope_only", len(so_done))
        all_new.extend(new)

    # Run naive_baseline if not done
    if len(nb_done) < _N:
        print(f"\nStarting naive_baseline from run {len(nb_done)}...")
        new = run_condition_from("naive_baseline", len(nb_done))
        all_new.extend(new)

    # Final stats
    all_records = existing + all_new
    valid = [r for r in all_records if "error" not in r]
    completed = list(dict.fromkeys(r["condition"] for r in valid))
    stats_list = [compute_stats(valid, c) for c in completed]
    print_stats_table(stats_list)

    so_s = next((s for s in stats_list if s["condition"] == "scope_only"), None)
    sr_s = next((s for s in stats_list if s["condition"] == "scope_reasoning"), None)
    if so_s and sr_s:
        print_fisher_test(so_s, sr_s)
    else:
        print("\n(Fisher's test skipped — one/both conditions incomplete)")

    # Write stats JSON
    import json as _json
    stats_path = os.path.join(_PROJECT_ROOT, "experiments", "s24_ablation_stats.json")
    with open(stats_path, "w") as f:
        _json.dump({"stats": stats_list, "conditions_completed": completed}, f, indent=2)
    print(f"\nStats written: {stats_path}")


if __name__ == "__main__":
    main()
