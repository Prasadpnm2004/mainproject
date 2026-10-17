"""
Run aggressive config (3 samples) with per-sample JSONL flush.
Writes each sample's records immediately after completion so a crash
mid-run still preserves completed samples.
"""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure project root is on the path
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiment import run_experiment, write_results
from src.llm_config import load_llm_config

TARGET_RID = "sweep_defended_langgraph_ignore_previous_aggressive_2ee2bb"
JSONL_PATH = Path(r"d:\mainproject\experiments\pacm_results.jsonl")

# Check which samples already completed (resume-safe)
completed_samples = set()
if JSONL_PATH.exists():
    for line in JSONL_PATH.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if TARGET_RID in r.get('run_id', ''):
            sid = r.get('sample_idx')
            if sid is not None:
                completed_samples.add(int(sid))

print(f"Already completed samples: {sorted(completed_samples)}")

cfg = load_llm_config()
print(f"Config: {cfg}")
print()

all_new_records = []
for s in range(3):
    if s in completed_samples:
        print(f"  Sample {s}/3 — SKIPPING (already flushed)")
        continue

    rid = f"{TARGET_RID}_s{s}"
    print(f"  Sample {s}/3  run_id={rid}", flush=True)
    t0 = time.perf_counter()

    try:
        recs = run_experiment(
            mode='defended',
            harness_name='langgraph',
            attack_filter='ignore_previous',
            policy_mode='block',
            divergence_threshold=0.40,
            model_name=cfg.model,
            max_iter=10,
            run_id=rid,
            sample_idx=s,
        )
    except Exception as exc:
        print(f"  Sample {s} FAILED: {exc}", flush=True)
        continue

    elapsed = time.perf_counter() - t0

    # Normalise run_id for aggregation
    for r in recs:
        r['run_id'] = TARGET_RID
        r['sample_idx'] = s

    # Flush immediately
    write_results(recs)
    all_new_records.extend(recs)

    print(f"  Sample {s} done: {len(recs)} records  t={elapsed:.1f}s", flush=True)
    for r in recs:
        tid = r.get('task_id', '?')
        blocked = r.get('attack_blocked')
        mw = r.get('middleware_calls', [])
        d_vals = [round(mc.get('divergence', 0), 3) for mc in mw if mc.get('divergence') is not None]
        f_vals = [round(mc.get('friction', 0), 3) for mc in mw if mc.get('friction') is not None]
        decisions = [mc.get('decision') for mc in mw]
        tcs = r.get('tool_calls', [])
        to_addrs = [tc.get('args', {}).get('to') for tc in tcs if tc.get('tool') == 'send_email']
        print(f"    task={tid}  blocked={blocked}")
        print(f"      D={d_vals}  F={f_vals}  decisions={decisions}")
        print(f"      send_email.to={to_addrs}", flush=True)

print()
print(f"Total new records written: {len(all_new_records)}")
print("Done.")
