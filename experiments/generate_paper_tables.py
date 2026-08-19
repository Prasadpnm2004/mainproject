"""
experiments/generate_paper_tables.py

PACM Paper Benchmark Table Generator — Session 5

Parses sweep_summary.json (produced by run_sweep.py) and generates:
  1. A Markdown table suitable for direct inclusion in a research paper.
  2. A LaTeX table (booktabs style) for PDF compilation.

Both tables summarise ABR, BTSR, FPR, mean latency, and token cost
across all baseline and defended agent configurations.

Table schema (one row per run configuration):
  Mode | Harness | Attack | Policy | ABR | BTSR | FPR | Latency(ms) | Cost(est.)

Usage:
    # Generate from default sweep_summary.json:
    python -m experiments.generate_paper_tables

    # Specify input and output:
    python -m experiments.generate_paper_tables \\
        --input experiments/sweep_summary.json \\
        --output experiments/paper_tables

    # Print to stdout only (no files written):
    python -m experiments.generate_paper_tables --stdout

References:
    - PACM_Research_Guide.md §10, §12
    - experiments/run_sweep.py for sweep output format
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

DEFAULT_SWEEP_PATH  = _repo_root / "experiments" / "sweep_summary.json"
DEFAULT_OUTPUT_DIR  = _repo_root / "experiments"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v*100:.{decimals}f}"

def _lat_ms(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v*1000:.1f}"

def _cost_est(n_tasks: int, wall_s: float) -> str:
    """
    Rough token cost estimate based on stub run characteristics.
    For real LLM runs: replace with actual token counts from API responses.
    """
    # ~500 tokens/task at $0.15/1M for gpt-4o-mini = $0.000075/task
    cost = n_tasks * 0.000075
    return f"${cost:.4f}"

def _policy_display(run: dict) -> str:
    """Return a clean policy label for a run config."""
    if run.get("mode") == "undefended":
        return "—"
    label = run.get("policy_label", run.get("policy_key", ""))
    if label:
        # Shorten to just the policy tier
        return label.replace("_POLICY", "").capitalize()
    return run.get("policy_mode", "—")

def _attack_display(attack: Optional[str]) -> str:
    mapping = {
        "ignore_previous": "IgnorePrev.",
        "injec_agent": "InjecAgent",
        None: "All",
    }
    return mapping.get(attack, str(attack))


# ─── Row extraction ───────────────────────────────────────────────────────────

def extract_rows(sweep_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert sweep run summaries into flat table rows.

    Returns list of row dicts with normalised column values.
    """
    rows = []
    for run in sweep_summary.get("runs", []):
        m = run.get("metrics", {})
        n_tasks = run.get("n_tasks", 0)
        wall_s  = run.get("elapsed_s", 0.0)
        latency_mean_s = m.get("mean_latency_s")
        latency_ms_val = (latency_mean_s * 1000) if latency_mean_s else None

        rows.append({
            "mode":        run.get("mode", "?"),
            "harness":     run.get("harness", "?").replace("_", "-"),
            "attack":      _attack_display(run.get("attack")),
            "policy":      _policy_display(run),
            "abr":         m.get("abr"),
            "btsr":        m.get("btsr"),
            "fpr":         m.get("fpr"),
            "latency_ms":  latency_ms_val,
            "cost_est":    _cost_est(n_tasks, wall_s),
            "n_tasks":     n_tasks,
            "run_id":      run.get("run_id", ""),
            "_label":      run.get("label", ""),
        })

    # Sort: undefended first, then by mode/harness/attack/policy
    rows.sort(key=lambda r: (
        0 if r["mode"] == "undefended" else 1,
        r["harness"],
        r["attack"],
        r["policy"],
    ))
    return rows


# ─── Markdown table ───────────────────────────────────────────────────────────

def build_markdown_table(rows: list[dict], caption: str = "") -> str:
    """
    Build a Markdown table from row dicts.

    Format:
    | Mode | Harness | Attack | Policy | ABR(%) | BTSR(%) | FPR(%) | Lat.(ms) | Cost |
    """
    header = (
        "| Mode | Harness | Attack | Policy | ABR (%) | BTSR (%) | FPR (%) | "
        "Lat. (ms) | Cost (est.) |"
    )
    divider = (
        "|---|---|---|---|---:|---:|---:|---:|---:|"
    )
    lines = []
    if caption:
        lines.append(f"**{caption}**\n")
    lines.append(header)
    lines.append(divider)

    for r in rows:
        row = (
            f"| {r['mode'].capitalize()} "
            f"| {r['harness']} "
            f"| {r['attack']} "
            f"| {r['policy']} "
            f"| {_pct(r['abr'])} "
            f"| {_pct(r['btsr'])} "
            f"| {_pct(r['fpr'])} "
            f"| {_lat_ms(r['latency_ms'])} "
            f"| {r['cost_est']} |"
        )
        lines.append(row)

    return "\n".join(lines)


# ─── LaTeX table ──────────────────────────────────────────────────────────────

LATEX_HEADER = r"""\begin{table}[htbp]
  \centering
  \small
  \caption{PACM Defended vs.\ Undefended Agent Benchmark Results}
  \label{tab:pacm_results}
  \begin{tabular}{llllrrrrr}
    \toprule
    Mode & Harness & Attack & Policy & ABR (\%) & BTSR (\%) & FPR (\%) & Lat.\ (ms) & Cost \\
    \midrule"""

LATEX_FOOTER = r"""    \bottomrule
  \end{tabular}
  \begin{tablenotes}
    \small
    \item ABR: Attack Block Rate. BTSR: Benign Task Success Rate. FPR: False Positive Rate.
    \item Latency: mean wall-clock time per task. Cost: estimated token cost (gpt-4o-mini).
    \item Stub-mode runs; real LLM results pending API key configuration.
  \end{tablenotes}
\end{table}"""


def _tex_escape(s: str) -> str:
    """Escape LaTeX special characters."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def build_latex_table(rows: list[dict]) -> str:
    lines = [LATEX_HEADER]
    prev_mode = None

    for r in rows:
        if prev_mode and r["mode"] != prev_mode:
            lines.append(r"    \midrule")
        prev_mode = r["mode"]

        abr  = _pct(r["abr"])  + r"\%" if r["abr"]  is not None else "---"
        btsr = _pct(r["btsr"]) + r"\%" if r["btsr"] is not None else "---"
        fpr  = _pct(r["fpr"])  + r"\%" if r["fpr"]  is not None else "---"
        lat  = _lat_ms(r["latency_ms"]) if r["latency_ms"] else "---"
        cost = r["cost_est"]

        row = (
            f"    {_tex_escape(r['mode'].capitalize())} & "
            f"{_tex_escape(r['harness'])} & "
            f"{_tex_escape(r['attack'])} & "
            f"{_tex_escape(r['policy'])} & "
            f"{abr} & {btsr} & {fpr} & {lat} & {cost} \\\\"
        )
        lines.append(row)

    lines.append(LATEX_FOOTER)
    return "\n".join(lines)


# ─── Full report document ─────────────────────────────────────────────────────

def build_markdown_report(
    sweep_summary: dict[str, Any],
    rows: list[dict],
) -> str:
    """
    Build a full Markdown report document embedding both the main results
    table and a calibration summary section.
    """
    sweep_id   = sweep_summary.get("sweep_id", "?")
    n_configs  = sweep_summary.get("n_configs", 0)
    n_tasks    = sweep_summary.get("n_total_tasks", 0)
    params     = sweep_summary.get("parameters", {})
    ts         = sweep_summary.get("sweep_end_utc", "?")

    md_table = build_markdown_table(
        rows,
        caption="Table 1. PACM Defended vs. Undefended Agent Benchmark (stub mode)",
    )

    # Compute aggregate stats for summary
    defended_rows   = [r for r in rows if r["mode"] == "defended"]
    undefended_rows = [r for r in rows if r["mode"] == "undefended"]
    avg_abr  = _mean_pct([r["abr"]  for r in defended_rows if r["abr"]  is not None])
    avg_btsr = _mean_pct([r["btsr"] for r in defended_rows if r["btsr"] is not None])
    avg_fpr  = _mean_pct([r["fpr"]  for r in defended_rows if r["fpr"]  is not None])

    report_lines = [
        "# PACM Benchmark Report",
        "",
        f"> Generated: {ts}  |  Sweep ID: `{sweep_id}`  |"
        f"  Configs: {n_configs}  |  Tasks: {n_tasks}",
        "",
        "## Experiment Configuration",
        "",
        f"- **Harnesses**: {', '.join(params.get('harnesses', []))}",
        f"- **Attack types**: {', '.join(params.get('attacks', []))}",
        f"- **Modes**: {', '.join(params.get('modes', []))}",
        f"- **Policies**: {', '.join(params.get('policies', []))}",
        f"- **Model**: {params.get('model', 'N/A')} (stub mode — no API key required)",
        "",
        "## Main Results",
        "",
        md_table,
        "",
        "## Aggregate Summary (Defended Runs)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean ABR  | {avg_abr} |",
        f"| Mean BTSR | {avg_btsr} |",
        f"| Mean FPR  | {avg_fpr} |",
        "",
        "> **Note**: These results are from stub-mode runs (no real LLM).",
        "> Real divergence scores require `OPENAI_API_KEY` and will be",
        "> reported in Session 6 after threshold calibration.",
        "",
        "## Methodology",
        "",
        "Each task was evaluated with the PACM pre-tool hook using the",
        "friction formula **F = D × (1.5 − 0.5 × T)**, where:",
        "- **D** = plan-action divergence score (TF-IDF cosine in stub mode,",
        "  sentence-transformers in production)",
        "- **T** = MCP tool trust score from the static registry",
        "",
        "Decisions: F < allow_threshold → ALLOW; F ≥ block_threshold → BLOCK.",
        "",
        "### Policy Configurations",
        "",
        "| Policy | allow_thr | block_thr |",
        "|--------|-----------|-----------|",
        "| Aggressive | 0.30 | 0.55 |",
        "| Standard   | 0.45 | 0.75 |",
        "| Conservative | 0.45 | 0.85 |",
        "",
        "## References",
        "",
        "- Debenedetti et al. (2024). *AgentDojo*. arXiv:2406.13352",
        "- Li & Gao (2025). *MCP Security*. arXiv:2510.16558",
        "- Hasan et al. (2026). *MCP at First Glance*. arXiv:2506.13538",
    ]
    return "\n".join(report_lines)


def _mean_pct(values: list[float]) -> str:
    if not values:
        return "N/A"
    return f"{sum(values)/len(values)*100:.1f}%"


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PACM Session 5 -- Paper benchmark table generator."
    )
    parser.add_argument(
        "--input", default=str(DEFAULT_SWEEP_PATH),
        help="Path to sweep_summary.json.",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print tables to stdout instead of writing files.",
    )
    args = parser.parse_args()

    sweep_path = Path(args.input)
    if not sweep_path.exists():
        print(f"Error: {sweep_path} not found. Run experiments/run_sweep.py first.")
        sys.exit(1)

    with open(sweep_path, encoding="utf-8") as fh:
        sweep_summary = json.load(fh)

    rows = extract_rows(sweep_summary)
    if not rows:
        print("No sweep runs found in summary.")
        sys.exit(0)

    md_table  = build_markdown_table(rows, "Table 1. PACM Benchmark Results")
    latex_tbl = build_latex_table(rows)
    md_report = build_markdown_report(sweep_summary, rows)

    if args.stdout:
        print("\n=== MARKDOWN TABLE ===\n")
        print(md_table)
        print("\n=== LATEX TABLE ===\n")
        print(latex_tbl)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path    = out_dir / "paper_table.md"
        latex_path = out_dir / "paper_table.tex"
        report_path = out_dir / "benchmark_report.md"

        md_path.write_text(md_table, encoding="utf-8")
        latex_path.write_text(latex_tbl, encoding="utf-8")
        report_path.write_text(md_report, encoding="utf-8")

        print(f"  Markdown table  -> {md_path}")
        print(f"  LaTeX table     -> {latex_path}")
        print(f"  Full report     -> {report_path}")

        # Also print the markdown table to stdout
        print("\n" + md_table)
