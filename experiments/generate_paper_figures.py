"""
experiments/generate_paper_figures.py

PACM Paper Figure Generator — Session 6

Generates two types of paper-ready figures from experiment data:

  (a) ABR vs. FPR Pareto Frontier (ROC curve) — from calibration_result.json
      - SVG vector figure for LaTeX inclusion
      - ASCII art version for terminal / log inspection

  (b) Latency overhead comparison — from sweep_summary.json
      - Bar-chart style ASCII figure comparing harness × policy configurations
      - SVG bar chart for LaTeX inclusion

Usage:
    # Generate all figures from default paths:
    python -m experiments.generate_paper_figures

    # Custom input files:
    python -m experiments.generate_paper_figures \\
        --calibration experiments/calibration_result.json \\
        --sweep       experiments/full_sweep_summary.json \\
        --output-dir  experiments/figures

    # ASCII-only (no SVG files):
    python -m experiments.generate_paper_figures --ascii-only

    # SVG-only (no terminal output):
    python -m experiments.generate_paper_figures --svg-only

Output files (in --output-dir):
    roc_curve.svg           — ABR vs. FPR Pareto frontier (SVG)
    latency_comparison.svg  — Latency bar chart (SVG)
    figure_data.json        — Raw data used for figures (for reproducibility)

References:
    - PACM_Research_Guide.md §10, §12
    - experiments/calibrate.py (produces calibration_result.json)
    - experiments/run_sweep.py (produces sweep_summary.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

DEFAULT_CALIBRATION = _repo_root / "experiments" / "calibration_result.json"
DEFAULT_SWEEP       = _repo_root / "experiments" / "sweep_summary.json"
DEFAULT_OUTPUT_DIR  = _repo_root / "experiments" / "figures"
DEFAULT_SENSITIVITY = _repo_root / "experiments" / "alpha_sensitivity.json"


# ─── SVG builder helpers ──────────────────────────────────────────────────────

def _svg_wrap(inner: str, width: int, height: int, title: str = "") -> str:
    """Wrap SVG content with standard header/footer."""
    title_el = f'  <title>{title}</title>\n' if title else ''
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="Arial, sans-serif">\n'
        f'{title_el}'
        f'{inner}'
        f'</svg>\n'
    )


def _svg_line(x1, y1, x2, y2, color="#333", width=1.5, dash="") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'  <line x1="{x1:.1f}" y1="{y1:.1f}" '
        f'x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>\n'
    )


def _svg_circle(cx, cy, r, color="#2563eb", stroke="#1d4ed8") -> str:
    return (
        f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
        f'fill="{color}" stroke="{stroke}" stroke-width="1.5"/>\n'
    )


def _svg_text(x, y, text, size=11, anchor="middle", color="#1e293b", bold=False) -> str:
    weight = ' font-weight="bold"' if bold else ''
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{color}"{weight}>'
        f'{text}</text>\n'
    )


def _svg_rect(x, y, w, h, fill="#3b82f6", stroke="#1d4ed8", rx=3) -> str:
    return (
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1" rx="{rx}"/>\n'
    )


# ─── Figure (a): ROC / Pareto Frontier ───────────────────────────────────────

PALETTE_ROC = {
    "point": "#2563eb",
    "optimal": "#dc2626",
    "line":  "#6b7280",
    "diag":  "#d1d5db",
    "grid":  "#f1f5f9",
    "axis":  "#374151",
    "bg":    "#ffffff",
}


def build_roc_svg(
    frontier: list[dict],
    optimal: Optional[dict] = None,
    width: int = 480,
    height: int = 400,
) -> str:
    """
    Generate an SVG Pareto frontier (ROC) plot.

    Args:
        frontier: List of {fpr, abr, allow_threshold, block_threshold} dicts.
        optimal:  The optimal operating point dict (highlighted in red).
        width:    SVG width in px.
        height:   SVG height in px.
    """
    pad_l, pad_r, pad_t, pad_b = 70, 30, 30, 60
    plot_w = width  - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def _px(fpr: float, abr: float):
        x = pad_l + fpr * plot_w
        y = pad_t + (1.0 - abr) * plot_h
        return x, y

    parts = []

    # Background
    parts.append(f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{PALETTE_ROC["bg"]}"/>\n')

    # Grid lines
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx, gy = _px(v, 0.0)
        _, gy2 = _px(v, 1.0)
        parts.append(_svg_line(gx, pad_t, gx, pad_t + plot_h, color=PALETTE_ROC["grid"], width=0.8))
        gx2, gy = _px(0.0, v)
        parts.append(_svg_line(pad_l, gy, pad_l + plot_w, gy, color=PALETTE_ROC["grid"], width=0.8))

    # Diagonal (random classifier)
    x0, y0 = _px(0.0, 0.0)
    x1, y1 = _px(1.0, 1.0)
    parts.append(_svg_line(x0, y0, x1, y1, color=PALETTE_ROC["diag"], width=1.0, dash="5,4"))

    # Frontier line (step-interpolated)
    sorted_pts = sorted(frontier, key=lambda p: p["fpr"])
    if len(sorted_pts) >= 2:
        path_d = ""
        for i, pt in enumerate(sorted_pts):
            px, py = _px(pt["fpr"], pt["abr"])
            if i == 0:
                path_d += f"M {px:.1f} {py:.1f}"
            else:
                prev_px, _ = _px(sorted_pts[i-1]["fpr"], sorted_pts[i-1]["abr"])
                path_d += f" L {px:.1f} {py:.1f}"
        parts.append(
            f'  <path d="{path_d}" fill="none" '
            f'stroke="{PALETTE_ROC["point"]}" stroke-width="2.0"/>\n'
        )

    # Data points
    for pt in sorted_pts:
        px, py = _px(pt["fpr"], pt["abr"])
        parts.append(_svg_circle(px, py, r=5))

    # Optimal point (highlighted)
    if optimal:
        opx, opy = _px(optimal["fpr"], optimal["abr"])
        parts.append(f'  <circle cx="{opx:.1f}" cy="{opy:.1f}" r="9" '
                     f'fill="{PALETTE_ROC["optimal"]}" stroke="#991b1b" stroke-width="2"/>\n')
        parts.append(_svg_text(opx + 14, opy - 8,
                               f'Optimal\n({optimal["abr"]*100:.0f}%, {optimal["fpr"]*100:.0f}%)',
                               size=9, anchor="start", color=PALETTE_ROC["optimal"]))

    # Axes
    ax_x1, ax_y1 = pad_l, pad_t + plot_h
    parts.append(_svg_line(ax_x1, ax_y1, ax_x1 + plot_w, ax_y1, color=PALETTE_ROC["axis"], width=1.5))
    parts.append(_svg_line(ax_x1, ax_y1, ax_x1, pad_t, color=PALETTE_ROC["axis"], width=1.5))

    # Axis labels & ticks
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tx, ty = _px(v, 0.0)
        parts.append(_svg_line(tx, ty, tx, ty + 5, color=PALETTE_ROC["axis"], width=1.0))
        parts.append(_svg_text(tx, ty + 18, f"{v:.0%}", size=9))
        lx, ly = _px(0.0, v)
        parts.append(_svg_line(lx, ly, lx - 5, ly, color=PALETTE_ROC["axis"], width=1.0))
        parts.append(_svg_text(lx - 8, ly + 4, f"{v:.0%}", size=9, anchor="end"))

    # Axis titles
    cx = pad_l + plot_w / 2
    parts.append(_svg_text(cx, height - 8, "False Positive Rate (FPR)", size=12, bold=True))
    # Rotated Y-axis label via transform
    cy = pad_t + plot_h / 2
    parts.append(
        f'  <text x="{14}" y="{cy:.1f}" font-size="12" '
        f'text-anchor="middle" fill="{PALETTE_ROC["axis"]}" font-weight="bold" '
        f'transform="rotate(-90, 14, {cy:.1f})">Attack Block Rate (ABR)</text>\n'
    )

    # Title
    parts.append(_svg_text(width / 2, 20, "PACM: ABR vs. FPR Pareto Frontier", size=13, bold=True))

    return _svg_wrap("".join(parts), width, height, "PACM ROC Curve")


def build_roc_ascii(frontier: list[dict], optimal: Optional[dict] = None,
                    cols: int = 60, rows: int = 20) -> str:
    """Generate a compact ASCII ROC plot."""
    grid = [[' '] * cols for _ in range(rows)]

    def _cell(fpr: float, abr: float):
        c = int(round(fpr * (cols - 1)))
        r = int(round((1.0 - abr) * (rows - 1)))
        return max(0, min(cols - 1, c)), max(0, min(rows - 1, r))

    # Plot frontier points
    sorted_pts = sorted(frontier, key=lambda p: p["fpr"])
    for pt in sorted_pts:
        c, r = _cell(pt["fpr"], pt["abr"])
        grid[r][c] = '*'

    # Optimal point
    if optimal:
        c, r = _cell(optimal["fpr"], optimal["abr"])
        grid[r][c] = 'O'

    # Border and axes
    lines = [f"  ABR^  PACM ROC Frontier ({len(frontier)} points)"]
    for r_idx, row in enumerate(grid):
        abr_val = 1.0 - r_idx / max(rows - 1, 1)
        prefix = f"  {abr_val:.0%}|" if r_idx % 5 == 0 else "      |"
        lines.append(prefix + ''.join(row))
    lines.append("      +" + '-' * cols)
    lines.append("       " + ''.join(
        f"{i/(cols-1):.0%}".center(cols // 5)
        for i in range(0, cols, cols // 5)
    ) + "  > FPR")
    if optimal:
        lines.append(f"\n  O = Optimal (ABR={optimal['abr']*100:.0f}%, "
                     f"FPR={optimal['fpr']*100:.0f}%)")
    return '\n'.join(lines)


# ─── Figure (b): Latency comparison ──────────────────────────────────────────

PALETTE_BAR = {
    "undefended": "#94a3b8",
    "standard":   "#3b82f6",
    "conservative": "#8b5cf6",
    "aggressive":  "#f59e0b",
    "axis":       "#374151",
    "bg":         "#ffffff",
    "grid":       "#f1f5f9",
}


def _bar_color(policy_key: Optional[str]) -> str:
    if not policy_key:
        return PALETTE_BAR["undefended"]
    return PALETTE_BAR.get(policy_key, "#6b7280")


def build_latency_svg(
    runs: list[dict],
    width: int = 560,
    height: int = 380,
) -> str:
    """
    Generate a grouped bar chart SVG for latency comparison.

    Args:
        runs: List of run dicts from sweep_summary.json with metrics.latency fields.
    """
    pad_l, pad_r, pad_t, pad_b = 80, 30, 40, 80
    plot_w = width  - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # Extract latency values
    bars = []
    for r in runs:
        m = r.get("metrics", {})
        lat_ms = (m.get("mean_latency_s") or 0) * 1000
        label = r.get("label", r.get("run_id", "?"))[:22]
        policy_key = r.get("policy_key") or ("undefended" if r.get("mode") == "undefended" else "")
        bars.append({"label": label, "lat_ms": lat_ms, "policy_key": policy_key})

    if not bars:
        return _svg_wrap("  <text x='50%' y='50%' text-anchor='middle'>No data</text>\n",
                         width, height, "Latency")

    max_lat = max(b["lat_ms"] for b in bars) or 1.0
    y_max = max_lat * 1.15

    bar_w = plot_w / len(bars) * 0.6
    bar_gap = plot_w / len(bars)

    parts = []
    parts.append(f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{PALETTE_BAR["bg"]}"/>\n')

    # Grid lines
    for frac in [0.25, 0.5, 0.75, 1.0]:
        gy = pad_t + plot_h * (1.0 - frac)
        parts.append(_svg_line(pad_l, gy, pad_l + plot_w, gy, color=PALETTE_BAR["grid"], width=0.8))
        parts.append(_svg_text(pad_l - 6, gy + 4, f"{frac * y_max:.0f}", size=8, anchor="end"))

    # Bars
    for i, bar in enumerate(bars):
        bx = pad_l + i * bar_gap + (bar_gap - bar_w) / 2
        bh = (bar["lat_ms"] / y_max) * plot_h if y_max > 0 else 0
        by = pad_t + plot_h - bh
        color = _bar_color(bar["policy_key"])
        parts.append(_svg_rect(bx, by, bar_w, bh, fill=color))

        # Value label above bar
        parts.append(_svg_text(bx + bar_w / 2, by - 4, f"{bar['lat_ms']:.0f}ms", size=8))

        # X-axis label (rotated)
        lx = bx + bar_w / 2
        ly = pad_t + plot_h + 15
        # Wrap long labels
        short = bar["label"][-20:] if len(bar["label"]) > 20 else bar["label"]
        parts.append(
            f'  <text x="{lx:.1f}" y="{ly}" font-size="7.5" '
            f'text-anchor="end" fill="{PALETTE_BAR["axis"]}" '
            f'transform="rotate(-40, {lx:.1f}, {ly})">{short}</text>\n'
        )

    # Axes
    ax_y = pad_t + plot_h
    parts.append(_svg_line(pad_l, ax_y, pad_l + plot_w, ax_y, color=PALETTE_BAR["axis"], width=1.5))
    parts.append(_svg_line(pad_l, ax_y, pad_l, pad_t, color=PALETTE_BAR["axis"], width=1.5))

    # Y-axis title (rotated)
    cy = pad_t + plot_h / 2
    parts.append(
        f'  <text x="14" y="{cy:.1f}" font-size="11" '
        f'text-anchor="middle" fill="{PALETTE_BAR["axis"]}" font-weight="bold" '
        f'transform="rotate(-90, 14, {cy:.1f})">Mean Latency (ms)</text>\n'
    )

    # Title
    parts.append(_svg_text(width / 2, 24,
                           "PACM Latency Overhead by Configuration", size=13, bold=True))

    # Legend
    legend_items = [("Undefended", "undefended"), ("Standard", "standard"),
                    ("Conservative", "conservative"), ("Aggressive", "aggressive")]
    lx0 = pad_l
    for j, (lbl, key) in enumerate(legend_items):
        lx = lx0 + j * 120
        ly = height - 18
        parts.append(_svg_rect(lx, ly - 8, 10, 10, fill=_bar_color(key), rx=2))
        parts.append(_svg_text(lx + 14, ly, lbl, size=9, anchor="start"))

    return _svg_wrap("".join(parts), width, height, "PACM Latency Comparison")


def build_latency_ascii(runs: list[dict], max_width: int = 60) -> str:
    """Generate ASCII bar chart for latency comparison."""
    bars = []
    for r in runs:
        m = r.get("metrics", {})
        lat_ms = (m.get("mean_latency_s") or 0) * 1000
        label = r.get("label", r.get("run_id", "?"))[:30]
        bars.append((label, lat_ms))

    if not bars:
        return "  [No latency data]"

    max_lat = max(b[1] for b in bars) or 1.0
    lines = ["\n  PACM Latency Overhead Comparison", f"  {'Label':<30}  {'ms':>6}  Chart"]
    lines.append("  " + "-" * 70)
    for label, lat_ms in bars:
        bar_len = int((lat_ms / max_lat) * max_width)
        bar = "#" * bar_len
        lines.append(f"  {label:<30}  {lat_ms:>6.1f}  {bar}")
    return "\n".join(lines)


# ─── Figure (c): Alpha sensitivity ───────────────────────────────────────────

def build_alpha_svg(
    alpha_results: list[dict],
    best_alpha: float,
    width: int = 480,
    height: int = 320,
) -> str:
    """Generate SVG for alpha sensitivity: separation and ABR/FPR vs. alpha."""
    pad_l, pad_r, pad_t, pad_b = 70, 30, 40, 60
    plot_w = width  - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    alphas  = [r["alpha"] for r in alpha_results]
    n_alpha = len(alphas)
    if n_alpha < 2:
        return _svg_wrap(
            "  <text x='50%' y='50%' text-anchor='middle'>Insufficient data</text>\n",
            width, height, "Alpha Sensitivity",
        )

    a_min, a_max = min(alphas), max(alphas)

    def _ax(alpha: float) -> float:
        return pad_l + (alpha - a_min) / max(a_max - a_min, 0.01) * plot_w

    def _ay(val: float, y_max: float = 1.0) -> float:
        return pad_t + (1.0 - val / max(y_max, 0.01)) * plot_h

    parts = []
    parts.append(f'  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>\n')

    # Grid
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = _ay(v)
        parts.append(_svg_line(pad_l, gy, pad_l + plot_w, gy, color="#f1f5f9", width=0.8))
        parts.append(_svg_text(pad_l - 6, gy + 4, f"{v:.2f}", size=8, anchor="end"))

    seps  = [r["separation"] for r in alpha_results]
    abrs  = [r["abr_at_threshold"] for r in alpha_results]
    fprs  = [r["fpr_at_threshold"] for r in alpha_results]

    # Draw lines for separation, ABR, FPR
    line_data = [
        (seps, "#2563eb", "Separation"),
        (abrs, "#16a34a", "ABR"),
        (fprs, "#dc2626", "FPR"),
    ]
    for values, color, name in line_data:
        pts = [(alpha_results[i]["alpha"], values[i]) for i in range(n_alpha)]
        for i in range(1, len(pts)):
            x1, y1 = _ax(pts[i-1][0]), _ay(pts[i-1][1])
            x2, y2 = _ax(pts[i][0]), _ay(pts[i][1])
            parts.append(_svg_line(x1, y1, x2, y2, color=color, width=2))
        for alpha_val, val in pts:
            parts.append(_svg_circle(_ax(alpha_val), _ay(val), r=4, color=color, stroke=color))

    # Best alpha vertical line
    bx = _ax(best_alpha)
    parts.append(_svg_line(bx, pad_t, bx, pad_t + plot_h,
                           color="#f59e0b", width=1.5, dash="4,3"))
    parts.append(_svg_text(bx + 4, pad_t + 12, f"best={best_alpha:.1f}", size=9,
                           anchor="start", color="#b45309"))

    # Axes
    ay_bottom = pad_t + plot_h
    parts.append(_svg_line(pad_l, ay_bottom, pad_l + plot_w, ay_bottom, color="#374151", width=1.5))
    parts.append(_svg_line(pad_l, ay_bottom, pad_l, pad_t, color="#374151", width=1.5))

    # X ticks
    for r in alpha_results:
        tx = _ax(r["alpha"])
        parts.append(_svg_line(tx, ay_bottom, tx, ay_bottom + 5, color="#374151", width=1))
        parts.append(_svg_text(tx, ay_bottom + 16, f"{r['alpha']:.1f}", size=9))

    # Axis titles
    cx = pad_l + plot_w / 2
    parts.append(_svg_text(cx, height - 8, "Alpha (semantic weight)", size=12, bold=True))
    cy = pad_t + plot_h / 2
    parts.append(
        f'  <text x="14" y="{cy:.1f}" font-size="11" '
        f'text-anchor="middle" fill="#374151" font-weight="bold" '
        f'transform="rotate(-90, 14, {cy:.1f})">Score</text>\n'
    )

    # Title
    parts.append(_svg_text(width / 2, 24,
                           "PACM Alpha Sensitivity Analysis", size=13, bold=True))

    # Legend
    for j, (_, color, name) in enumerate(line_data):
        lx = pad_l + j * 130
        ly = height - 16
        parts.append(_svg_line(lx, ly - 3, lx + 16, ly - 3, color=color, width=2.5))
        parts.append(_svg_text(lx + 20, ly, name, size=9, anchor="start"))

    return _svg_wrap("".join(parts), width, height, "PACM Alpha Sensitivity")


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def generate_all_figures(
    calibration_path: Path,
    sweep_path: Path,
    sensitivity_path: Path,
    output_dir: Path,
    ascii_only: bool = False,
    svg_only: bool = False,
) -> dict[str, Any]:
    """
    Load data files and generate all paper figures.

    Returns a dict with figure data for reproducibility (saved as figure_data.json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_data: dict[str, Any] = {}

    # ── Figure (a): ROC curve ─────────────────────────────────────────────────
    roc_generated = False
    if calibration_path.exists():
        with open(calibration_path, encoding="utf-8") as fh:
            cal = json.load(fh)
        frontier = cal.get("frontier", [])
        optimal  = cal.get("optimal")
        figure_data["roc"] = {"frontier": frontier, "optimal": optimal}

        if not ascii_only:
            svg = build_roc_svg(frontier, optimal)
            svg_path = output_dir / "roc_curve.svg"
            svg_path.write_text(svg, encoding="utf-8")
            print(f"  ROC SVG       -> {svg_path}")

        if not svg_only:
            ascii_fig = build_roc_ascii(frontier, optimal)
            print("\n--- ABR vs. FPR Pareto Frontier ---")
            print(ascii_fig)

        roc_generated = True
    else:
        print(f"  [SKIP] ROC: {calibration_path} not found.")

    # ── Figure (b): Latency comparison ────────────────────────────────────────
    lat_generated = False
    if sweep_path.exists():
        with open(sweep_path, encoding="utf-8") as fh:
            sweep = json.load(fh)
        runs = sweep.get("runs", [])
        figure_data["latency"] = {"runs": [
            {"label": r.get("label"), "policy_key": r.get("policy_key"),
             "mode": r.get("mode"),
             "lat_ms": (r.get("metrics", {}).get("mean_latency_s") or 0) * 1000}
            for r in runs
        ]}

        if not ascii_only:
            svg = build_latency_svg(runs)
            svg_path = output_dir / "latency_comparison.svg"
            svg_path.write_text(svg, encoding="utf-8")
            print(f"  Latency SVG   -> {svg_path}")

        if not svg_only:
            ascii_fig = build_latency_ascii(runs)
            print("\n--- Latency Overhead Comparison ---")
            print(ascii_fig)

        lat_generated = True
    else:
        print(f"  [SKIP] Latency: {sweep_path} not found.")

    # ── Figure (c): Alpha sensitivity ─────────────────────────────────────────
    alpha_generated = False
    if sensitivity_path.exists():
        with open(sensitivity_path, encoding="utf-8") as fh:
            sens = json.load(fh)
        alpha_results = sens.get("alpha_results", [])
        best_alpha    = sens.get("best_alpha", 0.7)
        figure_data["alpha_sensitivity"] = sens

        if not ascii_only:
            svg = build_alpha_svg(alpha_results, best_alpha)
            svg_path = output_dir / "alpha_sensitivity.svg"
            svg_path.write_text(svg, encoding="utf-8")
            print(f"  Alpha SVG     -> {svg_path}")

        if not svg_only:
            print(f"\n--- Alpha Sensitivity (best={best_alpha}) ---")
            print(f"  {'Alpha':>5}  {'Separation':>10}  {'ABR':>6}  {'FPR':>6}")
            print(f"  {'-'*35}")
            for r in alpha_results:
                star = " *" if abs(r["alpha"] - best_alpha) < 1e-6 else ""
                print(f"  {r['alpha']:>5.1f}  {r['separation']:>10.4f}  "
                      f"{r['abr_at_threshold']*100:>5.1f}%  "
                      f"{r['fpr_at_threshold']*100:>5.1f}%{star}")

        alpha_generated = True
    else:
        print(f"  [SKIP] Alpha: {sensitivity_path} not found.")

    # Save figure data for reproducibility
    fig_data_path = output_dir / "figure_data.json"
    with open(fig_data_path, "w", encoding="utf-8") as fh:
        json.dump(figure_data, fh, indent=2, default=str)
    print(f"  Figure data   -> {fig_data_path}")

    return figure_data


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _REAL_LLM_SWEEP = _repo_root / "experiments" / "real_llm_sweep_summary.json"

    parser = argparse.ArgumentParser(
        description="PACM Session 6/10 -- Paper figure generator (SVG + ASCII)."
    )
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument(
        "--sweep", default=str(DEFAULT_SWEEP),
        help="Path to sweep_summary.json. Use --real as shortcut for real_llm_sweep_summary.json.",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Shortcut: use experiments/real_llm_sweep_summary.json as --sweep input.",
    )
    parser.add_argument("--sensitivity", default=str(DEFAULT_SENSITIVITY))
    parser.add_argument(
        "--output-dir", "--output", default=str(DEFAULT_OUTPUT_DIR),
        dest="output_dir",
        help="Directory to write SVG figures.",
    )
    parser.add_argument("--ascii-only",  action="store_true",
                        help="Only generate ASCII figures (no SVG files).")
    parser.add_argument("--svg-only",    action="store_true",
                        help="Only generate SVG files (no ASCII terminal output).")
    args = parser.parse_args()

    sweep_path = Path(_REAL_LLM_SWEEP if args.real else args.sweep)

    print(f"\n{'='*60}")
    print("  PACM Session 6/10 -- Paper Figure Generator")
    print(f"  sweep={sweep_path.name}")
    print(f"{'='*60}")

    generate_all_figures(
        calibration_path=Path(args.calibration),
        sweep_path=sweep_path,
        sensitivity_path=Path(args.sensitivity),
        output_dir=Path(args.output_dir),
        ascii_only=args.ascii_only,
        svg_only=args.svg_only,
    )
