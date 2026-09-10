#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Parse a msprof markdown performance report and render Plotly line charts.

Usage:
    python3 plot_perf_report.py \
        --report /home/g00940185/work/ops-transformer/perf_dual_e4m3_alg0_msprof_report.md \
        --output perf_fused_vs_small_chart.html

It parses each "## T..." section and creates:
  1. Duration line chart: Fused vs Small total vs each small op
  2. Speedup line chart: SmallTotal / Fused
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_report(report_text):
    sections = []
    lines = report_text.splitlines()
    current_shape = None
    current_items = {}
    current_ratio = None
    current_speedup = None

    def flush():
        nonlocal current_shape, current_items, current_ratio, current_speedup
        if current_shape and current_items:
            sections.append({
                "shape": current_shape,
                "items": current_items,
                "ratio": current_ratio,
                "speedup": current_speedup,
            })
        current_shape = None
        current_items = {}
        current_ratio = None
        current_speedup = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("## "):
            flush()
            current_shape = line[3:].strip()
        elif line.startswith("|") and current_shape is not None:
            cells = [c.strip().replace("**", "").replace("`", "") for c in line.strip("|").split("|")]
            if len(cells) >= 2 and not set(cells[1]) <= {"-", ":"}:
                try:
                    value = float(cells[1])
                except ValueError:
                    continue
                current_items[cells[0].strip()] = value
        elif line.startswith("- Fused/SmallTotal:"):
            try:
                current_ratio = float(line.split(":", 1)[1].strip())
            except ValueError:
                current_ratio = None
        elif line.startswith("- SmallTotal/Fused speedup:"):
            try:
                current_speedup = float(line.split(":", 1)[1].strip())
            except ValueError:
                current_speedup = None
    flush()
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="path to markdown report")
    parser.add_argument("--output", default="perf_fused_vs_small_chart.html",
                        help="output html path")
    parser.add_argument("--title", default=None, help="chart title")
    args = parser.parse_args()

    report_text = Path(args.report).read_text(encoding="utf-8")
    sections = parse_report(report_text)
    if not sections:
        parser.error(f"No parsed shape sections in {args.report}")

    df = pd.DataFrame([
        {
            "shape": s["shape"],
            **s["items"],
            "speedup": s.get("speedup"),
        }
        for s in sections
    ])

    # Item order: Fused, Small total, then small components.
    preferred = [c for c in ["Fused", "Small total"] if c in df.columns]
    small_cols = [c for c in df.columns if c.startswith("Small ") and c != "Small total"]
    ordered = preferred + small_cols

    # Filter out rows that have no numeric data.
    df = df.dropna(subset=[c for c in ordered if c in df.columns])
    x = df["shape"].astype(str)

    title = args.title or "Fused vs Small Ops Performance (Task Duration)"
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Task Duration (us)", "SmallTotal / Fused Speedup"),
        vertical_spacing=0.12,
    )

    colors = {
        "Fused": "#d62728",
        "Small total": "#1f77b4",
    }
    color_pool = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    color_idx = 0
    for col in ordered:
        if col not in df.columns:
            continue
        color = colors.get(col, color_pool[color_idx % len(color_pool)])
        color_idx += 1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="lines+markers",
                name=col,
                line=dict(width=2.5, color=color),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["speedup"] if "speedup" in df.columns else None,
            mode="lines+markers",
            name="Speedup",
            line=dict(width=2.5, color="#17becf"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=100, b=80),
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Duration (us)", row=1, col=1)
    fig.update_yaxes(title_text="Speedup (x)", row=2, col=1)

    output = Path(args.output)
    output.write_text(fig.to_html(include_plotlyjs="cdn"), encoding="utf-8")
    print(f"Parsed {len(sections)} shapes.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
