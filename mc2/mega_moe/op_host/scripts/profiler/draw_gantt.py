#!/usr/bin/env python3
"""Draw per-rank Gantt SVG from prof_paired.csv.

Auto-assigns colors from a palette for any event names found in the CSV.
Events whose duration contains sub-events are auto-detected as containers.

Usage:
    python3 draw_gantt.py <prof_paired.csv> [rank] [output.svg] [title] [--freq MHZ]
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib required. Run: pip install matplotlib")
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas required. Run: pip install pandas")
    sys.exit(1)

from gantt_utils import (
    DEFAULT_FREQ,
    auto_classify,
    build_legend,
    compute_lane_sort_key,
    compute_xlimit,
    render_bars,
)


def draw(df: pd.DataFrame, rank: str, out_path: str, title: str | None = None, freq: float = DEFAULT_FREQ) -> None:
    """Draw a per-rank Gantt chart and save to SVG."""
    rdf = df[df['rank'].astype(int) == int(rank)].copy()
    if rdf.empty:
        print("[ERROR] No data for rank %s. Available ranks: %s" % (rank, sorted(df['rank'].unique())))
        sys.exit(1)
    t0 = rdf['begin_cycle'].min()
    rdf['b'] = (rdf['begin_cycle'] - t0) / freq
    rdf['e'] = (rdf['end_cycle'] - t0) / freq
    rdf['lane'] = rdf['role'] + rdf['local_id'].astype(str)
    rdf['sk'] = compute_lane_sort_key(rdf)
    rdf = rdf.sort_values('sk')
    lanes = rdf.groupby('lane')['sk'].first().sort_values().index.tolist()
    l2y = {l: i for i, l in enumerate(lanes)}

    ec, containers = auto_classify(rdf)
    zo = {}
    for ev in rdf['event'].unique():
        zo[ev] = 0 if ev in containers else 2
    rdf['zorder'] = rdf['event'].map(zo).fillna(2)

    xm = compute_xlimit(rdf)

    fig, ax = plt.subplots(figsize=(22, max(10, len(lanes) * 0.22)))
    rows_sorted = rdf.sort_values('zorder')
    render_bars(ax, rows_sorted, l2y, ec, containers)

    sep = sum(1 for l in lanes if l.startswith('AIV'))
    if sep > 0:
        ax.axhline(y=sep - 0.5, color='black', linewidth=1.2)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=6.5)
    ax.set_xlim(0, xm)
    ax.set_xlabel('Time (us)', fontsize=10)
    ax.invert_yaxis()
    if not title:
        title = 'Profiler - Rank %s' % rank
    ax.set_title(title, fontsize=11)

    build_legend(ax, ec, containers)
    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=150)
    except OSError as e:
        print("[ERROR] Failed to save %s: %s" % (out_path, e))
        sys.exit(1)
    plt.close(fig)
    print("[INFO] %s saved (xlim=%.0fus, lanes=%d)" % (out_path, xm, len(lanes)))


def main() -> None:
    """Parse CLI arguments and draw per-rank Gantt SVG."""
    parser = argparse.ArgumentParser(description='Draw per-rank Gantt SVG from prof_paired.csv')
    parser.add_argument('csv_path', help='Path to prof_paired.csv')
    parser.add_argument('rank', nargs='?', default='0', help='Rank to plot (default: 0)')
    parser.add_argument('output', nargs='?', default=None, help='Output SVG path (default: gantt_rank{rank}.svg)')
    parser.add_argument('title', nargs='?', default='', help='Chart title (default: auto)')
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ,
                        help='Clock frequency in MHz (default: %.1f)' % DEFAULT_FREQ)
    args = parser.parse_args()
    if args.freq <= 0:
        print("[ERROR] --freq must be positive, got %s" % args.freq)
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print("[ERROR] CSV file not found: %s" % args.csv_path)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("[ERROR] CSV file is empty: %s" % args.csv_path)
        sys.exit(1)

    required = ['rank', 'core_id', 'role', 'local_id', 'event', 'begin_cycle', 'end_cycle', 'duration_cycles']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("[ERROR] CSV missing required columns: %s" % missing)
        sys.exit(1)

    out_path = args.output if args.output else 'gantt_rank%s.svg' % args.rank
    draw(df, args.rank, out_path, args.title, freq=args.freq)


if __name__ == '__main__':
    main()
