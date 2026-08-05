#!/usr/bin/env python3
"""Draw combined Gantt chart: both ranks on same time axis, with AIV/AIC completion lines.

Auto-assigns colors and detects containers, same as draw_gantt.py.

Usage:
    python3 draw_combined.py <prof_paired.csv> <output.svg> [tag] [--freq MHZ]
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib required. Run: pip install matplotlib")
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas required. Run: pip install pandas")
    sys.exit(1)
from matplotlib.lines import Line2D

from gantt_utils import (
    DEFAULT_FREQ,
    auto_classify,
    compute_lane_sort_key,
    compute_xlimit,
    render_bars,
)

RK_PALETTE = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']


def rk_color(n: int) -> str:
    """Return a color from the rank palette for the given rank index."""
    return RK_PALETTE[n % len(RK_PALETTE)]


def draw_combined(df: pd.DataFrame, out_path: str, tag: str | None = None, freq: float = DEFAULT_FREQ) -> None:
    """Draw combined Gantt chart with all ranks on same time axis and save to SVG."""
    all_rows = []
    lane_map = {}
    y = 0
    rank_info = {}

    ranks = sorted(set(df['rank'].astype(str).unique()), key=lambda x: int(x))

    for rk in ranks:
        rdf = df[df['rank'].astype(int) == int(rk)].copy()
        if rdf.empty:
            print("[ERROR] No data for rank %s. Available ranks: %s" % (rk, sorted(df['rank'].unique())))
            sys.exit(1)
        t0 = rdf['begin_cycle'].min()
        rdf['b'] = (rdf['begin_cycle'] - t0) / freq
        rdf['e'] = (rdf['end_cycle'] - t0) / freq
        rdf['rel_rank'] = rk

        aiv_kern = rdf[(rdf['role'] == 'AIV') & (rdf['event'].str.contains('KERNEL', na=False))]
        aiv_end = aiv_kern['e'].max() if len(aiv_kern) else 0
        aic_kern = rdf[(rdf['role'] == 'AIC') & (rdf['event'].str.contains('KERNEL', na=False))]
        aic_end = aic_kern['e'].max() if len(aic_kern) else 0

        rank_info[rk] = {'aiv_end': aiv_end, 'aic_end': aic_end}

        rdf['_sk'] = compute_lane_sort_key(rdf)
        lanes = sorted(rdf['_sk'].unique(), key=lambda x: x)
        for sk in lanes:
            sub = rdf[rdf['_sk'] == sk]
            lane_name = 'R' + rk + '/' + sub.iloc[0]['role'] + str(sub.iloc[0]['local_id'])
            lane_map[lane_name] = y
            y += 1
        y += 1
        all_rows.append(rdf)

    combined = pd.concat(all_rows)
    combined['lane'] = 'R' + combined['rel_rank'].astype(str) + '/' + combined['role'] + combined['local_id'].astype(str)

    ec, containers = auto_classify(combined)
    zo = {}
    for ev in combined['event'].unique():
        zo[ev] = 0 if ev in containers else 2
    combined['zorder'] = combined['event'].map(zo).fillna(2)

    xm = compute_xlimit(combined)
    for rk in ranks:
        xm = max(xm, rank_info[rk]['aic_end'] * 1.05, rank_info[rk]['aiv_end'] * 1.05)

    n_lanes = len(lane_map)
    n_ranks = len(ranks)
    estimated_height = n_lanes * 0.22
    if n_ranks >= 8:
        print(f"[WARN] {n_ranks} ranks detected — SVG will be very tall ({estimated_height:.0f} inches). "
              f"Consider using to_chrome_trace.py + MindStudio Insight instead.")
    fig, ax = plt.subplots(figsize=(24, max(12, estimated_height)))

    rows_sorted = combined.sort_values('zorder')
    render_bars(ax, rows_sorted, lane_map, ec, containers)

    r_lanes_by_rank = {}
    for rk in ranks:
        r_lanes_by_rank[rk] = [l for l in lane_map if l.startswith('R' + rk + '/')]

    for rk_idx, rk in enumerate(ranks):
        info = rank_info[rk]
        color = rk_color(rk_idx)
        r_lanes = r_lanes_by_rank[rk]
        if not r_lanes:
            continue

        if info['aiv_end'] > 0:
            ax.axvline(x=info['aiv_end'], color=color, linewidth=1.5,
                       linestyle='--', alpha=0.85, zorder=10)
        if info['aic_end'] > 0:
            ax.axvline(x=info['aic_end'], color=color, linewidth=1.8,
                       linestyle='-', alpha=0.85, zorder=10)

    for i in range(len(ranks) - 1):
        r_top = r_lanes_by_rank.get(ranks[i], [])
        r_bot = r_lanes_by_rank.get(ranks[i + 1], [])
        if r_top and r_bot:
            sep_y = (lane_map[r_top[-1]] + lane_map[r_bot[0]]) / 2
            ax.axhline(y=sep_y, color='#333333', linewidth=2.0, zorder=12)

    for rk in ranks:
        r_lanes = r_lanes_by_rank.get(rk, [])
        if not r_lanes:
            continue
        mid_y = (lane_map[r_lanes[0]] + lane_map[r_lanes[-1]]) / 2
        ax.text(-xm * 0.03, mid_y, 'Rank ' + rk,
                ha='right', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(list(lane_map.values()))
    ax.set_yticklabels(list(lane_map.keys()), fontsize=5.5)
    ax.set_xlim(0, xm)
    ax.set_xlabel('Time (us) - per-rank relative (cross-rank cy not synchronized)', fontsize=9)
    ax.invert_yaxis()
    ax.set_title('Profiler - Combined Ranks (%s)' % tag, fontsize=11)

    ue = sorted(set(combined['event'].unique()),
                key=lambda e: list(ec.keys()).index(e) if e in ec else 99)
    handles = []
    for e in ue:
        c = ec.get(e, '#999')
        if e in containers:
            handles.append(mp.Patch(edgecolor=c, facecolor='none', linewidth=1.5, label=e))
        else:
            handles.append(mp.Patch(color=c, label=e))
    handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='AIV end'))
    handles.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=1.8, label='AIC end'))
    ax.legend(handles=handles, loc='upper right', fontsize=6)

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

    print("[INFO] %s saved (xlim=%.0fus, lanes=%d)" % (out_path, xm, n_lanes))
    for rk in ranks:
        info = rank_info[rk]
        gap = info['aic_end'] - info['aiv_end']
        slow = 'AIC' if info['aic_end'] > info['aiv_end'] else 'AIV'
        print("  Rank %s: AIV=%.1fus  AIC=%.1fus  gap=%.1fus (%s slower)" % (
            rk, info['aiv_end'], info['aic_end'], abs(gap), slow))


def main() -> None:
    """Parse CLI arguments and draw combined Gantt chart."""
    parser = argparse.ArgumentParser(description='Draw combined Gantt chart with both ranks on same time axis')
    parser.add_argument('csv_path', help='Path to prof_paired.csv')
    parser.add_argument('output', help='Output SVG path')
    parser.add_argument('tag', nargs='?', default=None, help='Tag label (default: parent directory name)')
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ,
                        help='Clock frequency in MHz (default: %.1f)' % DEFAULT_FREQ)
    parser.add_argument('--max-ranks', type=int, default=0,
                        help='Maximum number of ranks to display (0 = all). '
                             'Useful for 8+ rank datasets where full SVG is unreadable.')
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

    if args.max_ranks > 0:
        all_ranks = sorted(df['rank'].unique())
        if len(all_ranks) > args.max_ranks:
            keep_ranks = all_ranks[:args.max_ranks]
            df = df[df['rank'].isin(keep_ranks)]
            print(f"[INFO] Showing {args.max_ranks} of {len(all_ranks)} ranks: {keep_ranks}")

    tag = args.tag if args.tag else os.path.basename(os.path.dirname(args.csv_path))
    draw_combined(df, args.output, tag, freq=args.freq)


if __name__ == '__main__':
    main()
