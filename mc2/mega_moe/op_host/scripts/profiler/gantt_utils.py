"""Shared utilities for Gantt chart drawing scripts.

Provides a colorblind-safe palette, auto-classification of profiler events
into containers (events whose instances contain other events' instances),
and shared rendering helpers.
"""

from __future__ import annotations

import bisect

import pandas as pd

DEFAULT_FREQ: float = 1650.0  # override via --freq; actual from msprof op OpBasicInfo_*.csv Rated Freq or INI cube_freq

# Container auto-classification thresholds
CONTAINER_COVERAGE_THRESHOLD = 0.5   # min fraction of sub-events that must be contained
CONTAINER_SIZE_RATIO_THRESHOLD = 0.9 # max avg(sub_dur)/avg(container_dur) for container classification

# Okabe-Ito colorblind-safe palette
PALETTE = [
    '#0072B2',  # blue
    '#E69F00',  # orange
    '#009E73',  # green
    '#CC79A7',  # pink
    '#56B4E9',  # sky blue
    '#D55E00',  # vermillion
    '#F0E442',  # yellow
    '#000000',  # black
    '#0D0887',  # dark blue
    '#46039F',  # purple
]

_KERNEL_NAMES = ('KERNEL', 'EVT_KERNEL')


def auto_classify(events_df: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    """Auto-detect containers: events whose individual instances contain other events' instances."""
    containers = set()
    ec = {'KERNEL': '#cccccc', 'EVT_KERNEL': '#cccccc'}
    color_idx = 0

    unique_events = events_df['event'].unique()
    for ev in unique_events:
        if ev in _KERNEL_NAMES:
            continue
        if color_idx < len(PALETTE):
            ec[ev] = PALETTE[color_idx]
            color_idx += 1
        else:
            ec[ev] = '#999999'

    instances = {}
    sorted_starts = {}
    for ev in unique_events:
        ev_rows = events_df[events_df['event'] == ev]
        inst = sorted(zip(ev_rows['begin_cycle'].values, ev_rows['end_cycle'].values))
        instances[ev] = inst
        sorted_starts[ev] = [b for b, e in inst]

    for ev in unique_events:
        if ev in _KERNEL_NAMES:
            containers.add(ev)
            continue
        ev_instances = instances.get(ev, [])
        if not ev_instances:
            continue
        starts = sorted_starts[ev]
        max_ev_dur = max(ee - eb for eb, ee in ev_instances) if ev_instances else 0
        for other in unique_events:
            if other == ev:
                continue
            other_instances = instances.get(other, [])
            if not other_instances:
                continue
            contained_count = 0
            for ob, oe in other_instances:
                idx = bisect.bisect_right(starts, ob)
                for i in range(idx - 1, -1, -1):
                    eb, ee = ev_instances[i]
                    if eb < ob - max_ev_dur:
                        break
                    if ee >= oe:
                        contained_count += 1
                        break
            if contained_count / len(other_instances) > CONTAINER_COVERAGE_THRESHOLD:
                avg_other_dur = sum(oe - ob for ob, oe in other_instances) / len(other_instances)
                avg_ev_dur = sum(ee - eb for eb, ee in ev_instances) / len(ev_instances)
                ratio = avg_other_dur / max(avg_ev_dur, 1)
                if ratio < CONTAINER_SIZE_RATIO_THRESHOLD:
                    containers.add(ev)
                    break

    return ec, containers


def compute_lane_sort_key(df: pd.DataFrame) -> pd.Series:
    """Compute sort key for lanes: role_order * 1000 + local_id."""
    return df['role'].map({'AIV': 0, 'AIC': 1}).fillna(2) * 1000 + df['local_id']


def compute_xlimit(df: pd.DataFrame) -> float:
    """Compute x-axis limit: clip outliers at median*5, pad by 5%."""
    non_outlier = df[df['e'] < df['e'].median() * 10]['e']
    if len(non_outlier) == 0:
        return df['e'].max() * 1.05 if len(df) > 0 else 1.0
    return min(non_outlier.max() * 1.1, non_outlier.median() * 5)


def render_bars(ax, rows_sorted, lane_map: dict, ec: dict, containers: set) -> None:
    """Render event bars using PolyCollection for performance."""
    from matplotlib.collections import PolyCollection

    fill_verts: list[list[tuple[float, float]]] = []
    fill_colors: list[str] = []
    outline_verts: list[list[tuple[float, float]]] = []
    outline_colors: list[str] = []

    for rw in rows_sorted.itertuples():
        if rw.lane not in lane_map:
            continue
        y = lane_map[rw.lane]
        left = rw.b
        dur = rw.e - rw.b
        if dur <= 0:
            continue
        color = ec.get(rw.event, '#999999')

        if rw.event in containers:
            h = 0.3
            y0 = y - h / 2
            outline_verts.append([(left, y0), (left + dur, y0), (left + dur, y0 + h), (left, y0 + h)])
            outline_colors.append(color)
        else:
            h = 0.65
            y0 = y - h / 2
            fill_verts.append([(left, y0), (left + dur, y0), (left + dur, y0 + h), (left, y0 + h)])
            fill_colors.append(color)

    if fill_verts:
        fill_coll = PolyCollection(fill_verts, facecolors=fill_colors, edgecolors='none', zorder=2)
        ax.add_collection(fill_coll)
    if outline_verts:
        outline_coll = PolyCollection(outline_verts, facecolors='none', edgecolors=outline_colors, linewidths=0.8, zorder=0)
        ax.add_collection(outline_coll)


def build_legend(ax, ec: dict, containers: set) -> None:
    """Build and place legend with event color patches."""
    import matplotlib.patches as mp
    ue = sorted(ec.keys(),
                key=lambda e: list(ec.keys()).index(e) if e in ec else 99)
    handles = []
    for e in ue:
        c = ec.get(e, '#999')
        if e in containers:
            handles.append(mp.Patch(edgecolor=c, facecolor='none', linewidth=1.5, label=e))
        else:
            handles.append(mp.Patch(color=c, label=e))
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=6)
