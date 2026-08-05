#!/usr/bin/env python3
"""Convert prof_paired.csv to Chrome Trace Event JSON for MindStudio Insight.

Supports MIX_1_1, MIX_1_2, and AIV-only core layouts (auto-detected from CSV).
Flow arrows are configured via --flow-map JSON file.

pid/tid layout depends on detected mode:
  MIX_1_1:  pid = rank * AIC_CORES + local_id,  tid = 0 (AIV) / 1 (AIC)
  MIX_1_2:  pid = rank * AIC_CORES + aic_id,     tid = aiv_id%2 (AIV) / 2 (AIC)
  AIV_ONLY: pid = rank,                          tid = local_id
"""

from __future__ import annotations

import json
import sys

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas required. Run: pip install pandas")
    sys.exit(1)
import os
import argparse

from gantt_utils import DEFAULT_FREQ

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

REQUIRED_CSV_COLUMNS = [
    'rank', 'core_id', 'role', 'local_id', 'event',
    'begin_cycle', 'end_cycle', 'duration_cycles', 'payload'
]


def detect_layout(df: pd.DataFrame, aic_cores_override: int) -> tuple[str, int]:
    """Auto-detect core layout from CSV data for a single rank.

    Returns (mode, aic_cores):
      mode      - 'MIX_1_1', 'MIX_1_2', or 'AIV_ONLY'
      aic_cores - number of AIC cores per rank (used for pid calculation)
    """
    aiv_ids = sorted(df[df['role'] == 'AIV']['local_id'].astype(int).unique())
    aic_ids = sorted(df[df['role'] == 'AIC']['local_id'].astype(int).unique())

    aiv_count = len(aiv_ids)
    aic_count = len(aic_ids)

    if aic_count == 0 and aiv_count > 0:
        mode = 'AIV_ONLY'
        aic_cores = 0
    elif aic_count > 0 and aiv_count == 2 * aic_count:
        mode = 'MIX_1_2'
        aic_cores = aic_cores_override if aic_cores_override > 0 else aic_count
    elif aic_count > 0 and aiv_count == aic_count:
        mode = 'MIX_1_1'
        aic_cores = aic_cores_override if aic_cores_override > 0 else aic_count
    elif aic_count > 0:
        print("[WARN] Non-standard AIV:AIC ratio (%d:%d). Falling back to MIX_1_1 layout. "
              "Use --aic-cores to override." % (aiv_count, aic_count))
        mode = 'MIX_1_1'
        aic_cores = aic_cores_override if aic_cores_override > 0 else aic_count
    else:
        mode = 'AIV_ONLY'
        aic_cores = 0

    return mode, aic_cores


def aggregate_median(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Aggregate multi-round profiling data by taking median per event-pair group.

    Groups by (rank, core_id, role, local_id, event, payload) and takes median of
    begin_cycle, end_cycle, duration_cycles. This collapses N rounds of repeated
    measurements into a single representative trace per event pair, removing
    per-round jitter while preserving inter-event timing relationships.

    Returns (aggregated_df, n_rounds, original_row_count).
    n_rounds == 1 means single-round data — df returned unchanged.
    """
    group_cols = ['rank', 'core_id', 'role', 'local_id', 'event', 'payload']
    df = df.copy()
    df['payload'] = df['payload'].fillna(0)
    orig_count = len(df)
    rounds = df.groupby(group_cols).size()
    n_rounds = int(rounds.max()) if len(rounds) > 0 else 1
    if n_rounds <= 1:
        return df, 1, orig_count
    agg = df.groupby(group_cols).agg(
        begin_cycle=('begin_cycle', 'median'),
        end_cycle=('end_cycle', 'median'),
        duration_cycles=('duration_cycles', 'median'),
    ).reset_index()
    for c in ['begin_cycle', 'end_cycle', 'duration_cycles']:
        agg[c] = agg[c].round().astype('int64')
    agg['rank'] = agg['rank'].astype(int)
    return agg, n_rounds, orig_count


def build_flow_arrows(events: list[dict], flow_map: list[dict]) -> tuple[list[dict], list[tuple]]:
    """Generate flow arrow events from a flow-map configuration.

    Returns (flow_events list, per-entry arrow counts list).
    """
    flow_events = []
    flow_id = 1
    counts = []

    for fm_entry in flow_map:
        sf_pattern = fm_entry['setflag_pattern']
        wf_pattern = fm_entry['waitflag_pattern']
        payload_factor = fm_entry.get('payload_factor', 0)
        flow_name = fm_entry.get('name', 'SetFlag->WaitFlag')

        # Filter by role (cat field) for generality across layout modes
        setflag_evts = [e for e in events if sf_pattern in e['name'] and e['cat'] == 'AIV']
        waitflag_evts = [e for e in events if wf_pattern in e['name'] and e['cat'] == 'AIC']

        if not setflag_evts or not waitflag_evts:
            print("[WARN] Flow '%s': no events matched setflag_pattern='%s' (cat=AIV) or waitflag_pattern='%s' (cat=AIC)" % (
                flow_name, sf_pattern, wf_pattern))

        sf_by_payload = {}
        for sf in setflag_evts:
            p = sf['args']['payload']
            sf_by_payload.setdefault(p, []).append(sf)

        wf_by_payload = {}
        for wf in waitflag_evts:
            p = wf['args']['payload']
            wf_by_payload.setdefault(p, []).append(wf)

        # Auto-detect payload factor from waitflag payload stride if not specified
        if payload_factor == 0:
            wf_payloads_sorted = sorted(wf_by_payload.keys())
            if len(wf_payloads_sorted) >= 2:
                payload_factor = wf_payloads_sorted[1] - wf_payloads_sorted[0]
            else:
                payload_factor = 2

        arrow_count = 0
        for sf_payload, sf_list in sf_by_payload.items():
            wf_payload = sf_payload * payload_factor
            wf_list = wf_by_payload.get(wf_payload, [])
            if not wf_list:
                continue

            wf_by_pid = {}
            for wf in wf_list:
                wf_by_pid.setdefault(wf['pid'], []).append(wf)

            for sf in sf_list:
                wf_candidates = wf_by_pid.get(sf['pid'], [])
                if not wf_candidates:
                    continue
                wf = min(wf_candidates,
                         key=lambda w: abs((w['ts'] + w['dur']) - (sf['ts'] + sf['dur'])))
                s_ts = sf['ts'] + sf['dur']
                f_ts = max(wf['ts'] + wf['dur'], s_ts + 0.001)
                flow_events.append({
                    "name": flow_name,
                    "cat": "flow",
                    "ph": "s",
                    "id": flow_id,
                    "pid": sf['pid'],
                    "tid": sf['tid'],
                    "ts": round(s_ts, 3),
                })
                flow_events.append({
                    "name": flow_name,
                    "cat": "flow",
                    "ph": "f",
                    "bp": "e",
                    "id": flow_id,
                    "pid": wf['pid'],
                    "tid": wf['tid'],
                    "ts": round(f_ts, 3),
                })
                flow_id += 1
                arrow_count += 1

        counts.append((flow_name, sf_pattern, wf_pattern, payload_factor, arrow_count))

    return flow_events, counts


def convert(df: pd.DataFrame, out_path: str, freq: float, aic_cores_hint: int, flow_map: list[dict] | None, rank: str | None = None, compact: bool = False) -> None:
    """Convert a single rank's CSV data to Chrome Trace Event JSON."""
    if rank is not None:
        df = df[df['rank'].astype(int) == int(rank)].copy()

    # Detect layout from this rank's data
    mode, aic_cores = detect_layout(df, aic_cores_hint)
    print("[INFO] Rank %s: detected layout=%s, AIC_CORES=%d" % (rank, mode, aic_cores))

    df['rk'] = df['rank'].astype(int)
    df['lid'] = df['local_id'].astype(int)
    if mode == 'AIV_ONLY':
        df['pid'] = df['rk']
        df['tid'] = df['lid']
    elif mode == 'MIX_1_1':
        df['pid'] = df['rk'] * aic_cores + df['lid']
        df['tid'] = (df['role'] != 'AIV').astype(int)  # 0=AIV, 1=AIC
    elif mode == 'MIX_1_2':
        is_aiv = df['role'] == 'AIV'
        df['pid'] = df['rk'] * aic_cores + df['lid'].where(~is_aiv, df['lid'] // 2)
        df['tid'] = is_aiv.astype(int) * (df['lid'] % 2) + (~is_aiv).astype(int) * 2

    df['ts'] = (df['begin_cycle'] / freq).round(3)
    neg_count = (df['duration_cycles'] < 0).sum()
    if neg_count > 0:
        print("[WARN] %d events have negative duration_cycles (clipped to 0.001us)" % neg_count)
    df['dur'] = (df['duration_cycles'] / freq).round(3).clip(lower=0.001)
    df['payload_int'] = df['payload'].fillna(0).astype(int)
    df['cycles_int'] = df['duration_cycles'].astype(int)

    t0 = df['ts'].min()  # single t0 for entire rank — preserves cross-core timing
    df['ts'] = (df['ts'] - t0).round(3)

    pid_tid_role = df[['pid', 'tid', 'role']].drop_duplicates()
    tids_seen = {}
    tid_roles = {}
    for row in pid_tid_role.itertuples(index=False):
        tids_seen.setdefault(row.pid, set()).add(row.tid)
        tid_roles[(row.pid, row.tid)] = row.role

    events = []
    for row in df[['event', 'role', 'pid', 'tid', 'ts', 'dur',
                    'rk', 'lid', 'payload_int', 'cycles_int']].itertuples(index=False):
        events.append({
            "name": row.event,
            "cat": row.role,
            "ph": "X",
            "pid": row.pid,
            "tid": row.tid,
            "ts": row.ts,
            "dur": row.dur,
            "args": {
                "rank": row.rk,
                "local_id": row.lid,
                "payload": row.payload_int,
                "cycles": row.cycles_int,
            }
        })

    # Metadata: process and thread names
    meta = []
    for pid in sorted(tids_seen.keys()):
        if mode == 'AIV_ONLY':
            rk = pid
            meta.append({
                "name": "process_name", "ph": "M", "pid": pid,
                "args": {"name": "Rank%s" % rk}
            })
        else:
            rk = pid // aic_cores
            lid = pid % aic_cores
            meta.append({
                "name": "process_name", "ph": "M", "pid": pid,
                "args": {"name": "Core%02d Rank%s" % (lid, rk)}
            })

        for tid in sorted(tids_seen[pid]):
            role = tid_roles.get((pid, tid), "AIV" if tid == 0 else "AIC")
            meta.append({
                "name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                "args": {"name": role}
            })

    # Flow arrows from flow-map configuration
    flow_events = []
    flow_info = []
    if flow_map:
        flow_events, flow_info = build_flow_arrows(events, flow_map)
        for name, sf_pat, wf_pat, factor, count in flow_info:
            print("[INFO]   Flow '%s': %d arrows (%s -> %s, factor=%d)" %
                  (name, count, sf_pat, wf_pat, factor))

    all_events = meta + sorted(
        events, key=lambda e: (e['pid'], e['tid'], e['ts'])) + flow_events

    trace = {
        "traceEvents": all_events,
        "displayTimeUnit": "us",
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        if HAS_ORJSON:
            opt = 0
            if not compact:
                opt |= orjson.OPT_INDENT_2
            with open(out_path, 'wb') as f:
                f.write(orjson.dumps(trace, option=opt))
        else:
            json_kwargs = {'separators': (',', ':')} if compact else {'indent': 2}
            with open(out_path, 'w') as f:
                json.dump(trace, f, **json_kwargs)
    except OSError as e:
        print("[ERROR] Failed to write %s: %s" % (out_path, e))
        sys.exit(1)
    print("[INFO] %s: %d events, %d groups, %d flow arrows" % (
        out_path, len(events), len(tids_seen), len(flow_events) // 2))


def main() -> None:
    """CLI entry point: read CSV, process each rank, write trace JSON files."""
    parser = argparse.ArgumentParser(
        description='Convert prof_paired.csv to Chrome Trace Event JSON for MindStudio Insight.')
    parser.add_argument('csv', help='Path to prof_paired.csv')
    parser.add_argument('output_dir', nargs='?', default=None,
                        help='Output directory (default: same directory as CSV)')
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ,
                        help='NPU clock frequency in MHz for cycle->us conversion (default: %.1f)' % DEFAULT_FREQ)
    parser.add_argument('--aic-cores', type=int, default=0,
                        help='Number of AIC cores per rank for pid calculation (default: 0 = auto-detect)')
    parser.add_argument('--flow-map', type=str, default=None,
                        help='JSON file defining flow arrow relationships (default: no flow arrows)')
    parser.add_argument('--compact', action='store_true',
                        help='Compact JSON output (no indentation, smaller file, faster write)')
    parser.add_argument('--align-ranks', action='store_true',
                        help='Align all ranks to the first KERNEL_BEGIN event (cross-rank comparison)')
    parser.add_argument('--all-rounds', action='store_true',
                        help='Disable multi-round median aggregation; emit every round (default: aggregate by median)')

    args = parser.parse_args()

    if args.freq <= 0:
        print("[ERROR] --freq must be positive, got %s" % args.freq)
        sys.exit(1)

    csv_path = args.csv
    out_dir = args.output_dir if args.output_dir else os.path.dirname(csv_path) or '.'

    flow_map = None
    if args.flow_map:
        try:
            with open(args.flow_map, 'r') as f:
                flow_map = json.load(f)
        except FileNotFoundError:
            print("[ERROR] flow-map file not found: %s" % args.flow_map)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print("[ERROR] Invalid JSON in flow-map %s: %s" % (args.flow_map, e))
            sys.exit(1)
        if not isinstance(flow_map, list):
            print("[ERROR] flow-map must be a JSON array, got %s" % type(flow_map).__name__)
            sys.exit(1)
        for i, entry in enumerate(flow_map):
            if not isinstance(entry, dict) or 'setflag_pattern' not in entry or 'waitflag_pattern' not in entry:
                print("[ERROR] flow-map entry %d missing required keys 'setflag_pattern' and/or 'waitflag_pattern'" % i)
                sys.exit(1)
        print("[INFO] Loaded flow-map: %d entries from %s" % (len(flow_map), args.flow_map))

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[ERROR] CSV file not found: %s" % csv_path)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("[ERROR] CSV file is empty: %s" % csv_path)
        sys.exit(1)
    if df.empty:
        print("[ERROR] CSV file is empty: %s" % csv_path)
        sys.exit(1)
    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        print("[ERROR] CSV missing required columns: %s" % missing_cols)
        sys.exit(1)

    # Coerce numeric columns
    for col in ['rank', 'core_id', 'local_id', 'begin_cycle', 'end_cycle', 'duration_cycles', 'payload']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows with NaN in critical columns
    critical = ['rank', 'begin_cycle', 'end_cycle', 'duration_cycles']
    nan_count = df[critical].isna().any(axis=1).sum()
    if nan_count > 0:
        print("[WARN] Dropped %d rows with non-numeric values in critical columns" % nan_count)
        df = df.dropna(subset=critical)
    if df.empty:
        print("[ERROR] No valid rows remaining after numeric coercion")
        sys.exit(1)

    if not args.all_rounds:
        df, n_rounds, orig_count = aggregate_median(df)
        if n_rounds > 1:
            print("[INFO] Aggregated %d rounds → median per event (%d rows → %d). "
                  "Use --all-rounds to emit every round." % (n_rounds, orig_count, len(df)))

    if args.align_ranks:
        # In prof_paired.csv, event names have _BEGIN stripped by parse_prof.py's pair_events().
        # So "KERNEL_BEGIN" becomes "KERNEL" (or "EVT_KERNEL_BEGIN" becomes "EVT_KERNEL").
        kernel_begin = df[df['event'].str.contains('KERNEL', na=False)]
        if len(kernel_begin) > 0:
            # Use rank 0's first KERNEL as reference
            ref_rank = kernel_begin['rank'].astype(int).min()
            ref_ts = kernel_begin[kernel_begin['rank'].astype(int) == ref_rank]['begin_cycle'].min()
            # Compute offset per rank (in cycles, before conversion)
            rank_offsets = kernel_begin.groupby(kernel_begin['rank'].astype(int))['begin_cycle'].min() - ref_ts
            # Apply offsets
            df['begin_cycle'] = df.apply(
                lambda row: row['begin_cycle'] - rank_offsets.get(int(row['rank']), 0), axis=1)
            df['end_cycle'] = df.apply(
                lambda row: row['end_cycle'] - rank_offsets.get(int(row['rank']), 0), axis=1)
            print(f"[INFO] Aligned {len(rank_offsets)} ranks to rank {ref_rank} reference")
        else:
            print("[WARN] --align-ranks: no KERNEL events found, skipping alignment")

    ranks = sorted(set(df['rank'].astype(str).unique()), key=lambda x: int(x))
    print("[INFO] Processing %d rank(s) from %s" % (len(ranks), csv_path))
    for rk in ranks:
        out_path = os.path.join(out_dir, 'trace_rank%s.json' % rk)
        convert(df, out_path, args.freq, args.aic_cores, flow_map, rank=rk, compact=args.compact)


if __name__ == '__main__':
    main()
