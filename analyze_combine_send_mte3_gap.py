#!/usr/bin/env python3
"""Analyze send-side issue gaps and possible MTE3 starvation.

This script only consumes profiler CSV files.  It does not add device markers
or synchronization, so it does not change the kernel being measured.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_int(row, name):
    value = str(row.get(name, "")).strip()
    return int(value) if value else None


def read_send_tokens(path):
    lanes = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            start_cycle = read_int(row, "start_cycle")
            end_cycle = read_int(row, "end_cycle")
            if start_cycle is None or end_cycle is None:
                continue
            lane = (int(row["rank"]), int(row["core_id"]))
            row["_start_cycle"] = start_cycle
            row["_end_cycle"] = end_cycle
            send_order = read_int(row, "send_order")
            row["_send_order"] = send_order if send_order is not None else len(lanes[lane])
            lanes[lane].append(row)
    for rows in lanes.values():
        rows.sort(key=lambda row: (row["_send_order"], row["_start_cycle"]))
    return lanes


def read_stage_end(path):
    stages = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return stages
    with path.open("r", newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            stage = str(row.get("stage", "")).strip()
            if stage not in ("TRANSFER_TOKENS", "SEND_TOKENS"):
                continue
            begin_cycle = read_int(row, "begin_cycle")
            end_cycle = read_int(row, "end_cycle")
            if begin_cycle is None or end_cycle is None:
                continue
            lane = (int(row["rank"]), int(row["core_id"]))
            stages[lane][stage].append((begin_cycle, end_cycle))
    return stages


def stage_range(stage_rows):
    if not stage_rows:
        return None, None
    return min(row[0] for row in stage_rows), max(row[1] for row in stage_rows)


def analyze_lane(rank, core_id, rows, stages, cycles_per_us):
    first_start = rows[0]["_start_cycle"]
    last_end = max(row["_end_cycle"] for row in rows)
    issue_busy = sum(max(0, row["_end_cycle"] - row["_start_cycle"]) for row in rows)
    gaps = []
    for previous, current in zip(rows, rows[1:]):
        gap = max(0, current["_start_cycle"] - previous["_end_cycle"])
        gaps.append((gap, current))

    issue_gap = sum(gap for gap, _ in gaps)
    max_gap, max_gap_row = max(gaps, key=lambda item: item[0], default=(0, rows[0]))
    issue_span = max(0, last_end - first_start)

    transfer_begin, transfer_end = stage_range(stages.get("TRANSFER_TOKENS", []))
    send_begin, send_end = stage_range(stages.get("SEND_TOKENS", []))
    drain_tail = None
    if transfer_end is not None and send_end is not None:
        drain_tail = max(0, send_end - transfer_end)

    return {
        "rank": rank,
        "core_id": core_id,
        "send_count": len(rows),
        "first_send_start_cycle": first_start,
        "last_send_issue_end_cycle": last_end,
        "issue_span_cycles": issue_span,
        "issue_busy_cycles": issue_busy,
        "issue_gap_cycles": issue_gap,
        "issue_gap_us": issue_gap / cycles_per_us,
        "max_issue_gap_cycles": max_gap,
        "max_issue_gap_us": max_gap / cycles_per_us,
        "max_gap_send_order": max_gap_row.get("send_order", ""),
        "max_gap_token_id": max_gap_row.get("token_id", ""),
        "max_gap_topk_id": max_gap_row.get("topk_id", ""),
        "issue_busy_ratio": issue_busy / issue_span if issue_span else 0.0,
        "transfer_end_cycle": transfer_end if transfer_end is not None else "",
        "send_tokens_end_cycle": send_end if send_end is not None else "",
        "drain_tail_cycles": drain_tail if drain_tail is not None else "",
        "drain_tail_us": drain_tail / cycles_per_us if drain_tail is not None else "",
    }


def add_status(rows, gap_threshold_cycles, drain_threshold_cycles):
    for row in rows:
        has_gap = (
            row["issue_gap_cycles"] >= gap_threshold_cycles
            or row["max_issue_gap_cycles"] >= gap_threshold_cycles
        )
        drain_tail = row["drain_tail_cycles"]
        if drain_tail == "":
            status = "ISSUE_GAP_ONLY"
        elif has_gap and drain_tail <= drain_threshold_cycles:
            status = "MTE3_HUNGRY_CANDIDATE"
        elif has_gap:
            status = "ISSUE_GAP_BUT_MTE3_BACKLOG"
        elif drain_tail > drain_threshold_cycles:
            status = "MTE3_BACKLOG"
        else:
            status = "NO_OBVIOUS_GAP"
        row["status"] = status


def make_rank_summary(rows, cycles_per_us):
    by_rank = defaultdict(list)
    for row in rows:
        by_rank[row["rank"]].append(row)
    summaries = []
    for rank, rank_rows in sorted(by_rank.items()):
        drain_rows = [row for row in rank_rows if row["drain_tail_cycles"] != ""]
        candidate_count = sum(row["status"] == "MTE3_HUNGRY_CANDIDATE" for row in rank_rows)
        summaries.append({
            "rank": rank,
            "send_core_count": len(rank_rows),
            "first_send_start_cycle": min(row["first_send_start_cycle"] for row in rank_rows),
            "send_tokens_end_cycle": max(
                (row["send_tokens_end_cycle"] for row in rank_rows if row["send_tokens_end_cycle"] != ""),
                default="",
            ),
            "send_span_cycles": (
                max(row["send_tokens_end_cycle"] for row in rank_rows if row["send_tokens_end_cycle"] != "")
                - min(row["first_send_start_cycle"] for row in rank_rows)
                if any(row["send_tokens_end_cycle"] != "" for row in rank_rows) else ""
            ),
            "send_span_us": (
                (
                    max(row["send_tokens_end_cycle"] for row in rank_rows if row["send_tokens_end_cycle"] != "")
                    - min(row["first_send_start_cycle"] for row in rank_rows)
                ) / cycles_per_us
                if any(row["send_tokens_end_cycle"] != "" for row in rank_rows) else ""
            ),
            "max_issue_gap_cycles": max(row["max_issue_gap_cycles"] for row in rank_rows),
            "max_issue_gap_us": max(row["max_issue_gap_cycles"] for row in rank_rows) / cycles_per_us,
            "max_issue_gap_sum_cycles": max(row["issue_gap_cycles"] for row in rank_rows),
            "max_issue_gap_sum_us": max(row["issue_gap_cycles"] for row in rank_rows) / cycles_per_us,
            "max_drain_tail_cycles": max(
                (row["drain_tail_cycles"] for row in drain_rows), default=""
            ),
            "max_drain_tail_us": (
                max(row["drain_tail_cycles"] for row in drain_rows) / cycles_per_us
                if drain_rows else ""
            ),
            "mte3_hungry_candidate_cores": candidate_count,
        })
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="profiler result directory")
    parser.add_argument(
        "--send-csv", default="fullmesh_v2_send_token_rounds.csv",
        help="send token CSV name in result_dir",
    )
    parser.add_argument(
        "--stages-csv", default="fullmesh_v2_stages.csv",
        help="stage CSV name in result_dir",
    )
    parser.add_argument(
        "--cycles-per-us", type=float, default=1000.0,
        help="device cycles per microsecond (default: 1000)",
    )
    parser.add_argument(
        "--gap-threshold-us", type=float, default=0.1,
        help="gap threshold for a candidate (default: 0.1 us)",
    )
    parser.add_argument(
        "--drain-threshold-us", type=float, default=0.1,
        help="tail threshold used to identify MTE3 backlog (default: 0.1 us)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="output directory (default: result_dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.result_dir
    send_path = args.result_dir / args.send_csv
    stages_path = args.result_dir / args.stages_csv
    if not send_path.exists():
        raise SystemExit(f"send CSV not found: {send_path}")

    send_lanes = read_send_tokens(send_path)
    stage_lanes = read_stage_end(stages_path)
    rows = [
        analyze_lane(rank, core_id, lane_rows, stage_lanes[(rank, core_id)], args.cycles_per_us)
        for (rank, core_id), lane_rows in sorted(send_lanes.items())
    ]
    add_status(
        rows,
        args.gap_threshold_us * args.cycles_per_us,
        args.drain_threshold_us * args.cycles_per_us,
    )
    rank_rows = make_rank_summary(rows, args.cycles_per_us)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "send_mte3_gap_core_summary.csv", rows)
    write_csv(output_dir / "send_mte3_gap_rank_summary.csv", rank_rows)

    print("send-side gap summary")
    for row in sorted(rows, key=lambda item: item["issue_gap_cycles"], reverse=True)[:10]:
        print(
            f"rank={row['rank']} core={row['core_id']} "
            f"gap={row['issue_gap_us']:.3f}us max={row['max_issue_gap_us']:.3f}us "
            f"drain={row['drain_tail_us'] if row['drain_tail_us'] != '' else 'NA'}us "
            f"status={row['status']}"
        )
    print(f"core summary: {output_dir / 'send_mte3_gap_core_summary.csv'}")
    print(f"rank summary: {output_dir / 'send_mte3_gap_rank_summary.csv'}")
    end_cycles = [row["send_tokens_end_cycle"] for row in rows if row["send_tokens_end_cycle"] != ""]
    if end_cycles:
        first_start = min(row["first_send_start_cycle"] for row in rows)
        last_end = max(end_cycles)
        print(
            f"global_send_span={last_end - first_start} cycles "
            f"({(last_end - first_start) / args.cycles_per_us:.3f} us)"
        )
    print(
        "MTE3_HUNGRY_CANDIDATE is only a candidate: these CSV markers cannot "
        "prove that the MTE3 pipe was idle inside every issue gap."
    )


if __name__ == "__main__":
    main()
