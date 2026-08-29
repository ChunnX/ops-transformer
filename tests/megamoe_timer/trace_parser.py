import argparse
import csv
import json
from pathlib import Path

from timer_path import resolve_timer_dir


CYCLES_PER_US = 1000.0
DISPLAYED_STAGES = (
    "AIV_DISPATCH",
    "AIC_GMM1",
    "AIV_SWIGLU",
    "AIC_GMM2",
    "AIV_COMBINE",
    "AIV_UNPERMUTE",
)


def stage_name(event_name: str) -> str:
    """Remove the dynamic iteration suffix from a timer event name."""
    prefix, separator, iteration = event_name.rpartition("_")
    if separator and iteration.isdigit():
        return prefix
    return event_name


def load_rank_data(path: Path, rank: int):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [], {}, []
    event_names = [
        name.removesuffix("_start_cycle")
        for name in rows[0]
        if name.endswith("_start_cycle")
    ]
    valid_starts = [
        int(row[f"{name}_start_cycle"])
        for row in rows
        for name in event_names
        if int(row[f"{name}_start_cycle"]) > 0
        and int(row[f"{name}_end_cycle"]) > int(row[f"{name}_start_cycle"])
    ]
    base = min(valid_starts, default=0)
    events = []
    stage_ranges = {}
    stage_order = []
    for row in rows:
        for name in event_names:
            start = int(row[f"{name}_start_cycle"])
            end = int(row[f"{name}_end_cycle"])
            if start <= 0 or end <= start:
                continue
            stage = stage_name(name)
            if stage not in stage_ranges:
                stage_ranges[stage] = [start, end]
                stage_order.append(stage)
            else:
                stage_ranges[stage][0] = min(stage_ranges[stage][0], start)
                stage_ranges[stage][1] = max(stage_ranges[stage][1], end)
            events.append({
                "ph": "X",
                "cat": "aic_op" if row["core_type"] == "AIC" else "aiv_op",
                "pid": f"rank{rank}",
                "tid": f"group_{row['group_id']}_{row['core_type']}_{row['sub_id']}",
                "name": name,
                "ts": (start - base) / CYCLES_PER_US,
                "dur": (end - start) / CYCLES_PER_US,
            })
    stage_durations = {
        stage: (end - start) / CYCLES_PER_US
        for stage, (start, end) in stage_ranges.items()
    }
    return events, stage_durations, stage_order


def print_stage_durations(rank_durations, stage_order):
    available_stages = set(stage_order)
    displayed_stages = [
        stage for stage in DISPLAYED_STAGES if stage in available_stages
    ]
    if not rank_durations or not displayed_stages:
        print("[WARN] no valid stage timing data")
        return

    ranks = sorted(rank_durations)
    stage_width = max(len("Stage"), *(len(stage) for stage in displayed_stages))
    duration_width = len("Duration (us)")

    def print_row(stage, duration):
        print(f"{stage:<{stage_width}}  {duration:>{duration_width}}")

    print("\nStage elapsed time (earliest start to latest end across all cores):")
    for rank in ranks:
        print(f"\nRank {rank} / Card {rank}:")
        print_row("Stage", "Duration (us)")
        print_row("-" * stage_width, "-" * duration_width)
        for stage in displayed_stages:
            duration = rank_durations[rank].get(stage)
            print_row(stage, "-" if duration is None else f"{duration:.3f}")

    print("\nAverage across cards:")
    print_row("Stage", "Duration (us)")
    print_row("-" * stage_width, "-" * duration_width)
    for stage in displayed_stages:
        values = [rank_durations[rank].get(stage) for rank in ranks]
        valid_values = [value for value in values if value is not None]
        average = sum(valid_values) / len(valid_values)
        print_row(stage, f"{average:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Create Chrome Trace JSON from parsed MegaMoe timers")
    parser.add_argument(
        "--timer-dir",
        default=None,
        help=(
            "Timer case directory or timer root. By default, use "
            "tests/megamoe_w4a8/timer and select its latest parsed case."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to all_rank_trace.json in the selected timer case.",
    )
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, default=7)
    args = parser.parse_args()
    timer_dir = resolve_timer_dir(args.timer_dir, "rank_*_npu_time_parsed.csv")
    print(f"[INFO] timer directory: {timer_dir}")
    events = []
    rank_durations = {}
    stage_order = []
    for rank in range(args.rank_start, args.rank_end + 1):
        path = timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        if path.exists():
            rank_events, durations, rank_stage_order = load_rank_data(path, rank)
            events.extend(rank_events)
            rank_durations[rank] = durations
            for stage in rank_stage_order:
                if stage not in stage_order:
                    stage_order.append(stage)
    output = Path(args.output) if args.output else timer_dir / "all_rank_trace.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {len(events)} events -> {output}")
    print_stage_durations(rank_durations, stage_order)


if __name__ == "__main__":
    main()
