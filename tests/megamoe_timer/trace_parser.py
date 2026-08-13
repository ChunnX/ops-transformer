import argparse
import csv
import json
from pathlib import Path

from timer_path import resolve_timer_dir


CYCLES_PER_US = 1000.0


def load_events(path: Path, rank: int):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
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
    for row in rows:
        for name in event_names:
            start = int(row[f"{name}_start_cycle"])
            end = int(row[f"{name}_end_cycle"])
            if start <= 0 or end <= start:
                continue
            events.append({
                "ph": "X",
                "cat": "aic_op" if row["core_type"] == "AIC" else "aiv_op",
                "pid": f"rank{rank}",
                "tid": f"group_{row['group_id']}_{row['core_type']}_{row['sub_id']}",
                "name": name,
                "ts": (start - base) / CYCLES_PER_US,
                "dur": (end - start) / CYCLES_PER_US,
            })
    return events


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
    for rank in range(args.rank_start, args.rank_end + 1):
        path = timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        if path.exists():
            events.extend(load_events(path, rank))
    output = Path(args.output) if args.output else timer_dir / "all_rank_trace.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {len(events)} events -> {output}")


if __name__ == "__main__":
    main()
