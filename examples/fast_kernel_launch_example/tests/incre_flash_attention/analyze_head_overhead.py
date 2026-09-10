#!/usr/bin/env python3
import argparse
import csv
import sqlite3
import statistics
import struct
from pathlib import Path


STARS_RECORD_SIZE_A3 = 64
STARS_RECORD_A3 = struct.Struct("<IHHQIIII")
STARS_RECORD_A5 = struct.Struct("<IIQIIII")
STARS_DISPATCH = 0
FFTS_RECORD_A3 = struct.Struct("<HHHHQIIII12Q")
FFTS_TASK_PMU = 40
FFTS_BLOCK_PMU = 41
FFTS_MIX_PMU_A5 = 42


def find_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one {pattern!r} under {root}, found {len(paths)}"
        )
    return paths[0]


def load_ifa_task_keys(profile_dir: Path):
    db_path = find_one(profile_dir, "device_*/sqlite/ai_core_op_summary.db")
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT stream_id, task_id
            FROM ge_summary
            WHERE lower(op_name) LIKE '%incre_flash_attention%'
            ORDER BY rowid
            """
        ).fetchall()
    if not rows:
        raise RuntimeError(f"No incre_flash_attention tasks found in {db_path}")
    return {(int(stream_id), int(task_id)) for stream_id, task_id in rows}


def load_stars_dispatches(profile_dir: Path, ifa_task_keys):
    data_dir = find_one(profile_dir, "device_*/data")
    data_files = sorted(
        path
        for path in data_dir.glob("stars_soc.data.*")
        if not path.name.endswith((".done", ".complete"))
    )
    if not data_files:
        raise RuntimeError(f"No stars_soc.data files found under {data_dir}")

    dispatches = []
    for data_file in data_files:
        content = data_file.read_bytes()
        if len(content) % STARS_RECORD_SIZE_A3:
            raise RuntimeError(
                f"{data_file} size {len(content)} is not aligned to "
                f"{STARS_RECORD_SIZE_A3}-byte A3 STARS records"
            )
        for offset in range(0, len(content), STARS_RECORD_SIZE_A3):
            fields = STARS_RECORD_A3.unpack_from(content, offset)
            header, task_id_raw, stream_id, sys_cycle = fields[:4]
            func_type = header & 0x3F
            task_id = task_id_raw & 0x7FFF
            if (
                func_type == STARS_DISPATCH
                and (stream_id, task_id) in ifa_task_keys
            ):
                dispatches.append(
                    {
                        "stream_id": stream_id,
                        "task_id": task_id,
                        "task_id_raw": task_id_raw,
                        "stars_dispatch_cycle": sys_cycle,
                    }
                )
    if not dispatches:
        task_streams = {}
        for stream_id, task_id in ifa_task_keys:
            if task_id in task_streams and task_streams[task_id] != stream_id:
                raise RuntimeError(
                    f"IFA task ID {task_id} appears on multiple streams"
                )
            task_streams[task_id] = stream_id
        for data_file in data_files:
            content = data_file.read_bytes()
            if len(content) % STARS_RECORD_A5.size:
                raise RuntimeError(
                    f"{data_file} size {len(content)} is not aligned to "
                    f"{STARS_RECORD_A5.size}-byte A5 STARS records"
                )
            for offset in range(0, len(content), STARS_RECORD_A5.size):
                header, task_id, sys_cycle, _, _, _, _ = (
                    STARS_RECORD_A5.unpack_from(content, offset)
                )
                if (header & 0x3F) == STARS_DISPATCH and task_id in task_streams:
                    dispatches.append(
                        {
                            "stream_id": task_streams[task_id],
                            "task_id": task_id,
                            "task_id_raw": task_id,
                            "stars_dispatch_cycle": sys_cycle,
                        }
                    )

    dispatches.sort(key=lambda row: row["stars_dispatch_cycle"])
    if not dispatches:
        raise RuntimeError("No STARS dispatch records matched IFA tasks")
    return dispatches


def load_ffts_kicks(profile_dir: Path, ifa_task_keys):
    data_dir = find_one(profile_dir, "device_*/data")
    data_files = sorted(
        path
        for path in data_dir.glob("ffts_profile.data.*")
        if not path.name.endswith((".done", ".complete"))
    )
    if not data_files:
        raise RuntimeError(f"No ffts_profile.data files found under {data_dir}")

    task_streams = {task_id: stream_id for stream_id, task_id in ifa_task_keys}
    starts = {}
    for data_file in data_files:
        content = data_file.read_bytes()
        if len(content) % FFTS_RECORD_A3.size:
            raise RuntimeError(
                f"{data_file} size {len(content)} is not aligned to "
                f"{FFTS_RECORD_A3.size}-byte A3 FFTS records"
            )
        for offset in range(0, len(content), FFTS_RECORD_A3.size):
            fields = FFTS_RECORD_A3.unpack_from(content, offset)
            func_type = fields[0] & 0x3F
            task_id_raw = fields[2]
            stream_id = fields[3]
            task_id = task_id_raw & 0x7FFF
            pmu_start = fields[19]
            key = (stream_id, task_id)
            if key not in ifa_task_keys and task_id in task_streams:
                key = (task_streams[task_id], task_id)
            if (
                func_type in {FFTS_TASK_PMU, FFTS_BLOCK_PMU, FFTS_MIX_PMU_A5}
                and key in ifa_task_keys
                and pmu_start > 0
            ):
                starts.setdefault(key, set()).add(pmu_start)

    kicks = {}
    for key, values in starts.items():
        # A5 emits separate AIC/AIV PMU records whose starts can differ
        # by a few cycles. The first PMU start is the task kick boundary.
        kicks[key] = min(values)
    if not kicks:
        raise RuntimeError("No FFTS pmu_start records matched IFA tasks")
    return kicks


def load_timing_launches(timing_csv: Path):
    with timing_csv.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    launches = {}
    order = []
    for row in rows:
        key = (row["phase"], int(row["launch_index"]))
        if key not in launches:
            launches[key] = []
            order.append(key)
        launches[key].append(row)
    return [(key, launches[key]) for key in order]


def align_launches(dispatches, timing_launches):
    if len(dispatches) == len(timing_launches):
        return list(zip(dispatches, timing_launches))
    replay_launches = [
        launch for launch in timing_launches if launch[0][0] == "replay"
    ]
    if len(dispatches) == len(replay_launches):
        return list(zip(dispatches, replay_launches))
    # 多拍模式 (对齐 add/gmm): 每拍 replay 完整图 (N 节点), cycle 记录只反映
    # 本拍最后一次 launch -> 该拍对齐最后一条 dispatch
    # test.py warmup (capture 外 1 次发射) 会多 1 条 dispatch: 余 1 时剔除首条再对齐
    for extra in (0, 1):
        trimmed = dispatches[extra:] if extra and len(dispatches) % len(
            timing_launches or [1]) == extra else dispatches
        for launches in (timing_launches, replay_launches):
            n = len(launches)
            if n and len(trimmed) % n == 0:
                per = len(trimmed) // n
                return [
                    (trimmed[(i + 1) * per - 1], launch)
                    for i, launch in enumerate(launches)
                ]
    raise RuntimeError(
        "Cannot align launches: "
        f"{len(dispatches)} IFA STARS dispatches, "
        f"{len(timing_launches)} timing launches "
        f"({len(replay_launches)} replay launches)"
    )


def write_merged_rows(aligned, kicks, output_csv: Path, syscnt_mhz: float):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "measurement_point",
        "phase",
        "launch_index",
        "stream_id",
        "task_id",
        "task_id_raw",
        "vector_block",
        "ai_core",
        "stars_dispatch_cycle",
        "aicore_kick_cycle",
        "kernel_entry_cycle",
        "before_first_mte2_cycle",
        "kernel_end_cycle",
        "stars_to_kick_cycles",
        "kick_to_kernel_entry_cycles",
        "schedule_hardware_cycles",
        "software_init_cycles",
        "t2_from_stars_cycles",
        "kernel_end_from_stars_cycles",
        "stars_to_kick_us",
        "kick_to_kernel_entry_us",
        "schedule_hardware_us",
        "software_init_us",
        "t2_from_stars_us",
        "kernel_end_from_stars_us",
    ]
    merged = []
    for dispatch, ((phase, launch_index), timing_rows) in aligned:
        task_key = (dispatch["stream_id"], dispatch["task_id"])
        if task_key not in kicks:
            raise RuntimeError(f"No FFTS TKick matched IFA task {task_key}")
        tkick = kicks[task_key]
        for timing in timing_rows:
            t0 = dispatch["stars_dispatch_cycle"]
            t1 = int(timing["kernel_entry_cycle"])
            t2_text = timing["before_first_mte2_cycle"]
            t2 = int(t2_text) if t2_text else None
            # T2 (kernel 末打点, slot 7); 老数据无此列为空
            kend_text = timing.get("kernel_end_cycle", "")
            kend = int(kend_text) if kend_text else None
            if t1 < t0:
                raise RuntimeError(
                    f"Kernel entry precedes STARS dispatch for "
                    f"stream={dispatch['stream_id']} task={dispatch['task_id']}"
                )
            # A3: FFTS f[19] (pmu_start) is not the true kick moment
            # (measured as dispatch+20cyc), so TKick may precede T0.
            # Skip TKick fields when order is invalid; they are not used
            # in the final summary (schedule_hardware/software_init/t2_from_stars
            # all derive from T0/T1/T2 directly).
            valid_kick = t0 <= tkick <= t1
            stars_to_kick_cycles = (tkick - t0) if valid_kick else ""
            kick_to_kernel_entry_cycles = (t1 - tkick) if valid_kick else ""
            schedule_hardware_cycles = t1 - t0
            software_init_cycles = t2 - t1 if t2 is not None else None
            t2_from_stars_cycles = t2 - t0 if t2 is not None else None
            merged.append(
                {
                    "measurement_point": timing.get("measurement_point", ""),
                    "phase": phase,
                    "launch_index": launch_index,
                    "stream_id": dispatch["stream_id"],
                    "task_id": dispatch["task_id"],
                    "task_id_raw": dispatch["task_id_raw"],
                    "vector_block": timing["vector_block"],
                    "ai_core": timing["ai_core"],
                    "stars_dispatch_cycle": t0,
                    "aicore_kick_cycle": tkick,
                    "kernel_entry_cycle": t1,
                    "before_first_mte2_cycle": t2 if t2 is not None else "",
                    "kernel_end_cycle": kend if kend is not None else "",
                    "stars_to_kick_cycles": stars_to_kick_cycles,
                    "kick_to_kernel_entry_cycles": kick_to_kernel_entry_cycles,
                    "schedule_hardware_cycles": schedule_hardware_cycles,
                    "software_init_cycles": (
                        software_init_cycles
                        if software_init_cycles is not None
                        else ""
                    ),
                    "t2_from_stars_cycles": (
                        t2_from_stars_cycles
                        if t2_from_stars_cycles is not None
                        else ""
                    ),
                    "kernel_end_from_stars_cycles": (
                        kend - t0 if kend is not None else ""
                    ),
                    "stars_to_kick_us": (
                        stars_to_kick_cycles / syscnt_mhz
                        if stars_to_kick_cycles != ""
                        else ""
                    ),
                    "kick_to_kernel_entry_us": (
                        kick_to_kernel_entry_cycles / syscnt_mhz
                        if kick_to_kernel_entry_cycles != ""
                        else ""
                    ),
                    "schedule_hardware_us": schedule_hardware_cycles / syscnt_mhz,
                    "software_init_us": (
                        software_init_cycles / syscnt_mhz
                        if software_init_cycles is not None
                        else ""
                    ),
                    "t2_from_stars_us": (
                        t2_from_stars_cycles / syscnt_mhz
                        if t2_from_stars_cycles is not None
                        else ""
                    ),
                    "kernel_end_from_stars_us": (
                        (kend - t0) / syscnt_mhz if kend is not None else ""
                    ),
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    return merged


def summarize(values):
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def report_summary(rows, warmup: int, syscnt_mhz: float):
    stable = [
        row
        for row in rows
        if row["phase"] == "replay" and int(row["launch_index"]) >= warmup
    ]
    unique_tasks = {}
    for row in stable:
        key = (row["stream_id"], row["task_id"])
        unique_tasks[key] = row

    metrics = {
        "TKick-T0_stars_scheduling": summarize(
            [
                row["stars_to_kick_cycles"]
                for row in unique_tasks.values()
                if row["stars_to_kick_cycles"] != ""
            ]
        ),
        "T1-TKick_hardware_init": summarize(
            [
                row["kick_to_kernel_entry_cycles"]
                for row in stable
                if row["kick_to_kernel_entry_cycles"] != ""
            ]
        ),
        "T1-T0_schedule_plus_hardware": summarize(
            [row["schedule_hardware_cycles"] for row in stable]
        ),
        "T2-T1_software_init": summarize(
            [
                row["software_init_cycles"]
                for row in stable
                if row["software_init_cycles"] != ""
            ]
        ),
        "T2-T0_total_head": summarize(
            [
                row["t2_from_stars_cycles"]
                for row in stable
                if row["t2_from_stars_cycles"] != ""
            ]
        ),
        "TKend-T1_kernel_main": summarize(
            [
                int(row["kernel_end_cycle"]) - int(row["kernel_entry_cycle"])
                for row in stable
                if row.get("kernel_end_cycle", "") != ""
            ]
        ),
        "TKend-stars_kernel_end": summarize(
            [
                row["kernel_end_from_stars_cycles"]
                for row in stable
                if row.get("kernel_end_from_stars_cycles", "") != ""
            ]
        ),
    }
    all_t2_stars = [
        row["t2_from_stars_cycles"]
        for row in rows
        if row["t2_from_stars_cycles"] != ""
    ]
    if all_t2_stars:
        metrics["T2-T0_total_head_all_samples"] = summarize(all_t2_stars)

    print(
        f"stable replay rows={len(stable)}, "
        f"stable tasks={len(unique_tasks)}"
    )
    for name, stats in metrics.items():
        if stats is None:
            print(f"{name}: not collected")
            continue
        print(
            f"{name}: n={stats['n']} "
            f"min={stats['min']:.0f} "
            f"mean={stats['mean']:.2f} "
            f"median={stats['median']:.0f} "
            f"max={stats['max']:.0f} cycles "
            f"(max={stats['max'] / syscnt_mhz:.3f} us)"
        )
    if all_t2_stars:
        print(
            "T2-T0 all samples including warmup: "
            f"n={len(all_t2_stars)} max={max(all_t2_stars)} cycles "
            f"({max(all_t2_stars) / syscnt_mhz:.3f} us)"
        )
    return metrics


def write_summary(metrics, output_csv: Path, syscnt_mhz: float):
    fieldnames = [
        "metric",
        "samples",
        "min_cycles",
        "mean_cycles",
        "median_cycles",
        "max_cycles",
        "min_us",
        "mean_us",
        "median_us",
        "max_us",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for metric, stats in metrics.items():
            if stats is None:
                continue
            writer.writerow(
                {
                    "metric": metric,
                    "samples": stats["n"],
                    "min_cycles": stats["min"],
                    "mean_cycles": stats["mean"],
                    "median_cycles": stats["median"],
                    "max_cycles": stats["max"],
                    "min_us": stats["min"] / syscnt_mhz,
                    "mean_us": stats["mean"] / syscnt_mhz,
                    "median_us": stats["median"] / syscnt_mhz,
                    "max_us": stats["max"] / syscnt_mhz,
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Correlate A3 STARS dispatch records with IFA T1/T2 timestamps"
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument(
        "--timing-csv", type=Path, default=Path("ifa_head_timing_raw.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("ifa_head_overhead.csv"))
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--syscnt-mhz",
        type=float,
        default=50.0,
        help="A3 SYS_CNT frequency; existing profiler data validates 50 MHz",
    )
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    task_keys = load_ifa_task_keys(args.profile_dir)
    dispatches = load_stars_dispatches(args.profile_dir, task_keys)
    kicks = load_ffts_kicks(args.profile_dir, task_keys)
    timing_launches = load_timing_launches(args.timing_csv)
    aligned = align_launches(dispatches, timing_launches)
    merged = write_merged_rows(aligned, kicks, args.output, args.syscnt_mhz)
    metrics = report_summary(merged, args.warmup, args.syscnt_mhz)
    summary_output = (
        args.summary_output
        if args.summary_output is not None
        else args.output.with_name("ifa_head_summary.csv")
    )
    write_summary(metrics, summary_output, args.syscnt_mhz)
    print(f"Merged raw timestamps written to {args.output.resolve()}")
    print(f"Summary written to {summary_output.resolve()}")


if __name__ == "__main__":
    main()
