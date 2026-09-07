# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate and aggregate ordinary msprof CSV from paired Sage/FIA applications.

Primary timing is the sum of device task durations within each API call. Also
report first-task-start to last-task-end spans, including internal launch gaps.
Never add per-stage medians or include host/event durations in the primary sum.
"""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
import math
from pathlib import Path
import statistics


STAGES = ("Stats", "Quant", "Core")


def nanoseconds(value):
    return int(Decimal(value.strip()) * 1000)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {
        "count": len(values),
        "min": min(values),
        "p10": percentile(values, 0.1),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def stage(row):
    for name in STAGES:
        if row["OP Type"].startswith("SageAttention2" + name):
            return name
    if row["OP Type"].startswith("FusedInferAttentionScore"):
        return "FIA"
    return None


def task_key(row, summary=True):
    return (row["Device_id"], row["Stream ID" if summary else "stream_id"], row["Task ID" if summary else "task_id"])


def single_file(directory, pattern):
    files = list(directory.glob("PROF_*/mindstudio_profiler_output/" + pattern))
    if len(files) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(files)}")
    return files[0]


def parse_case(directory):
    metadata = json.loads((directory / "application.json").read_text())
    if not metadata.get("profile_only"):
        raise ValueError(f"Not a profiling application: {directory}")
    warmup, repeats = metadata["warmup"], metadata["repeats"]
    stages = tuple(metadata.get("operator_stages", STAGES))
    if stages != STAGES:
        raise ValueError(f"Unknown Sage pipeline: {stages}")
    total = warmup + repeats
    summary_file = single_file(directory, "op_summary*.csv")
    task_file = single_file(directory, "task_time*.csv")
    with summary_file.open() as stream:
        all_ops = list(csv.DictReader(stream))
    with task_file.open() as stream:
        all_tasks = list(csv.DictReader(stream))
    ops = sorted((r for r in all_ops if stage(r)), key=lambda r: nanoseconds(r["Task Start Time(us)"]))
    counts = Counter(stage(r) for r in ops)
    if any(counts[name] != (total if name in stages else 0) for name in STAGES):
        raise ValueError(f"Missing/extra Sage stage calls in {directory}: {counts}")
    if counts["FIA"] == 0 or counts["FIA"] % total:
        raise ValueError(f"Incomplete FIA calls in {directory}: {counts}")
    fia_tasks = counts["FIA"] // total
    task_map = {task_key(r, False): r for r in all_tasks}
    if len(task_map) != len(all_tasks):
        raise ValueError("Duplicate task IDs: timestamp-aware correlation is required")
    for row in ops:
        raw = task_map[task_key(row)]
        duration = nanoseconds(row["Task Duration(us)"])
        if duration <= 0 or duration != nanoseconds(raw["task_time(us)"]):
            raise ValueError(f"Invalid/mismatched device duration: {row}")

    first = nanoseconds(ops[0]["Task Start Time(us)"])
    last = max(nanoseconds(r["Task Start Time(us)"]) + nanoseconds(r["Task Duration(us)"]) for r in ops)
    unknown_ops = [r for r in all_ops if not stage(r) and first <= nanoseconds(r["Task Start Time(us)"]) < last]
    known_keys = {task_key(r) for r in ops}
    extra_tasks = [
        r
        for r in all_tasks
        if task_key(r, False) not in known_keys and first <= nanoseconds(r["task_start(us)"]) < last
    ]
    # torch.npu.synchronize itself records runtime events, even though the
    # application does not time events. Retain these in the audit, not the
    # operator kernel sum. Fail closed for auxiliary compute/copy tasks.
    sync_tasks = [r for r in extra_tasks if r["kernel_type"] == "DAVID_EVENT_RECORD"]
    unknown_tasks = [r for r in extra_tasks if r["kernel_type"] != "DAVID_EVENT_RECORD"]
    if unknown_ops or unknown_tasks:
        raise ValueError(f"Unattributed work in {directory}: ops={unknown_ops}, tasks={unknown_tasks}")

    calls = {"sage": [], "fia": []}
    cursor = 0
    mode = metadata.get("order", "sage-first")
    if mode not in ("sage-first", "fia-first"):
        raise ValueError(f"Expected sequential operator measurements, got {mode}")
    order = ("sage", "fia") if mode == "sage-first" else ("fia", "sage")
    schedule = [(operator, i, None) for operator in order for i in range(total)]
    for operator, iteration, position in schedule:
        expected = list(stages) if operator == "sage" else ["FIA"] * fia_tasks
        rows = ops[cursor : cursor + len(expected)]
        cursor += len(expected)
        if [stage(r) for r in rows] != expected:
            raise ValueError(f"Call ordering mismatch at call {iteration}: {directory}")
        streams = {(r["Device_id"], r["Stream ID"]) for r in rows}
        if len(streams) != 1:
            raise ValueError("Concurrent streams require interval-union timing")
        starts = [nanoseconds(r["Task Start Time(us)"]) for r in rows]
        durations = [nanoseconds(r["Task Duration(us)"]) for r in rows]
        ends = [s + d for s, d in zip(starts, durations)]
        if any(end > start for end, start in zip(ends, starts[1:])):
            raise ValueError("Overlapping tasks: a simple duration sum is invalid")
        summed = sum(durations)
        span = max(ends) - min(starts)
        calls[operator].append(
            {
                "call_index": iteration,
                "warmup": iteration < warmup,
                "order_in_pair": position,
                "task_sum_us": summed / 1000,
                "span_us": span / 1000,
                "internal_gap_us": (span - summed) / 1000,
                "stages_us": {
                    name: sum(d for r, d in zip(rows, durations) if stage(r) == name) / 1000
                    for name in dict.fromkeys(expected)
                },
                "tasks": [
                    {
                        "name": r["Op Name"],
                        "type": r["Task Type"],
                        "stream_id": r["Stream ID"],
                        "task_id": r["Task ID"],
                        "start_ns": start,
                        "duration_us": duration / 1000,
                        "cube_active_ratio": float(r.get("aic_mac_ratio", 0)),
                        "vector_active_ratio": float(r.get("aiv_vec_ratio", 0)),
                    }
                    for r, start, duration in zip(rows, starts, durations)
                ],
            }
        )
    if cursor != len(ops):
        raise ValueError("Unconsumed task rows")

    result = {
        "case": directory.name,
        "interface": directory.parent.name,
        "operator_stages": stages,
        "metadata": metadata,
        "csv": {"op_summary": str(summary_file), "task_time": str(task_file)},
        "audit": {
            "collected_counts": dict(counts),
            "fia_tasks_per_call": fia_tasks,
            "discarded_warmup_calls_per_operator": warmup,
            "measured_calls_per_operator": repeats,
            "unattributed_compute_or_copy_tasks": 0,
            "runtime_sync_event_tasks_excluded": len(sync_tasks),
            "runtime_sync_event_duration_us": sum(nanoseconds(r["task_time(us)"]) for r in sync_tasks) / 1000,
            "task_time_crosscheck": "passed",
        },
        "samples": calls,
    }
    summarize_calls(result)
    return result


def summarize_calls(result):
    calls = {operator: [s for s in samples if not s["warmup"]] for operator, samples in result["samples"].items()}
    for operator, measured in calls.items():
        result[operator] = {
            key: distribution([r[key] for r in measured]) for key in ("task_sum_us", "span_us", "internal_gap_us")
        }
        names = result["operator_stages"] if operator == "sage" else ("FIA",)
        result[operator]["stages_us"] = {name: distribution([r["stages_us"][name] for r in measured]) for name in names}
    sage = result["sage"]["task_sum_us"]["median"]
    fia = result["fia"]["task_sum_us"]["median"]
    result.update(
        sage_us=sage,
        fia_us=fia,
        speedup=fia / sage,
        span_speedup=result["fia"]["span_us"]["median"] / result["sage"]["span_us"]["median"],
        target_sage_us=fia / 1.6,
        required_reduction_us=max(0, sage - fia / 1.6),
        required_reduction_pct=max(0, 1 - fia / sage / 1.6) * 100,
        performance_gate_pass=fia / sage >= 1.6,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    results = {}
    for interface in ("pt", "aclnn"):
        cases = []
        for directory in sorted(
            p for p in (root / interface).glob("*") if p.is_dir() and p.name.startswith(("P-", "X-"))
        ):
            result = parse_case(directory)
            (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
            # Samples remain in each analysis.json; keep the aggregate compact.
            cases.append({k: v for k, v in result.items() if k != "samples"})
            print(
                f"{interface} {directory.name}: Sage={result['sage_us']:.3f} us "
                f"FIA={result['fia_us']:.3f} us speedup={result['speedup']:.4f} "
                f"need_reduce={result['required_reduction_pct']:.2f}%"
            )
        if cases:
            required = [r for r in cases if r["case"] in {f"P-{i:02}" for i in range(1, 8)}]
            results[interface] = {
                "complete": all(f"P-{i:02}" in {r["case"] for r in cases} for i in range(1, 8)),
                "cases": cases,
                "geometric_mean_speedup": math.exp(statistics.mean(math.log(r["speedup"]) for r in required))
                if required
                else None,
                "all_cases_pass": len(required) == 7 and all(r["performance_gate_pass"] for r in required),
                "extended_geometric_mean_speedup": math.exp(statistics.mean(math.log(r["speedup"]) for r in cases)),
            }
    (root / "summary.json").write_text(
        json.dumps(
            {
                "timing": "ordinary msprof CSV; median per-call device task duration sum, warmup excluded",
                "paths": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
