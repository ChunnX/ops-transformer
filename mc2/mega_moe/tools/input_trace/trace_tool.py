#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Decode MegaMoe input trace; timelines share a clock within each rank only."""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import replace
from pathlib import Path

from critical_path_report import ReportSchema, write_report_bundle, write_run_index

MAGIC = 0x4D495452
VERSION = 1
HEADER_WORDS = 16
RECORD_HEADER_WORDS = 8
EVENT_WORDS = 9
MAX_CAPACITY = 32
ROLES = ("AIC", "AIV0", "AIV1")
EVENTS = {
    1: "InputPrepareReturn",
    2: "BufferInitSubmit",
    3: "ResetSubmit",
    4: "QuantPackSubmit",
    5: "SendIndexSubmit",
    6: "SendCountSubmit",
    7: "AicInputSyncSubmit",
    8: "SharedGmm1ActivationSubmit",
    9: "SharedCompletionFence",
    10: "MoeEntrySyncSubmit",
    11: "MoeEntry",
    12: "InputResetScalarWait",
    13: "CountTableCheckWait",
    14: "CountTablePrepareComplete",
    15: "MoeCountReadyWait",
}
PARENTS = {12: 1, 13: 14}
POINTS = {
    1: (
        "ProcessInputPreparationStage entry",
        "existing MTE3_S ID3 returns; count DMA may remain",
    ),
    2: ("SendAndQuantBuffInit entry", "Scalar return; Vector clear may remain"),
    3: ("ResetSyncStatus entry", "Scalar return; reset DMA may remain"),
    4: (
        "QuantizeInputTokens / PackPreQuantizedLocalTokens entry",
        "Scalar return; final DMA may remain",
    ),
    5: ("SendTopkIdsIndexForExperts entry", "Scalar return; index DMA may remain"),
    6: ("SendTopkIdsCountForExperts entry", "Scalar return; count DMA may remain"),
    7: ("WaitForInputPreparation entry", "Scalar issued waits; FIX may still wait"),
    8: (
        "before SyncBeforeSharedExpertGmm1",
        "ProcessSharedExpertGmm1 returns; asynchronous work may remain",
    ),
    9: (
        "trace-only PIPE_ALL before",
        "trace-only PIPE_ALL after: local completion upper bound (mode 2)",
    ),
    10: (
        "SyncBeforeMoeExpertStages entry",
        "Scalar return; AIV downstream stream waits may remain",
    ),
    11: ("before ProcessMoeExpertStages", "Scalar entry, NOT first GMM DMA"),
    12: (
        "before existing WaitFlag MTE3_S ID3",
        "snapshot complete; includes earlier local MTE3 work",
    ),
    13: (
        "WaitForCountTable entry",
        "V_S + pending.GetValue()==0; includes copy/check/backoff",
    ),
    14: (
        "PrepareMoeExpertTokenCountTable AIV1 entry",
        "MTE3_S ID2 and countTableReady write finished",
    ),
    15: (
        "WaitForMoeExpertTokenCountReady entry",
        "Scalar observed countTableReady nonzero",
    ),
}
REPORT = ReportSchema(
    title="MegaMoe 输入准备 / 共享阶段 / MoE 入口",
    roles=ROLES,
    event_names=EVENTS,
    event_colors={
        i: (
            "#cbd5e1"
            if i in PARENTS or i in (9, 15)
            else ["#2563eb", "#16a34a", "#d97706", "#9333ea"][i % 4]
        )
        for i in EVENTS
    },
    event_trace_points=POINTS,
    summary_fields=(
        ("Mode", "trace_mode"),
        ("BS", "bs"),
        ("H", "h"),
        ("EP", "world_size"),
    ),
    iteration_label="Launch",
    unit_label="Scope",
    range_label="Token / Expert / Count slot",
    work_label="Range size",
    derived_metric_label="区间时长（含未拆分等待）",
    wait_metric_label="已记录检查/等待",
    event_groups={
        "Input": tuple(EVENTS[i] for i in (1, 2, 3, 4, 5, 6, 7, 12)),
        "Shared": tuple(EVENTS[i] for i in (8, 9)),
        "MoeEntry": tuple(EVENTS[i] for i in (10, 11, 13, 14, 15)),
    },
    phase_presets={
        "Shared / MoE": tuple(EVENTS[i] for i in (8, 9, 10, 11, 13, 14, 15))
    },
)


def delta(begin: int, end: int) -> int:
    return (end - begin) & 0xFFFFFFFF


def parse_capture(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path}: raw bytes must be uint32-aligned")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if len(words) < HEADER_WORDS:
        raise ValueError(f"{path}: incomplete header")
    magic, version, hw, rw, ew, records, cores, capacity = words[:8]
    if magic != MAGIC or version != VERSION:
        raise ValueError(
            f"{path}: magic/version mismatch; check trace build and raw dtype"
        )
    if hw != HEADER_WORDS or ew != EVENT_WORDS or not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError(f"{path}: invalid header/event layout or capacity")
    if (
        cores == 0
        or records != len(ROLES) * cores
        or rw != RECORD_HEADER_WORDS + capacity * ew
    ):
        raise ValueError(f"{path}: invalid record/core layout")
    if len(words) < hw + records * rw:
        raise ValueError(f"{path}: truncated record storage")
    bs, h, world, mode, shared_mode, moe_mode, reserved0, reserved1 = words[8:16]
    if mode not in (1, 2) or reserved0 or reserved1:
        raise ValueError(f"{path}: invalid mode/reserved header fields")
    result = []
    for index in range(records):
        offset = hw + index * rw
        role, core, count, truncated, begin, end, r0, r1 = words[
            offset : offset + RECORD_HEADER_WORDS
        ]
        if (
            (role, core) != divmod(index, cores)
            or count > capacity
            or truncated not in (0, 1)
            or r0
            or r1
        ):
            raise ValueError(f"{path}: invalid record {index}")
        process_duration = delta(begin, end)
        if process_duration >= 0x80000000:
            raise ValueError(f"{path}: ambiguous cycle wrap / stale record {index}")
        events = []
        for j in range(count):
            at = offset + RECORD_HEADER_WORDS + j * ew
            eid, iteration, unit, start, size, work, parent, eb, ee = words[
                at : at + ew
            ]
            if (
                eid not in EVENTS
                or parent != PARENTS.get(eid, 0)
                or (eid == 9 and mode != 2)
            ):
                raise ValueError(f"{path}: invalid event/parent {eid}/{parent}")
            duration = delta(eb, ee)
            if delta(begin, eb) + duration > process_duration:
                raise ValueError(f"{path}: event outside Process / ambiguous wrap")
            is_wait = eid in PARENTS or eid in (9, 15)
            events.append(
                dict(
                    event=EVENTS[eid],
                    event_id=eid,
                    parent_event=EVENTS.get(parent),
                    parent_event_id=parent or None,
                    is_wait_event=is_wait,
                    iteration=iteration,
                    unit=unit,
                    range_begin=start,
                    range_count=size,
                    work_count=work,
                    begin_cycle=eb,
                    end_cycle=ee,
                    duration_cycles=duration,
                    active_cycles=duration,
                    wait_cycles=duration if is_wait else None,
                    wait_event_count=None,
                    begin_trace_point=POINTS[eid][0],
                    end_trace_point=POINTS[eid][1],
                )
            )
        for event in events:
            parent = event["parent_event_id"]
            if parent is None:
                continue
            matches = [
                e
                for e in events
                if e["event_id"] == parent
                and all(
                    e[k] == event[k]
                    for k in (
                        "iteration",
                        "unit",
                        "range_begin",
                        "range_count",
                        "work_count",
                    )
                )
            ]
            if not matches:
                if truncated:
                    continue
                raise ValueError(f"{path}: missing parent for {event['event']}")
            if not any(
                delta(e["begin_cycle"], event["begin_cycle"]) + event["duration_cycles"]
                <= e["duration_cycles"]
                for e in matches
            ):
                raise ValueError(f"{path}: wait lies outside parent")
        # Do not report unrecorded pipeline waits as zero or label the remainder pure compute.
        result.append(
            dict(
                role=ROLES[role],
                core=core,
                process_begin_cycle=begin,
                process_end_cycle=end,
                process_duration_cycles=process_duration,
                truncated=bool(truncated),
                events=events,
            )
        )
    return dict(
        header=dict(
            version=version,
            logical_core_count=cores,
            record_count=records,
            event_capacity=capacity,
            bs=bs,
            h=h,
            world_size=world,
            trace_mode=mode,
            shared_mode=shared_mode,
            moe_mode=moe_mode,
            source=str(path),
        ),
        records=result,
    )


def discover(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        found.extend(path.glob("rank*.raw") if path.is_dir() else [path])

    def key(path: Path):
        match = re.search(r"rank(\d+)", path.name)
        return (str(path.parent), int(match[1]) if match else -1, path.name)

    found = sorted(set(found), key=key)
    if not found:
        raise ValueError("no rank*.raw captures found")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cycles-per-us",
        type=float,
        help="confirmed device SYS_CNT frequency; omit to show cycles",
    )
    args = parser.parse_args()
    if args.cycles_per_us is not None and args.cycles_per_us <= 0:
        parser.error("--cycles-per-us must be positive")
    paths = discover(args.raw)
    output = args.output_dir or (
        paths[0].parent
        / (paths[0].stem + "_input_trace" if len(paths) == 1 else "input_trace_report")
    )
    schema = replace(REPORT, cycles_per_us=args.cycles_per_us)
    summaries = []
    for index, path in enumerate(paths):
        name = f"{index:03d}_{path.stem}"
        target = output if len(paths) == 1 else output / name
        summary = write_report_bundle(parse_capture(path), target, schema)
        summary["name"] = name
        summaries.append(summary)
        print(
            f"{path}: {summary['event_count']} events, {summary['truncated_record_count']} truncated records -> {target}"
        )
    if len(paths) > 1:
        write_run_index(summaries, output, schema)
    (output / "capture-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
