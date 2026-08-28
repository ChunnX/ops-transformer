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
"""Parse MegaMoe A8W4 critical-path raw data and generate per-rank reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

MAGIC = 0x41385452
VERSION = 1
HEADER_WORDS = 24
RECORD_HEADER_WORDS = 8
EVENT_WORDS = 13
ROLE_NAMES = ("AIC", "AIV0", "AIV1")
NO_SCOPE = 0xFFFFFFFF
FLAG_COMPLETE = 1
FLAG_SUBMIT = 2
FLAG_WAIT = 4
FLAG_TRACE_ONLY = 8
EVENT_NAMES = {
    1: "OP_PRE_MOE",
    2: "DISPATCH_WORK",
    3: "WAIT_FOR_DISPATCH",
    4: "GMM1_PROLOGUE",
    5: "GMM1_COMPUTE",
    6: "WAIT_FOR_GMM1",
    7: "ACTIVATION_WORK",
    8: "WAIT_FOR_ACTIVATION",
    9: "GMM2_PROLOGUE",
    10: "GMM2_COMPUTE",
    11: "WAIT_FOR_GMM2",
    12: "COMBINE_WORK",
    13: "FINAL_COMBINE_OR_DRAIN",
    14: "COUNT_EXPORT",
    15: "UNPERMUTE",
    16: "OP_POST_MOE",
    17: "CROSS_RANK_INPUT_SYNC",
    18: "CROSS_RANK_OUTPUT_SYNC",
    19: "SHARED_GMM1_ACTIVATION",
    20: "SHARED_GMM2",
}
EVENT_COLORS = {
    1: "#64748b",
    2: "#06b6d4",
    3: "#cbd5e1",
    4: "#a855f7",
    5: "#2563eb",
    6: "#cbd5e1",
    7: "#22c55e",
    8: "#cbd5e1",
    9: "#c084fc",
    10: "#ef4444",
    11: "#cbd5e1",
    12: "#f97316",
    13: "#fb923c",
    14: "#14b8a6",
    15: "#0ea5e9",
    16: "#475569",
    17: "#94a3b8",
    18: "#94a3b8",
    19: "#84cc16",
    20: "#eab308",
}
EVENT_TRACE_POINTS = {
    1: ("ProcessWave 入口", "输入准备 SyncAll 返回"),
    2: ("DispatchMoeExpert 前", "现有 MTE3_S 与 row-ready 发布后"),
    3: ("dispatch count/row 轮询前", "轮询满足后"),
    4: ("GMM1 W4 prologue tile 前", "trace-only MTE3_S 后"),
    5: ("GMM1 scheduler tile 前", "现有 FIX_S 与 ready 发布后"),
    6: ("GMM1 tile ready 轮询前", "轮询满足后"),
    7: ("activation scheduler tile 前", "现有 MTE3_S 与 GMM2-ready 发布后"),
    8: ("activation flag 轮询前", "轮询满足后"),
    9: ("GMM2 W4 prologue tile 前", "trace-only MTE3_S 后"),
    10: ("GMM2 scheduler tile 前", "trace-only FIX_S 后"),
    11: ("GMM2 ready cache-line 扫描前", "MTE2/reduce/backoff 轮询满足后"),
    12: ("非 final combine 前", "combine send loop 返回（submit）"),
    13: ("final combine 或 wave drain 前", "submit；drain 实例在 trace-only MTE3_S 后"),
    14: ("compact count export 前", "现有 export MTE3_S 后"),
    15: ("unpermute 前", "trace-only tail MTE3_S 后"),
    16: ("MoE waves 后", "raw publication 前"),
    17: ("输入 GM signal 同步前", "正式同步返回"),
    18: ("输出 GM signal 同步前", "正式同步返回"),
    19: ("optional shared GMM1 前", "role-specific trace-only completion 后"),
    20: ("optional shared GMM2 前", "role-specific trace-only completion 后"),
}


def _u64(lo: int, hi: int) -> int:
    return lo | (hi << 32)


def _scope(value: int) -> int | None:
    return None if value == NO_SCOPE else value


def interval_is_within(child: dict, parent: dict) -> bool:
    return (
        child["begin_cycle"] >= parent["begin_cycle"]
        and child["end_cycle"] <= parent["end_cycle"]
    )


def attach_wait_intervals(events: list[dict]) -> None:
    wait_events = [event for event in events if event["is_wait_event"]]
    for event in events:
        if event["is_wait_event"]:
            event["wait_cycles"] = event["duration_cycles"]
            event["active_cycles"] = 0
            event["wait_event_count"] = 1
            continue
        matching_waits = [
            wait_event
            for wait_event in wait_events
            if wait_event["parent_event_id"] == event["event_id"]
            and all(
                wait_event[key] is None
                or event[key] is None
                or wait_event[key] == event[key]
                for key in ("wave", "expert", "tile")
            )
            and interval_is_within(wait_event, event)
        ]
        merged_wait_cycles = 0
        merged_end = None
        for wait_event in sorted(matching_waits, key=lambda item: item["begin_cycle"]):
            if merged_end is None or wait_event["begin_cycle"] >= merged_end:
                merged_wait_cycles += wait_event["duration_cycles"]
                merged_end = wait_event["end_cycle"]
            elif wait_event["end_cycle"] > merged_end:
                merged_wait_cycles += wait_event["end_cycle"] - merged_end
                merged_end = wait_event["end_cycle"]
        event["wait_event_count"] = len(matching_waits)
        event["wait_cycles"] = merged_wait_cycles
        event["active_cycles"] = max(0, event["duration_cycles"] - event["wait_cycles"])


def read_words(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path}: byte size {len(data)} is not divisible by four")
    return struct.unpack(f"<{len(data) // 4}I", data)


def parse_trace(words: tuple[int, ...]) -> dict:
    if len(words) < HEADER_WORDS:
        raise ValueError("capture is smaller than the trace header")
    if words[0] != MAGIC:
        raise ValueError(
            f"trace magic mismatch: got 0x{words[0]:08x}, expected 0x{MAGIC:08x}"
        )

    header = {
        "version": words[1],
        "header_words": words[2],
        "record_words": words[3],
        "event_words": words[4],
        "record_count": words[5],
        "logical_core_count": words[6],
        "event_capacity_per_record": words[7],
        "rank": words[8],
        "bs": words[9],
        "h": words[10],
        "hidden_dim": words[11],
        "top_k": words[12],
        "moe_expert_count": words[13],
        "world_size": words[14],
        "wave_m_group_target": words[15],
        "clock_bits": words[16],
    }
    if header["header_words"] != HEADER_WORDS:
        raise ValueError(f"unsupported header size: {header['header_words']}")
    if header["version"] != VERSION:
        raise ValueError(f"unsupported trace version: {header['version']}")
    if header["event_words"] != EVENT_WORDS:
        raise ValueError(
            f"version {VERSION} requires {EVENT_WORDS} event words, got {header['event_words']}"
        )
    expected_record_words = (
        RECORD_HEADER_WORDS + header["event_capacity_per_record"] * EVENT_WORDS
    )
    if header["record_words"] != expected_record_words:
        raise ValueError(
            f"record size mismatch: got {header['record_words']}, expected {expected_record_words}"
        )
    expected_record_count = len(ROLE_NAMES) * header["logical_core_count"]
    if header["record_count"] != expected_record_count:
        raise ValueError(
            f"record count mismatch: got {header['record_count']}, expected {expected_record_count}"
        )
    required_words = (
        header["header_words"] + header["record_count"] * header["record_words"]
    )
    if len(words) < required_words:
        raise ValueError(
            f"capture has {len(words)} words, protocol requires {required_words}"
        )

    records = []
    warnings = []
    for record_index in range(header["record_count"]):
        role_index = record_index // header["logical_core_count"]
        core_index = record_index % header["logical_core_count"]
        record_base = header["header_words"] + record_index * header["record_words"]
        process_begin = _u64(words[record_base], words[record_base + 1])
        process_end = _u64(words[record_base + 2], words[record_base + 3])
        event_count, dropped, rank, packed = words[record_base + 4 : record_base + 8]
        encoded_role, encoded_core = packed >> 16, packed & 0xFFFF
        if (encoded_role, encoded_core) != (role_index, core_index):
            raise ValueError(
                f"record {record_index}: role/core mismatch {encoded_role}/{encoded_core}"
            )
        if rank != header["rank"]:
            raise ValueError(
                f"record {record_index}: rank {rank} != header rank {header['rank']}"
            )
        if process_end < process_begin:
            raise ValueError(f"record {record_index}: process interval is reversed")
        if event_count > header["event_capacity_per_record"]:
            raise ValueError(
                f"record {record_index}: event count {event_count} exceeds capacity"
            )
        if dropped:
            warnings.append(
                f"rank {rank} {ROLE_NAMES[role_index]} core {core_index}: "
                f"truncated, dropped {dropped} event(s)"
            )
        events = []
        for event_index in range(event_count):
            base = (
                record_base + RECORD_HEADER_WORDS + event_index * header["event_words"]
            )
            event_words = words[base : base + header["event_words"]]
            event_type, parent_event_id, flags = event_words[:3]
            if event_type not in EVENT_NAMES:
                raise ValueError(
                    f"record {record_index} event {event_index}: unknown event id {event_type}"
                )
            begin, end = (
                _u64(event_words[9], event_words[10]),
                _u64(event_words[11], event_words[12]),
            )
            paired = end != 0
            if not paired:
                warnings.append(
                    f"rank {rank} {ROLE_NAMES[role_index]} core {core_index}: "
                    f"unpaired {EVENT_NAMES[event_type]} at {begin}"
                )
                end = begin
            if end < begin:
                raise ValueError(
                    f"record {record_index} event {event_index}: reversed interval"
                )
            duration_cycles = end - begin
            events.append(
                {
                    "event": EVENT_NAMES.get(event_type, f"unknown({event_type})"),
                    "event_id": event_type,
                    "is_wait_event": bool(flags & FLAG_WAIT),
                    "parent_event": EVENT_NAMES.get(parent_event_id)
                    if parent_event_id
                    else None,
                    "parent_event_id": parent_event_id,
                    "flags": flags,
                    "completion": "submit" if flags & FLAG_SUBMIT else "complete",
                    "trace_only_completion": bool(flags & FLAG_TRACE_ONLY),
                    "paired": paired,
                    "wave": _scope(event_words[3]),
                    "expert": _scope(event_words[4]),
                    "tile": _scope(event_words[5]),
                    "token_start_in_expert": event_words[6],
                    "token_count": event_words[7],
                    "work_count": event_words[8],
                    "begin_cycle": begin,
                    "end_cycle": end,
                    "duration_cycles": duration_cycles,
                    "wait_cycles": None,
                    "active_cycles": duration_cycles,
                    "wait_event_count": None,
                    "begin_trace_point": EVENT_TRACE_POINTS.get(
                        event_type, ("未知", "未知")
                    )[0],
                    "end_trace_point": EVENT_TRACE_POINTS.get(
                        event_type, ("未知", "未知")
                    )[1],
                }
            )
        events.sort(
            key=lambda event: (
                event["begin_cycle"],
                event["end_cycle"],
                event["event_id"],
            )
        )
        attach_wait_intervals(events)
        records.append(
            {
                "role": ROLE_NAMES[role_index],
                "core": core_index,
                "process_begin_cycle": process_begin,
                "process_end_cycle": process_end,
                "rank": rank,
                "process_duration_cycles": process_end - process_begin,
                "written_count": event_count,
                "dropped_count": dropped,
                "truncated": bool(dropped),
                "events": events,
            }
        )
    return {"header": header, "warnings": warnings, "records": records}


def write_csv(trace: dict, path: Path) -> None:
    columns = [
        "role",
        "core",
        "event",
        "event_id",
        "is_wait_event",
        "parent_event",
        "parent_event_id",
        "flags",
        "completion",
        "trace_only_completion",
        "paired",
        "wave",
        "expert",
        "tile",
        "token_start_in_expert",
        "token_count",
        "work_count",
        "begin_cycle",
        "end_cycle",
        "duration_cycles",
        "active_cycles",
        "wait_cycles",
        "wait_event_count",
        "begin_trace_point",
        "end_trace_point",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in trace["records"]:
            for event in record["events"]:
                writer.writerow(
                    {"role": record["role"], "core": record["core"], **event}
                )


def write_svg(trace: dict, path: Path) -> None:
    role_order = {"AIC": 0, "AIV0": 1, "AIV1": 2}
    records = sorted(
        (record for record in trace["records"] if record["events"]),
        key=lambda record: (
            record["core"],
            role_order.get(record["role"], len(role_order)),
        ),
    )
    lanes = []
    for record in records:
        wait_events = [event for event in record["events"] if event["is_wait_event"]]
        wait_cycles = (
            sum(event["duration_cycles"] for event in wait_events)
            if wait_events
            else sum(event["wait_cycles"] or 0 for event in record["events"])
        )
        lane_label = f"B{record['core']} · {record['role']}"
        lane_tip = f"Process={record['process_duration_cycles']} cycles, Wait={wait_cycles} cycles"
        lanes.append(
            (
                record["core"],
                lane_label,
                lane_tip,
                sorted(record["events"], key=lambda event: event["is_wait_event"]),
            )
        )
    width = 1880
    left = 105
    right = 30
    row_height = 24
    block_gap = 7
    top = 55
    lane_rows = []
    block_ranges = []
    current_y = top
    for row, lane in enumerate(lanes):
        if row > 0 and lane[0] != lanes[row - 1][0]:
            block_ranges[-1][2] = current_y
            current_y += block_gap
        if row == 0 or lane[0] != lanes[row - 1][0]:
            block_ranges.append([lane[0], current_y, 0])
        lane_rows.append(current_y)
        current_y += row_height
    if block_ranges:
        block_ranges[-1][2] = current_y
    height = current_y + 45
    begin_cycles = [
        event["begin_cycle"] for record in records for event in record["events"]
    ]
    if not begin_cycles:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="100"/>\n'
        )
        return
    origin = min(begin_cycles)
    relative_end = max(
        event["end_cycle"] - origin for record in records for event in record["events"]
    )
    scale = (width - left - right) / max(1, relative_end)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;font-size:11px}.label{font-weight:600}.event{stroke:#ffffff;stroke-width:.5}</style>",
        f'<text x="12" y="24" font-size="16" font-weight="700">A8W4 critical-path timeline (origin={origin})</text>',
    ]
    for block_idx, (_, block_begin, block_end) in enumerate(block_ranges):
        fill = "#f8fafc" if block_idx % 2 == 0 else "#f1f5f9"
        lines.append(
            f'<rect x="4" y="{block_begin - 2}" width="{width - right - 4}" '
            f'height="{block_end - block_begin + 4}" rx="5" fill="{fill}" stroke="#dbe2ea"/>'
        )
    for y, (_, lane_name, lane_tip, lane_events) in zip(lane_rows, lanes):
        lines.append(
            f'<text class="label" x="10" y="{y + 14}">{lane_name}<title>{lane_tip}</title></text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{y + 18}" x2="{width - right}" y2="{y + 18}" stroke="#e5e7eb"/>'
        )
        for event in lane_events:
            x = left + ((event["begin_cycle"] - origin) * scale)
            event_width = max(1.0, event["duration_cycles"] * scale)
            color = EVENT_COLORS.get(event["event_id"], "#64748b")
            title = (
                f"{event['event']} wave={event['wave']} expert={event['expert']} tile={event['tile']} "
                f"tokens=[{event['token_start_in_expert']},"
                f"{event['token_start_in_expert'] + event['token_count']}) "
                f"begin={event['begin_cycle']} end={event['end_cycle']} "
                f"duration={event['duration_cycles']} "
                f"parent={event['parent_event'] or 'none'} "
                f"begin-point={event['begin_trace_point']} end-point={event['end_trace_point']}"
            )
            lines.append(
                f'<rect class="event" x="{x:.2f}" y="{y + 3}" width="{event_width:.2f}" '
                f'height="14" fill="{color}"><title>{title}</title></rect>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(trace: dict, path: Path) -> None:
    """Write a self-contained interactive report without external JavaScript dependencies."""
    payload = json.dumps(trace, ensure_ascii=False).replace("</", "<\\/")
    header = trace["header"]
    event_count = sum(len(record["events"]) for record in trace["records"])
    truncated_count = sum(record["truncated"] for record in trace["records"])
    cycles_per_us = trace["header"].get("cycles_per_us")
    conversion_note = (
        f"按 {cycles_per_us:g} cycles/μs 换算。"
        if cycles_per_us
        else "raw 协议未携带时钟频率；报告只显示 cycles，可用 --cycles-per-us 指定换算值。"
    )
    warnings_html = (
        "".join(
            f"<li>{html.escape(warning)}</li>" for warning in trace.get("warnings", [])
        )
        or "<li>无</li>"
    )
    trace_point_rows = "".join(
        f"<tr><td>{html.escape(EVENT_NAMES[event_id])}</td>"
        f"<td>{html.escape(points[0])}</td><td>{html.escape(points[1])}</td></tr>"
        for event_id, points in EVENT_TRACE_POINTS.items()
    )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MegaMoe A8W4 critical-path trace</title>
<style>
:root {{ color-scheme: light; --bg:#f5f7fb; --panel:#fff; --line:#dbe2ea; --text:#172033; --muted:#667085;
        --lane-label-width:126px; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 Arial,"Microsoft YaHei",sans-serif; }}
main {{ max-width:1800px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 4px; font-size:24px; }}
h2 {{ margin:0 0 14px; font-size:17px; }}
.muted {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; margin:18px 0; }}
.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; box-shadow:0 1px 2px #1018280d; }}
.card {{ padding:13px 15px; }} .card b {{ display:block; margin-top:3px; font-size:19px; }}
.panel {{ padding:16px; margin:14px 0; overflow:hidden; }}
.controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:14px; }}
select,button {{ padding:6px 9px; border:1px solid #cbd5e1; border-radius:6px; background:#fff; color:var(--text); }}
button {{ cursor:pointer; }} button:hover {{ background:#f1f5f9; }}
.event-picker {{ position:relative; }}
.event-picker summary {{ padding:6px 9px; border:1px solid #cbd5e1; border-radius:6px; background:#fff;
                         cursor:pointer; list-style:none; }}
.event-picker summary::-webkit-details-marker {{ display:none; }}
.event-picker-menu {{ position:absolute; z-index:20; top:38px; left:0; width:min(720px,85vw); max-height:520px;
                      overflow:auto; padding:12px; border:1px solid #cbd5e1; border-radius:8px; background:#fff;
                      box-shadow:0 8px 24px #10182824; }}
.event-picker-actions {{ display:flex; gap:8px; margin-bottom:10px; }}
.event-groups {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:8px; }}
.event-group {{ padding:8px; border:1px solid #e5e7eb; border-radius:6px; background:#f8fafc; }}
.event-group > label {{ display:block; margin-bottom:5px; font-weight:700; }}
.event-option {{ display:block; margin:3px 0; font:11px/1.4 monospace; }}
.event-option input,.event-group input {{ margin-right:5px; vertical-align:middle; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px 16px; color:var(--muted); }}
.legend i {{ display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; }}
#timeline {{ min-height:140px; overflow-x:auto; }}
.axis {{ position:relative; height:24px; margin-left:142px; border-bottom:1px solid var(--line); min-width:900px; }}
.tick {{ position:absolute; bottom:0; height:7px; border-left:1px solid #98a2b3; }}
.tick span {{ position:absolute; top:-18px; left:3px; white-space:nowrap; font-size:10px; color:var(--muted); }}
.lane-wrap {{ display:grid; grid-template-columns:var(--lane-label-width) minmax(900px,1fr); gap:10px;
              align-items:center; margin-top:4px; }}
.lane-name {{ display:flex; align-items:center; gap:5px; min-width:0; font-weight:600; text-align:left; line-height:1.3; }}
.lane-name-text {{ min-width:0; }}
.lane-wrap.lane-pinned .lane {{ border-color:#f59e0b; box-shadow:0 0 0 1px #f59e0b33; }}
.lane-name small {{ display:block; margin-top:2px; overflow:hidden; color:var(--muted); font-size:10px;
                    font-weight:400; text-overflow:ellipsis; white-space:nowrap; }}
.pin-button {{ flex:0 0 auto; width:24px; height:24px; padding:0; border-color:transparent; color:#98a2b3;
               font-size:13px; line-height:22px; }}
.pin-button.active {{ border-color:#f59e0b; background:#fffbeb; color:#b45309; }}
.block-group {{ margin-top:9px; padding:6px 8px 8px; background:#f8fafc; border:1px solid var(--line);
                border-radius:8px; }}
.block-group:nth-of-type(even) {{ background:#f3f6fa; }}
.block-group.pinned {{ border-color:#f59e0b; background:#fffbeb66; }}
.block-header {{ display:flex; align-items:center; justify-content:space-between; min-height:25px; color:var(--muted);
                 font-size:11px; font-weight:600; }}
.pinned-section {{ margin:9px 0 13px; padding:8px; border:1px dashed #f59e0b; border-radius:8px; background:#fffbeb55; }}
.pinned-title {{ margin-bottom:4px; color:#92400e; font-size:12px; font-weight:700; }}
.pinned-lanes {{ display:grid; gap:5px; }}
.pinned-lane {{ padding:4px 7px 7px; border:1px solid #fbbf24; border-radius:7px; background:#fff; }}
.lane {{ position:relative; height:32px; overflow:hidden; background:#f8fafc; border:1px solid #e5e7eb; border-radius:5px; }}
.lane-canvas {{ display:block; width:100%; height:30px; cursor:crosshair; }}
.canvas-tooltip {{ display:none; position:fixed; z-index:100; max-width:min(620px,calc(100vw - 24px)); padding:8px 10px;
                   border:1px solid #94a3b8; border-radius:6px; background:#0f172a; color:#f8fafc;
                   box-shadow:0 8px 24px #0f172a33; font:11px/1.45 monospace; pointer-events:none; }}
.event-detail {{ min-height:54px; margin-top:12px; padding:10px 12px; background:#f8fafc; border:1px solid #e5e7eb;
                 border-radius:6px; color:var(--muted); }}
.event-detail b {{ color:var(--text); }}
.empty {{ padding:30px; text-align:center; color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ padding:7px 8px; border-bottom:1px solid #e5e7eb; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
thead th {{ position:sticky; top:0; background:#f8fafc; z-index:1; }}
.table-scroll {{ max-height:470px; overflow:auto; border:1px solid #e5e7eb; border-radius:6px; }}
#eventTable tbody tr:not(.virtual-spacer) {{ height:32px; }}
#eventTable .virtual-spacer td {{ height:0; padding:0; border:0; }}
.lane-summary-row {{ cursor:pointer; }}
.lane-summary-row:hover {{ background:#f1f5f9; }}
.block-summary-row {{ background:#f8fafc; font-weight:600; }}
.warn {{ color:#b42318; font-weight:600; }}
</style>
</head>
<body><main>
<h1>MegaMoe A8W4 critical-path trace</h1>
<div class="muted">同一份 rank trace 中的所有 lane 共用一个 SYS_CNT cycle 原点和刻度；跨 rank 报告不合并时间轴。
{conversion_note}</div>
<section class="cards">
  <div class="card"><span class="muted">Shape</span><b>BS {header["bs"]} · H {header["h"]}</b></div>
  <div class="card"><span class="muted">Intermediate</span><b>{header["hidden_dim"]}</b></div>
  <div class="card"><span class="muted">TopK / World</span><b>{header["top_k"]} / {header["world_size"]}</b></div>
  <div class="card"><span class="muted">MoE experts</span><b>{header["moe_expert_count"]}</b></div>
  <div class="card"><span class="muted">Wave M-group target</span><b>{header["wave_m_group_target"]}</b></div>
  <div class="card"><span class="muted">Events / truncated</span><b>{event_count} / {truncated_count}</b></div>
</section>
<section class="panel"><h2>协议告警</h2><ul>{warnings_html}</ul></section>
<section class="panel">
  <h2>流水时间线</h2>
  <div class="controls">
    <label>显示 Block <select id="coreSelect"></select></label>
    <label>统计 Block <select id="statsCoreSelect"></select></label>
    <label>Wave <select id="waveSelect"><option value="all">全部</option></select></label>
    <label>等待筛选 <select id="waitFilter">
      <option value="all">全部</option><option value="with-wait">存在等待</option>
      <option value="without-wait">无等待</option><option value="truncated">已截断</option>
    </select></label>
    <label>泳道排序 <select id="laneSort">
      <option value="block">Block / 角色</option>
      <option value="process-desc">Block Process 总时长 ↓</option>
      <option value="span-desc">Block 事件覆盖时长 ↓</option>
      <option value="wait-desc">Block 累计等待时长 ↓</option>
      <option value="wait-ratio-desc">Block 等待占比 ↓</option>
      <option value="last-desc">Block 最后打点位置 ↓</option>
      <option value="tail-desc">Block 尾部空洞 ↓</option>
    </select></label>
    <label>显示 <select id="laneLimit">
      <option value="all" selected>全部 Block</option><option value="5">前 5 个 Block</option>
      <option value="10">前 10 个 Block</option><option value="20">前 20 个 Block</option>
    </select></label>
    <label>快速视图 <select id="phasePreset">
      <option value="all">全部阶段</option>
      <option value="pre">PRE_MOE</option>
      <option value="moe">MoE 主流水</option>
      <option value="count">COUNT_EXPORT</option>
      <option value="unpermute">UNPERMUTE</option>
      <option value="post">POST_MOE</option>
      <option value="shared">Shared experts</option>
      <option value="cross-rank">跨 Rank 同步</option>
      <option value="custom" disabled>自定义事件</option>
    </select></label>
    <details id="eventPicker" class="event-picker">
      <summary>阶段/事件：<span id="eventSelectionCount"></span></summary>
      <div class="event-picker-menu">
        <div class="event-picker-actions">
          <button type="button" id="selectAllEvents">全选</button>
          <button type="button" id="clearAllEvents">清空</button>
        </div>
        <div id="eventGroups" class="event-groups"></div>
      </div>
    </details>
    <span id="selectionSummary" class="muted"></span>
  </div>
  <div class="legend" id="legend"></div>
  <div class="muted">每个 role/core 只使用一条泳道。父阶段显示完整执行区间，等待按真实 begin/end cycle
  覆盖在同一泳道；Active = Total - 这些真实等待区间。可固定整个 Block 或单条泳道到顶部对比区。</div>
  <div id="timeline"></div>
  <div id="eventDetail" class="event-detail">点击时间线中的事件块查看完整起止 cycle 与打点位置。</div>
  <div id="canvasTooltip" class="canvas-tooltip"></div>
</section>
<section class="panel">
  <h2>选中阶段/事件统计</h2>
  <div class="muted"><span id="statsSelectionSummary"></span>。统计 Block 与显示 Block 独立；统计不受显示 Top N 限制，
  但随 Wave、等待和阶段/事件筛选联动。累计值按事件区间求和；父阶段和子等待
  可能重叠，因此不等同于墙钟耗时。覆盖 Span 是选中区间最早 begin 到最晚 end 的时间范围。</div>
  <div class="table-scroll"><table id="selectionStatsTable"><thead><tr>
    <th>阶段组</th><th>事件</th><th>角色</th><th>数量</th><th>总时间</th><th>Active</th><th>Wait</th>
    <th>均值</th><th>P50</th><th>P95</th><th>最大</th><th>最早 begin</th><th>最晚 end</th><th>覆盖 Span</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<section class="panel">
  <h2>泳道概览</h2>
  <div class="muted">指标随当前 Block、Wave 和等待筛选更新；筛选和排序以 Block 为单位，AIC/AIV0/AIV1
  始终绑定显示。点击任意一行可定位到该 Block 的全部角色。</div>
  <div class="table-scroll"><table id="laneTable"><thead><tr>
    <th>AI Block</th><th>角色</th><th>Process begin</th><th>Process end</th><th>Process 总时长</th>
    <th>首事件</th><th>最后事件</th><th>最后结束 cycle</th><th>事件覆盖时长</th>
    <th>累计 Stage</th><th>累计 Wait</th><th>Wait/Stage</th><th>尾部空洞</th><th>事件数</th><th>截断</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<section class="panel">
  <h2>Wave 汇总</h2>
  <div class="table-scroll"><table id="waveTable"><thead><tr>
    <th>AI Block</th><th>Wave</th><th>角色</th><th>阶段</th><th>事件数</th><th>Token 数</th><th>Work count</th>
    <th>总时间</th><th>Active</th><th>Wait</th><th>Wait 占比</th><th>最长事件</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<section class="panel">
  <h2>打点位置</h2>
  <div class="table-scroll"><table><thead><tr><th>阶段</th><th>Begin 打点</th><th>End 打点/完成边界</th></tr></thead>
  <tbody>{trace_point_rows}</tbody></table></div>
</section>
<section class="panel">
  <h2>事件明细</h2>
  <div class="table-scroll" id="eventTableScroll"><table id="eventTable"><thead><tr>
    <th>AI Block</th><th>角色</th><th>事件</th><th>父阶段</th><th>Wave</th><th>Expert</th><th>Token 范围</th><th>Work count</th>
    <th>Begin cycle</th><th>End cycle</th><th>总 cycles</th><th>Active cycles</th><th>Wait cycles</th><th>Wait 占比</th>
    <th>总耗时 (μs)</th><th>Begin 打点</th><th>End 打点</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<script id="traceData" type="application/json">{payload}</script>
<script>
const trace = JSON.parse(document.getElementById('traceData').textContent);
for(const record of trace.records) {{
  for(const event of record.events) {{ event.role=record.role; event.core=record.core; }}
}}
const colors = {{OP_PRE_MOE:'#64748b',DISPATCH_WORK:'#06b6d4',WAIT_FOR_DISPATCH:'#cbd5e1',GMM1_PROLOGUE:'#a855f7',GMM1_COMPUTE:'#2563eb',WAIT_FOR_GMM1:'#cbd5e1',ACTIVATION_WORK:'#22c55e',WAIT_FOR_ACTIVATION:'#cbd5e1',GMM2_PROLOGUE:'#c084fc',GMM2_COMPUTE:'#ef4444',WAIT_FOR_GMM2:'#cbd5e1',COMBINE_WORK:'#f97316',FINAL_COMBINE_OR_DRAIN:'#fb923c',COUNT_EXPORT:'#14b8a6',UNPERMUTE:'#0ea5e9',OP_POST_MOE:'#475569',CROSS_RANK_INPUT_SYNC:'#94a3b8',CROSS_RANK_OUTPUT_SYNC:'#94a3b8',SHARED_GMM1_ACTIVATION:'#84cc16',SHARED_GMM2:'#eab308'}};
const legendColors = colors;
const eventGroups = {{
  '算子边界':['OP_PRE_MOE','OP_POST_MOE'],
  '跨 Rank 同步':['CROSS_RANK_INPUT_SYNC','CROSS_RANK_OUTPUT_SYNC'],
  'Dispatch':['DISPATCH_WORK','WAIT_FOR_DISPATCH'],
  'GMM1':['GMM1_PROLOGUE','GMM1_COMPUTE','WAIT_FOR_GMM1'],
  'Activation':['ACTIVATION_WORK','WAIT_FOR_ACTIVATION'],
  'GMM2':['GMM2_PROLOGUE','GMM2_COMPUTE','WAIT_FOR_GMM2'],
  'Combine / 收尾':['COMBINE_WORK','FINAL_COMBINE_OR_DRAIN','COUNT_EXPORT','UNPERMUTE'],
  'Shared':['SHARED_GMM1_ACTIVATION','SHARED_GMM2'],
}};
const phasePresets = {{
  pre:['OP_PRE_MOE'],
  moe:['DISPATCH_WORK','WAIT_FOR_DISPATCH','GMM1_PROLOGUE','GMM1_COMPUTE','WAIT_FOR_GMM1',
       'ACTIVATION_WORK','WAIT_FOR_ACTIVATION','GMM2_PROLOGUE','GMM2_COMPUTE','WAIT_FOR_GMM2',
       'COMBINE_WORK','FINAL_COMBINE_OR_DRAIN'],
  count:['COUNT_EXPORT'],
  unpermute:['UNPERMUTE'],
  post:['OP_POST_MOE'],
  shared:['SHARED_GMM1_ACTIVATION','SHARED_GMM2'],
  'cross-rank':['CROSS_RANK_INPUT_SYNC','CROSS_RANK_OUTPUT_SYNC'],
}};
const eventToGroup = new Map(Object.entries(eventGroups).flatMap(([group,names])=>names.map(name=>[name,group])));
const roles = ['AIC','AIV0','AIV1'];
const coreSelect = document.getElementById('coreSelect');
const statsCoreSelect = document.getElementById('statsCoreSelect');
const waveSelect = document.getElementById('waveSelect');
const waitFilter = document.getElementById('waitFilter');
const laneSort = document.getElementById('laneSort');
const laneLimit = document.getElementById('laneLimit');
const phasePreset = document.getElementById('phasePreset');
const eventGroupsRoot = document.getElementById('eventGroups');
const canvasTooltip = document.getElementById('canvasTooltip');
const eventTableScroll = document.getElementById('eventTableScroll');
const pinnedBlocks = new Set();
const pinnedLanes = new Set();
let currentBlockGroups=[];
let timelineResizeObserver=null;
let virtualEventRows=[];
let virtualEventFrame=null;
const eventRowHeight=32;
const eventRowOverscan=12;
const availableEvents = new Set(trace.records.flatMap(record=>record.events.map(event=>event.event)));
const selectedEvents = new Set(availableEvents);
coreSelect.add(new Option('全部','all'));
statsCoreSelect.add(new Option('全部','all'));
for (let core=0; core<trace.header.logical_core_count; core++) {{
  coreSelect.add(new Option(String(core),String(core)));
  statsCoreSelect.add(new Option(String(core),String(core)));
}}
const waves = [...new Set(trace.records.flatMap(r=>r.events.map(e=>e.wave)))].sort((a,b)=>a-b);
for (const wave of waves) waveSelect.add(new Option(String(wave),String(wave)));
function renderEventPicker() {{
  eventGroupsRoot.innerHTML=Object.entries(eventGroups).map(([group,names])=>{{
    const present=names.filter(name=>availableEvents.has(name));
    if(!present.length) return '';
    const checked=present.every(name=>selectedEvents.has(name));
    const children=present.map(name=>`<label class="event-option"><input type="checkbox" data-event="${{name}}" ${{selectedEvents.has(name)?'checked':''}}>${{name}}</label>`).join('');
    return `<div class="event-group"><label><input type="checkbox" data-event-group="${{group}}" ${{checked?'checked':''}}>${{group}}</label>${{children}}</div>`;
  }}).join('');
  const count=selectedEvents.size;
  document.getElementById('eventSelectionCount').textContent=`${{count}} / ${{availableEvents.size}}`;
  eventGroupsRoot.querySelectorAll('[data-event-group]').forEach(input=>{{
    const present=eventGroups[input.dataset.eventGroup].filter(name=>availableEvents.has(name));
    input.indeterminate=present.some(name=>selectedEvents.has(name))&&!present.every(name=>selectedEvents.has(name));
    input.addEventListener('change',()=>{{
      for(const name of present) input.checked?selectedEvents.add(name):selectedEvents.delete(name);
      phasePreset.value='custom';
      renderEventPicker(); render();
    }});
  }});
  eventGroupsRoot.querySelectorAll('[data-event]').forEach(input=>input.addEventListener('change',()=>{{
    input.checked?selectedEvents.add(input.dataset.event):selectedEvents.delete(input.dataset.event);
    phasePreset.value='custom';
    renderEventPicker(); render();
  }}));
}}
document.getElementById('selectAllEvents').addEventListener('click',()=>{{
  for(const name of availableEvents) selectedEvents.add(name); phasePreset.value='all'; renderEventPicker(); render();
}});
document.getElementById('clearAllEvents').addEventListener('click',()=>{{
  selectedEvents.clear(); phasePreset.value='custom'; renderEventPicker(); render();
}});
phasePreset.addEventListener('change',()=>{{
  if(phasePreset.value==='custom') return;
  selectedEvents.clear();
  const names=phasePreset.value==='all'?[...availableEvents]:(phasePresets[phasePreset.value]||[]);
  for(const name of names) if(availableEvents.has(name)) selectedEvents.add(name);
  renderEventPicker(); render();
}});
renderEventPicker();
const cyclesPerUs = trace.header.cycles_per_us || null;
const cycleAnchor = trace.records.length ? trace.records[0].process_begin_cycle : 0;
function signedCycleDelta(cycle) {{
  return cycle-cycleAnchor;
}}
const durationText=cycles=>cyclesPerUs?`${{cycles.toLocaleString()}} cycles (${{(cycles/cyclesPerUs).toFixed(3)}} μs)`:`${{cycles.toLocaleString()}} cycles`;
const cycleDelta=(begin,end)=>end-begin;
function withLaneMetrics(record) {{
  const stageEvents=record.events.filter(e=>!e.is_wait_event);
  const waitEvents=record.events.filter(e=>e.is_wait_event);
  const firstEvent=record.events.reduce((first,e)=>!first||signedCycleDelta(e.begin_cycle)<signedCycleDelta(first.begin_cycle)?e:first,null);
  const lastEvent=record.events.reduce((last,e)=>!last||signedCycleDelta(e.end_cycle)>signedCycleDelta(last.end_cycle)?e:last,null);
  const stageCycles=stageEvents.reduce((sum,e)=>sum+e.duration_cycles,0);
  const waitCycles=waitEvents.length?
    waitEvents.reduce((sum,e)=>sum+e.duration_cycles,0):
    stageEvents.reduce((sum,e)=>sum+(e.wait_cycles||0),0);
  const firstPosition=firstEvent?signedCycleDelta(firstEvent.begin_cycle):0;
  const lastPosition=lastEvent?signedCycleDelta(lastEvent.end_cycle):0;
  return {{...record,laneMetrics:{{
    firstEvent,lastEvent,firstPosition,lastPosition,stageCycles,waitCycles,
    waitRatio:stageCycles?waitCycles/stageCycles:0,
    eventSpanCycles:firstEvent&&lastEvent?Math.max(0,lastPosition-firstPosition):0,
    tailCycles:lastEvent?cycleDelta(lastEvent.end_cycle,record.process_end_cycle):record.process_duration_cycles,
  }}}};
}}
function withBlockMetrics(records) {{
  const recordsWithFirst=records.filter(r=>r.laneMetrics.firstEvent);
  const recordsWithLast=records.filter(r=>r.laneMetrics.lastEvent);
  const processBeginRecord=records.reduce((first,r)=>signedCycleDelta(r.process_begin_cycle)<signedCycleDelta(first.process_begin_cycle)?r:first);
  const processEndRecord=records.reduce((last,r)=>signedCycleDelta(r.process_end_cycle)>signedCycleDelta(last.process_end_cycle)?r:last);
  const firstRecord=recordsWithFirst.reduce((first,r)=>!first||r.laneMetrics.firstPosition<first.laneMetrics.firstPosition?r:first,null);
  const lastRecord=recordsWithLast.reduce((last,r)=>!last||r.laneMetrics.lastPosition>last.laneMetrics.lastPosition?r:last,null);
  const processBeginPosition=signedCycleDelta(processBeginRecord.process_begin_cycle);
  const processEndPosition=signedCycleDelta(processEndRecord.process_end_cycle);
  const firstPosition=firstRecord?firstRecord.laneMetrics.firstPosition:processBeginPosition;
  const lastPosition=lastRecord?lastRecord.laneMetrics.lastPosition:processBeginPosition;
  const stageCycles=records.reduce((sum,r)=>sum+r.laneMetrics.stageCycles,0);
  const waitCycles=records.reduce((sum,r)=>sum+r.laneMetrics.waitCycles,0);
  return {{
    core:records[0].core,
    records,
    blockMetrics:{{
      processDurationCycles:Math.max(0,processEndPosition-processBeginPosition),
      processBeginCycle:processBeginRecord.process_begin_cycle,processEndCycle:processEndRecord.process_end_cycle,
      eventSpanCycles:Math.max(0,lastPosition-firstPosition),
      stageCycles,waitCycles,waitRatio:stageCycles?waitCycles/stageCycles:0,
      lastPosition,tailCycles:Math.max(0,processEndPosition-lastPosition),
      firstEvent:firstRecord?firstRecord.laneMetrics.firstEvent:null,firstRole:firstRecord?firstRecord.role:null,
      lastEvent:lastRecord?lastRecord.laneMetrics.lastEvent:null,lastRole:lastRecord?lastRecord.role:null,
      truncated:records.some(r=>r.truncated),
    }},
  }};
}}
function compareBlockGroups(a,b) {{
  const am=a.blockMetrics,bm=b.blockMetrics;
  const sorters={{
    'process-desc':()=>bm.processDurationCycles-am.processDurationCycles,
    'span-desc':()=>bm.eventSpanCycles-am.eventSpanCycles,
    'wait-desc':()=>bm.waitCycles-am.waitCycles,
    'wait-ratio-desc':()=>bm.waitRatio-am.waitRatio,
    'last-desc':()=>bm.lastPosition-am.lastPosition,
    'tail-desc':()=>bm.tailCycles-am.tailCycles,
  }};
  const delta=laneSort.value==='block'?0:sorters[laneSort.value]();
  return delta||a.core-b.core;
}}
const laneKey=(core,role)=>`${{core}}|${{role}}`;
function allBlockGroupsForWave() {{
  const wave=waveSelect.value;
  const records=trace.records
    .map(r=>withLaneMetrics({{...r,events:r.events.filter(e=>
      selectedEvents.has(e.event)&&(wave==='all'||e.wave===Number(wave)))}}))
  return [...new Set(records.map(r=>r.core))].map(block=>
    withBlockMetrics(records.filter(r=>r.core===block).sort((a,b)=>roles.indexOf(a.role)-roles.indexOf(b.role))))
    .filter(group=>group.records.some(r=>r.events.length));
}}
function selectedBlockGroups(sourceGroups,core,applyLimit) {{
  const groups=sourceGroups
    .filter(group=>core==='all'||group.core===Number(core))
    .filter(group=>waitFilter.value==='all'||
      (waitFilter.value==='with-wait'&&group.blockMetrics.waitCycles>0)||
      (waitFilter.value==='without-wait'&&group.blockMetrics.waitCycles===0)||
      (waitFilter.value==='truncated'&&group.blockMetrics.truncated))
    .sort(compareBlockGroups);
  return !applyLimit||laneLimit.value==='all'?groups:groups.slice(0,Number(laneLimit.value));
}}
function esc(value) {{ return String(value).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function renderStatistics() {{
  const statsBlockGroups=selectedBlockGroups(currentBlockGroups,statsCoreSelect.value,false);
  const statsRecords=statsBlockGroups.flatMap(group=>group.records);
  const statsEvents=statsRecords.flatMap(r=>r.events);
  const statsBlockText=statsCoreSelect.value==='all'?'全部可见 Block':`Block ${{statsCoreSelect.value}}`;
  document.getElementById('statsSelectionSummary').textContent=`统计 ${{statsBlockText}}：${{statsBlockGroups.length}} 个 Block · ${{statsEvents.length}} 个事件`;
  renderSelectionStats(statsEvents);
}}
function render() {{
  currentBlockGroups=allBlockGroupsForWave();
  const blockGroups=selectedBlockGroups(currentBlockGroups,coreSelect.value,true);
  const records=blockGroups.flatMap(group=>group.records);
  const events=records.flatMap(r=>r.events);
  document.getElementById('selectionSummary').textContent=`${{blockGroups.length}} 个 Block · ${{records.length}} 条泳道 · ${{events.length}} 个事件 · ${{selectedEvents.size}} 种事件`;
  document.getElementById('legend').innerHTML = Object.entries(legendColors)
    .filter(([name])=>selectedEvents.has(name))
    .map(([name,color])=>`<span><i style="background:${{color}}"></i>${{name}}</span>`).join('');
  renderTimeline(blockGroups,currentBlockGroups,events);
  renderStatistics(); renderLaneTable(blockGroups); renderWaveTable(events); renderEventTable(events);
}}
function renderTimeline(blockGroups,pinSourceGroups,events) {{
  const root=document.getElementById('timeline');
  canvasTooltip.style.display='none';
  if(timelineResizeObserver) {{ timelineResizeObserver.disconnect(); timelineResizeObserver=null; }}
  const pinnedBlockGroups=pinSourceGroups.filter(group=>pinnedBlocks.has(group.core));
  const pinnedLaneRecords=pinSourceGroups.flatMap(group=>group.records)
    .filter(record=>pinnedLanes.has(laneKey(record.core,record.role))&&!pinnedBlocks.has(record.core));
  const pinnedEvents=[...pinnedBlockGroups.flatMap(group=>group.records.flatMap(record=>record.events)),
    ...pinnedLaneRecords.flatMap(record=>record.events)];
  const hasPinnedEvents=pinnedEvents.length>0;
  if (!events.length&&!hasPinnedEvents) {{
    root.innerHTML='<div class="empty">当前筛选条件下没有事件</div>'; return;
  }}
  const domainEvents=[...events,...pinnedEvents];
  let domainStart=Infinity,domainEnd=-Infinity;
  for(const event of domainEvents) {{
    domainStart=Math.min(domainStart,signedCycleDelta(event.begin_cycle));
    domainEnd=Math.max(domainEnd,signedCycleDelta(event.end_cycle));
  }}
  const domainSpan=Math.max(domainEnd-domainStart,1);
  const relativeCycle=cycle=>signedCycleDelta(cycle)-domainStart;
  const absoluteCycle=offset=>cycleAnchor+domainStart+offset;
  const laneModels=[];
  let out='<div class="axis">';
  for(let i=0;i<=5;i++) {{
    const x=i*20, offset=Math.round(domainSpan*i/5), cycle=Math.round(absoluteCycle(offset));
    out+=`<div class="tick" style="left:${{x}}%"><span>${{cycle.toLocaleString()}}<br>+${{offset.toLocaleString()}}</span></div>`;
  }}
  out+='</div>';

  const renderLane=(record,highlightPinned=false)=>{{
    const metrics=record.laneMetrics;
    const lastEvent=metrics.lastEvent;
    const key=laneKey(record.core,record.role);
    const lanePinned=pinnedLanes.has(key);
    const laneTip=`Process begin=${{record.process_begin_cycle}} end=${{record.process_end_cycle}} total=${{record.process_duration_cycles}} cycles | event span=${{metrics.eventSpanCycles}} | wait=${{metrics.waitCycles}} | last=${{lastEvent?lastEvent.event:'none'}}@${{lastEvent?lastEvent.end_cycle:'—'}} | tail=${{metrics.tailCycles}}`;
    const laneEvents=[...record.events].sort((a,b)=>Number(a.is_wait_event)-Number(b.is_wait_event));
    const laneIndex=laneModels.push({{record,events:laneEvents,hitBoxes:[]}})-1;
    let laneOut=`<div class="lane-wrap${{highlightPinned&&lanePinned?' lane-pinned':''}}"><div class="lane-name" title="${{esc(laneTip)}}">`
      +`<button class="pin-button${{lanePinned?' active':''}}" data-pin-lane="${{key}}" title="${{lanePinned?'取消固定':'固定该泳道'}}">📌</button>`
      +`<span class="lane-name-text">${{record.role}}`
      +`<small>P/W ${{durationText(record.process_duration_cycles)}} / ${{durationText(metrics.waitCycles)}}</small>`
      +`</span></div><div class="lane"><canvas class="lane-canvas" data-lane-index="${{laneIndex}}"></canvas></div></div>`;
    return laneOut;
  }};
  const renderBlockGroup=(group,highlightPinned=false)=>{{
    const blockPinned=pinnedBlocks.has(group.core);
    return `<div class="block-group${{highlightPinned&&blockPinned?' pinned':''}}"><div class="block-header"><span>Block ${{group.core}}</span>`
      +`<button class="pin-button${{blockPinned?' active':''}}" data-pin-block="${{group.core}}" title="${{blockPinned?'取消固定':'固定整个 Block'}}">📌</button></div>`
      +group.records.map(record=>renderLane(record,highlightPinned)).join('')+'</div>';
  }};

  if(pinnedBlockGroups.length||pinnedLaneRecords.length) {{
    out+='<div class="pinned-section"><div class="pinned-title">固定对比</div>';
    out+=pinnedBlockGroups.map(group=>renderBlockGroup(group,true)).join('');
    if(pinnedLaneRecords.length) {{
      out+='<div class="pinned-lanes">'+pinnedLaneRecords.map(record=>
        `<div class="pinned-lane"><div class="block-header">Block ${{record.core}}</div>${{renderLane(record,true)}}</div>`).join('')+'</div>';
    }}
    out+='</div>';
  }}
  out+=blockGroups.map(group=>renderBlockGroup(group,false)).join('');
  root.innerHTML=out;
  const eventTooltipHtml=(event,record)=>{{
    const waitText=event.wait_cycles===null?'未拆分':durationText(event.wait_cycles);
    const parentText=event.parent_event?` · parent=${{esc(event.parent_event)}}`:'';
    return `<b>Block ${{record.core}} · ${{record.role}} · ${{esc(event.event)}}</b>${{parentText}}<br>`
      +`begin=${{event.begin_cycle}} · end=${{event.end_cycle}} · total=${{durationText(event.duration_cycles)}}<br>`
      +`active=${{event.active_cycles}} cycles · wait=${{waitText}}<br>`
      +`wave=${{event.wave}} · expert=${{event.expert}} · tile=${{event.tile}} · tokens=[${{event.token_start_in_expert}},${{event.token_start_in_expert+event.token_count}})<br>`
      +`completion=${{esc(event.completion)}} · trace-only=${{event.trace_only_completion}}<br>`
      +`begin point=${{esc(event.begin_trace_point)}}<br>end point=${{esc(event.end_trace_point)}}`;
  }};
  const drawCanvases=()=>{{
    for(const canvas of root.querySelectorAll('.lane-canvas')) {{
      const model=laneModels[Number(canvas.dataset.laneIndex)];
      const width=Math.max(1,canvas.clientWidth),height=30,dpr=window.devicePixelRatio||1;
      const pixelWidth=Math.max(1,Math.round(width*dpr)),pixelHeight=Math.max(1,Math.round(height*dpr));
      if(canvas.width!==pixelWidth||canvas.height!==pixelHeight) {{ canvas.width=pixelWidth; canvas.height=pixelHeight; }}
      const context=canvas.getContext('2d');
      context.setTransform(dpr,0,0,dpr,0,0);
      context.clearRect(0,0,width,height);
      model.hitBoxes=[];
      for(const event of model.events) {{
        const left=relativeCycle(event.begin_cycle)/domainSpan*width;
        const visualWidth=Math.max(1,event.duration_cycles/domainSpan*width);
        context.globalAlpha=event.is_wait_event?.95:.9;
        context.fillStyle=colors[event.event]||'#64748b';
        context.fillRect(left,5,visualWidth,20);
        if(event.is_wait_event) {{
          context.globalAlpha=1; context.strokeStyle='#94a3b8'; context.lineWidth=1;
          context.strokeRect(left+.5,5.5,Math.max(0,visualWidth-1),19);
        }}
        model.hitBoxes.push({{left,right:left+visualWidth,event}});
      }}
      context.globalAlpha=1;
    }}
  }};
  const moveTooltip=(mouseEvent,model)=>{{
    const bounds=mouseEvent.currentTarget.getBoundingClientRect();
    const x=mouseEvent.clientX-bounds.left;
    let hit=null;
    for(let index=model.hitBoxes.length-1;index>=0;index--) {{
      const box=model.hitBoxes[index];
      if(x>=box.left&&x<=box.right) {{ hit=box.event; break; }}
    }}
    if(!hit) {{ canvasTooltip.style.display='none'; return; }}
    canvasTooltip.innerHTML=eventTooltipHtml(hit,model.record);
    canvasTooltip.style.display='block';
    const tooltipWidth=canvasTooltip.offsetWidth,tooltipHeight=canvasTooltip.offsetHeight;
    canvasTooltip.style.left=`${{Math.max(8,Math.min(mouseEvent.clientX+12,window.innerWidth-tooltipWidth-8))}}px`;
    canvasTooltip.style.top=`${{Math.max(8,Math.min(mouseEvent.clientY+12,window.innerHeight-tooltipHeight-8))}}px`;
  }};
  root.querySelectorAll('.lane-canvas').forEach(canvas=>{{
    const model=laneModels[Number(canvas.dataset.laneIndex)];
    canvas.addEventListener('mousemove',event=>moveTooltip(event,model));
    canvas.addEventListener('mouseleave',()=>{{ canvasTooltip.style.display='none'; }});
    canvas.addEventListener('click',event=>{{
      const bounds=canvas.getBoundingClientRect(),x=event.clientX-bounds.left;
      for(let index=model.hitBoxes.length-1;index>=0;index--) {{
        const box=model.hitBoxes[index];
        if(x>=box.left&&x<=box.right) {{ showEventDetail({{event:box.event,record:model.record}}); break; }}
      }}
    }});
  }});
  requestAnimationFrame(drawCanvases);
  if('ResizeObserver' in window) {{
    timelineResizeObserver=new ResizeObserver(()=>requestAnimationFrame(drawCanvases));
    timelineResizeObserver.observe(root);
  }}
  root.querySelectorAll('[data-pin-block]').forEach(button=>button.addEventListener('click',()=>{{
    const core=Number(button.dataset.pinBlock);
    if(pinnedBlocks.has(core)) pinnedBlocks.delete(core); else pinnedBlocks.add(core);
    render();
  }}));
  root.querySelectorAll('[data-pin-lane]').forEach(button=>button.addEventListener('click',()=>{{
    const key=button.dataset.pinLane;
    if(pinnedLanes.has(key)) pinnedLanes.delete(key); else pinnedLanes.add(key);
    render();
  }}));
}}
function showEventDetail(item) {{
  const e=item.event, r=item.record;
  const waitKnown=e.wait_cycles!==null;
  const waitText=waitKnown?durationText(e.wait_cycles):'未拆分';
  const waitPercent=waitKnown&&e.duration_cycles?(e.wait_cycles/e.duration_cycles*100).toFixed(1)+'%':'—';
  const parentText=e.parent_event?`　父阶段：<b>${{esc(e.parent_event)}}</b>`:'';
  const intervalText=e.is_wait_event?'这是由 kernel 直接记录的真实轮询等待区间。':
    `包含 <b>${{e.wait_event_count===null?'—':e.wait_event_count}}</b> 个独立等待区间。`;
  document.getElementById('eventDetail').innerHTML=`<b>Block ${{r.core}} · ${{r.role}} · ${{esc(e.event)}}</b>　`
    +`Wave ${{e.wave}}，Expert ${{e.expert}}，Tile ${{e.tile}}，Token [${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})${{parentText}}<br>`
    +`Begin cycle: <b>${{e.begin_cycle}}</b>　End cycle: <b>${{e.end_cycle}}</b>　Total: <b>${{e.duration_cycles}} cycles</b><br>`
    +`Active: <b>${{e.active_cycles}} cycles</b>　Wait: <b>${{waitText}}</b>　Wait 占比: <b>${{waitPercent}}</b><br>`
    +`${{intervalText}}<br>`
    +`Begin 打点：${{esc(e.begin_trace_point)}}<br>End 打点：${{esc(e.end_trace_point)}}`;
}}
function renderLaneTable(blockGroups) {{
  const tbody=document.querySelector('#laneTable tbody');
  tbody.innerHTML=blockGroups.map(group=>{{
    const bm=group.blockMetrics,first=bm.firstEvent,last=bm.lastEvent;
    const blockWaitRatio=bm.stageCycles?(bm.waitRatio*100).toFixed(1)+'%':'—';
    const lastOffset=bm.lastPosition>=0?`+${{bm.lastPosition.toLocaleString()}}`:bm.lastPosition.toLocaleString();
    const eventCount=group.records.reduce((sum,r)=>sum+r.events.length,0);
    const blockRow=`<tr class="lane-summary-row block-summary-row" data-core="${{group.core}}" title="点击定位到 Block ${{group.core}}">`
      +`<td>${{group.core}}</td><td>Block 汇总</td><td>${{bm.processBeginCycle}}</td><td>${{bm.processEndCycle}}</td>`
      +`<td>${{durationText(bm.processDurationCycles)}}</td>`
      +`<td>${{first?bm.firstRole+' / '+esc(first.event)+' @ '+first.begin_cycle:'—'}}</td>`
      +`<td>${{last?bm.lastRole+' / '+esc(last.event):'—'}}</td><td>${{last?last.end_cycle+' ('+lastOffset+')':'—'}}</td>`
      +`<td>${{bm.eventSpanCycles.toLocaleString()}}</td><td>${{bm.stageCycles.toLocaleString()}}</td>`
      +`<td>${{bm.waitCycles.toLocaleString()}}</td><td>${{blockWaitRatio}}</td><td>${{bm.tailCycles.toLocaleString()}}</td>`
      +`<td>${{eventCount}}</td><td class="${{bm.truncated?'warn':''}}">${{bm.truncated?'是':'否'}}</td></tr>`;
    const roleRows=group.records.map(r=>{{
      const m=r.laneMetrics,roleFirst=m.firstEvent,roleLast=m.lastEvent;
      const waitRatio=m.stageCycles?(m.waitRatio*100).toFixed(1)+'%':'—';
      const roleLastOffset=m.lastPosition>=0?`+${{m.lastPosition.toLocaleString()}}`:m.lastPosition.toLocaleString();
      return `<tr class="lane-summary-row" data-core="${{r.core}}" title="点击定位到 Block ${{r.core}} 的全部角色">`
        +`<td>${{r.core}}</td><td>　${{r.role}}</td><td>${{r.process_begin_cycle}}</td><td>${{r.process_end_cycle}}</td>`
        +`<td>${{durationText(r.process_duration_cycles)}}</td>`
        +`<td>${{roleFirst?esc(roleFirst.event)+' @ '+roleFirst.begin_cycle:'—'}}</td>`
        +`<td>${{roleLast?esc(roleLast.event):'—'}}</td><td>${{roleLast?roleLast.end_cycle+' ('+roleLastOffset+')':'—'}}</td>`
        +`<td>${{m.eventSpanCycles.toLocaleString()}}</td><td>${{m.stageCycles.toLocaleString()}}</td>`
        +`<td>${{m.waitCycles.toLocaleString()}}</td><td>${{waitRatio}}</td><td>${{m.tailCycles.toLocaleString()}}</td>`
        +`<td>${{r.events.length}}</td><td class="${{r.truncated?'warn':''}}">${{r.truncated?'是':'否'}}</td></tr>`;
    }}).join('');
    return blockRow+roleRows;
  }}).join('');
  tbody.querySelectorAll('.lane-summary-row').forEach(row=>row.addEventListener('click',()=>{{
    coreSelect.value=row.dataset.core;
    render();
  }}));
}}
function percentile(sortedValues,ratio) {{
  if(!sortedValues.length) return 0;
  return sortedValues[Math.ceil((sortedValues.length-1)*ratio)];
}}
function selectionMetricRow(group,event,role,items,summary=false) {{
  const durations=items.map(item=>item.duration_cycles).sort((a,b)=>a-b);
  const total=durations.reduce((sum,value)=>sum+value,0);
  const active=items.reduce((sum,item)=>sum+item.active_cycles,0);
  const waitKnown=items.every(item=>item.wait_cycles!==null);
  const wait=waitKnown?items.reduce((sum,item)=>sum+item.wait_cycles,0):null;
  let first=Infinity,last=-Infinity;
  for(const item of items) {{ first=Math.min(first,item.begin_cycle); last=Math.max(last,item.end_cycle); }}
  const waitText=wait===null?'—':durationText(wait);
  return `<tr class="${{summary?'block-summary-row':''}}"><td>${{esc(group)}}</td><td>${{esc(event)}}</td><td>${{role}}</td>`
    +`<td>${{items.length}}</td><td>${{durationText(total)}}</td><td>${{durationText(active)}}</td><td>${{waitText}}</td>`
    +`<td>${{durationText(Math.round(total/items.length))}}</td><td>${{durationText(percentile(durations,.5))}}</td>`
    +`<td>${{durationText(percentile(durations,.95))}}</td><td>${{durationText(durations[durations.length-1])}}</td>`
    +`<td>${{first}}</td><td>${{last}}</td><td>${{durationText(last-first)}}</td></tr>`;
}}
function renderSelectionStats(events) {{
  const tbody=document.querySelector('#selectionStatsTable tbody');
  if(!events.length) {{ tbody.innerHTML='<tr><td colspan="14" class="empty">当前筛选条件下没有事件</td></tr>'; return; }}
  const groups=new Map();
  for(const item of events) {{
    const group=eventToGroup.get(item.event)||'其他';
    if(!groups.has(group)) groups.set(group,[]);
    groups.get(group).push(item);
  }}
  const groupOrder=[...Object.keys(eventGroups),'其他'];
  const rows=[];
  for(const group of groupOrder) {{
    const groupItems=groups.get(group)||[];
    if(!groupItems.length) continue;
    rows.push(selectionMetricRow(group,'阶段汇总','全部',groupItems,true));
    const eventRoles=new Map();
    for(const item of groupItems) {{
      const key=`${{item.event}}|${{item.role}}`;
      if(!eventRoles.has(key)) eventRoles.set(key,[]);
      eventRoles.get(key).push(item);
    }}
    for(const [key,items] of [...eventRoles].sort((a,b)=>a[0].localeCompare(b[0]))) {{
      const separator=key.lastIndexOf('|');
      rows.push(selectionMetricRow(group,key.slice(0,separator),key.slice(separator+1),items));
    }}
  }}
  tbody.innerHTML=rows.join('');
}}
function renderWaveTable(events) {{
  const groups=new Map();
  for(const e of events) {{
    const key=`${{e.core}}|${{e.wave}}|${{e.role}}|${{e.event}}`;
    const g=groups.get(key)||{{core:e.core,wave:e.wave,role:e.role,event:e.event,count:0,tokens:0,mGroups:0,cycles:0,active:0,wait:0,waitKnown:true,max:0}};
    g.count++;
    if(e.is_wait_event) {{ g.tokens=Math.max(g.tokens,e.token_count); g.mGroups=Math.max(g.mGroups,e.work_count); }}
    else {{ g.tokens+=e.token_count; g.mGroups+=e.work_count; }}
    g.cycles+=e.duration_cycles; g.active+=e.active_cycles;
    if(e.wait_cycles===null) g.waitKnown=false; else g.wait+=e.wait_cycles;
    g.max=Math.max(g.max,e.duration_cycles); groups.set(key,g);
  }}
  const rows=[...groups.values()].sort((a,b)=>a.core-b.core||a.wave-b.wave||roles.indexOf(a.role)-roles.indexOf(b.role)||a.event.localeCompare(b.event));
  document.querySelector('#waveTable tbody').innerHTML=rows.map(g=>{{
    const wait=g.waitKnown?durationText(g.wait):'—', waitPercent=g.waitKnown&&g.cycles?(g.wait/g.cycles*100).toFixed(1)+'%':'—';
    return `<tr><td>${{g.core}}</td><td>${{g.wave}}</td><td>${{g.role}}</td><td>${{g.event}}</td><td>${{g.count}}</td><td>${{g.tokens}}</td><td>${{g.mGroups}}</td><td>${{durationText(g.cycles)}}</td><td>${{durationText(g.active)}}</td><td>${{wait}}</td><td>${{waitPercent}}</td><td>${{durationText(g.max)}}</td></tr>`;
  }}).join('');
}}
function eventTableRow(e) {{
    const wait=e.wait_cycles===null?'—':e.wait_cycles.toLocaleString();
    const waitPercent=e.wait_cycles!==null&&e.duration_cycles?(e.wait_cycles/e.duration_cycles*100).toFixed(1)+'%':'—';
    const converted=cyclesPerUs?(e.duration_cycles/cyclesPerUs).toFixed(3):'—';
    return `<tr><td>${{e.core}}</td><td>${{e.role}}</td><td>${{e.event}}</td><td>${{e.parent_event||'—'}}</td><td>${{e.wave}}</td><td>${{e.expert}}</td><td>[${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})</td><td>${{e.work_count}}</td><td>${{e.begin_cycle}}</td><td>${{e.end_cycle}}</td><td>${{e.duration_cycles.toLocaleString()}}</td><td>${{e.active_cycles.toLocaleString()}}</td><td>${{wait}}</td><td>${{waitPercent}}</td><td>${{converted}}</td><td>${{esc(e.begin_trace_point)}}</td><td>${{esc(e.end_trace_point)}}</td></tr>`;
}}
function renderVirtualEventRows() {{
  const tbody=document.querySelector('#eventTable tbody');
  if(!virtualEventRows.length) {{ tbody.innerHTML='<tr><td colspan="17" class="empty">当前筛选条件下没有事件</td></tr>'; return; }}
  const viewportHeight=Math.max(eventTableScroll.clientHeight,470);
  const first=Math.max(0,Math.floor(eventTableScroll.scrollTop/eventRowHeight)-eventRowOverscan);
  const last=Math.min(virtualEventRows.length,Math.ceil((eventTableScroll.scrollTop+viewportHeight)/eventRowHeight)+eventRowOverscan);
  const topHeight=first*eventRowHeight,bottomHeight=(virtualEventRows.length-last)*eventRowHeight;
  const top=`<tr class="virtual-spacer"><td colspan="17" style="height:${{topHeight}}px"></td></tr>`;
  const rows=virtualEventRows.slice(first,last).map(eventTableRow).join('');
  const bottom=`<tr class="virtual-spacer"><td colspan="17" style="height:${{bottomHeight}}px"></td></tr>`;
  tbody.innerHTML=top+rows+bottom;
}}
function renderEventTable(events) {{
  virtualEventRows=[...events].sort((a,b)=>a.core-b.core||a.wave-b.wave||roles.indexOf(a.role)-roles.indexOf(b.role)||signedCycleDelta(a.begin_cycle)-signedCycleDelta(b.begin_cycle));
  eventTableScroll.scrollTop=0;
  renderVirtualEventRows();
}}
eventTableScroll.addEventListener('scroll',()=>{{
  if(virtualEventFrame!==null) return;
  virtualEventFrame=requestAnimationFrame(()=>{{ virtualEventFrame=null; renderVirtualEventRows(); }});
}});
statsCoreSelect.addEventListener('change',renderStatistics);
for(const control of [coreSelect,waveSelect,waitFilter,laneSort,laneLimit]) control.addEventListener('change',render);
render();
</script>
</main></body></html>
"""
    path.write_text(page, encoding="utf-8")


def capture_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.fullmatch(r"rank(\d+)\.(?:raw|bin)", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def collect_captures(inputs: list[Path]) -> list[Path]:
    captures = []
    for input_path in inputs:
        if input_path.is_dir():
            candidates = list(input_path.glob("rank*.raw")) + list(
                input_path.glob("rank*.bin")
            )
            captures.extend(sorted(candidates, key=capture_sort_key))
        elif input_path.is_file():
            captures.append(input_path)
        else:
            raise FileNotFoundError(f"trace input does not exist: {input_path}")

    unique_captures = []
    seen = set()
    for capture in captures:
        resolved = capture.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_captures.append(capture)
    if not unique_captures:
        raise ValueError("no rank*.raw or rank*.bin captures found")
    return unique_captures


def write_trace_artifacts(
    capture: Path, output_dir: Path, cycles_per_us: float | None = 1000.0
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = parse_trace(read_words(capture))
    trace["header"]["cycles_per_us"] = cycles_per_us
    (output_dir / "trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(trace, output_dir / "events.csv")
    write_svg(trace, output_dir / "timeline.svg")
    write_html(trace, output_dir / "index.html")
    return {
        "capture": str(capture),
        "name": capture.stem,
        "header": trace["header"],
        "record_count": len(trace["records"]),
        "event_count": sum(len(record["events"]) for record in trace["records"]),
        "truncated_record_count": sum(
            record["truncated"] for record in trace["records"]
        ),
        "dropped_event_count": sum(
            record["dropped_count"] for record in trace["records"]
        ),
        "warning_count": len(trace["warnings"]),
    }


def write_run_index(summaries: list[dict], output_dir: Path) -> None:
    rows = []
    for summary in summaries:
        header = summary["header"]
        report_dir = summary["name"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(summary['name'])}</td>"
            f"<td>{header['bs']}</td><td>{header['h']}</td><td>{header['hidden_dim']}</td>"
            f"<td>{header['top_k']}</td><td>{header['world_size']}</td>"
            f"<td>{header['logical_core_count']}</td><td>{summary['event_count']}</td>"
            f"<td>{summary['truncated_record_count']}</td>"
            f'<td><a href="{report_dir}/index.html">交互报告</a> · '
            f'<a href="{report_dir}/events.csv">CSV</a> · '
            f'<a href="{report_dir}/timeline.svg">SVG</a></td>'
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MegaMoe 多 Rank Trace 汇总</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#172033}}
main{{max-width:1280px;margin:32px auto;padding:0 24px}}h1{{margin-bottom:8px}}p{{color:#667085}}
.panel{{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:20px;box-shadow:0 1px 3px #10182812}}
.table-scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:10px 12px;border-bottom:1px solid #eaecf0;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:last-child,td:last-child{{text-align:left}}th{{background:#f9fafb}}
a{{color:#175cd3;text-decoration:none}}a:hover{{text-decoration:underline}}
</style></head><body><main><h1>MegaMoe 多 Rank Trace 汇总</h1>
<p>共解析 {len(summaries)} 个 rank。点击某一行的交互报告查看该 rank 的 core/wave 时间线。</p>
<section class="panel"><div class="table-scroll"><table><thead><tr>
<th>Rank 文件</th><th>BS</th><th>H</th><th>Hidden</th><th>TopK</th><th>World size</th>
<th>AI Block</th><th>事件数</th><th>截断记录</th><th>产物</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "run-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


def aggregate_trace(captures: list[Path]) -> dict[tuple[int, str, str], dict]:
    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for capture in captures:
        trace = parse_trace(read_words(capture))
        rank = trace["header"]["rank"]
        for record in trace["records"]:
            for event in record["events"]:
                grouped.setdefault((rank, record["role"], event["event"]), []).append(
                    event
                )
    result = {}
    for key, events in grouped.items():
        totals = [event["duration_cycles"] for event in events]
        waits = [event["wait_cycles"] or 0 for event in events]
        result[key] = {
            "count": len(events),
            "total": sum(totals),
            "active": sum(event["active_cycles"] for event in events),
            "wait": sum(waits),
            "mean": sum(totals) / len(totals),
            "max": max(totals),
        }
    return result


def run_compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare good/bad A8W4 stage aggregates"
    )
    parser.add_argument("--good", type=Path, nargs="+", required=True)
    parser.add_argument("--bad", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    good, bad = (
        aggregate_trace(collect_captures(args.good)),
        aggregate_trace(collect_captures(args.bad)),
    )
    rows = []
    for rank, role, event in sorted(set(good) | set(bad)):
        row = {"rank": rank, "role": role, "event": event}
        good_row, bad_row = (
            good.get((rank, role, event), {}),
            bad.get((rank, role, event), {}),
        )
        for metric in ("count", "total", "active", "wait", "mean", "max"):
            before, after = good_row.get(metric), bad_row.get(metric)
            row[f"good_{metric}"] = before
            row[f"bad_{metric}"] = after
            row[f"delta_{metric}"] = (
                None if before is None or after is None else after - before
            )
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["rank", "role", "event"]
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"compared {len(rows)} rank/role/stage aggregates")
    print(f"artifacts: {args.output_dir}")
    return 0


def make_synthetic(path: Path, rank: int, events: list[dict], dropped: int = 0) -> None:
    core_count, capacity = 1, max(8, len(events))
    record_count = core_count * len(ROLE_NAMES)
    record_words = RECORD_HEADER_WORDS + capacity * EVENT_WORDS
    words = [0] * (HEADER_WORDS + record_count * record_words)
    words[:17] = [
        MAGIC,
        VERSION,
        HEADER_WORDS,
        record_words,
        EVENT_WORDS,
        record_count,
        core_count,
        capacity,
        rank,
        64,
        4096,
        7168,
        8,
        16,
        2,
        4,
        64,
    ]
    for record_index in range(record_count):
        base = HEADER_WORDS + record_index * record_words
        own = [event for event in events if event.get("role", 0) == record_index]
        begin = min((event["begin"] for event in own), default=100)
        end = max((event.get("end", 0) for event in own), default=begin) or begin
        words[base : base + 8] = [
            begin & NO_SCOPE,
            begin >> 32,
            end & NO_SCOPE,
            end >> 32,
            len(own),
            dropped if record_index == 0 else 0,
            rank,
            record_index << 16,
        ]
        for index, event in enumerate(own):
            pos = base + RECORD_HEADER_WORDS + index * EVENT_WORDS
            event_end = event.get("end", 0)
            words[pos : pos + EVENT_WORDS] = [
                event["event"],
                event.get("parent", 0),
                event.get("flags", FLAG_COMPLETE),
                event.get("wave", NO_SCOPE),
                event.get("expert", NO_SCOPE),
                event.get("tile", NO_SCOPE),
                event.get("token_start", 0),
                event.get("token_count", 0),
                event.get("work_count", 0),
                event["begin"] & NO_SCOPE,
                event["begin"] >> 32,
                event_end & NO_SCOPE,
                event_end >> 32,
            ]
    path.write_bytes(struct.pack(f"<{len(words)}I", *words))


def run_synthetic_tests() -> int:
    with tempfile.TemporaryDirectory() as directory_name:
        case = Path(directory_name)
        base = 1 << 32
        specs = [
            {
                "event": 5,
                "begin": base - 20,
                "end": base + 100,
                "wave": 0,
                "expert": 2,
                "tile": 7,
            },
            {
                "event": 3,
                "parent": 5,
                "flags": FLAG_COMPLETE | FLAG_WAIT,
                "begin": base + 10,
                "end": base + 40,
                "wave": 0,
                "expert": 2,
                "tile": 7,
            },
            {
                "event": 3,
                "parent": 5,
                "flags": FLAG_COMPLETE | FLAG_WAIT,
                "begin": base + 30,
                "end": base + 60,
                "wave": 0,
                "expert": 2,
                "tile": 7,
            },
            {
                "event": 3,
                "parent": 5,
                "flags": FLAG_COMPLETE | FLAG_WAIT,
                "begin": base + 12,
                "end": base + 18,
                "wave": 0,
                "expert": 2,
                "tile": 99,
            },
            {
                "event": 5,
                "begin": base + 200,
                "end": base + 260,
                "wave": 1,
                "expert": 2,
                "tile": 8,
            },
            {"event": 7, "begin": base + 300, "end": 0, "role": 2},
        ]
        make_synthetic(case / "rank10.raw", 10, specs, dropped=2)
        make_synthetic(case / "rank2.raw", 2, specs)
        captures = collect_captures([case])
        assert [path.name for path in captures] == ["rank2.raw", "rank10.raw"]
        trace = parse_trace(read_words(captures[0]))
        computes = [
            event for event in trace["records"][0]["events"] if event["event_id"] == 5
        ]
        assert (
            computes[0]["duration_cycles"] == 120 and computes[0]["wait_cycles"] == 50
        )
        assert [event["wave"] for event in computes] == [0, 1]
        assert trace["warnings"] and trace["records"][2]["events"][0]["paired"] is False
        output = case / "report"
        summaries = []
        for capture in captures:
            summaries.append(write_trace_artifacts(capture, output / capture.stem))
        write_run_index(summaries, output)
        required = [output / "index.html", output / "run-summary.json"]
        required += [
            output / "rank2" / name
            for name in ("trace.json", "events.csv", "timeline.svg", "index.html")
        ]
        assert all(path.exists() and path.stat().st_size for path in required)
        rank_page = (output / "rank2" / "index.html").read_text(encoding="utf-8")
        assert 'id="eventGroups"' in rank_page
        assert 'id="phasePreset"' in rank_page
        assert 'id="statsCoreSelect"' in rank_page
        assert 'id="selectionStatsTable"' in rank_page
        assert "moe:['DISPATCH_WORK'" in rank_page
        assert "按 1000 cycles/μs 换算" in rank_page
        assert "selectedEvents.has(e.event)" in rank_page
        assert "const domainEvents=[...events,...pinnedEvents]" in rank_page
        assert '<option value="all" selected>全部 Block</option>' in rank_page
        assert 'id="timelineEventLimit"' not in rank_page
        assert 'id="eventTableLimit"' not in rank_page
        assert 'class="lane-canvas"' in rank_page
        assert 'id="eventTableScroll"' in rank_page
        assert "const drawCanvases=()=>" in rank_page
        assert "new ResizeObserver" in rank_page
        assert "virtualEventRows.slice(first,last)" in rank_page
        assert '<div class="bar ' not in rank_page
        assert (
            "selectedBlockGroups(currentBlockGroups,statsCoreSelect.value,false)"
            in rank_page
        )
        assert (
            "statsCoreSelect.addEventListener('change',renderStatistics)" in rank_page
        )
        good_dir, bad_dir = case / "good", case / "bad"
        good_dir.mkdir()
        bad_dir.mkdir()
        make_synthetic(good_dir / "rank0.raw", 0, specs)
        slower = [
            dict(event, end=event.get("end", 0) + 10) if event.get("end") else event
            for event in specs
        ]
        make_synthetic(bad_dir / "rank0.raw", 0, slower)
        compare_output = case / "comparison"
        assert (
            run_compare(
                [
                    "--good",
                    str(good_dir),
                    "--bad",
                    str(bad_dir),
                    "--output-dir",
                    str(compare_output),
                ]
            )
            == 0
        )
        assert (compare_output / "comparison.csv").stat().st_size
        assert (compare_output / "comparison.json").stat().st_size
        bad_words = list(read_words(captures[0]))
        for index, value in ((0, 0), (1, VERSION + 1), (4, EVENT_WORDS - 1)):
            mutated = bad_words.copy()
            mutated[index] = value
            try:
                parse_trace(tuple(mutated))
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid protocol field {index} was accepted")
        node = shutil.which("node")
        if node:
            page = rank_page
            scripts = re.findall(r"<script([^>]*)>(.*?)</script>", page, re.DOTALL)
            js = "\n".join(
                body
                for attributes, body in scripts
                if "application/json" not in attributes
            )
            js_path = case / "report.js"
            js_path.write_text(js, encoding="utf-8")
            subprocess.run([node, "--check", str(js_path)], check=True)
    print("synthetic trace/report tests passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else __import__("sys").argv[1:]
    if argv and argv[0] == "compare":
        return run_compare(argv[1:])
    if argv == ["synthetic-test"]:
        return run_synthetic_tests()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="rank raw file(s), or a case directory containing rank*.raw/rank*.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="artifact directory; defaults beside the input capture or case directory",
    )
    parser.add_argument(
        "--cycles-per-us",
        type=float,
        default=1000.0,
        help="platform SYS_CNT frequency for time display (default: 1000)",
    )
    args = parser.parse_args(argv)
    captures = collect_captures(args.inputs)
    batch_mode = len(captures) > 1 or any(
        input_path.is_dir() for input_path in args.inputs
    )

    if not batch_mode:
        capture = captures[0]
        output_dir = args.output_dir or capture.parent / f"{capture.stem}-trace"
        summary = write_trace_artifacts(capture, output_dir, args.cycles_per_us)
        print(
            f"parsed {summary['event_count']} events from {summary['record_count']} records"
        )
        print(f"truncated records: {summary['truncated_record_count']}")
        print(f"artifacts: {output_dir}")
        return

    default_root = args.inputs[0] if args.inputs[0].is_dir() else captures[0].parent
    output_dir = args.output_dir or default_root / "trace-report"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for capture in captures:
        rank_output_dir = output_dir / capture.stem
        summary = write_trace_artifacts(capture, rank_output_dir, args.cycles_per_us)
        summaries.append(summary)
        print(
            f"{capture.name}: {summary['event_count']} events, "
            f"{summary['truncated_record_count']} truncated records"
        )
    write_run_index(summaries, output_dir)
    print(f"parsed {len(summaries)} rank captures")
    print(f"summary: {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
