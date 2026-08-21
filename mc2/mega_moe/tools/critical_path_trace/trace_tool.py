#!/usr/bin/env python3
"""Parse the minimal MegaMoe A8W8 wave trace carried in output y."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import struct
from pathlib import Path

MAGIC = 0x41575452
WAIT_CYCLES_UNAVAILABLE = 0xFFFFFFFF
HEADER_WORDS = 24
RECORD_HEADER_WORDS = 4
ROLE_NAMES = ("AIC", "AIV0", "AIV1")
EVENT_NAMES = {
    1: "Dispatch",
    2: "GMM1",
    3: "GMM1 prologue",
    4: "Activation",
    5: "GMM2",
    6: "GMM2 prologue",
    7: "Combine",
    8: "Wait Dispatch",
    9: "Wait GMM1",
    10: "Wait Activation",
    11: "Wait GMM2",
    12: "Dispatch transfer pipeline (aggregate)",
    13: "Wait Dispatch MTE3 reuse (aggregate)",
    14: "Wait Dispatch MTE3 drain (aggregate)",
    15: "Wait Dispatch ready sync (aggregate)",
}
EVENT_COLORS = {
    1: "#06b6d4",
    2: "#2563eb",
    3: "#a855f7",
    4: "#22c55e",
    5: "#ef4444",
    6: "#c084fc",
    7: "#f97316",
    8: "#cbd5e1",
    9: "#cbd5e1",
    10: "#cbd5e1",
    11: "#cbd5e1",
    12: "#0891b2",
    13: "#f59e0b",
    14: "#f97316",
    15: "#ef4444",
}
WAIT_EVENT_IDS = frozenset((8, 9, 10, 11, 13, 14, 15))
AGGREGATE_EVENT_IDS = frozenset((12, 13, 14, 15))
WAIT_INSTRUMENTED_PARENT_IDS = frozenset((1, 2, 4, 5, 7))
EVENT_TRACE_POINTS = {
    1: (
        "DispatchWaveAndGetEndCursor：开始规划并处理当前 wave",
        "当前 wave 的 Dispatch 数据和 ready flag 发布完成",
    ),
    2: (
        "ProcessGmm1Wave：开始当前 expert slice",
        "RunGmm1ActivationByMode 返回（复用内部 FIX_S 完成边界）",
    ),
    3: (
        "ProcessMoeExpertStages：调用 RunGmm1ActivationForExpert 前",
        "RunGmm1ActivationForExpert 返回并完成 GMM1 prologue 的 MTE3_S 后",
    ),
    4: (
        "ProcessGmm1Wave：AIV0 开始当前 expert slice",
        "RunGmm1ActivationByMode 返回后的 Activation 完成边界",
    ),
    5: (
        "ProcessGmm2Wave：开始当前 expert slice",
        "RunMoeExpertGmm2Stage 返回（复用内部 FIX_S 完成边界）",
    ),
    6: (
        "ProcessMoeExpertStages：调用 RunGmm2CombineForExpert 前",
        "RunGmm2CombineForExpert 返回并完成 GMM2 prologue 的 MTE3_S 后",
    ),
    7: (
        "ProcessCombineWave：开始当前 wave 的连续 row ownership 处理",
        "DrainWaveCombineRowRing 返回后的发送完成边界",
    ),
    8: (
        "GMM1 AIC：进入 WaitForGmm1InputReady",
        "Dispatch 输入 ready 后",
    ),
    9: (
        "Activation AIV0：进入 WaitGmm1TileStatus（prefetch 路径）",
        "GMM1 tile ready 后",
    ),
    10: (
        "GMM2 AIC：进入 WaitForGmm2InputReady",
        "Activation 输入 ready 后",
    ),
    11: (
        "Combine AIV1：进入 WaitWaveGmm2GroupReady",
        "GMM2 输出 ready 后",
    ),
    12: (
        "当前 wave 内进入 token/scale/meta 搬运 pipeline（多段累计）",
        "当前 wave 内对应搬运 pipeline 返回（多段累计）",
    ),
    13: (
        "Dispatch ring slot 复用前等待上一轮 MTE3 完成（累计值）",
        "对应 slot 可安全复用（累计值）",
    ),
    14: (
        "Dispatch 小批结束后开始 drain 已使用 ring slot（累计值）",
        "当前小批全部 MTE3 搬运完成（累计值）",
    ),
    15: (
        "发布 GMM1 ready 前执行 MTE3_S 同步（累计值）",
        "MTE3 数据可见，可发布 ready（累计值）",
    ),
}


def interval_is_within(child: dict, parent: dict) -> bool:
    child_begin_offset = (child["begin_cycle"] - parent["begin_cycle"]) & 0xFFFFFFFF
    child_end_offset = (child["end_cycle"] - parent["begin_cycle"]) & 0xFFFFFFFF
    return child_begin_offset <= parent["duration_cycles"] and child_end_offset <= parent["duration_cycles"]


def attach_wait_intervals(events: list[dict]) -> None:
    wait_events = [event for event in events if event["is_wait_event"]]
    for event in events:
        if event["is_wait_event"]:
            event["wait_cycles"] = event["duration_cycles"]
            event["active_cycles"] = 0
            event["wait_event_count"] = 1
            continue
        if event["event_id"] not in WAIT_INSTRUMENTED_PARENT_IDS:
            continue
        matching_waits = [
            wait_event
            for wait_event in wait_events
            if wait_event["parent_event_id"] == event["event_id"]
            and wait_event["wave"] == event["wave"]
            and wait_event["expert"] == event["expert"]
            and wait_event["token_start_in_expert"] == event["token_start_in_expert"]
            and wait_event["token_count"] == event["token_count"]
            and interval_is_within(wait_event, event)
        ]
        event["wait_event_count"] = len(matching_waits)
        event["wait_cycles"] = sum(wait_event["duration_cycles"] for wait_event in matching_waits)
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
        raise ValueError(f"trace magic mismatch: got 0x{words[0]:08x}, expected 0x{MAGIC:08x}")

    header = {
        "version": words[1],
        "header_words": words[2],
        "record_words": words[3],
        "event_words": words[4],
        "record_count": words[5],
        "logical_core_count": words[6],
        "event_capacity_per_record": words[7],
        "bs": words[8],
        "h": words[9],
        "hidden_dim": words[10],
        "top_k": words[11],
        "moe_expert_count": words[12],
        "world_size": words[13],
        "wave_m_group_target": words[14],
        "clock_bits": words[15],
    }
    if header["header_words"] != HEADER_WORDS:
        raise ValueError(f"unsupported header size: {header['header_words']}")
    expected_event_words = {1: 8, 2: 9, 3: 9}
    if header["version"] not in expected_event_words:
        raise ValueError(f"unsupported trace version: {header['version']}")
    if header["event_words"] != expected_event_words[header["version"]]:
        raise ValueError(
            f"version {header['version']} requires {expected_event_words[header['version']]} event words, "
            f"got {header['event_words']}"
        )
    expected_record_count = len(ROLE_NAMES) * header["logical_core_count"]
    if header["record_count"] != expected_record_count:
        raise ValueError(
            f"record count mismatch: got {header['record_count']}, expected {expected_record_count}"
        )
    required_words = header["header_words"] + header["record_count"] * header["record_words"]
    if len(words) < required_words:
        raise ValueError(f"capture has {len(words)} words, protocol requires {required_words}")

    records = []
    for record_index in range(header["record_count"]):
        role_index = record_index // header["logical_core_count"]
        core_index = record_index % header["logical_core_count"]
        record_base = header["header_words"] + record_index * header["record_words"]
        process_begin, process_end, event_count, truncated = words[record_base : record_base + 4]
        if event_count > header["event_capacity_per_record"]:
            raise ValueError(f"record {record_index}: event count {event_count} exceeds capacity")
        events = []
        for event_index in range(event_count):
            base = record_base + RECORD_HEADER_WORDS + event_index * header["event_words"]
            event_words = words[base : base + header["event_words"]]
            event_type, wave_index, expert_index, token_start, token_count, m_group_count = event_words[:6]
            parent_event_id = 0
            is_wait_event = False
            wait_event_count = None
            if header["version"] == 3:
                parent_event_id, begin, end = event_words[6:9]
                is_wait_event = event_type in WAIT_EVENT_IDS
                is_aggregate_event = event_type in AGGREGATE_EVENT_IDS
                wait_cycles = None
            elif header["version"] == 2:
                wait_cycles, begin, end = event_words[6:9]
                wait_cycles = None if wait_cycles == WAIT_CYCLES_UNAVAILABLE else wait_cycles
            else:
                wait_cycles = None
                begin, end = event_words[6:8]
            duration_cycles = (end - begin) & 0xFFFFFFFF
            active_cycles = duration_cycles if wait_cycles is None else max(0, duration_cycles - wait_cycles)
            events.append(
                {
                    "event": EVENT_NAMES.get(event_type, f"unknown({event_type})"),
                    "event_id": event_type,
                    "is_wait_event": is_wait_event,
                    "is_aggregate_event": is_aggregate_event if header["version"] == 3 else False,
                    "parent_event": EVENT_NAMES.get(parent_event_id) if parent_event_id else None,
                    "parent_event_id": parent_event_id,
                    "wave": wave_index,
                    "expert": expert_index,
                    "token_start_in_expert": token_start,
                    "token_count": token_count,
                    "m_group_count": m_group_count,
                    "begin_cycle": begin,
                    "end_cycle": end,
                    "duration_cycles": duration_cycles,
                    "wait_cycles": wait_cycles,
                    "active_cycles": active_cycles,
                    "wait_event_count": wait_event_count,
                    "begin_trace_point": EVENT_TRACE_POINTS.get(event_type, ("未知", "未知"))[0],
                    "end_trace_point": EVENT_TRACE_POINTS.get(event_type, ("未知", "未知"))[1],
                }
            )
        if header["version"] == 3:
            attach_wait_intervals(events)
        records.append(
            {
                "role": ROLE_NAMES[role_index],
                "core": core_index,
                "process_begin_cycle": process_begin,
                "process_end_cycle": process_end,
                "process_duration_cycles": (process_end - process_begin) & 0xFFFFFFFF,
                "truncated": bool(truncated),
                "events": events,
            }
        )
    return {"header": header, "records": records}


def write_csv(trace: dict, path: Path) -> None:
    columns = [
        "role",
        "core",
        "event",
        "event_id",
        "is_wait_event",
        "is_aggregate_event",
        "parent_event",
        "parent_event_id",
        "wave",
        "expert",
        "token_start_in_expert",
        "token_count",
        "m_group_count",
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
                writer.writerow({"role": record["role"], "core": record["core"], **event})


def write_svg(trace: dict, path: Path) -> None:
    role_order = {"AIC": 0, "AIV0": 1, "AIV1": 2}
    records = sorted(
        (record for record in trace["records"] if record["events"]),
        key=lambda record: (record["core"], role_order.get(record["role"], len(role_order))),
    )
    lanes = []
    for record in records:
        wait_events = [event for event in record["events"] if event["is_wait_event"]]
        wait_cycles = (
            sum(event["duration_cycles"] for event in wait_events)
            if wait_events
            else sum(event["wait_cycles"] or 0 for event in record["events"])
        )
        lane_label = f'B{record["core"]} · {record["role"]}'
        lane_tip = f'Process={record["process_duration_cycles"]} cycles, Wait={wait_cycles} cycles'
        lanes.append((record["core"], lane_label, lane_tip,
                      sorted(record["events"], key=lambda event: event["is_wait_event"])))
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
    begin_cycles = [event["begin_cycle"] for record in records for event in record["events"]]
    if not begin_cycles:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="100"/>\n')
        return
    origin = min(begin_cycles)
    relative_end = max(
        (event["end_cycle"] - origin) & 0xFFFFFFFF for record in records for event in record["events"]
    )
    scale = (width - left - right) / max(1, relative_end)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:11px}.label{font-weight:600}.event{stroke:#ffffff;stroke-width:.5}</style>',
        f'<text x="12" y="24" font-size="16" font-weight="700">A8W8 wave timeline (origin={origin})</text>',
    ]
    for block_idx, (_, block_begin, block_end) in enumerate(block_ranges):
        fill = "#f8fafc" if block_idx % 2 == 0 else "#f1f5f9"
        lines.append(
            f'<rect x="4" y="{block_begin - 2}" width="{width - right - 4}" '
            f'height="{block_end - block_begin + 4}" rx="5" fill="{fill}" stroke="#dbe2ea"/>'
        )
    for y, (_, lane_name, lane_tip, lane_events) in zip(lane_rows, lanes):
        lines.append(f'<text class="label" x="10" y="{y + 14}">{lane_name}<title>{lane_tip}</title></text>')
        lines.append(f'<line x1="{left}" y1="{y + 18}" x2="{width - right}" y2="{y + 18}" stroke="#e5e7eb"/>')
        for event in lane_events:
            if event.get("is_aggregate_event"):
                continue
            x = left + (((event["begin_cycle"] - origin) & 0xFFFFFFFF) * scale)
            event_width = max(1.0, event["duration_cycles"] * scale)
            color = EVENT_COLORS.get(event["event_id"], "#64748b")
            title = (
                f'{event["event"]} wave={event["wave"]} expert={event["expert"]} '
                f'tokens=[{event["token_start_in_expert"]},'
                f'{event["token_start_in_expert"] + event["token_count"]}) '
                f'begin={event["begin_cycle"]} end={event["end_cycle"]} '
                f'duration={event["duration_cycles"]} '
                f'parent={event["parent_event"] or "none"} '
                f'begin-point={event["begin_trace_point"]} end-point={event["end_trace_point"]}'
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
<title>MegaMoe A8W8 wave trace</title>
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
.lane {{ position:relative; height:32px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:5px; }}
.bar {{ position:absolute; top:5px; height:20px; min-width:2px; border-radius:3px; opacity:.9; cursor:pointer; }}
.bar.wait-bar {{ top:5px; height:20px; border:1px solid #94a3b8; opacity:.95; z-index:2; }}
.bar:hover {{ opacity:1; outline:2px solid #11182755; z-index:2; }}
.event-detail {{ min-height:54px; margin-top:12px; padding:10px 12px; background:#f8fafc; border:1px solid #e5e7eb;
                 border-radius:6px; color:var(--muted); }}
.event-detail b {{ color:var(--text); }}
.empty {{ padding:30px; text-align:center; color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ padding:7px 8px; border-bottom:1px solid #e5e7eb; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
thead th {{ position:sticky; top:0; background:#f8fafc; z-index:1; }}
.table-scroll {{ max-height:470px; overflow:auto; border:1px solid #e5e7eb; border-radius:6px; }}
.lane-summary-row {{ cursor:pointer; }}
.lane-summary-row:hover {{ background:#f1f5f9; }}
.block-summary-row {{ background:#f8fafc; font-weight:600; }}
.warn {{ color:#b42318; font-weight:600; }}
</style>
</head>
<body><main>
<h1>MegaMoe A8W8 wave trace</h1>
<div class="muted">同一份 rank trace 中的所有 lane 共用一个 SYS_CNT cycle 原点和刻度；跨 rank 报告不合并时间轴。
在当前平台上，1000 cycles = 1 μs。</div>
<section class="cards">
  <div class="card"><span class="muted">Shape</span><b>BS {header['bs']} · H {header['h']}</b></div>
  <div class="card"><span class="muted">Intermediate</span><b>{header['hidden_dim']}</b></div>
  <div class="card"><span class="muted">TopK / World</span><b>{header['top_k']} / {header['world_size']}</b></div>
  <div class="card"><span class="muted">MoE experts</span><b>{header['moe_expert_count']}</b></div>
  <div class="card"><span class="muted">Wave M-group target</span><b>{header['wave_m_group_target']}</b></div>
  <div class="card"><span class="muted">Events / truncated</span><b>{event_count} / {truncated_count}</b></div>
</section>
<section class="panel">
  <h2>流水时间线</h2>
  <div class="controls">
    <label>AI Block <select id="coreSelect"></select></label>
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
      <option value="all">全部 Block</option><option value="5">前 5 个 Block</option>
      <option value="10">前 10 个 Block</option><option value="20">前 20 个 Block</option>
    </select></label>
    <span id="selectionSummary" class="muted"></span>
  </div>
  <div class="legend" id="legend"></div>
  <div class="muted">每个 role/core 只使用一条泳道。父阶段显示完整执行区间，等待按真实 begin/end cycle
  覆盖在同一泳道；Active = Total - 这些真实等待区间。可固定整个 Block 或单条泳道到顶部对比区。</div>
  <div id="timeline"></div>
  <div id="eventDetail" class="event-detail">点击时间线中的事件块查看完整起止 cycle 与打点位置。</div>
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
    <th>AI Block</th><th>Wave</th><th>角色</th><th>阶段</th><th>事件数</th><th>Token 数</th><th>M group</th>
    <th>总 cycles</th><th>Active cycles</th><th>Wait cycles</th><th>Wait 占比</th><th>最长事件</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<section class="panel">
  <h2>打点位置</h2>
  <div class="table-scroll"><table><thead><tr><th>阶段</th><th>Begin 打点</th><th>End 打点/完成边界</th></tr></thead>
  <tbody>{trace_point_rows}</tbody></table></div>
</section>
<section class="panel">
  <h2>事件明细</h2>
  <div class="table-scroll"><table id="eventTable"><thead><tr>
    <th>AI Block</th><th>角色</th><th>事件</th><th>父阶段</th><th>Wave</th><th>Expert</th><th>Token 范围</th><th>M group</th>
    <th>Begin cycle</th><th>End cycle</th><th>总 cycles</th><th>Active cycles</th><th>Wait cycles</th><th>Wait 占比</th>
    <th>总耗时 (μs)</th><th>Begin 打点</th><th>End 打点</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<script id="traceData" type="application/json">{payload}</script>
<script>
const trace = JSON.parse(document.getElementById('traceData').textContent);
const colors = {{Dispatch:'#06b6d4',GMM1:'#2563eb','GMM1 prologue':'#a855f7',Activation:'#22c55e',GMM2:'#ef4444','GMM2 prologue':'#c084fc',Combine:'#f97316','Wait Dispatch':'#cbd5e1','Wait GMM1':'#cbd5e1','Wait Activation':'#cbd5e1','Wait GMM2':'#cbd5e1'}};
const legendColors = {{Dispatch:'#06b6d4',GMM1:'#2563eb','GMM1 prologue':'#a855f7',Activation:'#22c55e',GMM2:'#ef4444','GMM2 prologue':'#c084fc',Combine:'#f97316',Wait:'#cbd5e1'}};
const roles = ['AIC','AIV0','AIV1'];
const coreSelect = document.getElementById('coreSelect');
const waveSelect = document.getElementById('waveSelect');
const waitFilter = document.getElementById('waitFilter');
const laneSort = document.getElementById('laneSort');
const laneLimit = document.getElementById('laneLimit');
const pinnedBlocks = new Set();
const pinnedLanes = new Set();
coreSelect.add(new Option('全部','all'));
for (let core=0; core<trace.header.logical_core_count; core++) coreSelect.add(new Option(String(core),String(core)));
const waves = [...new Set(trace.records.flatMap(r=>r.events.map(e=>e.wave)))].sort((a,b)=>a-b);
for (const wave of waves) waveSelect.add(new Option(String(wave),String(wave)));
document.getElementById('legend').innerHTML = Object.entries(legendColors).map(([name,color])=>`<span><i style="background:${{color}}"></i>${{name}}</span>`).join('');
const cycleModulo = 4294967296;
const cycleAnchor = trace.records.length ? trace.records[0].process_begin_cycle : 0;
function signedCycleDelta(cycle) {{
  const delta=(cycle-cycleAnchor+cycleModulo)%cycleModulo;
  return delta>=cycleModulo/2 ? delta-cycleModulo : delta;
}}
const eventCycles=trace.records.flatMap(r=>r.events.flatMap(e=>[e.begin_cycle,e.end_cycle]));
const domainCycles=eventCycles.length ? eventCycles : trace.records.flatMap(r=>[r.process_begin_cycle,r.process_end_cycle]);
const domainStart=domainCycles.length ? Math.min(...domainCycles.map(signedCycleDelta)) : 0;
const domainEnd=domainCycles.length ? Math.max(...domainCycles.map(signedCycleDelta)) : 1;
const domainSpan=Math.max(domainEnd-domainStart,1);
const relativeCycle=cycle=>signedCycleDelta(cycle)-domainStart;
const absoluteCycle=offset=>((cycleAnchor+domainStart+offset)%cycleModulo+cycleModulo)%cycleModulo;
const cycleDelta=(begin,end)=>(end-begin+cycleModulo)%cycleModulo;
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
    .map(r=>withLaneMetrics({{...r,events:r.events.filter(e=>wave==='all'||e.wave===Number(wave))}}))
  return [...new Set(records.map(r=>r.core))].map(block=>
    withBlockMetrics(records.filter(r=>r.core===block).sort((a,b)=>roles.indexOf(a.role)-roles.indexOf(b.role))))
    .filter(group=>group.records.some(r=>r.events.length));
}}
function selectedBlockGroups() {{
  const core=coreSelect.value;
  const groups=allBlockGroupsForWave()
    .filter(group=>core==='all'||group.core===Number(core))
    .filter(group=>waitFilter.value==='all'||
      (waitFilter.value==='with-wait'&&group.blockMetrics.waitCycles>0)||
      (waitFilter.value==='without-wait'&&group.blockMetrics.waitCycles===0)||
      (waitFilter.value==='truncated'&&group.blockMetrics.truncated))
    .sort(compareBlockGroups);
  return laneLimit.value==='all'?groups:groups.slice(0,Number(laneLimit.value));
}}
function esc(value) {{ return String(value).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function render() {{
  const blockGroups=selectedBlockGroups();
  const records=blockGroups.flatMap(group=>group.records);
  const events=records.flatMap(r=>r.events.map(e=>({{...e,role:r.role,core:r.core}})));
  document.getElementById('selectionSummary').textContent=`${{blockGroups.length}} 个 Block · ${{records.length}} 条泳道 · ${{events.length}} 个事件`;
  renderTimeline(blockGroups,allBlockGroupsForWave(),events);
  renderLaneTable(blockGroups); renderWaveTable(events); renderEventTable(events);
}}
function renderTimeline(blockGroups,pinSourceGroups,events) {{
  const root=document.getElementById('timeline');
  const hasPinnedEvents=pinSourceGroups.some(group=>pinnedBlocks.has(group.core)||
    group.records.some(record=>pinnedLanes.has(laneKey(record.core,record.role))));
  if (!events.length&&!hasPinnedEvents) {{
    root.innerHTML='<div class="empty">当前筛选条件下没有事件</div>'; return;
  }}
  const visibleEvents=[];
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
    let laneOut=`<div class="lane-wrap${{highlightPinned&&lanePinned?' lane-pinned':''}}"><div class="lane-name" title="${{esc(laneTip)}}">`
      +`<button class="pin-button${{lanePinned?' active':''}}" data-pin-lane="${{key}}" title="${{lanePinned?'取消固定':'固定该泳道'}}">📌</button>`
      +`<span class="lane-name-text">${{record.role}}`
      +`<small>P/W ${{(record.process_duration_cycles/1000).toFixed(3)}}/${{(metrics.waitCycles/1000).toFixed(3)}} us</small>`
      +`</span></div><div class="lane">`;
    const laneEvents=record.events.filter(e=>!e.is_aggregate_event)
      .sort((a,b)=>Number(a.is_wait_event)-Number(b.is_wait_event));
    for(const e of laneEvents) {{
        const left=relativeCycle(e.begin_cycle)/domainSpan*100, width=Math.max(e.duration_cycles/domainSpan*100,.12);
        const waitKnown=e.wait_cycles!==null;
        const waitText=waitKnown?`${{e.wait_cycles}} cycles (${{(e.wait_cycles/1000).toFixed(3)}} μs)`:'未拆分';
        const parentText=e.parent_event?` | parent=${{e.parent_event}}`:'';
        const tip=`${{e.event}}${{parentText}} | begin=${{e.begin_cycle}} end=${{e.end_cycle}} total=${{e.duration_cycles}} cycles (${{(e.duration_cycles/1000).toFixed(3)}} μs) | active=${{e.active_cycles}} cycles | wait=${{waitText}} | wave=${{e.wave}} expert=${{e.expert}} | tokens=[${{e.token_start_in_expert}},${{e.token_start_in_expert+e.token_count}}) | begin point=${{e.begin_trace_point}} | end point=${{e.end_trace_point}}`;
        const eventIndex=visibleEvents.push({{event:e,record}})-1;
        laneOut+=`<div class="bar ${{e.is_wait_event?'wait-bar':''}}" data-event-index="${{eventIndex}}" title="${{esc(tip)}}" style="left:${{left}}%;width:${{width}}%;background:${{colors[e.event]||'#64748b'}}"></div>`;
    }}
    return laneOut+'</div></div>';
  }};
  const renderBlockGroup=(group,highlightPinned=false)=>{{
    const blockPinned=pinnedBlocks.has(group.core);
    return `<div class="block-group${{highlightPinned&&blockPinned?' pinned':''}}"><div class="block-header"><span>Block ${{group.core}}</span>`
      +`<button class="pin-button${{blockPinned?' active':''}}" data-pin-block="${{group.core}}" title="${{blockPinned?'取消固定':'固定整个 Block'}}">📌</button></div>`
      +group.records.map(record=>renderLane(record,highlightPinned)).join('')+'</div>';
  }};

  const pinnedBlockGroups=pinSourceGroups.filter(group=>pinnedBlocks.has(group.core));
  const pinnedLaneRecords=pinSourceGroups.flatMap(group=>group.records)
    .filter(record=>pinnedLanes.has(laneKey(record.core,record.role))&&!pinnedBlocks.has(record.core));
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
  root.querySelectorAll('.bar').forEach(bar=>bar.addEventListener('click',()=>showEventDetail(visibleEvents[Number(bar.dataset.eventIndex)])));
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
  const waitText=waitKnown?`${{e.wait_cycles}} cycles (${{(e.wait_cycles/1000).toFixed(3)}} μs)`:'未拆分';
  const waitPercent=waitKnown&&e.duration_cycles?(e.wait_cycles/e.duration_cycles*100).toFixed(1)+'%':'—';
  const parentText=e.parent_event?`　父阶段：<b>${{esc(e.parent_event)}}</b>`:'';
  const intervalText=e.is_aggregate_event?'这是当前 wave 内多个短区间的累计值；为避免事件溢出，不表示一段连续时间轴位置。':
    (e.is_wait_event?'这是由 kernel 直接记录的真实轮询等待区间。':
    `包含 <b>${{e.wait_event_count===null?'—':e.wait_event_count}}</b> 个独立等待区间。`);
  document.getElementById('eventDetail').innerHTML=`<b>Block ${{r.core}} · ${{r.role}} · ${{esc(e.event)}}</b>　`
    +`Wave ${{e.wave}}，Expert ${{e.expert}}，Token [${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})${{parentText}}<br>`
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
      +`<td>${{bm.processDurationCycles.toLocaleString()}} (${{(bm.processDurationCycles/1000).toFixed(3)}} μs)</td>`
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
        +`<td>${{r.process_duration_cycles.toLocaleString()}} (${{(r.process_duration_cycles/1000).toFixed(3)}} μs)</td>`
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
function renderWaveTable(events) {{
  const groups=new Map();
  for(const e of events) {{
    const key=`${{e.core}}|${{e.wave}}|${{e.role}}|${{e.event}}`;
    const g=groups.get(key)||{{core:e.core,wave:e.wave,role:e.role,event:e.event,count:0,tokens:0,mGroups:0,cycles:0,active:0,wait:0,waitKnown:true,max:0}};
    g.count++;
    if(e.is_wait_event) {{ g.tokens=Math.max(g.tokens,e.token_count); g.mGroups=Math.max(g.mGroups,e.m_group_count); }}
    else {{ g.tokens+=e.token_count; g.mGroups+=e.m_group_count; }}
    g.cycles+=e.duration_cycles; g.active+=e.active_cycles;
    if(e.wait_cycles===null) g.waitKnown=false; else g.wait+=e.wait_cycles;
    g.max=Math.max(g.max,e.duration_cycles); groups.set(key,g);
  }}
  const rows=[...groups.values()].sort((a,b)=>a.core-b.core||a.wave-b.wave||roles.indexOf(a.role)-roles.indexOf(b.role)||a.event.localeCompare(b.event));
  document.querySelector('#waveTable tbody').innerHTML=rows.map(g=>{{
    const wait=g.waitKnown?g.wait.toLocaleString():'—', waitPercent=g.waitKnown&&g.cycles?(g.wait/g.cycles*100).toFixed(1)+'%':'—';
    return `<tr><td>${{g.core}}</td><td>${{g.wave}}</td><td>${{g.role}}</td><td>${{g.event}}</td><td>${{g.count}}</td><td>${{g.tokens}}</td><td>${{g.mGroups}}</td><td>${{g.cycles.toLocaleString()}}</td><td>${{g.active.toLocaleString()}}</td><td>${{wait}}</td><td>${{waitPercent}}</td><td>${{g.max.toLocaleString()}}</td></tr>`;
  }}).join('');
}}
function renderEventTable(events) {{
  const rows=[...events].sort((a,b)=>a.core-b.core||a.wave-b.wave||roles.indexOf(a.role)-roles.indexOf(b.role)||signedCycleDelta(a.begin_cycle)-signedCycleDelta(b.begin_cycle));
  document.querySelector('#eventTable tbody').innerHTML=rows.map(e=>{{
    const wait=e.wait_cycles===null?'—':e.wait_cycles.toLocaleString();
    const waitPercent=e.wait_cycles!==null&&e.duration_cycles?(e.wait_cycles/e.duration_cycles*100).toFixed(1)+'%':'—';
    return `<tr><td>${{e.core}}</td><td>${{e.role}}</td><td>${{e.event}}</td><td>${{e.parent_event||'—'}}</td><td>${{e.wave}}</td><td>${{e.expert}}</td><td>[${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})</td><td>${{e.m_group_count}}</td><td>${{e.begin_cycle}}</td><td>${{e.end_cycle}}</td><td>${{e.duration_cycles.toLocaleString()}}</td><td>${{e.active_cycles.toLocaleString()}}</td><td>${{wait}}</td><td>${{waitPercent}}</td><td>${{(e.duration_cycles/1000).toFixed(3)}}</td><td>${{esc(e.begin_trace_point)}}</td><td>${{esc(e.end_trace_point)}}</td></tr>`;
  }}).join('');
}}
for(const control of [coreSelect,waveSelect,waitFilter,laneSort,laneLimit]) control.addEventListener('change',render);
render();
</script>
</main></body></html>
"""
    path.write_text(page, encoding="utf-8")


def capture_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.fullmatch(r"rank(\d+)\.raw", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def collect_captures(inputs: list[Path]) -> list[Path]:
    captures = []
    for input_path in inputs:
        if input_path.is_dir():
            captures.extend(sorted(input_path.glob("rank*.raw"), key=capture_sort_key))
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
        raise ValueError("no rank*.raw captures found")
    return unique_captures


def write_trace_artifacts(capture: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = parse_trace(read_words(capture))
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    write_csv(trace, output_dir / "events.csv")
    write_svg(trace, output_dir / "timeline.svg")
    write_html(trace, output_dir / "index.html")
    return {
        "capture": str(capture),
        "name": capture.stem,
        "header": trace["header"],
        "record_count": len(trace["records"]),
        "event_count": sum(len(record["events"]) for record in trace["records"]),
        "truncated_record_count": sum(record["truncated"] for record in trace["records"]),
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
</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "run-summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="rank raw file(s), or a case directory containing rank*.raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="artifact directory; defaults beside the input capture or case directory",
    )
    args = parser.parse_args()
    captures = collect_captures(args.inputs)
    batch_mode = len(captures) > 1 or any(input_path.is_dir() for input_path in args.inputs)

    if not batch_mode:
        capture = captures[0]
        output_dir = args.output_dir or capture.parent / f"{capture.stem}-trace"
        summary = write_trace_artifacts(capture, output_dir)
        print(f"parsed {summary['event_count']} events from {summary['record_count']} records")
        print(f"truncated records: {summary['truncated_record_count']}")
        print(f"artifacts: {output_dir}")
        return

    default_root = args.inputs[0] if args.inputs[0].is_dir() else captures[0].parent
    output_dir = args.output_dir or default_root / "trace-report"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for capture in captures:
        rank_output_dir = output_dir / capture.stem
        summary = write_trace_artifacts(capture, rank_output_dir)
        summaries.append(summary)
        print(
            f"{capture.name}: {summary['event_count']} events, "
            f"{summary['truncated_record_count']} truncated records"
        )
    write_run_index(summaries, output_dir)
    print(f"parsed {len(summaries)} rank captures")
    print(f"summary: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
