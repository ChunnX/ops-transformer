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

"""Render a normalized AscendC critical-path trace without knowing its raw protocol."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ReportSchema:
    title: str
    roles: tuple[str, ...]
    event_names: dict[int, str]
    event_colors: dict[int, str]
    event_trace_points: dict[int, tuple[str, str]]
    summary_fields: tuple[tuple[str, str], ...] = ()
    iteration_label: str = "Iteration"
    unit_label: str = "Unit"
    range_label: str = "Range"
    work_label: str = "Work"
    block_label: str = "Block"
    cycles_per_us: float | None = None
    wait_color: str = "#cbd5e1"
    event_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    phase_presets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    derived_metric_label: str = "Derived"
    wait_metric_label: str = "Wait"


NORMALIZED_EVENT_FIELDS = (
    "event",
    "event_id",
    "parent_event",
    "parent_event_id",
    "is_wait_event",
    "iteration",
    "unit",
    "range_begin",
    "range_count",
    "work_count",
    "begin_cycle",
    "end_cycle",
    "duration_cycles",
    "active_cycles",
    "wait_cycles",
    "wait_event_count",
    "begin_trace_point",
    "end_trace_point",
)


def cycle_delta(begin_cycle: int, end_cycle: int) -> int:
    return (end_cycle - begin_cycle) & 0xFFFFFFFF


def derive_wait_metrics(trace: dict) -> None:
    """Optional example derivation; call only when this formula matches the event table."""
    scope_fields = ("iteration", "unit", "range_begin", "range_count", "work_count")
    for record in trace["records"]:
        stage_events = [
            event for event in record["events"] if not event["is_wait_event"]
        ]
        wait_events = [event for event in record["events"] if event["is_wait_event"]]
        for stage in stage_events:
            matched_waits = [
                wait
                for wait in wait_events
                if wait["parent_event_id"] == stage["event_id"]
                and all(wait[field] == stage[field] for field in scope_fields)
                and cycle_delta(stage["begin_cycle"], wait["begin_cycle"])
                + wait["duration_cycles"]
                <= stage["duration_cycles"]
            ]
            stage["wait_cycles"] = sum(
                wait["duration_cycles"] for wait in matched_waits
            )
            stage["wait_event_count"] = len(matched_waits)
            stage["active_cycles"] = max(
                stage["duration_cycles"] - stage["wait_cycles"], 0
            )


def validate_normalized_trace(trace: dict, schema: ReportSchema) -> None:
    """Reject incomplete normalized data before producing partially misleading reports."""
    if "logical_core_count" not in trace.get("header", {}):
        raise ValueError("normalized trace header requires logical_core_count")
    for record_index, record in enumerate(trace.get("records", [])):
        missing_record_fields = {
            "role",
            "core",
            "process_begin_cycle",
            "process_end_cycle",
            "process_duration_cycles",
            "truncated",
            "events",
        } - record.keys()
        if missing_record_fields:
            raise ValueError(
                f"record {record_index} missing fields: {sorted(missing_record_fields)}"
            )
        if record["role"] not in schema.roles:
            raise ValueError(
                f"record {record_index} uses unknown role {record['role']!r}"
            )
        for event_index, event in enumerate(record["events"]):
            missing_event_fields = set(NORMALIZED_EVENT_FIELDS) - event.keys()
            if missing_event_fields:
                raise ValueError(
                    f"record {record_index} event {event_index} missing fields: {sorted(missing_event_fields)}"
                )
            if event["event_id"] not in schema.event_names:
                raise ValueError(
                    f"record {record_index} event {event_index} has unknown event id"
                )


def _to_render_model(trace: dict) -> dict:
    """Map generic scope names to the renderer's stable internal presentation slots."""
    records = []
    for record in trace["records"]:
        events = []
        for event in record["events"]:
            events.append(
                {
                    **event,
                    "wave": event["iteration"],
                    "expert": event["unit"],
                    "token_start_in_expert": event["range_begin"],
                    "token_count": event["range_count"],
                    "m_group_count": event["work_count"],
                }
            )
        records.append({**record, "events": events})
    return {"header": dict(trace["header"]), "records": records}


def _format_cycles(cycles: int, schema: ReportSchema) -> str:
    if schema.cycles_per_us is None:
        return f"{cycles:,} cycles"
    return f"{cycles:,} ({cycles / schema.cycles_per_us:.3f} us)"


def write_csv(trace: dict, path: Path) -> None:
    """Write the normalized schema without presentation-slot aliases."""
    columns = [
        "role",
        "core",
        "event",
        "event_id",
        "is_wait_event",
        "parent_event",
        "parent_event_id",
        "iteration",
        "unit",
        "range_begin",
        "range_count",
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


def write_svg(trace: dict, path: Path, schema: ReportSchema) -> None:
    trace = _to_render_model(trace)
    role_order = {role: index for index, role in enumerate(schema.roles)}
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
        lane_label = f"{schema.block_label[0]}{record['core']} · {record['role']}"
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
        (event["end_cycle"] - origin) & 0xFFFFFFFF
        for record in records
        for event in record["events"]
    )
    scale = (width - left - right) / max(1, relative_end)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;font-size:11px}.label{font-weight:600}.event{stroke:#ffffff;stroke-width:.5}</style>",
        f'<text x="12" y="24" font-size="16" font-weight="700">{html.escape(schema.title)} (origin={origin})</text>',
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
            x = left + (((event["begin_cycle"] - origin) & 0xFFFFFFFF) * scale)
            event_width = max(1.0, event["duration_cycles"] * scale)
            color = schema.event_colors.get(event["event_id"], "#64748b")
            title = (
                f"{event['event']} {schema.iteration_label}={event['wave']} "
                f"{schema.unit_label}={event['expert']} {schema.range_label}=["
                f"{event['token_start_in_expert']},"
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


def pack_html_trace(trace: dict, schema: ReportSchema) -> dict:
    """Pack repeated event dictionaries into compact, versioned rows for HTML."""
    records = []
    for record in trace["records"]:
        event_rows = [
            [
                event["event_id"],
                event["parent_event_id"],
                4 if event["is_wait_event"] else 0,
                event["wave"],
                event["expert"],
                event.get("tile", event["token_start_in_expert"]),
                event["token_start_in_expert"],
                event["token_count"],
                event["work_count"],
                event["begin_cycle"],
                event["end_cycle"],
                event["wait_cycles"],
                event["active_cycles"],
                event["wait_event_count"],
                event.get("paired", True),
            ]
            for event in record["events"]
        ]
        records.append(
            [
                schema.roles.index(record["role"]),
                record["core"],
                record["process_begin_cycle"],
                record["process_end_cycle"],
                record.get("rank", trace.get("header", {}).get("rank", 0)),
                record.get("written_count", len(event_rows)),
                record.get("dropped_count", int(bool(record["truncated"]))),
                event_rows,
            ]
        )
    metadata = {
        str(event_id): [
            name,
            *schema.event_trace_points.get(event_id, ("unknown", "unknown")),
        ]
        for event_id, name in schema.event_names.items()
    }
    return {
        "format": "ascendc-critical-path-html-columns-v1",
        "header": {
            **trace["header"],
            "cycles_per_us": schema.cycles_per_us
            or trace["header"].get("cycles_per_us"),
        },
        "warnings": trace.get("warnings", []),
        "event_metadata": metadata,
        "records": records,
    }


def write_html(trace: dict, path: Path, schema: ReportSchema) -> None:
    """Write a self-contained interactive report without external JavaScript dependencies."""
    trace = _to_render_model(trace)
    payload = json.dumps(
        pack_html_trace(trace, schema), ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    header = trace["header"]
    event_count = sum(len(record["events"]) for record in trace["records"])
    truncated_count = sum(record["truncated"] for record in trace["records"])
    cycles_per_us = schema.cycles_per_us or trace["header"].get("cycles_per_us")
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
        f"<tr><td>{html.escape(schema.event_names[event_id])}</td>"
        f"<td>{html.escape(points[0])}</td><td>{html.escape(points[1])}</td></tr>"
        for event_id, points in schema.event_trace_points.items()
    )
    summary_cards = "".join(
        f'<div class="card"><span class="muted">{html.escape(label)}</span>'
        f"<b>{html.escape(str(header.get(key, '—')))}</b></div>"
        for label, key in schema.summary_fields
    )
    event_colors = {
        schema.event_names[event_id]: schema.event_colors.get(event_id, "#64748b")
        for event_id in schema.event_names
    }
    event_groups = schema.event_groups or {"Events": tuple(schema.event_names.values())}
    phase_presets = schema.phase_presets
    phase_options = (
        '<option value="all">全部事件</option>'
        + "".join(
            f'<option value="{html.escape(key)}">{html.escape(key)}</option>'
            for key in phase_presets
        )
        + '<option value="custom" disabled>自定义事件</option>'
    )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(schema.title)}</title>
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
.controls {{ position:relative; display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:14px; }}
select,button {{ padding:6px 9px; border:1px solid #cbd5e1; border-radius:6px; background:#fff; color:var(--text);
                font:inherit; }}
button {{ cursor:pointer; }} button:hover {{ background:#f1f5f9; }}
.event-picker,.wave-picker {{ position:static; }}
.event-picker summary,.wave-picker summary {{ padding:6px 9px; border:1px solid #cbd5e1; border-radius:6px; background:#fff;
                                              cursor:pointer; list-style:none; }}
.event-picker summary::-webkit-details-marker,.wave-picker summary::-webkit-details-marker {{ display:none; }}
.event-picker-menu {{ position:absolute; z-index:20; top:calc(100% + 4px); left:0; width:min(720px,100%); max-width:100%; max-height:520px;
                      overflow:auto; padding:12px; border:1px solid #cbd5e1; border-radius:8px; background:#fff;
                      box-shadow:0 8px 24px #10182824; }}
.wave-picker-menu {{ position:absolute; z-index:21; top:calc(100% + 4px); left:0; width:min(440px,100%); max-height:420px;
                     overflow:auto; padding:12px; border:1px solid #cbd5e1; border-radius:8px; background:#fff;
                     box-shadow:0 8px 24px #10182824; }}
.event-picker-actions,.wave-picker-actions,.wave-range-actions {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
.wave-range-actions {{ align-items:center; padding-bottom:10px; border-bottom:1px solid #e5e7eb; }}
.wave-range-actions label {{ display:flex; align-items:center; gap:5px; font-size:12px; }}
.wave-range-actions input {{ width:64px; padding:5px 6px; border:1px solid #cbd5e1; border-radius:5px; font:inherit; }}
.wave-options {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(90px,1fr)); gap:6px 10px; }}
.wave-option {{ display:block; padding:3px 5px; border-radius:4px; background:#f8fafc; font-size:12px; line-height:1.5; }}
.wave-option input {{ margin-right:5px; vertical-align:middle; }}
.event-groups {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:8px; }}
.event-group {{ padding:8px; border:1px solid #e5e7eb; border-radius:6px; background:#f8fafc; }}
.event-group > label {{ display:block; margin-bottom:5px; font-weight:700; }}
.event-option {{ display:block; margin:3px 0; font-size:11px; line-height:1.4; }}
.event-option input,.event-group input {{ margin-right:5px; vertical-align:middle; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px 16px; color:var(--muted); }}
.legend-item {{ display:inline-flex; align-items:center; padding:4px 7px; border-color:#dbe2ea; color:var(--muted);
                font-size:11px; line-height:1.3; }}
.legend-item i {{ display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; }}
.legend-item:not(.active) {{ opacity:.42; background:#f8fafc; }}
.legend-item.active {{ border-color:#94a3b8; background:#fff; color:var(--text); box-shadow:0 1px 2px #10182814; }}
.legend-item:hover {{ opacity:1; }}
#timeline {{ min-height:140px; overflow-x:auto; }}
.timeline-viewport {{ position:relative; overflow:auto; min-height:140px; resize:vertical; border:1px solid var(--line);
                      border-radius:8px; background:#fff; }}
.timeline-content {{ position:relative; min-width:1050px; }}
.timeline-label-layer {{ position:absolute; left:0; top:0; z-index:3; width:142px; pointer-events:none; }}
.timeline-label {{ position:absolute; left:0; width:142px; display:flex; align-items:center; gap:5px; padding:0 7px;
                   overflow:hidden; border-right:1px solid var(--line); background:#fff; pointer-events:auto; }}
.timeline-label.block {{ background:#eef2f6; color:var(--muted); font-size:11px; font-weight:700; }}
.timeline-label.section {{ justify-content:space-between; background:#fffbeb; color:#92400e; font-size:11px; font-weight:700; }}
.pinned-toggle {{ width:24px; height:22px; padding:0; border-color:#fbbf24; background:#fff; color:#92400e;
                  font-size:12px; line-height:20px; }}
.timeline-label.fixed-pinned {{ z-index:2; }}
.timeline-pinned-divider {{ position:absolute; left:0; z-index:3; width:142px; height:2px; background:#f59e0b;
                            pointer-events:none; }}
.timeline-label.lane-label {{ font-weight:600; }}
.timeline-label.lane-label small {{ display:block; overflow:hidden; color:var(--muted); font-size:9px; font-weight:400;
                                    text-overflow:ellipsis; white-space:nowrap; }}
.timeline-canvas {{ position:absolute; left:142px; top:0; z-index:2; cursor:default; }}
.timeline-measurement {{ display:inline-block; margin-left:10px; color:#175cd3; font-weight:600; }}
.pin-button {{ flex:0 0 auto; width:24px; height:24px; padding:0; border-color:transparent; color:#98a2b3;
               font-size:13px; line-height:22px; }}
.pin-button.active {{ border-color:#f59e0b; background:#fffbeb; color:#b45309; }}
.canvas-tooltip {{ display:none; position:fixed; z-index:100; max-width:min(620px,calc(100vw - 24px)); padding:8px 10px;
                   border:1px solid #cbd5e1; border-radius:6px; background:#fff; color:var(--text);
                   box-shadow:0 8px 24px #10182824; font:11px/1.45 monospace; pointer-events:none; }}
.event-detail {{ min-height:54px; margin-top:12px; padding:10px 12px; background:#f8fafc; border:1px solid #e5e7eb;
                 border-radius:6px; color:var(--muted); }}
.event-detail b {{ color:var(--text); }}
.stats-heading {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }}
.stats-heading h2 {{ margin:0; }}
.empty {{ padding:30px; text-align:center; color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ padding:7px 8px; border-bottom:1px solid #e5e7eb; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
thead th {{ position:sticky; top:0; background:#f8fafc; z-index:1; }}
.table-scroll {{ max-height:470px; overflow:auto; border:1px solid #e5e7eb; border-radius:6px; }}
#eventTable tbody tr:not(.virtual-spacer) {{ height:32px; }}
#eventTable .virtual-spacer td {{ height:0; padding:0; border:0; }}
#waveTable tbody tr:not(.virtual-spacer) {{ height:32px; }}
#waveTable .virtual-spacer td {{ height:0; padding:0; border:0; }}
.lane-summary-row {{ cursor:pointer; }}
.lane-summary-row:hover {{ background:#f1f5f9; }}
.block-summary-row {{ background:#f8fafc; font-weight:600; }}
.warn {{ color:#b42318; font-weight:600; }}
</style>
</head>
<body><main>
<h1>{html.escape(schema.title)}</h1>
<div class="muted">同一份 rank trace 中的所有 lane 共用一个 SYS_CNT cycle 原点和刻度；跨 rank 报告不合并时间轴。
{conversion_note}</div>
<section class="cards">
  {summary_cards}
  <div class="card"><span class="muted">Events / truncated</span><b>{event_count} / {
        truncated_count
    }</b></div>
</section>
<section class="panel"><h2>协议告警</h2><ul>{warnings_html}</ul></section>
<section class="panel">
  <h2>流水时间线</h2>
  <div class="controls">
    <label>显示 {
        html.escape(schema.block_label)
    } <select id="coreSelect"></select></label>
    <details id="wavePicker" class="wave-picker">
      <summary>{
        html.escape(schema.iteration_label)
    }：<span id="waveSelectionCount"></span></summary>
      <div class="wave-picker-menu">
        <div class="wave-picker-actions">
          <button type="button" id="selectAllWaves">全选</button>
          <button type="button" id="clearAllWaves">清空</button>
          <button type="button" id="invertWaves">反选</button>
        </div>
        <div class="wave-range-actions">
          <label>起始 <input type="number" id="waveRangeStart"></label>
          <label>结束 <input type="number" id="waveRangeEnd"></label>
          <button type="button" id="selectWaveRange">仅选范围</button>
          <button type="button" id="addWaveRange">追加范围</button>
        </div>
        <div id="waveOptions" class="wave-options"></div>
      </div>
    </details>
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
      {phase_options}
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
  <div class="muted">点击图例可直接显示或隐藏单个事件；阶段/事件选择器用于批量选择。</div>
  <div class="legend" id="legend"></div>
  <div class="muted">每个 role/core 只使用一条泳道；事件分类、父子关系与派生指标由 adapter 提供。
  可固定整个 {html.escape(schema.block_label)} 或单条泳道到顶部对比区；拖动视口
  右下角可调整高度，固定对比区可单独收起。<span id="timelineMeasurement" class="timeline-measurement">测量：在时间线上水平拖动</span></div>
  <div id="timeline"></div>
  <div id="eventDetail" class="event-detail">点击时间线中的事件块查看完整起止 cycle 与打点位置。</div>
  <div id="canvasTooltip" class="canvas-tooltip"></div>
</section>
<section class="panel">
  <div class="stats-heading">
    <h2>选中阶段/事件统计</h2>
    <label>统计 {
        html.escape(schema.block_label)
    } <select id="statsCoreSelect"></select></label>
  </div>
  <div class="muted"><span id="statsSelectionSummary"></span>。统计 Block 与显示 Block 独立；统计不受显示 Top N 限制，
  但随 Wave、等待和阶段/事件筛选联动。累计值按事件区间求和；父阶段和子等待
  可能重叠，因此不等同于墙钟耗时。覆盖 Span 是选中区间最早 begin 到最晚 end 的时间范围。</div>
  <div class="table-scroll"><table id="selectionStatsTable"><thead><tr>
    <th>阶段组</th><th>事件</th><th>角色</th><th>数量</th><th>总时间</th><th>{
        html.escape(schema.derived_metric_label)
    }</th><th>{html.escape(schema.wait_metric_label)}</th>
    <th>均值</th><th>P50</th><th>P95</th><th>最大</th><th>最早 begin</th><th>最晚 end</th><th>覆盖 Span</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<section class="panel">
  <h2>泳道概览</h2>
  <div class="muted">指标随当前 {html.escape(schema.block_label)}、{
        html.escape(schema.iteration_label)
    } 和等待筛选更新；
  筛选和排序以 {html.escape(schema.block_label)} 为单位，同组角色始终绑定显示。</div>
  <div class="table-scroll"><table id="laneTable"><thead><tr>
    <th>{
        html.escape(schema.block_label)
    }</th><th>角色</th><th>Process begin</th><th>Process end</th><th>Process 总时长</th>
    <th>首事件</th><th>最后事件</th><th>最后结束 cycle</th><th>事件覆盖时长</th>
    <th>累计 Stage</th><th>累计 Wait</th><th>Wait/Stage</th><th>尾部空洞</th><th>事件数</th><th>截断</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<section class="panel">
  <h2>{html.escape(schema.iteration_label)} 汇总</h2>
  <div class="table-scroll" id="waveTableScroll"><table id="waveTable"><thead><tr>
    <th>{html.escape(schema.block_label)}</th><th>{
        html.escape(schema.iteration_label)
    }</th><th>角色</th><th>阶段</th><th>事件数</th><th>{
        html.escape(schema.range_label)
    } 数</th><th>{html.escape(schema.work_label)}</th>
    <th>总时间</th><th>{html.escape(schema.derived_metric_label)}</th><th>{
        html.escape(schema.wait_metric_label)
    }</th><th>Wait 占比</th><th>最长事件</th>
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
    <th>{
        html.escape(schema.block_label)
    }</th><th>角色</th><th>事件</th><th>父阶段</th><th>{
        html.escape(schema.iteration_label)
    }</th><th>{html.escape(schema.unit_label)}</th><th>{
        html.escape(schema.range_label)
    }</th><th>{html.escape(schema.work_label)}</th>
    <th>Begin cycle</th><th>End cycle</th><th>总 cycles</th><th>{
        html.escape(schema.derived_metric_label)
    }</th><th>{html.escape(schema.wait_metric_label)}</th><th>Wait 占比</th>
    <th>总耗时 (μs)</th><th>Begin 打点</th><th>End 打点</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<script id="traceWorkerSource" type="text/plain">
(()=>{{
let columns=null,eventNames={{}},groupNames={{}},groupOrder=[];
const metric=()=>({{count:0,total:0,active:0,wait:0,waitKnown:true,first:Infinity,last:-Infinity,durations:[]}});
const addMetric=(target,index)=>{{
  const duration=columns.duration[index],wait=columns.wait[index];
  target.count++; target.total+=duration; target.active+=columns.active[index];
  if(wait<0) target.waitKnown=false; else target.wait+=wait;
  target.first=Math.min(target.first,columns.begin[index]); target.last=Math.max(target.last,columns.end[index]);
  target.durations.push(duration);
}};
const finishMetric=(value,group,event,role,summary)=>{{
  value.durations.sort((a,b)=>a-b);
  const pick=ratio=>value.durations[Math.ceil((value.durations.length-1)*ratio)];
  return {{group,event,role,summary,count:value.count,total:value.total,active:value.active,
    wait:value.waitKnown?value.wait:null,mean:Math.round(value.total/value.count),p50:pick(.5),p95:pick(.95),
    max:value.durations[value.durations.length-1],first:value.first,last:value.last,span:value.last-value.first}};
}};
const aggregateStats=indices=>{{
  const groups=new Map(),details=new Map();
  for(const index of indices) {{
    const eventId=columns.eventId[index],group=groupNames[eventId]||'其他',role=columns.role[index];
    if(!groups.has(group)) groups.set(group,metric()); addMetric(groups.get(group),index);
    const key=`${{group}}|${{eventId}}|${{role}}`;
    if(!details.has(key)) details.set(key,metric()); addMetric(details.get(key),index);
  }}
  const rows=[];
  for(const group of [...groupOrder,'其他']) {{
    if(!groups.has(group)) continue;
    rows.push(finishMetric(groups.get(group),group,'阶段汇总','全部',true));
    const matching=[...details.entries()].filter(([key])=>key.startsWith(`${{group}}|`)).sort((a,b)=>a[0].localeCompare(b[0]));
    for(const [key,value] of matching) {{
      const parts=key.split('|'),eventId=Number(parts[1]),role=Number(parts[2]);
      rows.push(finishMetric(value,group,eventNames[eventId]||`unknown(${{eventId}})`,role, false));
    }}
  }}
  return rows;
}};
const aggregateWaves=indices=>{{
  const groups=new Map();
  for(const index of indices) {{
    const iteration=Number.isNaN(columns.wave[index])?null:columns.wave[index];
    const key=`${{columns.core[index]}}|${{iteration===null?'none':iteration}}|${{columns.role[index]}}|${{columns.eventId[index]}}`;
    let row=groups.get(key);
    if(!row) {{ row={{core:columns.core[index],wave:iteration,role:columns.role[index],event:eventNames[columns.eventId[index]],count:0,tokens:0,mGroups:0,cycles:0,active:0,wait:0,waitKnown:true,max:0}}; groups.set(key,row); }}
    row.count++;
    if(columns.isWait[index]) {{ row.tokens=Math.max(row.tokens,columns.tokens[index]); row.mGroups=Math.max(row.mGroups,columns.work[index]); }}
    else {{ row.tokens+=columns.tokens[index]; row.mGroups+=columns.work[index]; }}
    row.cycles+=columns.duration[index]; row.active+=columns.active[index];
    if(columns.wait[index]<0) row.waitKnown=false; else row.wait+=columns.wait[index];
    row.max=Math.max(row.max,columns.duration[index]);
  }}
  return [...groups.values()].sort((a,b)=>a.core-b.core||((a.wave??-1)-(b.wave??-1))||a.role-b.role||a.event.localeCompare(b.event));
}};
self.onmessage=message=>{{
  const data=message.data;
  if(data.type==='init') {{
    const count=data.records.reduce((sum,record)=>sum+record[7].length,0);
    columns={{eventId:new Uint16Array(count),role:new Uint8Array(count),core:new Uint16Array(count),wave:new Float64Array(count),
      duration:new Float64Array(count),active:new Float64Array(count),wait:new Float64Array(count),begin:new Float64Array(count),
      end:new Float64Array(count),tokens:new Uint32Array(count),work:new Uint32Array(count),isWait:new Uint8Array(count)}};
    let index=0;
    for(const record of data.records) for(const event of record[7]) {{
      columns.eventId[index]=event[0]; columns.role[index]=record[0]; columns.core[index]=record[1]; columns.wave[index]=event[3]===null?NaN:event[3];
      columns.duration[index]=event[10]-event[9]; columns.active[index]=event[12]; columns.wait[index]=event[11]===null?-1:event[11];
      columns.begin[index]=event[9]; columns.end[index]=event[10]; columns.tokens[index]=event[7]; columns.work[index]=event[8];
      columns.isWait[index]=Boolean(event[2]&4); index++;
    }}
    eventNames=data.eventNames; groupNames=data.groupNames; groupOrder=data.groupOrder;
    self.postMessage({{type:'ready'}}); return;
  }}
  const rows=data.type==='stats'?aggregateStats(data.indices):aggregateWaves(data.indices);
  self.postMessage({{type:data.type,requestId:data.requestId,rows}});
}};
}})();
</script>
<script id="traceData" type="application/json">{payload}</script>
<script>
const packedTrace = JSON.parse(document.getElementById('traceData').textContent);
if(packedTrace.format!=='ascendc-critical-path-html-columns-v1') throw new Error(`unsupported HTML trace format: ${{packedTrace.format}}`);
const packedRoles={json.dumps(schema.roles, ensure_ascii=False)};
const unpackEvent=row=>{{
  const metadata=packedTrace.event_metadata[String(row[0])]||[`unknown(${{row[0]}})`,'未知','未知'];
  return {{
    event:metadata[0],event_id:row[0],parent_event:row[1]?(packedTrace.event_metadata[String(row[1])]||[])[0]||null:null,
    parent_event_id:row[1],flags:row[2],is_wait_event:Boolean(row[2]&4),completion:(row[2]&2)?'submit':'complete',
    trace_only_completion:Boolean(row[2]&8),wave:row[3],expert:row[4],tile:row[5],token_start_in_expert:row[6],
    token_count:row[7],work_count:row[8],begin_cycle:row[9],end_cycle:row[10],duration_cycles:row[10]-row[9],
    wait_cycles:row[11],active_cycles:row[12],wait_event_count:row[13],paired:row[14],
    begin_trace_point:metadata[1],end_trace_point:metadata[2],
  }};
}};
const trace={{header:packedTrace.header,warnings:packedTrace.warnings,records:packedTrace.records.map(row=>({{
  role:packedRoles[row[0]],core:row[1],process_begin_cycle:row[2],process_end_cycle:row[3],rank:row[4],
  process_duration_cycles:row[3]-row[2],written_count:row[5],dropped_count:row[6],truncated:row[6]>0,
  events:row[7].map(unpackEvent),
}}))}};
let nextEventIndex=0;
for(const record of trace.records) {{
  for(const event of record.events) {{ event.role=record.role; event.core=record.core; event._index=nextEventIndex++; }}
}}
const colors = {json.dumps(event_colors, ensure_ascii=False)};
const legendColors = colors;
const eventGroups = {json.dumps(event_groups, ensure_ascii=False)};
const phasePresets = {json.dumps(phase_presets, ensure_ascii=False)};
const eventToGroup = new Map(Object.entries(eventGroups).flatMap(([group,names])=>names.map(name=>[name,group])));
const roles = packedRoles;
const labels = {
        json.dumps(
            {
                "iteration": schema.iteration_label,
                "unit": schema.unit_label,
                "range": schema.range_label,
                "work": schema.work_label,
                "block": schema.block_label,
                "derived": schema.derived_metric_label,
                "wait": schema.wait_metric_label,
            },
            ensure_ascii=False,
        )
    };
let aggregationWorker=null;
let workerRequestId=0;
const latestWorkerRequest={{stats:0,waves:0}};
try {{
  const workerSource=document.getElementById('traceWorkerSource').textContent;
  aggregationWorker=new Worker(URL.createObjectURL(new Blob([workerSource],{{type:'text/javascript'}})));
  aggregationWorker.onmessage=message=>{{
    const data=message.data;
    if(data.type==='stats'&&data.requestId===latestWorkerRequest.stats) applySelectionStatsRows(data.rows);
    if(data.type==='waves'&&data.requestId===latestWorkerRequest.waves) applyWaveRows(data.rows);
  }};
  aggregationWorker.onerror=()=>{{
    aggregationWorker=null;
    renderStatistics();
    renderWaveTable(currentDisplayEvents);
  }};
  const eventNames=Object.fromEntries(Object.entries(packedTrace.event_metadata).map(([id,metadata])=>[id,metadata[0]]));
  const groupNames={{}};
  for(const [id,name] of Object.entries(eventNames)) groupNames[id]=eventToGroup.get(name)||'其他';
  aggregationWorker.postMessage({{type:'init',records:packedTrace.records,eventNames,groupNames,groupOrder:Object.keys(eventGroups)}});
  packedTrace.records=[];
}} catch(error) {{ aggregationWorker=null; }}
function requestAggregation(type,events) {{
  if(!aggregationWorker) return false;
  const indices=new Uint32Array(events.length);
  for(let index=0;index<events.length;index++) indices[index]=events[index]._index;
  const requestId=++workerRequestId;
  latestWorkerRequest[type]=requestId;
  aggregationWorker.postMessage({{type,requestId,indices}},[indices.buffer]);
  return true;
}}
const coreSelect = document.getElementById('coreSelect');
const statsCoreSelect = document.getElementById('statsCoreSelect');
const wavePicker = document.getElementById('wavePicker');
const waveOptionsRoot = document.getElementById('waveOptions');
const waitFilter = document.getElementById('waitFilter');
const laneSort = document.getElementById('laneSort');
const laneLimit = document.getElementById('laneLimit');
const phasePreset = document.getElementById('phasePreset');
const eventPicker = document.getElementById('eventPicker');
const eventGroupsRoot = document.getElementById('eventGroups');
const canvasTooltip = document.getElementById('canvasTooltip');
const eventTableScroll = document.getElementById('eventTableScroll');
const waveTableScroll = document.getElementById('waveTableScroll');
const pinnedBlocks = new Set();
const pinnedLanes = new Set();
let currentBlockGroups=[];
let currentDisplayBlockGroups=[];
let currentDisplayEvents=[];
let timelineResizeObserver=null;
let selectedTimelineEvent=null;
let virtualEventRows=[];
let virtualEventFrame=null;
let virtualWaveRows=[];
let virtualWaveFrame=null;
let deferredRenderGeneration=0;
let visibleEventGeneration=0;
let redrawTimelineSelection=null;
let clearTimelineMeasurement=null;
let timelineViewportHeight=null;
let pinnedPartitionCollapsed=false;
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
const waves = [...new Set(trace.records.flatMap(r=>r.events.map(e=>e.wave)).filter(wave=>wave!==null))].sort((a,b)=>a-b);
const selectedWaves = new Set(waves);
const waveRangeStart = document.getElementById('waveRangeStart');
const waveRangeEnd = document.getElementById('waveRangeEnd');
if(waves.length) {{
  waveRangeStart.min=waveRangeEnd.min=waves[0];
  waveRangeStart.max=waveRangeEnd.max=waves[waves.length-1];
  waveRangeStart.value=waves[0];
  waveRangeEnd.value=waves[waves.length-1];
}}
let lastWaveClicked=null;
function updateWaveSelection() {{
  const count=selectedWaves.size;
  document.getElementById('waveSelectionCount').textContent=count===waves.length?`全部 (${{count}})`:`${{count}} / ${{waves.length}}`;
}}
function renderWavePicker() {{
  waveOptionsRoot.innerHTML=waves.map(wave=>`<label class="wave-option"><input type="checkbox" data-wave="${{wave}}" ${{selectedWaves.has(wave)?'checked':''}}>${{labels.iteration}} ${{wave}}</label>`).join('');
  waveOptionsRoot.querySelectorAll('[data-wave]').forEach(input=>input.addEventListener('click',event=>{{
    const wave=Number(input.dataset.wave);
    if(event.shiftKey&&lastWaveClicked!==null) {{
      const begin=Math.min(waves.indexOf(lastWaveClicked),waves.indexOf(wave));
      const end=Math.max(waves.indexOf(lastWaveClicked),waves.indexOf(wave));
      for(const rangeWave of waves.slice(begin,end+1)) input.checked?selectedWaves.add(rangeWave):selectedWaves.delete(rangeWave);
    }} else input.checked?selectedWaves.add(wave):selectedWaves.delete(wave);
    lastWaveClicked=wave;
    updateWaveSelection(); render();
  }}));
  updateWaveSelection();
}}
function applyWaveRange(additive) {{
  let begin=Number(waveRangeStart.value),end=Number(waveRangeEnd.value);
  if(!Number.isFinite(begin)||!Number.isFinite(end)) return;
  if(begin>end) [begin,end]=[end,begin];
  if(!additive) selectedWaves.clear();
  for(const wave of waves) if(wave>=begin&&wave<=end) selectedWaves.add(wave);
  renderWavePicker(); render();
}}
document.getElementById('selectAllWaves').addEventListener('click',()=>{{
  for(const wave of waves) selectedWaves.add(wave); renderWavePicker(); render();
}});
document.getElementById('clearAllWaves').addEventListener('click',()=>{{
  selectedWaves.clear(); renderWavePicker(); render();
}});
document.getElementById('invertWaves').addEventListener('click',()=>{{
  for(const wave of waves) selectedWaves.has(wave)?selectedWaves.delete(wave):selectedWaves.add(wave);
  renderWavePicker(); render();
}});
document.getElementById('selectWaveRange').addEventListener('click',()=>applyWaveRange(false));
document.getElementById('addWaveRange').addEventListener('click',()=>applyWaveRange(true));
wavePicker.addEventListener('toggle',()=>{{ if(wavePicker.open) eventPicker.open=false; }});
eventPicker.addEventListener('toggle',()=>{{ if(eventPicker.open) wavePicker.open=false; }});
document.addEventListener('pointerdown',event=>{{
  if(wavePicker.open&&!wavePicker.contains(event.target)) wavePicker.open=false;
  if(eventPicker.open&&!eventPicker.contains(event.target)) eventPicker.open=false;
}});
document.addEventListener('keydown',event=>{{
  if(event.key==='Escape') {{
    wavePicker.open=false; eventPicker.open=false;
    if(clearTimelineMeasurement) clearTimelineMeasurement();
  }}
}});
renderWavePicker();
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
const allEventsInTableOrder=trace.records.flatMap(record=>record.events).sort((a,b)=>
  a.core-b.core||a.wave-b.wave||roles.indexOf(a.role)-roles.indexOf(b.role)||a.begin_cycle-b.begin_cycle);
const durationText=cycles=>cyclesPerUs?`${{cycles.toLocaleString()}} cycles (${{(cycles/cyclesPerUs).toFixed(3)}} μs)`:`${{cycles.toLocaleString()}} cycles`;
const compactDurationValue=cycles=>cyclesPerUs?(cycles/cyclesPerUs).toFixed(1):cycles.toLocaleString();
const compactDurationUnit=cyclesPerUs?'μs':'cycles';
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
  const records=trace.records
    .map(r=>withLaneMetrics({{...r,events:r.events.filter(e=>
      selectedEvents.has(e.event)&&(e.wave===null||selectedWaves.has(e.wave)))}}))
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
  const statsBlockText=statsCoreSelect.value==='all'?`全部可见 ${{labels.block}}`:`${{labels.block}} ${{statsCoreSelect.value}}`;
  document.getElementById('statsSelectionSummary').textContent=`统计 ${{statsBlockText}}：${{statsBlockGroups.length}} 个 ${{labels.block}} · ${{statsEvents.length}} 个事件`;
  renderSelectionStats(statsEvents);
}}
function scheduleDeferredPanels(events,refreshStats) {{
  const generation=++deferredRenderGeneration;
  requestAnimationFrame(()=>{{
    if(generation!==deferredRenderGeneration) return;
    if(refreshStats) renderStatistics();
    requestAnimationFrame(()=>{{
      if(generation!==deferredRenderGeneration) return;
      renderWaveTable(events);
      requestAnimationFrame(()=>{{
        if(generation!==deferredRenderGeneration) return;
        renderEventTable(events);
      }});
    }});
  }});
}}
function renderDisplay(refreshStats=false) {{
  currentDisplayBlockGroups=selectedBlockGroups(currentBlockGroups,coreSelect.value,true);
  const records=currentDisplayBlockGroups.flatMap(group=>group.records);
  currentDisplayEvents=records.flatMap(r=>r.events);
  document.getElementById('selectionSummary').textContent=`${{currentDisplayBlockGroups.length}} 个 ${{labels.block}} · ${{records.length}} 条泳道 · ${{currentDisplayEvents.length}} 个事件 · ${{selectedEvents.size}} 种事件`;
  const legend=document.getElementById('legend');
  legend.innerHTML = Object.entries(legendColors)
    .filter(([name])=>availableEvents.has(name))
    .map(([name,color])=>{{
      const active=selectedEvents.has(name);
      return `<button type="button" class="legend-item${{active?' active':''}}" data-legend-event="${{name}}" aria-pressed="${{active}}"><i style="background:${{color}}"></i>${{name}}</button>`;
    }}).join('');
  legend.querySelectorAll('[data-legend-event]').forEach(button=>button.addEventListener('click',()=>{{
    const name=button.dataset.legendEvent;
    selectedEvents.has(name)?selectedEvents.delete(name):selectedEvents.add(name);
    phasePreset.value='custom';
    renderEventPicker(); render();
  }}));
  renderTimeline(currentDisplayBlockGroups,currentBlockGroups,currentDisplayEvents);
  renderLaneTable(currentDisplayBlockGroups);
  scheduleDeferredPanels(currentDisplayEvents,refreshStats);
}}
function render() {{
  currentBlockGroups=allBlockGroupsForWave();
  renderDisplay(true);
}}
function renderTimeline(blockGroups,pinSourceGroups,events) {{
  const root=document.getElementById('timeline');
  const measurementStatus=document.getElementById('timelineMeasurement');
  canvasTooltip.style.display='none';
  measurementStatus.textContent='测量：在时间线上水平拖动';
  redrawTimelineSelection=null;
  clearTimelineMeasurement=null;
  if(timelineResizeObserver) {{ timelineResizeObserver.disconnect(); timelineResizeObserver=null; }}
  const pinnedBlockGroups=pinSourceGroups.filter(group=>pinnedBlocks.has(group.core));
  const pinnedLaneRecords=pinSourceGroups.flatMap(group=>group.records)
    .filter(record=>pinnedLanes.has(laneKey(record.core,record.role))&&!pinnedBlocks.has(record.core));
  const allPinnedEvents=[...pinnedBlockGroups.flatMap(group=>group.records.flatMap(record=>record.events)),
    ...pinnedLaneRecords.flatMap(record=>record.events)];
  const hasPinnedItems=pinnedBlockGroups.length>0||pinnedLaneRecords.length>0;
  const pinnedEvents=pinnedPartitionCollapsed?[]:allPinnedEvents;
  if (!events.length&&!hasPinnedItems) {{
    selectedTimelineEvent=null;
    resetEventDetail();
    root.innerHTML='<div class="empty">当前筛选条件下没有事件</div>'; return;
  }}
  const visibleTimelineEvents=[...events,...pinnedEvents];
  const domainEvents=visibleTimelineEvents.length?visibleTimelineEvents:allPinnedEvents;
  if(selectedTimelineEvent&&!visibleTimelineEvents.includes(selectedTimelineEvent)) {{
    selectedTimelineEvent=null;
    resetEventDetail();
  }}
  let domainStart=Infinity,domainEnd=-Infinity;
  for(const event of domainEvents) {{
    domainStart=Math.min(domainStart,signedCycleDelta(event.begin_cycle));
    domainEnd=Math.max(domainEnd,signedCycleDelta(event.end_cycle));
  }}
  const domainSpan=Math.max(domainEnd-domainStart,1);
  const relativeCycle=cycle=>signedCycleDelta(cycle)-domainStart;
  const absoluteCycle=offset=>cycleAnchor+domainStart+offset;
  const axisHeight=32,blockHeight=28,laneHeight=38,sectionHeight=28,labelWidth=142;
  const rows=[];
  let rowTop=axisHeight;
  const appendRow=row=>{{ row.top=rowTop; rows.push(row); rowTop+=row.height; }};
  const appendBlock=(group,pinned=false,fixed=false)=>{{
    appendRow({{type:'block',core:group.core,pinned,fixed,height:blockHeight}});
    for(const record of group.records) appendRow({{type:'lane',record,pinned,fixed,height:laneHeight}});
  }};
  if(hasPinnedItems) {{
    appendRow({{type:'section',label:'固定对比',fixed:true,height:sectionHeight}});
    if(!pinnedPartitionCollapsed) {{
      for(const group of pinnedBlockGroups) appendBlock(group,true,true);
      for(const record of pinnedLaneRecords) {{
        appendRow({{type:'block',core:record.core,pinned:true,fixed:true,height:blockHeight}});
        appendRow({{type:'lane',record,pinned:true,fixed:true,height:laneHeight}});
      }}
    }}
  }}
  const pinnedBottom=rowTop;
  const hasPinnedPartition=pinnedBottom>axisHeight;
  for(const group of blockGroups) appendBlock(group,false);
  const contentHeight=Math.max(rowTop,140);
  const defaultViewportHeight=Math.min(Math.max(contentHeight,140),650);
  const viewportHeight=timelineViewportHeight??defaultViewportHeight;
  const laneLabelsHtml=rows.map(row=>{{
    const fixedClass=row.fixed?' fixed-pinned':'';
    if(row.type==='section') return `<div class="timeline-label section${{fixedClass}}" style="top:${{row.top}}px;height:${{row.height}}px"><span>${{row.label}}</span>`
      +`<button type="button" class="pinned-toggle" data-toggle-pinned title="${{pinnedPartitionCollapsed?'展开固定对比区':'收起固定对比区'}}">${{pinnedPartitionCollapsed?'▸':'▾'}}</button></div>`;
    if(row.type==='block') {{
      const active=pinnedBlocks.has(row.core);
      return `<div class="timeline-label block${{fixedClass}}" style="top:${{row.top}}px;height:${{row.height}}px"><span>${{labels.block}} ${{row.core}}</span>`
        +`<button class="pin-button${{active?' active':''}}" data-pin-block="${{row.core}}" title="${{active?'取消固定':'固定整个 Block'}}">📌</button></div>`;
    }}
    const record=row.record,metrics=record.laneMetrics,key=laneKey(record.core,record.role),active=pinnedLanes.has(key);
    const last=metrics.lastEvent;
    const tip=`Process begin=${{record.process_begin_cycle}} end=${{record.process_end_cycle}} total=${{record.process_duration_cycles}} cycles | event span=${{metrics.eventSpanCycles}} | wait=${{metrics.waitCycles}} | last=${{last?last.event:'none'}}@${{last?last.end_cycle:'—'}} | tail=${{metrics.tailCycles}}`;
    return `<div class="timeline-label lane-label${{fixedClass}}" style="top:${{row.top}}px;height:${{row.height}}px" title="${{esc(tip)}}">`
      +`<button class="pin-button${{active?' active':''}}" data-pin-lane="${{key}}" title="${{active?'取消固定':'固定该泳道'}}">📌</button>`
      +`<span>${{record.role}}<small>Span/W ${{compactDurationValue(metrics.eventSpanCycles)}}/${{compactDurationValue(metrics.waitCycles)}} ${{compactDurationUnit}}</small></span></div>`;
  }}).join('');
  const pinnedDivider=hasPinnedPartition?`<div class="timeline-pinned-divider" style="top:${{pinnedBottom-2}}px"></div>`:'';
  root.innerHTML=`<div class="timeline-viewport" style="height:${{viewportHeight}}px"><div class="timeline-content" style="height:${{contentHeight}}px">`
    +`<div class="timeline-label-layer">${{laneLabelsHtml}}${{pinnedDivider}}</div><canvas class="timeline-canvas"></canvas></div></div>`;
  const viewport=root.querySelector('.timeline-viewport'),content=root.querySelector('.timeline-content');
  const labelLayer=root.querySelector('.timeline-label-layer'),canvas=root.querySelector('.timeline-canvas');
  const eventTooltipHtml=(event,record)=>{{
    const waitText=event.wait_cycles===null?'未拆分':durationText(event.wait_cycles);
    const parentText=event.parent_event?` · parent=${{esc(event.parent_event)}}`:'';
    return `<b>${{labels.block}} ${{record.core}} · ${{record.role}} · ${{esc(event.event)}}</b>${{parentText}}<br>`
      +`begin=${{event.begin_cycle}} · end=${{event.end_cycle}} · total=${{durationText(event.duration_cycles)}}<br>`
      +`active=${{event.active_cycles}} cycles · wait=${{waitText}}<br>`
      +`${{labels.iteration}}=${{event.wave}} · ${{labels.unit}}=${{event.expert}} · ${{labels.range}}=[${{event.token_start_in_expert}},${{event.token_start_in_expert+event.token_count}})<br>`
      +`completion=${{esc(event.completion)}} · trace-only=${{event.trace_only_completion}}<br>`
      +`begin point=${{esc(event.begin_trace_point)}}<br>end point=${{esc(event.end_trace_point)}}`;
  }};
  let hoveredEvent=null;
  let hitBoxes=[];
  let measurement=null,measurementDrag=null,suppressTimelineClick=false,canvasCssWidth=1;
  const measurementText=value=>{{
    const cycles=Math.abs(Math.round(value));
    return `Δ ${{cycles.toLocaleString()}} cycles${{cyclesPerUs?` · ${{(cycles/cyclesPerUs).toFixed(3)}} μs`:''}}`;
  }};
  const updateMeasurementStatus=()=>{{
    measurementStatus.textContent=measurement?
      `测量：${{Math.min(measurement.startCycle,measurement.endCycle).toLocaleString()}} → ${{Math.max(measurement.startCycle,measurement.endCycle).toLocaleString()}} · ${{measurementText(measurement.endCycle-measurement.startCycle)}}`:
      '测量：在时间线上水平拖动';
  }};
  const emphasizeEvent=(context,box,selected)=>{{
    const width=Math.max(2,box.right-box.left),height=box.bottom-box.top;
    context.save();
    context.globalAlpha=1;
    context.fillStyle=colors[box.event.event]||'#64748b';
    context.shadowColor=selected?'#0f172acc':'#33415588';
    context.shadowBlur=selected?7:4;
    context.shadowOffsetY=selected?2:1;
    context.fillRect(box.left,box.top-1,width,height+2);
    context.shadowColor='transparent';
    context.lineWidth=selected?2:1.5;
    context.strokeStyle='#ffffff';
    context.strokeRect(box.left+.5,box.top-.5,Math.max(0,width-1),height+1);
    context.lineWidth=1;
    context.strokeStyle=selected?'#0f172a':'#334155';
    context.strokeRect(box.left+1.5,box.top+.5,Math.max(0,width-3),Math.max(0,height-1));
    if(selected) {{
      const markerX=box.left+Math.min(width/2,8);
      context.fillStyle='#0f172a';
      context.beginPath(); context.moveTo(markerX-4,box.top-4); context.lineTo(markerX+4,box.top-4); context.lineTo(markerX,box.top); context.closePath(); context.fill();
    }}
    context.restore();
  }};
  const drawCanvas=()=>{{
    const width=Math.max(1,content.clientWidth-labelWidth),height=Math.max(1,viewport.clientHeight),dpr=window.devicePixelRatio||1;
    canvasCssWidth=width;
    canvas.style.width=`${{width}}px`; canvas.style.height=`${{height}}px`;
    const pixelWidth=Math.max(1,Math.round(width*dpr)),pixelHeight=Math.max(1,Math.round(height*dpr));
    if(canvas.width!==pixelWidth||canvas.height!==pixelHeight) {{ canvas.width=pixelWidth; canvas.height=pixelHeight; }}
    const context=canvas.getContext('2d');
    context.setTransform(dpr,0,0,dpr,0,0);
    context.clearRect(0,0,width,height);
    hitBoxes=[];
    const scrollTop=viewport.scrollTop;
    for(let fixedPass=0;fixedPass<2;fixedPass++) {{
      const drawFixed=Boolean(fixedPass);
      context.save();
      if(!drawFixed) {{
        context.beginPath(); context.rect(0,hasPinnedPartition?pinnedBottom:axisHeight,width,
          Math.max(0,height-(hasPinnedPartition?pinnedBottom:axisHeight))); context.clip();
      }}
      const visibleTop=drawFixed?axisHeight:(hasPinnedPartition?pinnedBottom:axisHeight);
      for(const row of rows) {{
        if(Boolean(row.fixed)!==drawFixed) continue;
        const y=row.fixed?row.top:row.top-scrollTop;
        if(y+row.height<visibleTop||y>height) continue;
        if(row.type==='block'||row.type==='section') {{
          context.fillStyle=row.type==='section'?'#fffbeb':'#eef2f6'; context.fillRect(0,y,width,row.height-1); continue;
        }}
        context.fillStyle=row.pinned?'#fffbeb':'#f8fafc'; context.fillRect(0,y+2,width,row.height-4);
        context.strokeStyle=row.pinned?'#f59e0b':'#e5e7eb'; context.strokeRect(.5,y+2.5,width-1,row.height-5);
        for(let waitPass=0;waitPass<2;waitPass++) {{
          for(const event of row.record.events) {{
            if(Number(event.is_wait_event)!==waitPass) continue;
            const left=relativeCycle(event.begin_cycle)/domainSpan*width;
            const visualWidth=Math.max(1,event.duration_cycles/domainSpan*width),top=y+7,eventHeight=row.height-14;
            context.globalAlpha=event.is_wait_event?.95:.9;
            context.fillStyle=colors[event.event]||'#64748b';
            context.fillRect(left,top,visualWidth,eventHeight);
            if(event.is_wait_event) {{
              context.globalAlpha=1; context.strokeStyle='#94a3b8'; context.lineWidth=1;
              context.strokeRect(left+.5,top+.5,Math.max(0,visualWidth-1),Math.max(0,eventHeight-1));
            }}
            const hitTop=Math.max(top,visibleTop),hitBottom=Math.min(top+eventHeight,height);
            if(hitBottom>hitTop) hitBoxes.push({{left,right:left+visualWidth,top:hitTop,bottom:hitBottom,event,record:row.record}});
          }}
        }}
      }}
      context.restore();
    }}
    if(hasPinnedPartition) {{ context.fillStyle='#f59e0b'; context.fillRect(0,pinnedBottom-2,width,2); }}
    context.globalAlpha=1;
    const selectedBox=hitBoxes.find(box=>box.event===selectedTimelineEvent);
    if(selectedBox) emphasizeEvent(context,selectedBox,true);
    const hoveredBox=hitBoxes.find(box=>box.event===hoveredEvent);
    if(hoveredBox&&hoveredBox!==selectedBox) emphasizeEvent(context,hoveredBox,false);
    context.globalAlpha=1; context.fillStyle='#ffffff'; context.fillRect(0,0,width,axisHeight);
    context.strokeStyle='#dbe2ea'; context.beginPath(); context.moveTo(0,axisHeight-.5); context.lineTo(width,axisHeight-.5); context.stroke();
    context.fillStyle='#667085'; context.font='10px Arial';
    for(let tick=0;tick<=5;tick++) {{
      const x=width*tick/5,offset=Math.round(domainSpan*tick/5),cycle=Math.round(absoluteCycle(offset));
      context.strokeStyle='#98a2b3'; context.beginPath(); context.moveTo(x+.5,axisHeight-7); context.lineTo(x+.5,axisHeight); context.stroke();
      context.fillText(`${{cycle.toLocaleString()}}  +${{offset.toLocaleString()}}`,Math.min(x+3,width-125),12);
    }}
    if(measurement) {{
      const startX=relativeCycle(measurement.startCycle)/domainSpan*width;
      const endX=relativeCycle(measurement.endCycle)/domainSpan*width;
      const left=Math.min(startX,endX),right=Math.max(startX,endX);
      context.fillStyle='#2e90fa22';
      context.fillRect(left,axisHeight,Math.max(1,right-left),Math.max(0,height-axisHeight));
      context.strokeStyle='#1570ef'; context.lineWidth=1.5;
      for(const x of [startX,endX]) {{
        context.beginPath(); context.moveTo(x+.5,axisHeight); context.lineTo(x+.5,height); context.stroke();
      }}
      const label=measurementText(measurement.endCycle-measurement.startCycle);
      context.font='bold 11px Arial';
      const measureLabelWidth=context.measureText(label).width+12;
      const labelX=Math.max(2,Math.min((left+right-measureLabelWidth)/2,width-measureLabelWidth-2));
      context.fillStyle='#175cd3'; context.fillRect(labelX,axisHeight+4,measureLabelWidth,20);
      context.fillStyle='#fff'; context.fillText(label,labelX+6,axisHeight+18);
    }}
  }};
  redrawTimelineSelection=drawCanvas;
  const hitTest=(x,y)=>{{
    for(let index=hitBoxes.length-1;index>=0;index--) {{
      const box=hitBoxes[index];
      if(x>=box.left&&x<=box.right&&y>=box.top&&y<=box.bottom) return box;
    }}
    return null;
  }};
  const moveTooltip=mouseEvent=>{{
    const bounds=canvas.getBoundingClientRect(),box=hitTest(mouseEvent.clientX-bounds.left,mouseEvent.clientY-bounds.top);
    const hit=box?box.event:null;
    if(hoveredEvent!==hit) {{
      hoveredEvent=hit;
      drawCanvas();
    }}
    canvas.style.cursor=hit?'pointer':'crosshair';
    if(!hit) {{ canvasTooltip.style.display='none'; return; }}
    canvasTooltip.innerHTML=eventTooltipHtml(hit,box.record);
    canvasTooltip.style.display='block';
    const tooltipWidth=canvasTooltip.offsetWidth,tooltipHeight=canvasTooltip.offsetHeight;
    canvasTooltip.style.left=`${{Math.max(8,Math.min(mouseEvent.clientX+12,window.innerWidth-tooltipWidth-8))}}px`;
    canvasTooltip.style.top=`${{Math.max(8,Math.min(mouseEvent.clientY+12,window.innerHeight-tooltipHeight-8))}}px`;
  }};
  const pointerCycle=pointerEvent=>{{
    const bounds=canvas.getBoundingClientRect();
    const x=Math.max(0,Math.min(pointerEvent.clientX-bounds.left,canvasCssWidth));
    return Math.round(absoluteCycle(domainSpan*x/canvasCssWidth));
  }};
  canvas.addEventListener('pointerdown',event=>{{
    if(event.button!==0) return;
    measurementDrag={{pointerId:event.pointerId,startX:event.clientX,startCycle:pointerCycle(event),moved:false}};
    canvas.setPointerCapture(event.pointerId);
    canvasTooltip.style.display='none'; hoveredEvent=null; canvas.style.cursor='crosshair';
    event.preventDefault();
  }});
  canvas.addEventListener('pointermove',event=>{{
    if(!measurementDrag) {{ moveTooltip(event); return; }}
    if(Math.abs(event.clientX-measurementDrag.startX)>=3) measurementDrag.moved=true;
    if(!measurementDrag.moved) return;
    measurement={{startCycle:measurementDrag.startCycle,endCycle:pointerCycle(event)}};
    updateMeasurementStatus(); drawCanvas();
  }});
  const finishMeasurement=event=>{{
    if(!measurementDrag||event.pointerId!==measurementDrag.pointerId) return;
    const moved=measurementDrag.moved;
    if(canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    measurementDrag=null;
    if(moved) {{
      suppressTimelineClick=true;
      setTimeout(()=>{{ suppressTimelineClick=false; }},0);
      updateMeasurementStatus(); drawCanvas();
    }}
  }};
  canvas.addEventListener('pointerup',finishMeasurement);
  canvas.addEventListener('pointercancel',finishMeasurement);
  canvas.addEventListener('mouseleave',()=>{{
    canvasTooltip.style.display='none'; canvas.style.cursor='crosshair';
    if(hoveredEvent) {{ hoveredEvent=null; drawCanvas(); }}
  }});
  canvas.addEventListener('click',event=>{{
    if(suppressTimelineClick) {{ suppressTimelineClick=false; return; }}
    const bounds=canvas.getBoundingClientRect(),box=hitTest(event.clientX-bounds.left,event.clientY-bounds.top);
    if(!box) {{ measurement=null; updateMeasurementStatus(); drawCanvas(); clearTimelineSelection(); return; }}
    if(selectedTimelineEvent===box.event) {{ clearTimelineSelection(); return; }}
    selectedTimelineEvent=box.event;
    drawCanvas();
    showEventDetail({{event:box.event,record:box.record}});
  }});
  clearTimelineMeasurement=()=>{{
    if(measurementDrag&&canvas.hasPointerCapture(measurementDrag.pointerId))
      canvas.releasePointerCapture(measurementDrag.pointerId);
    measurement=null; measurementDrag=null; updateMeasurementStatus(); drawCanvas();
  }};
  updateMeasurementStatus();
  let scrollFrame=null;
  const syncViewport=()=>{{
    canvas.style.transform=`translateY(${{viewport.scrollTop}}px)`;
    labelLayer.style.transform=`translateX(${{viewport.scrollLeft}}px)`;
    labelLayer.querySelectorAll('.fixed-pinned,.timeline-pinned-divider').forEach(label=>{{
      label.style.transform=`translateY(${{viewport.scrollTop}}px)`;
    }});
    drawCanvas();
  }};
  viewport.addEventListener('scroll',()=>{{
    canvas.style.transform=`translateY(${{viewport.scrollTop}}px)`;
    labelLayer.style.transform=`translateX(${{viewport.scrollLeft}}px)`;
    labelLayer.querySelectorAll('.fixed-pinned,.timeline-pinned-divider').forEach(label=>{{
      label.style.transform=`translateY(${{viewport.scrollTop}}px)`;
    }});
    if(scrollFrame!==null) return;
    scrollFrame=requestAnimationFrame(()=>{{ scrollFrame=null; drawCanvas(); }});
  }});
  requestAnimationFrame(syncViewport);
  if('ResizeObserver' in window) {{
    timelineResizeObserver=new ResizeObserver(()=>{{
      timelineViewportHeight=Math.round(viewport.getBoundingClientRect().height);
      requestAnimationFrame(syncViewport);
    }});
    timelineResizeObserver.observe(viewport);
  }}
  root.querySelectorAll('[data-toggle-pinned]').forEach(button=>button.addEventListener('click',()=>{{
    pinnedPartitionCollapsed=!pinnedPartitionCollapsed;
    renderTimeline(currentDisplayBlockGroups,currentBlockGroups,currentDisplayEvents);
  }}));
  root.querySelectorAll('[data-pin-block]').forEach(button=>button.addEventListener('click',()=>{{
    const core=Number(button.dataset.pinBlock);
    if(pinnedBlocks.has(core)) pinnedBlocks.delete(core); else {{ pinnedBlocks.add(core); pinnedPartitionCollapsed=false; }}
    renderTimeline(currentDisplayBlockGroups,currentBlockGroups,currentDisplayEvents);
  }}));
  root.querySelectorAll('[data-pin-lane]').forEach(button=>button.addEventListener('click',()=>{{
    const key=button.dataset.pinLane;
    if(pinnedLanes.has(key)) pinnedLanes.delete(key); else {{ pinnedLanes.add(key); pinnedPartitionCollapsed=false; }}
    renderTimeline(currentDisplayBlockGroups,currentBlockGroups,currentDisplayEvents);
  }}));
}}
function resetEventDetail() {{
  document.getElementById('eventDetail').textContent='点击时间线中的事件块查看完整起止 cycle 与打点位置。';
}}
function clearTimelineSelection() {{
  if(!selectedTimelineEvent) return;
  selectedTimelineEvent=null;
  resetEventDetail();
  if(redrawTimelineSelection) redrawTimelineSelection();
}}
document.addEventListener('keydown',event=>{{ if(event.key==='Escape') clearTimelineSelection(); }});
function showEventDetail(item) {{
  const e=item.event, r=item.record;
  const waitKnown=e.wait_cycles!==null;
  const waitText=waitKnown?durationText(e.wait_cycles):'未拆分';
  const waitPercent=waitKnown&&e.duration_cycles?(e.wait_cycles/e.duration_cycles*100).toFixed(1)+'%':'—';
  const parentText=e.parent_event?`　父阶段：<b>${{esc(e.parent_event)}}</b>`:'';
  const intervalText=e.is_wait_event?'这是由 kernel 直接记录的真实轮询等待区间。':
    `包含 <b>${{e.wait_event_count===null?'—':e.wait_event_count}}</b> 个独立等待区间。`;
  document.getElementById('eventDetail').innerHTML=`<b>${{labels.block}} ${{r.core}} · ${{r.role}} · ${{esc(e.event)}}</b>　`
    +`${{labels.iteration}} ${{e.wave}}，${{labels.unit}} ${{e.expert}}，${{labels.range}} [${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})${{parentText}}<br>`
    +`Begin cycle: <b>${{e.begin_cycle}}</b>　End cycle: <b>${{e.end_cycle}}</b>　Total: <b>${{e.duration_cycles}} cycles</b><br>`
    +`${{labels.derived}}: <b>${{e.active_cycles}} cycles</b>　${{labels.wait}}: <b>${{waitText}}</b>　Wait 占比: <b>${{waitPercent}}</b><br>`
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
function renderSelectionStatsMain(events) {{
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
function applySelectionStatsRows(rows) {{
  const tbody=document.querySelector('#selectionStatsTable tbody');
  if(!rows.length) {{ tbody.innerHTML='<tr><td colspan="14" class="empty">当前筛选条件下没有事件</td></tr>'; return; }}
  tbody.innerHTML=rows.map(row=>{{
    const wait=row.wait===null?'—':durationText(row.wait);
    const role=typeof row.role==='number'?roles[row.role]:row.role;
    return `<tr class="${{row.summary?'block-summary-row':''}}"><td>${{esc(row.group)}}</td><td>${{esc(row.event)}}</td><td>${{role}}</td>`
      +`<td>${{row.count}}</td><td>${{durationText(row.total)}}</td><td>${{durationText(row.active)}}</td><td>${{wait}}</td>`
      +`<td>${{durationText(row.mean)}}</td><td>${{durationText(row.p50)}}</td><td>${{durationText(row.p95)}}</td>`
      +`<td>${{durationText(row.max)}}</td><td>${{row.first}}</td><td>${{row.last}}</td><td>${{durationText(row.span)}}</td></tr>`;
  }}).join('');
}}
function renderSelectionStats(events) {{
  if(!events.length) {{ latestWorkerRequest.stats=++workerRequestId; applySelectionStatsRows([]); return; }}
  if(!requestAggregation('stats',events)) renderSelectionStatsMain(events);
}}
function renderWaveTableMain(events) {{
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
  virtualWaveRows=[...groups.values()].sort((a,b)=>a.core-b.core||((a.wave??-1)-(b.wave??-1))||roles.indexOf(a.role)-roles.indexOf(b.role)||a.event.localeCompare(b.event));
  waveTableScroll.scrollTop=0;
  renderVirtualWaveRows();
}}
function applyWaveRows(rows) {{
  virtualWaveRows=rows.map(row=>({{...row,role:typeof row.role==='number'?roles[row.role]:row.role}}));
  waveTableScroll.scrollTop=0;
  renderVirtualWaveRows();
}}
function renderWaveTable(events) {{
  if(!requestAggregation('waves',events)) renderWaveTableMain(events);
}}
function waveTableRow(g) {{
  const wait=g.waitKnown?durationText(g.wait):'—', waitPercent=g.waitKnown&&g.cycles?(g.wait/g.cycles*100).toFixed(1)+'%':'—';
  return `<tr><td>${{g.core}}</td><td>${{g.wave===null?'—':g.wave}}</td><td>${{g.role}}</td><td>${{g.event}}</td><td>${{g.count}}</td><td>${{g.tokens}}</td><td>${{g.mGroups}}</td><td>${{durationText(g.cycles)}}</td><td>${{durationText(g.active)}}</td><td>${{wait}}</td><td>${{waitPercent}}</td><td>${{durationText(g.max)}}</td></tr>`;
}}
function renderVirtualWaveRows() {{
  const tbody=document.querySelector('#waveTable tbody');
  if(!virtualWaveRows.length) {{ tbody.innerHTML='<tr><td colspan="12" class="empty">当前筛选条件下没有事件</td></tr>'; return; }}
  const viewportHeight=Math.max(waveTableScroll.clientHeight,470);
  const first=Math.max(0,Math.floor(waveTableScroll.scrollTop/eventRowHeight)-eventRowOverscan);
  const last=Math.min(virtualWaveRows.length,Math.ceil((waveTableScroll.scrollTop+viewportHeight)/eventRowHeight)+eventRowOverscan);
  const topHeight=first*eventRowHeight,bottomHeight=(virtualWaveRows.length-last)*eventRowHeight;
  const top=`<tr class="virtual-spacer"><td colspan="12" style="height:${{topHeight}}px"></td></tr>`;
  const rows=virtualWaveRows.slice(first,last).map(waveTableRow).join('');
  const bottom=`<tr class="virtual-spacer"><td colspan="12" style="height:${{bottomHeight}}px"></td></tr>`;
  tbody.innerHTML=top+rows+bottom;
}}
function eventTableRow(e) {{
    const wait=e.wait_cycles===null?'—':e.wait_cycles.toLocaleString();
    const waitPercent=e.wait_cycles!==null&&e.duration_cycles?(e.wait_cycles/e.duration_cycles*100).toFixed(1)+'%':'—';
    const converted=cyclesPerUs?(e.duration_cycles/cyclesPerUs).toFixed(3):'—';
    return `<tr><td>${{e.core}}</td><td>${{e.role}}</td><td>${{e.event}}</td><td>${{e.parent_event||'—'}}</td><td>${{e.wave===null?'—':e.wave}}</td><td>${{e.expert}}</td><td>[${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})</td><td>${{e.work_count}}</td><td>${{e.begin_cycle}}</td><td>${{e.end_cycle}}</td><td>${{e.duration_cycles.toLocaleString()}}</td><td>${{e.active_cycles.toLocaleString()}}</td><td>${{wait}}</td><td>${{waitPercent}}</td><td>${{converted}}</td><td>${{esc(e.begin_trace_point)}}</td><td>${{esc(e.end_trace_point)}}</td></tr>`;
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
  const generation=++visibleEventGeneration;
  for(const event of events) event._visibleEventGeneration=generation;
  virtualEventRows=allEventsInTableOrder.filter(event=>event._visibleEventGeneration===generation);
  eventTableScroll.scrollTop=0;
  renderVirtualEventRows();
}}
waveTableScroll.addEventListener('scroll',()=>{{
  if(virtualWaveFrame!==null) return;
  virtualWaveFrame=requestAnimationFrame(()=>{{ virtualWaveFrame=null; renderVirtualWaveRows(); }});
}});
eventTableScroll.addEventListener('scroll',()=>{{
  if(virtualEventFrame!==null) return;
  virtualEventFrame=requestAnimationFrame(()=>{{ virtualEventFrame=null; renderVirtualEventRows(); }});
}});
statsCoreSelect.addEventListener('change',renderStatistics);
for(const control of [coreSelect,laneSort,laneLimit]) control.addEventListener('change',()=>renderDisplay(false));
waitFilter.addEventListener('change',render);
render();
</script>
</main></body></html>
"""
    path.write_text(page, encoding="utf-8")


def write_report_bundle(trace: dict, output_dir: Path, schema: ReportSchema) -> dict:
    """Write JSON, CSV, SVG and self-contained HTML from one normalized trace."""
    validate_normalized_trace(trace, schema)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(trace, output_dir / "events.csv")
    write_svg(trace, output_dir / "timeline.svg", schema)
    write_html(trace, output_dir / "index.html", schema)
    return {
        "header": trace["header"],
        "record_count": len(trace["records"]),
        "event_count": sum(len(record["events"]) for record in trace["records"]),
        "truncated_record_count": sum(
            bool(record["truncated"]) for record in trace["records"]
        ),
    }


def write_run_index(
    summaries: list[dict], output_dir: Path, schema: ReportSchema
) -> None:
    """Write a generic multi-capture entry page using schema-selected header fields."""
    header_columns = "".join(
        f"<th>{html.escape(label)}</th>" for label, _ in schema.summary_fields
    )
    rows = []
    for summary in summaries:
        report_dir = summary["name"]
        header_values = "".join(
            f"<td>{html.escape(str(summary['header'].get(key, '—')))}</td>"
            for _, key in schema.summary_fields
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(summary['name'])}</td>{header_values}"
            f"<td>{summary['event_count']}</td><td>{summary['truncated_record_count']}</td>"
            f'<td><a href="{report_dir}/index.html">交互报告</a> · '
            f'<a href="{report_dir}/events.csv">CSV</a> · '
            f'<a href="{report_dir}/timeline.svg">SVG</a></td></tr>'
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(schema.title)} 汇总</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#172033}}
main{{max-width:1500px;margin:32px auto;padding:0 24px}}p{{color:#667085}}
.panel{{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:20px}}
.table-scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:10px 12px;border-bottom:1px solid #eaecf0;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:last-child,td:last-child{{text-align:left}}th{{background:#f9fafb}}
a{{color:#175cd3;text-decoration:none}}</style></head><body><main><h1>{html.escape(schema.title)} 汇总</h1>
<p>共 {len(summaries)} 份 capture；不将不同 rank/capture 的 cycle 强行对齐。</p>
<section class="panel"><div class="table-scroll"><table><thead><tr><th>Capture</th>{header_columns}
<th>事件数</th><th>截断记录</th><th>产物</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
</main></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "run-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
