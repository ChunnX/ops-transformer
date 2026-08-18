#!/usr/bin/env python3
"""Parse the minimal MegaMoe A8W4 token-wave trace carried in output y."""

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
}
EVENT_COLORS = {
    1: "#06b6d4",
    2: "#2563eb",
    3: "#a855f7",
    4: "#22c55e",
    5: "#ef4444",
    6: "#c084fc",
    7: "#f97316",
    8: "#eab308",
    9: "#ca8a04",
    10: "#a16207",
    11: "#854d0e",
}
WAIT_EVENT_IDS = frozenset((8, 9, 10, 11))
WAIT_INSTRUMENTED_PARENT_IDS = frozenset((2, 4, 5, 7))
EVENT_TRACE_POINTS = {
    1: (
        "ProcessMoeExpertStages：调用 DispatchMoeExpert 前",
        "DispatchMoeExpert 返回后",
    ),
    2: (
        "ProcessMoeExpertStages：调用 RunGmm1ActivationForExpert 前",
        "RunGmm1ActivationForExpert 返回后（AIC 内部 FIX_S 边界）",
    ),
    3: (
        "ProcessMoeExpertStages：调用 RunGmm1ActivationForExpert 前",
        "RunGmm1ActivationForExpert 返回并完成 GMM1 prologue 的 MTE3_S 后",
    ),
    4: (
        "ProcessMoeExpertStages：调用 RunGmm1ActivationForExpert 前",
        "RunGmm1ActivationForExpert 返回后的 Activation 完成边界",
    ),
    5: (
        "ProcessMoeExpertStages：调用 RunGmm2CombineForExpert 前",
        "RunGmm2CombineForExpert 返回后（AIC 内部 FIX_S 边界）",
    ),
    6: (
        "ProcessMoeExpertStages：调用 RunGmm2CombineForExpert 前",
        "RunGmm2CombineForExpert 返回并完成 GMM2 prologue 的 MTE3_S 后",
    ),
    7: (
        "ProcessMoeExpertStages：调用 RunGmm2CombineForExpert 前",
        "RunGmm2CombineForExpert 返回后的 Combine 完成边界",
    ),
    8: (
        "GMM1 AIC：进入 WaitForGmm1InputReady 前",
        "Dispatch 输入 ready 后",
    ),
    9: (
        "Activation AIV1：开始轮询 GMM1 tile 前",
        "GMM1 tile ready 后",
    ),
    10: (
        "GMM2 AIC：进入 WaitForGmm2InputReady 前",
        "Activation 输入 ready 后",
    ),
    11: (
        "Combine AIV1：开始轮询 GMM2 completion 前",
        "GMM2 输出 ready 后",
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
    records = [record for record in trace["records"] if record["events"]]
    lanes = []
    for record in records:
        stage_events = [event for event in record["events"] if not event["is_wait_event"]]
        wait_events = [event for event in record["events"] if event["is_wait_event"]]
        if stage_events:
            lanes.append((f'{record["role"]} core {record["core"]}', stage_events))
        if wait_events:
            lanes.append((f'{record["role"]} core {record["core"]} wait', wait_events))
    width = 1800
    left = 150
    right = 30
    row_height = 24
    top = 55
    height = top + max(1, len(lanes)) * row_height + 55
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
        f'<text x="12" y="24" font-size="16" font-weight="700">A8W4 token-wave timeline (origin={origin})</text>',
    ]
    for row, (lane_name, lane_events) in enumerate(lanes):
        y = top + row * row_height
        lines.append(f'<text class="label" x="8" y="{y + 14}">{lane_name}</text>')
        lines.append(f'<line x1="{left}" y1="{y + 18}" x2="{width - right}" y2="{y + 18}" stroke="#e5e7eb"/>')
        for event in lane_events:
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
                f'<rect class="event" x="{x:.2f}" y="{y + 3}" width="{event_width:.2f}" height="14" '
                f'fill="{color}"><title>{title}</title></rect>'
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
<title>MegaMoe A8W4 token-wave trace</title>
<style>
:root {{ color-scheme: light; --bg:#f5f7fb; --panel:#fff; --line:#dbe2ea; --text:#172033; --muted:#667085; }}
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
select {{ padding:6px 9px; border:1px solid #cbd5e1; border-radius:6px; background:#fff; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px 16px; color:var(--muted); }}
.legend i {{ display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; }}
#timeline {{ min-height:140px; overflow-x:auto; }}
.axis {{ position:relative; height:24px; margin-left:160px; border-bottom:1px solid var(--line); min-width:900px; }}
.tick {{ position:absolute; bottom:0; height:7px; border-left:1px solid #98a2b3; }}
.tick span {{ position:absolute; top:-18px; left:3px; white-space:nowrap; font-size:10px; color:var(--muted); }}
.lane-wrap {{ display:grid; grid-template-columns:150px minmax(900px,1fr); gap:10px; align-items:center; margin-top:9px; }}
.lane-name {{ font-weight:600; text-align:right; }}
.lane {{ position:relative; height:32px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:5px; }}
.lane.wait-lane {{ background:#fffbeb; }}
.bar {{ position:absolute; top:5px; height:20px; min-width:2px; border-radius:3px; opacity:.9; cursor:pointer; }}
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
.warn {{ color:#b42318; font-weight:600; }}
</style>
</head>
<body><main>
<h1>MegaMoe A8W4 token-wave trace</h1>
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
    <span id="selectionSummary" class="muted"></span>
  </div>
  <div class="legend" id="legend"></div>
  <div class="muted">每个角色的 wait 子泳道直接绘制轮询等待的实际 begin/end cycle。父阶段仍覆盖完整执行区间，
  因此阶段块与其等待块会在时间轴上重叠；Active = Total - 这些真实等待区间。</div>
  <div id="timeline"></div>
  <div id="eventDetail" class="event-detail">点击时间线中的事件块查看完整起止 cycle 与打点位置。</div>
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
const colors = {{Dispatch:'#06b6d4',GMM1:'#2563eb','GMM1 prologue':'#a855f7',Activation:'#22c55e',GMM2:'#ef4444','GMM2 prologue':'#c084fc',Combine:'#f97316','Wait Dispatch':'#eab308','Wait GMM1':'#ca8a04','Wait Activation':'#a16207','Wait GMM2':'#854d0e'}};
const roles = ['AIC','AIV0','AIV1'];
const coreSelect = document.getElementById('coreSelect');
const waveSelect = document.getElementById('waveSelect');
coreSelect.add(new Option('全部','all'));
for (let core=0; core<trace.header.logical_core_count; core++) coreSelect.add(new Option(String(core),String(core)));
const waves = [...new Set(trace.records.flatMap(r=>r.events.map(e=>e.wave)))].sort((a,b)=>a-b);
for (const wave of waves) waveSelect.add(new Option(String(wave),String(wave)));
document.getElementById('legend').innerHTML = Object.entries(colors).map(([name,color])=>`<span><i style="background:${{color}}"></i>${{name}}</span>`).join('');
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
function selectedRecords() {{
  const core = coreSelect.value, wave = waveSelect.value;
  return trace.records.filter(r=>core==='all'||r.core===Number(core))
    .map(r=>({{...r,events:r.events.filter(e=>wave==='all'||e.wave===Number(wave))}}))
    .sort((a,b)=>a.core-b.core||roles.indexOf(a.role)-roles.indexOf(b.role));
}}
function esc(value) {{ return String(value).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function render() {{
  const records=selectedRecords(), events=records.flatMap(r=>r.events.map(e=>({{...e,role:r.role,core:r.core}})));
  document.getElementById('selectionSummary').textContent=`${{events.length}} 个事件`;
  renderTimeline(records,events); renderWaveTable(events); renderEventTable(events);
}}
function renderTimeline(records,events) {{
  const root=document.getElementById('timeline');
  if (!events.length) {{ root.innerHTML='<div class="empty">当前筛选条件下没有事件</div>'; return; }}
  const visibleEvents=[];
  let out='<div class="axis">';
  for(let i=0;i<=5;i++) {{
    const x=i*20, offset=Math.round(domainSpan*i/5), cycle=Math.round(absoluteCycle(offset));
    out+=`<div class="tick" style="left:${{x}}%"><span>${{cycle.toLocaleString()}}<br>+${{offset.toLocaleString()}}</span></div>`;
  }}
  out+='</div>';
  for(const record of records) {{
    const laneGroups=[
      {{name:`Block ${{record.core}} · ${{record.role}}`,isWait:false,events:record.events.filter(e=>!e.is_wait_event)}},
      {{name:`Block ${{record.core}} · ${{record.role}} · wait`,isWait:true,events:record.events.filter(e=>e.is_wait_event)}},
    ].filter(group=>group.events.length);
    for(const group of laneGroups) {{
      out+=`<div class="lane-wrap"><div class="lane-name">${{group.name}}</div><div class="lane ${{group.isWait?'wait-lane':''}}">`;
      for(const e of group.events) {{
        const left=relativeCycle(e.begin_cycle)/domainSpan*100, width=Math.max(e.duration_cycles/domainSpan*100,.12);
        const waitKnown=e.wait_cycles!==null;
        const waitText=waitKnown?`${{e.wait_cycles}} cycles (${{(e.wait_cycles/1000).toFixed(3)}} μs)`:'未拆分';
        const parentText=e.parent_event?` | parent=${{e.parent_event}}`:'';
        const tip=`${{e.event}}${{parentText}} | begin=${{e.begin_cycle}} end=${{e.end_cycle}} total=${{e.duration_cycles}} cycles (${{(e.duration_cycles/1000).toFixed(3)}} μs) | active=${{e.active_cycles}} cycles | wait=${{waitText}} | wave=${{e.wave}} expert=${{e.expert}} | tokens=[${{e.token_start_in_expert}},${{e.token_start_in_expert+e.token_count}}) | begin point=${{e.begin_trace_point}} | end point=${{e.end_trace_point}}`;
        const eventIndex=visibleEvents.push({{event:e,record}})-1;
        out+=`<div class="bar" data-event-index="${{eventIndex}}" title="${{esc(tip)}}" style="left:${{left}}%;width:${{width}}%;background:${{colors[e.event]||'#64748b'}}"></div>`;
      }}
      out+='</div></div>';
    }}
  }}
  root.innerHTML=out;
  root.querySelectorAll('.bar').forEach(bar=>bar.addEventListener('click',()=>showEventDetail(visibleEvents[Number(bar.dataset.eventIndex)])));
}}
function showEventDetail(item) {{
  const e=item.event, r=item.record;
  const waitKnown=e.wait_cycles!==null;
  const waitText=waitKnown?`${{e.wait_cycles}} cycles (${{(e.wait_cycles/1000).toFixed(3)}} μs)`:'未拆分';
  const waitPercent=waitKnown&&e.duration_cycles?(e.wait_cycles/e.duration_cycles*100).toFixed(1)+'%':'—';
  const parentText=e.parent_event?`　父阶段：<b>${{esc(e.parent_event)}}</b>`:'';
  const intervalText=e.is_wait_event?'这是由 kernel 直接记录的真实轮询等待区间。':
    `包含 <b>${{e.wait_event_count===null?'—':e.wait_event_count}}</b> 个独立等待区间。`;
  document.getElementById('eventDetail').innerHTML=`<b>Block ${{r.core}} · ${{r.role}} · ${{esc(e.event)}}</b>　`
    +`Wave ${{e.wave}}，Expert ${{e.expert}}，Token [${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})${{parentText}}<br>`
    +`Begin cycle: <b>${{e.begin_cycle}}</b>　End cycle: <b>${{e.end_cycle}}</b>　Total: <b>${{e.duration_cycles}} cycles</b><br>`
    +`Active: <b>${{e.active_cycles}} cycles</b>　Wait: <b>${{waitText}}</b>　Wait 占比: <b>${{waitPercent}}</b><br>`
    +`${{intervalText}}<br>`
    +`Begin 打点：${{esc(e.begin_trace_point)}}<br>End 打点：${{esc(e.end_trace_point)}}`;
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
coreSelect.addEventListener('change',render); waveSelect.addEventListener('change',render); render();
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
