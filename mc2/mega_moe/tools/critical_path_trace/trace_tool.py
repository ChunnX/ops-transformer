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
}
EVENT_COLORS = {
    1: "#06b6d4",
    2: "#2563eb",
    3: "#a855f7",
    4: "#22c55e",
    5: "#ef4444",
    6: "#c084fc",
    7: "#f97316",
}


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
    if header["event_words"] != 8:
        raise ValueError(f"unsupported event size: {header['event_words']}")
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
            event_type, wave_index, expert_index, token_start, token_count, m_group_count, begin, end = words[
                base : base + 8
            ]
            events.append(
                {
                    "event": EVENT_NAMES.get(event_type, f"unknown({event_type})"),
                    "event_id": event_type,
                    "wave": wave_index,
                    "expert": expert_index,
                    "token_start_in_expert": token_start,
                    "token_count": token_count,
                    "m_group_count": m_group_count,
                    "begin_cycle": begin,
                    "end_cycle": end,
                    "duration_cycles": (end - begin) & 0xFFFFFFFF,
                }
            )
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
        "wave",
        "expert",
        "token_start_in_expert",
        "token_count",
        "m_group_count",
        "begin_cycle",
        "end_cycle",
        "duration_cycles",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in trace["records"]:
            for event in record["events"]:
                writer.writerow({"role": record["role"], "core": record["core"], **event})


def write_svg(trace: dict, path: Path) -> None:
    records = [record for record in trace["records"] if record["events"]]
    width = 1800
    left = 150
    right = 30
    row_height = 24
    top = 55
    height = top + max(1, len(records)) * row_height + 55
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
    for row, record in enumerate(records):
        y = top + row * row_height
        lines.append(f'<text class="label" x="8" y="{y + 14}">{record["role"]} core {record["core"]}</text>')
        lines.append(f'<line x1="{left}" y1="{y + 18}" x2="{width - right}" y2="{y + 18}" stroke="#e5e7eb"/>')
        for event in record["events"]:
            x = left + (((event["begin_cycle"] - origin) & 0xFFFFFFFF) * scale)
            event_width = max(1.0, event["duration_cycles"] * scale)
            color = EVENT_COLORS.get(event["event_id"], "#64748b")
            title = (
                f'{event["event"]} wave={event["wave"]} expert={event["expert"]} '
                f'tokens=[{event["token_start_in_expert"]},'
                f'{event["token_start_in_expert"] + event["token_count"]}) '
                f'duration={event["duration_cycles"]}'
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
main {{ max-width:1500px; margin:0 auto; padding:24px; }}
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
.axis {{ position:relative; height:24px; margin-left:130px; border-bottom:1px solid var(--line); min-width:900px; }}
.tick {{ position:absolute; bottom:0; height:7px; border-left:1px solid #98a2b3; }}
.tick span {{ position:absolute; top:-18px; left:3px; white-space:nowrap; font-size:10px; color:var(--muted); }}
.lane-wrap {{ display:grid; grid-template-columns:120px minmax(900px,1fr); gap:10px; align-items:center; margin-top:9px; }}
.lane-name {{ font-weight:600; text-align:right; }}
.lane {{ position:relative; height:32px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:5px; }}
.bar {{ position:absolute; top:5px; height:20px; min-width:2px; border-radius:3px; opacity:.9; cursor:default; }}
.bar:hover {{ opacity:1; outline:2px solid #11182755; z-index:2; }}
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
<div class="muted">同一物理 AI Block 内比较 AIC/AIV0/AIV1；不同 core 的绝对时间不作横向比较。</div>
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
  <div id="timeline"></div>
</section>
<section class="panel">
  <h2>Wave 汇总</h2>
  <div class="table-scroll"><table id="waveTable"><thead><tr>
    <th>Wave</th><th>角色</th><th>阶段</th><th>事件数</th><th>Token 数</th><th>M group</th><th>总 cycles</th><th>最长事件</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<section class="panel">
  <h2>事件明细</h2>
  <div class="table-scroll"><table id="eventTable"><thead><tr>
    <th>角色</th><th>阶段</th><th>Wave</th><th>Expert</th><th>Token 范围</th><th>M group</th><th>Begin</th><th>End</th><th>Cycles</th>
  </tr></thead><tbody></tbody></table></div>
</section>
<script id="traceData" type="application/json">{payload}</script>
<script>
const trace = JSON.parse(document.getElementById('traceData').textContent);
const colors = {{Dispatch:'#06b6d4',GMM1:'#2563eb','GMM1 prologue':'#a855f7',Activation:'#22c55e',GMM2:'#ef4444','GMM2 prologue':'#c084fc',Combine:'#f97316'}};
const roles = ['AIC','AIV0','AIV1'];
const coreSelect = document.getElementById('coreSelect');
const waveSelect = document.getElementById('waveSelect');
for (let core=0; core<trace.header.logical_core_count; core++) coreSelect.add(new Option(String(core),String(core)));
const waves = [...new Set(trace.records.flatMap(r=>r.events.map(e=>e.wave)))].sort((a,b)=>a-b);
for (const wave of waves) waveSelect.add(new Option(String(wave),String(wave)));
document.getElementById('legend').innerHTML = Object.entries(colors).map(([name,color])=>`<span><i style="background:${{color}}"></i>${{name}}</span>`).join('');
function selectedRecords() {{
  const core = Number(coreSelect.value), wave = waveSelect.value;
  return trace.records.filter(r=>r.core===core).map(r=>({{...r,events:r.events.filter(e=>wave==='all'||e.wave===Number(wave))}}));
}}
function esc(value) {{ return String(value).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function render() {{
  const records=selectedRecords(), events=records.flatMap(r=>r.events.map(e=>({{...e,role:r.role}})));
  document.getElementById('selectionSummary').textContent=`${{events.length}} 个事件`;
  renderTimeline(records,events); renderWaveTable(events); renderEventTable(events);
}}
function renderTimeline(records,events) {{
  const root=document.getElementById('timeline');
  if (!events.length) {{ root.innerHTML='<div class="empty">当前筛选条件下没有事件</div>'; return; }}
  const origin=Math.min(...events.map(e=>e.begin_cycle));
  const rel=t=>(t-origin)>>>0;
  const end=Math.max(...events.map(e=>rel(e.end_cycle)),1);
  let out='<div class="axis">';
  for(let i=0;i<=5;i++) {{ const x=i*20, cycle=Math.round(end*i/5); out+=`<div class="tick" style="left:${{x}}%"><span>${{cycle.toLocaleString()}}</span></div>`; }}
  out+='</div>';
  for(const role of roles) {{
    const record=records.find(r=>r.role===role), laneEvents=record?record.events:[];
    out+=`<div class="lane-wrap"><div class="lane-name">${{role}}</div><div class="lane">`;
    for(const e of laneEvents) {{
      const left=rel(e.begin_cycle)/end*100, width=Math.max(e.duration_cycles/end*100,.12);
      const tip=`${{e.event}} | wave=${{e.wave}} expert=${{e.expert}} | tokens=[${{e.token_start_in_expert}},${{e.token_start_in_expert+e.token_count}}) | M groups=${{e.m_group_count}} | cycles=${{e.duration_cycles}}`;
      out+=`<div class="bar" title="${{esc(tip)}}" style="left:${{left}}%;width:${{width}}%;background:${{colors[e.event]||'#64748b'}}"></div>`;
    }}
    out+='</div></div>';
  }}
  root.innerHTML=out;
}}
function renderWaveTable(events) {{
  const groups=new Map();
  for(const e of events) {{
    const key=`${{e.wave}}|${{e.role}}|${{e.event}}`;
    const g=groups.get(key)||{{wave:e.wave,role:e.role,event:e.event,count:0,tokens:0,mGroups:0,cycles:0,max:0}};
    g.count++; g.tokens+=e.token_count; g.mGroups+=e.m_group_count; g.cycles+=e.duration_cycles; g.max=Math.max(g.max,e.duration_cycles); groups.set(key,g);
  }}
  const rows=[...groups.values()].sort((a,b)=>a.wave-b.wave||roles.indexOf(a.role)-roles.indexOf(b.role)||a.event.localeCompare(b.event));
  document.querySelector('#waveTable tbody').innerHTML=rows.map(g=>`<tr><td>${{g.wave}}</td><td>${{g.role}}</td><td>${{g.event}}</td><td>${{g.count}}</td><td>${{g.tokens}}</td><td>${{g.mGroups}}</td><td>${{g.cycles.toLocaleString()}}</td><td>${{g.max.toLocaleString()}}</td></tr>`).join('');
}}
function renderEventTable(events) {{
  const rows=[...events].sort((a,b)=>a.wave-b.wave||roles.indexOf(a.role)-roles.indexOf(b.role)||((a.begin_cycle-b.begin_cycle)|0));
  document.querySelector('#eventTable tbody').innerHTML=rows.map(e=>`<tr><td>${{e.role}}</td><td>${{e.event}}</td><td>${{e.wave}}</td><td>${{e.expert}}</td><td>[${{e.token_start_in_expert}}, ${{e.token_start_in_expert+e.token_count}})</td><td>${{e.m_group_count}}</td><td>${{e.begin_cycle}}</td><td>${{e.end_cycle}}</td><td>${{e.duration_cycles.toLocaleString()}}</td></tr>`).join('');
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
