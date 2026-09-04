from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import struct
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .common import percentile, resolve_frequency_mhz, write_csv
except ImportError:  # direct script execution
    from common import percentile, resolve_frequency_mhz, write_csv


MAGIC = 0x4D455046
VERSION = 0x00010000
HEADER_BYTES = 64
OP_NAMES = {
    1: "moe_ep_dispatch",
    2: "moe_ep_dispatch_epilogue",
    3: "moe_ep_combine",
    4: "moe_ep_combine_epilogue",
}
ENGINE_NAMES = {1: "Scalar", 2: "MTE2", 3: "Vector", 4: "MTE3", 5: "URMA", 6: "Wait"}
ENGINE_COLORS = {
    "Scalar": "#984ea3",
    "MTE2": "#377eb8",
    "Vector": "#4daf4a",
    "MTE3": "#00a6d6",
    "URMA": "#ff7f00",
    "Wait": "#e41a1c",
    "Unknown": "#777777",
}

# Unknown/new stage IDs remain visible as STAGE_<N>. This map is deliberately
# data-only so kernel changes do not make the parser reject valid raw files.
DEFAULT_STAGE_NAMES: dict[str, dict[int, str]] = {
    "moe_ep_dispatch": {
        0: "KERNEL",
        1: "BUFFER_INIT",
        2: "COUNT_BUILD",
        3: "COUNT_ISSUE",
        4: "COUNT_WAIT",
        5: "COUNT_AGGREGATE",
        6: "SLOT_PREFIX",
        7: "PAYLOAD_PREP",
        8: "PAYLOAD_ISSUE",
    },
    "moe_ep_dispatch_epilogue": {
        0: "KERNEL",
        1: "PREFIX",
        2: "PAYLOAD_WAIT",
        3: "WAIT_BARRIER",
        4: "COUNT_HITS",
        5: "WINDOW_OUTPUT",
    },
    "moe_ep_combine": {
        0: "KERNEL",
        1: "ADDRESS_TABLE",
        2: "DATA_WQE",
        3: "FLAG_COMMIT",
    },
    "moe_ep_combine_epilogue": {
        0: "KERNEL",
        1: "BUFFER_INIT",
        2: "COMPLETION_WAIT",
        3: "WAIT_BARRIER",
        4: "REDUCE_OUTPUT",
    },
}


@dataclass(frozen=True)
class RawHeader:
    op_id: int
    actual_aiv_cores: int
    max_cores: int
    slots_per_core: int
    record_bytes: int
    core_control_bytes: int
    core_stride: int
    flags: int
    local_record_capacity: int


def _load_stage_names(path: Path | None) -> dict[str, dict[int, str]]:
    result = {op: dict(values) for op, values in DEFAULT_STAGE_NAMES.items()}
    if path is None:
        return result
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages", payload)
    for op, mapping in stages.items():
        result.setdefault(op, {}).update(
            {
                int(key, 0) if isinstance(key, str) else int(key): str(value)
                for key, value in mapping.items()
            }
        )
    return result


def _rank_from_path(path: Path) -> int:
    match = re.search(r"rank[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _parse_header(data: bytes, path: Path) -> RawHeader:
    if len(data) < HEADER_BYTES:
        raise ValueError(f"{path}: truncated global header ({len(data)} bytes)")
    words = struct.unpack_from("<16I", data, 0)
    if words[0] != MAGIC:
        raise ValueError(f"{path}: bad magic 0x{words[0]:08X}, expected 0x{MAGIC:08X}")
    if words[1] != VERSION:
        raise ValueError(f"{path}: unsupported version 0x{words[1]:08X}")
    if words[2] != HEADER_BYTES:
        raise ValueError(f"{path}: unsupported header size {words[2]}")
    header = RawHeader(
        words[3],
        words[4],
        words[5],
        words[6],
        words[7],
        words[8],
        words[9],
        words[10],
        words[11],
    )
    if header.op_id not in OP_NAMES:
        raise ValueError(f"{path}: unsupported op id {header.op_id}")
    if header.max_cores <= 0 or header.slots_per_core <= 0:
        raise ValueError(f"{path}: max_cores and slots_per_core must be positive")
    if header.actual_aiv_cores > header.max_cores:
        raise ValueError(f"{path}: actual_aiv_cores exceeds max_cores")
    expected_stride = (
        header.core_control_bytes + header.slots_per_core * header.record_bytes
    )
    if (
        header.record_bytes != 16
        or header.core_control_bytes != 64
        or header.core_stride != expected_stride
    ):
        raise ValueError(f"{path}: inconsistent record/control/stride fields")
    expected_size = HEADER_BYTES + header.max_cores * header.core_stride
    if len(data) < expected_size:
        raise ValueError(f"{path}: truncated buffer ({len(data)} < {expected_size})")
    return header


def _is_unused_buffer(path: Path) -> bool:
    """A zero header denotes an allocated profiler buffer that its op never launched."""
    with path.open("rb") as stream:
        first_word = stream.read(4)
    return len(first_word) == 4 and struct.unpack("<I", first_word)[0] == 0


def _decode_event(
    event_id: int, stage_names: Mapping[str, Mapping[int, str]]
) -> dict[str, object]:
    op_id = (event_id >> 28) & 0xF
    role_id = (event_id >> 24) & 0xF
    engine_id = (event_id >> 20) & 0xF
    stage_id = (event_id >> 1) & 0x7FFFF
    boundary_id = event_id & 1
    op = OP_NAMES.get(op_id, f"op_{op_id}")
    engine = ENGINE_NAMES.get(engine_id, "Unknown")
    stage = stage_names.get(op, {}).get(stage_id, f"STAGE_{stage_id}")
    boundary = "BEGIN" if boundary_id == 0 else "END"
    return {
        "event_op_id": op_id,
        "event_op": op,
        "role_id": role_id,
        "role": "AIV" if role_id == 1 else f"ROLE_{role_id}",
        "engine_id": engine_id,
        "engine": engine,
        "stage_id": stage_id,
        "function": stage,
        "boundary": boundary,
        "pair_id": event_id & ~1,
        "event_name": f"{op}.{stage}.{engine}.{boundary}",
    }


def parse_raw_file(
    path: Path, stage_names: Mapping[str, Mapping[int, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], RawHeader]:
    data = path.read_bytes()
    header = _parse_header(data, path)
    header_op = OP_NAMES.get(header.op_id, f"op_{header.op_id}")
    rank = _rank_from_path(path)
    events: list[dict[str, object]] = []
    core_stats: list[dict[str, object]] = []
    for core_id in range(header.max_cores):
        core_base = HEADER_BYTES + core_id * header.core_stride
        control = struct.unpack_from("<16I", data, core_base)
        write_count = control[0] | (control[1] << 32)
        write_pos = control[2]
        flags = control[3]
        dropped_local = control[4] | (control[5] << 32)
        flush_count = control[6] | (control[7] << 32)
        last_batch_count = control[8]
        if write_pos >= header.slots_per_core:
            raise ValueError(
                f"{path}: core {core_id} write_pos {write_pos} is outside the ring"
            )
        valid = min(write_count, header.slots_per_core)
        core_stats.append(
            {
                "rank": rank,
                "op": header_op,
                "core_id": core_id,
                "write_count": write_count,
                "write_pos": write_pos,
                "flags": flags,
                "ring_wrapped": int(bool(flags & 1)),
                "local_dropped": dropped_local,
                "local_drop_seen": int(bool(flags & 2)),
                "flush_count": flush_count,
                "last_batch_count": last_batch_count,
                "valid_records": valid,
            }
        )
        if valid == 0:
            continue
        oldest = write_pos if write_count >= header.slots_per_core else 0
        records: list[tuple[int, int, int, int]] = []
        records_base = core_base + header.core_control_bytes
        for order in range(valid):
            slot = (oldest + order) % header.slots_per_core
            event_id, payload, cycle = struct.unpack_from(
                "<IIQ", data, records_base + slot * header.record_bytes
            )
            if event_id == 0 and cycle == 0:
                continue
            records.append((order, event_id, payload, cycle))
        kernel_begins = sum(
            1
            for _, event_id, _, _ in records
            if ((event_id >> 1) & 0x7FFFF) == 0 and (event_id & 1) == 0
        )
        invocation = max(0, int(flush_count) - kernel_begins)
        for order, event_id, payload, cycle in records:
            decoded = _decode_event(event_id, stage_names)
            if decoded["stage_id"] == 0 and decoded["boundary"] == "BEGIN":
                invocation += 1
            events.append(
                {
                    "source_file": str(path),
                    "rank": rank,
                    "header_op": header_op,
                    "core_id": core_id,
                    "physical_core": f"AIV_{core_id}",
                    "record_order": order,
                    "invocation": invocation - 1,
                    "event_id": f"0x{event_id:08X}",
                    "payload": payload,
                    "cycle": cycle,
                    **decoded,
                }
            )
    return events, core_stats, header


def pair_intervals(
    events: Sequence[Mapping[str, object]], frequency_mhz: float
) -> list[dict[str, object]]:
    by_lane: dict[tuple[int, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for event in events:
        by_lane[
            (int(event["rank"]), str(event["header_op"]), int(event["core_id"]))
        ].append(event)
    intervals: list[dict[str, object]] = []
    for lane, lane_events in by_lane.items():
        stacks: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
        for event in sorted(
            lane_events,
            key=lambda item: (int(item["record_order"]), int(item["cycle"])),
        ):
            key = (int(event["pair_id"]), int(event["invocation"]))
            if event["boundary"] == "BEGIN":
                stacks[key].append(event)
                continue
            if not stacks[key]:
                continue
            begin = stacks[key].pop()
            duration = int(event["cycle"]) - int(begin["cycle"])
            if duration < 0:
                continue
            intervals.append(
                {
                    "rank": lane[0],
                    "op": begin["event_op"],
                    "header_op": lane[1],
                    "core_id": lane[2],
                    "physical_core": begin["physical_core"],
                    "role": begin["role"],
                    "engine": begin["engine"],
                    "function": begin["function"],
                    "stage_id": begin["stage_id"],
                    "invocation": begin["invocation"],
                    "payload": begin["payload"],
                    "begin_cycle": begin["cycle"],
                    "end_cycle": event["cycle"],
                    "duration_cycles": duration,
                    "duration_us": duration / frequency_mhz,
                }
            )
    return intervals


def _aggregate(
    intervals: Iterable[Mapping[str, object]], scope: str
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for item in intervals:
        key = (item["rank"], item["op"], item["role"], item["engine"], item["function"])
        grouped[key].append(float(item["duration_cycles"]))
    rows: list[dict[str, object]] = []
    for key, values in sorted(
        grouped.items(), key=lambda entry: tuple(map(str, entry[0]))
    ):
        rows.append(
            {
                "scope": scope,
                "rank": key[0],
                "op": key[1],
                "role": key[2],
                "engine": key[3],
                "function": key[4],
                "count": len(values),
                "min_cycles": min(values),
                "p50_cycles": percentile(values, 0.5),
                "mean_cycles": statistics.fmean(values),
                "p90_cycles": percentile(values, 0.9),
                "max_cycles": max(values),
            }
        )
    return rows


def _load_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_trace(
    path: Path,
    rank: int,
    intervals: Sequence[Mapping[str, object]],
    frequency_mhz: float,
) -> None:
    selected = [item for item in intervals if int(item["rank"]) == rank]
    origin = min((int(item["begin_cycle"]) for item in selected), default=0)
    trace = []
    for item in selected:
        trace.append(
            {
                "name": str(item["function"]),
                "cat": f"{item['op']}/{item['engine']}",
                "ph": "X",
                "pid": f"rank{rank}:{item['op']}",
                "tid": f"{item['role']}:{item['core_id']}",
                "ts": (int(item["begin_cycle"]) - origin) / frequency_mhz,
                "dur": float(item["duration_cycles"]) / frequency_mhz,
                "args": {
                    "invocation": item["invocation"],
                    "payload": item["payload"],
                    "relative_time_only": True,
                },
            }
        )
    path.write_text(
        json.dumps({"traceEvents": trace, "displayTimeUnit": "us"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_summary_svg(
    path: Path, summary: Sequence[Mapping[str, object]], frequency_mhz: float
) -> None:
    rows = list(summary)
    left, width, row_h = 330, 1250, 25
    max_cycles = max((float(row["p50_cycles"]) for row in rows), default=1.0)
    scale = 820.0 / max(max_cycles, 1.0)
    height = max(100, 45 + len(rows) * row_h)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font:12px sans-serif}.grid{stroke:#ddd}</style><rect width="100%" height="100%" fill="white"/>',
    ]
    for index, row in enumerate(rows):
        y = 28 + index * row_h
        cycles = float(row["p50_cycles"])
        color = ENGINE_COLORS.get(str(row["engine"]), ENGINE_COLORS["Unknown"])
        label = html.escape(
            f"r{row['rank']} {row['op']} {row['role']} {row['function']}"
        )
        parts.append(f'<text x="5" y="{y + 14}">{label}</text>')
        parts.append(
            f'<rect x="{left}" y="{y + 2}" width="{cycles * scale:.2f}" height="16" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + cycles * scale + 5:.2f}" y="{y + 14}">{cycles / frequency_mhz:.3f} us</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_logical_svg(
    path: Path,
    rank: int,
    intervals: Sequence[Mapping[str, object]],
    frequency_mhz: float,
) -> None:
    selected = [
        item
        for item in intervals
        if int(item["rank"]) == rank and item["function"] != "KERNEL"
    ]
    kernel_start: dict[tuple[str, int, int], int] = {}
    for item in intervals:
        if int(item["rank"]) == rank and item["function"] == "KERNEL":
            kernel_start[
                (str(item["op"]), int(item["core_id"]), int(item["invocation"]))
            ] = int(item["begin_cycle"])
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for item in selected:
        key = (str(item["op"]), int(item["core_id"]), int(item["invocation"]))
        origin = kernel_start.get(key)
        if origin is None:
            continue
        grouped[(str(item["op"]), str(item["engine"]), str(item["function"]))].append(
            (int(item["begin_cycle"]) - origin, float(item["duration_cycles"]))
        )
    lanes = []
    for key, values in sorted(grouped.items()):
        lanes.append(
            (
                key,
                percentile([v[0] for v in values], 0.5),
                percentile([v[1] for v in values], 0.5),
            )
        )
    max_end = max((start + duration for _, start, duration in lanes), default=1.0)
    scale = 850.0 / max(max_end, 1.0)
    width, left, row_h = 1250, 300, 30
    height = max(110, 55 + len(lanes) * row_h)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font:12px sans-serif}.axis{stroke:#333}</style><rect width="100%" height="100%" fill="white"/>',
        f'<text x="5" y="18">rank {rank}: relative GetSystemCycle logical pipeline (no cross-rank epoch claim)</text>',
        f'<line class="axis" x1="{left}" y1="28" x2="{left + 850}" y2="28"/>',
    ]
    for index, (key, start, duration) in enumerate(lanes):
        y = 40 + index * row_h
        color = ENGINE_COLORS.get(key[1], ENGINE_COLORS["Unknown"])
        parts.append(
            f'<text x="5" y="{y + 15}">{html.escape(key[0] + " / " + key[2] + " / " + key[1])}</text>'
        )
        parts.append(
            f'<rect x="{left + start * scale:.2f}" y="{y + 2}" width="{max(1.0, duration * scale):.2f}" height="17" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + (start + duration) * scale + 4:.2f}" y="{y + 15}">{duration / frequency_mhz:.3f} us</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_logical_trace(
    path: Path,
    rank: int,
    intervals: Sequence[Mapping[str, object]],
    frequency_mhz: float,
) -> None:
    """Write the p50 logical pipeline relative to each invocation's KERNEL_BEGIN."""
    kernel_start: dict[tuple[str, int, int], int] = {}
    for item in intervals:
        if int(item["rank"]) == rank and item["function"] == "KERNEL":
            kernel_start[
                (str(item["op"]), int(item["core_id"]), int(item["invocation"]))
            ] = int(item["begin_cycle"])
    grouped: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(
        list
    )
    for item in intervals:
        if int(item["rank"]) != rank or item["function"] == "KERNEL":
            continue
        origin = kernel_start.get(
            (str(item["op"]), int(item["core_id"]), int(item["invocation"]))
        )
        if origin is None:
            continue
        key = (
            str(item["op"]),
            str(item["role"]),
            str(item["engine"]),
            str(item["function"]),
        )
        grouped[key].append(
            (int(item["begin_cycle"]) - origin, float(item["duration_cycles"]))
        )
    trace = []
    for (op, role, engine, function), values in sorted(grouped.items()):
        start = percentile([value[0] for value in values], 0.5)
        duration = percentile([value[1] for value in values], 0.5)
        trace.append(
            {
                "name": function,
                "cat": f"{op}/{engine}",
                "ph": "X",
                "pid": f"rank{rank}:{op}",
                "tid": f"{role}:{engine}",
                "ts": start / frequency_mhz,
                "dur": duration / frequency_mhz,
                "args": {
                    "samples": len(values),
                    "statistic": "p50",
                    "relative_to": "KERNEL_BEGIN",
                    "relative_time_only": True,
                },
            }
        )
    path.write_text(
        json.dumps({"traceEvents": trace, "displayTimeUnit": "us"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def analyze(
    run_dir: Path,
    output_dir: Path,
    warmup: int | None = None,
    tail_count: int | None = None,
    event_map: Path | None = None,
    frequency_search_root: Path | None = None,
    frequency_mhz_override: float | None = None,
) -> dict[str, object]:
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(run_dir)
    warmup = int(manifest.get("warmup", 5) if warmup is None else warmup)
    tail_count = int(
        manifest.get("performance_tail_count", 50) if tail_count is None else tail_count
    )
    stage_names = _load_stage_names(event_map)
    raw_files = sorted(run_dir.rglob("*.profbin"))
    if not raw_files:
        raise FileNotFoundError(f"no *.profbin files under {run_dir}")
    events: list[dict[str, object]] = []
    core_stats: list[dict[str, object]] = []
    headers = []
    unused_raw_files: list[str] = []
    for path in raw_files:
        if _is_unused_buffer(path):
            warnings.warn(
                f"{path}: unused zero-header profiler buffer; skipped", RuntimeWarning
            )
            unused_raw_files.append(str(path))
            continue
        parsed_events, parsed_stats, header = parse_raw_file(path, stage_names)
        events.extend(parsed_events)
        core_stats.extend(parsed_stats)
        headers.append({"path": str(path), **header.__dict__})
    frequency_root = (frequency_search_root or run_dir).resolve()
    if frequency_mhz_override is not None and frequency_mhz_override > 0:
        frequency = {
            "mhz": float(frequency_mhz_override),
            "source": "cli --frequency-mhz override",
            "source_files": [],
            "observed_mhz": [float(frequency_mhz_override)],
        }
    else:
        frequency = resolve_frequency_mhz(frequency_root)
    frequency_mhz = float(frequency["mhz"])
    intervals = pair_intervals(events, frequency_mhz)
    post_warmup = [item for item in intervals if int(item["invocation"]) >= warmup]
    max_invocations: dict[tuple[int, str, int], int] = defaultdict(lambda: -1)
    for item in intervals:
        key = (int(item["rank"]), str(item["op"]), int(item["core_id"]))
        max_invocations[key] = max(max_invocations[key], int(item["invocation"]))
    tail = [
        item
        for item in post_warmup
        if tail_count <= 0
        or int(item["invocation"])
        > max_invocations[(int(item["rank"]), str(item["op"]), int(item["core_id"]))]
        - tail_count
    ]
    function_summary = _aggregate(post_warmup, "post_warmup")
    tail_summary = _aggregate(
        tail, f"last_{tail_count}" if tail_count > 0 else "all_post_warmup"
    )
    for row in function_summary + tail_summary:
        for name in ("min", "p50", "mean", "p90", "max"):
            row[f"{name}_us"] = float(row[f"{name}_cycles"]) / frequency_mhz
    event_fields = [
        "source_file",
        "rank",
        "header_op",
        "core_id",
        "physical_core",
        "record_order",
        "invocation",
        "event_id",
        "payload",
        "cycle",
        "event_op_id",
        "event_op",
        "role_id",
        "role",
        "engine_id",
        "engine",
        "stage_id",
        "function",
        "boundary",
        "pair_id",
        "event_name",
    ]
    interval_fields = [
        "rank",
        "op",
        "header_op",
        "core_id",
        "physical_core",
        "role",
        "engine",
        "function",
        "stage_id",
        "invocation",
        "payload",
        "begin_cycle",
        "end_cycle",
        "duration_cycles",
        "duration_us",
    ]
    summary_fields = [
        "scope",
        "rank",
        "op",
        "role",
        "engine",
        "function",
        "count",
        "min_cycles",
        "p50_cycles",
        "mean_cycles",
        "p90_cycles",
        "max_cycles",
        "min_us",
        "p50_us",
        "mean_us",
        "p90_us",
        "max_us",
    ]
    write_csv(output_dir / "events.csv", event_fields, events)
    write_csv(output_dir / "intervals.csv", interval_fields, intervals)
    write_csv(output_dir / "function_summary.csv", summary_fields, function_summary)
    write_csv(output_dir / "tail_summary.csv", summary_fields, tail_summary)
    write_csv(
        output_dir / "core_status.csv",
        list(core_stats[0].keys()) if core_stats else ["rank"],
        core_stats,
    )
    for rank in sorted({int(item["rank"]) for item in intervals}):
        _write_trace(
            output_dir / f"functions_rank{rank:03d}.trace.json",
            rank,
            tail,
            frequency_mhz,
        )
        _write_logical_svg(
            output_dir / f"logical_pipeline_rank{rank:03d}.svg",
            rank,
            tail,
            frequency_mhz,
        )
        _write_logical_trace(
            output_dir / f"logical_pipeline_rank{rank:03d}.trace.json",
            rank,
            tail,
            frequency_mhz,
        )
    _write_summary_svg(output_dir / "function_summary.svg", tail_summary, frequency_mhz)
    analysis_manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "warmup": warmup,
        "tail_count": tail_count,
        "frequency": frequency,
        "frequency_search_root": str(frequency_root),
        "raw_files": [str(path) for path in raw_files],
        "unused_raw_files": unused_raw_files,
        "headers": headers,
        "events": len(events),
        "intervals": len(intervals),
        "cross_rank_time_semantics": "relative-per-rank; GetSystemCycle epochs are not assumed equal",
        "warnings": [
            f"rank={row['rank']} op={row['op']} core={row['core_id']}: ring wrapped"
            for row in core_stats
            if row["ring_wrapped"]
        ]
        + [
            f"rank={row['rank']} op={row['op']} core={row['core_id']}: local records dropped={row['local_dropped']}"
            for row in core_stats
            if row["local_drop_seen"]
        ]
        + [
            f"header/event op mismatch: rank={item['rank']} header={item['header_op']} event={item['event_op']}"
            for item in events
            if item["header_op"] != item["event_op"]
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return analysis_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse MoeEp self-describing GetSystemCycle raw files"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--tail-count", type=int)
    parser.add_argument("--event-map", type=Path)
    parser.add_argument(
        "--frequency-search-root",
        type=Path,
        help="root containing this run's OpBasicInfo.csv; platform_config/fallback remain automatic",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        help="override frequency in MHz; skips OpBasicInfo/platform_config/fallback",
    )
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "analysis"
    result = analyze(
        args.run_dir,
        output,
        args.warmup,
        args.tail_count,
        args.event_map,
        args.frequency_search_root,
        args.frequency_mhz,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
