#!/usr/bin/env python3
"""Convert FullMeshV2 per-core profiler binaries to CSV, SVG and Chrome Trace JSON."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import struct
from collections import defaultdict, deque
from pathlib import Path


MAGIC = 0x50524F46
HEADER_SIZE = 64
EVENT_NAMES = {
    0x0001: "KERNEL",
    0x0010: "SEND_TOKENS",
    0x0011: "SEND_TOKENS_END",
    0x0020: "SEND_COUNTS",
    0x0021: "SEND_COUNTS_END",
    0x0030: "WAIT_RECV_COUNTS",
    0x0031: "WAIT_RECV_COUNTS_END",
    0x0040: "LOCAL_WINDOW_COPY",
    0x0041: "LOCAL_WINDOW_COPY_END",
    0x0050: "WAIT_CUMSUM",
    0x0051: "WAIT_CUMSUM_END",
    0x0060: "FORMAT_OUTPUT",
    0x0061: "FORMAT_OUTPUT_END",
    0x0070: "PREPARE_TOKEN_INPUT",
    0x0071: "PREPARE_TOKEN_INPUT_END",
    0x0080: "TRANSFER_TOKENS",
    0x0081: "TRANSFER_TOKENS_END",
    0x0090: "PREPARE_COUNT_INPUT",
    0x0091: "PREPARE_COUNT_INPUT_END",
    0x00A0: "CLEAR_RECV_STATUS",
    0x00A1: "CLEAR_RECV_STATUS_END",
    0x00B0: "GATHER_RECV_COUNTS",
    0x00B1: "GATHER_RECV_COUNTS_END",
    0x00C0: "GET_CUMSUM",
    0x00C1: "GET_CUMSUM_END",
    0x00D0: "BUILD_RECV_CUMSUM",
    0x00D1: "BUILD_RECV_CUMSUM_END",
    0x00E0: "PUBLISH_RECV_COUNTS",
    0x00E1: "PUBLISH_RECV_COUNTS_END",
    0x0100: "RESET_CUMSUM_FLAG",
    0x0101: "RESET_CUMSUM_FLAG_END",
    0x0110: "BUILD_EXPERT_TASKS",
    0x0111: "BUILD_EXPERT_TASKS_END",
    0x0120: "CLEAR_LOCAL_FLAGS",
    0x0121: "CLEAR_LOCAL_FLAGS_END",
    0x0130: "DRAIN_PIPELINES",
    0x0131: "DRAIN_PIPELINES_END",
    0x0140: "NO_ACTIVE_TOKENS",
    0x0141: "NO_ACTIVE_TOKENS_END",
    0x0150: "NO_ASSIGNED_STATUS",
    0x0151: "NO_ASSIGNED_STATUS_END",
    0x0160: "NO_VALID_EXPERT",
    0x0161: "NO_VALID_EXPERT_END",
    0x0171: "SEND_TASK_COUNT",
    0x0172: "SEND_BUFFER_NUM",
    0x0173: "SEND_POSITION_CYCLES",
    0x0174: "SEND_INPUT_CYCLES",
    0x0175: "SEND_COMPUTE_CYCLES",
    0x0176: "SEND_PACK_CYCLES",
    0x0177: "SEND_BUFFER_WAIT_CYCLES",
    0x0178: "SERIAL_BUFFER_INIT_CYCLES",
    0x0180: "READ_RECV_COUNTS",
    0x0181: "READ_RECV_COUNTS_END",
    0x0190: "BUILD_LOCAL_PREFIX",
    0x0191: "BUILD_LOCAL_PREFIX_END",
    0x01A0: "PUBLISH_LOCAL_COUNTS",
    0x01A1: "PUBLISH_LOCAL_COUNTS_END",
    0x00FF: "KERNEL_END",
    0xFFFF: "OVERFLOW",
}
PAIR_MAP = {
    0x0001: 0x00FF,
    0x0010: 0x0011,
    0x0020: 0x0021,
    0x0030: 0x0031,
    0x0040: 0x0041,
    0x0050: 0x0051,
    0x0060: 0x0061,
    0x0070: 0x0071,
    0x0080: 0x0081,
    0x0090: 0x0091,
    0x00A0: 0x00A1,
    0x00B0: 0x00B1,
    0x00C0: 0x00C1,
    0x00D0: 0x00D1,
    0x00E0: 0x00E1,
    0x0100: 0x0101,
    0x0110: 0x0111,
    0x0120: 0x0121,
    0x0130: 0x0131,
    0x0140: 0x0141,
    0x0150: 0x0151,
    0x0160: 0x0161,
    0x0180: 0x0181,
    0x0190: 0x0191,
    0x01A0: 0x01A1,
}
COLORS = {
    "KERNEL": "#455a64",
    "SEND_TOKENS": "#1976d2",
    "SEND_COUNTS": "#7b1fa2",
    "WAIT_RECV_COUNTS": "#ef6c00",
    "LOCAL_WINDOW_COPY": "#388e3c",
    "WAIT_CUMSUM": "#d32f2f",
    "FORMAT_OUTPUT": "#00838f",
    "PREPARE_TOKEN_INPUT": "#64b5f6",
    "TRANSFER_TOKENS": "#0d47a1",
    "PREPARE_COUNT_INPUT": "#ba68c8",
    "CLEAR_RECV_STATUS": "#ffb300",
    "GATHER_RECV_COUNTS": "#e64a19",
    "GET_CUMSUM": "#5e35b1",
    "BUILD_RECV_CUMSUM": "#3949ab",
    "PUBLISH_RECV_COUNTS": "#283593",
    "RESET_CUMSUM_FLAG": "#ad1457",
    "BUILD_EXPERT_TASKS": "#7cb342",
    "CLEAR_LOCAL_FLAGS": "#00695c",
    "DRAIN_PIPELINES": "#78909c",
    "NO_ACTIVE_TOKENS": "#9e9e9e",
    "NO_ASSIGNED_STATUS": "#757575",
    "NO_VALID_EXPERT": "#424242",
    "READ_RECV_COUNTS": "#f9a825",
    "BUILD_LOCAL_PREFIX": "#6a1b9a",
    "PUBLISH_LOCAL_COUNTS": "#1565c0",
}
STAGE_DESCRIPTIONS = {
    "KERNEL": "Process total",
    "SEND_TOKENS": "AllToAllDispatch: send token data",
    "SEND_COUNTS": "CalAndSendCnt*: calculate and send token counts",
    "WAIT_RECV_COUNTS": "WaitDispatch: wait, clear status and sum received counts",
    "LOCAL_WINDOW_COPY": "LocalWindowCopy: compact local expert output",
    "WAIT_CUMSUM": "WaitCumSumFlag: wait for count prefix sum",
    "FORMAT_OUTPUT": "WaitAndFormatOutput: active cores write expert output",
    "PREPARE_TOKEN_INPUT": "ExpIdsCopyAndMaskCal: prepare token dispatch input",
    "TRANSFER_TOKENS": "AllToAllDispatch: copy token data to remote windows",
    "PREPARE_COUNT_INPUT": "ExpIdsCopyAndMaskCal: prepare count calculation input",
    "CLEAR_RECV_STATUS": "WaitDispatchClearStatus: clear received status flags",
    "GATHER_RECV_COUNTS": "GatherSumRecvCnt: sum per-core received counts",
    "GET_CUMSUM": "GetCumSum: calculate prefix across count cores",
    "BUILD_RECV_CUMSUM": "CalRecvAndSetFlag: build local received-count prefix sum",
    "PUBLISH_RECV_COUNTS": "CalRecvAndSetFlag: publish counts and completion flags",
    "RESET_CUMSUM_FLAG": "WaitCumSumFlag: record progress and clear sync flags",
    "BUILD_EXPERT_TASKS": "SetValidExpertInfo: select experts with received tokens",
    "CLEAR_LOCAL_FLAGS": "ClearLocalWindowDataFlags: clear consumed data flags",
    "DRAIN_PIPELINES": "PipeBarrier: wait for previous pipeline operations",
    "NO_ACTIVE_TOKENS": "AllToAllDispatch: this core has no active token to send",
    "NO_ASSIGNED_STATUS": "LocalWindowCopy: this core has no assigned status",
    "NO_VALID_EXPERT": "LocalWindowCopy: assigned experts received no token",
    "READ_RECV_COUNTS": "LocalWindowCopySerial: read compact received counts",
    "BUILD_LOCAL_PREFIX": "LocalWindowCopySerial: build this core's output positions",
    "PUBLISH_LOCAL_COUNTS": "LocalWindowCopySerial: publish sendCounts and expertTokenNums",
}
TIMELINE_STAGES = (
    "PREPARE_TOKEN_INPUT",
    "TRANSFER_TOKENS",
    "SEND_COUNTS",
    "PREPARE_COUNT_INPUT",
    "WAIT_RECV_COUNTS",
    "CLEAR_RECV_STATUS",
    "GATHER_RECV_COUNTS",
    "GET_CUMSUM",
    "BUILD_RECV_CUMSUM",
    "PUBLISH_RECV_COUNTS",
    "DRAIN_PIPELINES",
    "WAIT_CUMSUM",
    "RESET_CUMSUM_FLAG",
    "READ_RECV_COUNTS",
    "BUILD_LOCAL_PREFIX",
    "PUBLISH_LOCAL_COUNTS",
    "BUILD_EXPERT_TASKS",
    "FORMAT_OUTPUT",
    "CLEAR_LOCAL_FLAGS",
)
SEND_METRIC_FIELDS = {
    0x0171: "tasks",
    0x0173: "position_cycles",
    0x0174: "input_cycles",
    0x0175: "compute_cycles",
    0x0176: "pack_cycles",
    0x0177: "buffer_wait_cycles",
    0x0178: "serial_buffer_init_cycles",
}
SEND_SUMMARY_FIELDS = (
    "rank",
    "core_id",
    "serial_mode",
    "tasks",
    "send_buffers_allocated",
    "send_buffers_used",
    "buffer_init_cycles",
    "transfer_cycles",
    "position_cycles",
    "input_cycles",
    "compute_cycles",
    "pack_cycles",
    "buffer_wait_cycles",
    "other_cycles",
)
CORE_SUMMARY_FIELDS = (
    "rank", "core_id", "format_version", "serial_mode", "role", "kernel_cycles", "kernel_end_cycle",
    "prepare_token_cycles", "transfer_cycles", "send_count_cycles", "send_tasks", "send_buffers",
    "send_buffers_used", "send_buffer_init_cycles", "send_position_cycles", "send_input_cycles",
    "send_compute_cycles", "send_pack_cycles",
    "send_buffer_wait_cycles", "send_other_cycles", "serial_buffer_init_cycles", "prepare_count_cycles",
    "wait_recv_cycles", "wait_recv_polls", "wait_recv_other_cycles", "clear_recv_cycles", "gather_recv_cycles",
    "get_cumsum_cycles", "get_cumsum_polls", "build_recv_cumsum_cycles", "publish_recv_cycles", "drain_cycles",
    "wait_cumsum_cycles", "wait_cumsum_polls", "reset_cumsum_cycles", "read_recv_counts_cycles",
    "build_local_prefix_cycles", "publish_local_counts_cycles", "build_expert_tasks_cycles", "assigned_status",
    "valid_status", "assigned_tokens", "copy_batches", "format_polls", "format_cycles", "clear_local_flags_cycles",
)


def rank_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"rank(\d+)", path.name)
    return int(match.group(1)) if match else fallback


def parse_binary(path: Path, rank: int) -> list[dict[str, int | str]]:
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"{path}: file is shorter than the 64-byte header")
    header = struct.unpack_from("<8I", data, 0)
    magic, version_layout, core_num, record_size, capacity, count_stride, data_offset, _ = header
    if magic != MAGIC:
        raise ValueError(f"{path}: invalid magic 0x{magic:08X}")
    if record_size != 16:
        raise ValueError(f"{path}: unsupported record size {record_size}")
    expected_size = data_offset + core_num * capacity * record_size
    if len(data) < expected_size:
        raise ValueError(f"{path}: got {len(data)} bytes, expected at least {expected_size}")

    rows: list[dict[str, int | str]] = []
    for core_id in range(core_num):
        count_offset = HEADER_SIZE + core_id * count_stride
        count = struct.unpack_from("<I", data, count_offset)[0]
        if count > capacity:
            raise ValueError(f"{path}: core {core_id} count {count} exceeds capacity {capacity}")
        core_offset = data_offset + core_id * capacity * record_size
        for index in range(count):
            event_id, payload, cycle = struct.unpack_from("<IIQ", data, core_offset + index * record_size)
            rows.append({
                "rank": rank,
                "core_id": core_id,
                "index": index,
                "event_id": event_id,
                "event_name": EVENT_NAMES.get(event_id, f"EVT_0x{event_id:04X}"),
                "payload": payload,
                "cycle": cycle,
                "format_version": version_layout & 0xFFFF,
                "layout": version_layout >> 16,
            })
    return rows


def pair_events(rows: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    by_lane: dict[tuple[int, int], list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        by_lane[(int(row["rank"]), int(row["core_id"]))].append(row)

    pairs: list[dict[str, int | str]] = []
    end_to_begin = {end: begin for begin, end in PAIR_MAP.items()}
    for (rank, core_id), lane_rows in sorted(by_lane.items()):
        pending: dict[int, deque[dict[str, int | str]]] = defaultdict(deque)
        for row in sorted(lane_rows, key=lambda item: int(item["index"])):
            event_id = int(row["event_id"])
            if event_id in PAIR_MAP:
                pending[event_id].append(row)
            elif event_id in end_to_begin:
                begin_id = end_to_begin[event_id]
                if not pending[begin_id]:
                    continue
                begin = pending[begin_id].popleft()
                begin_cycle = int(begin["cycle"])
                end_cycle = int(row["cycle"])
                pairs.append({
                    "rank": rank,
                    "core_id": core_id,
                    "stage": EVENT_NAMES[begin_id],
                    "begin_cycle": begin_cycle,
                    "end_cycle": end_cycle,
                    "duration_cycles": end_cycle - begin_cycle,
                    "payload": int(begin["payload"]),
                    "begin_payload": int(begin["payload"]),
                    "end_payload": int(row["payload"]),
                    "format_version": int(begin["format_version"]),
                })
    return pairs


def collect_send_metrics(rows: list[dict[str, int | str]]) -> dict[tuple[int, int], dict[str, int]]:
    lane_metrics: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for row in rows:
        event_id = int(row["event_id"])
        if event_id == 0x0172:
            lane = (int(row["rank"]), int(row["core_id"]))
            payload = int(row["payload"])
            lane_metrics[lane]["send_buffers"] = payload & 0xFF
            lane_metrics[lane]["send_buffers_used"] = (payload >> 8) & 0xFF
            lane_metrics[lane]["buffer_init_cycles"] = payload >> 16
            continue
        if event_id in SEND_METRIC_FIELDS:
            lane = (int(row["rank"]), int(row["core_id"]))
            lane_metrics[lane][SEND_METRIC_FIELDS[event_id]] = int(row["payload"])
    return lane_metrics


def build_core_summary(rows: list[dict[str, int | str]],
                       pairs: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    by_lane: dict[tuple[int, int], dict[str, dict[str, int | str]]] = defaultdict(dict)
    for pair in pairs:
        lane = (int(pair["rank"]), int(pair["core_id"]))
        by_lane[lane][str(pair["stage"])] = pair
    lane_metrics = collect_send_metrics(rows)

    def field(stages: dict[str, dict[str, int | str]], stage: str, name: str) -> int | str:
        return stages[stage][name] if stage in stages else ""

    summaries: list[dict[str, int | str]] = []
    for (rank, core_id), stages in sorted(by_lane.items()):
        format_version = int(field(stages, "KERNEL", "format_version") or 0)
        kernel_payload = int(field(stages, "KERNEL", "begin_payload") or 0)
        serial_mode = bool(kernel_payload & 0x80000000)
        send_token_cores = (kernel_payload >> 16) & 0x7FFF
        metrics = lane_metrics.get((rank, core_id), {})
        transfer_cycles = field(stages, "TRANSFER_TOKENS", "duration_cycles")
        send_measured_cycles = sum(metrics.get(name, 0) for name in (
            "buffer_init_cycles", "position_cycles", "input_cycles", "compute_cycles", "pack_cycles",
            "buffer_wait_cycles"))
        send_other_cycles: int | str = ""
        if transfer_cycles != "" and metrics:
            send_other_cycles = int(transfer_cycles) - send_measured_cycles
        wait_recv_cycles = field(stages, "WAIT_RECV_COUNTS", "duration_cycles")
        clear_recv_cycles = field(stages, "CLEAR_RECV_STATUS", "duration_cycles")
        gather_recv_cycles = field(stages, "GATHER_RECV_COUNTS", "duration_cycles")
        wait_recv_other_cycles: int | str = ""
        if wait_recv_cycles != "":
            wait_recv_other_cycles = (int(wait_recv_cycles) - int(clear_recv_cycles or 0) -
                                      int(gather_recv_cycles or 0) - metrics.get("serial_buffer_init_cycles", 0))
        format_payload = field(stages, "CLEAR_LOCAL_FLAGS", "end_payload")
        copy_batches = int(format_payload) & 0xFFFF if format_payload != "" else ""
        format_polls = int(format_payload) >> 16 if format_payload != "" else ""
        summaries.append({
            "rank": rank,
            "core_id": core_id,
            "format_version": format_version,
            "serial_mode": int(serial_mode),
            "role": "serial" if serial_mode else ("send_token" if core_id < send_token_cores else "cumsum"),
            "kernel_cycles": field(stages, "KERNEL", "duration_cycles"),
            "kernel_end_cycle": field(stages, "KERNEL", "end_cycle"),
            "prepare_token_cycles": field(stages, "PREPARE_TOKEN_INPUT", "duration_cycles"),
            "transfer_cycles": transfer_cycles,
            "send_count_cycles": field(stages, "SEND_COUNTS", "duration_cycles"),
            "send_tasks": metrics.get("tasks", ""),
            "send_buffers": metrics.get("send_buffers", ""),
            "send_buffers_used": metrics.get("send_buffers_used", ""),
            "send_buffer_init_cycles": metrics.get("buffer_init_cycles", ""),
            "send_position_cycles": metrics.get("position_cycles", ""),
            "send_input_cycles": metrics.get("input_cycles", ""),
            "send_compute_cycles": metrics.get("compute_cycles", ""),
            "send_pack_cycles": metrics.get("pack_cycles", ""),
            "send_buffer_wait_cycles": metrics.get("buffer_wait_cycles", ""),
            "send_other_cycles": send_other_cycles,
            "serial_buffer_init_cycles": metrics.get("serial_buffer_init_cycles", ""),
            "prepare_count_cycles": field(stages, "PREPARE_COUNT_INPUT", "duration_cycles"),
            "wait_recv_cycles": wait_recv_cycles,
            "wait_recv_polls": field(stages, "WAIT_RECV_COUNTS", "end_payload"),
            "wait_recv_other_cycles": wait_recv_other_cycles,
            "clear_recv_cycles": clear_recv_cycles,
            "gather_recv_cycles": gather_recv_cycles,
            "get_cumsum_cycles": field(stages, "GET_CUMSUM", "duration_cycles"),
            "get_cumsum_polls": field(stages, "GET_CUMSUM", "end_payload"),
            "build_recv_cumsum_cycles": field(stages, "BUILD_RECV_CUMSUM", "duration_cycles"),
            "publish_recv_cycles": field(stages, "PUBLISH_RECV_COUNTS", "duration_cycles"),
            "drain_cycles": field(stages, "DRAIN_PIPELINES", "duration_cycles"),
            "wait_cumsum_cycles": field(stages, "WAIT_CUMSUM", "duration_cycles"),
            "wait_cumsum_polls": field(stages, "WAIT_CUMSUM", "end_payload"),
            "reset_cumsum_cycles": field(stages, "RESET_CUMSUM_FLAG", "duration_cycles"),
            "read_recv_counts_cycles": field(stages, "READ_RECV_COUNTS", "duration_cycles"),
            "build_local_prefix_cycles": field(stages, "BUILD_LOCAL_PREFIX", "duration_cycles"),
            "publish_local_counts_cycles": field(stages, "PUBLISH_LOCAL_COUNTS", "duration_cycles"),
            "build_expert_tasks_cycles": field(stages, "BUILD_EXPERT_TASKS", "duration_cycles"),
            "assigned_status": field(stages, "BUILD_EXPERT_TASKS", "begin_payload"),
            "valid_status": field(stages, "BUILD_EXPERT_TASKS", "end_payload"),
            "assigned_tokens": field(stages, "FORMAT_OUTPUT", "end_payload"),
            "copy_batches": copy_batches,
            "format_polls": format_polls,
            "format_cycles": field(stages, "FORMAT_OUTPUT", "duration_cycles"),
            "clear_local_flags_cycles": field(stages, "CLEAR_LOCAL_FLAGS", "duration_cycles"),
        })
    return summaries


def build_send_summary(rows: list[dict[str, int | str]],
                       pairs: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    lane_metrics = collect_send_metrics(rows)
    kernel_payloads = {
        (int(pair["rank"]), int(pair["core_id"])): int(pair["begin_payload"])
        for pair in pairs if pair["stage"] == "KERNEL"
    }

    transfer_pairs = [pair for pair in pairs if pair["stage"] == "TRANSFER_TOKENS"]
    transfer_pairs = [pair for pair in transfer_pairs
                      if (int(pair["rank"]), int(pair["core_id"])) in lane_metrics]
    summaries = []
    for transfer in transfer_pairs:
        rank = int(transfer["rank"])
        core_id = int(transfer["core_id"])
        metrics = lane_metrics[(rank, core_id)]
        measured_cycles = sum(metrics.get(name, 0) for name in (
            "buffer_init_cycles", "position_cycles", "input_cycles", "compute_cycles", "pack_cycles",
            "buffer_wait_cycles"))
        summaries.append({
            "rank": rank,
            "core_id": core_id,
            "serial_mode": int(bool(kernel_payloads.get((rank, core_id), 0) & 0x80000000)),
            "tasks": metrics.get("tasks", ""),
            "send_buffers_allocated": metrics.get("send_buffers", ""),
            "send_buffers_used": metrics.get("send_buffers_used", ""),
            "buffer_init_cycles": metrics.get("buffer_init_cycles", ""),
            "transfer_cycles": int(transfer["duration_cycles"]),
            "position_cycles": metrics.get("position_cycles", ""),
            "input_cycles": metrics.get("input_cycles", ""),
            "compute_cycles": metrics.get("compute_cycles", ""),
            "pack_cycles": metrics.get("pack_cycles", ""),
            "buffer_wait_cycles": metrics.get("buffer_wait_cycles", ""),
            "other_cycles": int(transfer["duration_cycles"]) - measured_cycles,
        })
    return sorted(summaries, key=lambda item: int(item["transfer_cycles"]), reverse=True)


def align_cycles(rows: list[dict[str, int | str]], pairs: list[dict[str, int | str]], keep_origin: bool) -> None:
    if keep_origin:
        return
    origin: dict[int, int] = {}
    for row in rows:
        rank = int(row["rank"])
        cycle = int(row["cycle"])
        origin[rank] = min(origin.get(rank, cycle), cycle)
    for row in rows:
        row["cycle"] = int(row["cycle"]) - origin[int(row["rank"])]
    for pair in pairs:
        rank_origin = origin[int(pair["rank"])]
        pair["begin_cycle"] = int(pair["begin_cycle"]) - rank_origin
        pair["end_cycle"] = int(pair["end_cycle"]) - rank_origin


def write_csv(path: Path, rows: list[dict[str, int | str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_trace(path: Path, pairs: list[dict[str, int | str]], freq_mhz: float) -> None:
    events = []
    for pair in pairs:
        stage = str(pair["stage"])
        payload = int(pair["payload"])
        args = {
            "description": STAGE_DESCRIPTIONS.get(stage, stage),
            "begin_payload": payload,
            "end_payload": int(pair["end_payload"]),
        }
        if stage == "KERNEL":
            args["serial_mode"] = bool(payload & 0x80000000)
            args["send_token_cores"] = (payload >> 16) & 0x7FFF
            args["cumsum_cores"] = payload & 0xFFFF
        events.append({
            "name": stage,
            "cat": "FullMeshV2",
            "ph": "X",
            "pid": f"rank{pair['rank']}",
            "tid": f"AIV{pair['core_id']}",
            "ts": float(pair["begin_cycle"]) / freq_mhz,
            "dur": float(pair["duration_cycles"]) / freq_mhz,
            "args": args,
        })
    path.write_text(json.dumps({"traceEvents": events}, indent=2), encoding="utf-8")


def write_svg(path: Path, pairs: list[dict[str, int | str]], freq_mhz: float) -> None:
    lanes = sorted({(int(pair["rank"]), int(pair["core_id"])) for pair in pairs})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    left = 130
    legend_columns = 2
    legend_rows = (len(TIMELINE_STAGES) + legend_columns - 1) // legend_columns
    top = 43 + legend_rows * 22
    lane_height = 24
    plot_width = 1400
    width = left + plot_width + 30
    height = top + max(1, len(lanes)) * lane_height + 55
    max_cycle = max((int(pair["end_cycle"]) for pair in pairs), default=1)
    scale = plot_width / max(1, max_cycle)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.lane{fill:#fafafa;stroke:#eeeeee}.label{fill:#263238}.title{font-size:16px;font-weight:bold}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="title" x="12" y="22">FullMeshV2 per-core timeline</text>',
    ]
    legend_column_width = (width - 24) / legend_columns
    for index, stage in enumerate(TIMELINE_STAGES):
        row = index // legend_columns
        column = index % legend_columns
        x = 12 + column * legend_column_width
        y = 35 + row * 22
        label = html.escape(f"{stage}: {STAGE_DESCRIPTIONS[stage]}")
        svg.append(f'<rect x="{x:.2f}" y="{y}" width="14" height="14" fill="{COLORS[stage]}"/>')
        svg.append(f'<text class="label" x="{x + 20:.2f}" y="{y + 12}">{label}</text>')
    for lane, index in lane_index.items():
        y = top + index * lane_height
        svg.append(f'<rect class="lane" x="{left}" y="{y}" width="{plot_width}" height="{lane_height}"/>')
        svg.append(f'<text class="label" x="8" y="{y + 16}">rank{lane[0]}/AIV{lane[1]}</text>')
    # Draw wider outer stages first so nested wait/format stages remain visible.
    for pair in sorted(pairs, key=lambda item: int(item["duration_cycles"]), reverse=True):
        stage = str(pair["stage"])
        if stage == "KERNEL":
            continue
        y = top + lane_index[(int(pair["rank"]), int(pair["core_id"]))] * lane_height + 3
        x = left + int(pair["begin_cycle"]) * scale
        rect_width = max(1.0, int(pair["duration_cycles"]) * scale)
        color = COLORS.get(stage, "#607d8b")
        duration_us = int(pair["duration_cycles"]) / freq_mhz
        tooltip = html.escape(
            f"{stage}: {duration_us:.3f} us, begin_payload={pair['begin_payload']}, "
            f"end_payload={pair['end_payload']}")
        svg.append(f'<rect x="{x:.2f}" y="{y}" width="{rect_width:.2f}" height="18" fill="{color}"><title>{tooltip}</title></rect>')
    total_us = max_cycle / freq_mhz
    svg.append(f'<text x="{left}" y="{height - 18}">0 us</text>')
    svg.append(f'<text x="{left + plot_width - 90}" y="{height - 18}">{total_us:.3f} us</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="fullmesh_v2_prof_rank*.bin")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--freq-mhz", type=float, default=1000.0,
                        help="device frequency in MHz; cycles / MHz = microseconds")
    parser.add_argument("--keep-cycle-origin", action="store_true",
                        help="do not align each rank's first event to cycle zero")
    args = parser.parse_args()
    if args.freq_mhz <= 0:
        parser.error("--freq-mhz must be greater than zero")

    rows: list[dict[str, int | str]] = []
    for index, file_path in enumerate(args.files):
        rows.extend(parse_binary(file_path, rank_from_path(file_path, index)))
    if not rows:
        raise SystemExit("no profiler events found")
    pairs = pair_events(rows)
    align_cycles(rows, pairs, args.keep_cycle_origin)
    summaries = build_core_summary(rows, pairs)
    send_summary = build_send_summary(rows, pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fullmesh_v2_events.csv", rows,
              ["rank", "core_id", "index", "event_id", "event_name", "payload", "cycle",
               "format_version", "layout"])
    write_csv(args.output_dir / "fullmesh_v2_stages.csv", pairs,
              ["rank", "core_id", "stage", "begin_cycle", "end_cycle", "duration_cycles", "payload",
               "begin_payload", "end_payload", "format_version"])
    write_csv(args.output_dir / "fullmesh_v2_core_summary.csv", summaries, list(CORE_SUMMARY_FIELDS))
    write_csv(args.output_dir / "fullmesh_v2_send_summary.csv", send_summary, list(SEND_SUMMARY_FIELDS))
    write_trace(args.output_dir / "fullmesh_v2_trace.json", pairs, args.freq_mhz)
    write_svg(args.output_dir / "fullmesh_v2_timeline.svg", pairs, args.freq_mhz)
    if send_summary:
        print(",".join(SEND_SUMMARY_FIELDS))
        print(",".join(str(send_summary[0][field]) for field in SEND_SUMMARY_FIELDS))
    else:
        print(f"parsed {len(rows)} events and {len(pairs)} stage intervals into {args.output_dir}")


if __name__ == "__main__":
    main()
