#!/usr/bin/env python3
"""Run the existing MegaMoe test case with the compact critical-path parser.

The runner executes mega_moe.py from memory.  It does not modify that test
file: its legacy diagnostic parser is renamed in memory, the compact parser is
injected, and the target performance case is selected before execution.
"""

import argparse
import html
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


TRACE_MAGIC = 0x43505452
TRACE_VERSION = 1
TRACE_HEADER_WORDS = 20
TRACE_RECORD_WORDS = 26
TRACE_POINT_NAMES = (
    "process_begin",
    "rolling_begin",
    "dispatch_next_wave_end",
    "dispatch_second_next_wave_end",
    "first_publish_begin",
    "first_publish_end",
    "last_publish_begin",
    "last_publish_end",
    "first_ready_begin",
    "first_ready_end",
    "last_ready_begin",
    "last_ready_end",
    "first_combine_begin",
    "first_combine_scalar_end",
    "last_combine_begin",
    "last_combine_scalar_end",
    "last_combine_actual_end",
    "rolling_body_end",
    "rolling_end",
    "local_sync_end",
    "cross_rank_end",
    "work_end",
)
TRACE_WAIT_NAMES = (
    "send_cnt_ready",
    "gmm1_input_ready",
    "gmm2_input_ready",
    "gmm2_combine_ready",
)
TRACE_SPANS = (
    (0, 1, "setup_to_rolling"),
    (1, 2, "rolling_to_dispatch_next"),
    (4, 5, "first_publish_fix_store_issue"),
    (6, 7, "last_publish_fix_store_issue"),
    (8, 9, "first_ready_fanin"),
    (10, 11, "last_ready_fanin"),
    (12, 13, "first_combine_scalar_issue"),
    (12, 16, "all_combine_to_actual_end"),
    (14, 15, "last_chunk_to_scalar_drain"),
    (15, 16, "final_async_completion"),
    (1, 17, "rolling_body"),
    (17, 18, "terminal_gmm1_drain"),
    (18, 19, "local_pipeall_sync"),
    (19, 20, "final_cross_rank"),
    (20, 21, "unpermute_actual"),
    (0, 21, "process_to_work_end"),
)

# This is the case discussed in the GMM2/Combine investigation.  Assignments
# are replaced in memory before the existing test script computes derived
# values such as global_bs and moe_expert_num.
TARGET_ASSIGNMENTS = (
    ("x_active_mask_dim", "0"),
    ("graph_type", "0"),
    ("quant_mode", "4"),
    ("comm_quant_mode", "0"),
    ("is_performance", "1"),
    ("is_variable_bs", "0"),
    ("is_full_core", "True"),
    ("server_num", "1"),
    ("tp_world_size", "1"),
    ("rank_per_dev", "2"),
    ("expert_per_rank", "16"),
    ("rank_num_per_shared_expert", "0"),
    ("bs", "512"),
    ("local_moe_expert_num", "16"),
    ("h", "4096"),
    ("n", "4096"),
    ("k", "8"),
    ("is_topk_balance", "True"),
)

EXPECTED_SOURCE_MACROS = (
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe_impl_base.h",
        "MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE",
        "1",
    ),
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp",
        "MEGA_MOE_WEIGHT1_INTERLEAVED",
        "1",
    ),
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe.h",
        "MEGA_MOE_GMM2_GM_READY_SMOKE_TEST",
        "0",
    ),
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe.h",
        "MEGA_MOE_GMM2_READY_MODE",
        "MEGA_MOE_GMM2_READY_PER_EXPERT",
    ),
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe.h",
        "MEGA_MOE_GMM2_FINAL_BATCH_COMBINE_ONLY",
        "0",
    ),
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe.h",
        "MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG",
        "0",
    ),
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe_workspace_info.h",
        "MEGA_MOE_GMM2_FANIN_MODE",
        "MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE",
    ),
    (
        "mc2/mega_moe/op_kernel/arch35/mega_moe_workspace_info.h",
        "MEGA_MOE_GMM2_READY_CORE_STRIDE",
        "16",
    ),
)


def _format_values(values):
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return "n/a"
    return (
        f"{int(values.min()):>10}/"
        f"{int(np.median(values)):>10}/"
        f"{int(values.max()):>10}"
    )


def _write_trace_artifacts(records, aic_num, rank, reference, header):
    """Write raw JSON plus a self-contained per-core SVG timeline."""
    def relative_value(value):
        value = int(value)
        if value == 0:
            return None
        delta = (value - reference) & 0xFFFFFFFF
        return delta - 0x100000000 if delta >= 0x80000000 else delta

    cores = []
    for record_idx, record in enumerate(records):
        if record_idx < aic_num:
            core_type = "AIC"
            logical_idx = record_idx
        else:
            aiv_physical_idx = record_idx - aic_num
            core_type = "AIV0" if aiv_physical_idx % 2 == 0 else "AIV1"
            logical_idx = aiv_physical_idx // 2
        points = {
            name: relative_value(record[point_idx])
            for point_idx, name in enumerate(TRACE_POINT_NAMES)
        }
        waits = {
            name: int(record[22 + wait_idx])
            for wait_idx, name in enumerate(TRACE_WAIT_NAMES)
        }
        cores.append(
            {
                "record_index": record_idx,
                "core_type": core_type,
                "logical_index": logical_idx,
                "label": f"{core_type}-{logical_idx:02d}",
                "points": points,
                "wait_cycles": waits,
            }
        )

    artifact_base = Path.cwd() / f"mega_moe_critical_path_rank{rank}"
    json_path = artifact_base.with_suffix(".json")
    json_path.write_text(
        json.dumps({"header": header, "cores": cores}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    colors = {
        "setup": "#64748b",
        "gmm": "#2563eb",
        "publish": "#dc2626",
        "swiglu": "#7c3aed",
        "dispatch": "#0891b2",
        "ready": "#eab308",
        "combine": "#16a34a",
        "drain": "#f97316",
        "sync": "#ef4444",
        "cross_rank": "#a855f7",
        "unpermute": "#14b8a6",
        "idle": "#94a3b8",
    }
    ordered_cores = (
        [core for core in cores if core["core_type"] == "AIC"]
        + [core for core in cores if core["core_type"] == "AIV0"]
        + [core for core in cores if core["core_type"] == "AIV1"]
    )
    plotted_cycles = [
        value
        for core in cores
        for value in core["points"].values()
        if value is not None
    ]
    min_cycle = min(plotted_cycles)
    max_cycle = max(plotted_cycles)
    if min_cycle == max_cycle:
        max_cycle += 1
    cycle_range = max_cycle - min_cycle
    label_width = 112
    plot_width = 1800
    wait_width = 380
    row_height = 25
    top = 105
    svg_width = label_width + plot_width + wait_width
    svg_height = top + len(ordered_cores) * row_height + 55

    def x_pos(cycle):
        return label_width + (cycle - min_cycle) * plot_width / cycle_range

    def add_segment(parts, y, begin, end, kind, label, exact):
        if begin is None or end is None or end < begin:
            return
        x_begin = x_pos(begin)
        width = max(1.0, x_pos(end) - x_begin)
        opacity = "0.92" if exact else "0.38"
        css_class = "exact" if exact else "inferred"
        title = html.escape(
            f"{label}: {begin} -> {end} cycles ({end - begin}); "
            f"{'exact boundary span' if exact else 'inferred between sparse critical points'}"
        )
        parts.append(
            f'<rect class="segment {css_class}" x="{x_begin:.2f}" y="{y + 5}" '
            f'width="{width:.2f}" height="15" rx="2" fill="{colors[kind]}" opacity="{opacity}">'
            f"<title>{title}</title></rect>"
        )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}">',
        "<style>",
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#0f172a}",
        ".core{font-size:12px;font-weight:600}.axis{font-size:11px}.wait{font-size:10px}",
        ".row:hover{fill:#e0f2fe}.marker{stroke:#111827;stroke-width:.7;opacity:.55}",
        ".group-line{stroke:#334155;stroke-width:1.5}.grid{stroke:#cbd5e1;stroke-width:.7}",
        ".inferred{stroke:#475569;stroke-width:.4;stroke-dasharray:2 2}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="8" y="22" font-size="16" font-weight="700">MegaMoe critical-path per-core timeline</text>',
        f'<text x="8" y="43" font-size="11">rank={rank} | cycles relative to AIC0 process_begin | '
        "solid/high-opacity = exact bounded span; translucent/dashed = inferred activity</text>",
    ]

    legend_x = 8
    for kind, label in (
        ("setup", "setup"),
        ("gmm", "GMM1/GMM2"),
        ("publish", "ready publish"),
        ("swiglu", "SwiGLU"),
        ("dispatch", "dispatch"),
        ("ready", "ready fan-in"),
        ("combine", "combine"),
        ("sync", "sync"),
        ("cross_rank", "cross-rank"),
        ("unpermute", "unpermute"),
    ):
        svg.append(
            f'<rect x="{legend_x}" y="56" width="13" height="10" fill="{colors[kind]}"/>'
            f'<text x="{legend_x + 17}" y="65" font-size="10">{label}</text>'
        )
        legend_x += 17 + len(label) * 7 + 13

    for tick_idx in range(11):
        cycle = min_cycle + cycle_range * tick_idx // 10
        x = x_pos(cycle)
        svg.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top - 15}" x2="{x:.2f}" y2="{svg_height - 25}"/>'
            f'<text class="axis" x="{x:.2f}" y="{top - 20}" text-anchor="middle">{cycle}</text>'
        )
    svg.append(
        f'<text class="axis" x="{label_width + plot_width + 8}" y="{top - 20}">accumulated waits (cycles)</text>'
    )

    previous_type = None
    for row_idx, core in enumerate(ordered_cores):
        y = top + row_idx * row_height
        if core["core_type"] != previous_type:
            svg.append(
                f'<line class="group-line" x1="0" y1="{y}" x2="{svg_width}" y2="{y}"/>'
            )
            previous_type = core["core_type"]
        if row_idx % 2:
            svg.append(
                f'<rect class="row" x="0" y="{y}" width="{svg_width}" height="{row_height}" fill="#f8fafc"/>'
            )
        else:
            svg.append(
                f'<rect class="row" x="0" y="{y}" width="{svg_width}" height="{row_height}" fill="#ffffff"/>'
            )
        svg.append(
            f'<text class="core" x="6" y="{y + 17}">{html.escape(core["label"])}</text>'
        )
        p = core["points"]
        add_segment(svg, y, p["process_begin"], p["rolling_begin"], "setup", "local setup", False)

        if core["core_type"] == "AIC":
            add_segment(
                svg, y, p["rolling_begin"], p["first_publish_begin"],
                "gmm", "GMM rolling before first publish", False,
            )
            add_segment(
                svg, y, p["first_publish_begin"], p["first_publish_end"],
                "publish", "first FIX->S and ready-store issue", True,
            )
            add_segment(
                svg, y, p["first_publish_end"], p["last_publish_begin"],
                "gmm", "GMM rolling between first/last publish", False,
            )
            add_segment(
                svg, y, p["last_publish_begin"], p["last_publish_end"],
                "publish", "last FIX->S and ready-store issue", True,
            )
            add_segment(
                svg, y, p["last_publish_end"], p["rolling_body_end"],
                "gmm", "GMM rolling tail", False,
            )
        elif core["core_type"] == "AIV0":
            add_segment(
                svg, y, p["rolling_begin"], p["rolling_body_end"],
                "swiglu", "GMM1 post/SwiGLU rolling", False,
            )
        else:
            add_segment(
                svg, y, p["rolling_begin"], p["rolling_body_end"],
                "dispatch", "Dispatch + Combine rolling", False,
            )
            add_segment(
                svg, y, p["rolling_begin"], p["dispatch_next_wave_end"],
                "dispatch", "dispatch next wave", True,
            )
            add_segment(
                svg, y, p["first_ready_begin"], p["first_ready_end"],
                "ready", "first ready fan-in", True,
            )
            add_segment(
                svg, y, p["first_combine_begin"], p["first_combine_scalar_end"],
                "combine", "first combine Scalar issue/drain", True,
            )
            add_segment(
                svg, y, p["last_ready_begin"], p["last_ready_end"],
                "ready", "last ready fan-in", True,
            )
            add_segment(
                svg, y, p["last_combine_begin"], p["last_combine_scalar_end"],
                "combine", "last combine Scalar issue/drain", True,
            )
            add_segment(
                svg, y, p["last_combine_scalar_end"], p["last_combine_actual_end"],
                "drain", "final combine async completion", True,
            )

        add_segment(
            svg, y, p["rolling_body_end"], p["rolling_end"],
            "drain", "terminal GMM1 sync/copyout", True,
        )
        add_segment(
            svg, y, p["rolling_end"], p["local_sync_end"],
            "sync", "local PIPE_ALL + core sync", True,
        )
        if core["core_type"] != "AIC":
            add_segment(
                svg, y, p["local_sync_end"], p["cross_rank_end"],
                "cross_rank", "final cross-rank sync", True,
            )
            add_segment(
                svg, y, p["cross_rank_end"], p["work_end"],
                "unpermute", "unpermute actual completion", True,
            )
        else:
            add_segment(
                svg, y, p["local_sync_end"], p["work_end"],
                "idle", "AIC tail/trace rendezvous", False,
            )

        for point_name, cycle in p.items():
            if cycle is None:
                continue
            x = x_pos(cycle)
            title = html.escape(f"{core['label']} {point_name}: {cycle} cycles")
            svg.append(
                f'<line class="marker" x1="{x:.2f}" y1="{y + 3}" x2="{x:.2f}" y2="{y + 22}">'
                f"<title>{title}</title></line>"
            )

        waits = core["wait_cycles"]
        wait_text = (
            f"send={waits['send_cnt_ready']} g1={waits['gmm1_input_ready']} "
            f"g2={waits['gmm2_input_ready']} ready={waits['gmm2_combine_ready']}"
        )
        svg.append(
            f'<text class="wait" x="{label_width + plot_width + 8}" y="{y + 16}">'
            f"{html.escape(wait_text)}</text>"
        )

    svg.append("</svg>")
    html_path = artifact_base.with_suffix(".html")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MegaMoe critical path</title>"
        "<style>body{margin:0;background:#e2e8f0}main{padding:12px}"
        ".panel{background:white;border-radius:8px;box-shadow:0 1px 5px #64748b;"
        "overflow:auto;max-height:96vh}</style></head><body><main><div class='panel'>"
        + "".join(svg)
        + "</div></main></body></html>",
        encoding="utf-8",
    )
    print(f"[TRACE_ARTIFACT] rank={rank} html={html_path}")
    print(f"[TRACE_ARTIFACT] rank={rank} json={json_path}")


def _print_critical_path_words(words, rank):
    if words.size < TRACE_HEADER_WORDS:
        raise RuntimeError(f"rank {rank}: y is too small for the critical-path header")

    (
        magic, version, header_words, record_words, aic_num, aiv_num,
        record_count, experts_per_wave, expert_per_rank, ready_mode,
        fanin_mode, ready_stride, final_only, wave_count, trace_flags,
        diag_m, diag_k, trace_rank, world_size, gmm1_interleaved,
    ) = (int(value) for value in words[:TRACE_HEADER_WORDS])
    if magic != TRACE_MAGIC:
        raise RuntimeError(
            f"rank {rank}: invalid trace magic 0x{magic:08x}; "
            "the installed kernel is not the critical-path build"
        )
    if version != TRACE_VERSION:
        raise RuntimeError(f"rank {rank}: unsupported trace version {version}")
    if header_words != TRACE_HEADER_WORDS or record_words != TRACE_RECORD_WORDS:
        raise RuntimeError(
            f"rank {rank}: invalid trace layout header={header_words}, record={record_words}"
        )
    if record_count != aic_num + aiv_num or aiv_num != aic_num * 2:
        raise RuntimeError(
            f"rank {rank}: invalid core layout aic={aic_num}, aiv={aiv_num}, records={record_count}"
        )

    required_words = header_words + record_count * record_words
    if words.size < required_words:
        raise RuntimeError(
            f"rank {rank}: y has {words.size} uint32 words; trace requires {required_words}"
        )
    records = words[header_words:required_words].reshape(record_count, record_words)
    aic_records = records[:aic_num]
    aiv_records = records[aic_num:]
    groups = (aic_records, aiv_records[0::2], aiv_records[1::2])
    mask32 = np.uint64(0xFFFFFFFF)

    origins = records[:, 0]
    nonzero_origins = origins[origins != 0]
    if nonzero_origins.size == 0:
        raise RuntimeError(f"rank {rank}: trace contains no process_begin")
    reference = np.uint64(
        int(aic_records[0, 0]) if aic_records[0, 0] != 0 else int(nonzero_origins[0])
    )

    def relative(values):
        deltas = (
            (np.asarray(values, dtype=np.uint64) - reference) & mask32
        ).astype(np.int64)
        deltas[deltas >= 0x80000000] -= 0x100000000
        return deltas

    def modular_span(begin, end):
        begin_u64 = np.asarray(begin, dtype=np.uint64)
        end_u64 = np.asarray(end, dtype=np.uint64)
        return (end_u64 - begin_u64) & mask32

    def format_timestamp(group, point_idx):
        values = group[:, point_idx]
        values = values[values != 0]
        return "n/a" if values.size == 0 else _format_values(relative(values))

    def format_span(group, begin_idx, end_idx):
        begin = group[:, begin_idx]
        end = group[:, end_idx]
        valid = (begin != 0) & (end != 0)
        if not np.any(valid):
            return "n/a"
        return _format_values(modular_span(begin[valid], end[valid]))

    ready_name = {0: "per_expert", 1: "per_batch"}.get(ready_mode, str(ready_mode))
    fanin_name = {
        0: "shared_atomic",
        1: "private_compact",
        2: "private_strided",
        3: "all_aiv_reduce",
    }.get(fanin_mode, str(fanin_mode))
    print(
        f"\n[CRITICAL_PATH_HEADER] rank={rank} trace_rank={trace_rank} "
        f"m={diag_m} k={diag_k} aic={aic_num} aiv={aiv_num} "
        f"experts_per_wave={experts_per_wave} experts={expert_per_rank} waves={wave_count} "
        f"ready={ready_name} fanin={fanin_name} stride={ready_stride} "
        f"final_only={final_only} interleaved={gmm1_interleaved} world={world_size} "
        f"terminal_pipeall={trace_flags & 1} final_mte2_s_bridge={(trace_flags >> 1) & 1}"
    )

    expected_runtime = {
        "ready_mode": (ready_mode, 0),
        "fanin_mode": (fanin_mode, 3),
        "ready_stride": (ready_stride, 16),
        "final_only": (final_only, 0),
        "gmm1_interleaved": (gmm1_interleaved, 1),
        "terminal_pipeall": (trace_flags & 1, 1),
        "final_mte2_s_bridge": ((trace_flags >> 1) & 1, 1),
        "trace_rank": (trace_rank, rank),
    }
    if os.environ.get("MEGA_MOE_TRACE_KEEP_CONFIG", "0") != "1":
        expected_runtime.update(
            {
                "m": (diag_m, 512),
                "k": (diag_k, 4096),
                "aic": (aic_num, 36),
                "experts_per_wave": (experts_per_wave, 5),
                "expert_per_rank": (expert_per_rank, 16),
                "wave_count": (wave_count, 4),
                "world_size": (world_size, 2),
            }
        )
    mismatches = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in expected_runtime.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            "runtime trace is not the requested target path: " + ", ".join(mismatches)
        )

    trace_header = {
        "schema": "mega_moe_critical_path_trace_v1",
        "rank": rank,
        "trace_rank": trace_rank,
        "reference": "AIC-00 process_begin",
        "reference_raw_u32": int(reference),
        "time_unit": "cycle",
        "m": diag_m,
        "k": diag_k,
        "aic_num": aic_num,
        "aiv_num": aiv_num,
        "experts_per_wave": experts_per_wave,
        "expert_per_rank": expert_per_rank,
        "wave_count": wave_count,
        "ready_mode": ready_name,
        "fanin_mode": fanin_name,
        "ready_core_stride_int32": ready_stride,
        "final_batch_combine_only": final_only,
        "gmm1_interleaved": gmm1_interleaved,
        "world_size": world_size,
        "terminal_pipeall": trace_flags & 1,
        "final_mte2_s_bridge": (trace_flags >> 1) & 1,
        "timeline_note": (
            "Exact spans have two recorded boundaries; broad GMM/SwiGLU/Dispatch+Combine "
            "bands are inferred between sparse critical points."
        ),
    }
    _write_trace_artifacts(records, aic_num, rank, int(reference), trace_header)

    print(
        "\n[CRITICAL_TIMELINE] cycle offset min/median/max from AIC0 process_begin; "
        "zero means that core does not execute the path"
    )
    print(f"{'point':<31} {'AIC':>32} {'AIV0':>32} {'AIV1':>32}")
    for point_idx, name in enumerate(TRACE_POINT_NAMES):
        print(
            f"{name:<31} "
            f"{format_timestamp(groups[0], point_idx):>32} "
            f"{format_timestamp(groups[1], point_idx):>32} "
            f"{format_timestamp(groups[2], point_idx):>32}"
        )

    print("\n[CRITICAL_SPANS] cycle min/median/max; subtraction is uint32 wrap-safe")
    print(f"{'span':<31} {'AIC':>32} {'AIV0':>32} {'AIV1':>32}")
    for begin_idx, end_idx, name in TRACE_SPANS:
        print(
            f"{name:<31} "
            f"{format_span(groups[0], begin_idx, end_idx):>32} "
            f"{format_span(groups[1], begin_idx, end_idx):>32} "
            f"{format_span(groups[2], begin_idx, end_idx):>32}"
        )

    print("\n[CRITICAL_WAITS] accumulated Scalar polling/fan-in cycles")
    print(f"{'wait':<31} {'AIC':>32} {'AIV0':>32} {'AIV1':>32}")
    for wait_idx, name in enumerate(TRACE_WAIT_NAMES, start=22):
        print(
            f"{name:<31} "
            f"{_format_values(groups[0][:, wait_idx]):>32} "
            f"{_format_values(groups[1][:, wait_idx]):>32} "
            f"{_format_values(groups[2][:, wait_idx]):>32}"
        )

    def relative_nonzero(group, point_idx):
        values = group[:, point_idx]
        values = values[values != 0]
        return relative(values).astype(np.int64)

    def gap_from_last_publish(publish_idx, ready_idx):
        publish = relative_nonzero(aic_records, publish_idx)
        ready = relative_nonzero(groups[2], ready_idx)
        if publish.size == 0 or ready.size == 0:
            return np.empty(0, dtype=np.int64)
        return ready - int(publish.max())

    def paired_gap(group, begin_idx, end_idx):
        begin = group[:, begin_idx]
        end = group[:, end_idx]
        valid = (begin != 0) & (end != 0)
        if not np.any(valid):
            return np.empty(0, dtype=np.int64)
        return (
            relative(end[valid]).astype(np.int64)
            - relative(begin[valid]).astype(np.int64)
        )

    first_publish = relative_nonzero(aic_records, 5)
    last_publish = relative_nonzero(aic_records, 7)
    derived = (
        (
            "first_publish_core_skew",
            np.array([first_publish.max() - first_publish.min()], dtype=np.int64)
            if first_publish.size else np.empty(0, dtype=np.int64),
        ),
        (
            "last_publish_core_skew",
            np.array([last_publish.max() - last_publish.min()], dtype=np.int64)
            if last_publish.size else np.empty(0, dtype=np.int64),
        ),
        ("first_ready_after_last_publish", gap_from_last_publish(5, 9)),
        ("last_ready_after_last_publish", gap_from_last_publish(7, 11)),
        ("dispatch_end_to_ready_begin", paired_gap(groups[2], 2, 8)),
        ("first_ready_to_combine", paired_gap(groups[2], 9, 12)),
        ("last_ready_to_actual_end", paired_gap(groups[2], 11, 16)),
    )
    print(
        "\n[CRITICAL_DERIVED] signed cycle gaps; a negative fan-in gap means "
        "the clocks/order are inconsistent"
    )
    for name, values in derived:
        print(f"{name:<36} {_format_values(values):>32}")
    print(
        "[CRITICAL_NOTE] combine_scalar_end is a Scalar issue/drain boundary; "
        "last_combine_actual_end is the only MTE2->Scalar completion-bridged combine point."
    )


def _print_wait_diagnostics(y_cpu, rank):
    # torch must be imported by the target script after it configures its
    # extension cache, so obtain the already-loaded module here.
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        raise RuntimeError("the target script did not import torch")
    raw = y_cpu.contiguous().view(torch_module.uint8).numpy().tobytes()
    words = np.frombuffer(raw, dtype=np.dtype("<u4"))
    _print_critical_path_words(words, rank)


def _read_default_macro(repo_root, relative_path, macro):
    source = (repo_root / relative_path).read_text(encoding="utf-8-sig")
    match = re.search(
        rf"^\s*#define\s+{re.escape(macro)}\s+(\S+)",
        source,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"cannot find default macro {macro} in {relative_path}")
    return match.group(1)


def _audit_source_macros(repo_root):
    print("[TRACE_SOURCE_CONFIG] checking compile-time defaults")
    failures = []
    for relative_path, macro, expected in EXPECTED_SOURCE_MACROS:
        actual = _read_default_macro(repo_root, relative_path, macro)
        print(f"  {macro}={actual}")
        if actual != expected:
            failures.append(f"{macro}={actual} (expected {expected})")
    if failures:
        raise RuntimeError("unexpected source macro defaults: " + ", ".join(failures))


def _is_source_repo_root(path):
    return all((path / relative_path).is_file()
               for relative_path, _macro, _expected in EXPECTED_SOURCE_MACROS)


def _find_source_repo_root(wrapper_path, explicit_root):
    if explicit_root:
        repo_root = Path(explicit_root).expanduser().resolve()
        if not _is_source_repo_root(repo_root):
            raise RuntimeError(
                f"--repo-root does not point to the MegaMoe source tree: {repo_root}"
            )
        return repo_root

    candidates = []
    for start in (wrapper_path.parent, Path.cwd()):
        candidates.extend((start, *start.parents))
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_source_repo_root(candidate):
            return candidate
    return None


def _replace_assignment(source, name, value, required=True):
    pattern = re.compile(rf"^(?P<indent>[ \t]*){re.escape(name)}\s*=.*$", re.MULTILINE)
    source, count = pattern.subn(rf"\g<indent>{name} = {value}", source, count=1)
    if required and count != 1:
        raise RuntimeError(f"cannot find global assignment for {name} in the test script")
    return source


def _prepare_test_source(test_script, keep_test_config):
    source = test_script.read_text(encoding="utf-8-sig")
    script_dir_assignment = "script_dir = os.path.dirname(os.path.abspath(__file__))"
    replaced_source = source.replace(
        script_dir_assignment,
        f"script_dir = {str(test_script.parent)!r}",
        1,
    )
    if replaced_source == source:
        raise RuntimeError(
            "cannot find the script_dir assignment required for multiprocessing"
        )
    source = replaced_source
    if not keep_test_config:
        for name, value in TARGET_ASSIGNMENTS:
            source = _replace_assignment(source, name, value)
        # This switch exists only in the camodel development variant.  Force
        # the real two-die path when present, but remain compatible with the
        # regular upstream test script where it does not exist.
        source = _replace_assignment(source, "FAKE_SEND_MODE", "0", required=False)

    parser_pattern = re.compile(
        r"^def print_wait_diagnostics\(y_cpu, rank\):",
        flags=re.MULTILINE,
    )
    source, count = parser_pattern.subn(
        "def _legacy_print_wait_diagnostics(y_cpu, rank):",
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError(
            "cannot find print_wait_diagnostics(y_cpu, rank) in the target script"
        )
    return source


def _parse_parent_args():
    default_script = os.environ.get(
        "MEGA_MOE_TRACE_TEST_SCRIPT",
        "/home/mace/workspace/moe_test/mega_moe/mega_moe.py",
    )
    parser = argparse.ArgumentParser(
        description="Run MegaMoe's target performance case and decode compact critical-path traces."
    )
    parser.add_argument("--test-script", default=default_script)
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("MEGA_MOE_TRACE_REPO_ROOT"),
        help="MegaMoe source repository root used only for the optional source-macro audit.",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Skip the source-macro audit; the installed kernel is still checked from its trace header.",
    )
    parser.add_argument(
        "--keep-test-config",
        action="store_true",
        help="Do not replace the test script's shape with bs512/topk8/16 experts/h=2n=4096.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check the source macro defaults without executing the NPU test.",
    )
    return parser.parse_known_args()


def _bootstrap_target():
    is_child_import = __name__ == "__mp_main__"
    if is_child_import:
        test_script_arg = os.environ.get("MEGA_MOE_TRACE_TEST_SCRIPT")
        if not test_script_arg:
            raise RuntimeError("MEGA_MOE_TRACE_TEST_SCRIPT is missing in the worker process")
        test_script = Path(test_script_arg).resolve()
        keep_test_config = os.environ.get("MEGA_MOE_TRACE_KEEP_CONFIG", "0") == "1"
    else:
        args, passthrough = _parse_parent_args()
        test_script = Path(args.test_script).resolve()
        keep_test_config = args.keep_test_config
        os.environ["MEGA_MOE_TRACE_TEST_SCRIPT"] = str(test_script)
        os.environ["MEGA_MOE_TRACE_KEEP_CONFIG"] = "1" if keep_test_config else "0"
        sys.argv = [sys.argv[0]] + passthrough

    if not test_script.is_file():
        raise FileNotFoundError(f"MegaMoe test script does not exist: {test_script}")

    wrapper_path = Path(__file__).resolve()
    if not is_child_import:
        if args.skip_source_check:
            if args.check_only:
                raise RuntimeError("--check-only cannot be combined with --skip-source-check")
            print(
                "[TRACE_SOURCE_CONFIG] skipped by request; "
                "the installed kernel will be validated from the runtime trace header"
            )
        else:
            repo_root = _find_source_repo_root(wrapper_path, args.repo_root)
            if repo_root is None:
                if args.check_only:
                    raise RuntimeError(
                        "cannot locate the MegaMoe source tree; pass --repo-root /path/to/repo"
                    )
                print(
                    "[TRACE_SOURCE_CONFIG] source tree not found; skipping the source-macro audit. "
                    "The installed kernel will be validated from the runtime trace header."
                )
            else:
                print(f"[TRACE_SOURCE_CONFIG] repo_root={repo_root}")
                _audit_source_macros(repo_root)
        if args.check_only:
            return
        if keep_test_config:
            print("[TRACE_CASE] preserving assignments from the target test script")
        else:
            print(
                "[TRACE_CASE] bs=512 topk=8 expert_per_rank=16 h=4096 "
                "gmm1_2n=4096 full_core balanced"
            )

    os.environ["MEGA_MOE_WAIT_DIAGNOSTICS"] = "1"
    os.environ["MEGA_MOE_CRITICAL_PATH_TRACE"] = "1"
    os.chdir(test_script.parent)
    if str(test_script.parent) not in sys.path:
        sys.path.insert(0, str(test_script.parent))

    source = _prepare_test_source(test_script, keep_test_config)
    globals()["print_wait_diagnostics"] = _print_wait_diagnostics
    # Execute in this module's globals so forkserver/spawn can resolve the test
    # script's worker functions from __main__/__mp_main__.  Keep __file__ on
    # this wrapper; script_dir was replaced above with the real test directory.
    exec(compile(source, str(test_script), "exec"), globals())


if __name__ in ("__main__", "__mp_main__"):
    _bootstrap_target()
