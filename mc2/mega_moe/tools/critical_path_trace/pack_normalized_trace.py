#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack a normalized MegaMoe trace into a small analysis-only JSON gzip."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


COMPACT_FORMAT = "mega_moe_compact_trace"
COMPACT_FORMAT_VERSION = 6
ROLES = ("AIC", "AIV0", "AIV1")
CONFIG_FIELDS = (
    "rank",
    "runtime_config_hash",
    "path_id",
    "bs",
    "h",
    "hidden_dim",
    "top_k",
    "moe_experts_per_rank",
    "shared_expert_count",
    "m_groups_per_wave",
    "world_size",
    "dispatch_quant_out_key",
    "combine_quant_mode",
    "topk_weights_prefetch",
    "weight1_interleaved",
    "role_counts",
    "repetition_count",
    "capture_capacity",
    "flags",
    "truncated",
)
TASK_LAYOUTS = (
    ("gs", "gmm1_scalar_setup", "gmm1_scalar_setup_begin_S", "gmm1_scalar_setup_end_S"),
    (
        "ds",
        "dispatch_scalar_gap",
        "dispatch_scalar_gap_begin_S",
        "dispatch_scalar_gap_end_S",
    ),
    (
        "gd",
        "gmm1_dispatch_wait",
        "gmm1_dispatch_wait_begin_S",
        "gmm1_dispatch_wait_end_S",
    ),
    ("gv", "gmm1_vector_wait", "gmm1_vector_wait_begin_S", "gmm1_vector_wait_end_S"),
    ("g1", "gmm1_problem", "gmm1_problem_begin_S", "gmm1_problem_end_S"),
    ("g2", "gmm2_problem", "gmm2_problem_begin_S", "gmm2_problem_end_S"),
    ("s", "swiglu_problem", "swiglu_problem_begin_S", "swiglu_problem_end_S"),
    ("d", "dispatch_task", "dispatch_task_begin_S", "dispatch_task_end_S"),
    ("c", "combine_group_task", "combine_group_begin_S", "combine_group_end_S"),
)
FIXED_PHASE_LAYOUTS = (
    ("ib", "input_buffer_init_begin_S", "input_buffer_init_end_S"),
    ("iq", "input_quantize_begin_S", "input_quantize_end_S"),
    ("ri", "route_index_begin_S", "route_index_end_S"),
    ("wr", "workspace_reset_begin_S", "workspace_reset_end_S"),
    ("ls", "local_prepare_sync_begin_S", "local_prepare_sync_end_S"),
    ("rs", "cross_rank_sync_begin_S", "cross_rank_sync_end_S"),
    ("eb", "expert_buffer_init_begin_S", "expert_buffer_init_end_S"),
    ("fd", "final_combine_drain_begin_S", "final_combine_drain_end_S"),
    ("me", "moe_epilogue_begin_S", "moe_epilogue_end_S"),
    ("mt", "moe_tail_sync_begin_S", "moe_tail_sync_end_S"),
    ("sg", "shared_expert_gmm2_begin_S", "shared_expert_gmm2_end_S"),
    ("os", "output_cross_rank_sync_begin_S", "output_cross_rank_sync_end_S"),
    ("ui", "unpermute_init_begin_S", "unpermute_init_end_S"),
    ("up", "unpermute_begin_S", "unpermute_end_S"),
)


class CompactTraceError(RuntimeError):
    """Raised when compacting would silently discard required trace semantics."""


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CompactTraceError(f"{location} must be an object")
    return value


def _require_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompactTraceError(f"{location} must be an integer")
    return value


def _require_timestamp_list(
    timestamps: dict[str, Any], name: str, location: str
) -> list[int]:
    values = timestamps.get(name, [])
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise CompactTraceError(f"{location}.{name} must be an integer array")
    return values


def _single_timestamp(timestamps: dict[str, Any], name: str, location: str) -> int:
    values = _require_timestamp_list(timestamps, name, location)
    if len(values) != 1:
        raise CompactTraceError(f"{location}.{name} must contain exactly one timestamp")
    return values[0]


def compact_normalized_trace(normalized: dict[str, Any]) -> dict[str, Any]:
    """Retain only configuration, per-core lifetime, and sparse task intervals."""

    metadata = _require_mapping(normalized.get("metadata"), "$.metadata")
    header = _require_mapping(metadata.get("header"), "$.metadata.header")
    records = normalized.get("records")
    if not isinstance(records, list):
        raise CompactTraceError("$.records must be an array")

    protocol_version = _require_int(
        normalized.get("protocol_version"), "$.protocol_version"
    )
    schema_id = normalized.get("schema_id")
    if not isinstance(schema_id, str) or not schema_id.startswith("0x"):
        raise CompactTraceError("$.schema_id must be a hexadecimal string")
    clock_bits = _require_int(header.get("clock_bits"), "$.metadata.header.clock_bits")
    if clock_bits <= 0:
        raise CompactTraceError("$.metadata.header.clock_bits must be positive")

    config: dict[str, Any] = {}
    for field in CONFIG_FIELDS:
        if field not in header:
            raise CompactTraceError(f"$.metadata.header lacks required field {field!r}")
        config[field] = header[field]
    role_counts = config["role_counts"]
    if (
        not isinstance(role_counts, list)
        or len(role_counts) != len(ROLES)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in role_counts
        )
    ):
        raise CompactTraceError(
            "$.metadata.header.role_counts must contain three non-negative integers"
        )
    expected_records = {
        (role, core) for role, count in zip(ROLES, role_counts) for core in range(count)
    }
    if len(records) != len(expected_records):
        raise CompactTraceError(
            f"$.records contains {len(records)} records; role_counts requires {len(expected_records)}"
        )

    known_spans = {layout[1] for layout in TASK_LAYOUTS}
    known_timestamp_names = {"process_begin_S", "work_end_S"}
    for _, begin_name, end_name in FIXED_PHASE_LAYOUTS:
        known_timestamp_names.update((begin_name, end_name))
    for _, _, begin_name, end_name in TASK_LAYOUTS:
        known_timestamp_names.update((begin_name, end_name))

    compact_records: list[list[Any]] = []
    actual_records: set[tuple[str, int]] = set()
    cycle_mask = (1 << clock_bits) - 1
    for record_index, record_value in enumerate(records):
        location = f"$.records[{record_index}]"
        record = _require_mapping(record_value, location)
        role = record.get("role")
        core = record.get("core")
        if role not in ROLES:
            raise CompactTraceError(f"{location}.role is unsupported: {role!r}")
        core = _require_int(core, f"{location}.core")
        actual_records.add((role, core))
        timestamps = _require_mapping(
            record.get("timestamps"), f"{location}.timestamps"
        )
        unknown_timestamps = set(timestamps) - known_timestamp_names
        if unknown_timestamps:
            raise CompactTraceError(
                f"{location}.timestamps contains unsupported points: {sorted(unknown_timestamps)}"
            )
        segments = _require_mapping(record.get("segments", {}), f"{location}.segments")
        unknown_spans = set(segments) - known_spans
        if unknown_spans:
            raise CompactTraceError(
                f"{location}.segments contains unsupported spans: {sorted(unknown_spans)}"
            )

        process_begin = _single_timestamp(
            timestamps, "process_begin_S", f"{location}.timestamps"
        )
        work_end = _single_timestamp(timestamps, "work_end_S", f"{location}.timestamps")
        phases: list[list[Any]] = []
        for kind, begin_name, end_name in FIXED_PHASE_LAYOUTS:
            begins = _require_timestamp_list(
                timestamps, begin_name, f"{location}.timestamps"
            )
            ends = _require_timestamp_list(
                timestamps, end_name, f"{location}.timestamps"
            )
            if len(begins) != len(ends) or len(begins) > 1:
                raise CompactTraceError(
                    f"{location} fixed phase {kind} must have zero or one paired boundary"
                )
            if begins:
                phases.append([kind, begins[0], ends[0]])
        phases.sort(key=lambda phase: (phase[1] - process_begin) & cycle_mask)
        tasks: list[list[Any]] = []
        for kind, span_name, begin_name, end_name in TASK_LAYOUTS:
            segment_values = segments.get(span_name, [])
            if not isinstance(segment_values, list):
                raise CompactTraceError(
                    f"{location}.segments.{span_name} must be an array"
                )
            begins = _require_timestamp_list(
                timestamps, begin_name, f"{location}.timestamps"
            )
            ends = _require_timestamp_list(
                timestamps, end_name, f"{location}.timestamps"
            )
            if len(segment_values) != len(begins) or len(begins) != len(ends):
                raise CompactTraceError(
                    f"{location} {span_name} metadata/begin/end lengths disagree"
                )
            for occurrence, (segment_value, begin, end) in enumerate(
                zip(segment_values, begins, ends)
            ):
                segment = _require_mapping(
                    segment_value, f"{location}.segments.{span_name}[{occurrence}]"
                )
                wave = _require_int(
                    segment.get("wave"),
                    f"{location}.segments.{span_name}[{occurrence}].wave",
                )
                expert = _require_int(
                    segment.get("expert"),
                    f"{location}.segments.{span_name}[{occurrence}].expert",
                )
                tasks.append([kind, wave, expert, begin, end])
        tasks.sort(key=lambda task: ((task[3] - process_begin) & cycle_mask, task[0]))
        compact_records.append([role, core, process_begin, work_end, phases, tasks])

    if actual_records != expected_records:
        missing = sorted(expected_records - actual_records)
        extra = sorted(actual_records - expected_records)
        raise CompactTraceError(
            f"record role/core set disagrees with role_counts; missing={missing}, extra={extra}"
        )
    compact_records.sort(key=lambda record: (record[1], ROLES.index(record[0])))

    compact: dict[str, Any] = {
        "format": COMPACT_FORMAT,
        "format_version": COMPACT_FORMAT_VERSION,
        "trace_name": normalized.get("trace_name"),
        "protocol": [protocol_version, schema_id, clock_bits],
        "config": config,
        "record_columns": [
            "role",
            "core",
            "process_begin",
            "work_end",
            "phases",
            "tasks",
        ],
        "phase_columns": ["kind", "begin", "end"],
        "phase_kinds": {
            "ib": "input_buffer_init",
            "iq": "input_quantize",
            "ri": "route_index",
            "wr": "workspace_reset",
            "ls": "local_prepare_sync",
            "rs": "cross_rank_sync",
            "eb": "expert_buffer_init",
        },
        "task_columns": ["kind", "wave", "expert", "begin", "end"],
        "task_kinds": {
            "gs": "gmm1_scalar_setup",
            "ds": "dispatch_scalar_gap",
            "gd": "gmm1_dispatch_wait",
            "gv": "gmm1_vector_wait",
            "g1": "gmm1_problem",
            "g2": "gmm2_problem",
            "s": "swiglu_problem",
            "d": "dispatch_task",
            "c": "combine_group_task",
        },
        "records": compact_records,
    }
    capture_kind = metadata.get("capture_kind")
    if isinstance(capture_kind, str):
        compact["capture_kind"] = capture_kind
    return compact


def _read_json(path: Path) -> tuple[dict[str, Any], int]:
    raw = path.read_bytes()
    payload = gzip.decompress(raw) if path.suffix == ".gz" else raw
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactTraceError(
            f"cannot decode normalized trace {path}: {exc}"
        ) from exc
    return _require_mapping(value, "$"), len(raw)


def pack_trace_file(
    input_path: Path, output_path: Path, *, lossless: bool = False, plain: bool = False
) -> dict[str, int]:
    normalized, input_bytes = _read_json(input_path)
    packed = normalized if lossless else compact_normalized_trace(normalized)
    payload = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    output = payload if plain else gzip.compress(payload, compresslevel=9, mtime=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    return {
        "input_bytes": input_bytes,
        "compact_json_bytes": len(payload),
        "output_bytes": len(output),
    }


def _default_output(input_path: Path, *, lossless: bool, plain: bool) -> Path:
    name = input_path.name
    for suffix in (".normalized.json.gz", ".normalized.json", ".json.gz", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    kind = "min" if lossless else "compact"
    extension = ".json" if plain else ".json.gz"
    return input_path.with_name(f"{name}.{kind}{extension}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strip a normalized MegaMoe trace to the minimum analysis contract and gzip it. "
            "The default output retains shape, Wave/core topology, and every sparse timestamp."
        )
    )
    parser.add_argument("input", type=Path, help="rankN.normalized.json or .json.gz")
    parser.add_argument("-o", "--output", type=Path, help="output path")
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="minify and gzip the complete normalized JSON instead of stripping metadata",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="write compact uncompressed JSON (gzip is the default)",
    )
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output(input_path, lossless=args.lossless, plain=args.plain)
    )
    try:
        stats = pack_trace_file(
            input_path,
            output_path,
            lossless=args.lossless,
            plain=args.plain,
        )
    except (CompactTraceError, OSError) as exc:
        parser.error(str(exc))
    ratio = (
        stats["input_bytes"] / stats["output_bytes"]
        if stats["output_bytes"] != 0
        else 0.0
    )
    print(
        f"[OK] packed {input_path} -> {output_path}; "
        f"input={stats['input_bytes']} bytes, compact-json={stats['compact_json_bytes']} bytes, "
        f"output={stats['output_bytes']} bytes, reduction={ratio:.1f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
