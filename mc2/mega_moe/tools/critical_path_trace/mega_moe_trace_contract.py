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
"""Target-specific validation for the MegaMoeWave critical-path protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when a capture cannot be proven to match the trace contract."""


DISPATCH_API_TO_TILING_KEY = {23: 3, 24: 4, 296: 5}
EXPECTED_PATH_ID = 2
EXPECTED_FLAGS_BY_PROTOCOL = {13: 0x1FFF}
DISPATCH_GROUP_READY_FLAG = 1 << 13
TRACE_ROLE_COUNT = 3
TRACE_RECORD_WORDS = 26
TRACE_SEGMENT_WORDS = 4
TRACE_STAGE_COUNT = 5
TRACE_SCRATCH_HEADER_WORDS = 22
TRACE_SCRATCH_RESERVE_BYTES = 50 * 1024 * 1024
GMM_TILE_N = 256
GMM1_MIN_LOGICAL_TILES_PER_CORE = 4
FIXED_PRE_DISPATCH_POINTS_BY_ROLE = {
    "AIC": (
        "process_begin_S",
        "local_prepare_sync_begin_S",
        "local_prepare_sync_end_S",
        "cross_rank_sync_begin_S",
        "cross_rank_sync_end_S",
        "expert_buffer_init_begin_S",
        "expert_buffer_init_end_S",
    ),
    "AIV0": (
        "process_begin_S",
        "input_buffer_init_begin_S",
        "input_buffer_init_end_S",
        "input_quantize_begin_S",
        "input_quantize_end_S",
        "route_index_begin_S",
        "route_index_end_S",
        "workspace_reset_begin_S",
        "workspace_reset_end_S",
        "local_prepare_sync_begin_S",
        "local_prepare_sync_end_S",
        "cross_rank_sync_begin_S",
        "cross_rank_sync_end_S",
        "expert_buffer_init_begin_S",
        "expert_buffer_init_end_S",
    ),
    "AIV1": (
        "process_begin_S",
        "input_buffer_init_begin_S",
        "input_buffer_init_end_S",
        "input_quantize_begin_S",
        "input_quantize_end_S",
        "route_index_begin_S",
        "route_index_end_S",
        "workspace_reset_begin_S",
        "workspace_reset_end_S",
        "local_prepare_sync_begin_S",
        "local_prepare_sync_end_S",
        "cross_rank_sync_begin_S",
        "cross_rank_sync_end_S",
        "expert_buffer_init_begin_S",
        "expert_buffer_init_end_S",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_uint(value: int | str, name: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be an unsigned integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value, 0)
        except ValueError as exc:
            raise ContractError(f"{name} is not an integer: {value!r}") from exc
    else:
        raise ContractError(f"{name} must be an integer, got {type(value).__name__}")
    if parsed < 0 or parsed > 0xFFFFFFFF:
        raise ContractError(f"{name} is outside uint32: {parsed}")
    return parsed


def _decode_header_value(
    words: list[int], field: dict[str, Any], word_bits: int, endianness: str
) -> int | list[int]:
    start = int(field["word"])
    width = int(field["width_words"])
    selected = words[start : start + width]
    if field["name"] == "role_counts":
        return selected
    if width == 1:
        return selected[0]
    ordered = selected if endianness == "little" else list(reversed(selected))
    result = 0
    for index, word in enumerate(ordered):
        result |= word << (index * word_bits)
    return result


def _raw_words(raw: bytes) -> list[int]:
    if len(raw) % 4:
        raise ContractError(f"raw y byte count {len(raw)} is not uint32 aligned")
    return [
        int.from_bytes(raw[offset : offset + 4], "little")
        for offset in range(0, len(raw), 4)
    ]


def read_header(raw: bytes, spec: dict[str, Any]) -> dict[str, int | list[int]]:
    trace = spec["trace"]
    protocol = trace["protocol"]
    word_bits = int(protocol["word_bits"])
    if word_bits != 32 or protocol["endianness"] != "little":
        raise ContractError("MegaMoe validator requires little-endian uint32 words")
    header_words = int(protocol["header_words"])
    if len(raw) < header_words * 4:
        raise ContractError(
            f"raw y has {len(raw)} bytes; header requires {header_words * 4}"
        )
    words = _raw_words(raw)[:header_words]
    return {
        field["name"]: _decode_header_value(words, field, word_bits, "little")
        for field in protocol["header_fields"]
    }


def config_hash(values: list[int]) -> int:
    result = 2166136261
    for value in values:
        result = ((result ^ (value & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return result


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ContractError(f"invalid non-positive divisor {divisor}")
    return (value + divisor - 1) // divisor


def resolve_record_words(protocol: dict[str, Any], _capture_capacity: int) -> int:
    value = protocol["record_words"]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(
            "MegaMoe sparse protocol requires fixed positive record_words"
        )
    return value


def calc_m_groups_per_wave(*, h: int, hidden_dim: int, aic_count: int) -> int:
    """Mirror arch35 CalcMGroupsPerWave; drift must fail runtime validation."""
    if h == 0 or hidden_dim == 0 or aic_count == 0:
        return 1
    gmm1_tiles_per_m_group = ceil_div(hidden_dim, GMM_TILE_N)
    gmm2_tiles_per_m_group = ceil_div(h, GMM_TILE_N)
    gmm1_required_m_groups = ceil_div(
        aic_count * GMM1_MIN_LOGICAL_TILES_PER_CORE,
        gmm1_tiles_per_m_group,
    )
    gmm2_required_m_groups = ceil_div(aic_count, gmm2_tiles_per_m_group)
    return max(gmm1_required_m_groups, gmm2_required_m_groups)


def calc_max_m_groups(
    *, bs: int, ep_world_size: int, top_k: int, expert_count: int
) -> int:
    if expert_count == 0:
        return 0
    max_routed_rows = bs * ep_world_size * min(top_k, expert_count)
    first_m_groups = min(max_routed_rows, expert_count)
    return first_m_groups + (max_routed_rows - first_m_groups) // 256


def calc_wave_capacity(
    *,
    bs: int,
    ep_world_size: int,
    top_k: int,
    expert_count: int,
    m_groups_per_wave: int,
) -> int:
    if expert_count == 0 or m_groups_per_wave == 0:
        return 0
    max_waves = ceil_div(
        calc_max_m_groups(
            bs=bs,
            ep_world_size=ep_world_size,
            top_k=top_k,
            expert_count=expert_count,
        ),
        m_groups_per_wave,
    )
    return max(1, max_waves)


def calc_segment_capacity(
    *, bs: int, ep_world_size: int, top_k: int, expert_count: int, aic_count: int
) -> tuple[int, int, int]:
    required = max(
        expert_count,
        calc_max_m_groups(
            bs=bs,
            ep_world_size=ep_world_size,
            top_k=top_k,
            expert_count=expert_count,
        ),
    )
    record_count = TRACE_ROLE_COUNT * aic_count
    if record_count <= 0:
        return required, 0, 0
    reserve_words = TRACE_SCRATCH_RESERVE_BYTES // 4
    words_per_record = reserve_words // record_count
    limit = max(
        0,
        (words_per_record - TRACE_SCRATCH_HEADER_WORDS)
        // (TRACE_STAGE_COUNT * TRACE_SEGMENT_WORDS),
    )
    return required, limit, min(required, limit)


def _required_meta_int(metadata: dict[str, Any], name: str) -> int:
    if name not in metadata:
        raise ContractError(f"runtime metadata is missing {name!r}")
    return parse_uint(metadata[name], f"runtime metadata {name}")


def expected_header_from_metadata(
    metadata: dict[str, Any], raw_bytes: int, spec: dict[str, Any]
) -> dict[str, int]:
    bs = _required_meta_int(metadata, "bs")
    h = _required_meta_int(metadata, "h")
    hidden_dim = _required_meta_int(metadata, "n")
    top_k = _required_meta_int(metadata, "k")
    moe_experts = _required_meta_int(metadata, "local_moe_expert_num")
    shared_experts = _required_meta_int(metadata, "shared_expert_num")
    ep_world_size = _required_meta_int(metadata, "ep_world_size")
    aic_count = _required_meta_int(metadata, "aic_core_num")
    dispatch_api_type = _required_meta_int(metadata, "dispatch_quant_out_type")
    combine_quant_mode = _required_meta_int(metadata, "comm_quant_mode")
    rank = _required_meta_int(metadata, "rank")
    try:
        dispatch_key = DISPATCH_API_TO_TILING_KEY[dispatch_api_type]
    except KeyError as exc:
        raise ContractError(
            f"unsupported dispatch_quant_out_type={dispatch_api_type}; cannot prove tiling key"
        ) from exc
    if dispatch_key not in (3, 4):
        raise ContractError(
            f"dispatch tiling key {dispatch_key} is not the A8W8 MegaMoeWave path"
        )
    m_groups_per_wave = calc_m_groups_per_wave(
        h=h, hidden_dim=hidden_dim, aic_count=aic_count
    )
    wave_capacity = calc_wave_capacity(
        bs=bs,
        ep_world_size=ep_world_size,
        top_k=top_k,
        expert_count=moe_experts,
        m_groups_per_wave=m_groups_per_wave,
    )
    required_segments, scratch_capacity_limit, segment_capacity = calc_segment_capacity(
        bs=bs,
        ep_world_size=ep_world_size,
        top_k=top_k,
        expert_count=moe_experts,
        aic_count=aic_count,
    )
    topk_prefetch = int(bool(metadata.get("topk_weights_type", False)))
    interleaved = int(bool(metadata.get("is_interleaved", False)))
    protocol_version = int(spec["trace"]["protocol"]["version"])
    try:
        expected_flags = EXPECTED_FLAGS_BY_PROTOCOL[protocol_version]
    except KeyError as exc:
        raise ContractError(
            f"unsupported MegaMoe trace protocol version {protocol_version}"
        ) from exc
    dispatch_group_ready = _required_meta_int(metadata, "dispatch_group_ready")
    if dispatch_group_ready not in (0, 1):
        raise ContractError("runtime metadata dispatch_group_ready must be 0 or 1")
    expected_flags |= dispatch_group_ready * DISPATCH_GROUP_READY_FLAG
    values_for_hash = [
        EXPECTED_PATH_ID,
        bs,
        h,
        hidden_dim,
        top_k,
        moe_experts,
        shared_experts,
        m_groups_per_wave,
        ep_world_size,
        aic_count,
        dispatch_key,
        combine_quant_mode,
        topk_prefetch,
        interleaved,
    ]
    return {
        "record_count": TRACE_ROLE_COUNT * aic_count,
        "clock_bits": 32,
        "capture_capacity": segment_capacity,
        "flags": expected_flags,
        "rank": rank,
        "runtime_config_hash": config_hash(values_for_hash),
        "path_id": EXPECTED_PATH_ID,
        "bs": bs,
        "h": h,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "moe_experts_per_rank": moe_experts,
        "shared_expert_count": shared_experts,
        "m_groups_per_wave": m_groups_per_wave,
        "world_size": ep_world_size,
        "dispatch_quant_out_key": dispatch_key,
        "combine_quant_mode": combine_quant_mode,
        "topk_weights_prefetch": topk_prefetch,
        "weight1_interleaved": interleaved,
        "carrier_words": raw_bytes // 4,
        "aic_count": aic_count,
        "wave_capacity": wave_capacity,
        "required_segment_capacity": required_segments,
        "scratch_capacity_limit": scratch_capacity_limit,
    }


def _modular_delta(begin: int, end: int, bits: int = 32) -> int:
    return (end - begin) & ((1 << bits) - 1)


def _validate_sparse_wire_layout(
    *, words: list[int], header: dict[str, Any], spec: dict[str, Any]
) -> int:
    protocol = spec["trace"]["protocol"]
    sparse = protocol["sparse_segments"]
    header_words = int(protocol["header_words"])
    record_words = int(protocol["record_words"])
    record_count = int(header["record_count"])
    capture_capacity = int(header["capture_capacity"])
    carrier_words = int(header["carrier_words"])
    role_counts = header["role_counts"]
    role_by_record: list[str] = []
    for role, count in zip(spec["trace"]["roles"], role_counts):
        role_by_record.extend([role] * int(count))
    payload_cursor = header_words + record_count * record_words
    half_range = 1 << 31

    for record_index, role in enumerate(role_by_record):
        descriptor = header_words + record_index * record_words
        process_begin = words[descriptor]
        work_end = words[descriptor + 1]
        if process_begin == 0 or work_end == 0:
            raise ContractError(
                f"{role} record {record_index} lacks process boundaries"
            )
        process_duration = _modular_delta(process_begin, work_end)
        if process_duration >= half_range:
            raise ContractError(
                f"{role} record {record_index} has backward process span"
            )
        for stage in sparse["stages"]:
            offset = words[descriptor + int(stage["offset_record_word"])]
            count = words[descriptor + int(stage["count_record_word"])]
            stage_name = stage["name"]
            if count > capture_capacity:
                raise ContractError(
                    f"{role} record {record_index} {stage_name} count={count} exceeds "
                    f"capture_capacity={capture_capacity}"
                )
            if role not in stage["role_points"] and count != 0:
                raise ContractError(
                    f"{role} record {record_index} unexpectedly has {count} {stage_name} segments"
                )
            if offset != payload_cursor:
                raise ContractError(
                    f"{role} record {record_index} {stage_name} offset={offset}, "
                    f"expected compact cursor {payload_cursor}"
                )
            payload_end = offset + count * int(sparse["entry_words"])
            if payload_end > carrier_words or payload_end > len(words):
                raise ContractError(
                    f"{role} record {record_index} {stage_name} payload exceeds y"
                )
            previous_begin_delta = -1
            previous_end_delta = -1
            for occurrence in range(count):
                entry = offset + occurrence * int(sparse["entry_words"])
                wave = words[entry + int(sparse["entry_fields"]["wave"])]
                expert = words[entry + int(sparse["entry_fields"]["expert"])]
                begin = words[entry + int(sparse["entry_fields"]["begin"])]
                end = words[entry + int(sparse["entry_fields"]["end"])]
                if wave >= int(header["repetition_count"]):
                    raise ContractError(
                        f"{role} record {record_index} {stage_name}[{occurrence}] "
                        f"wave={wave} is outside runtime Wave count"
                    )
                if expert >= int(header["moe_experts_per_rank"]):
                    raise ContractError(
                        f"{role} record {record_index} {stage_name}[{occurrence}] "
                        f"expert={expert} is outside expert count"
                    )
                if begin == 0 or end == 0:
                    raise ContractError(
                        f"{role} record {record_index} {stage_name}[{occurrence}] "
                        "has a missing boundary"
                    )
                duration = _modular_delta(begin, end)
                begin_delta = _modular_delta(process_begin, begin)
                end_delta = begin_delta + duration
                if duration >= half_range or end_delta > process_duration:
                    raise ContractError(
                        f"{role} record {record_index} {stage_name}[{occurrence}] "
                        "is backward or outside process bounds"
                    )
                if (
                    begin_delta < previous_begin_delta
                    or begin_delta < previous_end_delta
                ):
                    raise ContractError(
                        f"{role} record {record_index} {stage_name} segments overlap "
                        "or are not time-monotonic"
                    )
                previous_begin_delta = begin_delta
                previous_end_delta = end_delta
            payload_cursor = payload_end
    return payload_cursor


def validate_capture_contract(
    *, raw_path: Path, metadata_path: Path, spec_path: Path, allow_truncated: bool
) -> tuple[dict[str, Any], dict[str, int]]:
    spec = load_json(spec_path)
    metadata = load_json(metadata_path)
    raw = raw_path.read_bytes()
    words = _raw_words(raw)
    header = read_header(raw, spec)
    protocol = spec["trace"]["protocol"]
    expected = expected_header_from_metadata(metadata, len(raw), spec)
    protocol_expected = {
        "magic": parse_uint(protocol["magic"], "protocol magic"),
        "version": int(protocol["version"]),
        "schema_id": parse_uint(protocol["schema_id"], "protocol schema_id"),
        "header_words": int(protocol["header_words"]),
        "record_words": TRACE_RECORD_WORDS,
    }
    expected.update(protocol_expected)

    ignored_runtime_fields = {
        "aic_count",
        "wave_capacity",
        "required_segment_capacity",
        "scratch_capacity_limit",
    }
    mismatches: list[str] = []
    for name, wanted in expected.items():
        if name in ignored_runtime_fields:
            continue
        actual = header.get(name)
        if not isinstance(actual, int) or actual != wanted:
            if name in {"magic", "schema_id"} and isinstance(actual, int):
                mismatches.append(
                    f"{name}={actual} (0x{actual:08x}), expected {wanted} "
                    f"(0x{wanted:08x})"
                )
            else:
                mismatches.append(f"{name}={actual!r}, expected {wanted}")
    wanted_role_counts = [expected["aic_count"]] * TRACE_ROLE_COUNT
    if header.get("role_counts") != wanted_role_counts:
        mismatches.append(
            f"role_counts={header.get('role_counts')!r}, expected {wanted_role_counts!r}"
        )
    if mismatches:
        raise ContractError("runtime header/config mismatch: " + "; ".join(mismatches))

    repetition_count = header.get("repetition_count")
    if (
        not isinstance(repetition_count, int)
        or repetition_count <= 0
        or repetition_count > expected["wave_capacity"]
    ):
        raise ContractError(
            f"runtime repetition_count={repetition_count!r}, expected "
            f"1..{expected['wave_capacity']} token-group Waves"
        )
    expected["repetition_count"] = repetition_count
    truncated = header.get("truncated")
    if truncated not in (0, 1):
        raise ContractError(f"runtime truncated={truncated!r} is not boolean")
    if truncated and not allow_truncated:
        raise ContractError(
            "runtime sparse trace is truncated by scratch or y capacity; rerun a smaller "
            "case instead of treating the partial capture as complete"
        )

    required_words = _validate_sparse_wire_layout(words=words, header=header, spec=spec)
    preflight = metadata.get("capacity_preflight")
    if not isinstance(preflight, dict):
        raise ContractError("runtime metadata lacks adapter capacity_preflight")
    scratch_record_words = (
        TRACE_SCRATCH_HEADER_WORDS
        + TRACE_STAGE_COUNT * TRACE_SEGMENT_WORDS * expected["capture_capacity"]
    )
    scratch_required_bytes = 4 * expected["record_count"] * scratch_record_words
    descriptor_required_bytes = 4 * (
        int(protocol["header_words"]) + expected["record_count"] * TRACE_RECORD_WORDS
    )
    wanted_preflight = {
        "protocol_version": int(protocol["version"]),
        "traced_ai_blocks": expected["aic_count"],
        "aic_core_num": expected["aic_count"],
        "local_moe_expert_num": expected["moe_experts_per_rank"],
        "m_groups_per_wave": expected["m_groups_per_wave"],
        "wave_capacity": expected["wave_capacity"],
        "required_segment_capacity": expected["required_segment_capacity"],
        "scratch_capacity_limit": expected["scratch_capacity_limit"],
        "segment_capacity": expected["capture_capacity"],
        "record_words": TRACE_RECORD_WORDS,
        "record_count": expected["record_count"],
        "descriptor_required_y_bytes": descriptor_required_bytes,
        "available_y_bytes": len(raw),
        "scratch_record_words": scratch_record_words,
        "scratch_required_bytes": scratch_required_bytes,
        "scratch_reserve_bytes": TRACE_SCRATCH_RESERVE_BYTES,
    }
    mismatched_preflight = [
        f"{name}={preflight.get(name)!r}, expected {wanted}"
        for name, wanted in wanted_preflight.items()
        if preflight.get(name) != wanted
    ]
    if mismatched_preflight:
        raise ContractError(
            "adapter capacity_preflight mismatch: " + "; ".join(mismatched_preflight)
        )
    if expected["required_segment_capacity"] > expected["scratch_capacity_limit"]:
        raise ContractError(
            "shape-derived full segment capacity does not fit the reserved trace scratch"
        )
    if metadata.get("output_dtype") != "torch.bfloat16":
        raise ContractError(
            f"trace carrier dtype must be torch.bfloat16, got {metadata.get('output_dtype')!r}"
        )
    if metadata.get("output_shape") != [expected["bs"], expected["h"]]:
        raise ContractError(
            f"captured y shape {metadata.get('output_shape')!r} does not match "
            f"[{expected['bs']}, {expected['h']}]"
        )
    if metadata.get("output_nbytes") != len(raw):
        raise ContractError(
            f"metadata output_nbytes={metadata.get('output_nbytes')!r} but raw has {len(raw)} bytes"
        )
    actual_raw_sha256 = sha256_bytes(raw)
    if metadata.get("raw_sha256") != actual_raw_sha256:
        raise ContractError(
            f"metadata raw_sha256={metadata.get('raw_sha256')!r} but raw is "
            f"{actual_raw_sha256}"
        )
    return (
        {
            "header": header,
            "metadata": metadata,
            "raw_sha256": actual_raw_sha256,
            "required_trace_bytes": required_words * 4,
            "available_carrier_bytes": len(raw),
        },
        expected,
    )


def validate_timestamp_order(
    normalized_path: Path, spec_path: Path, allow_truncated: bool = False
) -> dict[str, Any]:
    normalized = load_json(normalized_path)
    spec = load_json(spec_path)
    records = normalized.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("normalized trace has no records")
    header = normalized.get("metadata", {}).get("header", {})
    role_counts = header.get("role_counts")
    roles = spec["trace"]["roles"]
    if not isinstance(role_counts, list) or len(role_counts) != len(roles):
        raise ContractError("normalized header has invalid role_counts")
    expected_records = {
        (role, core)
        for role, count in zip(roles, role_counts)
        for core in range(int(count))
    }
    actual_records = {(record.get("role"), record.get("core")) for record in records}
    if actual_records != expected_records or len(records) != len(expected_records):
        raise ContractError(
            "capture must contain AIC/AIV0/AIV1 for every physical AI Block"
        )
    if header.get("truncated") and not allow_truncated:
        raise ContractError("normalized trace is truncated")

    bits = int(spec["trace"]["clock"]["bits"])
    missing = int(spec["trace"]["clock"]["missing_value"])
    half_range = 1 << (bits - 1)
    spans = spec["spans"]
    span_counts = {span["id"]: 0 for span in spans}
    span_durations: dict[str, list[int]] = {span["id"]: [] for span in spans}
    runtime_waves = int(header["repetition_count"])
    expert_count = int(header["moe_experts_per_rank"])
    segment_capacity = int(header["capture_capacity"])
    sparse_span_ids = {
        mapping["span"]
        for stage in spec["trace"]["protocol"]["sparse_segments"]["stages"]
        for mapping in stage["role_points"].values()
    }

    for record in records:
        label = record.get("label", "unknown")
        timestamps = record.get("timestamps", {})
        process_begin = timestamps.get("process_begin_S", [missing])[0]
        work_end = timestamps.get("work_end_S", [missing])[0]
        if process_begin == missing or work_end == missing:
            raise ContractError(f"{label} lacks process_begin_S/work_end_S")
        process_duration = _modular_delta(process_begin, work_end, bits)
        if process_duration >= half_range:
            raise ContractError(f"{label} has backward/ambiguous process interval")

        previous_delta = -1
        for point_name in FIXED_PRE_DISPATCH_POINTS_BY_ROLE[record["role"]]:
            values = timestamps.get(point_name, [])
            if len(values) != 1 or values[0] == missing:
                raise ContractError(f"{label} lacks fixed boundary {point_name}")
            point_delta = _modular_delta(process_begin, values[0], bits)
            if point_delta < previous_delta or point_delta > process_duration:
                raise ContractError(
                    f"{label} fixed pre-Dispatch boundaries are backward at {point_name}"
                )
            previous_delta = point_delta

        for span in spans:
            if record.get("role") not in span["roles"]:
                continue
            begins = timestamps.get(span["begin"], [])
            ends = timestamps.get(span["end"], [])
            if len(begins) != len(ends):
                raise ContractError(f"{label} {span['id']} begin/end counts differ")
            if span["id"] != "process" and len(begins) > segment_capacity:
                raise ContractError(f"{label} {span['id']} exceeds segment capacity")
            metadata = record.get("segments", {}).get(span["id"], [])
            if span["id"] in sparse_span_ids and len(metadata) != len(begins):
                raise ContractError(
                    f"{label} {span['id']} metadata/boundary counts differ"
                )
            previous_begin_delta = -1
            previous_end_delta = -1
            for occurrence, (begin, end) in enumerate(zip(begins, ends)):
                if begin == missing or end == missing:
                    raise ContractError(
                        f"{label} {span['id']}[{occurrence}] has a missing boundary"
                    )
                duration = _modular_delta(begin, end, bits)
                begin_delta = _modular_delta(process_begin, begin, bits)
                end_delta = begin_delta + duration
                if duration >= half_range or end_delta > process_duration:
                    raise ContractError(
                        f"{label} {span['id']}[{occurrence}] is backward or outside process"
                    )
                if (
                    begin_delta < previous_begin_delta
                    or begin_delta < previous_end_delta
                ):
                    raise ContractError(
                        f"{label} {span['id']} segments overlap or are not time-monotonic"
                    )
                previous_begin_delta = begin_delta
                previous_end_delta = end_delta
                if span["id"] in sparse_span_ids:
                    wave = int(metadata[occurrence]["wave"])
                    expert = int(metadata[occurrence]["expert"])
                    if wave >= runtime_waves or expert >= expert_count:
                        raise ContractError(
                            f"{label} {span['id']}[{occurrence}] metadata is out of bounds"
                        )
                span_counts[span["id"]] += 1
                span_durations[span["id"]].append(duration)

        if record["role"] == "AIV1":
            dispatch_begins = timestamps.get("dispatch_task_begin_S", [])
            dispatch_ends = timestamps.get("dispatch_task_end_S", [])
            dispatch_metadata = record.get("segments", {}).get("dispatch_task", [])
            gap_begins = timestamps.get("dispatch_scalar_gap_begin_S", [])
            gap_ends = timestamps.get("dispatch_scalar_gap_end_S", [])
            gap_metadata = record.get("segments", {}).get("dispatch_scalar_gap", [])
            task_intervals = [
                (
                    _modular_delta(process_begin, begin, bits),
                    _modular_delta(process_begin, begin, bits)
                    + _modular_delta(begin, end, bits),
                )
                for begin, end in zip(dispatch_begins, dispatch_ends)
            ]
            matched_next_tasks: set[int] = set()
            for gap_index, (gap_begin, gap_end) in enumerate(zip(gap_begins, gap_ends)):
                gap_begin_delta = _modular_delta(process_begin, gap_begin, bits)
                gap_end_delta = gap_begin_delta + _modular_delta(
                    gap_begin, gap_end, bits
                )
                next_task_index = next(
                    (
                        task_index
                        for task_index in range(1, len(task_intervals))
                        if (
                            task_intervals[task_index - 1][1] <= gap_begin_delta
                            and gap_end_delta <= task_intervals[task_index][0]
                        )
                    ),
                    None,
                )
                if next_task_index is None or next_task_index in matched_next_tasks:
                    raise ContractError(
                        f"{label} dispatch_scalar_gap[{gap_index}] is not uniquely "
                        "contained between consecutive Dispatch tasks"
                    )
                matched_next_tasks.add(next_task_index)
                if gap_metadata[gap_index] != dispatch_metadata[next_task_index]:
                    raise ContractError(
                        f"{label} dispatch_scalar_gap[{gap_index}] metadata does not "
                        "identify its following Dispatch task"
                    )

    if span_counts["process"] != len(records):
        raise ContractError("not every role record has one process span")
    duration_summary = {
        name: {
            "count": len(values),
            "min_cycles": min(values) if values else None,
            "max_cycles": max(values) if values else None,
        }
        for name, values in span_durations.items()
    }
    return {
        "status": "passed",
        "checks": [
            "AIC/AIV0/AIV1 records exist for every physical AI Block",
            "each AI Block is normalized independently while its three roles remain directly comparable",
            "all role-appropriate fixed pre-Dispatch boundaries exist and are time-monotonic",
            "every sparse segment has paired forward boundaries inside process_begin..work_end",
            "segment metadata wave/expert indices are in bounds",
            "segments on one role/stage are non-overlapping and time-monotonic",
            "every Dispatch Scalar gap is contained between consecutive AIV1 tasks and identifies the following task",
            "no inferred idle interval is converted into a colored work span",
            "GMM1/GMM2 exact spans use the protocol-v13 tracing-only FIX completion contract",
        ],
        "record_count": len(records),
        "runtime_waves": runtime_waves,
        "segment_capacity": segment_capacity,
        "fine_truncated_records": [],
        "span_summary": duration_summary,
    }
