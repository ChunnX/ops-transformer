#!/usr/bin/env python3
"""Target-specific validation for the MegaMoeWave critical-path protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when a capture cannot be proven to match the trace contract."""


DISPATCH_API_TO_TILING_KEY = {23: 3, 24: 4, 296: 5}
EXPECTED_PATH_ID = 1
EXPECTED_CAPTURE_CAPACITY = 4
EXPECTED_FLAGS_BY_PROTOCOL = {5: 0xFF, 6: 0x1FF}


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


def read_header(raw: bytes, spec: dict[str, Any]) -> dict[str, int | list[int]]:
    trace = spec["trace"]
    protocol = trace["protocol"]
    word_bits = int(protocol["word_bits"])
    word_bytes = word_bits // 8
    if word_bits != 32 or protocol["endianness"] != "little":
        raise ContractError(
            "MegaMoe target validator requires little-endian uint32 words"
        )
    if len(raw) % word_bytes:
        raise ContractError(
            f"raw y byte count {len(raw)} is not aligned to the {word_bytes}-byte trace word"
        )
    header_words = int(protocol["header_words"])
    if len(raw) < header_words * word_bytes:
        raise ContractError(
            f"raw y has {len(raw)} bytes; header requires {header_words * word_bytes}"
        )
    words = [
        int.from_bytes(raw[offset : offset + word_bytes], "little")
        for offset in range(0, header_words * word_bytes, word_bytes)
    ]
    return {
        field["name"]: _decode_header_value(
            words, field, word_bits, protocol["endianness"]
        )
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


def calc_experts_per_wave(
    *,
    bs: int,
    top_k: int,
    moe_experts_per_rank: int,
    h: int,
    hidden_dim: int,
    aic_count: int,
) -> int:
    """Mirror arch35 CalcExpertsPerBatch; drift here must fail runtime validation."""
    if moe_experts_per_rank == 0 or h == 0 or aic_count == 0:
        return 1
    expected_tokens = ceil_div(bs * top_k, moe_experts_per_rank)
    expected_m_waves = max(1, ceil_div(expected_tokens, 256))
    gmm1_tiles = expected_m_waves * ceil_div(hidden_dim, 512)
    gmm2_tiles = ceil_div(expected_tokens, 256) * ceil_div(h, 256)
    limiting_tiles = max(1, min(gmm1_tiles, gmm2_tiles))
    experts_to_fill_aic = ceil_div(aic_count, limiting_tiles)
    experts_per_wave = max(1, min(moe_experts_per_rank, experts_to_fill_aic))
    experts_per_wave *= ceil_div(hidden_dim, h)
    return max(1, min(moe_experts_per_rank, experts_per_wave))


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
    world_size = _required_meta_int(metadata, "world_size")
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
    experts_per_wave = calc_experts_per_wave(
        bs=bs,
        top_k=top_k,
        moe_experts_per_rank=moe_experts,
        h=h,
        hidden_dim=hidden_dim,
        aic_count=aic_count,
    )
    repetitions = ceil_div(moe_experts, experts_per_wave) if moe_experts else 0
    topk_prefetch = int(bool(metadata.get("topk_weights_type", False)))
    interleaved = int(bool(metadata.get("is_interleaved", False)))
    carrier_words = raw_bytes // 4
    protocol_version = int(spec["trace"]["protocol"]["version"])
    try:
        expected_flags = EXPECTED_FLAGS_BY_PROTOCOL[protocol_version]
    except KeyError as exc:
        raise ContractError(
            f"unsupported MegaMoe trace protocol version {protocol_version}"
        ) from exc
    values_for_hash = [
        EXPECTED_PATH_ID,
        bs,
        h,
        hidden_dim,
        top_k,
        moe_experts,
        shared_experts,
        experts_per_wave,
        world_size,
        aic_count,
        dispatch_key,
        combine_quant_mode,
        topk_prefetch,
        interleaved,
    ]
    return {
        "record_count": 3 * aic_count,
        "clock_bits": 32,
        "repetition_count": repetitions,
        "capture_capacity": EXPECTED_CAPTURE_CAPACITY,
        "truncated": int(repetitions > EXPECTED_CAPTURE_CAPACITY),
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
        "experts_per_wave": experts_per_wave,
        "world_size": world_size,
        "dispatch_quant_out_key": dispatch_key,
        "combine_quant_mode": combine_quant_mode,
        "topk_weights_prefetch": topk_prefetch,
        "weight1_interleaved": interleaved,
        "carrier_words": carrier_words,
        "aic_count": aic_count,
    }


def validate_capture_contract(
    *, raw_path: Path, metadata_path: Path, spec_path: Path, allow_truncated: bool
) -> tuple[dict[str, Any], dict[str, int]]:
    spec = load_json(spec_path)
    metadata = load_json(metadata_path)
    raw = raw_path.read_bytes()
    header = read_header(raw, spec)
    protocol = spec["trace"]["protocol"]
    protocol_expected = {
        "magic": parse_uint(protocol["magic"], "protocol magic"),
        "version": int(protocol["version"]),
        "schema_id": parse_uint(protocol["schema_id"], "protocol schema_id"),
        "header_words": int(protocol["header_words"]),
        "record_words": int(protocol["record_words"]),
    }
    expected = expected_header_from_metadata(metadata, len(raw), spec)
    expected.update(protocol_expected)

    mismatches: list[str] = []
    for name, wanted in expected.items():
        if name == "aic_count":
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
    role_counts = header.get("role_counts")
    wanted_role_counts = [expected["aic_count"]] * 3
    if role_counts != wanted_role_counts:
        mismatches.append(
            f"role_counts={role_counts!r}, expected {wanted_role_counts!r}"
        )
    if mismatches:
        raise ContractError("runtime header/config mismatch: " + "; ".join(mismatches))

    if expected["truncated"] and not allow_truncated:
        raise ContractError(
            f"capture has {expected['repetition_count']} waves but capacity is "
            f"{expected['capture_capacity']}; pass --allow-truncated only for intentional partial analysis"
        )
    required_words = int(protocol["header_words"]) + expected["record_count"] * int(
        protocol["record_words"]
    )
    fine_tail = protocol.get("fine_metadata_tail")
    if fine_tail is not None:
        required_words += expected["aic_count"] * int(fine_tail["words_per_core"])
    if required_words > expected["carrier_words"]:
        raise ContractError(
            f"trace requires {required_words * 4} bytes but y carries only {len(raw)} bytes"
        )
    if metadata.get("output_dtype") != "torch.bfloat16":
        raise ContractError(
            f"trace carrier dtype must be torch.bfloat16, got {metadata.get('output_dtype')!r}"
        )
    output_shape = metadata.get("output_shape")
    if output_shape != [expected["bs"], expected["h"]]:
        raise ContractError(
            f"captured y shape {output_shape!r} does not match bs*h "
            f"[{expected['bs']}, {expected['h']}]"
        )
    if metadata.get("output_nbytes") != len(raw):
        raise ContractError(
            f"metadata output_nbytes={metadata.get('output_nbytes')!r} but raw has {len(raw)} bytes"
        )
    actual_raw_sha256 = sha256_bytes(raw)
    if metadata.get("raw_sha256") != actual_raw_sha256:
        raise ContractError(
            f"metadata raw_sha256={metadata.get('raw_sha256')!r} but captured raw is "
            f"{actual_raw_sha256}"
        )
    result = {
        "header": header,
        "metadata": metadata,
        "raw_sha256": actual_raw_sha256,
        "required_trace_bytes": required_words * 4,
        "available_carrier_bytes": len(raw),
    }
    return result, expected


def _modular_delta(begin: int, end: int, bits: int) -> int:
    return (end - begin) & ((1 << bits) - 1)


def validate_timestamp_order(
    normalized_path: Path, spec_path: Path, allow_truncated: bool = False
) -> dict[str, Any]:
    normalized = load_json(normalized_path)
    spec = load_json(spec_path)
    records = normalized.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("normalized trace has no records")
    clock = spec["trace"]["clock"]
    bits = int(clock["bits"])
    missing = int(clock["missing_value"])
    half_range = 1 << (bits - 1)
    header = normalized.get("metadata", {}).get("header", {})
    repetitions = int(header.get("repetition_count", 0))
    capacity = int(header.get("capture_capacity", 0))
    captured = min(repetitions, capacity)
    spans = spec["trace"].get("spans", spec.get("spans", []))
    fine_visualization = spec["trace"].get("visualization", {}).get("fine_grained")
    fine_gap_by_span = {
        gap["span"]: gap
        for gap in (
            fine_visualization.get("derived_idle_gaps", [])
            if isinstance(fine_visualization, dict)
            else []
        )
    }
    fine_point_ids = {
        point_id
        for span in spans
        if span["id"] in fine_gap_by_span
        for point_id in (span["begin"], span["end"])
    }
    span_counts = {span["id"]: 0 for span in spans}
    span_durations: dict[str, list[int]] = {span["id"]: [] for span in spans}
    fine_truncated_records: list[str] = []

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

        for point, values in timestamps.items():
            if not isinstance(values, list):
                raise ContractError(f"{label} point {point} is not an occurrence list")
            for occurrence, value in enumerate(values):
                if value == missing:
                    continue
                from_process = _modular_delta(process_begin, value, bits)
                if from_process > process_duration:
                    raise ContractError(
                        f"{label} {point}[{occurrence}] lies outside process_begin_S..work_end_S"
                    )
                if (
                    point not in fine_point_ids
                    and len(values) > 1
                    and occurrence >= captured
                ):
                    raise ContractError(
                        f"{label} {point}[{occurrence}] is populated beyond captured wave count {captured}"
                    )

        for span in spans:
            if record.get("role") not in span["roles"]:
                continue
            begins = timestamps.get(span["begin"], [])
            ends = timestamps.get(span["end"], [])
            if len(begins) != len(ends):
                raise ContractError(
                    f"{label} {span['id']} begin/end occurrence counts differ"
                )
            gap_contract = fine_gap_by_span.get(span["id"])
            if gap_contract is not None:
                counters = record.get("counters", {})
                if counters.get("fine_enabled") != 1:
                    raise ContractError(
                        f"{label} lacks enabled fine metadata for {span['id']}"
                    )
                captured_count = counters.get(gap_contract["captured_count_counter"])
                if (
                    not isinstance(captured_count, int)
                    or isinstance(captured_count, bool)
                    or captured_count < 0
                    or captured_count > len(begins)
                ):
                    raise ContractError(
                        f"{label} has invalid captured count for {span['id']}: {captured_count!r}"
                    )
                truncated_name = gap_contract["truncated_counter"]
                fine_truncated = counters.get(truncated_name)
                if fine_truncated not in (0, 1):
                    raise ContractError(
                        f"{label} has invalid {truncated_name}={fine_truncated!r}"
                    )
                if fine_truncated:
                    fine_truncated_records.append(f"{label}/{span['id']}")
                    if not allow_truncated:
                        raise ContractError(
                            f"{label} {span['id']} exceeded its {len(begins)} active-expert slots; "
                            "pass --allow-truncated only for intentional partial analysis"
                        )
                wave_prefix = gap_contract["wave_counter_prefix"]
                previous_wave = -1
                for occurrence in range(captured_count):
                    if begins[occurrence] == missing or ends[occurrence] == missing:
                        raise ContractError(
                            f"{label} {span['id']}[{occurrence}] is missing inside captured_count={captured_count}"
                        )
                    wave_id = counters.get(f"{wave_prefix}_{occurrence}")
                    if (
                        not isinstance(wave_id, int)
                        or isinstance(wave_id, bool)
                        or wave_id < 0
                        or wave_id >= captured
                    ):
                        raise ContractError(
                            f"{label} {span['id']}[{occurrence}] has invalid captured wave id {wave_id!r}"
                        )
                    if wave_id < previous_wave:
                        raise ContractError(
                            f"{label} {span['id']} fine slot wave ids are not monotonic"
                        )
                    previous_wave = wave_id
                for occurrence in range(captured_count, len(begins)):
                    if begins[occurrence] != missing or ends[occurrence] != missing:
                        raise ContractError(
                            f"{label} {span['id']}[{occurrence}] is populated beyond captured_count={captured_count}"
                        )
            for occurrence, (begin, end) in enumerate(zip(begins, ends)):
                if (begin == missing) != (end == missing):
                    raise ContractError(
                        f"{label} {span['id']}[{occurrence}] has only one boundary"
                    )
                if begin == missing:
                    continue
                duration = _modular_delta(begin, end, bits)
                if duration >= half_range:
                    raise ContractError(
                        f"{label} {span['id']}[{occurrence}] is backward/ambiguous"
                    )
                span_counts[span["id"]] += 1
                span_durations[span["id"]].append(duration)

    if span_counts.get("process", 0) != len(records):
        raise ContractError(
            f"validated {span_counts.get('process', 0)} process spans for {len(records)} records"
        )
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
            "all record-local process boundaries are present and forward within the 32-bit half-range",
            "every populated stage has a paired begin/end and a forward duration",
            "every populated timestamp lies inside its role-local useful-work interval",
            "uncaptured wave/expert slots retain the zero missing sentinel",
            "fine expert counts, wave ids, monotonic slot order, and per-record truncation metadata agree",
        ],
        "record_count": len(records),
        "runtime_waves": repetitions,
        "captured_waves": captured,
        "fine_truncated_records": fine_truncated_records,
        "span_summary": duration_summary,
    }
