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
EXPECTED_COARSE_CAPTURE_CAPACITY = 4
TRACE_CONTRACTS = {
    "mega-moe-wave-critical-path": {
        "path_id": 1,
        "flags_by_protocol": {5: 0xFF, 6: 0x1FF, 7: 0x3FF},
        "experts_per_wave": "derived",
        "weight1_interleaved": "metadata",
    },
    "mega-moe-a8w4-critical-path": {
        "path_id": 2,
        "flags_by_protocol": {1: 0xF, 2: 0x1F},
        "experts_per_wave": 1,
        "weight1_interleaved": 0,
    },
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


def get_wave_expert_count(
    wave_begin: int, expert_count: int, experts_per_wave: int
) -> int:
    """Mirror arch35 GetWaveExpertCount, including small-tail absorption."""
    if wave_begin >= expert_count:
        return 0
    remaining = expert_count - wave_begin
    if experts_per_wave <= 0 or remaining <= experts_per_wave:
        return remaining
    tail = remaining - experts_per_wave
    return (
        remaining
        if tail < experts_per_wave and 2 * tail < experts_per_wave
        else experts_per_wave
    )


def count_expert_waves(expert_count: int, experts_per_wave: int) -> int:
    waves = 0
    begin = 0
    while begin < expert_count:
        count = get_wave_expert_count(begin, expert_count, experts_per_wave)
        if count <= 0:
            raise ContractError("cannot derive routed-wave count")
        begin += count
        waves += 1
    return waves


def resolve_record_words(
    protocol: dict[str, Any], header_or_expert_count: dict[str, int] | int
) -> int:
    layout = protocol["record_words"]
    if isinstance(layout, int) and not isinstance(layout, bool):
        return layout
    if not isinstance(layout, dict):
        raise ContractError("protocol record_words is neither fixed nor dynamic")
    header = (
        {"moe_experts_per_rank": header_or_expert_count}
        if isinstance(header_or_expert_count, int)
        else header_or_expert_count
    )
    terms = layout.get("terms", [layout])
    result = int(layout["base_words"])
    for term in terms:
        count_field = term.get("count_header_field")
        count = header.get(count_field)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ContractError(
                f"cannot resolve record_words from header field {count_field!r}: {count!r}"
            )
        result += int(term["words_per_count"]) * count
    return result


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
    metadata: dict[str, Any],
    raw_bytes: int,
    spec: dict[str, Any],
    runtime_aic_count: int | None = None,
) -> dict[str, int]:
    bs = _required_meta_int(metadata, "bs")
    h = _required_meta_int(metadata, "h")
    hidden_dim = _required_meta_int(metadata, "n")
    top_k = _required_meta_int(metadata, "k")
    moe_experts = _required_meta_int(metadata, "local_moe_expert_num")
    shared_experts = _required_meta_int(metadata, "shared_expert_num")
    world_size = _required_meta_int(metadata, "world_size")
    aic_count = (
        _required_meta_int(metadata, "aic_core_num")
        if runtime_aic_count is None
        else runtime_aic_count
    )
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
            f"dispatch tiling key {dispatch_key} is not supported by the traced MTE paths"
        )
    trace_name = spec["trace"]["name"]
    try:
        trace_contract = TRACE_CONTRACTS[trace_name]
    except KeyError as exc:
        raise ContractError(f"unsupported MegaMoe trace contract {trace_name!r}") from exc
    if trace_contract["experts_per_wave"] == "derived":
        experts_per_wave = calc_experts_per_wave(
            bs=bs,
            top_k=top_k,
            moe_experts_per_rank=moe_experts,
            h=h,
            hidden_dim=hidden_dim,
            aic_count=aic_count,
        )
        repetitions = count_expert_waves(moe_experts, experts_per_wave)
    else:
        experts_per_wave = int(trace_contract["experts_per_wave"])
        repetitions = moe_experts
    topk_prefetch = int(bool(metadata.get("topk_weights_type", False)))
    interleaved = (
        int(bool(metadata.get("is_interleaved", False)))
        if trace_contract["weight1_interleaved"] == "metadata"
        else int(trace_contract["weight1_interleaved"])
    )
    carrier_words = raw_bytes // 4
    protocol_version = int(spec["trace"]["protocol"]["version"])
    try:
        expected_flags = trace_contract["flags_by_protocol"][protocol_version]
    except KeyError as exc:
        raise ContractError(
            f"unsupported MegaMoe trace protocol version {protocol_version}"
        ) from exc
    values_for_hash = [
        trace_contract["path_id"],
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
    fine_dynamic = isinstance(spec["trace"]["protocol"]["record_words"], dict)
    expected = {
        "record_count": 3 * aic_count,
        "clock_bits": 32,
        "repetition_count": repetitions,
        "capture_capacity": moe_experts
        if fine_dynamic
        else EXPECTED_COARSE_CAPTURE_CAPACITY,
        "truncated": 0
        if fine_dynamic
        else int(repetitions > EXPECTED_COARSE_CAPTURE_CAPACITY),
        "flags": expected_flags,
        "rank": rank,
        "runtime_config_hash": config_hash(values_for_hash),
        "path_id": trace_contract["path_id"],
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
    if trace_name == "mega-moe-a8w4-critical-path" and protocol_version >= 2:
        max_output_size = _required_meta_int(metadata, "max_output_size")
        max_gmm2_tiles = ceil_div(max_output_size, 256) * ceil_div(h, 256)
        combine_tiles_per_expert_per_core = ceil_div(max_gmm2_tiles, aic_count)
        expected.update(
            {
                "max_output_size": max_output_size,
                "combine_tiles_per_expert_per_core": combine_tiles_per_expert_per_core,
                "combine_tile_capacity": moe_experts
                * combine_tiles_per_expert_per_core,
            }
        )
    return expected


def validate_capture_contract(
    *, raw_path: Path, metadata_path: Path, spec_path: Path, allow_truncated: bool
) -> tuple[dict[str, Any], dict[str, int]]:
    spec = load_json(spec_path)
    metadata = load_json(metadata_path)
    raw = raw_path.read_bytes()
    header = read_header(raw, spec)
    protocol = spec["trace"]["protocol"]
    role_counts = header.get("role_counts")
    if (
        not isinstance(role_counts, list)
        or len(role_counts) != 3
        or any(not isinstance(count, int) or count <= 0 for count in role_counts)
        or len(set(role_counts)) != 1
    ):
        raise ContractError(
            "runtime header has invalid AIC/AIV0/AIV1 role_counts: "
            f"{role_counts!r}"
        )
    # Tiling uses the platform-reported AIC count, while the adapted test script may
    # carry a larger configured count. The published device header is authoritative.
    runtime_aic_count = role_counts[0]
    expected = expected_header_from_metadata(
        metadata, len(raw), spec, runtime_aic_count=runtime_aic_count
    )
    protocol_expected = {
        "magic": parse_uint(protocol["magic"], "protocol magic"),
        "version": int(protocol["version"]),
        "schema_id": parse_uint(protocol["schema_id"], "protocol schema_id"),
        "header_words": int(protocol["header_words"]),
        "record_words": resolve_record_words(protocol, expected),
    }
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
    required_words = (
        int(protocol["header_words"])
        + expected["record_count"] * expected["record_words"]
    )
    fine_tail = protocol.get("fine_metadata_tail")
    if fine_tail is not None:
        required_words += expected["aic_count"] * int(fine_tail["words_per_core"])
    if required_words > expected["carrier_words"]:
        raise ContractError(
            f"trace requires {required_words * 4} bytes but y carries only {len(raw)} bytes"
        )
    if isinstance(protocol["record_words"], dict):
        preflight = metadata.get("capacity_preflight")
        if not isinstance(preflight, dict):
            raise ContractError("runtime metadata lacks adapter capacity_preflight")
        preflight_aic_count = parse_uint(
            preflight.get("aic_core_num"), "capacity_preflight aic_core_num"
        )
        preflight_record_header = {
            "moe_experts_per_rank": expected["moe_experts_per_rank"]
        }
        if "max_output_size" in expected:
            preflight_tiles_per_expert = ceil_div(
                ceil_div(expected["max_output_size"], 256)
                * ceil_div(expected["h"], 256),
                preflight_aic_count,
            )
            preflight_record_header["combine_tile_capacity"] = (
                expected["moe_experts_per_rank"] * preflight_tiles_per_expert
            )
        preflight_record_words = resolve_record_words(
            protocol, preflight_record_header
        )
        preflight_required_words = (
            int(protocol["header_words"])
            + 3 * preflight_aic_count * preflight_record_words
        )
        wanted_preflight = {
            "protocol_version": int(protocol["version"]),
            "local_moe_expert_num": expected["moe_experts_per_rank"],
            "record_words": preflight_record_words,
            "required_y_bytes": preflight_required_words * 4,
            "available_y_bytes": len(raw),
        }
        if "max_output_size" in expected:
            wanted_preflight.update(
                {
                    "max_output_size": expected["max_output_size"],
                    "combine_tiles_per_expert_per_core": preflight_tiles_per_expert,
                    "combine_tile_capacity": preflight_record_header[
                        "combine_tile_capacity"
                    ],
                }
            )
        mismatched_preflight = [
            f"{name}={preflight.get(name)!r}, expected {wanted}"
            for name, wanted in wanted_preflight.items()
            if preflight.get(name) != wanted
        ]
        if mismatched_preflight:
            raise ContractError(
                "adapter capacity_preflight mismatch: "
                + "; ".join(mismatched_preflight)
            )
        if preflight_aic_count < expected["aic_count"]:
            raise ContractError(
                "adapter capacity_preflight underestimates runtime AIC count: "
                f"aic_core_num={preflight_aic_count}, runtime={expected['aic_count']}"
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
    dynamic_fine = isinstance(fine_visualization, dict)
    process_spans = [span for span in spans if span.get("stage") == "process" or span["id"].endswith("process")]
    if len(process_spans) != 1:
        raise ContractError("trace spec must define exactly one process span")
    process_span = process_spans[0]
    wave_starts: set[int] = set()
    if dynamic_fine:
        expert_count = int(header.get("moe_experts_per_rank", 0))
        experts_per_wave = int(header.get("experts_per_wave", 0))
        wave_begin = 0
        while wave_begin < expert_count:
            wave_starts.add(wave_begin)
            wave_begin += get_wave_expert_count(
                wave_begin, expert_count, experts_per_wave
            )
        if len(wave_starts) != repetitions:
            raise ContractError(
                "dynamic fine wave partition disagrees with repetition_count"
            )
    fine_point_ids = {
        point_id
        for span in spans
        if span["id"] in fine_gap_by_span
        for point_id in (span["begin"], span["end"])
    }
    variable_capacity_point_ids = {
        field["name"]
        for field in spec["trace"]["protocol"]["record_fields"]
        if isinstance(field.get("count"), dict)
        and field["count"].get("header_field")
        not in (None, "moe_experts_per_rank")
    }
    span_counts = {span["id"]: 0 for span in spans}
    span_durations: dict[str, list[int]] = {span["id"]: [] for span in spans}
    fine_truncated_records: list[str] = []

    for record in records:
        label = record.get("label", "unknown")
        timestamps = record.get("timestamps", {})
        process_begin = timestamps.get(process_span["begin"], [missing])[0]
        work_end = timestamps.get(process_span["end"], [missing])[0]
        if process_begin == missing or work_end == missing:
            raise ContractError(
                f"{label} lacks {process_span['begin']}/{process_span['end']}"
            )
        process_duration = _modular_delta(process_begin, work_end, bits)
        if process_duration >= half_range:
            raise ContractError(f"{label} has backward/ambiguous process interval")

        if dynamic_fine and record.get("role") == "AIV1":
            dispatch_begins = timestamps.get("dispatch_begin_S", [])
            dispatch_ends = timestamps.get("dispatch_end_S", [])
            combine_begins = timestamps.get("combine_begin_S", [])
            combine_ends = timestamps.get("combine_end_S", [])
            for expert in range(len(dispatch_begins)):
                dispatch_present = (
                    dispatch_begins[expert] != missing
                    and dispatch_ends[expert] != missing
                )
                if dispatch_present != (expert in wave_starts):
                    raise ContractError(
                        f"{label} dispatch slot {expert} disagrees with routed-wave starts "
                        f"{sorted(wave_starts)}"
                    )
                combine_present = (
                    combine_begins[expert] != missing or combine_ends[expert] != missing
                )
                if combine_present and expert not in wave_starts:
                    raise ContractError(
                        f"{label} combine slot {expert} is not a routed-wave start"
                    )

        if (
            spec["trace"]["name"] == "mega-moe-a8w4-critical-path"
            and record.get("role") == "AIV1"
            and "combine_tile_capacity" in header
        ):
            tile_capacity = int(header["combine_tile_capacity"])
            tiles_per_expert = int(header["combine_tiles_per_expert_per_core"])
            tile_points = {
                name: timestamps.get(name, [])
                for name in (
                    "a8w4_combine_tile_wait_begin_S",
                    "a8w4_combine_tile_wait_end_S",
                    "a8w4_combine_tile_begin_S",
                    "a8w4_combine_tile_end_S",
                )
            }
            if any(len(values) != tile_capacity for values in tile_points.values()):
                raise ContractError(
                    f"{label} Combine tile arrays do not match capacity {tile_capacity}"
                )
            expert_count = int(header["moe_experts_per_rank"])
            if tile_capacity != expert_count * tiles_per_expert:
                raise ContractError(
                    f"{label} Combine tile capacity does not match expert slot layout"
                )
            for expert in range(expert_count):
                seen_missing = False
                for tile_ordinal in range(tiles_per_expert):
                    slot = expert * tiles_per_expert + tile_ordinal
                    values = [point_values[slot] for point_values in tile_points.values()]
                    present = [value != missing for value in values]
                    if any(present) and not all(present):
                        raise ContractError(
                            f"{label} expert {expert} tile {tile_ordinal} has incomplete boundaries"
                        )
                    if not any(present):
                        seen_missing = True
                        continue
                    if seen_missing:
                        raise ContractError(
                            f"{label} expert {expert} tile slots are not contiguous"
                        )
                    if values[1] != values[2]:
                        raise ContractError(
                            f"{label} expert {expert} tile {tile_ordinal} wait end "
                            "does not equal work begin"
                        )

        for point, values in timestamps.items():
            if not isinstance(values, list):
                raise ContractError(f"{label} point {point} is not an occurrence list")
            for occurrence, value in enumerate(values):
                if value == missing:
                    continue
                from_process = _modular_delta(process_begin, value, bits)
                if from_process > process_duration:
                    raise ContractError(
                        f"{label} {point}[{occurrence}] lies outside the process span"
                    )
                if (
                    not dynamic_fine
                    and point not in fine_point_ids
                    and point not in variable_capacity_point_ids
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
                if dynamic_fine:
                    if len(begins) != int(header.get("moe_experts_per_rank", 0)):
                        raise ContractError(
                            f"{label} {span['id']} does not expose one slot per routed expert"
                        )
                    previous_begin_delta = -1
                    for occurrence, (begin, end) in enumerate(zip(begins, ends)):
                        if begin == missing or end == missing:
                            continue
                        begin_delta = _modular_delta(process_begin, begin, bits)
                        if begin_delta < previous_begin_delta:
                            raise ContractError(
                                f"{label} {span['id']} active expert indices are not time-monotonic"
                            )
                        previous_begin_delta = begin_delta
                else:
                    counters = record.get("counters", {})
                    if counters.get("fine_enabled") != 1:
                        raise ContractError(
                            f"{label} lacks enabled fine metadata for {span['id']}"
                        )
                    captured_count = counters.get(
                        gap_contract["captured_count_counter"]
                    )
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
                                f"{label} {span['id']}[{occurrence}] is missing inside "
                                f"captured_count={captured_count}"
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
                                f"{label} {span['id']}[{occurrence}] is populated beyond "
                                f"captured_count={captured_count}"
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

    if span_counts.get(process_span["id"], 0) != len(records):
        raise ContractError(
            f"validated {span_counts.get(process_span['id'], 0)} process spans for {len(records)} records"
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
            "dynamic fine records expose every routed expert index and active intervals are time-monotonic",
            "AIV1 Dispatch is present exactly at every derived routed-wave start and Combine cannot use non-start slots",
        ],
        "record_count": len(records),
        "runtime_waves": repetitions,
        "captured_waves": captured,
        "fine_truncated_records": fine_truncated_records,
        "span_summary": duration_summary,
    }
