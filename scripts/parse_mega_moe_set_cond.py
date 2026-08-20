#!/usr/bin/env python3
"""Parse and decode MegaMoe set_cond pipeline checkpoints."""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


SAMPLE_COUNT = 20
SCALE_SAMPLE_COUNT = 8
GMM2_DEBUG_WINDOW_COUNT = 4
GMM2_DEBUG_UINT_BIAS = 1_000_000
GMM2_DEBUG_HASH_BIAS = 0x1_0000_0000

COND_VALUE_RE = re.compile(
    r"MOV_SPR_XN\s+SPR:COND,\s+XN:[^=]+=(?P<value>0x[0-9a-fA-F]+)"
)
LOG_LINE_RE = re.compile(r":(?P<line>\d+):\[info\]")


@dataclass
class SampleGroup:
    kind: str
    marker: int
    marker_line: int
    values: list[int]


@dataclass(frozen=True)
class SectionSpec:
    name: str
    count: int
    dtype: str


@dataclass(frozen=True)
class MarkerSpec:
    name: str
    sections: tuple[SectionSpec, ...]

    @property
    def value_count(self) -> int:
        return sum(section.count for section in self.sections)


MARKER_SPECS = {
    910001: MarkerSpec("dispatch", (SectionSpec("dispatch_data", SAMPLE_COUNT, "fp8"),)),
    910002: MarkerSpec(
        "gmm1",
        (
            SectionSpec("gate", SAMPLE_COUNT, "bf16"),
            SectionSpec("up", SAMPLE_COUNT, "bf16"),
        ),
    ),
    910003: MarkerSpec(
        "activation",
        (
            SectionSpec("swiglu", SAMPLE_COUNT, "bf16"),
            SectionSpec("quant_data", SAMPLE_COUNT, "fp8"),
            SectionSpec("quant_scale", 8, "e8m0"),
        ),
    ),
    910004: MarkerSpec("gmm2", (SectionSpec("gmm2_output", SAMPLE_COUNT, "bf16"),)),
    910005: MarkerSpec(
        "combine_receive",
        (
            SectionSpec("combine_mode", 1, "mode"),
            SectionSpec("combine_data", SAMPLE_COUNT, "combine"),
        ),
    ),
    910006: MarkerSpec(
        "gmm2_input_a",
        (
            SectionSpec("physical_block_idx", 1, "debug_uint"),
            SectionSpec("m_n_k_scale_k", 4, "debug_uint"),
            SectionSpec("k_offsets", GMM2_DEBUG_WINDOW_COUNT, "debug_uint"),
            SectionSpec("scale_offsets", GMM2_DEBUG_WINDOW_COUNT, "debug_uint"),
            *(
                SectionSpec(f"activation_data_window_{window_idx}", SAMPLE_COUNT, "fp8")
                for window_idx in range(GMM2_DEBUG_WINDOW_COUNT)
            ),
            *(
                SectionSpec(f"activation_scale_window_{window_idx}", SCALE_SAMPLE_COUNT, "e8m0")
                for window_idx in range(GMM2_DEBUG_WINDOW_COUNT)
            ),
        ),
    ),
    910007: MarkerSpec(
        "gmm2_input_b",
        (
            SectionSpec("physical_block_idx", 1, "debug_uint"),
            SectionSpec("is_weight_nz", 1, "debug_uint"),
            SectionSpec("raw_weight_offsets", GMM2_DEBUG_WINDOW_COUNT, "debug_uint"),
            SectionSpec("raw_scale_offsets", GMM2_DEBUG_WINDOW_COUNT, "debug_uint"),
            *(
                SectionSpec(f"weight2_data_window_{window_idx}", SAMPLE_COUNT, "fp8")
                for window_idx in range(GMM2_DEBUG_WINDOW_COUNT)
            ),
            *(
                SectionSpec(f"weight2_scale_window_{window_idx}", SCALE_SAMPLE_COUNT, "e8m0")
                for window_idx in range(GMM2_DEBUG_WINDOW_COUNT)
            ),
        ),
    ),
    910008: MarkerSpec(
        "gmm1_aic_tile_config",
        (
            SectionSpec("physical_block_idx", 1, "debug_uint"),
            SectionSpec("logical_block_count", 1, "debug_uint"),
            SectionSpec("loop_idx_start_loop_idx_tile_count", 3, "debug_uint"),
            SectionSpec("tile_coord_m_n_k", 3, "debug_uint"),
            SectionSpec("actual_shape_m_n_k", 3, "debug_uint"),
            SectionSpec("problem_m_n_k", 3, "debug_uint"),
            SectionSpec("output_n_scheduler_n_scale_k", 3, "debug_uint"),
            SectionSpec("scheduler_tile_m_activation_tile_m", 2, "debug_uint"),
            SectionSpec("requested_block_mmad_tile_m_n", 2, "debug_uint"),
            SectionSpec("requested_k_l1_scale_k_l1", 2, "debug_uint"),
            SectionSpec("requested_k_l1_scale_l1_l0_loop_counts", 3, "debug_uint"),
            SectionSpec("requested_scale_window_elements", 1, "debug_uint"),
            SectionSpec("requested_l1_window_half_l1_bytes", 2, "debug_uint"),
            SectionSpec("is_weight_nz", 1, "debug_uint"),
            SectionSpec("gm_tile_sequence_before", 1, "debug_uint"),
            SectionSpec("effective_problem_k_k_l1_scale_k_l1", 3, "debug_uint"),
            SectionSpec("effective_base_m_n_k", 3, "debug_uint"),
            SectionSpec("block_mmad_indices_before_l1_scale_l1_l0", 3, "debug_uint"),
            SectionSpec("current_l1_offsets_a_b_scale_a_scale_b", 4, "debug_uint"),
        ),
    ),
    910009: MarkerSpec(
        "gmm1_aiv_tile_mapping",
        (
            SectionSpec("physical_block_idx", 1, "debug_uint"),
            SectionSpec("logical_block_count", 1, "debug_uint"),
            SectionSpec("loop_idx_start_loop_idx_tile_count", 3, "debug_uint"),
            SectionSpec("tile_coord_m_n_k", 3, "debug_uint"),
            SectionSpec("actual_shape_m_n_k", 3, "debug_uint"),
            SectionSpec("expected_observed_sequence", 2, "debug_uint"),
        ),
    ),
    910010: MarkerSpec(
        "gmm2_aic_tile_config",
        (
            SectionSpec("physical_block_idx", 1, "debug_uint"),
            SectionSpec("logical_block_count", 1, "debug_uint"),
            SectionSpec("loop_idx_start_loop_idx_tile_count", 3, "debug_uint"),
            SectionSpec("tile_coord_m_n_k", 3, "debug_uint"),
            SectionSpec("actual_shape_m_n_k", 3, "debug_uint"),
            SectionSpec("problem_m_n_k", 3, "debug_uint"),
            SectionSpec("output_n_scheduler_n_scale_k", 3, "debug_uint"),
            SectionSpec("scheduler_tile_m_activation_tile_m", 2, "debug_uint"),
            SectionSpec("requested_block_mmad_tile_m_n", 2, "debug_uint"),
            SectionSpec("requested_k_l1_scale_k_l1", 2, "debug_uint"),
            SectionSpec("requested_k_l1_scale_l1_l0_loop_counts", 3, "debug_uint"),
            SectionSpec("requested_scale_window_elements", 1, "debug_uint"),
            SectionSpec("requested_l1_window_half_l1_bytes", 2, "debug_uint"),
            SectionSpec("is_weight_nz", 1, "debug_uint"),
            SectionSpec("gm_tile_sequence_before", 1, "debug_uint"),
            SectionSpec("effective_problem_k_k_l1_scale_k_l1", 3, "debug_uint"),
            SectionSpec("effective_base_m_n_k", 3, "debug_uint"),
            SectionSpec("block_mmad_indices_before_l1_scale_l1_l0", 3, "debug_uint"),
            SectionSpec("current_l1_offsets_a_b_scale_a_scale_b", 4, "debug_uint"),
        ),
    ),
    910011: MarkerSpec(
        "gmm2_aiv_tile_mapping",
        (
            SectionSpec("physical_block_idx", 1, "debug_uint"),
            SectionSpec("logical_block_count", 1, "debug_uint"),
            SectionSpec("loop_idx_start_loop_idx_tile_count", 3, "debug_uint"),
            SectionSpec("tile_coord_m_n_k", 3, "debug_uint"),
            SectionSpec("actual_shape_m_n_k", 3, "debug_uint"),
            SectionSpec("expected_observed_sequence", 2, "debug_uint"),
        ),
    ),
    910012: MarkerSpec(
        "gmm2_complete_input_hashes",
        (
            SectionSpec("physical_block_idx", 1, "debug_uint"),
            SectionSpec("is_weight_nz", 1, "debug_uint"),
            SectionSpec("k_scale_k", 2, "debug_uint"),
            SectionSpec("activation_data_hash", 1, "debug_hash"),
            SectionSpec("activation_scale_hash", 1, "debug_hash"),
            SectionSpec("weight2_data_row_hashes", 20, "debug_hash"),
            SectionSpec("weight2_scale_row_hashes", 20, "debug_hash"),
        ),
    ),
    910013: MarkerSpec(
        "gmm1_block_mmad_state_after_tile",
        (
            SectionSpec("physical_block_idx_loop_idx", 2, "debug_uint"),
            SectionSpec("tile_coord_m_n_k", 3, "debug_uint"),
            SectionSpec("effective_problem_k_k_l1_scale_k_l1", 3, "debug_uint"),
            SectionSpec("effective_base_m_n_k", 3, "debug_uint"),
            SectionSpec("block_mmad_indices_after_l1_scale_l1_l0", 3, "debug_uint"),
            SectionSpec("current_l1_offsets_a_b_scale_a_scale_b", 4, "debug_uint"),
        ),
    ),
    910014: MarkerSpec(
        "gmm2_block_mmad_state_after_tile",
        (
            SectionSpec("physical_block_idx_loop_idx", 2, "debug_uint"),
            SectionSpec("tile_coord_m_n_k", 3, "debug_uint"),
            SectionSpec("effective_problem_k_k_l1_scale_k_l1", 3, "debug_uint"),
            SectionSpec("effective_base_m_n_k", 3, "debug_uint"),
            SectionSpec("block_mmad_indices_after_l1_scale_l1_l0", 3, "debug_uint"),
            SectionSpec("current_l1_offsets_a_b_scale_a_scale_b", 4, "debug_uint"),
        ),
    ),
}


def decode_fp8_e4m3fn(raw: int) -> float:
    """Decode one raw FP8 E4M3FN byte."""
    raw &= 0xFF
    sign = -1.0 if raw & 0x80 else 1.0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7

    if exponent == 0:
        if mantissa == 0:
            return math.copysign(0.0, sign)
        return sign * math.ldexp(float(mantissa), -9)
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    return sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def decode_fp8_e5m2(raw: int) -> float:
    """Decode one raw FP8 E5M2 byte."""
    raw &= 0xFF
    sign = -1.0 if raw & 0x80 else 1.0
    exponent = (raw >> 2) & 0x1F
    mantissa = raw & 0x3

    if exponent == 0:
        if mantissa == 0:
            return math.copysign(0.0, sign)
        return sign * math.ldexp(float(mantissa), -16)
    if exponent == 0x1F:
        return math.copysign(math.inf, sign) if mantissa == 0 else math.nan
    return sign * math.ldexp(1.0 + mantissa / 4.0, exponent - 15)


def decode_bfloat16(raw: int) -> float:
    """Decode one raw BF16 word using the host IEEE-754 float implementation."""
    bits = (raw & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def decode_e8m0(raw: int) -> float:
    raw &= 0xFF
    if raw == 0xFF:
        return math.nan
    return math.ldexp(1.0, raw - 127)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0"
    return f"{value:.9g}"


def get_log_line(text: str, fallback: int) -> int:
    match = LOG_LINE_RE.search(text)
    return int(match.group("line")) if match else fallback


def parse_file(path: Path) -> list[SampleGroup]:
    groups: list[SampleGroup] = []
    current: SampleGroup | None = None

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for physical_line, text in enumerate(stream, start=1):
            match = COND_VALUE_RE.search(text)
            if not match:
                continue

            value = int(match.group("value"), 16)
            marker_spec = MARKER_SPECS.get(value)
            if marker_spec is not None:
                if current is not None:
                    groups.append(current)
                current = SampleGroup(
                    kind=marker_spec.name,
                    marker=value,
                    marker_line=get_log_line(text, physical_line),
                    values=[],
                )
                continue

            if current is None:
                continue

            current.values.append(value)
            if len(current.values) == MARKER_SPECS[current.marker].value_count:
                groups.append(current)
                current = None

    if current is not None:
        groups.append(current)
    return groups


def decode_value(raw: int, dtype: str, fp8_format: str, combine_mode: int | None) -> tuple[int, float | str]:
    if dtype == "debug_uint":
        return 8, str(raw - GMM2_DEBUG_UINT_BIAS)
    if dtype == "debug_hash":
        return 9, f"0x{raw - GMM2_DEBUG_HASH_BIAS:08x}"
    if dtype == "bf16":
        return 4, decode_bfloat16(raw)
    if dtype == "e8m0":
        return 2, decode_e8m0(raw)
    if dtype == "mode":
        return 1, "BF16" if raw == 0 else "quantized FP8"
    if dtype == "combine":
        dtype = "bf16" if combine_mode == 0 else "fp8"
        return decode_value(raw, dtype, fp8_format, combine_mode)
    decoder = decode_fp8_e4m3fn if fp8_format == "e4m3fn" else decode_fp8_e5m2
    return 2, decoder(raw)


def print_group(path: Path, group_index: int, group: SampleGroup, fp8_format: str) -> None:
    marker_spec = MARKER_SPECS[group.marker]
    expected_count = marker_spec.value_count
    status = (
        "complete"
        if len(group.values) == expected_count
        else f"incomplete: {len(group.values)}/{expected_count}"
    )

    print(
        f"\n[{group_index}] {group.kind.upper()}  marker=0x{group.marker:x}  "
        f"log_line={group.marker_line}  {status}"
    )
    print(f"source: {path}")
    value_offset = 0
    combine_mode = group.values[0] if group.kind == "combine_receive" and group.values else None
    for section in marker_spec.sections:
        section_values = group.values[value_offset : value_offset + section.count]
        value_offset += section.count
        if not section_values:
            continue
        print(f"\n{section.name}:")
        print("index  raw     decoded")
        print("-----  ------  ------------")
        for index, raw in enumerate(section_values):
            raw_width, decoded = decode_value(raw, section.dtype, fp8_format, combine_mode)
            decoded_text = decoded if isinstance(decoded, str) else format_float(decoded)
            print(f"{index:>5}  0x{raw:0{raw_width}x}  {decoded_text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MegaMoe set_cond pipeline checkpoints 910001 through 910014."
        )
    )
    parser.add_argument("logs", nargs="+", type=Path, help="instr_log.dump or filtered log file")
    parser.add_argument(
        "--fp8-format",
        choices=("e4m3fn", "e5m2"),
        default="e4m3fn",
        help="FP8 format used by Dispatch and activation quantization (default: e4m3fn)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    found = 0
    had_error = False

    for path in args.logs:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            had_error = True
            continue

        groups = parse_file(path)
        for group in groups:
            found += 1
            print_group(path, found, group, args.fp8_format)

    if found == 0:
        print("No MegaMoe set_cond marker groups found.", file=sys.stderr)
        return 1
    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
