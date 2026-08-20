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
            "Extract MegaMoe set_cond pipeline checkpoints 910001 through 910005."
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
