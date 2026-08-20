#!/usr/bin/env python3
"""Parse MegaMoe set_cond traces and decode Dispatch/GMM1 samples."""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


SAMPLE_COUNT = 20
DISPATCH_MARKER = 910001
GMM1_MARKER = 910002

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


def decode_bfloat16(raw: int) -> float:
    """Decode one raw BF16 word using the host IEEE-754 float implementation."""
    bits = (raw & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", bits))[0]


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
            marker_kind = None
            if value == DISPATCH_MARKER:
                marker_kind = "dispatch"
            elif value == GMM1_MARKER:
                marker_kind = "gmm1"

            if marker_kind is not None:
                if current is not None:
                    groups.append(current)
                current = SampleGroup(
                    kind=marker_kind,
                    marker=value,
                    marker_line=get_log_line(text, physical_line),
                    values=[],
                )
                continue

            if current is None:
                continue

            current.values.append(value)
            if len(current.values) == SAMPLE_COUNT:
                groups.append(current)
                current = None

    if current is not None:
        groups.append(current)
    return groups


def print_group(path: Path, group_index: int, group: SampleGroup) -> None:
    decoder = decode_fp8_e4m3fn if group.kind == "dispatch" else decode_bfloat16
    raw_width = 2 if group.kind == "dispatch" else 4
    dtype = "FP8 E4M3FN" if group.kind == "dispatch" else "BF16"
    status = "complete" if len(group.values) == SAMPLE_COUNT else f"incomplete: {len(group.values)}/{SAMPLE_COUNT}"

    print(
        f"\n[{group_index}] {group.kind.upper()}  dtype={dtype}  "
        f"marker=0x{group.marker:x}  log_line={group.marker_line}  {status}"
    )
    print(f"source: {path}")
    print("index  raw     decoded")
    print("-----  ------  ------------")
    for index, raw in enumerate(group.values):
        decoded = decoder(raw)
        print(f"{index:>5}  0x{raw:0{raw_width}x}  {format_float(decoded)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MegaMoe set_cond groups marked by 910001 (Dispatch FP8) "
            "and 910002 (GMM1 BF16)."
        )
    )
    parser.add_argument("logs", nargs="+", type=Path, help="instr_log.dump or filtered log file")
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
            print_group(path, found, group)

    if found == 0:
        print("No MegaMoe set_cond marker groups found.", file=sys.stderr)
        return 1
    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
