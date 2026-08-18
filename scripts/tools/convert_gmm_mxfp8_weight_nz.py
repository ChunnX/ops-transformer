#!/usr/bin/env python3
"""Convert the weight shape in a GMM MXFP8 CSV from ND to FRACTAL_NZ."""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def extract_bool_property(value: str, name: str) -> bool:
    """Extract one boolean property without parsing the surrounding dictionary."""
    match = re.search(
        rf"(?<![A-Za-z0-9_])['\"]?{re.escape(name)}['\"]?\s*:\s*"
        r"(?:['\"]\s*)?(true|false)(?:\s*['\"])?",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        preview = value.strip()
        if len(preview) > 160:
            preview = f"{preview[:157]}..."
        raise ValueError(
            f"cannot find boolean property {name!r} in "
            f"other_compilation_params: {preview!r}"
        )
    return match.group(1).lower() == "true"


def to_weight_nz_shape(
    nd_shape: list[Any], transpose_weight: bool, shape_convention: str
) -> list[int]:
    if len(nd_shape) != 3:
        raise ValueError(f"weight shape must be three-dimensional, got {nd_shape!r}")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in nd_shape):
        raise ValueError(f"weight dimensions must be positive integers, got {nd_shape!r}")

    group, first, second = nd_shape
    if shape_convention == "api-order":
        # API order is [G,K,N] for non-transpose and [G,N,K] for transpose.
        return [group, ceil_div(second, 32), ceil_div(first, 16), 16, 32]

    # canonical-kn means the CSV always stores [G,K,N].  When weight is
    # transposed, the NZ outer axes therefore use K/32 and N/16.
    k_value, n_value = first, second
    if transpose_weight:
        return [group, ceil_div(k_value, 32), ceil_div(n_value, 16), 16, 32]
    return [group, ceil_div(n_value, 32), ceil_div(k_value, 16), 16, 32]


def parse_stc_inputs(value: str) -> list[Any]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"stc_inputs is not a valid Python list: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"stc_inputs must be a list, got {type(parsed).__name__}")
    return parsed


def detect_encoding(path: Path) -> str:
    with path.open("rb") as file:
        return "utf-8-sig" if file.read(3) == b"\xef\xbb\xbf" else "utf-8"


def find_stc_inputs_span(line: str) -> tuple[int, int]:
    """Find the first complete [[...]] expression in the original CSV row."""
    start = line.find("[[")
    if start < 0:
        raise ValueError("cannot find stc_inputs list starting with '[['")
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(line)):
        char = line[index]
        if escaped:
            escaped = False
            continue
        if quote:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError("stc_inputs list has no matching closing bracket")


def convert(args: argparse.Namespace) -> tuple[int, int, list[str]]:
    input_path = args.input.resolve()
    encoding = detect_encoding(input_path)
    with input_path.open("r", encoding=encoding, newline="") as file:
        lines = file.readlines()
    if not lines:
        raise ValueError("CSV is empty")
    fieldnames = next(csv.reader([lines[0]], delimiter=args.delimiter))
    if args.stc_inputs_column not in fieldnames:
        raise ValueError(f"column {args.stc_inputs_column!r} does not exist")
    if args.compilation_params_column not in fieldnames:
        raise ValueError(f"column {args.compilation_params_column!r} does not exist")

    changes: list[str] = []
    output_lines = [lines[0]]
    converted_count = 0
    removed_count = 0
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            output_lines.append(line)
            continue
        try:
            # Read transpose properties from the complete row. This remains
            # reliable even when list/dict fields containing commas were not
            # quoted correctly and a conventional CSV parser shifts columns.
            transpose_x = extract_bool_property(line, "transpose_x")
            if transpose_x:
                removed_count += 1
                changes.append(f"line {line_number}: removed because transpose_x=true")
                continue
            transpose_weight = extract_bool_property(line, "transpose_weight")

            stc_start, stc_end = find_stc_inputs_span(line)
            inputs = parse_stc_inputs(line[stc_start:stc_end])
            if args.weight_index >= len(inputs):
                raise ValueError(
                    f"weight index {args.weight_index} is outside stc_inputs (size {len(inputs)})"
                )
            old_shape = inputs[args.weight_index]
            if isinstance(old_shape, list) and len(old_shape) == 5:
                if args.already_converted == "skip":
                    output_lines.append(line)
                    continue
                raise ValueError(f"weight shape is already five-dimensional: {old_shape!r}")
            new_shape = to_weight_nz_shape(
                old_shape, transpose_weight, args.shape_convention
            )
            inputs[args.weight_index] = new_shape
            output_lines.append(f"{line[:stc_start]}{inputs!r}{line[stc_end:]}")
            converted_count += 1
            changes.append(
                f"line {line_number}: transWeight={str(transpose_weight).lower()} "
                f"{old_shape} -> {new_shape}"
            )
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc

    if args.dry_run:
        return converted_count, removed_count, changes

    output_path = input_path if args.in_place else args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding=encoding, newline="") as file:
            file.writelines(output_lines)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return converted_count, removed_count, changes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert stc_inputs[1] from GMM MXFP8 weight ND shape to FRACTAL_NZ."
    )
    parser.add_argument("input", type=Path, help="source CSV file")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("-o", "--output", type=Path, help="output CSV file")
    destination.add_argument("--in-place", action="store_true", help="replace the source CSV")
    parser.add_argument("--stc-inputs-column", default="stc_inputs")
    parser.add_argument("--compilation-params-column", default="other_compilation_params")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter, default: comma")
    parser.add_argument("--weight-index", type=int, default=1)
    parser.add_argument(
        "--shape-convention",
        choices=("canonical-kn", "api-order"),
        default="api-order",
        help=(
            "canonical-kn: source is always [G,K,N]; api-order: source is "
            "[G,K,N] or [G,N,K] according to transWeight"
        ),
    )
    parser.add_argument("--already-converted", choices=("skip", "error"), default="skip")
    parser.add_argument("--dry-run", action="store_true", help="show changes without writing")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.in_place and args.output is None and not args.dry_run:
        args.output = args.input.with_name(f"{args.input.stem}_weight_nz{args.input.suffix}")
    try:
        converted_count, removed_count, changes = convert(args)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    for change in changes:
        print(change)
    if args.dry_run:
        print(
            f"dry run: {converted_count} row(s) would be converted, "
            f"{removed_count} transpose_x row(s) would be removed"
        )
    else:
        target = args.input if args.in_place else args.output
        print(
            f"converted {converted_count} row(s), removed {removed_count} "
            f"transpose_x row(s), output: {target}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
