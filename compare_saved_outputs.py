#!/usr/bin/env python3
"""Compare saved cascade and MegaMoe text outputs against one Golden."""

import argparse
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np


FLOAT_TOKEN = r"[+-]?(?:nan|inf(?:inity)?|(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))"
SMALL_ROW_RE = re.compile(
    rf"^\s*\|?\s*(\d+)\s+({FLOAT_TOKEN})\s+({FLOAT_TOKEN})\s+"
    rf"({FLOAT_TOKEN})\s+({FLOAT_TOKEN})\s*\|?\s*$",
    re.IGNORECASE,
)
MEGA_ROW_RE = SMALL_ROW_RE


@dataclass
class SavedTable:
    indexes: np.ndarray
    golden: np.ndarray
    actual: np.ndarray
    duplicate_actual: Optional[np.ndarray] = None
    actual_decimal_places: Optional[np.ndarray] = None


def is_separator(line):
    stripped = line.strip()
    return not stripped or (set(stripped) <= {"-", "=", "_"})


def is_small_header(line):
    lowered = line.lower()
    return "idx" in lowered and "golden" in lowered and "actual" in lowered


def is_mega_header(line):
    lowered = line.lower()
    return (
        "loop" in lowered
        and "cascade_npu_out" in lowered
        and "golden" in lowered
        and "npu_out" in lowered
    )


def parse_sections(path, header_predicate, row_builder):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    sections = []
    rows = None

    for line in lines:
        if header_predicate(line):
            if rows:
                sections.append(rows)
            rows = []
            continue

        if rows is None:
            continue

        match = SMALL_ROW_RE.match(line)
        if match:
            rows.append(row_builder(match.groups()))
            continue

        if is_separator(line):
            continue

        if rows:
            sections.append(rows)
        rows = None

    if rows:
        sections.append(rows)
    return sections


def parse_small_sections(path):
    def build(groups):
        index, golden, actual, _, _ = groups
        return int(index), float(golden), float(actual), None, None

    return parse_sections(path, is_small_header, build)


def parse_mega_sections(path, index_base):
    def build(groups):
        loop, cascade_output, golden, npu_output, _ = groups
        decimal_value = Decimal(npu_output)
        decimal_places = None
        if decimal_value.is_finite():
            decimal_places = max(0, -int(decimal_value.as_tuple().exponent))
        return (
            int(loop) - index_base,
            float(golden),
            float(npu_output),
            float(cascade_output),
            decimal_places,
        )

    return parse_sections(path, is_mega_header, build)


def select_table(sections, section_index, path, kind):
    if not sections:
        raise ValueError(f"No {kind} result table found in {path}")
    if section_index < 0 or section_index >= len(sections):
        raise ValueError(
            f"{path} contains {len(sections)} {kind} table(s); "
            f"section {section_index} is out of range"
        )

    rows = sections[section_index]
    indexes = np.asarray([row[0] for row in rows], dtype=np.int64)
    if len(np.unique(indexes)) != len(indexes):
        raise ValueError(f"Duplicate indexes found in {kind} section {section_index}")

    duplicate_values = [row[3] for row in rows]
    duplicate_actual = None
    if all(value is not None for value in duplicate_values):
        duplicate_actual = np.asarray(duplicate_values, dtype=np.float32)

    decimal_place_values = [row[4] for row in rows]
    actual_decimal_places = None
    if all(value is not None for value in decimal_place_values):
        actual_decimal_places = np.asarray(decimal_place_values, dtype=np.int32)

    return SavedTable(
        indexes=indexes,
        golden=np.asarray([row[1] for row in rows], dtype=np.float32),
        actual=np.asarray([row[2] for row in rows], dtype=np.float32),
        duplicate_actual=duplicate_actual,
        actual_decimal_places=actual_decimal_places,
    )


def align_tables(small, mega):
    small_positions = {int(index): position for position, index in enumerate(small.indexes)}
    mega_positions = {int(index): position for position, index in enumerate(mega.indexes)}
    common_indexes = sorted(set(small_positions) & set(mega_positions))
    if not common_indexes:
        raise ValueError("The two files have no common logical indexes")

    small_pos = np.asarray([small_positions[index] for index in common_indexes])
    mega_pos = np.asarray([mega_positions[index] for index in common_indexes])
    aligned = {
        "indexes": np.asarray(common_indexes, dtype=np.int64),
        "small_golden": small.golden[small_pos],
        "small_actual": small.actual[small_pos],
        "mega_golden": mega.golden[mega_pos],
        "mega_actual": mega.actual[mega_pos],
    }
    if mega.duplicate_actual is not None:
        aligned["mega_duplicate_actual"] = mega.duplicate_actual[mega_pos]
    if mega.actual_decimal_places is not None:
        aligned["mega_actual_decimal_places"] = mega.actual_decimal_places[mega_pos]
    return aligned


def finite_max(values):
    finite_values = values[np.isfinite(values)]
    return float(np.max(finite_values)) if finite_values.size else math.nan


def filter_by_golden_difference(aligned, threshold):
    if threshold <= 0:
        raise ValueError("Golden difference threshold must be greater than zero")

    small_golden = aligned["small_golden"]
    mega_golden = aligned["mega_golden"]
    abs_diff = np.abs(small_golden - mega_golden)
    trusted = np.isfinite(abs_diff) & (abs_diff < threshold)
    trusted |= np.isnan(small_golden) & np.isnan(mega_golden)
    trusted |= np.isposinf(small_golden) & np.isposinf(mega_golden)
    trusted |= np.isneginf(small_golden) & np.isneginf(mega_golden)

    total_count = int(trusted.size)
    trusted_count = int(np.count_nonzero(trusted))
    excluded_count = total_count - trusted_count
    if trusted_count == 0:
        raise ValueError(
            f"No element has Golden absolute difference strictly below {threshold:.8g}"
        )

    trusted_finite_diff = abs_diff[trusted & np.isfinite(abs_diff)]
    max_trusted_diff = finite_max(trusted_finite_diff)
    print(
        "[INFO] Golden trust filter: "
        f"abs_diff < {threshold:.8g}, kept={trusted_count}/{total_count} "
        f"({trusted_count / total_count:.4%}), excluded={excluded_count}, "
        f"max_kept_abs_diff={max_trusted_diff:.8g}"
    )

    if excluded_count:
        excluded_positions = np.flatnonzero(~trusted)
        print(
            "[WARN] Excluded Golden-inconsistent elements: "
            f"count={excluded_count}, max_abs_diff={finite_max(abs_diff[~trusted]):.8g}"
        )
        for position in excluded_positions[:10]:
            print(
                "  index={} small_golden={:.8g} mega_golden={:.8g} abs_diff={:.8g}".format(
                    int(aligned["indexes"][position]),
                    float(small_golden[position]),
                    float(mega_golden[position]),
                    float(abs_diff[position]),
                )
            )

    filtered = {}
    for name, values in aligned.items():
        if isinstance(values, np.ndarray) and values.shape[0] == total_count:
            filtered[name] = values[trusted]
        else:
            filtered[name] = values
    return filtered, total_count, excluded_count


def validate_mega_duplicate_columns(aligned):
    duplicate = aligned.get("mega_duplicate_actual")
    if duplicate is None:
        return
    matched = np.isclose(duplicate, aligned["mega_actual"], rtol=0.0, atol=0.0, equal_nan=True)
    mismatch_count = int(np.count_nonzero(~matched))
    if mismatch_count:
        max_abs_diff = finite_max(np.abs(duplicate - aligned["mega_actual"]))
        print(
            "[WARN] Mega cascade_npu_out and npu_out differ at "
            f"{mismatch_count}/{matched.size} positions; using npu_out. "
            f"max_abs_diff={max_abs_diff:.8g}"
        )
    else:
        print("[INFO] Mega cascade_npu_out and npu_out are identical.")


def match_small_precision_to_mega(aligned):
    decimal_places = aligned.get("mega_actual_decimal_places")
    if decimal_places is None:
        raise ValueError("Unable to determine the printed precision of Mega npu_out")

    original = aligned["small_actual"].copy()
    normalized = np.asarray(
        [
            float(f"{float(value):.{int(places)}f}")
            if np.isfinite(value)
            else float(value)
            for value, places in zip(original, decimal_places)
        ],
        dtype=np.float32,
    )
    aligned["small_actual_original"] = original
    aligned["small_actual"] = normalized

    unchanged = np.isclose(original, normalized, rtol=0.0, atol=0.0, equal_nan=True)
    changed_count = int(np.count_nonzero(~unchanged))
    max_adjustment = finite_max(np.abs(original - normalized))
    unique_places = sorted(set(int(value) for value in decimal_places.tolist()))
    print(
        "[INFO] Small-op actual values matched to Mega printed precision: "
        f"decimal_places={unique_places}, changed={changed_count}/{len(original)}, "
        f"max_adjustment={max_adjustment:.8g}"
    )


def write_report(
    path, aligned, golden, result, reason, better, equal, worse,
    source_elements, excluded_elements, golden_diff_threshold,
):
    small_abs = np.abs(aligned["small_actual"] - golden)
    mega_abs = np.abs(aligned["mega_actual"] - golden)
    small_original = aligned.get("small_actual_original", aligned["small_actual"])
    with path.open("w", encoding="utf-8") as output:
        output.write(f"three_way_result: {result}\n")
        output.write(f"reason: {reason.strip() if reason.strip() else 'none'}\n")
        output.write(f"source_elements: {source_elements}\n")
        output.write(f"compared_elements: {len(golden)}\n")
        output.write(f"excluded_golden_mismatch: {excluded_elements}\n")
        output.write(f"golden_diff_rule: abs_diff < {golden_diff_threshold:.9e}\n")
        output.write(f"mega_better: {better}\n")
        output.write(f"mega_equal: {equal}\n")
        output.write(f"mega_worse: {worse}\n\n")
        output.write(
            "idx golden small_actual_original small_actual_matched mega_actual "
            "small_abs_diff mega_abs_diff mega_vs_small\n"
        )
        for position, index in enumerate(aligned["indexes"]):
            if mega_abs[position] < small_abs[position]:
                relation = "better"
            elif mega_abs[position] > small_abs[position]:
                relation = "worse"
            else:
                relation = "equal"
            output.write(
                f"{int(index):8d} {golden[position]: .9e} "
                f"{small_original[position]: .9e} "
                f"{aligned['small_actual'][position]: .9e} "
                f"{aligned['mega_actual'][position]: .9e} "
                f"{small_abs[position]: .9e} {mega_abs[position]: .9e} {relation}\n"
            )


def compare_results(args):
    try:
        import torch
        import toolss_pro
    except ImportError as error:
        raise RuntimeError(
            "The comparison stage requires the test environment's torch and toolss_pro dependencies"
        ) from error

    small_sections = parse_small_sections(args.small_txt)
    mega_sections = parse_mega_sections(args.mega_txt, args.mega_index_base)
    print(
        f"[INFO] Found {len(small_sections)} small-op table(s) and "
        f"{len(mega_sections)} MegaMoe table(s)."
    )
    small = select_table(small_sections, args.small_section, args.small_txt, "small-op")
    mega = select_table(mega_sections, args.mega_section, args.mega_txt, "MegaMoe")
    aligned = align_tables(small, mega)

    coverage = len(aligned["indexes"]) / max(len(small.indexes), 1)
    print(
        f"[INFO] Aligned {len(aligned['indexes'])} element(s); "
        f"small={len(small.indexes)}, mega={len(mega.indexes)}, "
        f"coverage_vs_small={coverage:.2%}."
    )
    if len(aligned["indexes"]) != len(small.indexes) or len(aligned["indexes"]) != len(mega.indexes):
        if not args.allow_partial:
            raise ValueError(
                "Incomplete index coverage. The saved Mega table may contain only the "
                "first/last displayed rows; pass --allow-partial only for local diagnostics."
            )
        print("[WARN] Only common indexes will participate in this partial comparison.")

    validate_mega_duplicate_columns(aligned)
    aligned, source_elements, excluded_elements = filter_by_golden_difference(
        aligned, args.golden_diff_threshold
    )
    match_small_precision_to_mega(aligned)

    golden = (
        aligned["small_golden"]
        if args.golden_source == "small"
        else aligned["mega_golden"]
    )
    toolss_pro.data_compare_triangle(
        aligned["small_actual"],
        golden,
        aligned["mega_actual"],
        diff_thd=args.diff_threshold,
    )

    small_result = toolss_pro.checkResult(
        torch.from_numpy(aligned["small_actual"].copy()),
        torch.from_numpy(golden.copy()),
        "small_vs_golden",
    )
    mega_result = toolss_pro.checkResult(
        torch.from_numpy(aligned["mega_actual"].copy()),
        torch.from_numpy(golden.copy()),
        "mega_vs_golden",
    )
    if small_result is None or mega_result is None:
        raise ValueError("Unable to calculate comparison metrics")

    if args.print_details:
        small_result.print_result()
        mega_result.print_result()
    status, reason = mega_result.check_result(small_result)
    result = "PASS" if status == "ok" else "FAIL"

    small_abs = np.abs(aligned["small_actual"] - golden)
    mega_abs = np.abs(aligned["mega_actual"] - golden)
    better = int(np.count_nonzero(mega_abs < small_abs))
    equal = int(np.count_nonzero(mega_abs == small_abs))
    worse = int(np.count_nonzero(mega_abs > small_abs))
    print(
        f"[SUMMARY] MegaMoe vs small-op: better={better}, equal={equal}, worse={worse}, "
        f"total={len(golden)}"
    )
    print(f"[RESULT] {result}{(': ' + reason.strip()) if reason.strip() else ''}")

    if args.report is not None:
        write_report(
            args.report, aligned, golden, result, reason, better, equal, worse,
            source_elements, excluded_elements, args.golden_diff_threshold,
        )
        print(f"[INFO] Full aligned report written to {args.report}")
    return 0 if result == "PASS" else 1


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a saved small-op table and a saved MegaMoe table against "
            "one Golden using the existing MegaMoe precision protocol."
        )
    )
    parser.add_argument("small_txt", type=Path, help="Small-op TXT containing idx/golden/actual")
    parser.add_argument("mega_txt", type=Path, help="MegaMoe TXT containing Loop/cascade/golden/npu")
    parser.add_argument("--small-section", type=int, default=0, help="Zero-based small-op table index")
    parser.add_argument("--mega-section", type=int, default=0, help="Zero-based MegaMoe table index")
    parser.add_argument("--mega-index-base", type=int, default=1, choices=(0, 1))
    parser.add_argument("--golden-source", choices=("small", "mega"), default="mega")
    parser.add_argument(
        "--golden-diff-threshold", type=float, default=1e-5,
        help="Compare only indexes whose two Golden values have abs_diff strictly below this value",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--diff-threshold", type=float, default=0.001)
    parser.add_argument("--print-details", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional full per-element report path")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        return compare_results(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
