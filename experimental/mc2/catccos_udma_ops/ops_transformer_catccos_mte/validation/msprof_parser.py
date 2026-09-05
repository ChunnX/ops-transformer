# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Parse per-device msprof rows and quantify cross-rank launch skew."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


KERNEL_NAME_SUBSTRINGS = {
    "all_to_allv_grouped_mat_mul": (
        "Ascend950AllToAllVGroupedMatmulMte",
        "all_to_allv_grouped_mat_mul_mte",
        "AllToAllVGroupedMatMulMte",
    ),
    "grouped_mat_mul_all_to_allv": (
        "GroupedMatmulAllToAllVMte",
        "grouped_mat_mul_all_to_allv_mte",
        "GroupedMatMulAllToAllVMte",
    ),
}


def _field(row: Mapping[str, str], names: Sequence[str], default: str = "") -> str:
    lowered = {key.strip().lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            value = lowered[name.lower()].strip()
            if value:
                return value
    return default


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _read_rows(
    profile_dir: Path, kernel_name_substrings: Sequence[str]
) -> Tuple[Dict[str, List[Tuple[float, float]]], List[str], List[str]]:
    op_files = sorted(profile_dir.rglob("op_summary_*.csv"))
    if not op_files:
        raise FileNotFoundError("no exported op_summary_*.csv found under {}".format(profile_dir))
    devices = defaultdict(list)
    candidate_names = set()
    used_files = []
    fallback_order = 0.0
    for csv_path in op_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                name = _field(row, ("Op Name", "Name", "OP Type", "Op Type", "Type"))
                if name:
                    candidate_names.add(name)
                if not any(fragment.lower() in name.lower() for fragment in kernel_name_substrings):
                    continue
                device = _field(row, ("Device_id", "Device ID", "device_id"), "unknown")
                duration_text = _field(row, ("Task Duration(us)", "Duration(us)", "Task Duration"))
                if not duration_text:
                    continue
                start_text = _field(row, ("Task Start Time(us)", "Start Time(us)", "Task Start Time"))
                try:
                    duration = float(duration_text)
                    start = float(start_text) if start_text else fallback_order
                except ValueError:
                    continue
                fallback_order += 1.0
                devices[device].append((start, duration))
                used_files.append(str(csv_path))
    for entries in devices.values():
        entries.sort(key=lambda item: item[0])
    return dict(devices), sorted(candidate_names), sorted(set(used_files))


def parse_msprof_case(profile_dir: Union[Path, str], case: Mapping[str, Any]) -> Dict[str, Any]:
    operator = case.get("operator", "all_to_allv_grouped_mat_mul")
    if operator not in KERNEL_NAME_SUBSTRINGS:
        raise ValueError("unsupported operator for msprof parsing: {}".format(operator))
    devices, candidate_names, used_files = _read_rows(
        Path(profile_dir), KERNEL_NAME_SUBSTRINGS[operator]
    )
    performance = case["performance"]
    expected = performance["warmup"] + performance["repeats"] * (
        performance["align_iterations"] + performance["iterations"]
    )
    errors = []
    if len(devices) != case["world_size"]:
        errors.append("expected {} devices, found {}".format(case["world_size"], len(devices)))
    for device, entries in sorted(devices.items()):
        if len(entries) != expected:
            errors.append("device {}: expected {} target rows, found {}".format(device, expected, len(entries)))
    if errors:
        return {
            "case_id": case["id"], "valid": False, "passed": False,
            "errors": errors, "candidate_op_names": candidate_names,
            "op_summary_files": used_files,
            "expected_target_launches_per_device": expected,
            "found_target_launches_per_device": {key: len(value) for key, value in devices.items()},
        }

    selected = {}
    for device, entries in sorted(devices.items()):
        durations = [item[1] for item in entries]
        measured = []
        cursor = performance["warmup"]
        for _ in range(performance["repeats"]):
            cursor += performance["align_iterations"]
            measured.extend(durations[cursor:cursor + performance["iterations"]])
            cursor += performance["iterations"]
        selected[device] = measured

    device_names = sorted(
        selected, key=lambda item: (0, int(item)) if item.isdigit() else (1, item)
    )
    iteration_rows = []
    for iteration_index, rank_values in enumerate(zip(*(selected[name] for name in device_names))):
        center = median(rank_values)
        skew = (max(rank_values) - min(rank_values)) / center if center else 0.0
        iteration_rows.append({
            "iteration": iteration_index,
            "repeat": iteration_index // performance["iterations"],
            "iteration_in_repeat": iteration_index % performance["iterations"],
            "rank_durations_us": {name: value for name, value in zip(device_names, rank_values)},
            "min_us": min(rank_values), "median_us": center, "max_us": max(rank_values),
            "rank_skew_ratio": skew,
        })
    max_rank_durations = [row["max_us"] for row in iteration_rows]
    skew_values = [row["rank_skew_ratio"] for row in iteration_rows]
    skew_p90 = _quantile(skew_values, 0.90)
    skew_limit = performance["max_rank_skew_ratio"]
    per_device = {}
    for device, values in selected.items():
        per_device[device] = {
            "samples": len(values), "mean_us": sum(values) / len(values),
            "p50_us": _quantile(values, 0.50), "p90_us": _quantile(values, 0.90),
            "p99_us": _quantile(values, 0.99), "min_us": min(values), "max_us": max(values),
        }
    valid = skew_p90 <= skew_limit
    return {
        "case_id": case["id"], "valid": valid, "passed": valid,
        "errors": [] if valid else [
            "p90 rank skew {:.2%} exceeds {:.2%}; result is invalid for regression gating".format(
                skew_p90, skew_limit
            )
        ],
        "metric_definition": "p50 of iteration-wise maximum device duration",
        "latency_us": {
            "p50": _quantile(max_rank_durations, 0.50),
            "p90": _quantile(max_rank_durations, 0.90),
            "p99": _quantile(max_rank_durations, 0.99),
            "mean": sum(max_rank_durations) / len(max_rank_durations),
            "min": min(max_rank_durations), "max": max(max_rank_durations),
        },
        "rank_skew_ratio": {
            "p50": _quantile(skew_values, 0.50), "p90": skew_p90,
            "p99": _quantile(skew_values, 0.99), "limit": skew_limit,
        },
        "per_device": per_device, "iterations": iteration_rows,
        "occurrence_selection": {
            "warmup_skipped": performance["warmup"],
            "primer_skipped_per_repeat": performance["align_iterations"],
            "measured_per_repeat": performance["iterations"],
            "repeats": performance["repeats"],
        },
        "expected_target_launches_per_device": expected,
        "op_summary_files": used_files,
    }
