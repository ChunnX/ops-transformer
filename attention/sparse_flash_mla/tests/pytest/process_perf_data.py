#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Merge all SparseFlashMla msprof runs into one Excel result file."""

import argparse
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
OP_NAME_COLUMN = "Op Name"
TASK_DURATION_COLUMN = "Task Duration(us)"
PERF_COLUMNS = [
    "profile_folder",
    "profile_run",
    "metadata_duration",
    "smla_duration",
]
INVALID_RESULTS = {"npu error", "exception"}


@dataclass(frozen=True)
class PerfSample:
    """One SparseFlashMla invocation and its matching metadata invocation."""

    profile_folder: str
    profile_run: int
    metadata_duration: object = None
    smla_duration: object = None


def _normalise_op_name(op_name):
    return re.sub(r"[^a-z0-9]", "", str(op_name).lower())


def _classify_op(op_name):
    normalised = _normalise_op_name(op_name)
    if "sparseflashmlametadata" in normalised:
        return "metadata"
    if "sparseflashmla" in normalised:
        return "smla"
    return None


def _profile_sort_key(profile_folder):
    try:
        modified_time = profile_folder.stat().st_mtime_ns
    except OSError:
        modified_time = 0
    return modified_time, profile_folder.name


def find_profile_folders(profile_root):
    """Return every msprof output directory in collection order."""
    root = Path(profile_root)
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("PROF")
        ),
        key=_profile_sort_key,
    )


def _read_op_summary_files(profile_folder):
    profiler_output = profile_folder / "mindstudio_profiler_output"
    if not profiler_output.is_dir():
        LOGGER.warning(
            "Skip %s: mindstudio_profiler_output does not exist", profile_folder
        )
        return []

    summary_files = sorted(profiler_output.glob("op_summary*.csv"))
    if not summary_files:
        LOGGER.warning("Skip %s: op_summary CSV does not exist", profile_folder)
        return []

    summaries = []
    for summary_file in summary_files:
        try:
            summary = pd.read_csv(summary_file)
        except Exception as error:  # keep processing the other profiler outputs
            LOGGER.warning("Failed to read %s: %s", summary_file, error)
            continue
        missing_columns = {
            OP_NAME_COLUMN,
            TASK_DURATION_COLUMN,
        } - set(summary.columns)
        if missing_columns:
            LOGGER.warning(
                "Skip %s: missing columns %s",
                summary_file,
                sorted(missing_columns),
            )
            continue
        summaries.append(summary)
    return summaries


def extract_profile_samples(profile_folder):
    """Extract all SparseFlashMla measurements from one PROF directory."""
    metadata_durations = []
    smla_durations = []

    for summary in _read_op_summary_files(Path(profile_folder)):
        for _, row in summary.iterrows():
            op_type = _classify_op(row[OP_NAME_COLUMN])
            if op_type == "metadata":
                metadata_durations.append(row[TASK_DURATION_COLUMN])
            elif op_type == "smla":
                smla_durations.append(row[TASK_DURATION_COLUMN])

    sample_count = max(len(metadata_durations), len(smla_durations))
    samples = []
    for index in range(sample_count):
        samples.append(
            PerfSample(
                profile_folder=Path(profile_folder).name,
                profile_run=index + 1,
                metadata_duration=(
                    metadata_durations[index]
                    if index < len(metadata_durations)
                    else None
                ),
                smla_duration=(
                    smla_durations[index] if index < len(smla_durations) else None
                ),
            )
        )
    return samples


def collect_profile_samples(profile_root):
    """Extract measurements from all PROF directories without dropping runs."""
    samples = []
    for profile_folder in find_profile_folders(profile_root):
        folder_samples = extract_profile_samples(profile_folder)
        LOGGER.info(
            "%s: collected %d SparseFlashMla run(s)",
            profile_folder.name,
            len(folder_samples),
        )
        samples.extend(folder_samples)
    return samples


def _valid_result_indices(result_frame):
    if "result" not in result_frame.columns:
        return list(result_frame.index)
    statuses = result_frame["result"].fillna("").astype(str).str.strip().str.lower()
    return [
        index
        for index in result_frame.index
        if statuses.at[index] not in INVALID_RESULTS
    ]


def _same_value(left, right):
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    return left == right


def _same_result_row(result_frame, existing_frame, index):
    return all(
        column in existing_frame.columns
        and _same_value(result_frame.at[index, column], existing_frame.at[index, column])
        for column in result_frame.columns
    )


def _sample_key(sample):
    values = (
        sample.profile_folder,
        sample.profile_run,
        sample.metadata_duration,
        sample.smla_duration,
    )
    return tuple(None if pd.isna(value) else value for value in values)


def _row_sample_key(row):
    return tuple(
        None if pd.isna(row.get(column)) else row.get(column)
        for column in PERF_COLUMNS
    )


def merge_results_and_perf(result_frame, samples, existing_frame=None):
    """Map new profiler samples to pytest rows while preserving earlier runs."""
    merged = result_frame.copy()
    for column in PERF_COLUMNS:
        merged[column] = pd.NA

    if existing_frame is not None:
        shared_row_count = min(len(merged), len(existing_frame))
        for index in range(shared_row_count):
            if not _same_result_row(result_frame, existing_frame, index):
                continue
            for column in PERF_COLUMNS:
                if column in existing_frame.columns:
                    merged.at[index, column] = existing_frame.at[index, column]

    valid_indices = _valid_result_indices(merged)
    empty_indices = [
        index
        for index in valid_indices
        if pd.isna(merged.at[index, "metadata_duration"])
        and pd.isna(merged.at[index, "smla_duration"])
    ]
    existing_keys = {
        _row_sample_key(row)
        for _, row in merged.iterrows()
        if not (
            pd.isna(row["metadata_duration"]) and pd.isna(row["smla_duration"])
        )
    }
    new_samples = [
        sample for sample in samples if _sample_key(sample) not in existing_keys
    ]
    # msprof may reuse and overwrite the same PROF directory. In that case the
    # profile key can repeat, but newly appended pytest rows still need the
    # current measurements.
    if empty_indices and len(new_samples) < len(empty_indices):
        new_samples = samples[-len(empty_indices) :]

    for result_index, sample in zip(empty_indices, new_samples):
        for column in PERF_COLUMNS:
            merged.at[result_index, column] = getattr(sample, column)

    unmatched_samples = new_samples[len(empty_indices) :]
    if unmatched_samples:
        LOGGER.warning(
            "New profiler runs (%d) exceed empty pytest rows (%d); "
            "retaining %d run(s) as performance-only rows",
            len(new_samples),
            len(empty_indices),
            len(unmatched_samples),
        )
        extra_rows = []
        for sample in unmatched_samples:
            row = {column: pd.NA for column in merged.columns}
            for column in PERF_COLUMNS:
                row[column] = getattr(sample, column)
            extra_rows.append(row)
        merged = pd.concat([merged, pd.DataFrame(extra_rows)], ignore_index=True)

    return merged


def default_perf_path(test_result_path):
    result_path = Path(test_result_path)
    if result_path.suffix.lower() == ".xlsx":
        return result_path.with_name(f"{result_path.stem}_perf.xlsx")
    return result_path.with_name(f"{result_path.name}_perf.xlsx")


def process_profiler_data(
    test_result_path,
    profile_root=".",
    output_path=None,
    delete_perf_files=False,
):
    """Merge all available profiler runs and save them atomically."""
    result_path = Path(test_result_path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Test result file does not exist: {result_path}")

    result_frame = pd.read_excel(result_path)
    profile_folders = find_profile_folders(profile_root)
    if not profile_folders:
        raise FileNotFoundError(f"No PROF directory found under: {profile_root}")

    samples = collect_profile_samples(profile_root)
    if not samples:
        raise ValueError("No SparseFlashMla performance data found in PROF directories")

    destination = Path(output_path) if output_path else default_perf_path(result_path)
    existing_frame = pd.read_excel(destination) if destination.is_file() else None
    merged = merge_results_and_perf(result_frame, samples, existing_frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.tmp.xlsx")
    merged.to_excel(temporary_path, index=False)
    os.replace(temporary_path, destination)

    LOGGER.info(
        "Saved %d profiler run(s) from %d folder(s) to %s",
        len(samples),
        len(profile_folders),
        destination,
    )

    if delete_perf_files:
        for profile_folder in profile_folders:
            shutil.rmtree(profile_folder)
            LOGGER.info("Deleted profiler directory: %s", profile_folder)

    return destination


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge all SparseFlashMla msprof data into one Excel table"
    )
    parser.add_argument(
        "--test_result_path",
        default="./result/smla_result.xlsx",
        help="pytest result Excel path",
    )
    parser.add_argument(
        "--profile_root",
        default=".",
        help="directory containing one or more PROF* folders",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="output Excel path; defaults to <test_result>_perf.xlsx",
    )
    parser.add_argument(
        "--delete_perf_files",
        action="store_true",
        help="delete PROF* folders after a successful merge",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    try:
        process_profiler_data(
            test_result_path=args.test_result_path,
            profile_root=args.profile_root,
            output_path=args.output_path,
            delete_perf_files=args.delete_perf_files,
        )
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as error:
        LOGGER.error("Failed to process profiler data: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
