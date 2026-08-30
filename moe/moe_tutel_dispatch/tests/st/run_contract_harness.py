#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Validate and execute directed MoeTutelDispatch contract-test entries.

The manifest keeps invalid Host/GE descriptor cases out of positive TTK CSVs
while still giving every design row one deterministic execution target.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ALLOWED = {
    "ttk_kernel",
    "ttk_geir",
    "host_ut",
    "build_scan",
    "static_contract",
    "test_infra_ut",
}


def load_manifest(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    ids = [row["case_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise RuntimeError("manifest must be non-empty and case_id must be unique")
    bad = [
        row
        for row in rows
        if row["harness"] not in ALLOWED
        or not row["target"]
        or int(row["repeats"]) < 1
        or int(row["parallelism"]) < 1
        or row["dynamic"] not in {"0", "1"}
    ]
    if bad:
        raise RuntimeError(f"manifest contains unroutable rows: {bad}")
    return rows


def split_csv_target(st_dir, target):
    try:
        relative, case_name = target.split("#", 1)
    except ValueError as error:
        raise RuntimeError(f"invalid CSV target: {target!r}") from error
    if not relative or not case_name:
        raise RuntimeError(f"invalid CSV target: {target!r}")

    root = st_dir.resolve()
    csv_path = (root / relative).resolve()
    try:
        csv_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"CSV target escapes the ST directory: {target!r}"
        ) from error
    if not csv_path.is_file():
        raise RuntimeError(f"CSV target is not a regular file: {csv_path}")
    return csv_path, case_name


def command_for(row, args, st_dir, result_path=None):
    harness = row["harness"]
    if harness in {"ttk_kernel", "ttk_geir"}:
        if args.ttk_root is None:
            raise RuntimeError("--ttk-root is required for TTK entries")
        if args.manual_data_dir is None:
            raise RuntimeError("--manual-data-dir is required for directed TTK entries")
        csv_path, case_name = split_csv_target(st_dir, row["target"])
        mode = "kernel" if harness == "ttk_kernel" else "geir"
        repeats = int(row["repeats"])
        command = [
            sys.executable,
            "-m",
            "ttk",
            mode,
            "-i",
            str(csv_path),
            "-t",
            case_name,
            "--pc=1",
            f"--run={repeats}",
            "--compare=binary",
            "--warmup",
            "false",
            "--manual-data-dirs",
            str(args.manual_data_dir),
        ]
        if harness == "ttk_geir" and repeats > 1:
            command.append("--dl=1")
        if harness == "ttk_kernel":
            command.extend(["-d=false", "-b=release"])
        elif row["dynamic"] == "1":
            command.extend(["-c=false", "-d=true"])
        if args.plugin is not None:
            command.extend(["--plugin", str(args.plugin)])
        if result_path is not None:
            command.extend(["--output", str(result_path)])
        return command, args.ttk_root
    if harness == "host_ut":
        if args.host_ut_binary is None:
            raise RuntimeError("--host-ut-binary is required for Host UT entries")
        return [str(args.host_ut_binary), f"--gtest_filter={row['target']}"], None
    if harness == "test_infra_ut":
        if args.ttk_root is None:
            raise RuntimeError("--ttk-root is required for test-infra entries")
        return [sys.executable, "-m", "pytest", "-q", row["target"]], args.ttk_root
    if harness == "build_scan":
        if args.build_scan is None:
            raise RuntimeError("--build-scan is required for build-scan entries")
        return [str(args.build_scan), "--case", row["case_id"]], None
    if harness == "static_contract":
        if args.static_contract is None:
            raise RuntimeError(
                "--static-contract is required for static-contract entries"
            )
        return [str(args.static_contract), "--case", row["case_id"]], None
    raise RuntimeError(f"unsupported harness: {harness}")


def validate_ttk_result(path, row):
    if not path.is_file():
        raise RuntimeError(f"TTK did not create result CSV: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        results = list(csv.DictReader(file))
    if len(results) != 1:
        raise RuntimeError(f"TTK result must contain exactly one row: {path}")
    result = results[0]
    if result.get("testcase_name") != row["case_id"]:
        raise RuntimeError(
            f"TTK result case mismatch: expected {row['case_id']}, "
            f"got {result.get('testcase_name')!r}"
        )
    if result.get("precision_status") != "PASS":
        raise RuntimeError(
            f"TTK precision failed for {row['case_id']}: "
            f"{result.get('precision_status')!r}"
        )
    if row["harness"] == "ttk_kernel":
        if result.get("bin_compile_s") != "BINARY_MATCH":
            raise RuntimeError(
                f"TTK binary was not matched for {row['case_id']}: "
                f"{result.get('bin_compile_s')!r}"
            )
        if result.get("bin_precision") != "100%":
            raise RuntimeError(
                f"TTK binary comparison was not exact for {row['case_id']}: "
                f"{result.get('bin_precision')!r}"
            )


def execute_row(row, args, st_dir):
    parallelism = int(row["parallelism"])
    is_ttk = row["harness"] in {"ttk_kernel", "ttk_geir"}
    if is_ttk and args.result_dir is None:
        raise RuntimeError("--result-dir is required for TTK execution")
    if is_ttk:
        args.result_dir.mkdir(parents=True, exist_ok=True)

    def run_one(lane=None):
        result_path = None
        if is_ttk:
            suffix = "" if lane is None else f"_p{lane}"
            result_path = args.result_dir / f"{row['case_id']}{suffix}.csv"
        command, cwd = command_for(row, args, st_dir, result_path)
        print(
            f"RUN {row['case_id']}: parallelism={parallelism} {' '.join(command)}",
            flush=True,
        )
        subprocess.run(command, cwd=cwd, check=True)
        if result_path is not None:
            validate_ttk_result(result_path, row)

    if parallelism == 1:
        run_one()
        return
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(run_one, lane) for lane in range(parallelism)]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ttk-root", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--manual-data-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--host-ut-binary", type=Path)
    parser.add_argument("--build-scan", type=Path)
    parser.add_argument("--static-contract", type=Path)
    args = parser.parse_args()

    st_dir = Path(__file__).resolve().parent
    manifest = (
        args.manifest
        or st_dir / "testcases/MoeTutelDispatch_contract_harness_manifest.csv"
    )
    rows = load_manifest(manifest)
    if args.case:
        selected = set(args.case)
        rows = [row for row in rows if row["case_id"] in selected]
        missing = selected - {row["case_id"] for row in rows}
        if missing:
            raise RuntimeError(f"unknown case id(s): {sorted(missing)}")

    print(
        f"PASS: manifest rows={len(rows)}, unique targets={len({row['target'] for row in rows})}"
    )
    if args.list or not args.execute:
        for row in rows:
            print(
                f"{row['case_id']}\t{row['harness']}\t{row['target']}\t{row['expected']}"
            )
    if not args.execute:
        return

    for row in rows:
        execute_row(row, args, st_dir)


if __name__ == "__main__":
    main()
