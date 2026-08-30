#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

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


def command_for(row, args, st_dir):
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
            "--manual-data-dirs",
            str(args.manual_data_dir),
        ]
        if harness == "ttk_geir" and repeats > 1:
            command.append("--dl=1")
        if row["dynamic"] == "1":
            command.extend(["-c=false", "-d=true"])
        if args.plugin is not None:
            command.extend(["--plugin", str(args.plugin)])
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


def execute_row(row, args, st_dir):
    command, cwd = command_for(row, args, st_dir)
    parallelism = int(row["parallelism"])
    print(
        f"RUN {row['case_id']}: parallelism={parallelism} {' '.join(command)}",
        flush=True,
    )
    if parallelism == 1:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [
            executor.submit(subprocess.run, command, cwd=cwd, check=True)
            for _ in range(parallelism)
        ]
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
