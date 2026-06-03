#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[5]
TORCH_EXTENSION_DIR = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_DIR.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_DIR))

from npu_ops_transformer.ops import npu_tri_inverse
from test_tri_inverse_paramset import PROFILE_SHAPES, shape_to_tile_num
from tri_inverse_golden import make_cc_input


OP_TYPE = "TriInverse"
SUMMARY_COLUMNS = [
    "shape",
    "Duration(us)",
    "aic_mac_time(us)",
    "aiv_vec_time(us)",
    "aic_mte2_time(us)",
    "aic_mac_ratio",
]


def first_value(row, names):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return str(row[name]).strip()
    return "0"


def to_float(value):
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def run_once(shape, seed=0):
    tile_num = shape_to_tile_num(shape)
    cc_in = make_cc_input(tile_num, seed=seed, device="npu")
    actual = npu_tri_inverse(cc_in)
    torch_npu.npu.synchronize()
    return actual


def profile_shape(shape, output_dir, warmup, repeat, seed):
    case_dir = output_dir / f"shape_{shape}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(warmup):
        run_once(shape, seed)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        export_type="text",
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=repeat, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(case_dir), analyse_flag=True),
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(repeat):
            run_once(shape, seed)
            prof.step()

    return case_dir


def find_kernel_details(case_dir):
    files = sorted(case_dir.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if files:
        return files[-1]
    files = sorted(case_dir.glob("ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if files:
        return files[-1]
    files = sorted(case_dir.glob("mindstudio_profiler_output/op_summary_*.csv"))
    return files[-1] if files else None


def parse_profile_case(shape, case_dir):
    summary_path = find_kernel_details(case_dir)
    if summary_path is None:
        raise FileNotFoundError(f"kernel details csv not found under {case_dir}")

    rows = []
    with summary_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("Type") == OP_TYPE or row.get("OP Type") == OP_TYPE:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"{OP_TYPE} row not found in {summary_path}")

    row = rows[-1]
    return {
        "shape": shape,
        "Duration(us)": to_float(first_value(row, ["Duration(us)", "Task Duration(us)", "task_duration(us)"])),
        "aic_mac_time(us)": to_float(first_value(row, ["aic_mac_time(us)"])),
        "aiv_vec_time(us)": to_float(first_value(row, ["aiv_vec_time(us)"])),
        "aic_mte2_time(us)": to_float(first_value(row, ["aic_mte2_time(us)"])),
        "aic_mac_ratio": to_float(first_value(row, ["aic_mac_ratio"])),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Profile tri_inverse and extract compact CSV metrics.")
    parser.add_argument("--output-dir", default="prof_out/tri_inverse_profile", help="Profiler output root.")
    parser.add_argument("--csv", default=None, help="Summary CSV path. Defaults to <output-dir>/tri_inverse_profile.csv.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shapes", type=int, nargs="*", default=PROFILE_SHAPES)
    args = parser.parse_args()

    torch_npu.npu.set_device(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else output_dir / "tri_inverse_profile.csv"

    rows = []
    for shape in args.shapes:
        print(f"[PROFILE] shape={shape}, tile_num={shape_to_tile_num(shape)}")
        case_dir = profile_shape(shape, output_dir, args.warmup, args.repeat, args.seed)
        rows.append(parse_profile_case(shape, case_dir))
        write_csv(csv_path, rows)

    print(f"[PROFILE] summary csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
