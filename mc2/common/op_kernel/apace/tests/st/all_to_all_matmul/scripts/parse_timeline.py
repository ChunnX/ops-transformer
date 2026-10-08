#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# ---------------------------------------------------------------------------

"""Parse msprof op_summary CSVs and break down AllToAllMmKernel by task type.

背景：融合内核（AIC 计算 + AIV 通信 + 搬数）在 op_summary 里可能被拆成多个
Task Type 行（AI Core / AI Vector / M2XGE 等）。本脚本把同一 kernel 启动的
所有任务类型行汇总，打印每类任务的耗时与占比，用于回答：
  - 一个 kernel 启动是 1 行还是多行（按 Task Type 拆分）？
  - 计算(AI Core) / 通信(AIV) / 搬数(M2XGE) 各占多少？
  - FUSED / COMM_ONLY / MM_ONLY 与各 lm 之间的差异落在哪一类任务上？

用法：
  python3 scripts/parse_timeline.py                 # 最新 case
  python3 scripts/parse_timeline.py --all           # 全部 case
  python3 scripts/parse_timeline.py --case 3        # 第 3 个 case
  python3 scripts/parse_timeline.py --prof-dir DIR  # 指定 prof 目录
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

KERNEL_SUBSTR = "AllToAllMmKernel"
_TS_RE = re.compile(r"PROF_\d+_(\d{17})_")
GROUP_THRESHOLD_S = 5

_DEFAULT_PROF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prof")


def parse_ts(dirname):
    m = _TS_RE.search(dirname)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S%f") if m else None


def norm_col(name):
    return re.sub(r"[^a-z0-9_]", "", name.strip().lower())


def get(row, candidates):
    for c in candidates:
        if c in row:
            v = row[c]
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def getf(row, candidates):
    v = get(row, candidates)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def find_op_summary(prof_base, prof_dir):
    return glob.glob(os.path.join(prof_base, prof_dir, "mindstudio_profiler_output", "op_summary_*.csv"))


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        return [r for r in csv.DictReader(f)]


def group_cases(prof_dirs):
    prof_dirs_sorted = sorted(prof_dirs, key=parse_ts)
    cases = []
    current = []
    for d in prof_dirs_sorted:
        if not current or (parse_ts(d) - parse_ts(current[0])).total_seconds() < GROUP_THRESHOLD_S:
            current.append(d)
        else:
            cases.append(current)
            current = [d]
    if current:
        cases.append(current)
    return cases


def analyze_case(prof_base, case_dirs, case_idx):
    rows = []
    for d in case_dirs:
        for p in find_op_summary(prof_base, d):
            rows.extend(read_rows(p))

    kernel_rows = [r for r in rows if KERNEL_SUBSTR in (get(r, ["Op Name", "op name"]) or "")]
    print(f"  [Case {case_idx}]  ({len(case_dirs)} dirs, {case_dirs[0]})")
    if not kernel_rows:
        print("    no kernel rows")
        return

    # 1) 每个 (device, task_id/start) 启动是否被拆成多行
    launch = defaultdict(list)
    for r in kernel_rows:
        dev = get(r, ["Device_id", "device id"]) or "?"
        tid = get(r, ["Task ID", "task id"]) or get(r, ["Task Start Time(us)", "task start time(us)"]) or "?"
        launch[(dev, tid)].append(r)

    split_cnt = sum(1 for v in launch.values() if len(v) > 1)
    print(f"    kernel rows={len(kernel_rows)}  launches={len(launch)}  "
          f"(launches split into >1 task-type rows: {split_cnt})")

    # 2) 按 Task Type 汇总
    by_type = defaultdict(list)
    for r in kernel_rows:
        tt = get(r, ["Task Type", "task type", "Type", "type"]) or "?"
        by_type[tt].append(r)

    print("    per-task-type duration(us):")
    for tt in sorted(by_type.keys()):
        durs = sorted(getf(r, ["Task Duration(us)", "task duration(us)"]) for r in by_type[tt])
        waits = [getf(r, ["Task Wait Time(us)", "task wait time(us)"]) for r in by_type[tt]]
        if not durs:
            continue
        print(f"      {tt:<14} n={len(durs):3d}  min={durs[0]:9.3f}  avg={sum(durs) / len(durs):9.3f}  "
              f"max={durs[-1]:9.3f}  waitAvg={sum(waits) / len(waits):9.3f}")

    # 3) 每次 launch 的总时长（若拆分则按行求和）
    totals = sorted(sum(getf(r, ["Task Duration(us)", "task duration(us)"]) for r in rs)
                    for rs in launch.values())
    if totals:
        print(f"    per-launch total(us): min={totals[0]:.3f}  avg={sum(totals) / len(totals):.3f}  "
              f"max={totals[-1]:.3f}  (n={len(totals)})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Parse msprof op_summary by task type.")
    parser.add_argument("--prof-dir", default=_DEFAULT_PROF)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--case", type=int)
    g.add_argument("--latest", action="store_true", default=True)
    args = parser.parse_args()

    prof_base = os.path.abspath(args.prof_dir)
    prof_dirs = [d for d in os.listdir(prof_base) if re.match(r"PROF_\d+_\d{17}_", d)]
    if not prof_dirs:
        print(f"No PROF directories under {prof_base}", file=sys.stderr)
        return 1

    cases = group_cases(prof_dirs)
    if args.all:
        for i, case_dirs in enumerate(cases, 1):
            analyze_case(prof_base, case_dirs, i)
    elif args.case is not None:
        if not 1 <= args.case <= len(cases):
            print(f"--case {args.case} out of range (1..{len(cases)})", file=sys.stderr)
            return 1
        analyze_case(prof_base, cases[args.case - 1], args.case)
    else:
        analyze_case(prof_base, cases[-1], len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
