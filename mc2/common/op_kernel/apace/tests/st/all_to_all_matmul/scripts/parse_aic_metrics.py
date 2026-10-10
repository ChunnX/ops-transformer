#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# ---------------------------------------------------------------------------

"""Read ai_core_op_summary.db and break AllToAllMmKernel time down by engine.

ai_core_metrics 表把每个任务拆成独立引擎时间列：
  aic_total_time  AIC 总时长
  aic_mac_time    矩阵乘（cube MAC）纯计算时间
  aic_mte1_time   L1->L0 拷贝
  aic_mte2_time   GM->L1 拷贝
  aic_mte3_time   搬出
  aic_fixpipe_time fixpipe（L0C->GM）搬出
  aic_vec_time    向量
  aiv_total_time  AIV 总时长
  start/end_time  任务时间戳

用法：
  python3 scripts/parse_aic_metrics.py <PROF_CASE_DIR>
  # e.g. python3 scripts/parse_aic_metrics.py \
  #   prof/PROF_000001_20261008203713936_03750045MPENNLDD
"""

import os
import sys
import sqlite3
import glob
from collections import defaultdict

KERNEL_SUBSTR = "AllToAllMmKernel"

FIELDS = [
    ("aic_total_time", "aic_total"),
    ("aic_mac_time", "aic_mac"),
    ("aic_mte1_time", "aic_mte1"),
    ("aic_mte2_time", "aic_mte2"),
    ("aic_mte3_time", "aic_mte3"),
    ("aic_fixpipe_time", "aic_fixpipe"),
    ("aic_vec_time", "aic_vec"),
    ("aiv_total_time", "aiv_total"),
]


def query(db_path):
    rows = []
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for r in cur.execute(
            "SELECT m.task_id, m.stream_id, m.subtask_id, m.start_time, m.end_time, "
            "m.aic_total_time, m.aic_mac_time, m.aic_mte1_time, m.aic_mte2_time, m.aic_mte3_time, "
            "m.aic_fixpipe_time, m.aic_vec_time, m.aiv_total_time, g.op_name "
            "FROM ai_core_metrics m "
            "JOIN ge_summary g ON g.task_id = m.task_id AND g.stream_id = m.stream_id "
            "WHERE g.op_name LIKE ? "
            "ORDER BY m.start_time",
            ("%" + KERNEL_SUBSTR + "%",),
        ):
            rows.append(r)
    finally:
        con.close()
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    case_dir = sys.argv[1]
    dbs = sorted(glob.glob(os.path.join(case_dir, "device_*", "sqlite", "ai_core_op_summary.db")))
    if not dbs:
        print(f"no ai_core_op_summary.db under {case_dir}", file=sys.stderr)
        return 1

    for db in dbs:
        dev = os.path.basename(os.path.dirname(os.path.dirname(db)))
        rows = query(db)
        print(f"==== {dev} ({db})")
        if not rows:
            print("  no kernel rows")
            continue

        # 按 (task_id, stream_id) 分 launch
        launch = defaultdict(list)
        for r in rows:
            launch[(r[0], r[1])].append(r)
        print(f"  rows={len(rows)}  launches={len(launch)}  "
              f"(rows/launch: {sorted({len(v) for v in launch.values()})})")

        # 每个 launch 的引擎时间 = 该 launch 所有行对应列求和
        col_idx = {name: 5 + i for i, (name, _) in enumerate(FIELDS)}
        per_launch = []
        for key, rs in launch.items():
            rec = {}
            for name, _ in FIELDS:
                rec[name] = sum(float(r[col_idx[name]]) for r in rs if r[col_idx[name]] is not None)
            rec["start"] = min(float(r[3]) for r in rs)
            rec["end"] = max(float(r[4]) for r in rs)
            rec["span"] = rec["end"] - rec["start"]
            per_launch.append(rec)
        per_launch.sort(key=lambda x: x["start"])

        print(f"  per-launch engine time (us), avg over {len(per_launch)} launches:")
        hdr = f"    {'span':>9} {'aic_total':>10} {'aic_mac':>8} {'aic_mte2':>8} {'aic_mte3':>8} " \
              f"{'fixpipe':>8} {'aic_vec':>8} {'aiv_total':>9}"
        print(hdr)
        for rec in per_launch:
            print(f"    {rec['span']:9.1f} {rec['aic_total_time']:10.1f} {rec['aic_mac_time']:8.1f} "
                  f"{rec['aic_mte2_time']:8.1f} {rec['aic_mte3_time']:8.1f} {rec['aic_fixpipe_time']:8.1f} "
                  f"{rec['aic_vec_time']:8.1f} {rec['aiv_total_time']:9.1f}")
        n = len(per_launch)
        if n:
            avg = lambda k: sum(r[k] for r in per_launch) / n
            print(f"    {'avg':>9} {avg('aic_total_time'):10.1f} {avg('aic_mac_time'):8.1f} "
                  f"{avg('aic_mte2_time'):8.1f} {avg('aic_mte3_time'):8.1f} {avg('aic_fixpipe_time'):8.1f} "
                  f"{avg('aic_vec_time'):8.1f} {avg('aiv_total_time'):9.1f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
