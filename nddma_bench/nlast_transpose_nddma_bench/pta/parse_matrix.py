# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""解析 NLastTranspose 四配置 msprof op_summary（dim0=56），对齐 MANIFEST 执行顺序，
计算三 dtype（FP16/FP32/B8）时延/带宽矩阵。

CASES/CAP/ELEM/D0/DT_NAME 从 prof_pta_matrix.py 导入，保证与采集驱动完全同源（含 per-case dim1
与 int8）。每用例在会话内跑 (1 预热 + N 测量) 个 kernel task；op_summary 按 Task Start Time 升序
= 执行顺序，与驱动 RUN manifest 一一对齐。取 N 个测量 task 的 min（峰值带宽）与 mean。

带宽口径:
  纯读   BW = numel×elemSize / t
  读写   BW = 2×numel×elemSize / t  (转置 W=R)
  numel = D0·d1·d2·d3·d4；elemSize FP16=2/FP32=4/B8=1

用法: python3 parse_matrix.py <matrix_dir> [iters]
输出: <matrix_dir>/parsed.json
"""

import csv
import glob
import json
import os
import sys

from prof_pta_matrix import CASES, CAP, ELEM, D0, DT_NAME

CFG2MODE = {"db_rw": "db", "db_r": "db", "sb_rw": "sb", "sb_r": "sb"}
CFG2MTE3 = {"db_rw": 1, "db_r": 0, "sb_rw": 1, "sb_r": 0}


def run_order(mode):
    """该 mode 下按 CASES 顺序过滤容量后的 run 用例（与驱动一致）。tile=d2*d3*d4。
    CASES 元组 (label, d2, d3, d4, dt, dim1)。"""
    cap = CAP[mode]
    return [c for c in CASES if c[1] * c[2] * c[3] <= cap[c[4]]]


def load_durations(matrix_dir, cfg):
    """读取该配置 op_summary，返回按执行顺序（Task Start Time 升序）的 Task Duration 列表。"""
    pat = os.path.join(matrix_dir, cfg, "PROF_*", "mindstudio_profiler_output",
                       "op_summary_*.csv")
    files = glob.glob(pat)
    assert len(files) == 1, f"{cfg}: expect 1 op_summary, got {files}"
    rows = []
    with open(files[0]) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["Task Start Time(us)"].strip()),
                         float(r["Task Duration(us)"].strip())))
    rows.sort(key=lambda x: x[0])          # 执行顺序
    return [d for _, d in rows], os.path.relpath(files[0], matrix_dir)


def parse(matrix_dir, iters):
    per = iters + 1                        # 1 预热 + iters 测量
    result = {}
    csvmap = {}
    for cfg in ("db_rw", "sb_rw", "db_r", "sb_r"):
        mode = CFG2MODE[cfg]
        is_mte3 = CFG2MTE3[cfg]
        order = run_order(mode)
        durs, rel = load_durations(matrix_dir, cfg)
        csvmap[cfg] = rel
        exp = len(order) * per
        assert len(durs) == exp, f"{cfg}: {len(durs)} tasks != {len(order)}*{per}={exp}"
        cfg_res = {}
        for i, (lb, d2, d3, d4, dt, d1) in enumerate(order):
            meas = durs[i * per:(i + 1) * per][1:]   # 丢弃预热
            mn = min(meas)
            mean = sum(meas) / len(meas)
            n = D0 * d1 * d2 * d3 * d4
            R = n * ELEM[dt]
            factor = 2 if is_mte3 else 1   # 读写=R+W=2R；纯读=R
            bw = factor * R / (mn * 1e-6) / 1e9   # GB/s（min 时延对应峰值带宽）
            cfg_res[lb] = {
                "min_us": round(mn, 3), "mean_us": round(mean, 4),
                "bw_gbs": round(bw, 1), "numel": n, "dt": dt,
                "dt_name": DT_NAME[dt], "shape": [D0, d1, d2, d3, d4],
            }
        result[cfg] = cfg_res
    return result, csvmap


if __name__ == "__main__":
    matrix_dir = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    res, csvmap = parse(matrix_dir, iters)
    out = os.path.join(matrix_dir, "parsed.json")
    json.dump({"cfgs": res, "csv": csvmap, "iters": iters}, open(out, "w"),
              ensure_ascii=False, indent=1)
    for cfg in ("db_rw", "sb_rw", "db_r", "sb_r"):
        print(f"{cfg}: {len(res[cfg])} cases")
    print("saved", out)
