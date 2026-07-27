# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""解析 LastTranspose B8(int8) 四配置 msprof op_summary，对齐 MANIFEST 执行顺序，
计算时延/带宽矩阵 + 流水 ratio。仅处理 dtype=2(int8) 用例（L_B8_0~19）。

CASES/CAP 从 prof_pta_matrix.py 导入，保证与采集驱动完全同源。
每用例在会话内跑 (1 预热 + N 测量) 个 kernel task；op_summary 按 Task Start Time 升序
= 执行顺序，与驱动 RUN manifest 一一对齐。取 N 个测量 task 的 min（峰值带宽）与 mean。

带宽口径 (B8 elemSize=1):
  纯读   BW = numel×1 / t
  读写   BW = 2×numel×1 / t  (转置 W=R)
  numel = D0·d1·d2·d3

用法: python3 parse_b8.py <matrix_dir> [iters]
输出: <matrix_dir>/parsed.json
"""

import csv
import glob
import json
import os
import sys

from prof_pta_matrix import CASES, CAP, ELEM, D0

# B8 用例子集（dtype==2）。
B8_CASES = [c for c in CASES if c[4] == 2]

CFG2MODE = {"db_rw": "db", "db_r": "db", "sb_rw": "sb", "sb_r": "sb"}
CFG2MTE3 = {"db_rw": 1, "db_r": 0, "sb_rw": 1, "sb_r": 0}

# op_summary 流水 ratio 列（AIV 侧）。
RATIO_COLS = {
    "mte2": "aiv_mte2_ratio",
    "mte3": "aiv_mte3_ratio",
    "vec": "aiv_vec_ratio",
    "scalar": "aiv_scalar_ratio",
}


def run_order(mode):
    """该 mode 下按 B8_CASES 顺序过滤容量后的 run 用例（与驱动一致）。block=d2*d3。"""
    cap = CAP[mode]
    return [c for c in B8_CASES if c[2] * c[3] <= cap[c[4]]]


def load_rows(matrix_dir, cfg):
    """读取该配置 op_summary，返回按执行顺序（Task Start Time 升序）的行 dict 列表。"""
    pat = os.path.join(matrix_dir, cfg, "PROF_*", "mindstudio_profiler_output",
                       "op_summary_*.csv")
    files = glob.glob(pat)
    assert len(files) == 1, f"{cfg}: expect 1 op_summary, got {files}"
    rows = []
    with open(files[0]) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: float(r["Task Start Time(us)"].strip()))
    return rows, files[0]


def fnum(r, key):
    v = r.get(key, "").strip()
    try:
        return float(v)
    except ValueError:
        return 0.0


def numel(d1, d2, d3):
    return D0 * d1 * d2 * d3


def parse(matrix_dir, iters):
    per = iters + 1                        # 1 预热 + iters 测量
    result = {}
    csvmap = {}
    for cfg in ("db_rw", "sb_rw", "db_r", "sb_r"):
        mode = CFG2MODE[cfg]
        is_mte3 = CFG2MTE3[cfg]
        order = run_order(mode)
        rows, path = load_rows(matrix_dir, cfg)
        csvmap[cfg] = os.path.relpath(path, matrix_dir)
        exp = len(order) * per
        assert len(rows) == exp, f"{cfg}: {len(rows)} tasks != {len(order)}*{per}={exp}"
        cfg_res = {}
        for i, (lb, d1, d2, d3, dt) in enumerate(order):
            blk = rows[i * per:(i + 1) * per]
            meas = blk[1:]                 # 丢弃预热
            durs = [fnum(r, "Task Duration(us)") for r in meas]
            mn = min(durs)
            mean = sum(durs) / len(durs)
            n = numel(d1, d2, d3)
            R = n * ELEM[dt]               # bytes 读 (elemSize=1)
            factor = 2 if is_mte3 else 1   # 读写=R+W=2R；纯读=R
            bw = factor * R / (mn * 1e-6) / 1e9   # GB/s，min 时延=峰值带宽
            # ratio 取测量段均值（同一 shape 各次一致，均值稳）
            ratios = {}
            for k, col in RATIO_COLS.items():
                vals = [fnum(r, col) for r in meas]
                ratios[k] = sum(vals) / len(vals)
            block_num = int(fnum(meas[0], "Block Num"))
            cfg_res[lb] = {
                "min_us": round(mn, 3), "mean_us": round(mean, 4),
                "bw_gbs": round(bw, 1), "block": d2 * d3, "numel": n,
                "block_num": block_num, "ratios": {k: round(v, 4) for k, v in ratios.items()},
                "shape": [D0, d1, d2, d3],
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
        print(f"{cfg}: {len(res[cfg])} B8 cases")
    print("saved", out)
