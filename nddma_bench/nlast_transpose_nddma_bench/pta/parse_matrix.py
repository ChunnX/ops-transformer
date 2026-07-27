# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""解析 4 配置 msprof op_summary，对齐 MANIFEST 执行顺序，计算 NLastTranspose 时延/带宽矩阵。

每个用例在会话内跑 (1 预热 + N 测量) 个 kernel task；op_summary 按 Task Start Time 升序，
即执行顺序，与 prof_pta_matrix.py 打印的 RUN manifest 一一对齐。
取每用例 N 个测量 task 的 min（最优）与 mean 时延，带宽由 numel×elemSize 推出。

用法: python3 parse_matrix.py [iters]   (iters 默认 5)
输出: ../output/matrix/parsed.json
"""

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATRIX = os.path.join(ROOT, "output", "matrix")

ELEM = {0: 2, 1: 4}      # fp16 / fp32 elem bytes
# 分核维 dim0=56（= 物理 vector core 数，单波均衡 per=1/rem=0），固定不变。
# 带宽 numel = 56 × dim1_i × d2×d3×d4（dim1 逐用例，round_004 引入）。
D0 = 56

# (label, d2, d3, d4, dtype, dim1) —— 与 prof_pta_matrix.CASES 同源（含 per-case dim1）。
# dim1 逐用例：round_003 min 时延 <50us 的用例已放大 dim1，min ≥50us 的保持 dim1=8。
CASES = [
    ("NL_0", 128, 8, 16, 0, 32), ("NL_1", 128, 8, 32, 0, 24), ("NL_2", 64, 8, 64, 0, 24),
    ("NL_3", 64, 8, 128, 0, 8), ("NL_4", 32, 8, 256, 0, 8), ("NL_5", 16, 8, 512, 0, 8),
    ("NL_6", 128, 8, 15, 0, 32), ("NL_7", 128, 8, 31, 0, 16), ("NL_8", 64, 8, 63, 0, 16),
    ("NL_9", 64, 8, 127, 0, 8), ("NL_10", 32, 8, 255, 0, 8), ("NL_11", 16, 8, 511, 0, 8),
    ("NL_12", 8, 128, 16, 0, 24), ("NL_13", 8, 128, 32, 0, 24), ("NL_14", 8, 64, 64, 0, 24),
    ("NL_15", 8, 64, 128, 0, 8), ("NL_16", 8, 32, 256, 0, 8), ("NL_17", 8, 16, 512, 0, 8),
    ("NL_18", 8, 128, 15, 0, 32), ("NL_19", 8, 128, 31, 0, 24), ("NL_20", 8, 64, 63, 0, 24),
    ("NL_21", 8, 64, 127, 0, 16), ("NL_22", 8, 32, 255, 0, 8), ("NL_23", 8, 16, 511, 0, 8),
    ("NL_24", 128, 8, 16, 1, 24), ("NL_25", 128, 8, 32, 1, 24), ("NL_26", 64, 8, 64, 1, 24),
    ("NL_27", 64, 8, 128, 1, 8), ("NL_28", 32, 8, 256, 1, 8), ("NL_29", 16, 8, 512, 1, 8),
    ("NL_30", 128, 8, 15, 1, 16), ("NL_31", 128, 8, 31, 1, 8), ("NL_32", 64, 8, 63, 1, 16),
    ("NL_33", 64, 8, 127, 1, 8), ("NL_34", 32, 8, 255, 1, 8), ("NL_35", 16, 8, 511, 1, 8),
    ("NL_36", 8, 128, 16, 1, 16), ("NL_37", 8, 128, 32, 1, 24), ("NL_38", 8, 64, 64, 1, 24),
    ("NL_39", 8, 64, 128, 1, 8), ("NL_40", 8, 32, 256, 1, 8), ("NL_41", 8, 16, 512, 1, 8),
    ("NL_42", 8, 128, 15, 1, 32), ("NL_43", 8, 128, 31, 1, 16), ("NL_44", 8, 64, 63, 1, 16),
    ("NL_45", 8, 64, 127, 1, 8), ("NL_46", 8, 32, 255, 1, 8), ("NL_47", 8, 16, 511, 1, 8),
]

CAP = {
    "db": {0: 65536, 1: 16384},
    "sb": {0: 131072, 1: 32768},
}

CFG2MODE = {"db_rw": "db", "db_r": "db", "sb_rw": "sb", "sb_r": "sb"}
CFG2MTE3 = {"db_rw": 1, "db_r": 0, "sb_rw": 1, "sb_r": 0}


def run_order(mode):
    """返回该 mode 下按 CASES 顺序过滤容量后的 run 用例（与驱动一致）。"""
    cap = CAP[mode]
    return [c for c in CASES if c[1] * c[2] * c[3] <= cap[c[4]]]


def load_durations(cfg):
    """读取该配置 op_summary，返回按执行顺序（Task Start Time 升序）的 Task Duration 列表。"""
    pat = os.path.join(MATRIX, cfg, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    files = glob.glob(pat)
    assert len(files) == 1, f"{cfg}: expect 1 op_summary, got {files}"
    rows = []
    with open(files[0]) as f:
        for r in csv.DictReader(f):
            start = r["Task Start Time(us)"].strip()
            dur = r["Task Duration(us)"].strip()
            rows.append((float(start), float(dur)))
    rows.sort(key=lambda x: x[0])          # 执行顺序
    return [d for _, d in rows]


def numel(d1, d2, d3, d4):
    return D0 * d1 * d2 * d3 * d4


def parse(iters):
    per = iters + 1                        # 1 预热 + iters 测量
    result = {}
    for cfg in ("db_rw", "sb_rw", "db_r", "sb_r"):
        mode = CFG2MODE[cfg]
        is_mte3 = CFG2MTE3[cfg]
        order = run_order(mode)
        durs = load_durations(cfg)
        exp = len(order) * per
        assert len(durs) == exp, f"{cfg}: {len(durs)} tasks != {len(order)}*{per}={exp}"
        cfg_res = {}
        for i, (lb, d2, d3, d4, dt, d1) in enumerate(order):
            block = durs[i * per: (i + 1) * per]
            meas = block[1:]               # 丢弃预热
            mn = min(meas)
            mean = sum(meas) / len(meas)
            n = numel(d1, d2, d3, d4)
            R = n * ELEM[dt]               # bytes 读
            factor = 2 if is_mte3 else 1   # 读写=R+W=2R；纯读=R
            bw = factor * R / (mn * 1e-6) / 1e9   # GB/s（min 时延对应峰值带宽）
            cfg_res[lb] = [round(mn, 3), round(mean, 4), bw]
        result[cfg] = cfg_res
    return result


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    res = parse(iters)
    out = os.path.join(MATRIX, "parsed.json")
    json.dump(res, open(out, "w"), ensure_ascii=False)
    for cfg in ("db_rw", "sb_rw", "db_r", "sb_r"):
        print(f"{cfg}: {len(res[cfg])} cases")
    print("saved", out)
