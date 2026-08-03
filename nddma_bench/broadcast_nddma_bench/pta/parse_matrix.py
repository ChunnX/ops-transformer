# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""解析 DESIGN §9.6 5D 广播用例四配置 msprof op_summary，对齐 MANIFEST5D 执行顺序，
计算时延（min/mean us + aiv_total_cycles）、R_logical/RW 带宽（GB/s）与 PipeUtilization 指标。

RUN/SKIP 与执行顺序完全复用 prof_pta_matrix.run_matrix_5d 两判据 AND（§9.4）：
  tile=dim2*dim3*dim4≤halfCap[dtype] 且 dim4*elemSize%8==0（inner 8B 对齐）才 RUN。
（AB_align_fp16 dim4=6 非对齐 → 恒 SKIP，不发 kernel、不进 op_summary。）

带宽口径（DESIGN §9.6/§7.7，读写不对称，dtype 固定 float32 elemSize=4）：
  numel_dst = dim0*dim1*dim2*dim3*dim4     （5D 满输出，§7.7 满口径）
  R_logical = numel_dst × 4；W = numel_dst × 4
  isMte3=1（读写）→ (R_logical+W)=2×R_logical；isMte3=0（纯复制读）→ R_logical

用法: python3 parse_matrix.py [iters] [matrix_dir]   (iters 默认 5；matrix_dir 默认 ../output/matrix5d)
输出: <matrix_dir>/parsed5d.json + analysis5d.json
"""

import csv
import glob
import json
import os
import sys

from prof_pta_matrix import CASES_5D, CAP5D, SCENE_NAME, D0, ELEM, DT_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# ELEM/DT_NAME 由 prof_pta_matrix 提供（per-dtype elemSize：fp16=2/fp32=4/int8=1，§8.3.3）。

CFG2MODE = {"db_rw": "db", "db_r": "db", "sb_rw": "sb", "sb_r": "sb"}
CFG2MTE3 = {"db_rw": 1, "db_r": 0, "sb_rw": 1, "sb_r": 0}
FIELDS = ["aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_scalar_ratio"]
KERNEL_OP_NAME = "BroadcastNDDMABenchKernel"


def run_order(mode):
    """返回该 mode 下的 RUN 用例（两判据 AND：tile=d2*d3*d4≤cap[dtype] 且 dim4*elem%8==0），
    顺序与 run_matrix_5d 完全一致。每项 (label, scene, d2, d3, d4, dim1, dt, tile)。cap 随 dtype（§9.4）。"""
    cap = CAP5D[mode]
    run = []
    for lb, scene, d2, d3, d4, dim1, dt in CASES_5D:
        tile = d2 * d3 * d4
        if tile <= cap[dt] and (d4 * ELEM[dt]) % 8 == 0:
            run.append((lb, scene, d2, d3, d4, dim1, dt, tile))
    return run


def load_rows(cfg, matrix):
    pat = os.path.join(matrix, cfg, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    files = glob.glob(pat)
    assert len(files) == 1, f"{cfg}: expect 1 op_summary, got {files}"
    rows = []
    with open(files[0]) as f:
        for r in csv.DictReader(f):
            if r.get("Op Name", "").strip() != KERNEL_OP_NAME:
                continue  # 跳过输入组装引入的 aclnn 辅助算子
            row = {
                "start": float(r["Task Start Time(us)"].strip()),
                "dur": float(r["Task Duration(us)"].strip()),
                "block": int(float(r["Block Num"].strip())),
                "aiv_cycles": int(float(r["aiv_total_cycles"].strip()))
                if r.get("aiv_total_cycles", "").strip() else 0,
            }
            for fld in FIELDS:
                row[fld] = float(r[fld].strip()) if fld in r and r[fld].strip() else 0.0
            rows.append(row)
    rows.sort(key=lambda x: x["start"])  # 执行顺序 = Task Start Time 升序
    return rows


def numel_dst(d2, d3, d4, dim1):
    return D0 * dim1 * d2 * d3 * d4  # 满口径 dim0*dim1*dim2*dim3*dim4


def parse(iters, matrix):
    per = iters + 1  # 1 预热 + iters 测量
    parsed = {}
    analysis = {}
    for cfg in ("db_rw", "sb_rw", "db_r", "sb_r"):
        mode = CFG2MODE[cfg]
        is_mte3 = CFG2MTE3[cfg]
        order = run_order(mode)
        rows = load_rows(cfg, matrix)
        exp = len(order) * per
        assert len(rows) == exp, f"{cfg}: {len(rows)} tasks != {len(order)}*{per}={exp}"
        cfg_parsed = {}
        cases = {}
        agg = {fld: [] for fld in FIELDS}
        blocks = set()
        for i, (lb, scene, d2, d3, d4, dim1, dt, tile) in enumerate(order):
            blk = rows[i * per:(i + 1) * per][1:]  # drop warmup
            durs = [x["dur"] for x in blk]
            mn = min(durs)
            mean = sum(durs) / len(durs)
            cyc = blk[durs.index(mn)]["aiv_cycles"]  # cycle 取 min 时延对应测量
            n = numel_dst(d2, d3, d4, dim1)
            R = n * ELEM[dt]                          # per-dtype elemSize（§8.3.3）
            factor = 2 if is_mte3 else 1
            bw = factor * R / (mn * 1e-6) / 1e9  # GB/s（min 时延 → 峰值带宽）
            ratios = {fld: sum(x[fld] for x in blk) / len(blk) for fld in FIELDS}
            for fld in FIELDS:
                agg[fld].append(ratios[fld])
            for x in blk:
                blocks.add(x["block"])
            cfg_parsed[lb] = [round(mn, 3), round(mean, 4), round(bw, 2)]
            cases[lb] = {
                "scene": SCENE_NAME[scene], "dtype": DT_NAME[dt], "elem": ELEM[dt],
                "d2": d2, "d3": d3, "d4": d4, "dim1": dim1,
                "tile": tile, "dstBlkFull": d2 * d3 * d4,
                "numel_dst": n, "Rlog_MB": round(R / 1e6, 2),
                "min": round(mn, 3), "mean": round(mean, 3), "bw": round(bw, 2),
                "cycles": cyc,
                **{fld: round(ratios[fld], 4) for fld in FIELDS},
            }
        parsed[cfg] = cfg_parsed
        stat = {fld: {"min": round(min(agg[fld]), 4),
                      "mean": round(sum(agg[fld]) / len(agg[fld]), 4),
                      "max": round(max(agg[fld]), 4)} for fld in FIELDS}
        all_min = [c["min"] for c in cases.values()]
        all_bw = [(lb, c["bw"]) for lb, c in cases.items()]
        peak = max(all_bw, key=lambda x: x[1])
        analysis[cfg] = {
            "n": len(order), "blocks": sorted(blocks), "stat": stat,
            "lat_min": round(min(all_min), 3), "lat_mean": round(sum(all_min) / len(all_min), 3),
            "lat_max": round(max(all_min), 3),
            "peak_bw": peak[1], "peak_case": peak[0],
            "under40": [(lb, c["min"]) for lb, c in cases.items() if c["min"] <= 40.0],
            "cases": cases,
        }
    return parsed, analysis


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    matrix = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(ROOT, "output", "matrix5d")
    parsed, analysis = parse(iters, matrix)
    json.dump(parsed, open(os.path.join(matrix, "parsed5d.json"), "w"), ensure_ascii=False)
    json.dump(analysis, open(os.path.join(matrix, "analysis5d.json"), "w"), ensure_ascii=False, indent=1)
    for cfg in ("db_rw", "sb_rw", "db_r", "sb_r"):
        r = analysis[cfg]
        s = r["stat"]
        print(f"\n[{cfg}] {r['n']} RUN, Block Num={r['blocks']}")
        print(f"  aiv_mte2 min/mean/max = {s['aiv_mte2_ratio']['min']}/{s['aiv_mte2_ratio']['mean']}/{s['aiv_mte2_ratio']['max']}")
        print(f"  aiv_mte3 min/mean/max = {s['aiv_mte3_ratio']['min']}/{s['aiv_mte3_ratio']['mean']}/{s['aiv_mte3_ratio']['max']}")
        print(f"  aiv_vec  mean = {s['aiv_vec_ratio']['mean']}   aiv_scalar mean = {s['aiv_scalar_ratio']['mean']}")
        print(f"  时延 min/mean/max = {r['lat_min']}/{r['lat_mean']}/{r['lat_max']} us")
        print(f"  峰值 BW = {r['peak_bw']} GB/s ({r['peak_case']})")
        print(f"  min<=40us 用例数 = {len(r['under40'])}: {r['under40'][:12]}")
