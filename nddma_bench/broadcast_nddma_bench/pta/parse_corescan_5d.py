# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""解析 DESIGN §7.6 5D 核数扫描 msprof op_summary，对齐 MANIFEST5D_CORESCAN 执行顺序，
输出每 (case, blockNum) 的时延（min us）、带宽（GB/s，§7.6b 口径）与不均衡度。

执行顺序（完全复用 prof_pta_matrix.run_corescan_5d）：
  for lb in CORESCAN_LABELS(AB_0/BA_0/BAB_0/ABA_0):
    for bn in CORESCAN_BLOCKNUMS(1,2,4,8,16,28,56):
      (1 预热 + iters 测量)
  1 核护栏（§9.6/§7.6a）：bn==1 时若 est_us=dim0*dim1*t_nddma > T_max(缺省5ms) 自动降 dim1
  （3D 方案下每拍一条 3D NDDMA、仅 dim1 for，无 dim2-for，故无 dim2eff 因子）；
  故 numel_dst 用护栏后的 eff_dim1 计算（与驱动逐字节一致）。

带宽口径（§7.6b/§7.7，float32 elemSize=4）：
  分母 t = 最慢核时延（Task Duration 即整卡 kernel 墙钟=最慢核）——不取均值，如实反映尾核拖累。
  分子 numel_dst = dim0*eff_dim1*dim2*dim3*dim4（满输出，与核数无关，保证跨核数曲线分子一致可比）。
  imbalance = ceil(numBlocks/bn)/floor(numBlocks/bn)（尾核块数/常核块数；bn>numBlocks 记 inf）。
  isMte3=1 → BW=2*R/t；isMte3=0 → BW=R/t。

用法: python3 parse_corescan_5d.py <cfg_dir> <isMte3:0|1> [t_nddma_us] [T_max_ms] [iters]
输出: <cfg_dir 上级>/corescan5d_<cfg>.json + 打印曲线表
"""

import csv
import glob
import json
import math
import os
import sys

from prof_pta_matrix import (CASES_5D, SCENE_NAME, D0,
                             CORESCAN_LABELS, CORESCAN_BLOCKNUMS)

ELEM = 4
KERNEL_OP_NAME = "BroadcastNDDMABenchKernel"
FIELDS = ["aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_scalar_ratio"]


def eff_dim1_for(scene, d2, dim1, bn, t_nddma_us, t_max_ms):
    """复现 run_corescan_5d 的 1 核护栏：bn==1 且 est>T_max 时降 dim1。
    3D 方案：est=D0*dim1*t_nddma（每拍一条 3D NDDMA，仅 dim1 for，无 dim2eff 因子）。"""
    if bn != 1:
        return dim1
    est_us = D0 * dim1 * t_nddma_us
    if est_us > t_max_ms * 1000:
        new_dim1 = max(1, int((t_max_ms * 1000) // (D0 * t_nddma_us)))
        if new_dim1 < dim1:
            return new_dim1
    return dim1


def build_order(t_nddma_us, t_max_ms):
    # CORESCAN_LABELS 全为 fp32 代表 case（dt=1），ELEM=4 恒成立；dt 仅用于解包对齐 7 元组。
    reps = {lb: (sc, d2, d3, d4, dim1) for lb, sc, d2, d3, d4, dim1, dt in CASES_5D}
    order = []
    for lb in CORESCAN_LABELS:
        sc, d2, d3, d4, dim1 = reps[lb]
        for bn in CORESCAN_BLOCKNUMS:
            edim1 = eff_dim1_for(sc, d2, dim1, bn, t_nddma_us, t_max_ms)
            per = D0 // bn
            rem = D0 % bn
            ceil_b = per + (1 if rem else 0)
            imbalance = (ceil_b / per) if per > 0 else float("inf")
            order.append((lb, sc, d2, d3, d4, edim1, bn, imbalance))
    return order


def load_rows(cfg_dir):
    pat = os.path.join(cfg_dir, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    files = glob.glob(pat)
    assert len(files) == 1, f"expect 1 op_summary, got {files}"
    rows = []
    with open(files[0]) as f:
        for r in csv.DictReader(f):
            if r.get("Op Name", "").strip() != KERNEL_OP_NAME:
                continue
            row = {
                "start": float(r["Task Start Time(us)"].strip()),
                "dur": float(r["Task Duration(us)"].strip()),
                "block": int(float(r["Block Num"].strip())),
            }
            for fld in FIELDS:
                row[fld] = float(r[fld].strip()) if fld in r and r[fld].strip() else 0.0
            rows.append(row)
    rows.sort(key=lambda x: x["start"])
    return rows


def main():
    cfg_dir = sys.argv[1]
    is_mte3 = int(sys.argv[2])
    t_nddma_us = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
    t_max_ms = float(sys.argv[4]) if len(sys.argv) >= 5 else 5.0
    iters = int(sys.argv[5]) if len(sys.argv) >= 6 else 5
    per = iters + 1
    order = build_order(t_nddma_us, t_max_ms)
    rows = load_rows(cfg_dir)
    exp = len(order) * per
    assert len(rows) == exp, f"{len(rows)} tasks != {len(order)}*{per}={exp}"
    factor = 2 if is_mte3 else 1

    curves = {lb: [] for lb in CORESCAN_LABELS}
    for i, (lb, sc, d2, d3, d4, edim1, bn, imbalance) in enumerate(order):
        blk = rows[i * per:(i + 1) * per][1:]
        durs = [x["dur"] for x in blk]
        mn = min(durs)
        numel = D0 * edim1 * d2 * d3 * d4
        R = numel * ELEM
        bw = factor * R / (mn * 1e-6) / 1e9
        vec = sum(x["aiv_vec_ratio"] for x in blk) / len(blk)
        mte2 = sum(x["aiv_mte2_ratio"] for x in blk) / len(blk)
        blkn = blk[0]["block"]
        curves[lb].append({
            "blockNum": bn, "block_reported": blkn, "dim1": edim1,
            "numel_dst": numel, "min_us": round(mn, 3), "bw": round(bw, 2),
            "imbalance": round(imbalance, 3) if imbalance != float("inf") else "inf",
            "vec": round(vec, 4), "mte2": round(mte2, 4),
        })

    cfg = os.path.basename(cfg_dir.rstrip("/"))
    out = os.path.join(os.path.dirname(cfg_dir.rstrip("/")), f"corescan5d_{cfg}.json")
    json.dump(curves, open(out, "w"), ensure_ascii=False, indent=1)

    print(f"# corescan5d cfg={cfg} isMte3={is_mte3} factor={factor} "
          f"t_nddma={t_nddma_us}us T_max={t_max_ms}ms")
    for lb in CORESCAN_LABELS:
        sc = {l: s for l, s, *_ in CASES_5D}[lb]
        print(f"\n[{lb}] scene={SCENE_NAME[sc]}")
        print(f"  {'blockNum':>8} {'blk_rep':>7} {'dim1':>5} {'min_us':>9} {'BW_GB/s':>9} {'imbal':>6} {'vec':>5} {'mte2':>5}")
        for p in curves[lb]:
            print(f"  {p['blockNum']:>8} {p['block_reported']:>7} {p['dim1']:>5} "
                  f"{p['min_us']:>9} {p['bw']:>9} {str(p['imbalance']):>6} {p['vec']:>5} {p['mte2']:>5}")
    print(f"\n# written {out}")


if __name__ == "__main__":
    main()
