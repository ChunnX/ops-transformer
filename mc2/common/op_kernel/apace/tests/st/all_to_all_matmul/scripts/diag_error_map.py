#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""非量化 a2a_mm ST 精度错误定位工具。

在 run.sh 执行完（output/{r}/npu_out.bin 与 cpu_output.bin 就位）后运行：

  python3 scripts/diag_error_map.py <m> <n> <rank_num> [--base ./output] [--bm 256] [--bn 256] \
      [--old <快照目录>] [--expect <期望均值>]

输出：
  1. NaN/inf 统计（verify 只显示 max=nan，不给数量）
  2. allclose 真实超差 vs 1~2 ulp 良性舍入（verify 的 |err|>0.01 会误计 ulp 噪音）
  3. 按 baseM x baseN 块网格的错误/NaN 分布（定位错误空间结构）
  4. 超差样本（npu/golden/比值——区分缺贡献/垃圾/舍入）
  5. --old：与上一次运行的输出快照比对（确定性检验：垃圾位两次相同=未写入残留，不同=竞争写入）
"""

import argparse
import os

import numpy as np


def load(base, rank, m, n):
    out = np.fromfile(os.path.join(base, str(rank), "npu_out.bin"), dtype=np.float16)
    gold = np.fromfile(os.path.join(base, str(rank), "cpu_output.bin"), dtype=np.float16)
    if out.size != m * n or gold.size != m * n:
        raise ValueError(f"[Rank {rank}] 文件大小不符: npu={out.size} golden={gold.size} 期望={m * n}")
    return out.reshape(m, n).astype(np.float32), gold.reshape(m, n).astype(np.float32)


def block_grid(mask, m, n, bm, bn, title):
    if m % bm != 0 or n % bn != 0:
        print(f"  ({title}: {m}x{n} 不能按 {bm}x{bn} 整除，跳过块网格)")
        return None
    grid = mask.reshape(m // bm, bm, n // bn, bn).sum(axis=(1, 3))
    print(f"  {title} 按 {bm}x{bn} 块分布 (行=m块, 列=n块):")
    print(grid)
    print(f"  行块合计: {grid.sum(axis=1).tolist()}")
    print(f"  列块合计: {grid.sum(axis=0).tolist()}")
    return grid


def analyze_rank(rank, out, gold, m, n, bm, bn, expect, old_out=None):
    diff = out - gold
    nan_o = np.isnan(out)
    bad = np.abs(diff) > (0.01 + 0.01 * np.abs(gold))  # allclose 真实判据（NaN 不计入）
    rnd = (np.abs(diff) > 0) & (np.abs(diff) <= 2.0)  # 1~2 ulp 良性舍入

    print(f"===== Rank {rank} =====")
    print(f"  NPU输出: NaN={nan_o.sum()}  +inf={(out == np.inf).sum()}  -inf={(out == -np.inf).sum()}")
    print(f"  golden : NaN={np.isnan(gold).sum()}  inf={np.isinf(gold).sum()}")
    print(f"  allclose 真实超差: {bad.sum()}/{m * n} = {bad.sum() * 100.0 / (m * n):.3f}%"
          f"   (1~2ulp 舍入: {rnd.sum()})")
    d = np.abs(diff[bad])
    if d.size:
        print(f"  超差|diff|: min={d.min():.1f} p50={np.percentile(d, 50):.1f} max={d.max():.1f}")

    block_grid(bad, m, n, bm, bn, "超差")
    block_grid(nan_o, m, n, bm, bn, "NaN")

    for i in np.argwhere(bad)[:6]:
        rr, cc = int(i[0]), int(i[1])
        ratio = out[rr, cc] / gold[rr, cc] if gold[rr, cc] != 0 else float("nan")
        print(f"  超差样本[{rr},{cc}]: npu={out[rr, cc]:.2f} gold={gold[rr, cc]:.2f} 比值={ratio:.4f}")
    nidx = np.argwhere(nan_o)
    if nidx.size:
        print(f"  NaN 行范围: [{nidx[:, 0].min()},{nidx[:, 0].max()}]  "
              f"列范围: [{nidx[:, 1].min()},{nidx[:, 1].max()}]")

    if old_out is not None:
        same = (out == old_out) | (np.isnan(out) & np.isnan(old_out))
        print(f"  [确定性] 与上次运行相同: {same.sum()}/{m * n} = {same.sum() * 100.0 / (m * n):.1f}%")
        if expect is not None:
            garbage = np.abs(out - expect) > 25  # 极端垃圾位（NaN 比较为 False，不计入）
            print(f"  [确定性] 极端垃圾位({garbage.sum()}个)中两次相同的: "
                  f"{same[garbage].sum()}/{garbage.sum()} = "
                  f"{same[garbage].sum() * 100.0 / max(garbage.sum(), 1):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="a2a_mm ST 精度错误定位")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("rank_num", type=int)
    parser.add_argument("--base", default="./output", help="output 目录（含 {r}/npu_out.bin 与 cpu_output.bin）")
    parser.add_argument("--bm", type=int, default=256, help="块网格 M 粒度（baseM）")
    parser.add_argument("--bn", type=int, default=256, help="块网格 N 粒度（baseN）")
    parser.add_argument("--old", default=None, help="上一次运行的 output 快照目录（确定性比对）")
    parser.add_argument("--expect", type=float, default=None, help="输出期望量级（确定性垃圾位判定阈值用）")
    args = parser.parse_args()

    for r in range(args.rank_num):
        out, gold = load(args.base, r, args.m, args.n)
        old_out = None
        if args.old is not None:
            old_out, _ = load(args.old, r, args.m, args.n)
        analyze_rank(r, out, gold, args.m, args.n, args.bm, args.bn, args.expect, old_out)


if __name__ == "__main__":
    main()
