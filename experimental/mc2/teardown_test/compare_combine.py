#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_combine.py —— CombineTeardown npu 输出 vs golden 比对。
用法: compare_combine.py <bs> <topk> <h> <E> <R> <datadir>
x(bf16) mae<5e-3 且 cos>0.999（加权求和有 bf16 舍入，非 bit 精确）。
"""
import sys, os
import numpy as np
import ml_dtypes
BF16 = ml_dtypes.bfloat16


def main():
    bs, topk, h, E, R = (int(sys.argv[i]) for i in range(1, 6))
    d = sys.argv[6]
    npu = np.fromfile(os.path.join(d, "x.npu.bin"), dtype=BF16).reshape(bs, h).astype(np.float32)
    gold = np.fromfile(os.path.join(d, "x.golden.bin"), dtype=BF16).reshape(bs, h).astype(np.float32)
    mae = float(np.abs(npu - gold).mean())
    nn, ng = float(np.linalg.norm(npu)), float(np.linalg.norm(gold))
    denom = nn * ng
    # 零范数守卫：仅当两者都≈0 才算相似；一方为 0(如 kernel 静默写全 0)判 0，避免伪「完美」漏判
    if denom > 0:
        cos = float((npu * gold).sum() / denom)
    else:
        cos = 1.0 if (nn == 0.0 and ng == 0.0) else 0.0
    ok = (mae < 5e-3) and (cos > 0.999)
    print(f"  [{ 'PASS' if ok else 'FAIL'}] x mae={mae:.3e} cos={cos:.6f} (n={npu.size})")
    if not ok:
        print(f"   npu row0[:6] ={npu[0,:6].tolist()}")
        print(f"   gold row0[:6]={gold[0,:6].tolist()}")
    print("=== COMBINE_TEARDOWN " + ("PASS" if ok else "FAIL") + " ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
