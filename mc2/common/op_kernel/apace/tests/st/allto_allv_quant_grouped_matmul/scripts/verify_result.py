#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch

# FP8(MXFP8) 量化输入 + BF16 输出。NPU 与 CPU golden 的 K 轴累加顺序不同，
# fp32 累加后再舍入 bf16 会产生少量 1~2 ulp 差异（bf16 ulp 相对值 ~0.4%）。
# 判据：|diff| <= atol + rtol * |golden|（对齐低精度量化算子通用验收门槛）。
ATOL = 1e-2
RTOL = 2e-2
DATA_TYPE = np.uint16


def load_counts(base_dir, rank_id):
    path = os.path.join(base_dir, str(rank_id), "..", "..", "input", str(rank_id), "counts.bin")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        # 兼容直接在 output 目录下找不到 input 的情况
        path = os.path.join(base_dir, str(rank_id), "counts.bin")
    counts = np.fromfile(path, dtype=np.uint32)
    return counts


def verify_single_rank(n, rank_id, base_dir="./output"):
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output file not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden file not found: {golden_path}")

    output = np.fromfile(output_path, dtype=DATA_TYPE)
    golden = np.fromfile(golden_path, dtype=DATA_TYPE)
    if output.size != golden.size:
        raise ValueError(f"[Rank {rank_id}] output size {output.size} != golden size {golden.size}")

    m = golden.size // n
    if m == 0:
        print(f"[PASS] Rank {rank_id}: empty output (0 recv rows), trivially matched")
        return True
    output_t = torch.from_numpy(output.view(np.uint16)).view(torch.bfloat16).reshape(m, n)
    golden_t = torch.from_numpy(golden.view(np.uint16)).view(torch.bfloat16).reshape(m, n)

    g = golden_t.float()
    p = output_t.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    over_tol = (abs_err > (ATOL + RTOL * g.abs())).sum().item()
    print(f"\n[Rank {rank_id}] [verify] rows={m}, n={n}, elements={golden.size}")
    print(f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}")
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {ATOL:g} + {RTOL:g}*|golden|): {over_tol} / {golden.size}")
    print(f"  cpu golden (top-left 4x4):\n{golden_t[:4, :4]}")
    print(f"  npu output (top-left 4x4):\n{output_t[:4, :4]}")

    if over_tol > 0:
        print(f"[FAIL] Rank {rank_id}: {over_tol} elements exceed tolerance {ERROR_TOL}")
        return False
    print(f"[PASS] Rank {rank_id}: NPU results match CPU golden")
    return True


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 verify_result.py n rank_num base_dir")
        sys.exit(1)
    n = int(sys.argv[1])
    rank_num = int(sys.argv[2])
    base_dir = sys.argv[3]

    all_pass = True
    for rank_id in range(rank_num):
        try:
            if not verify_single_rank(n, rank_id, base_dir):
                all_pass = False
        except Exception as e:  # noqa: B902
            print(f"[FAIL] Rank {rank_id}: {e}")
            all_pass = False

    print("\n============================================================")
    print(f"Verification Summary: {'ALL PASS' if all_pass else 'FAILED'}")
    print("============================================================")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
