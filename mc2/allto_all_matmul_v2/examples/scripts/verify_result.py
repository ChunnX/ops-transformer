#!/usr/bin/python3
# coding=utf-8
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# ------------------------------------------------------------------------------
"""Precision verification for AlltoAllMatmulV2 aclnn test.

Adapted from mc2/common/op_kernel/apace/tests/st/all_to_all_quant_matmul/scripts/verify_result.py
"""

import os
import sys

import numpy as np
import torch

ERROR_TOL = 1e-2
DATA_TYPE = np.uint16
FULL_PRINT_MAX = 1024
CORNER = 4


def verify_single_rank(m, n, rank_id, base_dir="./output"):
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden not found: {golden_path}")

    output = np.fromfile(output_path, dtype=DATA_TYPE)
    golden = np.fromfile(golden_path, dtype=DATA_TYPE)

    expected = m * n
    if output.size != expected:
        raise ValueError(f"Rank {rank_id}: npu output size {output.size} != {expected}")
    if golden.size != expected:
        raise ValueError(f"Rank {rank_id}: cpu golden size {golden.size} != {expected}")

    npu_t = torch.from_numpy(output).view(torch.bfloat16).reshape(m, n)
    gold_t = torch.from_numpy(golden).view(torch.bfloat16).reshape(m, n)

    numel = m * n
    if numel <= FULL_PRINT_MAX:
        print(f"\n[Rank {rank_id}] cpu golden:\n", gold_t)
        print(f"[Rank {rank_id}] npu output:\n", npu_t)
    else:
        g = gold_t.float()
        p = npu_t.float()
        diff = p - g
        abs_err = diff.abs()
        rmse = diff.pow(2).mean().sqrt()
        denom = g.abs().clamp_min(1e-8)
        rel_err = abs_err / denom
        over_tol = (abs_err > ERROR_TOL).sum().item()
        print(f"\n[Rank {rank_id}] shape=({m},{n}) elements={numel}")
        print(
            f"  abs_err: max={abs_err.max().item():.6e} mean={abs_err.mean().item():.6e} rmse={rmse.item():.6e}"
        )
        print(f"  rel_err: max={rel_err.max().item():.6e}")
        print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol}/{numel}")
        cr, cc = min(CORNER, m), min(CORNER, n)
        if cr > 0 and cc > 0:
            print(f"  cpu[:{cr},:{cc}]:\n{gold_t[:cr, :cc]}")
            print(f"  npu[:{cr},:{cc}]:\n{npu_t[:cr, :cc]}")

    return torch.allclose(gold_t, npu_t, rtol=ERROR_TOL, atol=ERROR_TOL, equal_nan=True)


def verify_result(m, n, rank_num, base_dir="./output"):
    all_pass = True
    print(f"\n{'=' * 60}")
    print(f"Verification: {rank_num} ranks, M_per_rank={m}, N={n}")
    print(f"{'=' * 60}")

    for r in range(rank_num):
        try:
            ok = verify_single_rank(m, n, r, base_dir)
            if ok:
                print(f"[PASS] Rank {r}")
            else:
                print(f"[FAIL] Rank {r}: allclose returned False")
                all_pass = False
        except Exception as e:
            print(f"[FAIL] Rank {r}: {e}")
            all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print("[ALL PASS] All ranks passed verification!")
    else:
        print("[FAIL] Some ranks differ from golden.")
    return all_pass


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 verify_result.py m n rank_num [base_dir]")
        sys.exit(1)
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    rank_num = int(sys.argv[3])
    base_dir = sys.argv[4] if len(sys.argv) > 4 else "./output"
    try:
        ok = verify_result(m, n, rank_num, base_dir)
        if not ok:
            raise SystemExit(1)
    except Exception as e:
        print(e)
        sys.exit(1)
