#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# quantMatmulReduceScatter 精度比对
#
# 每卡输出: [m/rankSize, n] bfloat16
# ----------------------------------------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch

ERROR_TOL = 1e-2
ATOL_TOL = 1e4  # BF16 大数值绝对误差容忍 (2M * 0.4% = 8192)
DATA_TYPE = np.uint16


def verify_single_rank(chunk_m, n, rank_id, base_dir="./output"):
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output file not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden file not found: {golden_path}")

    output = np.fromfile(output_path, dtype=DATA_TYPE)
    golden = np.fromfile(golden_path, dtype=DATA_TYPE)

    expected_size = chunk_m * n
    if output.size != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] npu output size {output.size} != expected size {expected_size}"
        )
    if golden.size != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] cpu output size {golden.size} != expected size {expected_size}"
        )

    npu_output_tensor = (
        torch.from_numpy(output).view(torch.bfloat16).reshape(chunk_m, n)
    )
    golden_tensor = torch.from_numpy(golden).view(torch.bfloat16).reshape(chunk_m, n)

    g = golden_tensor.float()
    p = npu_output_tensor.float()
    abs_err = (p - g).abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    over_tol = (abs_err > ERROR_TOL).sum().item()
    print(f"\n[Rank {rank_id}] shape=({chunk_m}, {n}), elements={expected_size}")
    print(
        f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}"
    )
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol} / {expected_size}")
    print(f"  cpu golden (top-left 4x4):\n{golden_tensor[:4, :4]}")
    print(f"  npu output (top-left 4x4):\n{npu_output_tensor[:4, :4]}")

    return torch.allclose(
        golden_tensor, npu_output_tensor, rtol=ERROR_TOL, atol=ATOL_TOL, equal_nan=True
    )


def verify_result(m, n, rank_num, base_dir="./output"):
    chunk_m = m // rank_num
    all_pass = True
    results = []

    print(f"\n{'=' * 60}")
    print(f"Verifying outputs for {rank_num} ranks")
    print(f"Matrix shape: M={m}, N={n}, chunkM={chunk_m}")
    print(f"{'=' * 60}")

    for rank_id in range(rank_num):
        try:
            res = verify_single_rank(chunk_m, n, rank_id, base_dir)
            results.append(
                (rank_id, bool(res), None if res else "allclose returned False")
            )
            if res:
                print(f"[PASS] Rank {rank_id}: NPU results match CPU golden")
            else:
                print(
                    f"[FAIL] Rank {rank_id}: allclose returned False (rtol=atol={ERROR_TOL})"
                )
                all_pass = False
        except Exception as e:
            results.append((rank_id, False, str(e)))
            print(f"[FAIL] Rank {rank_id}: {e}")
            all_pass = False

    print(f"\n{'=' * 60}")
    print("Verification Summary:")
    for rank_id, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"  Rank {rank_id}: {status}")
        if error:
            print(f"    Error: {error}")

    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify QuantMatmulReduceScatter output"
    )
    parser.add_argument("m", type=int, help="matrix M dimension")
    parser.add_argument("n", type=int, help="matrix N dimension")
    parser.add_argument("rank_num", type=int, help="number of ranks")
    parser.add_argument(
        "base_dir", nargs="?", default="./output", help="output directory"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Silent mode: print only PASS or FAIL and exit",
    )
    args = parser.parse_args()

    m, n, rank_num, base_dir = args.m, args.n, args.rank_num, args.base_dir

    if m % rank_num != 0:
        print(f"Error: m={m} is not divisible by rank_num={rank_num}")
        sys.exit(1)

    try:
        if args.check:
            # Quiet mode for run.sh --perf — only print PASS or FAIL
            import io
            from contextlib import redirect_stdout, redirect_stderr

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    all_pass = verify_result(m, n, rank_num, base_dir)
                except Exception:
                    all_pass = False
            print("PASS" if all_pass else "FAIL")
            sys.exit(0 if all_pass else 1)
        else:
            all_pass = verify_result(m, n, rank_num, base_dir)
            if not all_pass:
                raise ValueError("[ERROR] Some NPU results differ from CPU.\n")
            print(f"\n[ALL PASS] All {rank_num} ranks passed verification!\n")
            sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
