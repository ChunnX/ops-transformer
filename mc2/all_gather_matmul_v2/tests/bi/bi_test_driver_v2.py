#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Batch-invariance test driver V2 for npu_all_gather_quant_mm (AllGatherMatmulV2).

V2 改进点（相对 V1 bi_test_driver.py）：
  1. Cross-rank BI：各 rank 使用 rank-specific seed 生成**不同** x1 数据，
     AllGather 后全局输入各 rank 一致，比较 full output byte-exact 一致性。
     （V1 所有 rank 输入相同，实际测的是确定性而非 BI）
  2. Row-BI：固定 target_row 数据，变化其他行，比较 target_row 输出是否不变。
     （V1 用不同 seed 改变了 target_row 本身，测试逻辑错误）
  3. 覆盖多个 target_row 位置（首行、末行），而非仅 row 0。
  4. x2_scale 始终 varying（触发 BI 差异的关键条件）。
  5. Warmup 后加 dist.barrier() 稳定 HCCL state。

Two independent BI dimensions:
  A) **Cross-rank BI**: 同输入下所有 rank 的 y 输出必须 byte-exact 一致
     （issue #2872 报告的 LOCAL/GATHER 双路径 divergence）
  B) **Row-BI (literature BI)**: y[i] = f(x[i], W)，row i 输出只取决于 row i + 共享权重

Methodology:
  Cross-rank BI:
    1. 各 rank 使用 rank-specific seed 生成不同的 x1 数据
    2. AllGather 后所有 rank 看到相同的全局输入 [x1_r0; x1_r1; ...]
    3. 每个 rank 独立计算 local output (M_per_rank rows)
    4. 通过 all_gather 收集所有 rank 的 local output，重建 full Y
    5. 比较所有 rank 的 full Y 是否 byte-exact 一致
    6. 多次 trial 验证确定性

  Row-BI:
    1. 固定 target_row 数据（所有 trial 相同）
    2. 变化其他行数据（不同 trial 使用不同 seed）
    3. 比较 target_row 的输出是否不变
    4. 在多个 target_row 位置测试（首行、末行）
"""
import os
import sys
import hashlib
import torch
import torch_npu
import torch.distributed as dist


BLOCK_SIZE = 0
N_WARMUP = 3
N_TRIALS = 3


def sha_row(t, row_idx):
    return hashlib.sha256(t[row_idx].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def sha_full(t):
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def build_scale_varying(shape):
    bs = torch.empty(shape, dtype=torch.uint8)
    ndim = len(shape)
    if ndim == 3:
        d0, d1, _ = shape
        for i in range(d0):
            for j in range(d1):
                for slot in (0, 1):
                    bs[i, j, slot] = 0x78 + ((i + j * 2 + slot) % 8)
    return bs


def run_op(dev, hcom, ws, M_per_rank, K, N, seed, rank,
           target_row=None, target_row_bytes=None, x1_scale_varying=False):
    """Run npu_all_gather_quant_mm.

    x1: 各 rank 使用 (seed * ws + rank) 生成不同数据，AllGather 后全局输入各 rank 一致。
    x2: 所有 rank 相同（固定 seed 9000）。
    x2_scale: 始终 varying（触发 BI 差异的关键条件）。
    x1_scale: 由 x1_scale_varying 控制。
    """
    n_pack = (K + 63) // 64

    g_w = torch.Generator()
    g_w.manual_seed(9000)
    x2_bytes = torch.randint(0x10, 0x50, (N, K), dtype=torch.uint8, generator=g_w)
    x2 = x2_bytes.to(dev).reshape(N, K).view(torch.float8_e4m3fn)
    x2s_bytes = build_scale_varying((N, n_pack, 2))
    x2s = x2s_bytes.to(dev).reshape(N, n_pack, 2).view(torch.float8_e8m0fnu)

    g_o = torch.Generator()
    g_o.manual_seed(seed * ws + rank)
    x1_bytes = torch.randint(0x10, 0x50, (M_per_rank, K), dtype=torch.uint8, generator=g_o)
    if target_row is not None and target_row_bytes is not None and target_row < M_per_rank:
        x1_bytes[target_row] = target_row_bytes
    x1 = x1_bytes.to(dev).reshape(M_per_rank, K).view(torch.float8_e4m3fn)

    if x1_scale_varying:
        x1s_bytes = build_scale_varying((M_per_rank, n_pack, 2))
    else:
        x1s_bytes = torch.full((M_per_rank, n_pack, 2), 0x7F, dtype=torch.uint8)
    x1s = x1s_bytes.to(dev).reshape(M_per_rank, n_pack, 2).view(torch.float8_e8m0fnu)

    out, _, _ = torch_npu.npu_all_gather_quant_mm(
        x1, x2, hcom, ws,
        bias=None,
        x1_scale=x1s, x2_scale=x2s,
        quant_scale=None,
        block_size=BLOCK_SIZE,
        gather_index=0,
        gather_output=False,
        comm_turn=0,
        group_sizes=None,
        amax_output=False,
        y_dtype=torch_npu.bfloat16,
        x1_dtype=torch_npu.float8_e4m3fn,
        x2_dtype=torch_npu.float8_e4m3fn,
        x1_scale_dtype=torch_npu.float8_e8m0fnu,
        x2_scale_dtype=torch_npu.float8_e8m0fnu,
    )
    torch_npu.npu.synchronize()
    return out


def gather_and_compare(y):
    """All-gather local outputs, compare byte-exact. Returns (all_agree, diff_counts_per_rank)."""
    ws = dist.get_world_size()
    all_y = [torch.empty_like(y) for _ in range(ws)]
    dist.all_gather(all_y, y)
    local_bytes = y.cpu().contiguous().view(torch.uint8)
    diff_counts = []
    for r in range(ws):
        d = (all_y[r].cpu().contiguous().view(torch.uint8) != local_bytes).sum().item()
        diff_counts.append(d)
    return all(d == 0 for d in diff_counts), diff_counts


def _xrank_status(trial_agrees, trial_diffs):
    if all(trial_agrees):
        return "X_PASS"
    if not any(trial_agrees):
        frozen = [tuple(d) for d in trial_diffs]
        if len(set(frozen)) == 1:
            return "X_FAIL"
        return "X_NON_DET"
    return "X_NON_DET"


def run_xrank_bi_test(dev, hcom, ws, M_per_rank, K, N, seed, rank, x1_scale_varying):
    """Cross-rank BI: 各 rank 不同 x1 -> AllGather 后全局输入相同 -> full output 必须 byte-exact 一致。"""
    full_shas = []
    xrank_agrees = []
    xrank_diffs = []
    try:
        for _ in range(N_TRIALS):
            y = run_op(dev, hcom, ws, M_per_rank, K, N, seed, rank, x1_scale_varying=x1_scale_varying)
            full_shas.append(sha_full(y))
            agree, diffs = gather_and_compare(y)
            xrank_agrees.append(agree)
            xrank_diffs.append(diffs)
    except Exception:
        return "X_ERROR", full_shas, xrank_diffs

    det = len(set(full_shas)) == 1
    if not det:
        return "X_NON_DET", full_shas, xrank_diffs
    return _xrank_status(xrank_agrees, xrank_diffs), full_shas, xrank_diffs


def run_row_bi_test(dev, hcom, ws, M_per_rank, K, N, seed, rank, target_row, x1_scale_varying):
    """Row-BI: target_row 固定，其他行变化 -> target_row 输出必须不变。

    两次 trial 使用不同 seed 生成不同的非 target_row 数据，但 target_row 数据固定。
    """
    g_t = torch.Generator()
    g_t.manual_seed(12345 + target_row)
    target_row_bytes = torch.randint(0x10, 0x50, (K,), dtype=torch.uint8, generator=g_t)

    shas = []
    try:
        for trial in range(2):
            trial_seed = seed + (trial + 1) * 10000
            y = run_op(dev, hcom, ws, M_per_rank, K, N, trial_seed, rank,
                       target_row=target_row, target_row_bytes=target_row_bytes,
                       x1_scale_varying=x1_scale_varying)
            if target_row >= y.shape[0]:
                return "SKIP", None
            shas.append(sha_row(y, target_row))
    except Exception:
        return "ERROR", None

    if len(shas) < 2:
        return "ERROR", None
    if shas[0] == shas[1]:
        return "ROW_PASS", shas
    return "ROW_FAIL", shas


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dev = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl", world_size=ws, rank=rank)
    hcom = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(local_rank)

    if rank == 0:
        print(f"=== AllGatherMatmulV2 BI Test V2  WS={ws}  N_WARMUP={N_WARMUP}  N_TRIALS={N_TRIALS} ===", flush=True)

    if rank == 0:
        print("  Phase 0: HCCL warm-up...", flush=True)
    try:
        for _ in range(N_WARMUP):
            _ = run_op(dev, hcom, ws, 8, 256, 64, 42, rank)
    except Exception as e:
        if rank == 0:
            print(f"    warmup ERROR (continuing): {str(e)[:100]}", flush=True)
    dist.barrier()

    M_PER_RANKS = [4, 8, 16]
    KS = [256, 512, 1024]
    NS = [64, 128]

    n_xr_pass = n_xr_fail = n_xr_nondet = 0
    n_row_pass = n_row_fail = n_row_nondet = n_row_skip = 0
    xrank_fail_examples = []
    row_fail_examples = []
    nondet_examples = []

    cell_idx = 0
    for M_per_rank in M_PER_RANKS:
        for K in KS:
            for N in NS:
                base_seed = 100 + cell_idx * 7
                x1_var = (cell_idx % 2 == 0)
                cell_idx += 1

                xr_st, xr_shas, xr_diffs = run_xrank_bi_test(
                    dev, hcom, ws, M_per_rank, K, N, base_seed, rank, x1_var)

                if rank == 0:
                    if xr_st == "X_PASS":
                        n_xr_pass += 1
                    elif xr_st == "X_FAIL":
                        n_xr_fail += 1
                        xrank_fail_examples.append((M_per_rank * ws, K, N, xr_diffs))
                    else:
                        n_xr_nondet += 1
                        nondet_examples.append((M_per_rank * ws, K, N, xr_st, xr_shas))

                for target_row in [0, M_per_rank - 1]:
                    row_st, row_shas = run_row_bi_test(
                        dev, hcom, ws, M_per_rank, K, N, base_seed, rank,
                        target_row, x1_var)

                    if rank == 0:
                        M_total = M_per_rank * ws
                        if row_st == "ROW_PASS":
                            n_row_pass += 1
                        elif row_st == "ROW_FAIL":
                            n_row_fail += 1
                            row_fail_examples.append((M_total, K, N, target_row, row_shas))
                        elif row_st == "SKIP":
                            n_row_skip += 1
                        else:
                            n_row_nondet += 1

    if rank == 0:
        print(flush=True)
        print(f"  [Cross-rank BI] X_PASS={n_xr_pass}  X_FAIL={n_xr_fail}  X_NON_DET/ERR={n_xr_nondet}", flush=True)
        print(f"  [Row-BI]        ROW_PASS={n_row_pass}  ROW_FAIL={n_row_fail}  "
              f"NON_DET/ERR={n_row_nondet}  SKIP={n_row_skip}", flush=True)

        if xrank_fail_examples:
            print(flush=True)
            print("  X_FAIL (cross-rank BI violation):", flush=True)
            for ex in xrank_fail_examples[:5]:
                print(f"    M_total={ex[0]} K={ex[1]} N={ex[2]}", flush=True)
                print(f"      byte_diffs={ex[3][0]}", flush=True)

        if row_fail_examples:
            print(flush=True)
            print("  ROW_FAIL (row-BI violation):", flush=True)
            for ex in row_fail_examples[:5]:
                print(f"    M_total={ex[0]} K={ex[1]} N={ex[2]} row={ex[3]} shas={ex[4]}", flush=True)

        if nondet_examples:
            print(flush=True)
            print("  NON_DET (non-deterministic):", flush=True)
            for ex in nondet_examples[:5]:
                print(f"    M_total={ex[0]} K={ex[1]} N={ex[2]} status={ex[3]} shas={ex[4]}", flush=True)

    dist.destroy_process_group()
    sys.exit(0 if (n_xr_fail == 0 and n_row_fail == 0) else 1)


if __name__ == "__main__":
    main()
