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

V2 improvements over V1 (bi_test_driver.py):
  1. Cross-rank BI: each rank uses rank-specific seed for different x1 data,
     AllGather produces identical global input, compare full output byte-exact.
  2. Row-BI: fix target_row data across trials, vary only other rows.
  3. Cover multiple target_row positions (rank 0 rows + non-rank 0 rows).
  4. x2_scale varies across K-axis (key BI trigger condition).
  5. Add dist.barrier() after warmup for stable HCCL state.
  6. Synchronized error handling to prevent collective op deadlocks.
  7. Exit code accounts for NON_DET results.

Two independent BI dimensions:
  A) **Cross-rank BI**: all ranks must produce byte-exact identical output
     (issue #2872: LOCAL/GATHER dual-path divergence)
  B) **Row-BI (literature BI)**: y[i] = f(x[i], W), row i output depends
     only on row i input + shared weights

Output shape: (M_total, N) regardless of gather_output parameter.
All ranks hold the full output after AllGather + matmul.
"""
import os
import sys
import traceback
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


def build_x2_scale_varying(n_pack, N):
    """Build x2_scale with shape (n_pack, N, 2), varying across K-axis blocks."""
    bs = torch.empty(n_pack, N, 2, dtype=torch.uint8)
    for p in range(n_pack):
        for c in range(N):
            for slot in (0, 1):
                bs[p, c, slot] = 0x78 + ((p * 2 + slot) % 8)
    return bs


def build_x1_scale_varying(M_per_rank, n_pack):
    """Build x1_scale with shape (M_per_rank, n_pack, 2), varying across M and K."""
    bs = torch.empty(M_per_rank, n_pack, 2, dtype=torch.uint8)
    for m in range(M_per_rank):
        for p in range(n_pack):
            for slot in (0, 1):
                bs[m, p, slot] = 0x78 + ((m + p * 2 + slot) % 8)
    return bs


def sync_error(dev, local_error):
    """Synchronize error state across all ranks via all_reduce.
    Returns True if ANY rank reported an error.
    """
    err_tensor = torch.tensor([1.0 if local_error else 0.0], device=dev)
    dist.all_reduce(err_tensor)
    return err_tensor.item() > 0.5


def run_op(dev, hcom, ws, M_per_rank, K, N, seed, rank,
           target_row=None, target_row_bytes=None, x1_scale_varying=False):
    """Run npu_all_gather_quant_mm.

    x1: each rank uses (seed * ws + rank) for different data,
        AllGather produces identical global input across ranks.
    x2: all ranks identical (fixed seed 9000), shape (K, N).
    x2_scale: varies across K-axis, shape (n_pack, N, 2).
    x1_scale: controlled by x1_scale_varying, shape (M_per_rank, n_pack, 2).

    Output shape: (M_total, N) where M_total = M_per_rank * ws.
    """
    n_pack = (K + 63) // 64

    g_w = torch.Generator()
    g_w.manual_seed(9000)
    x2_bytes = torch.randint(0x10, 0x50, (K, N), dtype=torch.uint8, generator=g_w)
    x2 = x2_bytes.to(dev).reshape(K, N).view(torch.float8_e4m3fn)
    x2s_bytes = build_x2_scale_varying(n_pack, N)
    x2s = x2s_bytes.to(dev).reshape(n_pack, N, 2).view(torch.float8_e8m0fnu)

    g_o = torch.Generator()
    g_o.manual_seed(seed * ws + rank)
    x1_bytes = torch.randint(0x10, 0x50, (M_per_rank, K), dtype=torch.uint8, generator=g_o)
    if target_row is not None and target_row_bytes is not None and target_row < M_per_rank:
        x1_bytes[target_row] = target_row_bytes
    x1 = x1_bytes.to(dev).reshape(M_per_rank, K).view(torch.float8_e4m3fn)

    if x1_scale_varying:
        x1s_bytes = build_x1_scale_varying(M_per_rank, n_pack)
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


def safe_gather_and_compare(y, dev):
    """All-gather outputs from all ranks, compare byte-exact.
    All ranks participate even if local y is a dummy (error case).

    Returns (all_agree, diff_counts_per_rank).
    """
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
    """Cross-rank BI: each rank has different x1 -> AllGather -> same global input
    -> full output (M_total, N) must be byte-exact identical across all ranks.
    """
    M_total = M_per_rank * ws
    full_shas = []
    xrank_agrees = []
    xrank_diffs = []
    local_error = False
    for _ in range(N_TRIALS):
        try:
            y = run_op(dev, hcom, ws, M_per_rank, K, N, seed, rank, x1_scale_varying=x1_scale_varying)
        except Exception:
            if rank == 0:
                print(f"    [xrank] run_op Exception: {traceback.format_exc()}", flush=True)
            local_error = True
            y = torch.zeros(M_total, N, dtype=torch.bfloat16, device=dev)

        full_shas.append(sha_full(y))
        agree, diffs = safe_gather_and_compare(y, dev)
        xrank_agrees.append(agree)
        xrank_diffs.append(diffs)

    if sync_error(dev, local_error):
        return "X_ERROR", full_shas, xrank_diffs

    det = len(set(full_shas)) == 1
    if not det:
        return "X_NON_DET", full_shas, xrank_diffs
    return _xrank_status(xrank_agrees, xrank_diffs), full_shas, xrank_diffs


def run_row_bi_test(dev, hcom, ws, M_per_rank, K, N, seed, rank, target_row, x1_scale_varying):
    """Row-BI: target_row data is fixed, other rows vary -> target_row output must not change.

    Two trials use different seeds to generate different non-target_row data,
    but target_row data is overwritten with the same target_row_bytes on the
    rank that owns that row.
    """
    M_total = M_per_rank * ws
    owner_rank = target_row // M_per_rank
    local_idx = target_row % M_per_rank

    g_t = torch.Generator()
    g_t.manual_seed(12345 + target_row)
    target_row_bytes = torch.randint(0x10, 0x50, (K,), dtype=torch.uint8, generator=g_t)

    shas = []
    local_error = False
    for trial in range(2):
        trial_seed = seed + (trial + 1) * 10000
        try:
            y = run_op(dev, hcom, ws, M_per_rank, K, N, trial_seed, rank,
                       target_row=local_idx if rank == owner_rank else None,
                       target_row_bytes=target_row_bytes if rank == owner_rank else None,
                       x1_scale_varying=x1_scale_varying)
        except Exception:
            if rank == 0:
                print(f"    [row-BI] run_op Exception: {traceback.format_exc()}", flush=True)
            local_error = True
            y = torch.zeros(M_total, N, dtype=torch.bfloat16, device=dev)

        if target_row >= y.shape[0]:
            shas.append(None)
        else:
            shas.append(sha_row(y, target_row))

    if sync_error(dev, local_error):
        return "ERROR", None

    if any(s is None for s in shas):
        return "SKIP", None
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
    for _ in range(N_WARMUP):
        _ = run_op(dev, hcom, ws, 8, 256, 64, 42, rank)
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

                target_rows = [0, M_per_rank - 1]
                if M_per_rank > 1 and ws > 1:
                    target_rows.append(M_per_rank)
                for target_row in target_rows:
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
    sys.exit(0 if (n_xr_fail == 0 and n_row_fail == 0 and n_xr_nondet == 0 and n_row_nondet == 0) else 1)


if __name__ == "__main__":
    main()
