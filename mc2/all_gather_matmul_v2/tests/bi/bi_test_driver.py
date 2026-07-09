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

"""Batch-invariance test driver for npu_all_gather_quant_mm (AllGatherMatmulV2).

Two independent BI dimensions:
  A) **Row-BI (literature BI)**: y[i] = f(x[i], W)，row i 输出只取决于 row i + 共享权重
  B) **Cross-rank BI**: 同输入下所有 rank 的 y 输出必须 byte-exact 一致
     （issue #2872 报告的 LOCAL/GATHER 双路径 divergence）

方法学（关键，详见 README.md §2 Methodology Note）：
    每个 cell 四步走，缺一不可：
      0. **HCCL warm-up**：稳定 HCCL state
      1. **Multi-trial determinism baseline**：同 input N_TRIALS 次，所有 SHA 必须一致
      2. **A vs B row-BI check**：只有 baseline 过了，才比较 variant A 和 B 的 SHA
      3. **Cross-rank BI check**：all_gather 各 rank 的 full-y SHA，比较是否一致
"""
import os
import sys
import hashlib
import torch
import torch_npu
import torch.distributed as dist


BLOCK_SIZE = 32
N_WARMUP = 2
N_TRIALS = 3


def sha_row(t, row_idx):
    return hashlib.sha256(t[row_idx].cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def sha_full(t):
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def build_x2_scale_k_varying(n_pack, N):
    bs = torch.empty(n_pack, N, 2, dtype=torch.uint8)
    for p in range(n_pack):
        for c in range(N):
            for slot in (0, 1):
                bs[p, c, slot] = 0x78 + ((p * 2 + slot) % 8)
    return bs


def build_x1_scale_varying(M_total, n_pack):
    bs = torch.empty(M_total, n_pack, 2, dtype=torch.uint8)
    for m in range(M_total):
        for p in range(n_pack):
            for slot in (0, 1):
                bs[m, p, slot] = 0x78 + ((m + p * 2 + slot) % 8)
    return bs


def make_and_call(dev, hcom, ws, M_per_rank, K, N, target_row, seed, x1_scale_varying=False):
    M_total = M_per_rank * ws
    n_pack = (K + BLOCK_SIZE * 2 - 1) // (BLOCK_SIZE * 2)

    g_t = torch.Generator(); g_t.manual_seed(12345)
    target_row_bytes = torch.randint(0x10, 0x50, (K,), dtype=torch.uint8, generator=g_t)

    g_w = torch.Generator(); g_w.manual_seed(9000)
    x2_bytes = torch.randint(0x10, 0x50, (K, N), dtype=torch.uint8, generator=g_w)
    x2 = x2_bytes.to(dev).reshape(K, N).view(torch.float8_e4m3fn)

    x2s_bytes = build_x2_scale_k_varying(n_pack, N)
    x2s = x2s_bytes.to(dev).reshape(n_pack, N, 2).view(torch.float8_e8m0fnu)

    g_o = torch.Generator(); g_o.manual_seed(seed)
    x1_bytes = torch.randint(0x10, 0x50, (M_per_rank, K), dtype=torch.uint8, generator=g_o)
    if target_row < M_per_rank:
        x1_bytes[target_row] = target_row_bytes
    if x1_scale_varying:
        x1s_bytes = build_x1_scale_varying(M_per_rank, n_pack)
    else:
        x1s_bytes = torch.full((M_per_rank, n_pack, 2), 0x7F, dtype=torch.uint8)
    x1 = x1_bytes.to(dev).reshape(M_per_rank, K).view(torch.float8_e4m3fn)
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


def cross_rank_check(y):
    """各 rank 计算 full-y SHA，all_gather 后比较。返回 (all_agree, all_shas)。"""
    local_sha = sha_full(y)
    ws = dist.get_world_size()
    all_shas = [None] * ws
    dist.all_gather_object(all_shas, local_sha)
    all_agree = len(set(all_shas)) == 1
    return all_agree, all_shas


def _xrank_status(trial_agrees, trial_shas):
    """从多次 trial 的 cross-rank 结果判定 xrank 状态。

    trial_agrees: [bool] — 每次 trial 是否所有 rank SHA 一致
    trial_shas:   [[sha_r0, sha_r1, ...], ...] — 每次 trial 各 rank 的 SHA
    """
    if all(trial_agrees):
        return "X_PASS"
    if not any(trial_agrees):
        frozen = [tuple(s) for s in trial_shas]
        if len(set(frozen)) == 1:
            return "X_FAIL"
        return "X_NON_DET"
    return "X_NON_DET"


def rigorous_bi_cell(dev, hcom, ws, M_per_rank, K, N, target_row, seed_a, seed_b, x1_scale_varying=False):
    """BI cell 判定。

    返回 (bi_status, xrank_status, sha_a, sha_b, shas_a_all, shas_b_all, xrank_info)

    bi_status  ∈ {BI_PASS, BI_FAIL, NON_DET, ERROR, SKIP}
    xrank_status ∈ {X_PASS, X_FAIL, X_NON_DET, X_ERROR, X_SKIP}
    """
    shas_a = []
    shas_b = []
    xrank_agree_a = []
    xrank_agree_b = []
    xrank_shas_a = []
    xrank_shas_b = []
    try:
        for _ in range(N_TRIALS):
            y = make_and_call(dev, hcom, ws, M_per_rank, K, N, target_row, seed_a, x1_scale_varying)
            if target_row >= y.shape[0]:
                return ("SKIP", "X_SKIP", None, None, shas_a, shas_b, None)
            shas_a.append(sha_row(y, target_row))
            agree, all_shas = cross_rank_check(y)
            xrank_agree_a.append(agree)
            xrank_shas_a.append(all_shas)
        for _ in range(N_TRIALS):
            y = make_and_call(dev, hcom, ws, M_per_rank, K, N, target_row, seed_b, x1_scale_varying)
            shas_b.append(sha_row(y, target_row))
            agree, all_shas = cross_rank_check(y)
            xrank_agree_b.append(agree)
            xrank_shas_b.append(all_shas)
    except Exception:
        return ("ERROR", "X_ERROR", None, None, shas_a, shas_b, None)

    det_a = len(set(shas_a)) == 1
    det_b = len(set(shas_b)) == 1
    if not det_a or not det_b:
        bi_status = "NON_DET"
    else:
        bi_status = "BI_PASS" if shas_a[0] == shas_b[0] else "BI_FAIL"

    xr_a = _xrank_status(xrank_agree_a, xrank_shas_a)
    xr_b = _xrank_status(xrank_agree_b, xrank_shas_b)
    if xr_a == "X_FAIL" or xr_b == "X_FAIL":
        xrank_status = "X_FAIL"
    elif xr_a == "X_NON_DET" or xr_b == "X_NON_DET":
        xrank_status = "X_NON_DET"
    else:
        xrank_status = "X_PASS"

    xrank_info = {"a": xrank_shas_a, "b": xrank_shas_b}
    sha_a = shas_a[0] if shas_a else None
    sha_b = shas_b[0] if shas_b else None
    return (bi_status, xrank_status, sha_a, sha_b, shas_a, shas_b, xrank_info)


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dev = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl", world_size=ws, rank=rank)
    hcom = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(local_rank)

    if rank == 0:
        print(f"=== all_gather_matmul_v2 BI rigorous WS={ws} N_WARMUP={N_WARMUP} N_TRIALS={N_TRIALS} ===", flush=True)

    if rank == 0:
        print("  Phase 0: HCCL warm-up...", flush=True)
    try:
        for _ in range(N_WARMUP):
            _ = make_and_call(dev, hcom, ws, 8, 256, 64, 0, 42)
    except Exception as e:
        if rank == 0:
            print(f"    warmup ERROR (continuing): {str(e)[:100]}", flush=True)

    M_PER_RANKS = [4, 8, 16]
    KS = [256, 512, 1024]
    NS = [64, 128]
    TARGET_ROW = 0
    X1_VARYING_MOD = 4

    n_bi_pass = n_bi_fail = n_nondet = n_err = n_skip = 0
    n_xr_pass = n_xr_fail = n_xr_nondet = 0
    bi_fail_examples = []
    nondet_examples = []
    xrank_fail_examples = []

    cell_idx = 0
    for M_per_rank in M_PER_RANKS:
        for K in KS:
            for N in NS:
                M_total = M_per_rank * ws
                shape_hash = hash((M_total, K, N)) % 10000
                seed_a = 100 + shape_hash
                seed_b = 200 + shape_hash
                x1_var = (cell_idx % X1_VARYING_MOD == 0)
                cell_idx += 1
                bi_st, xr_st, sa, sb, sas, sbs, xr_info = rigorous_bi_cell(
                    dev, hcom, ws, M_per_rank, K, N, TARGET_ROW,
                    seed_a=seed_a, seed_b=seed_b, x1_scale_varying=x1_var)
                if rank == 0:
                    if bi_st == "BI_PASS": n_bi_pass += 1
                    elif bi_st == "BI_FAIL":
                        n_bi_fail += 1
                        bi_fail_examples.append((M_total, K, N, sa, sb))
                    elif bi_st == "NON_DET":
                        n_nondet += 1
                        nondet_examples.append((M_total, K, N, sas, sbs))
                    elif bi_st == "ERROR": n_err += 1
                    elif bi_st == "SKIP": n_skip += 1

                    if xr_st == "X_PASS": n_xr_pass += 1
                    elif xr_st == "X_FAIL":
                        n_xr_fail += 1
                        xrank_fail_examples.append((M_total, K, N, xr_info))
                    elif xr_st in ("X_NON_DET", "X_ERROR"):
                        n_xr_nondet += 1

    if rank == 0:
        print(f"  [Row-BI]   BI_PASS={n_bi_pass} BI_FAIL={n_bi_fail} NON_DET={n_nondet} ERROR={n_err} SKIP={n_skip}", flush=True)
        print(f"  [XRank-BI] X_PASS={n_xr_pass} X_FAIL={n_xr_fail} X_NON_DET={n_xr_nondet}", flush=True)
        if bi_fail_examples:
            print(f"  BI_FAIL (row-BI 违例 — deterministic but A != B):", flush=True)
            for ex in bi_fail_examples[:5]:
                print(f"    M_total={ex[0]} K={ex[1]} N={ex[2]} A_sha={ex[3]} B_sha={ex[4]}", flush=True)
        if xrank_fail_examples:
            print(f"  X_FAIL (cross-rank BI 违例 — 同输入不同 rank 输出不一致):", flush=True)
            for ex in xrank_fail_examples[:5]:
                info = ex[3]
                print(f"    M_total={ex[0]} K={ex[1]} N={ex[2]}", flush=True)
                print(f"      variant A rank_shas={info['a'][0]}", flush=True)
                print(f"      variant B rank_shas={info['b'][0]}", flush=True)
        if nondet_examples:
            print(f"  NON_DET (同 input 跨 trial 不一致 — 无法判别 BI):", flush=True)
            for ex in nondet_examples[:5]:
                print(f"    M_total={ex[0]} K={ex[1]} N={ex[2]} A_trials={ex[3]} B_trials={ex[4]}", flush=True)

    dist.destroy_process_group()
    sys.exit(0 if (n_bi_fail == 0 and n_xr_fail == 0) else 1)


if __name__ == "__main__":
    main()
