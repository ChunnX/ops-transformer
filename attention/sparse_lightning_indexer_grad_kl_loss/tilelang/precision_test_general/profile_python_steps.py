#!/usr/bin/env python3
"""
Python端步骤耗时分析 - Case 0
定位端到端3.8ms中各步骤的耗时分布
"""

import torch
import torch_npu
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_lightning_indexer_grad_kl_loss import (
    sparse_lightning_indexer_grad_kl_loss,
    _preprocess_gather,
    _compute_d_key_index,
    check_param_valid,
    _clear_caches,
    _kernel_cache,
    _workspace_cache,
)
from sparse_lightning_indexer_grad_kl_loss import (
    tilelang_grad_kl_loss_single_kernel,
    ceil_div,
    BLOCK_M, BLOCK_N, BLOCK_K, VECTOR_BASEG, VECTOR_BASEN,
    DQUERY, DQUERY_INDEX,
)

import tilelang

WARMUP = 5
BENCH = 20

B, S1, S2 = 1, 1, 2048
N1, N2 = 128, 1
Nidx1, Nidx2 = 64, 1
DQuery, DQueryIndex = 512, 128
d_rope = 64
K = 2048
G = N1 // N2
scale_value = 1.0 / math.sqrt(DQuery)


def bench_fn(fn, warmup=WARMUP, iters=BENCH):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    arr = sorted(times)
    return arr[len(arr)//2] * 1e3


def main():
    print("=" * 60)
    print("Python端步骤耗时分析 - Case 0")
    print("=" * 60)

    torch.manual_seed(42)
    dtype = torch.float16

    query = torch.randn(B, S1, N1, DQuery, dtype=dtype).npu()
    key = torch.randn(B, S2, N2, DQuery, dtype=dtype).npu()
    query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).npu()
    key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).npu()
    query_rope = torch.randn(B, S1, N1, d_rope, dtype=dtype).npu()
    key_rope = torch.randn(B, S2, N2, d_rope, dtype=dtype).npu()
    weights = torch.randn(B, S1, Nidx1, dtype=dtype).npu()
    sparse_indices = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).npu()

    s2_real_len = min(S2, K)
    indices = torch.randperm(S2)[:s2_real_len]
    for s1_idx in range(S1):
        sparse_indices[0, s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
    torch.npu.synchronize()

    with torch.no_grad():
        k_mqa = key.squeeze(2)
        k_rope_mqa = key_rope.squeeze(2)
        q_full = torch.cat([query, query_rope], dim=-1)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
        k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
        sm_max_list, sm_sum_list = [], []
        for s1_idx in range(S1):
            row = torch.matmul(q_full[:, s1_idx:s1_idx+1], k_full_t)
            row_scaled = row.float() * scale_value
            row_max = row_scaled.max(dim=-1).values
            row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
            sm_max_list.append(row_max)
            sm_sum_list.append(row_sum)
        sm_max_val = torch.cat(sm_max_list, dim=1)
        sm_sum_val = torch.cat(sm_sum_list, dim=1)
        softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
        softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
    torch.npu.synchronize()

    # Step-by-step profiling
    results = {}

    # 1. Gather key + keyIndex
    def step_gather():
        _preprocess_gather(key, key_index, sparse_indices, B, S1, S2, N2, Nidx2, K, DQuery, DQueryIndex)
    results["1. gather_key+keyIndex"] = bench_fn(step_gather)

    # 2. Gather rope key
    def step_gather_rope():
        _sparse_idx_rope = sparse_indices[:, :, 0, :].clamp(min=0)
        gather_key_rope = key_rope[torch.arange(B, device=query.device).view(B, 1, 1),
                                   _sparse_idx_rope, 0, :].contiguous()
        _valid_mask_rope = (sparse_indices[:, :, 0, :] >= 0).unsqueeze(-1).to(query.dtype)
        gather_key_rope.mul_(_valid_mask_rope)
    results["2. gather_rope_key"] = bench_fn(step_gather_rope)

    # 3. Pre-transpose
    gather_key, gather_key_index = _preprocess_gather(
        key, key_index, sparse_indices, B, S1, S2, N2, Nidx2, K, DQuery, DQueryIndex)
    _sparse_idx_rope = sparse_indices[:, :, 0, :].clamp(min=0)
    gather_key_rope = key_rope[torch.arange(B, device=query.device).view(B, 1, 1),
                               _sparse_idx_rope, 0, :].contiguous()
    _valid_mask_rope = (sparse_indices[:, :, 0, :] >= 0).unsqueeze(-1).to(query.dtype)
    gather_key_rope.mul_(_valid_mask_rope)

    def step_transpose():
        gather_key_index.permute(0, 1, 3, 2).contiguous()
    results["3. transpose_GKI"] = bench_fn(step_transpose)

    gather_key_index_t = gather_key_index.permute(0, 1, 3, 2).contiguous()

    # 4. TileLang kernel (with workspace)
    _clear_caches()
    input_dtype = query.dtype
    input_dtype_str = "float16"
    calc_dtype_str = "float"

    S2_pad = (S2 + VECTOR_BASEN - 1) // VECTOR_BASEN * VECTOR_BASEN
    if S2_pad > S2:
        key_p = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
        key_index_p = torch.nn.functional.pad(key_index, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
    else:
        key_p = key
        key_index_p = key_index

    d_rope_aligned = ((d_rope + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    query_rope_p = torch.nn.functional.pad(query_rope, (0, d_rope_aligned - d_rope)) if d_rope_aligned > d_rope else query_rope
    gather_key_rope_p = torch.nn.functional.pad(gather_key_rope, (0, d_rope_aligned - d_rope)) if d_rope_aligned > d_rope else gather_key_rope

    func = tilelang_grad_kl_loss_single_kernel(
        B, S1, S2_pad, N1, N2, G, Nidx1, Nidx2, DQuery, DQueryIndex, K,
        d_rope_aligned, input_dtype_str, calc_dtype_str, scale_value,
    )

    bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=query.device)
    bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=query.device)
    p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
    reduce_sum_y_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
    softmax_i_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=query.device)
    d_query_index_ws = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=torch.float32, device=query.device)
    d_query_index_out = torch.zeros(B, S1, Nidx1, DQueryIndex, dtype=input_dtype, device=query.device)
    d_weight_out = torch.zeros(B, S1, Nidx1, dtype=input_dtype, device=query.device)
    relu_grad_out = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=query.device)
    loss_out = torch.zeros(B, S1, 16, dtype=torch.float32, device=query.device)

    torch.npu.synchronize()

    def step_kernel():
        func(
            query, query_rope_p, gather_key, gather_key_rope_p,
            query_index, gather_key_index, gather_key_index_t,
            weights, softmax_max, softmax_sum,
            bmm1_res_ws, bmm2_res_ws,
            p_reduced_ws, reduce_sum_y_ws, softmax_i_ws,
            d_query_index_out, d_weight_out, relu_grad_out, loss_out,
            d_query_index_ws,
        )
    results["4. tilelang_kernel"] = bench_fn(step_kernel)

    # 5. dKeyIndex computation
    def step_dkeyindex():
        _compute_d_key_index(
            relu_grad_out, query_index, sparse_indices,
            B, S1, S2, Nidx1, Nidx2, K, DQueryIndex,
            query.device, input_dtype,
        )
    results["5. compute_dKeyIndex"] = bench_fn(step_dkeyindex)

    # 6. Full end-to-end
    def step_full():
        sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index,
            weights, sparse_indices, softmax_max, softmax_sum,
            scale_value,
            query_rope=query_rope, key_rope=key_rope,
            layout_query="BSND",
            actual_seq_lengths_query=[S1], actual_seq_lengths_key=[S2],
            sparse_mode=3,
        )
    results["6. end_to_end"] = bench_fn(step_full)

    # Print results
    print("\n步骤耗时分解 (中位数, ms):")
    print("-" * 50)
    total_measured = 0
    for name, ms in results.items():
        print(f"  {name:<35s} {ms:>8.3f}")
        if name != "6. end_to_end":
            total_measured += ms
    print("-" * 50)
    print(f"  {'步骤1~5合计':<35s} {total_measured:>8.3f}")
    print(f"  {'端到端':<35s} {results['6. end_to_end']:>8.3f}")
    print(f"  {'未测量开销':<35s} {results['6. end_to_end'] - total_measured:>8.3f}")


if __name__ == "__main__":
    main()