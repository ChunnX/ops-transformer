#!/usr/bin/env python3
"""
Profiling test: use torch.npu.profiler + SLIG_PROFILE to capture pipeline timing.
Usage:
    SLIG_PROFILE=1 python test_0616_profile.py          # print Python-level breakdown
    NPU_PROFILE=1 python test_0616_profile.py           # capture NPU timeline (chrome trace)
"""

import torch
import torch_npu
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import import_module
mod = import_module("0616")
sparse_lightning_indexer_grad_kl_loss = mod.sparse_lightning_indexer_grad_kl_loss
BLOCK_M, BLOCK_N, BLOCK_K = mod.BLOCK_M, mod.BLOCK_N, mod.BLOCK_K

CASE_CONFIG = {
    "B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
    "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
    "d_rope": 64, "K": 2048, "layout": "BSND",
}

def gen_inputs(seed=42):
    B, S1, S2, N1 = CASE_CONFIG["B"], CASE_CONFIG["S1"], CASE_CONFIG["S2"], CASE_CONFIG["N1"]
    N2, Nidx1, Nidx2 = CASE_CONFIG["N2"], CASE_CONFIG["Nidx1"], CASE_CONFIG["Nidx2"]
    DQuery, DQueryIndex, d_rope, K = CASE_CONFIG["DQuery"], CASE_CONFIG["DQueryIndex"], CASE_CONFIG["d_rope"], CASE_CONFIG["K"]
    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)
    torch.manual_seed(seed)
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
        if s2_real_len < K:
            sparse_indices[0, s1_idx, 0, s2_real_len:] = -1
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

    return {
        "query": query, "key": key,
        "query_index": query_index, "key_index": key_index,
        "weights": weights, "sparse_indices": sparse_indices,
        "softmax_max": softmax_max, "softmax_sum": softmax_sum,
        "scale_value": scale_value,
        "query_rope": query_rope, "key_rope": key_rope,
    }

def call_tl(inputs):
    return sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        layout_query="BSND",
        actual_seq_lengths_query=[CASE_CONFIG["S1"]] * CASE_CONFIG["B"],
        actual_seq_lengths_key=[CASE_CONFIG["S2"]] * CASE_CONFIG["B"],
        sparse_mode=3,
    )

def print_kernel_info():
    K = CASE_CONFIG["K"]
    B, S1, N1, Nidx1 = CASE_CONFIG["B"], CASE_CONFIG["S1"], CASE_CONFIG["N1"], CASE_CONFIG["Nidx1"]
    n_num = K // BLOCK_N
    block_num = B * ((S1 + BLOCK_M - 1) // BLOCK_M) * n_num

    print("\n=== Kernel Structure ===")
    print(f"  block_num  = {block_num} (B={B} x S1_tiles={((S1+BLOCK_M-1)//BLOCK_M)} x K/N={n_num})")
    print(f"  n_num      = {n_num} (K={K} / BLOCK_N={BLOCK_N})")
    print(f"  Each block processes 1/{n_num} of K blocks")
    print(f"")
    print(f"  C1: Matmul1 ({N1}x512 @ 512x{BLOCK_N}) + Matmul2 ({Nidx1}x128 @ 128x{BLOCK_N})")
    print(f"  V1: softmax/P_reduced/ReduceSumY/SoftmaxI/dWeight/reluGrad/Loss")
    print(f"  C2: Matmul4 ({Nidx1}x{BLOCK_N} @ {BLOCK_N}x128) [REDUNDANT: {n_num}x per core]")
    print(f"  V3: Cast dQueryIndex FP32->FP16")
    print(f"")
    print(f"  C2 redundancy: {n_num} cores x {n_num} iterations = {n_num*n_num}x total matmul4 ops")
    print(f"  Ideal (AscendC): each core does 1/{n_num} of matmul4 (ping-pong partitioned)")

def main():
    do_npu_profile = os.environ.get("NPU_PROFILE", "0") == "1"

    print("=" * 60)
    print("0616.py Profiling Test")
    print("=" * 60)

    inputs = gen_inputs()
    print_kernel_info()

    # Warmup
    print("\n[Warmup] ...")
    for _ in range(3):
        call_tl(inputs)
    torch.npu.synchronize()
    print("  done")

    if do_npu_profile:
        print("\n[NPU Profiler] Capturing timeline ...")
        output_dir = "./profiling_output"
        os.makedirs(output_dir, exist_ok=True)

        with torch.npu.profiler.profile(
            activities=[torch.npu.profiler.ProfilerActivity.NPU],
            schedule=torch.npu.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            on_trace_ready=torch.npu.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                call_tl(inputs)
                torch.npu.synchronize()
                prof.step()

        print(f"  Trace saved to {output_dir}/")
        print(f"  Open with: chrome://tracing or msprof --parse=on {output_dir}/*.json")

    else:
        print("\n[Timing] Python-level breakdown ...")
        os.environ["SLIG_PROFILE"] = "1"
        call_tl(inputs)
        os.environ["SLIG_PROFILE"] = "0"

        print(f"\n  To capture NPU-level timeline (C1/V1/C2/V3 breakdown):")
        print(f"    NPU_PROFILE=1 python test_0616_profile.py")

if __name__ == "__main__":
    main()
