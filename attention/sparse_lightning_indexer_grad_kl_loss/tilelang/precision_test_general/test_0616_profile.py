#!/usr/bin/env python3
"""Profiling test for 0616.py"""
import torch, torch_npu, math, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
mod = import_module("0616")
sparse_lightning_indexer_grad_kl_loss = mod.sparse_lightning_indexer_grad_kl_loss

CASE_CONFIG = {"B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
    "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
    "d_rope": 64, "K": 2048, "layout": "BSND"}

def gen_inputs():
    B, S1 = CASE_CONFIG["B"], CASE_CONFIG["S1"]
    S2, N1 = CASE_CONFIG["S2"], CASE_CONFIG["N1"]
    N2 = CASE_CONFIG["N2"]
    Nidx1, Nidx2 = CASE_CONFIG["Nidx1"], CASE_CONFIG["Nidx2"]
    DQuery = CASE_CONFIG["DQuery"]
    DQueryIndex = CASE_CONFIG["DQueryIndex"]
    d_rope, K = CASE_CONFIG["d_rope"], CASE_CONFIG["K"]
    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)
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
    return {"query": query, "key": key, "query_index": query_index, "key_index": key_index,
            "weights": weights, "sparse_indices": sparse_indices,
            "softmax_max": softmax_max, "softmax_sum": softmax_sum,
            "scale_value": scale_value, "query_rope": query_rope, "key_rope": key_rope}

def call_tl(inputs):
    return sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"], inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"], inputs["scale_value"],
        query_rope=inputs["query_rope"], key_rope=inputs["key_rope"],
        layout_query="BSND",
        actual_seq_lengths_query=[CASE_CONFIG["S1"]] * CASE_CONFIG["B"],
        actual_seq_lengths_key=[CASE_CONFIG["S2"]] * CASE_CONFIG["B"],
        sparse_mode=3)

def main():
    print("=" * 60)
    print("0616.py Profiling Test")
    print("=" * 60)
    inputs = gen_inputs()
    # Warmup
    print("Warmup (3 iters)...")
    for _ in range(3):
        call_tl(inputs)
    torch.npu.synchronize()
    # Capture: 5 iterations within one sync region
    print("Capture (5 iters within one sync)...")
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        call_tl(inputs)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    print(f"5x avg: {(t1-t0)*200:.3f}ms per call")

if __name__ == "__main__":
    main()
