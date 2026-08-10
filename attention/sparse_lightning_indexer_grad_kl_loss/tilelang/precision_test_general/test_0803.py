#!/usr/bin/env python3
"""
Case 0 测试 - 基于0803.py (gather移入kernel V0 scope)
"""
import torch
import torch_npu
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import import_module
mod = import_module("0803_gather32")
sparse_lightning_indexer_grad_kl_loss = mod.sparse_lightning_indexer_grad_kl_loss

WARMUP_ITERS = 5
BENCH_ITERS = 10

CASE_CONFIGS = (
    (
        "small_b1_s1_1_k_2048",
        {
            "B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
            "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
            "d_rope": 64, "K": 2048, "layout": "BSND",
        },
    ),
    (
        "large_s1_b1_s1_1024_k_5120",
        {
            "B": 1, "S1": 1024, "S2": 5120, "N1": 128, "N2": 1,
            "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
            "d_rope": 64, "K": 5120, "layout": "BSND",
        },
    ),
)


def gen_inputs(case_config, seed=42):
    B = case_config["B"]
    S1 = case_config["S1"]
    S2 = case_config["S2"]
    N1 = case_config["N1"]
    N2 = case_config["N2"]
    Nidx1 = case_config["Nidx1"]
    Nidx2 = case_config["Nidx2"]
    DQuery = case_config["DQuery"]
    DQueryIndex = case_config["DQueryIndex"]
    d_rope = case_config["d_rope"]
    K = case_config["K"]

    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    torch.manual_seed(seed)
    dtype = torch.bfloat16

    query = torch.randn(B, S1, N1, DQuery, dtype=dtype).npu()
    key = torch.randn(B, S2, N2, DQuery, dtype=dtype).npu()
    query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).npu()
    key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).npu()
    query_rope = torch.randn(B, S1, N1, d_rope, dtype=dtype).npu()
    key_rope = torch.randn(B, S2, N2, d_rope, dtype=dtype).npu()
    weights = torch.randn(B, S1, Nidx1, dtype=dtype).npu()
    sparse_indices_cpu = torch.full((B, S1, Nidx2, K), -1, dtype=torch.int32)
    sparse_generator = torch.Generator(device="cpu")
    sparse_generator.manual_seed(seed + 1009)
    for batch_id in range(B):
        for s1_idx in range(S1):
            visible_keys = S2 - S1 + s1_idx + 1
            s2_real_len = min(visible_keys, K)
            indices = torch.randperm(visible_keys, generator=sparse_generator)[:s2_real_len]
            sparse_indices_cpu[batch_id, s1_idx, 0, :s2_real_len] = indices.to(torch.int32)
    sparse_indices = sparse_indices_cpu.npu()
    torch.npu.synchronize()

    with torch.no_grad():
        k_mqa = key.squeeze(2)
        k_rope_mqa = key_rope.squeeze(2)
        q_full = torch.cat([query, query_rope], dim=-1)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
        k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
        sm_max_list = []
        sm_sum_list = []
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


def call_tl(inputs, case_config):
    return sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        layout_query="BSND",
        actual_seq_lengths_query=[case_config["S1"]] * case_config["B"],
        actual_seq_lengths_key=[case_config["S2"]] * case_config["B"],
        sparse_mode=3,
    )


def call_official(inputs, case_config):
    return torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        actual_seq_qlen=[case_config["S1"]] * case_config["B"],
        actual_seq_klen=[case_config["S2"]] * case_config["B"],
        layout="BSND", sparse_mode=3,
        pre_tokens=65536, next_tokens=65536,
    )


def bench(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
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
    med = arr[len(arr) // 2]
    avg = sum(arr) / len(arr)
    mn = arr[0]
    return avg, med, mn


def run_case(case_name, case_config):
    print("=" * 60)
    print(f"{case_name} - 0803_gather32.py (GATHER_CHUNK_SIZE=32)")
    print("=" * 60)

    mod._clear_caches()
    inputs = gen_inputs(case_config)

    # TileLang
    print("\n[1] TileLang (0803 kernel-gather)...")
    torch.npu.synchronize()
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = call_tl(inputs, case_config)
        torch.npu.synchronize()
        print("  完成")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 官方
    print("\n[2] 官方...")
    torch.npu.synchronize()
    d_qi_off, d_ki_off, dw_off, loss_off = call_official(inputs, case_config)
    torch.npu.synchronize()
    print("  完成")

    # 精度
    print("\n[3] 精度对比...")
    for name, tl_val, off_val in [
        ("dQueryIndex", d_qi_tl, d_qi_off),
        ("dKeyIndex", d_ki_tl, d_ki_off),
        ("dWeights", dw_tl, dw_off),
    ]:
        diff = (tl_val.float() - off_val.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_err = (diff / (off_val.float().abs() + 1e-6)).mean().item()
        print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, rel_err={rel_err:.6e}")

    loss_abs_diff = abs(loss_tl.float().item() - loss_off.float().item())
    loss_rel_err = loss_abs_diff / (abs(loss_off.float().item()) + 1e-6)
    print(f"  Loss: abs_diff={loss_abs_diff:.6e}, rel_err={loss_rel_err:.6e}")

    # 性能
    print(f"\n[4] 性能测试 (warmup={WARMUP_ITERS}, bench={BENCH_ITERS})...")
    torch.npu.synchronize()
    tl_avg, tl_med, tl_min = bench(lambda: call_tl(inputs, case_config))
    torch.npu.synchronize()
    off_avg, off_med, off_min = bench(lambda: call_official(inputs, case_config))
    speedup = off_med / tl_med
    print(f"  TileLang: avg={tl_avg*1e3:.3f}ms, med={tl_med*1e3:.3f}ms, min={tl_min*1e3:.3f}ms")
    print(f"  官方:     avg={off_avg*1e3:.3f}ms, med={off_med*1e3:.3f}ms, min={off_min*1e3:.3f}ms")
    print(f"  加速比:   {speedup:.2f}x (基于中位数)")


def main():
    for case_name, case_config in CASE_CONFIGS:
        run_case(case_name, case_config)


if __name__ == "__main__":
    main()
