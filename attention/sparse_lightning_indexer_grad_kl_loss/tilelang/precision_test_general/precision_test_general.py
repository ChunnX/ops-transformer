#!/usr/bin/env python3
"""
泛化精度+性能测试 - SparseLightningIndexerGradKLLoss TileLang
共5个测试用例，覆盖关键维度：
- Case 0: BSND S1=1, S2=2048, K=2048
- Case 1: TND S1=128, S2=2048, K=2048
- Case 2: TND S1=64, S2=2048, K=2048
- Case 3: TND S1=512, S2=2048, K=2048
- Case 4: TND S1=2048, S2=8192, K=2048

关键约束：
- K必须≤S2（否则sparse_indices中-1被clamp(0)导致gather错误）
- softmax_max/sum从实际QK计算，dtype=FP32
- BSND布局下官方算子S1>1时可能卡住，因此S1>1的case使用TND布局
"""

import torch
import torch_npu
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss

WARMUP_ITERS = 3
BENCH_ITERS = 10

TEST_CASES = [
    {
        "name": "Case 0 - BSND S1=1",
        "B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
        "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
        "d_rope": 64, "K": 2048, "layout": "BSND",
    },
    {
        "name": "Case 1 - TND S1=128",
        "B": 1, "S1": 128, "S2": 2048, "N1": 128, "N2": 1,
        "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
        "d_rope": 64, "K": 2048, "layout": "TND",
    },
    {
        "name": "Case 2 - TND S1=64",
        "B": 1, "S1": 64, "S2": 2048, "N1": 128, "N2": 1,
        "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
        "d_rope": 64, "K": 2048, "layout": "TND",
    },
    {
        "name": "Case 3 - TND S1=512",
        "B": 1, "S1": 512, "S2": 2048, "N1": 128, "N2": 1,
        "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
        "d_rope": 64, "K": 2048, "layout": "TND",
    },
    {
        "name": "Case 4 - TND S1=2048",
        "B": 1, "S1": 2048, "S2": 8192, "N1": 128, "N2": 1,
        "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
        "d_rope": 64, "K": 2048, "layout": "TND",
    },
]


def gen_inputs_corrected(case_config, seed=42):
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
    layout = case_config["layout"]

    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    torch.manual_seed(seed)
    dtype = torch.float16

    query_bsnd = torch.randn(B, S1, N1, DQuery, dtype=dtype).npu()
    key_bsnd = torch.randn(B, S2, N2, DQuery, dtype=dtype).npu()
    query_index_bsnd = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).npu()
    key_index_bsnd = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).npu()
    query_rope_bsnd = torch.randn(B, S1, N1, d_rope, dtype=dtype).npu()
    key_rope_bsnd = torch.randn(B, S2, N2, d_rope, dtype=dtype).npu()
    weights_bsnd = torch.randn(B, S1, Nidx1, dtype=dtype).npu()
    sparse_indices_bsnd = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).npu()

    if layout == "TND":
        query = query_bsnd.squeeze(0)
        key = key_bsnd.squeeze(0)
        query_index = query_index_bsnd.squeeze(0)
        key_index = key_index_bsnd.squeeze(0)
        query_rope = query_rope_bsnd.squeeze(0)
        key_rope = key_rope_bsnd.squeeze(0)
        weights = weights_bsnd.squeeze(0)
        sparse_indices = sparse_indices_bsnd.squeeze(0)
    else:
        query = query_bsnd
        key = key_bsnd
        query_index = query_index_bsnd
        key_index = key_index_bsnd
        query_rope = query_rope_bsnd
        key_rope = key_rope_bsnd
        weights = weights_bsnd
        sparse_indices = sparse_indices_bsnd

    if layout == "TND":
        for s1_idx in range(S1):
            s2_real_size = S2 - S1 + s1_idx + 1
            if s2_real_size <= 0:
                s2_real_size = S2
            s2_real_len = min(s2_real_size, K)
            indices = torch.randperm(s2_real_size)[:s2_real_len]
            sparse_indices[s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
            if s2_real_len < K:
                sparse_indices[s1_idx, 0, s2_real_len:] = -1
        torch.npu.synchronize()
    else:
        s2_real_len = min(S2, K)
        indices = torch.randperm(S2)[:s2_real_len]
        for s1_idx in range(S1):
            sparse_indices[0, s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
            if s2_real_len < K:
                sparse_indices[0, s1_idx, 0, s2_real_len:] = -1
        torch.npu.synchronize()

    with torch.no_grad():
        if layout == "TND":
            k_mqa = key.squeeze(1)
            k_rope_mqa = key_rope.squeeze(1)
            q_full = torch.cat([query, query_rope], dim=-1)
            k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
            k_full_t = k_full.transpose(-1, -2)
            sm_max_list = []
            sm_sum_list = []
            for s1_idx in range(S1):
                row = torch.matmul(q_full[s1_idx:s1_idx+1], k_full_t)
                row_scaled = row.float() * scale_value
                row_max = row_scaled.max(dim=-1).values
                row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
                sm_max_list.append(row_max)
                sm_sum_list.append(row_sum)
            sm_max_val = torch.cat(sm_max_list, dim=0)
            sm_sum_val = torch.cat(sm_sum_list, dim=0)
            softmax_max = sm_max_val.reshape(1, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
            softmax_sum = sm_sum_val.reshape(1, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
            softmax_max = softmax_max.squeeze(0)
            softmax_sum = softmax_sum.squeeze(0)
        else:
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
        "layout": layout,
        "B": B, "S1": S1, "S2": S2, "N1": N1, "N2": N2,
        "Nidx1": Nidx1, "Nidx2": Nidx2, "K": K, "G": G,
    }


def _call_tl(inputs):
    if inputs["layout"] == "TND":
        return sparse_lightning_indexer_grad_kl_loss(
            inputs["query"], inputs["key"],
            inputs["query_index"], inputs["key_index"],
            inputs["weights"], inputs["sparse_indices"],
            inputs["softmax_max"], inputs["softmax_sum"],
            inputs["scale_value"],
            query_rope=inputs["query_rope"],
            key_rope=inputs["key_rope"],
            layout_query="TND",
            actual_seq_lengths_query=[inputs["S1"]],
            actual_seq_lengths_key=[inputs["S2"]],
            sparse_mode=3,
        )
    else:
        return sparse_lightning_indexer_grad_kl_loss(
            inputs["query"], inputs["key"],
            inputs["query_index"], inputs["key_index"],
            inputs["weights"], inputs["sparse_indices"],
            inputs["softmax_max"], inputs["softmax_sum"],
            inputs["scale_value"],
            query_rope=inputs["query_rope"],
            key_rope=inputs["key_rope"],
            layout_query="BSND",
            actual_seq_lengths_query=[inputs["S1"]] * inputs["B"],
            actual_seq_lengths_key=[inputs["S2"]] * inputs["B"],
            sparse_mode=3,
        )


def _call_official(inputs):
    if inputs["layout"] == "TND":
        return torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            inputs["query"], inputs["key"],
            inputs["query_index"], inputs["key_index"],
            inputs["weights"], inputs["sparse_indices"],
            inputs["softmax_max"], inputs["softmax_sum"],
            inputs["scale_value"],
            query_rope=inputs["query_rope"],
            key_rope=inputs["key_rope"],
            actual_seq_qlen=[inputs["S1"]],
            actual_seq_klen=[inputs["S2"]],
            layout="TND", sparse_mode=3,
            pre_tokens=65536, next_tokens=65536,
        )
    else:
        return torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            inputs["query"], inputs["key"],
            inputs["query_index"], inputs["key_index"],
            inputs["weights"], inputs["sparse_indices"],
            inputs["softmax_max"], inputs["softmax_sum"],
            inputs["scale_value"],
            query_rope=inputs["query_rope"],
            key_rope=inputs["key_rope"],
            actual_seq_qlen=[inputs["S1"]] * inputs["B"],
            actual_seq_klen=[inputs["S2"]] * inputs["B"],
            layout="BSND", sparse_mode=3,
            pre_tokens=65536, next_tokens=65536,
        )


def _bench(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
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


def run_single_test(case_idx):
    case = TEST_CASES[case_idx]
    print("=" * 80)
    print(f"测试 {case['name']}")
    print("=" * 80)

    config_str = (f"B={case['B']}, S1={case['S1']}, S2={case['S2']}, K={case['K']}, "
                  f"N1={case['N1']}, Nidx1={case['Nidx1']}, layout={case['layout']}")
    print(f"配置: {config_str}")

    inputs = gen_inputs_corrected(case)

    # --- TileLang ---
    print("\n[1] TileLang 计算...")
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = _call_tl(inputs)
        torch.npu.synchronize()
        print("  完成")
        tl_available = True
    except Exception as e:
        print(f"  失败: {e}")
        tl_available = False

    # --- 官方 ---
    print("\n[2] 官方 Ascend C 算子计算...")
    try:
        d_qi_off, d_ki_off, dw_off, loss_off = _call_official(inputs)
        torch.npu.synchronize()
        print("  完成")
        official_available = True
    except Exception as e:
        print(f"  失败: {e}")
        official_available = False

    # --- 精度对比 ---
    results = {}
    if tl_available and official_available:
        print("\n[3] 精度对比...")
        if inputs["layout"] == "TND":
            S2_real = inputs["S2"]
            if d_ki_tl.shape[0] != d_ki_off.shape[0]:
                d_ki_tl = d_ki_tl[:S2_real]

        for name, tl_val, off_val in [
            ("dQueryIndex", d_qi_tl, d_qi_off),
            ("dKeyIndex", d_ki_tl, d_ki_off),
            ("dWeights", dw_tl, dw_off),
        ]:
            diff = (tl_val.float() - off_val.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_err = (diff / (off_val.float().abs() + 1e-6)).mean().item()
            results[name] = {"max_diff": max_diff, "mean_diff": mean_diff, "rel_err": rel_err}
            print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, rel_err={rel_err:.6e}")

        loss_tl_val = loss_tl.float().item()
        loss_off_val = loss_off.float().item()
        loss_abs_diff = abs(loss_tl_val - loss_off_val)
        loss_rel_err = loss_abs_diff / (abs(loss_off_val) + 1e-6)
        results["loss"] = {
            "abs_diff": loss_abs_diff, "rel_err": loss_rel_err,
            "tl_value": loss_tl_val, "off_value": loss_off_val,
        }
        print(f"  Loss: abs_diff={loss_abs_diff:.6e}, rel_err={loss_rel_err:.6e}, "
              f"tl={loss_tl_val:.6e}, off={loss_off_val:.6e}")
    else:
        print("\n[3] 无法对比（至少一个算子失败）")

    # --- 性能测试 ---
    print(f"\n[4] 性能测试 (warmup={WARMUP_ITERS}, bench={BENCH_ITERS})...")
    if tl_available:
        tl_avg, tl_med, tl_min = _bench(lambda: _call_tl(inputs))
        results["tl_perf"] = {"avg_ms": tl_avg*1e3, "med_ms": tl_med*1e3, "min_ms": tl_min*1e3}
        print(f"  TileLang: avg={tl_avg*1e3:.3f}ms, med={tl_med*1e3:.3f}ms, min={tl_min*1e3:.3f}ms")
    if official_available:
        off_avg, off_med, off_min = _bench(lambda: _call_official(inputs))
        results["off_perf"] = {"avg_ms": off_avg*1e3, "med_ms": off_med*1e3, "min_ms": off_min*1e3}
        print(f"  官方:     avg={off_avg*1e3:.3f}ms, med={off_med*1e3:.3f}ms, min={off_min*1e3:.3f}ms")
    if tl_available and official_available:
        speedup = off_med / tl_med
        results["speedup"] = speedup
        print(f"  加速比:   {speedup:.2f}x (基于中位数)")

    return results


def run_all_tests():
    print("=" * 80)
    print("SparseLightningIndexerGradKLLoss TileLang 泛化精度+性能测试")
    print("=" * 80)
    print(f"共 {len(TEST_CASES)} 个测试用例")
    print()

    all_results = {}
    for i in range(len(TEST_CASES)):
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        results = run_single_test(i)
        all_results[TEST_CASES[i]["name"]] = results
        print()
        try:
            torch.npu.synchronize()
        except Exception:
            pass

    # --- 精度总结 ---
    print("=" * 80)
    print("精度统计 (相对误差 < 2e-3 为达标)")
    print("=" * 80)
    for name, res in all_results.items():
        if res is None or "dQueryIndex" not in res:
            print(f"  {name}: 无法对比")
            continue
        max_rel_err = max(
            res["dQueryIndex"]["rel_err"],
            res["dKeyIndex"]["rel_err"],
            res["dWeights"]["rel_err"],
            res["loss"]["rel_err"],
        )
        status = "✓ 达标" if max_rel_err < 2e-3 else "✗ 未达标"
        print(f"  {name}: {status} (max_rel_err={max_rel_err:.6e})")

    # --- 性能总结 ---
    print("\n" + "=" * 80)
    print("性能统计 (中位数延迟, ms)")
    print("=" * 80)
    header = f"  {'Case':<25s} {'TileLang':>10s} {'官方':>10s} {'加速比':>8s}"
    print(header)
    print("  " + "-" * 55)
    for name, res in all_results.items():
        if res is None:
            continue
        tl_ms = res.get("tl_perf", {}).get("med_ms", None)
        off_ms = res.get("off_perf", {}).get("med_ms", None)
        sp = res.get("speedup", None)
        tl_s = f"{tl_ms:.3f}" if tl_ms is not None else "N/A"
        off_s = f"{off_ms:.3f}" if off_ms is not None else "N/A"
        sp_s = f"{sp:.2f}x" if sp is not None else "N/A"
        print(f"  {name:<25s} {tl_s:>10s} {off_s:>10s} {sp_s:>8s}")


if __name__ == "__main__":
    run_all_tests()
