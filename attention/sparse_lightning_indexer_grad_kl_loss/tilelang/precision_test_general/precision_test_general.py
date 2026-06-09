#!/usr/bin/env python3
"""
泛化精度测试 - SparseLightningIndexerGradKLLoss TileLang
共6个测试用例，覆盖关键维度：
- Case 0: 现有配置 S1=1, S2=2048, K=2048 (BSND)
- Case 1: 官方风格 S1=128, S2=128, K=2048 (TND, 动态填充)
- Case 2: 短序列 S1=64, S2=512, K=1024
- Case 3: 中序列 S1=512, S2=2048, K=2048
- Case 4: 长序列 S1=2048, S2=8192, K=4096 (动态填充)
- Case 5: 大TopK S1=512, S2=2048, K=8192

关键改进：
- weights固定范围[-0.05, 0.05]
- sparse_indices使用randperm（不重复索引）
- softmax_max/sum从实际QK计算（修正官方错误）
"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss


TEST_CASES = [
    {
        "name": "Case 0 - 现有配置",
        "B": 1,
        "S1": 1,
        "S2": 2048,
        "N1": 128,
        "N2": 1,
        "Nidx1": 64,
        "Nidx2": 1,
        "DQuery": 512,
        "DQueryIndex": 128,
        "d_rope": 64,
        "K": 2048,
        "layout": "BSND",
        "dynamic_fill": False,
    },
    {
        "name": "Case 1 - 官方风格TND",
        "B": 1,
        "S1": 128,
        "S2": 128,
        "N1": 64,
        "N2": 1,
        "Nidx1": 64,
        "Nidx2": 1,
        "DQuery": 512,
        "DQueryIndex": 128,
        "d_rope": 64,
        "K": 2048,
        "layout": "TND",
        "dynamic_fill": True,
    },
    {
        "name": "Case 2 - 短序列",
        "B": 1,
        "S1": 64,
        "S2": 512,
        "N1": 64,
        "N2": 1,
        "Nidx1": 32,
        "Nidx2": 1,
        "DQuery": 512,
        "DQueryIndex": 128,
        "d_rope": 64,
        "K": 1024,
        "layout": "BSND",
        "dynamic_fill": False,
    },
    {
        "name": "Case 3 - 中序列",
        "B": 1,
        "S1": 512,
        "S2": 2048,
        "N1": 64,
        "N2": 1,
        "Nidx1": 32,
        "Nidx2": 1,
        "DQuery": 512,
        "DQueryIndex": 128,
        "d_rope": 64,
        "K": 2048,
        "layout": "BSND",
        "dynamic_fill": False,
    },
    {
        "name": "Case 4 - 长序列动态填充",
        "B": 1,
        "S1": 2048,
        "S2": 8192,
        "N1": 128,
        "N2": 1,
        "Nidx1": 64,
        "Nidx2": 1,
        "DQuery": 512,
        "DQueryIndex": 128,
        "d_rope": 64,
        "K": 4096,
        "layout": "BSND",
        "dynamic_fill": True,
    },
    {
        "name": "Case 5 - 大TopK",
        "B": 1,
        "S1": 512,
        "S2": 2048,
        "N1": 64,
        "N2": 1,
        "Nidx1": 32,
        "Nidx2": 1,
        "DQuery": 512,
        "DQueryIndex": 128,
        "d_rope": 64,
        "K": 8192,
        "layout": "BSND",
        "dynamic_fill": False,
    },
]


def gen_inputs_corrected(case_config, seed=42):
    """
    修正后的入参生成函数
    - weights范围[-0.05, 0.05]
    - sparse_indices使用randperm（不重复）
    - softmax_max/sum从实际QK计算
    """
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
    dynamic_fill = case_config["dynamic_fill"]

    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    torch.manual_seed(seed)
    dtype = torch.float16

    if layout == "TND":
        query = torch.randn(S1, N1, DQuery, dtype=dtype).npu()
        key = torch.randn(S2, N2, DQuery, dtype=dtype).npu()
        query_index = torch.randn(S1, Nidx1, DQueryIndex, dtype=dtype).npu()
        key_index = torch.randn(S2, Nidx2, DQueryIndex, dtype=dtype).npu()
        query_rope = torch.randn(S1, N1, d_rope, dtype=dtype).npu()
        key_rope = torch.randn(S2, N2, d_rope, dtype=dtype).npu()
        weights = torch.randn(S1, Nidx1, dtype=dtype).npu()
        sparse_indices = torch.zeros(S1, Nidx2, K, dtype=torch.int32).npu()
    else:
        query = torch.randn(B, S1, N1, DQuery, dtype=dtype).npu()
        key = torch.randn(B, S2, N2, DQuery, dtype=dtype).npu()
        query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).npu()
        key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).npu()
        query_rope = torch.randn(B, S1, N1, d_rope, dtype=dtype).npu()
        key_rope = torch.randn(B, S2, N2, d_rope, dtype=dtype).npu()
        weights = torch.randn(B, S1, Nidx1, dtype=dtype).npu()
        sparse_indices = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).npu()

    weights = weights * 0.0167

    if layout == "TND":
        for s1_idx in range(S1):
            s2_real_size = S2 - S1 + s1_idx + 1
            if dynamic_fill:
                if s2_real_size <= 0:
                    s2_real_size = S2
                s2_real_len = min(s2_real_size, K)
                indices = torch.randperm(s2_real_size)[:s2_real_len]
                sparse_indices[s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
                sparse_indices[s1_idx, 0, s2_real_len:] = -1
            else:
                s2_real_len = min(S2, K)
                indices = torch.randperm(S2)[:s2_real_len]
                sparse_indices[s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
                if s2_real_len < K:
                    sparse_indices[s1_idx, 0, s2_real_len:] = -1
    else:
        for s1_idx in range(S1):
            s2_real_size = S2 - S1 + s1_idx + 1
            if dynamic_fill:
                if s2_real_size <= 0:
                    s2_real_size = S2
                s2_real_len = min(s2_real_size, K)
                indices = torch.randperm(s2_real_size)[:s2_real_len]
                sparse_indices[0, s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
                sparse_indices[0, s1_idx, 0, s2_real_len:] = -1
            else:
                s2_real_len = min(S2, K)
                indices = torch.randperm(S2)[:s2_real_len]
                sparse_indices[0, s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
                if s2_real_len < K:
                    sparse_indices[0, s1_idx, 0, s2_real_len:] = -1

    with torch.no_grad():
        if layout == "TND":
            k_mqa = key.squeeze(1)
            k_rope_mqa = key_rope.squeeze(1)
            q_full = torch.cat([query, query_rope], dim=-1)
            k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
            k_full_t = k_full.transpose(-1, -2).unsqueeze(0)
            bmm1_full = torch.matmul(q_full.unsqueeze(0), k_full_t)
            bmm1_scaled = bmm1_full.float() * scale_value
            sm_max_val = bmm1_scaled.max(dim=-1).values
            sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
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
            bmm1_full = torch.matmul(q_full, k_full_t)
            bmm1_scaled = bmm1_full.float() * scale_value
            sm_max_val = bmm1_scaled.max(dim=-1).values
            sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
            softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
            softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()

    return {
        "query": query,
        "key": key,
        "query_index": query_index,
        "key_index": key_index,
        "weights": weights,
        "sparse_indices": sparse_indices,
        "softmax_max": softmax_max,
        "softmax_sum": softmax_sum,
        "scale_value": scale_value,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "layout": layout,
        "B": B,
        "S1": S1,
        "S2": S2,
        "N1": N1,
        "N2": N2,
        "Nidx1": Nidx1,
        "Nidx2": Nidx2,
        "K": K,
        "G": G,
    }


def run_single_test(case_idx):
    case = TEST_CASES[case_idx]
    print("=" * 80)
    print(f"测试 {case['name']}")
    print("=" * 80)

    config_str = f"B={case['B']}, S1={case['S1']}, S2={case['S2']}, K={case['K']}, " \
                f"N1={case['N1']}, Nidx1={case['Nidx1']}, layout={case['layout']}, " \
                f"dynamic_fill={case['dynamic_fill']}"
    print(f"配置: {config_str}")

    inputs = gen_inputs_corrected(case)

    print("\n[1] TileLang 计算...")
    try:
        if inputs["layout"] == "TND":
            d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
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
            d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
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
        torch.npu.synchronize()
        print("  完成")
        tl_available = True
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
        tl_available = False

    print("\n[2] 官方 Ascend C 算子计算...")
    try:
        if inputs["layout"] == "TND":
            d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                inputs["query"], inputs["key"],
                inputs["query_index"], inputs["key_index"],
                inputs["weights"], inputs["sparse_indices"],
                inputs["softmax_max"], inputs["softmax_sum"],
                inputs["scale_value"],
                query_rope=inputs["query_rope"],
                key_rope=inputs["key_rope"],
                actual_seq_qlen=[inputs["S1"]],
                actual_seq_klen=[inputs["S2"]],
                layout="TND",
                sparse_mode=3,
                pre_tokens=65536,
                next_tokens=65536,
            )
        else:
            d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                inputs["query"], inputs["key"],
                inputs["query_index"], inputs["key_index"],
                inputs["weights"], inputs["sparse_indices"],
                inputs["softmax_max"], inputs["softmax_sum"],
                inputs["scale_value"],
                query_rope=inputs["query_rope"],
                key_rope=inputs["key_rope"],
                actual_seq_qlen=[inputs["S1"]] * inputs["B"],
                actual_seq_klen=[inputs["S2"]] * inputs["B"],
                layout="BSND",
                sparse_mode=3,
                pre_tokens=65536,
                next_tokens=65536,
            )
        torch.npu.synchronize()
        print("  完成")
        official_available = True
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
        official_available = False

    if tl_available and official_available:
        print("\n[3] 精度对比...")
        results = {}
        for name, tl_val, off_val in [
            ("dQueryIndex", d_qi_tl, d_qi_off),
            ("dKeyIndex", d_ki_tl, d_ki_off),
            ("dWeights", dw_tl, dw_off),
        ]:
            diff = (tl_val.float() - off_val.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_err = (diff / (off_val.float().abs() + 1e-6)).mean().item()
            results[name] = {
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "rel_err": rel_err,
            }
            print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, rel_err={rel_err:.6e}")

        loss_tl_val = loss_tl.float().item()
        loss_off_val = loss_off.float().item()
        loss_abs_diff = abs(loss_tl_val - loss_off_val)
        loss_rel_err = loss_abs_diff / (abs(loss_off_val) + 1e-6)
        results["loss"] = {
            "abs_diff": loss_abs_diff,
            "rel_err": loss_rel_err,
            "tl_value": loss_tl_val,
            "off_value": loss_off_val,
        }
        print(f"  Loss: abs_diff={loss_abs_diff:.6e}, rel_err={loss_rel_err:.6e}, " \
              f"tl={loss_tl_val:.6e}, off={loss_off_val:.6e}")

        return results
    else:
        print("\n[3] 无法对比（至少一个算子失败）")
        return None


def run_all_tests():
    print("=" * 80)
    print("SparseLightningIndexerGradKLLoss TileLang 泛化测试")
    print("=" * 80)
    print(f"共 {len(TEST_CASES)} 个测试用例")
    print()

    all_results = {}
    passed_count = 0
    failed_count = 0

    for i in range(len(TEST_CASES)):
        results = run_single_test(i)
        all_results[TEST_CASES[i]["name"]] = results

        if results is not None:
            passed_count += 1
        else:
            failed_count += 1
        print()

    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总计: {len(TEST_CASES)} 个测试")
    print(f"成功: {passed_count} 个")
    print(f"失败: {failed_count} 个")

    if passed_count > 0:
        print("\n精度统计 (相对误差 < 2e-3 为达标):")
        for name, res in all_results.items():
            if res is None:
                print(f"  {name}: 失败")
                continue

            max_rel_err = max(
                res["dQueryIndex"]["rel_err"],
                res["dKeyIndex"]["rel_err"],
                res["dWeights"]["rel_err"],
                res["loss"]["rel_err"],
            )
            status = "✓ 达标" if max_rel_err < 2e-3 else "✗ 未达标"
            print(f"  {name}: {status} (max_rel_err={max_rel_err:.6e})")


if __name__ == "__main__":
    run_all_tests()