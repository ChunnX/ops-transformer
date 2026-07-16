#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLASST 稀疏注意力精度测试用例
===============================
测试 kernel 侧硬编码 sparseLambda=-7.0 的 NPU 输出精度，
对比 Python BLASST 仿真（同阈值）和 Python Dense（无跳过）参考。

用法：
    python test_blasst_precision.py
    python test_blasst_precision.py --verbose
    python test_blasst_precision.py --case 1 2 3
"""

import math
import sys
import os
import argparse
import torch
import numpy as np

# 添加 test_res 目录到路径以导入 row_loop_python_blasst
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from row_loop_python_blasst import optimized_blasst_sim, GlobalConfig


# ============================================================
# 测试参数配置
# ============================================================
TEST_CASES = [
    # (batch, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, sparse_mode, description)
    # Case 1: 小规模 MHA dense
    {"B": 1, "Nq": 2, "Nkv": 2, "Sq": 64, "Skv": 64, "D": 128, "sparse_mode": 0, "desc": "MHA dense S=64"},
    # Case 2: 小规模 GQA
    {"B": 1, "Nq": 4, "Nkv": 2, "Sq": 128, "Skv": 128, "D": 128, "sparse_mode": 0, "desc": "GQA dense S=128"},
    # Case 3: GQA rightDownCausal (有mask场景)
    {"B": 1, "Nq": 4, "Nkv": 2, "Sq": 256, "Skv": 256, "D": 128, "sparse_mode": 3, "desc": "GQA causal S=256"},
    # Case 4: 中规模 GQA
    {"B": 1, "Nq": 8, "Nkv": 2, "Sq": 512, "Skv": 512, "D": 128, "sparse_mode": 0, "desc": "GQA dense S=512"},
    # Case 5: 多batch GQA
    {"B": 2, "Nq": 4, "Nkv": 2, "Sq": 128, "Skv": 128, "D": 128, "sparse_mode": 0, "desc": "Multi-batch GQA"},
]

# NPU operator supports BNSD layout by default
INPUT_LAYOUT = "BSND"

# BLASST threshold — must match kernel hardcoded sparseLambda
BLASST_THRESHOLD_LOG = -7.0


def create_test_inputs(case, device="npu:0", dtype=torch.float16):
    """创建测试输入张量"""
    B, Nq, Nkv, Sq, Skv, D = case["B"], case["Nq"], case["Nkv"], case["Sq"], case["Skv"], case["D"]

    query = torch.randn(B, Sq, Nq, D, dtype=dtype, device=device)  # BSND layout
    key = torch.randn(B, Skv, Nkv, D, dtype=dtype, device=device)
    value = torch.randn(B, Skv, Nkv, D, dtype=dtype, device=device)

    actual_seq_q = torch.tensor([Sq] * B, dtype=torch.int64)
    actual_seq_kv = torch.tensor([Skv] * B, dtype=torch.int64)

    scale = 1.0 / math.sqrt(D)

    return query, key, value, actual_seq_q, actual_seq_kv, scale


def run_npu_fia(query, key, value, actual_seq_q, actual_seq_kv, case, scale):
    """运行 NPU FIA 算子（kernel 内已硬编码 sparseLambda=-7.0）"""
    import torch_npu
    B, Nq, Nkv, Sq, Skv, D = case["B"], case["Nq"], case["Nkv"], case["Sq"], case["Skv"], case["D"]
    sparse_mode = case["sparse_mode"]

    # NPU FIA 调用
    # 注意：当前 op_api 没有 sparseLambda 参数，kernel 硬编码 -7.0
    result, _ = torch_npu.npu_fused_infer_attention_score(
        query, key, value,
        actual_seq_lengths=actual_seq_q.tolist(),
        actual_seq_lengths_kv=actual_seq_kv.tolist(),
        num_heads=Nq,
        num_key_value_heads=Nkv,
        input_layout=INPUT_LAYOUT,
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        sparse_mode=sparse_mode,
    )
    torch.npu.synchronize()
    return result


def run_python_dense(query, key, value, actual_seq_q, actual_seq_kv, case, scale):
    """Python Dense FlashAttention 参考（无 BLASST）"""
    B, Nq, Nkv, Sq, Skv, D = case["B"], case["Nq"], case["Nkv"], case["Sq"], case["Skv"], case["D"]
    sparse_mode = case["sparse_mode"]

    # 转 BN2D 格式 (B, N, S, D)
    q_bnsd = query.float().transpose(1, 2)  # BSND -> BNSD
    k_bnsd = key.float().transpose(1, 2)
    v_bnsd = value.float().transpose(1, 2)

    Sq_act = int(actual_seq_q[0].item())
    Skv_act = int(actual_seq_kv[0].item())
    q_eff = q_bnsd[:, :, :Sq_act, :]
    k_eff = k_bnsd[:, :, :Skv_act, :]
    v_eff = v_bnsd[:, :, :Skv_act, :]

    group_size = Nq // Nkv
    K_exp = k_eff.repeat_interleave(group_size, dim=1)
    V_exp = v_eff.repeat_interleave(group_size, dim=1)

    S = torch.matmul(q_eff, K_exp.transpose(-2, -1)) * scale

    if sparse_mode == 3:
        q_range = torch.arange(Sq_act, device=q_eff.device).view(1, 1, -1, 1)
        k_range = torch.arange(Skv_act, device=q_eff.device).view(1, 1, 1, -1)
        mask = q_range < (k_range - (Skv_act - Sq_act))
        S = S.masked_fill(mask, -1e9)

    S_max = S.amax(dim=-1, keepdim=True)
    S_exp = torch.exp(S - S_max)
    S_sum = S_exp.sum(dim=-1, keepdim=True)
    P = S_exp / S_sum
    O = torch.matmul(P.float(), V_exp.float())

    return O.transpose(1, 2)  # BNSD -> BSND


def run_python_blasst(query, key, value, actual_seq_q, actual_seq_kv, case, scale):
    """Python BLASST 仿真（使用 optimized_blasst_sim，与 NPU 同阈值）"""
    B, Nq, Nkv, Sq, Skv, D = case["B"], case["Nq"], case["Nkv"], case["Sq"], case["Skv"], case["D"]
    sparse_mode = case["sparse_mode"]

    if B > 1:
        results = []
        for b in range(B):
            # optimized_blasst_sim expects per-batch tensors
            q_b = query[b].float()      # [Sq, Nq, D]
            k_b = key[b].float()        # [Skv, Nkv, D]
            v_b = value[b].float()      # [Skv, Nkv, D]
            aq_b = actual_seq_q[b]
            akv_b = actual_seq_kv[b]

            out = optimized_blasst_sim(
                q_b, k_b, v_b,
                actual_qlen=aq_b,
                actual_kvlen=akv_b,
                scale=scale,
                threshold_log=BLASST_THRESHOLD_LOG,
                sparse_mode=sparse_mode,
            )[0]  # returns (output, row_sparsity, block_sparsity)
            results.append(out.unsqueeze(0))
        return torch.cat(results, dim=0)
    else:
        q = query[0].float()
        k = key[0].float()
        v = value[0].float()
        return optimized_blasst_sim(
            q, k, v,
            actual_qlen=actual_seq_q[0],
            actual_kvlen=actual_seq_kv[0],
            scale=scale,
            threshold_log=BLASST_THRESHOLD_LOG,
            sparse_mode=sparse_mode,
        )[0].unsqueeze(0)  # returns (output, row_sparsity, block_sparsity)


def compute_metrics(result, reference, name=""):
    """计算精度指标"""
    result_f = result.float()
    ref_f = reference.float()

    diff = (result_f - ref_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Cosine similarity
    flat_r = result_f.reshape(-1)
    flat_ref = ref_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_r.unsqueeze(0), flat_ref.unsqueeze(0)
    ).item()

    # Relative error (avoid div by zero)
    rel_diff = diff / (ref_f.abs() + 1e-8)
    mean_rel = rel_diff.mean().item()
    max_rel = rel_diff.max().item()

    return {
        "name": name,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
    }


def print_metrics(metrics):
    """打印精度指标"""
    print(f"  [{metrics['name']}]")
    print(f"    max_diff   = {metrics['max_diff']:.6f}")
    print(f"    mean_diff  = {metrics['mean_diff']:.8f}")
    print(f"    cos_sim    = {metrics['cos_sim']:.6f}")
    print(f"    max_rel    = {metrics['max_rel']:.6f}")
    print(f"    mean_rel   = {metrics['mean_rel']:.8f}")


def test_case(case_idx, verbose=False, no_npu=False):
    """测试单个 case"""
    case = TEST_CASES[case_idx]
    B, Nq, Nkv, Sq, Skv, D = case["B"], case["Nq"], case["Nkv"], case["Sq"], case["Skv"], case["D"]
    desc = case["desc"]

    print(f"\n{'='*60}")
    print(f"Case {case_idx+1}: {desc}")
    print(f"  B={B} Nq={Nq} Nkv={Nkv} Sq={Sq} Skv={Skv} D={D}")
    print(f"  sparse_mode={case['sparse_mode']}")
    print(f"  BLASST threshold_log={BLASST_THRESHOLD_LOG}")
    print(f"{'='*60}")

    device = "cpu" if no_npu else "npu:0"
    query, key, value, act_q, act_kv, scale = create_test_inputs(case, device=device)

    # 1. Python Dense 参考
    print("\n[1/3] Computing Python Dense reference...")
    out_dense = run_python_dense(query, key, value, act_q, act_kv, case, scale)

    # 2. Python BLASST (同阈值 -7.0)
    print("[2/3] Computing Python BLASST simulation (threshold=-7.0)...")
    out_blasst = run_python_blasst(query, key, value, act_q, act_kv, case, scale)

    if no_npu:
        # NPU 不可用: 只比较 Python 参考
        m_blasst_dense = compute_metrics(out_blasst, out_dense, "Python_BLASST(sim) vs Python_Dense")
        print("\n--- Precision Metrics ---")
        print_metrics(m_blasst_dense)
        return None, None, m_blasst_dense

    # 3. NPU FIA (kernel 内 sparseLambda=-7.0 硬编码)
    print("[3/3] Running NPU FIA (BLASST hardcoded)...")
    query_npu = query.to("npu:0")
    key_npu = key.to("npu:0")
    value_npu = value.to("npu:0")
    out_npu = run_npu_fia(query_npu, key_npu, value_npu, act_q, act_kv, case, scale)

    # 4. 比较
    print("\n--- Precision Metrics ---")

    # NPU vs Python Dense: 衡量 BLASST 引入的误差
    m_npu_dense = compute_metrics(out_npu.cpu(), out_dense, "NPU_BLASST vs Python_Dense")
    print_metrics(m_npu_dense)

    # NPU vs Python BLASST: 衡量 NPU 实现精度
    m_npu_blasst = compute_metrics(out_npu.cpu(), out_blasst, "NPU_BLASST vs Python_BLASST(sim)")
    print_metrics(m_npu_blasst)

    # Python BLASST vs Python Dense: BLASST 仿真自身的误差
    m_blasst_dense = compute_metrics(out_blasst, out_dense, "Python_BLASST(sim) vs Python_Dense")
    print_metrics(m_blasst_dense)

    # 5. 判定
    print("\n--- Verdict ---")
    # NPU vs Python Dense: 期望有一定差异（BLASST 稀疏化），但 cos_sim 应 > 0.99
    if m_npu_dense["cos_sim"] > 0.99:
        print(f"  NPU vs Dense: PASS (cos_sim={m_npu_dense['cos_sim']:.6f} > 0.99)")
    else:
        print(f"  NPU vs Dense: WARN (cos_sim={m_npu_dense['cos_sim']:.6f})")

    # NPU vs Python BLASST: 应该高度一致
    if m_npu_blasst["cos_sim"] > 0.999:
        print(f"  NPU vs BLASST(sim): PASS (cos_sim={m_npu_blasst['cos_sim']:.6f} > 0.999)")
    elif m_npu_blasst["cos_sim"] > 0.99:
        print(f"  NPU vs BLASST(sim): PASS_WITH_NOTES (cos_sim={m_npu_blasst['cos_sim']:.6f} > 0.99)")
    else:
        print(f"  NPU vs BLASST(sim): CHECK (cos_sim={m_npu_blasst['cos_sim']:.6f})")

    return m_npu_dense, m_npu_blasst, m_blasst_dense


def main():
    parser = argparse.ArgumentParser(description="BLASST 稀疏注意力精度测试")
    parser.add_argument("--case", type=int, nargs="+", help="指定测试 case 编号 (1-based)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--no-npu", action="store_true", help="仅运行 Python 参考（不需要 NPU）")
    args = parser.parse_args()

    if args.case:
        cases = [c - 1 for c in args.case if 1 <= c <= len(TEST_CASES)]
    else:
        cases = list(range(len(TEST_CASES)))

    if not args.no_npu:
        try:
            import torch_npu
            torch_npu.npu.set_device(0)
            print(f"NPU device: {torch_npu.npu.get_device_name(0)}")
        except Exception as e:
            print(f"NPU 初始化失败: {e}")
            print("使用 --no-npu 可仅运行 Python 参考实现")
            return

    all_results = []
    for ci in cases:
        try:
            r = test_case(ci, verbose=args.verbose, no_npu=args.no_npu)
            all_results.append(r)
        except Exception as e:
            print(f"\nCase {ci+1} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(all_results)}/{len(cases)} cases completed")
    for i, (m_d, m_b, m_bd) in enumerate(all_results):
        ci = cases[i]
        print(f"  Case {ci+1}: NPUvsDense cos={m_d['cos_sim']:.4f}, "
              f"NPUvsBLASST cos={m_b['cos_sim']:.4f}, "
              f"BLASSTvsDense cos={m_bd['cos_sim']:.4f}")


if __name__ == "__main__":
    main()
