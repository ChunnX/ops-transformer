#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
调试脚本：打印QLI golden计算的中间结果
用于验证计算顺序修改的正确性

计算公式（按README）：
out = Top-k{ [1]_{g} @ [(W) ⊙ ReLU((Scale_Q ⊙ Scale_K) ⊙ (Q @ K^T))] }

正确计算顺序：
1. Q @ K^T 得到相关性
2. 相关性结果与 Scale_Q 和 Scale_K 相乘（反量化）
3. ReLU 过滤负值
4. 与权重 W 相乘（广播）
5. 沿g的方向求和
6. 取 Top-k

当前代码的计算顺序（优化后）：
1. Q @ K^T
2. weight * qscale（合并）
3. ReLU
4. (weight * qscale) @ ReLU(Q@K^T)（矩阵乘法实现广播+组内求和）
5. 结果 * kscale（最后乘kscale）
"""

import torch
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_simple_test_data_fp8():
    """创建简单的测试数据 - FP8版本"""
    batch_size = 1
    q_seq = 2
    k_seq = 4
    q_head_num = 4  # g = q_head_num / k_head_num = 4
    k_head_num = 1
    head_dim = 4
    
    torch.manual_seed(42)
    
    query = torch.randn(1, 2, 4, 4).to(torch.float8_e4m3fn)
    key = torch.randn(1, 4, 1, 4).to(torch.float8_e4m3fn)
    weights = torch.randn(1, 2, 4).to(torch.bfloat16)
    q_scale = torch.randn(1, 2, 4).abs().to(torch.float32) + 0.1
    k_scale = torch.randn(1, 4, 1).abs().to(torch.float32) + 0.1
    
    act_seq_q = [2]
    act_seq_k = [4.0]
    sparse_count = 2
    sparse_mode = 0
    
    return query, key, weights, q_scale, k_scale, act_seq_q, act_seq_k, sparse_count, sparse_mode

def create_simple_test_data_int8():
    """创建简单的测试数据 - INT8版本"""
    batch_size = 1
    q_seq = 2
    k_seq = 4
    q_head_num = 4
    k_head_num = 1
    head_dim = 4
    
    torch.manual_seed(42)
    
    query = torch.randint(-10, 10, (1, 2, 4, 4)).to(torch.int8)
    key = torch.randint(-10, 10, (1, 4, 1, 4)).to(torch.int8)
    weights = torch.randn(1, 2, 4).to(torch.bfloat16)
    q_scale = torch.randn(1, 2, 4).abs().to(torch.float32) + 0.1
    k_scale = torch.randn(1, 4, 1).abs().to(torch.float32) + 0.1
    
    act_seq_q = [2]
    act_seq_k = [4.0]
    sparse_count = 2
    sparse_mode = 0
    
    return query, key, weights, q_scale, k_scale, act_seq_q, act_seq_k, sparse_count, sparse_mode

def print_tensor_info(name, tensor, detailed=True):
    """打印tensor的基本信息"""
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    if detailed and tensor.numel() < 100:
        if tensor.dtype in [torch.float8_e4m3fn, torch.int8]:
            print(f"  values: {tensor.to(torch.float32)}")
        else:
            print(f"  values:\n{tensor}")
    else:
        print(f"  min: {tensor.min().item() if tensor.numel() > 0 else 'N/A'}, max: {tensor.max().item() if tensor.numel() > 0 else 'N/A'}")

def manual_calc_original_fp8(query, key, weights, q_scale, k_scale, group_size=4):
    """手动计算原始实现（优化后的顺序）- FP8"""
    print("\n" + "=" * 70)
    print("【原始实现】FP8版本 - 优化后的计算顺序")
    print("=" * 70)
    
    N = query.shape[2]  # q_head_num
    S1 = query.shape[1]  # q_seq
    S2 = key.shape[1]    # k_seq
    g = group_size       # N1 / N2
    
    cur_q = query.to(torch.float32)  # [1, 1, N, S1, D] -> [B, N, S, D] -> cur_q is [1, N, S, D]
    cur_k = key.to(torch.float32)
    cur_wt = weights.to(torch.float32)
    cur_q_scale = q_scale.to(torch.float32)
    cur_k_scale = k_scale.to(torch.float32)
    
    print_tensor_info("Step 0: 输入数据")
    print_tensor_info("  Query (float32)", cur_q.squeeze(0))
    print_tensor_info("  Key (float32)", cur_k.squeeze(0))
    print_tensor_info("  Weights (float32)", cur_wt.squeeze(0))
    print_tensor_info("  Q_scale", cur_q_scale.squeeze(0))
    print_tensor_info("  K_scale", cur_k_scale.squeeze(0))
    
    # 转换为BNSD格式（假设已经是）
    cur_q_bnsd = cur_q.permute(0, 2, 1, 3)  # [1, N, S1, D]
    cur_k_bnsd = cur_k.permute(0, 2, 1, 3)  # [1, 1, S2, D] -> [1, N, S2, D] after broadcast
    cur_wt_bnsd = cur_wt.permute(0, 2, 1).unsqueeze(-1)  # [1, N, S1, 1]
    cur_q_scale_bnsd = q_scale.permute(0, 2, 1).unsqueeze(-1)  # [1, N, S1, 1]
    cur_k_scale_bnsd = k_scale.permute(0, 2, 1).unsqueeze(-1)  # [1, 1, S2, 1]
    
    # 广播k的n轴
    if N != 1:
        cur_k_bnsd = cur_k_bnsd.repeat(1, N, 1, 1)
        cur_k_scale_bnsd = cur_k_scale_bnsd.repeat(1, N, 1, 1)
    
    print_tensor_info("\nStep 1: Q @ K^T (矩阵乘法)", None)
    qk_bmm_res = torch.matmul(
        cur_q_bnsd.squeeze(0),  # [N, S1, D]
        cur_k_bnsd.squeeze(0).transpose(-1, -2)  # [N, D, S2]
    ).unsqueeze(0)  # [1, N, S1, S2]
    print_tensor_info("  QK_result (Q @ K^T)", qk_bmm_res)
    
    print_tensor_info("\nStep 2: weight * q_scale (合并计算)", None)
    cur_w = cur_wt_bnsd * cur_q_scale_bnsd  # [1, N, S1, 1]
    print_tensor_info("  W * Q_scale", cur_w)
    
    print_tensor_info("\nStep 3: ReLU(QK)", None)
    qk_relu = qk_bmm_res.clamp_min(0.0)
    print_tensor_info("  ReLU(QK)", qk_relu)
    
    print_tensor_info("\nStep 4: (W*Qscale) @ ReLU(QK) - 组内求和", None)
    print("  当前实现使用矩阵乘法实现广播和组内求和:")
    print("  cur_w shape: [1, N, S1, 1] -> permute为 [1, S1, 1, N]")
    print("  qk_relu shape: [1, N, S1, S2] -> permute为 [1, S1, N, S2]")
    
    cur_w_permuted = cur_w.permute(0, 2, 3, 1)  # [1, S1, 1, N]
    qk_relu_permuted = qk_relu.permute(0, 2, 1, 3)  # [1, S1, N, S2]
    
    print_tensor_info("  cur_w (permuted)", cur_w_permuted)
    print_tensor_info("  qk_relu (permuted)", qk_relu_permuted)
    
    brc_vmul = torch.matmul(
        cur_w_permuted.squeeze(0),  # [S1, 1, N]
        qk_relu_permuted.squeeze(0)  # [S1, N, S2]
    ).unsqueeze(0)  # [1, S1, 1, S2]
    
    print_tensor_info("  结果 (组内求和后)", brc_vmul)
    print("  注意：这里通过矩阵乘法实现了沿N轴(g轴)的求和")
    
    print_tensor_info("\nStep 5: 结果 * k_scale (最后乘)", None)
    reduce_sum = brc_vmul * cur_k_scale_bnsd[:, 0:1, :, :]  # k_scale是 [1, 1, S2, 1]
    print_tensor_info("  最终结果 (乘K_scale)", reduce_sum)
    
    print("\n总结：原始实现的计算顺序")
    print("  QK = Q @ K^T")
    print("  W' = W * Q_scale")
    print("  Result = W' @ ReLU(QK)  (同时实现广播和组内求和)")
    print("  Final = Result * K_scale")
    
    return reduce_sum

def manual_calc_correct_fp8(query, key, weights, q_scale, k_scale, group_size=4):
    """手动计算正确的顺序（按公式）- FP8"""
    print("\n" + "=" * 70)
    print("【修改后实现】FP8版本 - 按README公式的正确计算顺序")
    print("=" * 70)
    
    N = query.shape[2]  # q_head_num
    S1 = query.shape[1]  # q_seq
    S2 = key.shape[1]    # k_seq
    g = group_size       # N1 / N2
    N2 = 1  # k_head_num
    
    cur_q = query.to(torch.float32)
    cur_k = key.to(torch.float32)
    cur_wt = weights.to(torch.float32)
    cur_q_scale = q_scale.to(torch.float32)
    cur_k_scale = k_scale.to(torch.float32)
    
    print_tensor_info("Step 0: 输入数据")
    print_tensor_info("  Query (float32)", cur_q.squeeze(0))
    print_tensor_info("  Key (float32)", cur_k.squeeze(0))
    print_tensor_info("  Weights (float32)", cur_wt.squeeze(0))
    print_tensor_info("  Q_scale", cur_q_scale.squeeze(0))
    print_tensor_info("  K_scale", cur_k_scale.squeeze(0))
    
    # 转换为BNSD格式
    cur_q_bnsd = cur_q.permute(0, 2, 1, 3)  # [1, N, S1, D]
    cur_k_bnsd = cur_k.permute(0, 2, 1, 3)  # [1, 1, S2, D]
    cur_wt_bnsd = cur_wt.permute(0, 2, 1).unsqueeze(-1)  # [1, N, S1, 1]
    cur_q_scale_bnsd = q_scale.permute(0, 2, 1).unsqueeze(-1)  # [1, N, S1, 1]
    cur_k_scale_bnsd = k_scale.permute(0, 2, 1).unsqueeze(-1)  # [1, 1, S2, 1]
    
    print_tensor_info("\nStep 1: Q @ K^T", None)
    qk_bmm_res = torch.matmul(
        cur_q_bnsd.squeeze(0),
        cur_k_bnsd.squeeze(0).transpose(-1, -2)
    ).unsqueeze(0)  # [1, N, S1, S2]
    print_tensor_info("  QK_result", qk_bmm_res)
    
    print_tensor_info("\nStep 2: QK * Q_scale * K_scale (反量化)", None)
    print("  按公式：QK先乘Scale_Q和Scale_K")
    print("  Q_scale shape: [1, N, S1, 1]")
    print("  K_scale shape: [1, 1, S2, 1] -> unsqueeze扩展")
    
    k_scale_expanded = cur_k_scale_bnsd.unsqueeze(1)  # [1, 1, 1, S2, 1] -> [1, N, 1, S2, 1]
    k_scale_broadcast = cur_k_scale_bnsd.expand(1, N, S2, 1)  # [1, N, S2, 1]
    
    qk_scaled = qk_bmm_res * cur_q_scale_bnsd * k_scale_broadcast
    print_tensor_info("  QK_scaled (QK * Qscale * Kscale)", qk_scaled)
    
    print_tensor_info("\nStep 3: ReLU", None)
    qk_relu = qk_scaled.clamp_min(0.0)
    print_tensor_info("  ReLU(QK_scaled)", qk_relu)
    
    print_tensor_info("\nStep 4: Weight广播乘法", None)
    print("  Weight shape: [1, N, S1, 1]")
    print("  QK_relu shape: [1, N, S1, S2]")
    
    result_weighted = qk_relu * cur_wt_bnsd
    print_tensor_info("  Weighted_result", result_weighted)
    
    print_tensor_info("\nStep 5: 组内求和 (沿N轴/g轴)", None)
    print(f"  g = {g} (N1/N2)")
    print("  将 [1, N, S1, S2] reshape为 [1, N2, S1, S2, g]")
    print("  然后沿最后一维求和")
    
    result_sum = result_weighted.reshape(1, N2, S1, S2, g).sum(dim=-1)
    print_tensor_info("  最终结果 (组内求和)", result_sum)
    
    print("\n总结：修改后实现的计算顺序")
    print("  QK = Q @ K^T")
    print("  QK_scaled = QK * Q_scale * K_scale  (先反量化)")
    print("  QK_relu = ReLU(QK_scaled)")
    print("  Weighted = QK_relu * W  (权重广播)")
    print("  Result = sum(Weighted, axis=g)  (组内求和)")
    
    return result_sum

def compare_results(original_result, modified_result):
    """对比两种实现的结果"""
    print("\n" + "=" * 70)
    print("【结果对比】")
    print("=" * 70)
    
    print_tensor_info("原始实现结果", original_result)
    print_tensor_info("修改后结果", modified_result)
    
    diff = torch.abs(original_result - modified_result)
    print_tensor_info("绝对差异", diff)
    
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n最大差异: {max_diff}")
    print(f"平均差异: {mean_diff}")
    
    if max_diff < 1e-5:
        print("\n✓ 结果基本一致（差异极小）")
    else:
        print(f"\n✗ 结果有明显差异，最大差异为 {max_diff}")

def manual_calc_original_int8(query, key, weights, q_scale, k_scale, group_size=4):
    """手动计算原始实现 - INT8"""
    print("\n" + "=" * 70)
    print("【原始实现】INT8版本 - 优化后的计算顺序")
    print("=" * 70)
    
    N = query.shape[2]
    S1 = query.shape[1]
    S2 = key.shape[1]
    g = group_size
    
    cur_q = query.to(torch.int32)
    cur_k = key.to(torch.int32)
    cur_wt = weights.to(torch.float16)
    cur_q_scale = q_scale.to(torch.float16)
    cur_k_scale = k_scale.to(torch.float16)
    
    print_tensor_info("Step 0: 输入数据")
    print_tensor_info("  Query (int8)", query.squeeze(0))
    print_tensor_info("  Key (int8)", key.squeeze(0))
    print_tensor_info("  Weights", cur_wt.squeeze(0))
    print_tensor_info("  Q_scale", cur_q_scale.squeeze(0))
    print_tensor_info("  K_scale", cur_k_scale.squeeze(0))
    
    # BNSD格式
    cur_q_bnsd = cur_q.permute(0, 2, 1, 3)
    cur_k_bnsd = cur_k.permute(0, 2, 1, 3)
    cur_wt_bnsd = cur_wt.permute(0, 2, 1).unsqueeze(-1)
    cur_q_scale_bnsd = q_scale.permute(0, 2, 1).unsqueeze(-1).to(torch.float16)
    cur_k_scale_bnsd = k_scale.permute(0, 2, 1).unsqueeze(-1).to(torch.float16)
    
    if N != 1:
        cur_k_bnsd = cur_k_bnsd.repeat(1, N, 1, 1)
        cur_k_scale_bnsd = cur_k_scale_bnsd.repeat(1, N, 1, 1)
    
    print_tensor_info("\nStep 1: Q @ K^T (int32)", None)
    qk_bmm_res = torch.matmul(
        cur_q_bnsd.squeeze(0),
        cur_k_bnsd.squeeze(0).transpose(-1, -2)
    ).unsqueeze(0)
    print_tensor_info("  QK_result (int32)", qk_bmm_res)
    
    print_tensor_info("\nStep 2: QK / 1024 + ReLU", None)
    qk_relu = (qk_bmm_res.to(torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
    print_tensor_info("  QK/1024 + ReLU (float16)", qk_relu)
    
    print_tensor_info("\nStep 3: weight * q_scale", None)
    cur_w = cur_wt_bnsd * cur_q_scale_bnsd
    print_tensor_info("  W * Q_scale", cur_w)
    
    print_tensor_info("\nStep 4: (W*Qscale) @ QK_relu", None)
    cur_w_permuted = cur_w.permute(0, 2, 3, 1)
    qk_relu_permuted = qk_relu.permute(0, 2, 1, 3)
    
    brc_vmul = torch.matmul(
        cur_w_permuted.squeeze(0),
        qk_relu_permuted.squeeze(0)
    ).unsqueeze(0)
    
    print_tensor_info("  结果", brc_vmul)
    
    print_tensor_info("\nStep 5: * k_scale", None)
    reduce_sum = brc_vmul * cur_k_scale_bnsd[:, 0:1, :, :]
    print_tensor_info("  最终结果", reduce_sum)
    
    return reduce_sum

def manual_calc_correct_int8(query, key, weights, q_scale, k_scale, group_size=4):
    """手动计算正确顺序 - INT8"""
    print("\n" + "=" * 70)
    print("【修改后实现】INT8版本 - 按公式的正确计算顺序")
    print("=" * 70)
    
    N = query.shape[2]
    S1 = query.shape[1]
    S2 = key.shape[1]
    g = group_size
    N2 = 1
    
    cur_q = query.to(torch.int32)
    cur_k = key.to(torch.int32)
    cur_wt = weights.to(torch.float16)
    cur_q_scale = q_scale.to(torch.float16)
    cur_k_scale = k_scale.to(torch.float16)
    
    print_tensor_info("Step 0: 输入数据")
    print_tensor_info("  Query (int8)", query.squeeze(0))
    print_tensor_info("  Key (int8)", key.squeeze(0))
    
    cur_q_bnsd = cur_q.permute(0, 2, 1, 3)
    cur_k_bnsd = cur_k.permute(0, 2, 1, 3)
    cur_wt_bnsd = cur_wt.permute(0, 2, 1).unsqueeze(-1)
    cur_q_scale_bnsd = q_scale.permute(0, 2, 1).unsqueeze(-1).to(torch.float16)
    cur_k_scale_bnsd = k_scale.permute(0, 2, 1).unsqueeze(-1).to(torch.float16)
    
    k_scale_broadcast = cur_k_scale_bnsd.expand(1, N, S2, 1)
    
    print_tensor_info("\nStep 1: Q @ K^T", None)
    qk_bmm_res = torch.matmul(
        cur_q_bnsd.squeeze(0),
        cur_k_bnsd.squeeze(0).transpose(-1, -2)
    ).unsqueeze(0)
    print_tensor_info("  QK (int32)", qk_bmm_res)
    
    print_tensor_info("\nStep 2: QK * Q_scale * K_scale / 1024", None)
    print("  INT8需要除以1024进行scale调整")
    qk_scaled = (qk_bmm_res.to(torch.float32) * cur_q_scale_bnsd.to(torch.float32) * k_scale_broadcast.to(torch.float32) / 1024.0)
    print_tensor_info("  QK_scaled", qk_scaled)
    
    print_tensor_info("\nStep 3: ReLU", None)
    qk_relu = qk_scaled.clamp_min(0.0).to(torch.float16)
    print_tensor_info("  ReLU(QK_scaled)", qk_relu)
    
    print_tensor_info("\nStep 4: Weight广播", None)
    result_weighted = qk_relu.to(torch.float32) * cur_wt_bnsd.to(torch.float32)
    print_tensor_info("  Weighted", result_weighted)
    
    print_tensor_info("\nStep 5: 组内求和", None)
    result_sum = result_weighted.reshape(1, N2, S1, S2, g).sum(dim=-1)
    print_tensor_info("  最终结果", result_sum)
    
    return result_sum

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("QLI Golden 计算顺序调试脚本")
    print("=" * 70)
    
    print("\n\n" + "=" * 70)
    print("第一部分：FP8版本测试")
    print("=" * 70)
    
    query_fp8, key_fp8, weights_fp8, q_scale_fp8, k_scale_fp8, act_seq_q, act_seq_k, sparse_count, sparse_mode = create_simple_test_data_fp8()
    
    original_result_fp8 = manual_calc_original_fp8(query_fp8, key_fp8, weights_fp8, q_scale_fp8, k_scale_fp8, group_size=4)
    modified_result_fp8 = manual_calc_correct_fp8(query_fp8, key_fp8, weights_fp8, q_scale_fp8, k_scale_fp8, group_size=4)
    compare_results(original_result_fp8, modified_result_fp8)
    
    print("\n\n" + "=" * 70)
    print("第二部分：INT8版本测试")
    print("=" * 70)
    
    query_int8, key_int8, weights_int8, q_scale_int8, k_scale_int8, _, _, _, _ = create_simple_test_data_int8()
    
    original_result_int8 = manual_calc_original_int8(query_int8, key_int8, weights_int8, q_scale_int8, k_scale_int8, group_size=4)
    modified_result_int8 = manual_calc_correct_int8(query_int8, key_int8, weights_int8, q_scale_int8, k_scale_int8, group_size=4)
    compare_results(original_result_int8, modified_result_int8)
    
    print("\n\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
如果结果显示有明显差异，说明修改后的计算顺序与原始不同，
这正是我们期望的结果 - 将qscale和weight的计算位置前移。

下一步：
1. 修改 quant_lightning_indexer_golden.py 中的 cal_atten_per_batch_fp8 和 cal_atten_per_batch_int8
2. 按正确的计算顺序重新实现
3. 运行 pytest 测试验证修改后的代码
""")

if __name__ == "__main__":
    main()