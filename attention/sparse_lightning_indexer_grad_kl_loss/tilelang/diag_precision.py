#!/usr/bin/env python3
"""诊断精度问题：逐步对比 TileLang kernel 与 Python 参考实现"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_lightning_indexer_grad_kl_loss import (
    sparse_lightning_indexer_grad_kl_loss,
    _preprocess_gather,
    _compute_d_key_index,
    _compute_vector_outputs,
    check_param_valid,
    tilelang_grad_kl_loss_single_kernel,
    BLOCK_M, BLOCK_N, BLOCK_K, VECTOR_BASEG, VECTOR_BASEN,
)

torch.manual_seed(42)

B, S1, S2 = 1, 1, 2048
N1, N2 = 128, 1
Nidx1, Nidx2 = 64, 1
DQuery, DQueryIndex, d_rope = 512, 128, 64
K = 2048
G = N1 // N2
scale_value = 1.0 / math.sqrt(DQuery)

print(f"配置: B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, G={G}, Nidx1={Nidx1}, K={K}")
print(f"  DQuery={DQuery}, DQueryIndex={DQueryIndex}, d_rope={d_rope}")

# 生成输入
q = torch.randn(B, S1, N1, DQuery, dtype=torch.float16, device='npu')
k = torch.randn(B, S2, N2, DQuery, dtype=torch.float16, device='npu')
q_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=torch.float16, device='npu')
k_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=torch.float16, device='npu')
q_rope = torch.randn(B, S1, N1, d_rope, dtype=torch.float16, device='npu')
k_rope = torch.randn(B, S2, N2, d_rope, dtype=torch.float16, device='npu')

weights = torch.randn(B, S1, Nidx1, dtype=torch.float16, device='npu')
kk = 3.0
weights = weights * (0.05 - (-0.05)) / (2 * kk) + (0.05 + (-0.05)) / 2

sparse_indices = torch.full((B, S1, N2, K), -1, dtype=torch.int32, device='npu')
for i in range(S2):
    sparse_indices[0, 0, 0, i] = i

# 计算 softmax_max/softmax_sum
with torch.no_grad():
    k_mqa = k.squeeze(2)
    k_rope_mqa = k_rope.squeeze(2)
    q_full = torch.cat([q.float(), q_rope.float()], dim=-1)
    k_full = torch.cat([k_mqa.float(), k_rope_mqa.float()], dim=-1)
    bmm1_full = torch.matmul(q_full, k_full.transpose(-1, -2).unsqueeze(1))
    bmm1_scaled = bmm1_full * scale_value
    sm_max_val = bmm1_scaled.max(dim=-1).values
    sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
    softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device='npu')
    softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device='npu')

print(f"\nsoftmax_max shape={softmax_max.shape}, dtype={softmax_max.dtype}")
print(f"softmax_sum shape={softmax_sum.shape}, dtype={softmax_sum.dtype}")

# ============================================================
# Python 参考实现（逐步计算）
# ============================================================
print("\n" + "="*60)
print("Python 参考实现")
print("="*60)

with torch.no_grad():
    # Gather key/key_index
    gather_key, gather_key_index = _preprocess_gather(
        k, k_index, sparse_indices, B, S1, S2, N2, Nidx2, K, DQuery, DQueryIndex)
    gather_key_rope = k_rope[torch.arange(B, device='npu').view(B,1,1),
                              sparse_indices[:,:,0,:].clamp(min=0), 0, :].contiguous()

    # Matmul1: [Query||QueryRope] @ [GatherKey||GatherKeyRope]^T
    q_full_gather = torch.cat([q.float(), q_rope.float()], dim=-1)
    gk_full = torch.cat([gather_key.float(), gather_key_rope.float()], dim=-1)
    bmm1_ref = torch.matmul(q_full_gather, gk_full.transpose(-1, -2))  # (B,S1,N1,K)
    print(f"bmm1_ref: shape={bmm1_ref.shape}, range=[{bmm1_ref.min().item():.4f}, {bmm1_ref.max().item():.4f}]")

    # Matmul2: QueryIndex @ GatherKeyIndex^T, then ReLU
    bmm2_raw_ref = torch.matmul(q_index.float(), gather_key_index.float().transpose(-1, -2))
    bmm2_ref = torch.relu(bmm2_raw_ref)  # (B,S1,Nidx1,K)
    print(f"bmm2_ref (after ReLU): shape={bmm2_ref.shape}, range=[{bmm2_ref.min().item():.4f}, {bmm2_ref.max().item():.4f}]")
    print(f"  bmm2_raw (before ReLU): range=[{bmm2_raw_ref.min().item():.4f}, {bmm2_raw_ref.max().item():.4f}]")
    print(f"  ReLU zero fraction: {(bmm2_ref == 0).float().mean().item():.4f}")

    # P = softmax(bmm1 * scale)
    bmm1_scaled_ref = bmm1_ref * scale_value
    sm_max_ref = softmax_max.float().squeeze(1)  # (B,S1,G)
    sm_sum_ref = softmax_sum.float().squeeze(1)  # (B,S1,G)
    P_ref = torch.exp(bmm1_scaled_ref - sm_max_ref.unsqueeze(-1)) / sm_sum_ref.unsqueeze(-1)  # (B,S1,N1,K)
    P_reduced_ref = P_ref.mean(dim=2)  # (B,S1,K) - average over heads
    print(f"P_reduced_ref: range=[{P_reduced_ref.min().item():.6f}, {P_reduced_ref.max().item():.6f}]")
    print(f"  P_ref: range=[{P_ref.min().item():.6f}, {P_ref.max().item():.6f}]")

    # ReduceSumY = sum(weights * bmm2) over Nidx1
    w_ref = weights.float().unsqueeze(-1)  # (B,S1,Nidx1,1)
    reduce_sum_y_ref = (bmm2_ref * w_ref).sum(dim=2)  # (B,S1,K)
    print(f"reduce_sum_y_ref: range=[{reduce_sum_y_ref.min().item():.6f}, {reduce_sum_y_ref.max().item():.6f}]")

    # SoftmaxI = softmax(reduce_sum_y)
    rsy_max_ref = reduce_sum_y_ref.max(dim=-1, keepdim=True).values
    rsy_exp_ref = torch.exp(reduce_sum_y_ref - rsy_max_ref)
    rsy_sum_ref = rsy_exp_ref.sum(dim=-1, keepdim=True)
    softmax_i_ref = rsy_exp_ref / rsy_sum_ref  # (B,S1,K)
    print(f"softmax_i_ref: range=[{softmax_i_ref.min().item():.6f}, {softmax_i_ref.max().item():.6f}]")

    # diff = softmaxI - P_reduced
    diff_ref = softmax_i_ref - P_reduced_ref  # (B,S1,K)
    print(f"diff_ref: range=[{diff_ref.min().item():.6f}, {diff_ref.max().item():.6f}]")

    # dWeight = sum(diff * bmm2) over K
    d_weight_ref = (diff_ref.unsqueeze(2) * bmm2_ref).sum(dim=-1)  # (B,S1,Nidx1)
    print(f"d_weight_ref: range=[{d_weight_ref.min().item():.6f}, {d_weight_ref.max().item():.6f}]")

    # reluGrad = weight * diff * I(bmm2 > 0)
    relu_mask_ref = (bmm2_ref > 0).float()
    relu_grad_ref = w_ref * diff_ref.unsqueeze(2) * relu_mask_ref  # (B,S1,Nidx1,K)
    print(f"relu_grad_ref: range=[{relu_grad_ref.min().item():.6f}, {relu_grad_ref.max().item():.6f}]")

    # dQueryIndex = reluGrad @ GatherKeyIndex
    d_qi_ref = torch.matmul(relu_grad_ref, gather_key_index.float())  # (B,S1,Nidx1,DQueryIndex)
    print(f"d_qi_ref: range=[{d_qi_ref.min().item():.6f}, {d_qi_ref.max().item():.6f}]")

    # KL Loss
    eps = 1e-8
    kl_loss_ref = (P_reduced_ref * (torch.log(P_reduced_ref + eps) - torch.log(softmax_i_ref + eps))).sum()
    print(f"kl_loss_ref: {kl_loss_ref.item():.6f}")

# ============================================================
# TileLang kernel 计算
# ============================================================
print("\n" + "="*60)
print("TileLang kernel 计算")
print("="*60)

d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
    q, k, q_index, k_index, weights, sparse_indices,
    softmax_max, softmax_sum, scale_value,
    query_rope=q_rope, key_rope=k_rope,
    layout_query="BSND", sparse_mode=3)
torch.npu.synchronize()

print(f"d_qi_tl: range=[{d_qi_tl.float().min().item():.6f}, {d_qi_tl.float().max().item():.6f}]")
print(f"dw_tl: range=[{dw_tl.float().min().item():.6f}, {dw_tl.float().max().item():.6f}]")
print(f"loss_tl: {loss_tl.float().item():.6f}")

# ============================================================
# 官方算子计算
# ============================================================
print("\n" + "="*60)
print("官方算子计算")
print("="*60)

d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
    q, k, q_index, k_index, weights, sparse_indices,
    softmax_max, softmax_sum, scale_value,
    query_rope=q_rope, key_rope=k_rope,
    actual_seq_qlen=[S1], actual_seq_klen=[S2],
    layout="BSND", sparse_mode=3, pre_tokens=65536, next_tokens=65536)
torch.npu.synchronize()

print(f"d_qi_off: range=[{d_qi_off.float().min().item():.6f}, {d_qi_off.float().max().item():.6f}]")
print(f"dw_off: range=[{dw_off.float().min().item():.6f}, {dw_off.float().max().item():.6f}]")
print(f"loss_off: {loss_off.float().item():.6f}")

# ============================================================
# 三方对比
# ============================================================
print("\n" + "="*60)
print("三方对比: Python_ref vs TileLang vs 官方")
print("="*60)

for name, py_v, tl_v, off_v in [
    ("dQueryIndex", d_qi_ref, d_qi_tl.float(), d_qi_off.float()),
    ("dWeights", d_weight_ref, dw_tl.float(), dw_off.float()),
]:
    tl_vs_py = (tl_v - py_v).abs().max().item()
    tl_vs_off = (tl_v - off_v).abs().max().item()
    py_vs_off = (py_v - off_v).abs().max().item()
    off_max = off_v.abs().max().item() + 1e-8
    py_max = py_v.abs().max().item() + 1e-8
    print(f"\n{name}:")
    print(f"  TL vs Python_ref: max_diff={tl_vs_py:.6f}, rel_err={tl_vs_py/py_max:.6f}")
    print(f"  TL vs Official:   max_diff={tl_vs_off:.6f}, rel_err={tl_vs_off/off_max:.6f}")
    print(f"  Python vs Official: max_diff={py_vs_off:.6f}, rel_err={py_vs_off/off_max:.6f}")

loss_tl_v = loss_tl.float().item()
loss_off_v = loss_off.float().item()
print(f"\nLoss:")
print(f"  TL vs Python_ref: abs_diff={abs(loss_tl_v - kl_loss_ref.item()):.6f}")
print(f"  TL vs Official:   abs_diff={abs(loss_tl_v - loss_off_v):.6f}")
print(f"  Python vs Official: abs_diff={abs(kl_loss_ref.item() - loss_off_v):.6f}")

# ============================================================
# 关键诊断：检查 Python_ref 中 ReLU mask 的两种实现差异
# ============================================================
print("\n" + "="*60)
print("ReLU mask 诊断: I(x>0) vs 1-exp(-x*10000)")
print("="*60)

with torch.no_grad():
    mask_step = (bmm2_ref > 0).float()
    mask_exp = 1.0 - torch.exp(-bmm2_ref * 10000.0)
    mask_diff = (mask_step - mask_exp).abs()
    print(f"mask_step range: [{mask_step.min().item():.6f}, {mask_step.max().item():.6f}]")
    print(f"mask_exp  range: [{mask_exp.min().item():.6f}, {mask_exp.max().item():.6f}]")
    print(f"max |mask_step - mask_exp|: {mask_diff.max().item():.6f}")
    print(f"mean |mask_step - mask_exp|: {mask_diff.mean().item():.6f}")

    # 在 bmm2 接近 0 的区域，两种 mask 差异最大
    near_zero = (bmm2_ref > 0) & (bmm2_ref < 0.01)
    if near_zero.any():
        print(f"Near-zero bmm2 count: {near_zero.sum().item()}")
        print(f"  mask_step near zero: {mask_step[near_zero].unique().tolist()}")
        print(f"  mask_exp near zero range: [{mask_exp[near_zero].min().item():.6f}, {mask_exp[near_zero].max().item():.6f}]")

    # 用 mask_exp 重新计算 reluGrad 和 dWeight
    relu_grad_exp = w_ref * diff_ref.unsqueeze(2) * mask_exp
    d_weight_exp = (diff_ref.unsqueeze(2) * bmm2_ref).sum(dim=-1)  # dWeight 不依赖 mask
    d_qi_exp = torch.matmul(relu_grad_exp, gather_key_index.float())

    # 对比用 exp mask 的结果 vs 官方
    dqi_exp_vs_off = (d_qi_exp - d_qi_off.float()).abs().max().item()
    dqi_step_vs_off = (d_qi_ref - d_qi_off.float()).abs().max().item()
    off_max = d_qi_off.float().abs().max().item() + 1e-8
    print(f"\ndQueryIndex (exp mask) vs Official: max_diff={dqi_exp_vs_off:.6f}, rel_err={dqi_exp_vs_off/off_max:.6f}")
    print(f"dQueryIndex (step mask) vs Official: max_diff={dqi_step_vs_off:.6f}, rel_err={dqi_step_vs_off/off_max:.6f}")

print("\n完成")