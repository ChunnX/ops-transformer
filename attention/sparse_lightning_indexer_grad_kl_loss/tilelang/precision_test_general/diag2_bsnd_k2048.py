#!/usr/bin/env python3
"""最小诊断脚本 - 打印TileLang输出与Python参考逐层对比"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sparse_lightning_indexer_grad_kl_loss import (
    sparse_lightning_indexer_grad_kl_loss,
    _preprocess_gather,
    _compute_vector_outputs,
    check_param_valid,
    _compute_d_key_index,
    tilelang_grad_kl_loss_single_kernel,
)
import tilelang

B, S1, S2 = 1, 1, 2048
N1, N2, Nidx1 = 128, 1, 64
D, DIndex, DR, K = 512, 128, 64, 2048
G = N1 // N2
scale = 1.0 / math.sqrt(D)

torch.manual_seed(42)
output_dtype = torch.float16
device = torch.device('npu')

q = torch.randn(B, S1, N1, D, dtype=output_dtype, device=device)
k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=device)
q_index = torch.randn(B, S1, Nidx1, DIndex, dtype=output_dtype, device=device)
k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=device)
q_rope = torch.randn(B, S1, N1, DR, dtype=output_dtype, device=device)
k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=device)
weights = torch.randn(B, S1, Nidx1, dtype=output_dtype, device=device)
a, b, kk = -0.05, 0.05, 3.0
scale_w = (b - a) / (2 * kk)
shift_w = (a + b) / 2
weights = weights * scale_w + shift_w
sparse_indices = torch.zeros(B, S1, N2, K).to(torch.int32).npu()
for s1Idx in range(S1):
    s2RealSize = (int)(S2 - S1 + s1Idx + 1)
    if s2RealSize <= 0:
        s2RealSize = S2
    s2RealLen = min(s2RealSize, K)
    sparse_indices[:, s1Idx, 0, 0:s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
    sparse_indices[:, s1Idx, 0, s2RealLen:K] = -1

sc = 1.0 / math.sqrt(D)
q_f = q.float().reshape(B, S1, N2, G, D)
k_f = k.float()
scores = torch.einsum('bsngd,bkmd->bngsk', q_f, k_f) * sc
sm_max_val = scores.max(dim=-1).values
exp_s = torch.exp(scores - sm_max_val.unsqueeze(-1))
sm_sum_val = exp_s.sum(dim=-1)
softmax_max = sm_max_val.permute(0, 1, 3, 2).to(device=device)
softmax_sum = sm_sum_val.permute(0, 1, 3, 2).to(device=device)

# === Python参考: 用TileLang的输入(gather_key等)重新计算 ===
S2_pad = (S2 + 256 - 1) // 256 * 256
k_p = torch.nn.functional.pad(k, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k
ki_p = torch.nn.functional.pad(k_index, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k_index
gk, gki = _preprocess_gather(k_p, ki_p, sparse_indices, B, S1, S2_pad, N2, 1, K, D, DIndex)

# bmm1 = Q @ GK^T (noscale), bmm2 = ReLU(QI @ GKI^T)
bmm1_noscale = q.float() @ gk.float().transpose(-2, -1)
bmm2_ref = torch.relu(q_index.float() @ gki.float().transpose(-2, -1))

# P_reduced
bmm1_scaled = bmm1_noscale * scale
sm_max_sq = softmax_max.float().squeeze(1)  # (B, S1, G)
sm_sum_sq = softmax_sum.float().squeeze(1)  # (B, S1, G)
P = torch.exp(bmm1_scaled - sm_max_sq.unsqueeze(-1)) / sm_sum_sq.unsqueeze(-1)
P_reduced_ref = P.mean(dim=2)

# ReduceSumY
w = weights.float().unsqueeze(-1)
weighted_relu = bmm2_ref * w
reduce_sum_y_ref = weighted_relu.sum(dim=2)

# SoftmaxI
rsy_max = reduce_sum_y_ref.max(dim=-1, keepdim=True).values
rsy_exp = torch.exp(reduce_sum_y_ref - rsy_max)
rsy_sum = rsy_exp.sum(dim=-1, keepdim=True)
softmax_i_ref = rsy_exp / rsy_sum

# diff and dWeights
diff_ref = softmax_i_ref - P_reduced_ref
d_weights_ref = (diff_ref.unsqueeze(2) * bmm2_ref).sum(dim=-1)

# reluGrad
relu_mask = (bmm2_ref > 0).float()
relu_grad_ref = w * diff_ref.unsqueeze(2) * relu_mask

print("=== Python参考 (用TileLang输入) ===")
print(f"P_reduced: range=[{P_reduced_ref.min():.8f}, {P_reduced_ref.max():.8f}], sum={P_reduced_ref.sum():.8f}")
print(f"ReduceSumY: range=[{reduce_sum_y_ref.min():.8f}, {reduce_sum_y_ref.max():.8f}]")
print(f"SoftmaxI: range=[{softmax_i_ref.min():.8f}, {softmax_i_ref.max():.8f}], sum={softmax_i_ref.sum():.8f}")
print(f"diff: range=[{diff_ref.min():.8f}, {diff_ref.max():.8f}], mean={diff_ref.mean():.8f}")
print(f"dWeights: range=[{d_weights_ref.min():.8f}, {d_weights_ref.max():.8f}]")
print(f"reluGrad: range=[{relu_grad_ref.min():.8f}, {relu_grad_ref.max():.8f}]")

# === TileLang输出 ===
d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
    q, k, q_index, k_index, weights, sparse_indices,
    softmax_max, softmax_sum, scale,
    query_rope=q_rope, key_rope=k_rope,
    layout_query="BSND", sparse_mode=3)
torch.npu.synchronize()

print(f"\n=== TileLang输出 ===")
print(f"dQueryIndex: range=[{d_qi_tl.float().min():.6f}, {d_qi_tl.float().max():.6f}]")
print(f"dKeyIndex: range=[{d_ki_tl.float().min():.6f}, {d_ki_tl.float().max():.6f}]")
print(f"dWeights: range=[{dw_tl.float().min():.6f}, {dw_tl.float().max():.6f}]")
print(f"Loss: {loss_tl.float().item():.6f}")

# === 官方算子输出 ===
d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
    q, k, q_index, k_index, weights, sparse_indices,
    softmax_max, softmax_sum, scale,
    query_rope=q_rope, key_rope=k_rope,
    actual_seq_qlen=[S1], actual_seq_klen=[S2],
    layout="BSND", sparse_mode=3,
    pre_tokens=65536, next_tokens=65536)
torch.npu.synchronize()

print(f"\n=== 官方算子输出 ===")
print(f"dQueryIndex: range=[{d_qi_off.float().min():.6f}, {d_qi_off.float().max():.6f}]")
print(f"dKeyIndex: range=[{d_ki_off.float().min():.6f}, {d_ki_off.float().max():.6f}]")
print(f"dWeights: range=[{dw_off.float().min():.6f}, {dw_off.float().max():.6f}]")
print(f"Loss: {loss_off.float().item():.6f}")

# === 逐层对比 ===
print(f"\n=== 逐层对比 ===")
print(f"|dw_tl - dw_ref|: max={(dw_tl.float().cpu() - d_weights_ref.cpu()).abs().max():.6f}")
print(f"|dw_off - dw_ref|: max={(dw_off.float().cpu() - d_weights_ref.cpu()).abs().max():.6f}")
print(f"|dw_tl - dw_off|: max={(dw_tl.float().cpu() - dw_off.float().cpu()).abs().max():.6f}")

# 检查softmax_max/softmax_sum的dtype和值
print(f"\n=== softmax_max/softmax_sum 检查 ===")
print(f"softmax_max: dtype={softmax_max.dtype}, shape={softmax_max.shape}")
print(f"softmax_sum: dtype={softmax_sum.dtype}, shape={softmax_sum.shape}")
print(f"softmax_max range: [{softmax_max.float().min():.6f}, {softmax_max.float().max():.6f}]")
print(f"softmax_sum range: [{softmax_sum.float().min():.6f}, {softmax_sum.float().max():.6f}]")

# 检查如果softmax_max/sum被截断为FP16会怎样
sm_max_fp16 = softmax_max.to(torch.float16).float()
sm_sum_fp16 = softmax_sum.to(torch.float16).float()
print(f"softmax_max FP16截断误差: max={(softmax_max.float().cpu() - sm_max_fp16.cpu()).abs().max():.8f}")
print(f"softmax_sum FP16截断误差: max={(softmax_sum.float().cpu() - sm_sum_fp16.cpu()).abs().max():.8f}")

# 用FP16截断后的softmax_max/sum重新计算P_reduced
sm_max_sq_fp16 = sm_max_fp16.squeeze(1)
sm_sum_sq_fp16 = sm_sum_fp16.squeeze(1)
P_fp16 = torch.exp(bmm1_scaled - sm_max_sq_fp16.unsqueeze(-1)) / sm_sum_sq_fp16.unsqueeze(-1)
P_reduced_fp16 = P_fp16.mean(dim=2)
diff_fp16 = softmax_i_ref - P_reduced_fp16
d_weights_fp16 = (diff_fp16.unsqueeze(2) * bmm2_ref).sum(dim=-1)
print(f"\n=== FP16截断softmax_max/sum后的影响 ===")
print(f"P_reduced (FP16 sm): range=[{P_reduced_fp16.min():.8f}, {P_reduced_fp16.max():.8f}], sum={P_reduced_fp16.sum():.8f}")
print(f"|P_reduced_fp16 - P_reduced_ref|: max={(P_reduced_fp16.cpu() - P_reduced_ref.cpu()).abs().max():.8f}")
print(f"dWeights (FP16 sm): range=[{d_weights_fp16.min():.8f}, {d_weights_fp16.max():.8f}]")
print(f"|dWeights_fp16 - dWeights_ref|: max={(d_weights_fp16.cpu() - d_weights_ref.cpu()).abs().max():.8f}")