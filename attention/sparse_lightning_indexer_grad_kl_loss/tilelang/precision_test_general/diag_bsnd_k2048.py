#!/usr/bin/env python3
"""诊断脚本 - 打印TileLang kernel中间结果与Python参考对比"""

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

print(f"=== 输入统计 ===")
print(f"q: shape={q.shape}, dtype={q.dtype}, range=[{q.float().min():.4f}, {q.float().max():.4f}]")
print(f"k: shape={k.shape}, dtype={k.dtype}, range=[{k.float().min():.4f}, {k.float().max():.4f}]")
print(f"weights: shape={weights.shape}, dtype={weights.dtype}, range=[{weights.float().min():.6f}, {weights.float().max():.6f}]")
print(f"softmax_max: shape={softmax_max.shape}, dtype={softmax_max.dtype}, range=[{softmax_max.float().min():.4f}, {softmax_max.float().max():.4f}]")
print(f"softmax_sum: shape={softmax_sum.shape}, dtype={softmax_sum.dtype}, range=[{softmax_sum.float().min():.4f}, {softmax_sum.float().max():.4f}]")

# Python参考计算
S2_pad = (S2 + 256 - 1) // 256 * 256
k_p = torch.nn.functional.pad(k, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k
ki_p = torch.nn.functional.pad(k_index, (0,0,0,0,0,S2_pad-S2,0,0)) if S2_pad > S2 else k_index
gk, gki = _preprocess_gather(k_p, ki_p, sparse_indices, B, S1, S2_pad, N2, 1, K, D, DIndex)

bmm1_noscale = q.float() @ gk.float().transpose(-2, -1)
bmm2_ref = torch.relu(q_index.float() @ gki.float().transpose(-2, -1))

print(f"\n=== Python参考 bmm1/bmm2 ===")
print(f"bmm1_noscale: shape={bmm1_noscale.shape}, range=[{bmm1_noscale.min():.4f}, {bmm1_noscale.max():.4f}]")
print(f"bmm1_scaled: range=[{(bmm1_noscale*scale).min():.4f}, {(bmm1_noscale*scale).max():.4f}]")
print(f"bmm2_ref (ReLU): shape={bmm2_ref.shape}, range=[{bmm2_ref.min():.4f}, {bmm2_ref.max():.4f}]")

# Python参考 P_reduced
bmm1 = bmm1_noscale * scale
sm_max_sq = softmax_max.float().squeeze(1)
sm_sum_sq = softmax_sum.float().squeeze(1)
P = torch.exp(bmm1 - sm_max_sq.unsqueeze(-1)) / sm_sum_sq.unsqueeze(-1)
P_reduced_ref = P.mean(dim=2)

print(f"\n=== Python参考 P_reduced ===")
print(f"P: shape={P.shape}, range=[{P.min():.6f}, {P.max():.6f}], sum={P.sum():.4f}")
print(f"P_reduced_ref: shape={P_reduced_ref.shape}, range=[{P_reduced_ref.min():.6f}, {P_reduced_ref.max():.6f}], sum={P_reduced_ref.sum():.4f}")

# Python参考 ReduceSumY
w = weights.float().unsqueeze(-1)
weighted_relu = bmm2_ref * w
reduce_sum_y_ref = weighted_relu.sum(dim=2)

print(f"\n=== Python参考 ReduceSumY ===")
print(f"reduce_sum_y_ref: shape={reduce_sum_y_ref.shape}, range=[{reduce_sum_y_ref.min():.6f}, {reduce_sum_y_ref.max():.6f}]")

# Python参考 SoftmaxI
rsy_max = reduce_sum_y_ref.max(dim=-1, keepdim=True).values
rsy_exp = torch.exp(reduce_sum_y_ref - rsy_max)
rsy_sum = rsy_exp.sum(dim=-1, keepdim=True)
softmax_i_ref = rsy_exp / rsy_sum

print(f"\n=== Python参考 SoftmaxI ===")
print(f"softmax_i_ref: shape={softmax_i_ref.shape}, range=[{softmax_i_ref.min():.6f}, {softmax_i_ref.max():.6f}], sum={softmax_i_ref.sum():.6f}")

# Python参考 dWeights
diff_ref = softmax_i_ref - P_reduced_ref
d_weights_ref = (diff_ref.unsqueeze(2) * bmm2_ref).sum(dim=-1)

print(f"\n=== Python参考 dWeights ===")
print(f"diff (softmaxI - P_reduced): range=[{diff_ref.min():.6f}, {diff_ref.max():.6f}]")
print(f"d_weights_ref: shape={d_weights_ref.shape}, range=[{d_weights_ref.min():.6f}, {d_weights_ref.max():.6f}]")

# 运行TileLang kernel并获取workspace
tiling = check_param_valid(
    q, k, q_index, k_index, weights, sparse_indices,
    softmax_max, softmax_sum, sparse_mode=3, layout_query="BSND")

input_dtype = q.dtype
calc_dtype_str = "float"
input_dtype_str = "float16"

d_rope = q_rope.shape[-1]

gather_key_rope = k_rope[torch.arange(B, device=device).view(B, 1, 1),
                          sparse_indices[:, :, 0, :].clamp(min=0), 0, :].contiguous()
gather_key_index_t = gki.permute(0, 1, 3, 2).contiguous()

func = tilelang_grad_kl_loss_single_kernel(
    B, S1, S2_pad, N1, N2, G, Nidx1, 1, D, DIndex, K,
    d_rope, input_dtype_str, calc_dtype_str, scale,
)

bmm1_res_ws = torch.zeros(B, S1, N1, K, dtype=torch.float32, device=device)
bmm2_res_ws = torch.zeros(B, S1, Nidx1, K, dtype=torch.float32, device=device)
p_reduced_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
reduce_sum_y_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
softmax_i_ws = torch.zeros(B, S1, K, dtype=torch.float32, device=device)
d_query_index_ws = torch.zeros(B, S1, Nidx1, DIndex, dtype=torch.float32, device=device)
d_query_index_out = torch.zeros(B, S1, Nidx1, DIndex, dtype=input_dtype, device=device)
d_weight_out = torch.zeros(B, S1, Nidx1, dtype=input_dtype, device=device)
relu_grad_out = torch.zeros(B, S1, Nidx1, K, dtype=input_dtype, device=device)
loss_out = torch.zeros(B, S1, 16, dtype=torch.float32, device=device)

torch.npu.synchronize()
results = func(
    q, q_rope, gk, gather_key_rope,
    q_index, gki, gather_key_index_t,
    weights, softmax_max, softmax_sum,
    bmm1_res_ws, bmm2_res_ws,
    p_reduced_ws, reduce_sum_y_ws, softmax_i_ws,
    d_query_index_out, d_weight_out, relu_grad_out, loss_out,
    d_query_index_ws,
)
torch.npu.synchronize()

print(f"\n=== TileLang kernel workspace ===")
print(f"bmm1_res_ws: shape={bmm1_res_ws.shape}, range=[{bmm1_res_ws.min():.4f}, {bmm1_res_ws.max():.4f}]")
print(f"bmm2_res_ws: shape={bmm2_res_ws.shape}, range=[{bmm2_res_ws.min():.4f}, {bmm2_res_ws.max():.4f}]")
print(f"p_reduced_ws: shape={p_reduced_ws.shape}, range=[{p_reduced_ws.min():.6f}, {p_reduced_ws.max():.6f}], sum={p_reduced_ws.sum():.6f}")
print(f"reduce_sum_y_ws: shape={reduce_sum_y_ws.shape}, range=[{reduce_sum_y_ws.min():.6f}, {reduce_sum_y_ws.max():.6f}]")
print(f"softmax_i_ws: shape={softmax_i_ws.shape}, range=[{softmax_i_ws.min():.6f}, {softmax_i_ws.max():.6f}], sum={softmax_i_ws.sum():.6f}")
print(f"d_weight_out: shape={d_weight_out.shape}, dtype={d_weight_out.dtype}, range=[{d_weight_out.float().min():.6f}, {d_weight_out.float().max():.6f}]")
print(f"relu_grad_out: shape={relu_grad_out.shape}, range=[{relu_grad_out.float().min():.6f}, {relu_grad_out.float().max():.6f}]")
print(f"loss_out[0,0,0]={loss_out[0,0,0].item():.6f}")

# 对比bmm1
print(f"\n=== bmm1对比 (noscale) ===")
bmm1_diff = (bmm1_res_ws.cpu() - bmm1_noscale.cpu()).abs()
print(f"|bmm1_tl - bmm1_ref|: max={bmm1_diff.max():.6f}, mean={bmm1_diff.mean():.6f}")

# 对比bmm2
print(f"\n=== bmm2对比 (ReLU) ===")
bmm2_diff = (bmm2_res_ws.cpu() - bmm2_ref.cpu()).abs()
print(f"|bmm2_tl - bmm2_ref|: max={bmm2_diff.max():.6f}, mean={bmm2_diff.mean():.6f}")

# 对比P_reduced
print(f"\n=== P_reduced对比 ===")
p_diff = (p_reduced_ws.cpu() - P_reduced_ref.cpu()).abs()
p_off_max = P_reduced_ref.cpu().abs().max()
print(f"|P_tl - P_ref|: max={p_diff.max():.6f}, mean={p_diff.mean():.6f}")
print(f"P_ref max_abs={p_off_max:.6f}, rel_err={p_diff.max()/(p_off_max+1e-8):.6f}")

# 对比ReduceSumY
print(f"\n=== ReduceSumY对比 ===")
rsy_diff = (reduce_sum_y_ws.cpu() - reduce_sum_y_ref.cpu()).abs()
rsy_off_max = reduce_sum_y_ref.cpu().abs().max()
print(f"|rsy_tl - rsy_ref|: max={rsy_diff.max():.6f}, mean={rsy_diff.mean():.6f}")
print(f"rsy_ref max_abs={rsy_off_max:.6f}, rel_err={rsy_diff.max()/(rsy_off_max+1e-8):.6f}")

# 对比SoftmaxI
print(f"\n=== SoftmaxI对比 ===")
si_diff = (softmax_i_ws.cpu() - softmax_i_ref.cpu()).abs()
si_off_max = softmax_i_ref.cpu().abs().max()
print(f"|si_tl - si_ref|: max={si_diff.max():.6f}, mean={si_diff.mean():.6f}")
print(f"si_ref max_abs={si_off_max:.6f}, rel_err={si_diff.max()/(si_off_max+1e-8):.6f}")

# 对比dWeights
print(f"\n=== dWeights对比 ===")
dw_diff = (d_weight_out.float().cpu() - d_weights_ref.cpu()).abs()
dw_off_max = d_weights_ref.cpu().abs().max()
print(f"|dw_tl - dw_ref|: max={dw_diff.max():.6f}, mean={dw_diff.mean():.6f}")
print(f"dw_ref max_abs={dw_off_max:.6f}, rel_err={dw_diff.max()/(dw_off_max+1e-8):.6f}")

# 官方算子结果
print(f"\n=== 官方算子 ===")
d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
    q, k, q_index, k_index, weights, sparse_indices,
    softmax_max, softmax_sum, scale,
    query_rope=q_rope, key_rope=k_rope,
    actual_seq_qlen=[S1], actual_seq_klen=[S2],
    layout="BSND", sparse_mode=3,
    pre_tokens=65536, next_tokens=65536)
torch.npu.synchronize()
print(f"dw_off: range=[{dw_off.float().min():.6f}, {dw_off.float().max():.6f}]")
print(f"loss_off={loss_off.float().item():.6f}")