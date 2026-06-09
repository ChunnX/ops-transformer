#!/usr/bin/env python3
"""诊断 - 对比BSND和TND入口的结果，定位TND精度问题"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss

B, S1, S2 = 1, 1, 2048
N1, N2, Nidx1 = 128, 1, 64
D, DIndex, DR, K = 512, 128, 64, 2048
G = N1 // N2
scale = 1.0 / math.sqrt(D)
device = torch.device('npu')
dtype = torch.float16

torch.manual_seed(42)
q = torch.randn(B, S1, N1, D, dtype=dtype, device=device)
k = torch.randn(B, S2, N2, D, dtype=dtype, device=device)
q_index = torch.randn(B, S1, Nidx1, DIndex, dtype=dtype, device=device)
k_index = torch.randn(B, S2, N2, DIndex, dtype=dtype, device=device)
q_rope = torch.randn(B, S1, N1, DR, dtype=dtype, device=device)
k_rope = torch.randn(B, S2, N2, DR, dtype=dtype, device=device)
weights = torch.randn(B, S1, Nidx1, dtype=dtype, device=device)
a, b, kk = -0.05, 0.05, 3.0
weights = weights * (b - a) / (2 * kk) + (a + b) / 2

sparse_indices = torch.zeros(B, S1, N2, K, dtype=torch.int32).npu()
for s1Idx in range(S1):
    s2RealSize = S2 - S1 + s1Idx + 1
    s2RealLen = min(s2RealSize, K)
    sparse_indices[:, s1Idx, 0, 0:s2RealLen] = torch.randperm(s2RealSize, dtype=torch.int32)[:s2RealLen].npu()
    sparse_indices[:, s1Idx, 0, s2RealLen:K] = -1

with torch.no_grad():
    q_full = torch.cat([q.float(), q_rope.float()], dim=-1)
    k_full = torch.cat([k.float().squeeze(2), k_rope.float().squeeze(2)], dim=-1)
    bmm1_full = torch.matmul(q_full, k_full.transpose(-1, -2).unsqueeze(1))
    bmm1_scaled = bmm1_full * scale
    sm_max_val = bmm1_scaled.max(dim=-1).values
    sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
    softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device)
    softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device)

# BSND call
print("[1] BSND...")
d_qi_bsnd, d_ki_bsnd, dw_bsnd, loss_bsnd = sparse_lightning_indexer_grad_kl_loss(
    q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale,
    query_rope=q_rope, key_rope=k_rope, layout_query="BSND", sparse_mode=3)
torch.npu.synchronize()

# TND call (squeeze B dim)
print("[2] TND...")
q_tnd = q.squeeze(0)
k_tnd = k.squeeze(0)
qi_tnd = q_index.squeeze(0)
ki_tnd = k_index.squeeze(0)
qr_tnd = q_rope.squeeze(0)
kr_tnd = k_rope.squeeze(0)
w_tnd = weights.squeeze(0)
si_tnd = sparse_indices.squeeze(0)
sm_tnd = softmax_max.squeeze(0)
ss_tnd = softmax_sum.squeeze(0)

d_qi_tnd, d_ki_tnd, dw_tnd, loss_tnd = sparse_lightning_indexer_grad_kl_loss(
    q_tnd, k_tnd, qi_tnd, ki_tnd, w_tnd, si_tnd, sm_tnd, ss_tnd, scale,
    query_rope=qr_tnd, key_rope=kr_tnd, layout_query="TND", sparse_mode=3)
torch.npu.synchronize()

# Official TND
print("[3] Official TND...")
d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
    q_tnd, k_tnd, qi_tnd, ki_tnd, w_tnd, si_tnd, sm_tnd, ss_tnd, scale,
    query_rope=qr_tnd, key_rope=kr_tnd, actual_seq_qlen=[S1], actual_seq_klen=[S2],
    layout="TND", sparse_mode=3, pre_tokens=65536, next_tokens=65536)
torch.npu.synchronize()

# Compare BSND vs TND (our impl)
print("\n=== TileLang BSND vs TND ===")
for name, bsnd_v, tnd_v in [("dQueryIndex", d_qi_bsnd, d_qi_tnd), 
                              ("dKeyIndex", d_ki_bsnd, d_ki_tnd),
                              ("dWeights", dw_bsnd, dw_tnd)]:
    diff = (bsnd_v.float() - tnd_v.float()).abs()
    print(f"  {name}: max_diff={diff.max().item():.8f}")

# Compare TND vs Official
print("\n=== TileLang TND vs Official TND ===")
for name, tl_v, off_v in [("dQueryIndex", d_qi_tnd, d_qi_off),
                            ("dKeyIndex", d_ki_tnd, d_ki_off),
                            ("dWeights", dw_tnd, dw_off)]:
    diff = (tl_v.float() - off_v.float()).abs()
    off_max = off_v.float().abs().max().item()
    re = diff.max().item() / (off_max + 1e-8)
    print(f"  {name}: max_diff={diff.max().item():.6f}, rel_err={re:.6f}")

# Compare BSND vs Official (BSND already passed)
print("\n=== TileLang BSND vs Official TND (should match) ===")
for name, bsnd_v, off_v in [("dQueryIndex", d_qi_bsnd, d_qi_off),
                              ("dKeyIndex", d_ki_bsnd, d_ki_off),
                              ("dWeights", dw_bsnd, dw_off)]:
    diff = (bsnd_v.float() - off_v.float()).abs()
    off_max = off_v.float().abs().max().item()
    re = diff.max().item() / (off_max + 1e-8)
    print(f"  {name}: max_diff={diff.max().item():.6f}, rel_err={re:.6f}")