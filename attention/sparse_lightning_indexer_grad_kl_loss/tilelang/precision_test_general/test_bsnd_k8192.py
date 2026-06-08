#!/usr/bin/env python3
"""泛化精度测试 - BSND K=8192"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss

B, S1, S2 = 1, 1, 8192
N1, N2 = 128, 1
Nidx1, Nidx2 = 64, 1
DQuery, DQueryIndex, d_rope, K = 512, 128, 64, 8192
G = N1 // N2
layout = "BSND"

print(f"泛化精度测试 - {layout} K={K}")
print(f"  B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, Nidx1={Nidx1}, Nidx2={Nidx2}")
print(f"  DQuery={DQuery}, DQueryIndex={DQueryIndex}, d_rope={d_rope}, K={K}, G={G}")

torch.manual_seed(42)

query = torch.randn(B, S1, N1, DQuery, dtype=torch.float16).npu()
key = torch.randn(B, S2, N2, DQuery, dtype=torch.float16).npu()
query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=torch.float16).npu()
key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=torch.float16).npu()
weights = torch.randn(B, S1, Nidx1, dtype=torch.float16).npu()
query_rope = torch.randn(B, S1, N1, d_rope, dtype=torch.float16).npu()
key_rope = torch.randn(B, S2, N2, d_rope, dtype=torch.float16).npu()

sparse_indices = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).npu()
for s1_idx in range(S1):
    s2_real_size = S2 - S1 + s1_idx + 1
    if s2_real_size <= 0:
        s2_real_size = S2
    s2_real_len = min(s2_real_size, K)
    sparse_indices[:, s1_idx, 0, 0:s2_real_len] = (torch.randint(0, s2_real_size, (s2_real_len,)).to(torch.int32)).npu()
    sparse_indices[:, s1_idx, 0, s2_real_len:K] = -1

scale_value = 1.0 / math.sqrt(DQuery)
with torch.no_grad():
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

print("\n[1] TileLang...")
try:
    d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
        query, key, query_index, key_index, weights, sparse_indices,
        softmax_max, softmax_sum, scale_value,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", sparse_mode=3,
    )
    torch.npu.synchronize()
    print("  OK")
    tl_ok = True
except Exception as e:
    print(f"  FAIL: {e}")
    tl_ok = False

print("\n[2] 官方算子...")
try:
    d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        query, key, query_index, key_index, weights, sparse_indices,
        softmax_max, softmax_sum, scale_value,
        query_rope=query_rope, key_rope=key_rope,
        actual_seq_qlen=[S1], actual_seq_klen=[S2],
        layout="BSND", sparse_mode=3,
        pre_tokens=65536, next_tokens=65536,
    )
    torch.npu.synchronize()
    print("  OK")
    off_ok = True
except Exception as e:
    print(f"  FAIL: {e}")
    off_ok = False

if tl_ok and off_ok:
    print("\n[3] 精度对比:")
    all_pass = True
    for name, tl_val, off_val in [("dQueryIndex", d_qi_tl, d_qi_off), ("dKeyIndex", d_ki_tl, d_ki_off), ("dWeights", dw_tl, dw_off)]:
        diff = (tl_val.float() - off_val.float()).abs()
        max_diff = diff.max().item()
        rel_err = (diff / (off_val.float().abs() + 1e-6)).mean().item()
        ok = max_diff < 0.01
        if not ok:
            all_pass = False
        print(f"  {name}: max_diff={max_diff:.6f}, rel_err={rel_err:.6f} [{'PASS' if ok else 'FAIL'}]")
    loss_tl_v = loss_tl.float().item()
    loss_off_v = loss_off.float().item()
    loss_abs = abs(loss_tl_v - loss_off_v)
    loss_rel = loss_abs / (abs(loss_off_v) + 1e-6)
    ok = loss_rel < 0.01
    if not ok:
        all_pass = False
    print(f"  Loss: abs_diff={loss_abs:.6f}, rel_err={loss_rel:.6f}, tl={loss_tl_v:.6f}, off={loss_off_v:.6f} [{'PASS' if ok else 'FAIL'}]")
    print(f"\n结果: {'所有测试通过!' if all_pass else '部分测试失败'}")
else:
    print("\n结果: 无法对比")