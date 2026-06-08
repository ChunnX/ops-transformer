#!/usr/bin/env python3
"""泛化精度测试 - TND K=4096"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss

t1 = 1
t2 = 4096
n1 = 128
n2 = 1
n_query_index = 64
d_query = 512
d_query_index = 128
d_rope = 64
topk = 4096
G = n1 // n2

print(f"泛化精度测试 - TND K={topk}")
print(f"  t1={t1}, t2={t2}, n1={n1}, n2={n2}, nQueryIndex={n_query_index}")
print(f"  d_query={d_query}, d_query_index={d_query_index}, d_rope={d_rope}, topk={topk}, G={G}")

torch.manual_seed(42)

query = torch.randn(t1, n1, d_query, dtype=torch.float16).npu()
key = torch.randn(t2, n2, d_query, dtype=torch.float16).npu()
query_index = torch.randn(t1, n_query_index, d_query_index, dtype=torch.float16).npu()
key_index = torch.randn(t2, n2, d_query_index, dtype=torch.float16).npu()
weights = torch.randn(t1, n_query_index, dtype=torch.float16).npu()
query_rope = torch.randn(t1, n1, d_rope, dtype=torch.float16).npu()
key_rope = torch.randn(t2, n2, d_rope, dtype=torch.float16).npu()

sparse_indices = torch.full((t1, n2, topk), -1, dtype=torch.int32).npu()
for i in range(t2):
    sparse_indices[0, 0, i] = i

scale_value = 1.0 / math.sqrt(d_query)

with torch.no_grad():
    k_mqa = key.squeeze(1)
    k_rope_mqa = key_rope.squeeze(1)
    q_full = torch.cat([query, query_rope], dim=-1)
    k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
    bmm1_full = torch.matmul(q_full, k_full.t())
    bmm1_scaled = bmm1_full.float() * scale_value
    sm_max_val = bmm1_scaled.max(dim=-1).values
    sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
    softmax_max = sm_max_val.reshape(n2, t1, G).contiguous().float()
    softmax_sum = sm_sum_val.reshape(n2, t1, G).contiguous().float()

print("\n[1] TileLang...")
try:
    d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
        query, key, query_index, key_index, weights, sparse_indices,
        softmax_max, softmax_sum, scale_value,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="TND", sparse_mode=3,
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
        actual_seq_qlen=[t1], actual_seq_klen=[t2],
        layout="TND", sparse_mode=3,
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
