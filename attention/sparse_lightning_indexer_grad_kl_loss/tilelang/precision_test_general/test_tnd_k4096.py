#!/usr/bin/env python3
"""泛化精度测试 - TND K=4096"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss


def gen_inputs(B, S1, S2, N1, Nidx1, N2, topk, D, DIndex, DR, is_tnd=False, seed=42):
    G = N1 // N2; output_dtype = torch.float16; torch.manual_seed(seed)
    q = torch.randn(B, S1, N1, D, dtype=output_dtype, device=torch.device('npu'))
    k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
    q_index = torch.randn(B, S1, Nidx1, DIndex, dtype=output_dtype, device=torch.device('npu'))
    k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=torch.device('npu'))
    if DR != 0:
        q_rope = torch.randn(B, S1, N1, DR, dtype=output_dtype, device=torch.device('npu'))
        k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=torch.device('npu'))
    else: q_rope, k_rope = None, None
    weights = torch.randn(B, S1, Nidx1, dtype=output_dtype, device=torch.device('npu'))
    a, b, kk = -0.05, 0.05, 3.0; scale_w = (b - a) / (2 * kk); shift_w = (a + b) / 2
    weights = weights * scale_w + shift_w
    sparse_indices = torch.zeros(B, S1, N2, topk).to(torch.int32).npu()
    for s1Idx in range(S1):
        s2RealSize = (int)(S2 - S1 + s1Idx + 1)
        if s2RealSize <= 0: s2RealSize = S2
        s2RealLen = min(s2RealSize, topk)
        sparse_indices[:, s1Idx, 0, 0:s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
        sparse_indices[:, s1Idx, 0, s2RealLen:topk] = -1
    softmax_max = torch.randn(B, N2, S1, G, dtype=torch.float, device=torch.device('npu'))
    softmax_sum = torch.randn(B, N2, S1, G, dtype=torch.float, device=torch.device('npu'))
    if is_tnd:
        return (q.squeeze(0), k.squeeze(0), q_index.squeeze(0), k_index.squeeze(0),
                q_rope.squeeze(0) if q_rope is not None else None, k_rope.squeeze(0) if k_rope is not None else None,
                weights.squeeze(0), sparse_indices.squeeze(0), softmax_max.squeeze(0), softmax_sum.squeeze(0))
    return (q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum)


B, S1, S2 = 1, 1, 4096; N1, N2, Nidx1 = 128, 1, 64; D, DIndex, DR, K = 512, 128, 64, 4096; scale = 1.0 / math.sqrt(D)
print(f"泛化精度测试 - TND K={K}\n  B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, Nidx1={Nidx1}, K={K}")
q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum = gen_inputs(B, S1, S2, N1, Nidx1, N2, K, D, DIndex, DR, is_tnd=True)
print("\n[1] TileLang...")
try:
    d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale, query_rope=q_rope, key_rope=k_rope, layout_query="TND", sparse_mode=3)
    torch.npu.synchronize(); print("  OK"); tl_ok = True
except Exception as e: print(f"  FAIL: {e}"); tl_ok = False
print("\n[2] 官方算子...")
try:
    d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale, query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=[S1], actual_seq_klen=[S2], layout="TND", sparse_mode=3, pre_tokens=65536, next_tokens=65536)
    torch.npu.synchronize(); print("  OK"); off_ok = True
except Exception as e: print(f"  FAIL: {e}"); off_ok = False
if tl_ok and off_ok:
    print("\n[3] 精度对比:"); all_pass = True
    for name, tl_v, off_v in [("dQueryIndex", d_qi_tl, d_qi_off), ("dKeyIndex", d_ki_tl, d_ki_off), ("dWeights", dw_tl, dw_off)]:
        has_nan = torch.isnan(tl_v).any().item()
        if has_nan:
            print(f"  {name}: contains NaN! [FAIL]"); all_pass = False; continue
        diff = (tl_v.float() - off_v.float()).abs(); mx = diff.max().item(); re = (diff / (off_v.float().abs() + 1e-6)).mean().item(); ok = mx < 0.01
        if not ok: all_pass = False
        print(f"  {name}: max_diff={mx:.6f}, rel_err={re:.6f} [{'PASS' if ok else 'FAIL'}]")
    ltl, loff = loss_tl.float().item(), loss_off.float().item()
    if math.isnan(ltl):
        print(f"  Loss: tl=NaN, off={loff:.6f} [FAIL]"); all_pass = False
    else:
        la = abs(ltl - loff); lr = la / (abs(loff) + 1e-6); ok = lr < 0.01
        if not ok: all_pass = False
        print(f"  Loss: abs_diff={la:.6f}, rel_err={lr:.6f}, tl={ltl:.6f}, off={loff:.6f} [{'PASS' if ok else 'FAIL'}]")
    print(f"\n结果: {'所有测试通过!' if all_pass else '部分测试失败'}")
else: print("\n结果: 无法对比")
