#!/usr/bin/env python3
"""泛化精度测试 - TND K=2048"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Force clear tilelang JIT cache before importing the kernel
import tilelang
tilelang.disable_cache()
import glob as _glob
import shutil as _shutil
for _cache_dir in _glob.glob(os.path.expanduser("~/.cache/tilelang*")) + _glob.glob("/tmp/tilelang_*"):
    _shutil.rmtree(_cache_dir, ignore_errors=True)

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss


def gen_inputs(B, S1, S2, N1, Nidx1, N2, topk, D, DIndex, DR, is_tnd=False, seed=42):
    G = N1 // N2
    output_dtype = torch.float16
    torch.manual_seed(seed)
    q = torch.randn(B, S1, N1, D, dtype=output_dtype, device=torch.device('npu'))
    k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
    q_index = torch.randn(B, S1, Nidx1, DIndex, dtype=output_dtype, device=torch.device('npu'))
    k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=torch.device('npu'))
    if DR != 0:
        q_rope = torch.randn(B, S1, N1, DR, dtype=output_dtype, device=torch.device('npu'))
        k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=torch.device('npu'))
    else:
        q_rope, k_rope = None, None
    weights = torch.randn(B, S1, Nidx1, dtype=output_dtype, device=torch.device('npu'))
    a, b, kk = -0.05, 0.05, 3.0
    scale_w = (b - a) / (2 * kk)
    shift_w = (a + b) / 2
    weights = weights * scale_w + shift_w
    sparse_indices = torch.zeros(B, S1, N2, topk).to(torch.int32).npu()
    for s1Idx in range(S1):
        s2RealSize = (int)(S2 - S1 + s1Idx + 1)
        if s2RealSize <= 0:
            s2RealSize = S2
        if s2RealSize > topk:
            s2RealLen = topk
        else:
            s2RealLen = s2RealSize
        sparse_indices[:, s1Idx, 0, 0:s2RealLen] = torch.randperm(s2RealSize, dtype=torch.int32)[:s2RealLen].npu()
        sparse_indices[:, s1Idx, 0, s2RealLen:topk] = -1
    sc = 1.0 / math.sqrt(D)
    with torch.no_grad():
        q_full = torch.cat([q.float(), q_rope.float()], dim=-1) if q_rope is not None else q.float()
        k_full = torch.cat([k.float().squeeze(2), k_rope.float().squeeze(2)], dim=-1) if k_rope is not None else k.float().squeeze(2)
        bmm1_full = torch.matmul(q_full, k_full.transpose(-1, -2).unsqueeze(1))
        bmm1_scaled = bmm1_full * sc
        sm_max_val = bmm1_scaled.max(dim=-1).values
        sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
        softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device=torch.device('npu'))
        softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().to(device=torch.device('npu'))
    if is_tnd:
        q_tnd = q.squeeze(dim=0)
        k_tnd = k.squeeze(dim=0)
        q_index_tnd = q_index.squeeze(dim=0)
        k_index_tnd = k_index.squeeze(dim=0)
        q_rope_tnd = q_rope.squeeze(dim=0) if q_rope is not None else None
        k_rope_tnd = k_rope.squeeze(dim=0) if k_rope is not None else None
        weights_tnd = weights.squeeze(dim=0)
        sparse_indices_tnd = sparse_indices.squeeze(dim=0)
        softmax_max_tnd = softmax_max.squeeze(dim=0)
        softmax_sum_tnd = softmax_sum.squeeze(dim=0)
        return (q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd,
                weights_tnd, sparse_indices_tnd, softmax_max_tnd, softmax_sum_tnd)
    else:
        return (q, k, q_index, k_index, q_rope, k_rope,
                weights, sparse_indices, softmax_max, softmax_sum)


B, S1, S2 = 1, 1, 2048; N1, N2, Nidx1 = 128, 1, 64; D, DIndex, DR, K = 512, 128, 64, 2048; scale = 1.0 / math.sqrt(D)
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
        diff = (tl_v.float() - off_v.float()).abs(); mx = diff.max().item(); off_max = off_v.float().abs().max().item(); re = mx / (off_max + 1e-8); ok = re < 2e-3
        if not ok: all_pass = False
        print(f"  {name}: max_diff={mx:.6f}, rel_err={re:.6f} [{'PASS' if ok else 'FAIL'}]")
    ltl, loff = loss_tl.float().item(), loss_off.float().item()
    if math.isnan(ltl):
        print(f"  Loss: tl=NaN, off={loff:.6f} [FAIL]"); all_pass = False
    else:
        la = abs(ltl - loff); lr = la / (abs(loff) + 1e-8); ok = lr < 2e-3
        if not ok: all_pass = False
        print(f"  Loss: abs_diff={la:.6f}, rel_err={lr:.6f}, tl={ltl:.6f}, off={loff:.6f} [{'PASS' if ok else 'FAIL'}]")
    print(f"\n结果: {'所有测试通过!' if all_pass else '部分测试失败'}")
else: print("\n结果: 无法对比")
