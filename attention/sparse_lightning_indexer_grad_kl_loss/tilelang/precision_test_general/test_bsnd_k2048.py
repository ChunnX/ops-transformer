#!/usr/bin/env python3
"""泛化精度测试 - BSND K=2048"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss


def gen_inputs(B, S1, S2, N1, Nidx1, N2, topk, D, DIndex, DR, w_min=-0.05, w_max=0.05, seed=42):
    G = N1 // N2
    torch.manual_seed(seed)
    q = torch.randn(B, S1, N1, D, dtype=torch.float16, device=torch.device('npu'))
    k = torch.randn(B, S2, N2, D, dtype=torch.float16, device=torch.device('npu'))
    q_index = torch.randn(B, S1, Nidx1, DIndex, dtype=torch.float16, device=torch.device('npu'))
    k_index = torch.randn(B, S2, N2, DIndex, dtype=torch.float16, device=torch.device('npu'))
    q_rope = torch.randn(B, S1, N1, DR, dtype=torch.float16, device=torch.device('npu')) if DR > 0 else None
    k_rope = torch.randn(B, S2, N2, DR, dtype=torch.float16, device=torch.device('npu')) if DR > 0 else None
    weights = torch.randn(B, S1, Nidx1, dtype=torch.float16, device=torch.device('npu'))
    kk = 3.0
    weights = weights * (w_max - w_min) / (2 * kk) + (w_max + w_min) / 2
    sparse_indices = torch.full((B, S1, N2, topk), -1, dtype=torch.int32, device=torch.device('npu'))
    for i in range(S2):
        sparse_indices[0, 0, 0, i] = i
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
    return q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum


def run_test(B, S1, S2, N1, Nidx1, N2, K, D, DIndex, DR, w_min, w_max, layout, label):
    scale = 1.0 / math.sqrt(D)
    q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum = gen_inputs(
        B, S1, S2, N1, Nidx1, N2, K, D, DIndex, DR, w_min, w_max)
    print(f"\n{label}")
    print(f"  B={B}, S1={S1}, S2={S2}, N1={N1}, Nidx1={Nidx1}, K={K}, weights=[{w_min},{w_max}]")
    tl_ok = off_ok = False
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
            q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale,
            query_rope=q_rope, key_rope=k_rope, layout_query=layout, sparse_mode=3)
        torch.npu.synchronize(); tl_ok = True
    except Exception as e: print(f"  TileLang FAIL: {e}")
    try:
        d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale,
            query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=[S1], actual_seq_klen=[S2],
            layout=layout, sparse_mode=3, pre_tokens=65536, next_tokens=65536)
        torch.npu.synchronize(); off_ok = True
    except Exception as e: print(f"  Official FAIL: {e}")
    if tl_ok and off_ok:
        all_pass = True
        for name, tl_v, off_v in [("dQueryIndex", d_qi_tl, d_qi_off), ("dKeyIndex", d_ki_tl, d_ki_off), ("dWeights", dw_tl, dw_off)]:
            if torch.isnan(tl_v).any().item():
                print(f"  {name}: NaN! [FAIL]"); all_pass = False; continue
            diff = (tl_v.float() - off_v.float()).abs(); mx = diff.max().item()
            off_max = off_v.float().abs().max().item(); re = mx / (off_max + 1e-8)
            ok = re < 2e-3
            if not ok: all_pass = False
            print(f"  {name}: max_diff={mx:.6f}, rel_err={re:.6f} [{'PASS' if ok else 'FAIL'}]")
        ltl, loff = loss_tl.float().item(), loss_off.float().item()
        if math.isnan(ltl): print(f"  Loss: NaN! [FAIL]"); all_pass = False
        else:
            la = abs(ltl - loff); lr = la / (abs(loff) + 1e-8); ok = lr < 2e-3
            if not ok: all_pass = False
            print(f"  Loss: abs_diff={la:.6f}, rel_err={lr:.6f} [{'PASS' if ok else 'FAIL'}]")
        return all_pass
    return False


D, DIndex, DR = 512, 128, 64
print(f"泛化精度测试 - BSND K=2048")
results = []
results.append(run_test(1, 1, 2048, 128, 64, 1, 2048, D, DIndex, DR, -0.05, 0.05, "BSND", "[1] N1=128, w=[-0.05,0.05]"))
results.append(run_test(1, 1, 2048, 64, 32, 1, 2048, D, DIndex, DR, -0.05, 0.05, "BSND", "[2] N1=64, w=[-0.05,0.05]"))
results.append(run_test(1, 1, 2048, 128, 64, 1, 2048, D, DIndex, DR, -0.01, 0.01, "BSND", "[3] N1=128, w=[-0.01,0.01]"))
results.append(run_test(1, 1, 2048, 128, 64, 1, 2048, D, DIndex, DR, -0.1, 0.1, "BSND", "[4] N1=128, w=[-0.1,0.1]"))
print(f"\n结果: {'所有测试通过!' if all(results) else '部分测试失败'}")
