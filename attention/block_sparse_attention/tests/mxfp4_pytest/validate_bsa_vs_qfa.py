#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BSA mxfp4 golden 可靠性交叉验证:
  若 Q/K/V 一致 + block_sparse_mask 全 1(dense) + MHA(N1==N2) + S<=4096(qfa kv 分块塌成单 tile)
  + s_dtype 一致, 则 BSA block-sparse 退化为稠密 attention, 应与 qfa flash golden 逐位接近
  (仅剩 fp16 P-量化噪声 + 极小 scale-cast/累加顺序差异).
  本脚本以 qfa 树里【真实】 flash_attention_cpu_golden_varlen 作独立 oracle, 与本仓 BSA golden
  跑同一份 QKV, 比对. 用作 BSA golden 是否忠实于 qfa 逻辑的评判标准.
  纯 CPU, 无需 NPU/torch_npu.
"""
import os
import sys
import math
import tempfile
import io
import contextlib

# 在临时目录工作, 吞掉 qfa golden 的 dump_data/ 与 print
os.chdir(tempfile.mkdtemp())

QFA_DIR = r'D:/work/w001-bsa/code/for-glm/ops-transformer-6060e1b8f886da2ca6bbccbf5836ee8981719418/attention/quant_flash_attn/tests/pytest'
BSA_DIR = r'D:/work/w001-bsa/code/for-glm/ops-transformer_0703/attention/block_sparse_attention/tests'
sys.path.insert(0, QFA_DIR)
sys.path.insert(0, BSA_DIR)

import torch
import numpy as np
from flash_attention_cpu_golden import flash_attention_cpu_golden_varlen
from test_bsa_mxfp4_golden import cpu_block_sparse_attention_bnsd_mxfp4_with_lse


def qfa_flash_bnsd(q, k, v, scale, s_dtype='fp16', quantize_p=True, p_mode='blockwise_snap_local'):
    """qfa 判决基准(flash golden, MHA, dense). 输入 BNSD fp32 -> 输出 BNSD fp32."""
    B, N, S, D = q.shape
    q_p = q.permute(0, 2, 1, 3).flatten(0, 1).to(torch.float32)   # [B*S, N, D]
    k_p = k.permute(0, 2, 1, 3).flatten(0, 1).to(torch.float32)
    v_p = v.permute(0, 2, 1, 3).flatten(0, 1).to(torch.float32)
    cu_q = [i * S for i in range(B + 1)]
    cu_kv = [i * S for i in range(B + 1)]
    with contextlib.redirect_stdout(io.StringIO()):  # 吞掉 qfa 的 dump print
        out = flash_attention_cpu_golden_varlen(
            q_p, k_p, v_p, cu_q, cu_kv, [S] * B, [S] * B,
            softmax_scale=scale, causal=False, block_q=128, block_kv=4096,
            quantize=True, mx_block_size=32, mx_mode='baseline', v_quant_axis='seq_k',
            quantize_p=quantize_p, s_layout='DN', quantize_p_mode=p_mode, s_dtype=s_dtype)
    return out.reshape(B, S, N, D).permute(0, 2, 1, 3).to(torch.float32)  # [B,N,S,D]


def bsa_golden_bnsd(q, k, v, scale, block_shape, p_quant='snap_local', s_dtype='fp16'):
    B, N, S, D = q.shape
    bx, by = block_shape
    cq = (S + bx - 1) // bx
    ckv = (S + by - 1) // by
    mask = torch.ones(B, N, cq, ckv, dtype=torch.int8)   # 全 1 = dense
    out, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
        q, k, v, mask, block_shape, scale, N, p_quant=p_quant, s_dtype=s_dtype)
    return out.to(torch.float32)   # [B,N,S,D]


def compare(a, b, label):
    a = a.flatten().cpu().numpy().astype(np.float32)
    b = b.flatten().cpu().numpy().astype(np.float32)
    max_abs = float(np.max(np.abs(a - b)))
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-10
    max_rel = float(np.max(np.abs(a - b) / denom))
    close = np.isclose(a, b, rtol=1 / 128, atol=1e-4, equal_nan=True)   # qfa check_result 档
    pct = float(close.sum()) / close.size * 100
    tight = np.isclose(a, b, rtol=2e-3, atol=1e-3, equal_nan=True)     # 更严: ~fp16 量级
    pct_tight = float(tight.sum()) / tight.size * 100
    verdict = 'MATCH ' if (pct >= 99.5 and max_rel < 0.1) else 'DIVERGE'
    print(f"[{verdict}] {label}: max_abs={max_abs:.6f} max_rel={max_rel:.6f} | "
          f"pass@1/128={pct:.3f}%  pass@2e-3={pct_tight:.3f}%")


def gen(B, N, S, D=128):
    torch.manual_seed(21)
    np.random.seed(21)
    q = torch.randn(B, N, S, D, dtype=torch.float32)
    k = torch.randn(B, N, S, D, dtype=torch.float32)
    v = torch.randn(B, N, S, D, dtype=torch.float32)
    return q, k, v


print("=== BSA golden vs qfa flash golden (同 QKV, dense 全 1, MHA, s_dtype=fp16) ===")
print("--- [1] snap_local P 量化路径 (主路径, qfa 判决基准) ---")
for (B, N, S, blk) in [(1, 8, 128, [128, 128]),
                       (1, 8, 256, [128, 128]),
                       (2, 8, 128, [128, 128]),
                       (1, 8, 256, [64, 128])]:
    q, k, v = gen(B, N, S)
    scale = 1.0 / math.sqrt(128)
    bsa = bsa_golden_bnsd(q, k, v, scale, blk, p_quant='snap_local', s_dtype='fp16')
    qfa = qfa_flash_bnsd(q, k, v, scale, s_dtype='fp16', quantize_p=True,
                          p_mode='blockwise_snap_local')
    compare(bsa, qfa, f"B={B} N={N} S={S} blk={blk} snap_local")

print("--- [2] 纯 softmax 路径 (BSA p_quant='none' vs qfa quantize_p=False) ---")
for (B, N, S, blk) in [(1, 8, 128, [128, 128]), (1, 8, 256, [128, 128])]:
    q, k, v = gen(B, N, S)
    scale = 1.0 / math.sqrt(128)
    bsa = bsa_golden_bnsd(q, k, v, scale, blk, p_quant='none', s_dtype='fp16')
    qfa = qfa_flash_bnsd(q, k, v, scale, s_dtype='fp16', quantize_p=False, p_mode='global')
    compare(bsa, qfa, f"B={B} N={N} S={S} blk={blk} none-vs-noPquant")

print("--- [3] s_dtype 敏感性 (snap_local, 看 fp16 vs fp32 偏差) ---")
for sd in ['fp16', 'fp32']:  # bf16: qfa 的 S_ij.numpy() dump 不支持, 跳过
    q, k, v = gen(1, 8, 128)
    scale = 1.0 / math.sqrt(128)
    bsa = bsa_golden_bnsd(q, k, v, scale, [128, 128], p_quant='snap_local', s_dtype=sd)
    qfa = qfa_flash_bnsd(q, k, v, scale, s_dtype=sd, quantize_p=True,
                          p_mode='blockwise_snap_local')
    compare(bsa, qfa, f"s_dtype={sd}")
print("=== done ===")
