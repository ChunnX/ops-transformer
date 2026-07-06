#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import math
import torch_npu
from torchair.configs.compiler_config import CompilerConfig
import torchair as tng
import result_compare_method
from load_external_data import load_data_from_dir
# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 5=动态图, 7=aclgraph
GRAPH_PATH = 0
DEVICE_ID = 0

B = 1
N_q = 1
N_kv = 1
D = 128

ACTUAL_SEQ_Q = [128]
ACTUAL_SEQ_KV = [256]

# Layout 选择
INPUT_LAYOUT = "NTD_TND"
OUTPUT_LAYOUT = "TND"
Q_SCALE_LAYOUT = "NT"

# PA KV Cache Layout
KV_CACHE_LAYOUT = "BnNBsD"

# Data Range
Q_DATA_RANGE = (-400, 400)
K_DATA_RANGE = (-400, 400)
V_DATA_RANGE = (-400, 400)

ENABLE_PA = True
ENABLE_LSE = True
GOLDEN_MODE = True
BLOCK_SIZE = 128
SPARSE_MODE = 3
SCALE_VALUE = None
IS_CONTIGUOUS = True

# Seed
SEED_Q = 54
SEED_K = 3
SEED_V = 20
SEED_BLOCK_TABLE = 1234
FP8_DTYPE = torch.float8_e4m3fn
OUTPUT_DETYPE = torch.bfloat16
P_SCALE = 1.0
EPSILON = 1e-20

Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 256

SAVE_PT = False
SAVE_PT_DIR = ''

# 是否使用外部 pt 文件作为输入 (True=加载外部文件, False=随机生成数据)
USE_EXTERNAL_INPUT = False
# 外部 pt 文件所在目录，USE_EXTERNAL_INPUT=True 时生效
LOAD_PT_DIR = ''


# ==============================================================================
# 数据生成函数
# ==============================================================================
def get_fp8_per_token_head_quant_scale(tensor):
    """
    用于生成 query/key quant scale
    per-token-head quant scale: shape (B, N, S, 1)
    """
    tensor = tensor.contiguous()
    B, N, S, _ = tensor.shape
    fp8_e4m3_max = 448.0
    row_max = torch.abs(tensor).max(dim=3, keepdim=True).values
    row_max = torch.max(row_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / row_max
    return scale.view(B, N, S, 1).float().contiguous()

def get_fp8_per_head_quant_scale(tensor):
    """
    用于生成 value quant scale
    per-head quant scale: shape (1, N, 1, 1)
    """
    tensor = tensor.contiguous()
    fp8_e4m3_max = 448.0
    head_max = torch.abs(tensor).amax(dim=(0, 2, 3), keepdim=True)
    head_max = torch.max(head_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / head_max
    return scale.float().contiguous()

def quant_fp16_to_fp8(tensor, scale):
    """将 fp16 数据量化为 fp8_e4m3"""
    tensor = tensor.contiguous()
    scale = scale.contiguous()
    result = tensor.float() * scale
    result = torch.clamp(result, -448.0, 448.0)
    return result.to(FP8_DTYPE).contiguous()

def create_block_table(actual_seq_kv, block_size, seed=SEED_BLOCK_TABLE):
    """创建 block table"""
    block_num_per_batch = [math.ceil(int(seq_len) / block_size) for seq_len in actual_seq_kv]
    total_blocks = sum(block_num_per_batch)
    max_blocks = max(block_num_per_batch)
    block_idx_list = np.random.default_rng(seed).permutation(np.arange(total_blocks, dtype=np.int32))
    block_table = np.full((len(actual_seq_kv), max_blocks), -1, dtype=np.int32)
    idx = 0

    for b_index, block_num in enumerate(block_num_per_batch):
        block_table[b_index, :block_num] = block_idx_list[idx:idx + block_num]
        idx += block_num
    return block_table

def bnsd_to_k_cache(k_fp8_bnsd, k_scale_fp32_bnsd, seq_lens, block_size, block_table):
    """BNSD to PA K cache, with k scale (fp32) stored in the 4 extra rows"""
    k_fp8_bnsd = k_fp8_bnsd.contiguous()
    k_scale_fp32_bnsd = k_scale_fp32_bnsd.contiguous()
    B_dim, N_dim, S_dim, D_dim = k_fp8_bnsd.shape
    scale_rows = 4
    block_num_per_seq = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_seq)

    cache = torch.zeros((total_blocks, N_dim, block_size + scale_rows, D_dim),
                        dtype=torch.uint8, device=k_fp8_bnsd.device).contiguous()

    for b in range(B_dim):
        bid_table = block_table[b]
        for blk_idx in range(block_num_per_seq[b]):
            blockid = int(bid_table[blk_idx])
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_lens[b])
            valid = end_s - start_s
            if valid <= 0:
                continue
            k_data = k_fp8_bnsd[b, :, start_s:end_s, :].contiguous()
            cache[blockid, :, :valid, :] = k_data.view(torch.uint8)
            scales_all = k_scale_fp32_bnsd[b, :, start_s:end_s, 0].contiguous()  # (N, valid)
            scale_buf = torch.zeros(N_dim, scale_rows, D_dim // 4, dtype=torch.float32, device=cache.device)
            flat_scale = scale_buf.reshape(N_dim, -1)  # (N, 128)
            if valid <= flat_scale.shape[1]:
                flat_scale[:, :valid] = scales_all
            cache[blockid, :, block_size:block_size + scale_rows, :] = scale_buf.view(torch.uint8).reshape(N_dim, scale_rows, D_dim)

    return cache.view(FP8_DTYPE).reshape(total_blocks, N_dim, block_size + scale_rows, D_dim).contiguous()

def bnsd_to_v_cache(tensor_bnsd, seq_lens, block_size, block_table):
    """BNSD to V cache - V cache 使用 FP8 类型"""
    tensor_bnsd = tensor_bnsd.contiguous()
    device = tensor_bnsd.device
    batch, heads, S, dim = tensor_bnsd.shape
    block_num_per_batch = [math.ceil(int(s) / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_batch)

    # V cache 使用 FP8 类型
    out_cache = torch.zeros((total_blocks, heads, block_size + 4, dim),
                            dtype=FP8_DTYPE, device=device).contiguous()

    for b in range(batch):
        for blk_idx in range(block_num_per_batch[b]):
            block_id = int(block_table[b, blk_idx].item())
            block_offset = blk_idx * block_size
            valid_len = min(block_size, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[block_id, :, :valid_len, :] = tensor_bnsd[b, :, block_offset:block_offset + valid_len, :].contiguous()
    
    return out_cache.contiguous()

def generate_data():
    """生成 BNSD FP16 Q/K/V 并做 FP8 量化"""
    max_sq = max(ACTUAL_SEQ_Q)
    max_skv = max(ACTUAL_SEQ_KV) if max(ACTUAL_SEQ_KV) > 0 else 1
    print(f"[INFO] max_sq={max_sq}, max_skv={max_skv}")

    # 使用随机数据
    np.random.seed(SEED_Q)
    q_amp_hi = max(abs(Q_DATA_RANGE[0]), abs(Q_DATA_RANGE[1]))
    q_amp_lo = q_amp_hi * 0.01
    q_token_amps = np.power(
        10.0,
        np.random.uniform(np.log10(q_amp_lo), np.log10(q_amp_hi), size=(B, N_q, max_sq, 1))
    ).astype(np.float32)  # (B, N_q, max_sq, 1)
    q_base = np.random.uniform(low=-1.0, high=1.0, size=(B, N_q, max_sq, D)).astype(np.float32)
    q_data = (q_base * q_token_amps).astype(np.float16)
    q_fp16 = torch.from_numpy(q_data)

    np.random.seed(SEED_K)
    k_amp_hi = max(abs(K_DATA_RANGE[0]), abs(K_DATA_RANGE[1]))
    k_amp_lo = 1.0
    k_token_amps = np.power(
        10.0,
        np.random.uniform(np.log10(k_amp_lo), np.log10(k_amp_hi), size=(B, N_kv, max_skv, 1))
    ).astype(np.float32)  # (B, N_kv, max_skv, 1)
    k_base = np.random.uniform(low=-1.0, high=1.0, size=(B, N_kv, max_skv, D)).astype(np.float32)
    k_data = (k_base * k_token_amps).astype(np.float16)
    k_fp16 = torch.from_numpy(k_data)

    np.random.seed(SEED_V)
    v_head_amps = np.power(
        10.0,
        np.random.uniform(0.0, np.log10(V_DATA_RANGE[1]), size=(B, N_kv, 1, 1))
    ).astype(np.float32)  # (B, N_kv, 1, 1) —— 幅度范围 [1, V_DATA_RANGE[1]]
    v_data_base = np.random.uniform(
        low=-1.0, high=1.0, size=(B, N_kv, max_skv, D),
    ).astype(np.float32)
    v_data = (v_data_base * v_head_amps).astype(np.float16)
    v_fp16 = torch.from_numpy(v_data)

    q_fp16 = q_fp16.cpu().contiguous()
    k_fp16 = k_fp16.cpu().contiguous()
    v_fp16 = v_fp16.cpu().contiguous()

    # 计算量化scale
    quant_scale_q = get_fp8_per_token_head_quant_scale(q_fp16)
    quant_scale_k = get_fp8_per_token_head_quant_scale(k_fp16)
    quant_scale_v = get_fp8_per_head_quant_scale(v_fp16)

    # 反量化scale
    dequant_scale_q = (1.0 / quant_scale_q).contiguous()
    dequant_scale_k = (1.0 / quant_scale_k).contiguous()
    dequant_scale_v = (1.0 / quant_scale_v).contiguous()

    # 量化到fp8
    q_fp8 = quant_fp16_to_fp8(q_fp16, quant_scale_q)
    k_fp8 = quant_fp16_to_fp8(k_fp16, quant_scale_k)
    v_fp8 = quant_fp16_to_fp8(v_fp16, quant_scale_v)

    if max(ACTUAL_SEQ_KV) == 0:
        real_skv = max(ACTUAL_SEQ_KV)
        k_fp8 = k_fp8[:, :, :real_skv, :].contiguous()
        v_fp8 = v_fp8[:, :, :real_skv, :].contiguous()

    print(f"[INFO] q_fp8 shape: {q_fp8.shape}, dtype: {q_fp8.dtype}")
    print(f"[INFO] k_fp8 shape: {k_fp8.shape}, dtype: {k_fp8.dtype}")
    print(f"[INFO] v_fp8 shape: {v_fp8.shape}, dtype: {v_fp8.dtype}")

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32).cpu().contiguous()

    return (q_fp8, k_fp8, v_fp8, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale)


# ==============================================================================
# NPU 排布 → BNSD 转换
# ==============================================================================
def _ntd_to_bnsd(q_ntd, actual_seq_q, N_q):
    """NTD (N_q, T, D) → BNSD (B, N_q, max_Sq, D)"""
    B = len(actual_seq_q)
    max_Sq = max(actual_seq_q)
    q_bnsd = torch.zeros((B, N_q, max_Sq, q_ntd.shape[-1]), dtype=q_ntd.dtype)
    offset = 0
    for b in range(B):
        act = actual_seq_q[b]
        for n in range(N_q):
            q_bnsd[b, n, :act, :] = q_ntd[n, offset:offset + act, :]
        offset += act
    return q_bnsd


def _bnbd_to_bnsd(kv_bnbd, block_table, actual_seq_kv, block_size):
    """BNBD (block_num, N_kv, block_size, D) → BNSD (B, N_kv, max_Skv, D)

    反向还原 bnsd_to_k_cache / bnsd_to_v_cache
    """
    B = len(actual_seq_kv)
    N_kv = kv_bnbd.shape[1]
    D = kv_bnbd.shape[-1]
    max_Skv = max(max(actual_seq_kv), 1)
    kv_bnsd = torch.zeros((B, N_kv, max_Skv, D), dtype=kv_bnbd.dtype)

    for b in range(B):
        seq_len = actual_seq_kv[b]
        block_num_per_seq = math.ceil(seq_len / block_size)
        for blk_idx in range(block_num_per_seq):
            block_id = int(block_table[b, blk_idx])
            if block_id < 0:
                continue
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_len)
            valid = end_s - start_s
            if valid <= 0:
                continue
            kv_bnsd[b, :, start_s:end_s, :] = kv_bnbd[block_id, :, :valid, :]
    return kv_bnsd


def _nt_to_bns1(q_scale_nt, actual_seq_q, N_q):
    """NT (N_q, T) → BNS1 (B, N_q, max_Sq, 1)"""
    B = len(actual_seq_q)
    max_Sq = max(actual_seq_q)
    q_scale_bns1 = torch.zeros((B, N_q, max_Sq, 1), dtype=torch.float32)
    offset = 0
    for b in range(B):
        act = actual_seq_q[b]
        for n in range(N_q):
            q_scale_bns1[b, n, :act, 0] = q_scale_nt[n, offset:offset + act]
        offset += act
    return q_scale_bns1


def _bnb_to_bns1(k_scale_bnb, block_table, actual_seq_kv, block_size):
    """BNB (block_num, N_kv, block_size) → BNS1 (B, N_kv, max_Skv, 1)"""
    B = len(actual_seq_kv)
    N_kv = k_scale_bnb.shape[1]
    max_Skv = max(max(actual_seq_kv), 1)
    k_scale_bns1 = torch.zeros((B, N_kv, max_Skv, 1), dtype=torch.float32)

    for b in range(B):
        seq_len = actual_seq_kv[b]
        block_num_per_seq = math.ceil(seq_len / block_size)
        for blk_idx in range(block_num_per_seq):
            block_id = int(block_table[b, blk_idx])
            if block_id < 0:
                continue
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_len)
            valid = end_s - start_s
            if valid <= 0:
                continue
            # print(f"##### {start_s=}, {end_s=}, {block_size=}, {seq_len=}")
            k_scale_bns1[b, :, start_s:end_s, 0] = k_scale_bnb[block_id, :, :valid]
    return k_scale_bns1


def external_to_bnsd(ext):
    """将外部 NPU 排布 tensor 转换为 BNSD 格式，供 CPU Golden 使用

    返回 (q_fp8_bnsd, k_fp8_bnsd, v_fp8_bnsd,
          deq_q_bns1, deq_k_bns1, deq_v_1N11, p_scale)
    """
    print("[INFO] Converting external (NPU layout) → BNSD for CPU golden...")

    q_bnsd = _ntd_to_bnsd(ext.q_fp8, ACTUAL_SEQ_Q, N_q)
    k_bnsd = _bnbd_to_bnsd(ext.k_fp8, ext.block_table, ACTUAL_SEQ_KV, ext.block_size)
    v_bnsd = _bnbd_to_bnsd(ext.v_fp8, ext.block_table, ACTUAL_SEQ_KV, ext.block_size)

    deq_q_bns1 = _nt_to_bns1(ext.deq_q, ACTUAL_SEQ_Q, N_q)
    print(f"######## ext.deq_k shape{ext.deq_k.shape}")
    deq_k_bns1 = _bnb_to_bns1(ext.deq_k, ext.block_table, ACTUAL_SEQ_KV, ext.block_size)
    # v scale: N → (1, N, 1, 1)
    deq_v_1N11 = ext.deq_v.reshape(1, N_kv, 1, 1).float()

    print(f"  q_bnsd: {q_bnsd.shape}")
    print(f"  k_bnsd: {k_bnsd.shape}")
    print(f"  v_bnsd: {v_bnsd.shape}")
    print(f"  deq_q_bns1: {deq_q_bns1.shape}")
    print(f"  deq_k_bns1: {deq_k_bns1.shape}")
    print(f"  deq_v_1N11: {deq_v_1N11.shape}")

    return (q_bnsd, k_bnsd, v_bnsd, deq_q_bns1, deq_k_bns1, deq_v_1N11, ext.p_scale)


# ==============================================================================
# CPU Golden 函数
# ==============================================================================
def get_softmax_scale(scale_value, head_dim):
    if scale_value is not None:
        return float(scale_value)
    return 1.0 / math.sqrt(head_dim)

def torch_broadcast_kv(num_heads, num_kv_heads, tensor):
    if num_heads == num_kv_heads:
        return tensor.contiguous()
    factor = num_heads // num_kv_heads
    return tensor.repeat_interleave(factor, dim=1).contiguous()

def cpu_fp8_fullquant_gqa_golden(q_fp8, k_fp8, v_fp8, 
                            deq_q, deq_k, deq_v, p_scale,
                            actual_seq_q, actual_seq_kv):
    """CPU golden reference - 所有操作在CPU上执行"""
    softmax_scale = get_softmax_scale(SCALE_VALUE, D)
    q_tensor = q_fp8.cpu().to(torch.float32).contiguous()
    batch, heads, q_seq, d_dim = q_tensor.shape

    k_tensor = k_fp8.cpu().to(torch.float32).contiguous()
    v_tensor = v_fp8.cpu().to(torch.float32).contiguous()
    deq_q = deq_q.cpu().float().contiguous()
    deq_k = deq_k.cpu().float().contiguous()
    deq_v = deq_v.cpu().float().contiguous()

    if N_q != N_kv:
        k_tensor = torch_broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = torch_broadcast_kv(N_q, N_kv, v_tensor)
        deq_k = torch_broadcast_kv(N_q, N_kv, deq_k)
        deq_v = torch_broadcast_kv(N_q, N_kv, deq_v)

    batch, heads, q_seq, _ = q_tensor.shape
    v_dim = v_tensor.shape[-1]

    # 空 KV 场景：无 KV token 可参与 attention，输出全零，LSE 为 inf
    if k_tensor.shape[2] == 0:
        result = torch.zeros((batch, heads, q_seq, v_dim), dtype=torch.float32).contiguous()
        lse = torch.full((batch, heads, q_seq, 1), float('inf'), dtype=torch.float32).contiguous()
        return result, lse

    out = torch.zeros((batch, heads, q_seq, v_dim), dtype=torch.float32).contiguous()
    o_sum = torch.zeros(q_tensor.shape[:-1], dtype=torch.float32)[..., None].contiguous()
        # 修改1: 使用 NPU 对应的最小值初始化 (0xFF7FFFFF 对应 -FLT_MAX)
    minValue = torch.tensor(-3.402823466e+38, dtype=torch.float32)
    o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float32)[..., None].contiguous()
    # o_max = torch.ones(q_tensor.shape[:-1], dtype=torch.float32)[..., None] * torch.finfo(torch.float32).min
    # o_max = o_max.contiguous()

    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32).contiguous()
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32).contiguous()
    q_lens_acl = q_lens_t.view(batch, 1, 1, 1).contiguous()
    k_lens_acl = k_lens_t.view(batch, 1, 1, 1).contiguous()

    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]
    q_range = torch.arange(Sq).view(1, 1, -1, 1).contiguous()
    k_range = torch.arange(Skv).view(1, 1, 1, -1).contiguous()
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if SPARSE_MODE == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        mask_global = causal_mask | q_padding_mask | k_padding_mask
    else:
        mask_global = q_padding_mask | k_padding_mask
    mask_global = mask_global.contiguous()
    
    mask_q_blocks = list(torch.split(mask_global, Q_BLOCK_SIZE, dim=2))
    mask_blocks = []
    for mask_q_block in mask_q_blocks:
        mask_blocks.append(list(torch.split(mask_q_block, KV_BLOCK_SIZE, dim=3)))

    q_blocks = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    k_blocks = list(torch.split(k_tensor, KV_BLOCK_SIZE, dim=2))
    v_blocks = list(torch.split(v_tensor, KV_BLOCK_SIZE, dim=2))
    o_blocks = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_blocks = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_blocks = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))
    deq_q_blocks = list(torch.split(deq_q, Q_BLOCK_SIZE, dim=2))
    deq_k_blocks = list(torch.split(deq_k, KV_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale.item())], dtype=torch.float32).contiguous()
    
    for j, (kj, vj) in enumerate(zip(k_blocks, v_blocks)):
        kj = kj.contiguous()
        kj_T = kj.transpose(-1, -2).contiguous()
        vj = vj.contiguous()
        deq_kj = deq_k_blocks[j]
        deq_kj_T = deq_kj.transpose(-1, -2).contiguous()

        for i, qi in enumerate(q_blocks):
            oi = o_blocks[i]
            si = s_blocks[i]
            mi = m_blocks[i]
            deq_qi = deq_q_blocks[i]

            sij = torch.matmul(qi, kj_T)
            sij = sij * deq_qi * deq_kj_T
            sij = sij * softmax_scale

            causal_mask = mask_blocks[i][j].contiguous()
            sij = sij.masked_fill(causal_mask, float('-inf'))

            m_block, _ = torch.max(sij, dim=-1, keepdims=True)
            m_block = m_block - ln_p_scale
            mi_new = torch.maximum(m_block, mi)
            all_masked_block = (m_block == float('-inf'))
            pij = torch.where(all_masked_block,
                              torch.zeros_like(sij),
                              torch.exp(sij - mi_new))
            s_block = torch.sum(pij, dim=-1, keepdims=True)
            pij_drop = pij.to(FP8_DTYPE).to(torch.float32)
            pij_v = torch.matmul(pij_drop, vj)
            pij_v = pij_v * deq_v
            scale = torch.where(mi_new == float('-inf'),
                                torch.ones_like(mi_new),
                                torch.exp(mi - mi_new))
            si_new = scale * si + s_block
            o_blocks[i] = (si * torch.exp(mi - mi_new) * oi + pij_v) / (si_new + EPSILON)
            s_blocks[i] = si_new
            m_blocks[i] = mi_new

    result = torch.cat(o_blocks, dim=2).contiguous()
    out_sum = torch.cat(s_blocks, dim=2).contiguous()
    out_max = torch.cat(m_blocks, dim=2).contiguous()
    lse = out_max + torch.log(out_sum + EPSILON).contiguous()

    # 当 out_max 等于 minValue (0xFF7FFFFF) 时，说明该位置被全 mask，输出 inf
    all_masked = (out_max <= minValue.item())
    lse = torch.where(all_masked, 
                      torch.full_like(out_max, float('inf')),
                      out_max + torch.log(out_sum + EPSILON)).contiguous()
    result = torch.where(all_masked, torch.zeros_like(result), result)
    return result, lse

# ==============================================================================
# Layout 转换
# ==============================================================================
def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    """BNSD → 各种 layout"""
    tensor = tensor_bnsd if isinstance(tensor_bnsd, torch.Tensor) else torch.as_tensor(tensor_bnsd)
    tensor = tensor.cpu().contiguous()
    B, N, _, D = tensor.shape
    max_org_s = max(seq_lens)
    
    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).reshape(B, max_org_s, N * D).contiguous()
    elif layout == "TND":
        T = sum(seq_lens)
        result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t:t + act_s, n, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    elif layout == "NTD_TND":
        T = sum(seq_lens)
        result = torch.zeros((N, T, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[n, t:t + act_s, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")

def convert_scale_to_layout(tensor, seq_lens, scale_type):
    """Scale to layout"""
    tensor = tensor.cpu().contiguous()
    if scale_type == "deq_q":
        B, N, _, _ = tensor.shape
        T = sum(seq_lens)
        if Q_SCALE_LAYOUT == "NT":
            result = torch.zeros((N, T), dtype=torch.float32)
            t = 0
            for b in range(B):
                act_s = seq_lens[b]
                for n in range(N):
                    result[n, t:t + act_s] = tensor[b, n, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif Q_SCALE_LAYOUT == "TN":
            result = torch.zeros((T, N), dtype=torch.float32)
            t = 0
            for b in range(B):
                act_s = seq_lens[b]
                for n in range(N):
                    result[t:t + act_s, n] = tensor[b, n, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif Q_SCALE_LAYOUT == "BNSD":
            result = torch.zeros((B, N, max(seq_lens), 1), dtype=torch.float32)
            for b in range(B):
                act_s = seq_lens[b]
                for n in range(N):
                    result[b, n, :act_s, 0] = tensor[b, n, :act_s, 0]
            return result.contiguous()
        else:
            return tensor.float().contiguous()
    elif scale_type == "deq_v":
        return tensor.reshape(tensor.shape[1]).float().contiguous()
    return tensor.squeeze(-1).contiguous()

def make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result

# ==============================================================================
# NPU 调用
# GRAPH_PATH: 0=单算子, 5=动态图, 7=aclgraph
# ==============================================================================
def get_npu_fa_kwargs():
    return dict(
        query_quant_mode=3,
        key_quant_mode=3,
        value_quant_mode=2,
        query_dtype=FP8_DTYPE,
        key_dtype=FP8_DTYPE,
        value_dtype=FP8_DTYPE,
        dequant_scale_query_dtype=torch.float32,
        dequant_scale_key_dtype=torch.float32,
        dequant_scale_value_dtype=torch.float32,
        return_softmax_lse=ENABLE_LSE,
    )

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, q, k, v, mask, actual_seq_q, actual_seq_kv,
                dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                block_table, q_n, kv_n, softmax_scale, layout, block_size, out_dtype):
        atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
            q, k, v,
            atten_mask=mask,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            dequant_scale_query=dequant_scale_q,
            dequant_scale_key=dequant_scale_k,
            dequant_scale_value=dequant_scale_v,
            block_table=block_table,
            block_size=block_size,
            num_query_heads=q_n,
            num_key_value_heads=kv_n,
            softmax_scale=softmax_scale,
            input_layout=layout,
            sparse_mode=SPARSE_MODE,
            quant_scale_p=p_scale,
            out_dtype=out_dtype,
            **get_npu_fa_kwargs(),
        )
        return atten_out, lse_out

def call_npu_fa_op(q, k, v, mask, actual_seq_q, actual_seq_kv,
                    dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                    block_table, q_n, kv_n, softmax_scale, layout, block_size, out_dtype):
    torch.npu.synchronize()
    atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
        q, k, v,
        atten_mask=mask,
        actual_seq_qlen=actual_seq_q,
        actual_seq_kvlen=actual_seq_kv,
        dequant_scale_query=dequant_scale_q,
        dequant_scale_key=dequant_scale_k,
        dequant_scale_value=dequant_scale_v,
        block_table=block_table,
        block_size=block_size,
        num_query_heads=q_n,
        num_key_value_heads=kv_n,
        softmax_scale=softmax_scale,
        input_layout=layout,
        sparse_mode=SPARSE_MODE,
        quant_scale_p=p_scale,
        out_dtype=out_dtype,
        **get_npu_fa_kwargs(),
    )
    torch.npu.synchronize()
    return atten_out, lse_out

def fia_gqa_torch_npu(q, k, v, mask, actual_seq_q, actual_seq_kv,
                      dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                      block_table, q_n, kv_n, softmax_scale, layout, block_size, out_dtype):
    if GRAPH_PATH == 0:
        print("[INFO] GRAPH_PATH == 0, 单算子模式...")
        return call_npu_fa_op(
            q, k, v, mask, actual_seq_q, actual_seq_kv,
            dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
            block_table, q_n, kv_n, softmax_scale, layout, block_size, out_dtype)

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    config = CompilerConfig()
    with torch.no_grad():
        torch.npu.synchronize()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        fa_args = (q, k, v, mask, actual_seq_q, actual_seq_kv,
                   dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                   block_table, q_n, kv_n, softmax_scale, layout, block_size, out_dtype)

        if GRAPH_PATH == 5:
            print("[INFO] GRAPH_PATH == 5, 动态图...")
            torch._dynamo.reset()
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)
            atten_out, lse_out = npu_mode(*fa_args)
        elif GRAPH_PATH == 7:
            print("[INFO] GRAPH_PATH == 7, aclgraph...")
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
            config.mode = "reduce-overhead"
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)
            for t in (q, k, v, mask,
                      dequant_scale_q, dequant_scale_k, dequant_scale_v,
                      p_scale, block_table):
                if t is not None:
                    torch._dynamo.mark_static(t)
            atten_out, lse_out = npu_mode(*fa_args)
        else:
            raise ValueError(f"Unsupported GRAPH_PATH: {GRAPH_PATH}, only support 0/5/7")

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        torch.npu.synchronize()
        return atten_out, lse_out

def fa_run_npu(q, k, v, mask, actual_seq_q, actual_seq_kv,
                dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                block_table, block_size, q_n, kv_n, softmax_scale, layout, out_dtype):
    """将数据转移到NPU上并调用NPU算子"""
    device_id = DEVICE_ID
    torch_npu.npu.set_device(int(device_id))
    
    # 确保所有输入都在NPU上且有正确的数据类型
    q = q.npu()
    k = k.npu()
    v = v.npu()

    # 取出deq_k，数据在NPU上，且是float32类型
    k_pa_f32 = (
        k
        .view(torch.uint8)                  # (Bn, N, Bs+4, D)   → uint8，仍是 view，stride 不变
        .view(k.shape[0], k.shape[1], -1)   # (Bn, N, 16896)     → 合并最后两维，仍是 view
        .view(torch.float32)                # (Bn, N, 4224)      → float32，仍是 view
    )
    dequant_scale_k = k_pa_f32[:, :, -BLOCK_SIZE:]   # (Bn, N, 128) float32

    # dequant scales 必须是 float32 类型
    dequant_scale_q = dequant_scale_q.float().npu()
    dequant_scale_v = dequant_scale_v.float().npu()
    p_scale = p_scale.float().npu()
    
    if not IS_CONTIGUOUS and ENABLE_PA:
        fake_kscale_tensor = torch.ones_like(dequant_scale_k)
        double_kscale = torch.stack([dequant_scale_k, fake_kscale_tensor], dim=2)
        double_kscale = double_kscale.npu()
        dequant_scale_k = double_kscale[:, :, 0] # 覆写为非连续

    # block_table 必须是 int32 类型
    block_table = block_table.int().npu() if ENABLE_PA else None

    # mask 如果有的话，转换为 bool 类型
    if mask is not None:
        mask = mask.bool().npu()

    # 将kv从cache中取切片
    k = k[:, :, :128, :]
    v = v[:, :, :128, :]

    # 打印调试信息
    print(f"[INFO] q dtype: {q.dtype}, shape: {q.shape}")
    print(f"[INFO] k dtype: {k.dtype}, shape: {k.shape}")
    print(f"[INFO] v dtype: {v.dtype}, shape: {v.shape}")
    print(f"[INFO] deq_q dtype: {dequant_scale_q.dtype}, shape: {dequant_scale_q.shape}")
    print(f"[INFO] deq_k dtype: {dequant_scale_k.dtype}, shape: {dequant_scale_k.shape}")
    print(f"[INFO] deq_v dtype: {dequant_scale_v.dtype}, shape: {dequant_scale_v.shape}")
    print(f"[INFO] NPU input layout: {layout}, sparse_mode: {SPARSE_MODE}")
    print(f"[INFO] key is_contiguous: {k.is_contiguous()}, value is_contiguous: {v.is_contiguous()}")
    print(f"[INFO] key stride: {k.stride()}, value stride: {v.stride()}")
    print(f"[INFO] deq_k is_contiguous: {dequant_scale_k.is_contiguous()}")
    print(f"[INFO] deq_k stride: {dequant_scale_k.stride()}")
    print(f"[INFO] --- tensor devices ---")
    print(f"  q.device={q.device}, k.device={k.device}, v.device={v.device}")
    print(f"  deq_q.device={dequant_scale_q.device}, deq_k.device={dequant_scale_k.device}, deq_v.device={dequant_scale_v.device}")
    print(f"  p_scale.device={p_scale.device}, block_table.device={block_table.device if block_table is not None else None}")
    print(f"  mask.device={mask.device if mask is not None else None}")

    # 从这里就开始进入NPU调用流程，不再处理数据
    atten_out, lse_out = fia_gqa_torch_npu(
        q, k, v, mask, actual_seq_q, actual_seq_kv,
        dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
        block_table, q_n, kv_n, softmax_scale, layout, block_size, out_dtype)

    if GRAPH_PATH == 0:
        atten_out = atten_out.cpu()
        lse_out = lse_out.cpu()
    return atten_out, lse_out

def npu_fp8_full_quant(q_fp8, k_fp8, v_fp8,
                       dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                       actual_seq_q, actual_seq_kv):
    """Main NPU quant function - prepares data and calls NPU"""
    """这个函数没有调用.npu(), 数据全都在CPU上, 需要在调用NPU前将数据转移到NPU上"""
    softmax_scale = 1.0 / math.sqrt(D)
    T = sum(actual_seq_q)
    out_dtype = OUTPUT_DETYPE

    accum_seq_q = make_accum_seq(actual_seq_q) if INPUT_LAYOUT in ("NTD_TND", "TND") else actual_seq_q

    npu_input_layout = INPUT_LAYOUT
    
    # 确保 q 是 FP8 类型
    q_npu = convert_q_bnsd_to_layout(q_fp8, actual_seq_q, npu_input_layout)

    # dequant scales 使用 float32
    deq_q_npu = convert_scale_to_layout(dequant_scale_q, ACTUAL_SEQ_Q, 'deq_q')
    deq_v_npu = convert_scale_to_layout(dequant_scale_v, ACTUAL_SEQ_KV, 'deq_v')

    if SPARSE_MODE == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
    else:
        mask = None

    if ENABLE_PA or not GOLDEN_MODE:
        # 在CPU上准备block table和cache
        block_table = create_block_table(ACTUAL_SEQ_KV, BLOCK_SIZE)
        block_table_tensor = torch.as_tensor(block_table, dtype=torch.int32)
        # 生成k和v的cache
        k_pa = bnsd_to_k_cache(k_fp8, dequant_scale_k, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table)
        v_pa = bnsd_to_v_cache(v_fp8, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table)

        # 重要！！！此处取出deq_k其实没用，因为传入NPU时会使用.npu(), 导致stride变成连续的！！！
        # 在这里取出deq_k只是为了保存为pt文件（其实也用不到deq_k的pt文件）
        # 将deq_k从k_pa中提取出来，确保deq_k和k_pa共享内存
        k_pa_f32 = (
            k_pa
            .view(torch.uint8)                        # (Bn, N, Bs+4, D)   → uint8，仍是 view，stride 不变
            .view(k_pa.shape[0], k_pa.shape[1], -1)   # (Bn, N, 16896)     → 合并最后两维，仍是 view
            .view(torch.float32)                      # (Bn, N, 4224)      → float32，仍是 view
        )
        deq_k_npu = k_pa_f32[:, :, -BLOCK_SIZE:]   # (Bn, N, 128) float32

        # 构造kvcache非连续
        if not IS_CONTIGUOUS:
            kv_cache = torch.stack([k_pa, v_pa], dim=2)
            kv_cache = kv_cache.npu()
            k_pa = kv_cache[:, :, 0]
            v_pa = kv_cache[:, :, 1]

        if SAVE_PT:
            import os
            os.makedirs(SAVE_PT_DIR, exist_ok=True)
            torch.save(q_npu, os.path.join(SAVE_PT_DIR, "q_fp8.pt"))
            torch.save(k_pa, os.path.join(SAVE_PT_DIR, "k_fp8.pt"))
            torch.save(v_pa, os.path.join(SAVE_PT_DIR, "v_fp8.pt"))
            torch.save(deq_q_npu, os.path.join(SAVE_PT_DIR, "deq_q.pt"))
            torch.save(deq_k_npu, os.path.join(SAVE_PT_DIR, "deq_k.pt"))
            torch.save(deq_v_npu, os.path.join(SAVE_PT_DIR, "deq_v.pt"))  
            torch.save(block_table_tensor, os.path.join(SAVE_PT_DIR, "block_table.pt"))
            torch.save(accum_seq_q, os.path.join(SAVE_PT_DIR, "seq_q.pt"))
            torch.save(actual_seq_kv, os.path.join(SAVE_PT_DIR, "seq_kv.pt"))
            torch.save(mask, os.path.join(SAVE_PT_DIR, "mask.pt"))

        output = fa_run_npu(q_npu, k_pa, v_pa, mask, accum_seq_q, actual_seq_kv,
                           deq_q_npu, deq_k_npu, deq_v_npu, p_scale, 
                           block_table_tensor, BLOCK_SIZE, N_q, N_kv, 
                           softmax_scale, npu_input_layout, out_dtype)
    else:
        raise NotImplementedError("当前仅支持 PA 模式")

    atten_out = output[0]
    T_actual = sum(actual_seq_q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]
    return output

def npu_fp8_full_quant_external(ext):
    """使用外部 NPU 排布数据直接调用 NPU 算子"""
    print(f"[INFO] Using external NPU data")
    softmax_scale = 1.0 / math.sqrt(D)
    T = sum(ACTUAL_SEQ_Q)
    out_dtype = OUTPUT_DETYPE

    if max(ACTUAL_SEQ_KV) == 0:
        atten_out = torch.zeros((T, N_q, D), dtype=out_dtype)
        lse_out = torch.full((T, N_q, 1), float('inf'), dtype=torch.float32)
        return (atten_out, lse_out)

    accum_seq_q = make_accum_seq(ACTUAL_SEQ_Q) if INPUT_LAYOUT in ("NTD_TND", "TND") else ACTUAL_SEQ_Q

    # 设置 device
    torch_npu.npu.set_device(int(DEVICE_ID))

    q_npu = ext.q_fp8.npu()
    k_npu = ext.k_fp8.npu()
    v_npu = ext.v_fp8.npu()
    deq_q_npu = ext.deq_q.float().npu()
    # deq_k_npu = ext.deq_k.float().npu()
    deq_v_npu = ext.deq_v.float().npu()
    p_scale_npu = ext.p_scale.float().npu()
    block_table_npu = ext.block_table.int().npu()

    k_pa_f32 = (
        k_npu
        .view(torch.uint8)                          # (Bn, N, Bs+4, D)   → uint8，仍是 view，stride 不变
        .view(k_npu.shape[0], k_npu.shape[1], -1)   # (Bn, N, 16896)     → 合并最后两维，仍是 view
        .view(torch.float32)                        # (Bn, N, 4224)      → float32，仍是 view
    )
    deq_k_npu = k_pa_f32[:, :, -BLOCK_SIZE:]   # (Bn, N, 128) float32

    if ext.mask is not None:
        mask = ext.mask.bool().npu()
        print(f"[INFO] Using external mask, shape: {mask.shape}")
    elif SPARSE_MODE == 3:
        mask = torch.triu(torch.ones(1, 2048, 2048, dtype=torch.bool), diagonal=1).npu()
    else:
        mask = None

    print(f"[INFO] q dtype: {q_npu.dtype}, shape: {q_npu.shape}")
    print(f"[INFO] k dtype: {k_npu.dtype}, shape: {k_npu.shape}")
    print(f"[INFO] v dtype: {v_npu.dtype}, shape: {v_npu.shape}")
    print(f"[INFO] deq_q_npu dtype: {deq_q_npu.dtype}, shape: {deq_q_npu.shape}")
    print(f"[INFO] deq_k_npu dtype: {deq_k_npu.dtype}, shape: {deq_k_npu.shape}")
    print(f"[INFO] deq_v_npu dtype: {deq_v_npu.dtype}, shape: {deq_v_npu.shape}")
    print(f"[INFO] key is_contiguous: {k_npu.is_contiguous()}, value is_contiguous: {v_npu.is_contiguous()}")
    print(f"[INFO] key stride: {k_npu.stride()}, value stride: {v_npu.stride()}")
    print(f"[INFO] deq_k is_contiguous: {deq_k_npu.is_contiguous()}")
    print(f"[INFO] deq_k stride: {deq_k_npu.stride()}")
    print(f"[INFO] --- tensor devices ---")
    print(f"  q_npu.device={q_npu.device}, k_npu.device={k_npu.device}, v_npu.device={v_npu.device}")
    print(f"  deq_q_npu.device={deq_q_npu.device}, deq_k_npu.device={deq_k_npu.device}, deq_v_npu.device={deq_v_npu.device}")
    print(f"  block_table_npu.device={block_table_npu.device}")
    print(f"  mask.device={mask.device if mask is not None else None}")

    torch.npu.synchronize()
    atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
        q_npu, k_npu[:, :, :128, :], v_npu[:, :, :128, :],
        atten_mask=mask,
        actual_seq_qlen=accum_seq_q,
        actual_seq_kvlen=ACTUAL_SEQ_KV,
        dequant_scale_query=deq_q_npu,
        dequant_scale_key=deq_k_npu,
        dequant_scale_value=deq_v_npu,
        block_table=block_table_npu,
        block_size=ext.block_size,
        num_query_heads=N_q,
        num_key_value_heads=N_kv,
        softmax_scale=softmax_scale,
        input_layout=INPUT_LAYOUT,
        sparse_mode=SPARSE_MODE,
        quant_scale_p=p_scale_npu,
        out_dtype=out_dtype,
        query_quant_mode=3,
        key_quant_mode=3,
        value_quant_mode=2,
        query_dtype=FP8_DTYPE,
        key_dtype=FP8_DTYPE,
        value_dtype=FP8_DTYPE,
        dequant_scale_query_dtype=torch.float32,
        dequant_scale_key_dtype=torch.float32,
        dequant_scale_value_dtype=torch.float32,
        return_softmax_lse=ENABLE_LSE,
    )
    torch.npu.synchronize()
    atten_out = atten_out.cpu()
    lse_out = lse_out.cpu()

    T_actual = sum(ACTUAL_SEQ_Q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]
    return (atten_out, lse_out)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("FIA Full Quant GQA GOLDEN")
    print("=" * 60)
    print(f"[INFO] 场景: {'PA' if ENABLE_PA else 'noPA'}, INPUT_LAYOUT: {INPUT_LAYOUT}, OUTPUT_LAYOUT: {OUTPUT_LAYOUT}")

    print("\n[Step 1] 数据加载")
    external_data = None
    if USE_EXTERNAL_INPUT:
        if LOAD_PT_DIR is None:
            raise ValueError("USE_EXTERNAL_INPUT=True but LOAD_PT_DIR is None, please set LOAD_PT_DIR")
        external_data = load_data_from_dir(LOAD_PT_DIR)
        # 从 ExternalData 中读取推断值，同步到模块级全局变量
        B = external_data.B
        N_q = external_data.N_q
        N_kv = external_data.N_kv
        D = external_data.D
        ACTUAL_SEQ_Q = external_data.ACTUAL_SEQ_Q
        ACTUAL_SEQ_KV = external_data.ACTUAL_SEQ_KV
        BLOCK_SIZE = external_data.block_size
        bnsd_tup = external_to_bnsd(external_data)
    else:
        bnsd_tup = generate_data()
    print(f"[INFO] B={B}, N_q={N_q}, N_kv={N_kv}, D={D}")
    print(f"[INFO] ACTUAL_SEQ_Q={ACTUAL_SEQ_Q}")
    print(f"[INFO] ACTUAL_SEQ_KV={ACTUAL_SEQ_KV}")
    (q_fp8, k_fp8, v_fp8, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale) = bnsd_tup
    
    print("\n[Step 2] CPU Golden")
    if GOLDEN_MODE:
        cpu_out= cpu_fp8_fullquant_gqa_golden(q_fp8, k_fp8, v_fp8,
                                                        dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                                                        ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
        print("[INFO] cpu_out[0] shape", cpu_out[0].shape)
        if ENABLE_LSE:
            print("[INFO] cpu_out[1] shape", cpu_out[1].shape)
    else:
        print("\n[Step 2] GOLDEN_MODE=False, skip CPU Golden.")
    # print(f"{cpu_out[0]=}")
    # print(f"{cpu_out[1]=}")

    print("\n[Step 3] NPU: 执行NPU计算")
    if USE_EXTERNAL_INPUT:
        npu_out = npu_fp8_full_quant_external(external_data)
    else:
        npu_out = npu_fp8_full_quant(q_fp8, k_fp8, v_fp8,
                                     dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                                     ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
    # torch.save(npu_out[0], './wk_out/atten_out_5.pt')
    print(f"[INFO] npu_out[0] shape: {npu_out[0].shape}, dtype: {npu_out[0].dtype}")
    if ENABLE_LSE:
        print(f"[INFO] npu_out[1] shape: {npu_out[1].shape}, dtype: {npu_out[1].dtype}")

    if GOLDEN_MODE:
        print("\n[Step 4] Atten out 精度对比")
        cpu_tnd_torch = convert_q_bnsd_to_layout(cpu_out[0], ACTUAL_SEQ_Q, OUTPUT_LAYOUT)
        print(f"cpu_out[0] final shape = {cpu_tnd_torch.shape}")
        result_compare_method.check_result(cpu_tnd_torch, npu_out[0])
        if ENABLE_LSE:
            print("\n[Step 5] LSE 精度对比")
            cpu_lse_tnd_torch = convert_q_bnsd_to_layout(cpu_out[1], ACTUAL_SEQ_Q, "TND")
            result_compare_method.check_result(cpu_lse_tnd_torch, npu_out[1])
    else:
        print("\n[Step 4] GOLDEN_MODE=False, skip precision check.")
    print("#" * 60)
    print()
