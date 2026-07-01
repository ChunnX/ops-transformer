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
# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 5=动态图, 7=aclgraph
GRAPH_PATH = 0
DEVICE_ID = 0

B = 1
N_q = 4
N_kv = 2
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
Q_DATA_RANGE = (-1.0, 1.0)
K_DATA_RANGE = (-5.0, 5.0)
V_DATA_RANGE = (-5.0, 5.0)

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
SEED_V = 4
SEED_BLOCK_TABLE = 1234
FP8_DTYPE = torch.float8_e4m3fn
P_SCALE = 1.0
EPSILON = 1e-20

Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 256

# ==============================================================================
# CPU 数据生成函数
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
    return scale.contiguous()

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
    q_fp16 = torch.from_numpy(
        np.random.uniform(
            low=Q_DATA_RANGE[0],
            high=Q_DATA_RANGE[1],
            size=(B, N_q, max_sq, D),
        ).astype(np.float16)
    )
    np.random.seed(SEED_K)
    k_fp16 = torch.from_numpy(
        np.random.uniform(
            low=K_DATA_RANGE[0],
            high=K_DATA_RANGE[1],
            size=(B, N_kv, max_skv, D),
        ).astype(np.float16)
    )
    np.random.seed(SEED_V)
    v_fp16 = torch.from_numpy(
        np.random.uniform(
            low=V_DATA_RANGE[0],
            high=V_DATA_RANGE[1],
            size=(B, N_kv, max_skv, D),
        ).astype(np.float16)
    )

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
    """NPU execution - only this function runs on NPU"""
    device_id = DEVICE_ID
    torch_npu.npu.set_device(int(device_id))
    
    # 确保所有输入都在NPU上且有正确的数据类型
    q = q.npu()
    k = k.npu()
    v = v.npu()
    
    # dequant scales 必须是 float32 类型
    dequant_scale_q = dequant_scale_q.float().npu()
    dequant_scale_k = dequant_scale_k.float().npu()
    dequant_scale_v = dequant_scale_v.float().npu()
    p_scale = p_scale.float().npu()
    if not IS_CONTIGUOUS and ENABLE_PA: # .float()之后tensor会变成连续，因此构造scale非连续需在float后
        fake_kscale_tensor = torch.ones_like(dequant_scale_k)
        double_kscale = torch.stack([dequant_scale_k, fake_kscale_tensor], dim=2)
        double_kscale = double_kscale.npu()
        dequant_scale_k = double_kscale[:, :, 0] # 覆写为非连续
    # block_table 必须是 int32 类型
    block_table = block_table.int().npu() if ENABLE_PA else None
    
    # mask 如果有的话，转换为 bool 类型
    if mask is not None:
        mask = mask.bool().npu()
    k = k[:, :, :128, :]
    v = v[:, :, :128, :]

    # 打印调试信息
    print(f"[INFO] q dtype: {q.dtype}, shape: {q.shape}")
    print(f"[INFO] k dtype: {k.dtype}, shape: {k.shape}")
    print(f"[INFO] v dtype: {v.dtype}, shape: {v.shape}")
    print(f"[INFO] deq_q dtype: {dequant_scale_q.dtype}, shape: {dequant_scale_q.shape}")
    print(f"[INFO] deq_k dtype: {dequant_scale_k.dtype}, shape: {dequant_scale_k.shape}")
    print(f"[INFO] deq_v dtype: {dequant_scale_v.dtype}, shape: {dequant_scale_v.shape}")
    print(f"[INFO] NPU input layout: {layout}")
    print(f"[INFO] key is_contiguous: {k.is_contiguous()}, value is_contiguous: {v.is_contiguous()}")
    print(f"[INFO] key stride: {k.stride()}, value stride: {v.stride()}")
    print(f"[INFO] deq_k is_contiguous: {dequant_scale_k.is_contiguous()}")
    print(f"[INFO] deq_k stride: {dequant_scale_k.stride()}")

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
    softmax_scale = 1.0 / math.sqrt(D)
    T = sum(actual_seq_q)
    out_dtype = torch.float16

    # 空 KV 场景：无 KV token 可参与 attention，跳过 NPU 调用，直接返回全零 + inf
    if max(actual_seq_kv) == 0:
        atten_out = torch.zeros((T, N_q, D), dtype=out_dtype)
        lse_out = torch.full((T, N_q, 1), float('inf'), dtype=torch.float32)
        return (atten_out, lse_out)

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
        # 确保 k_pa 和 v_pa 是正确的数据类型
        k_pa = bnsd_to_k_cache(k_fp8, dequant_scale_k, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table)
        v_pa = bnsd_to_v_cache(v_fp8, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table)

        # 从k cache的4行scale区域提取k scale (fp32字节存在fp8 cache中)
        scale_region = k_pa[:, :, BLOCK_SIZE:, :]
        deq_k_npu = scale_region.view(torch.uint8).view(torch.float32).reshape(scale_region.shape[0], scale_region.shape[1], -1)

        # 构造kvcache非连续
        if not IS_CONTIGUOUS:
            kv_cache = torch.stack([k_pa, v_pa], dim=2)
            kv_cache = kv_cache.npu()
            k_pa = kv_cache[:, :, 0]
            v_pa = kv_cache[:, :, 1]

        # 调用NPU
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

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("FIA Full Quant GQA GOLDEN")
    print("=" * 60)
    print(f"[INFO] 场景: {'PA' if ENABLE_PA else 'noPA'}, INPUT_LAYOUT: {INPUT_LAYOUT}, OUTPUT_LAYOUT: {OUTPUT_LAYOUT}")
    print(f"[INFO] B={B}, N_q={N_q}, N_kv={N_kv}, D={D}")
    print(f"[INFO] ACTUAL_SEQ_Q={ACTUAL_SEQ_Q}")
    print(f"[INFO] ACTUAL_SEQ_KV={ACTUAL_SEQ_KV}")

    print("\n[Step 1] CPU: 数据生成")
    (q_fp8, k_fp8, v_fp8, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale) = generate_data()

    if GOLDEN_MODE:
        print("\n[Step 2] CPU Golden")
        cpu_out= cpu_fp8_fullquant_gqa_golden(q_fp8, k_fp8, v_fp8,
                                                        dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                                                        ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
        print("[INFO] cpu_out[0] shape", cpu_out[0].shape)
        if ENABLE_LSE:
            print("[INFO] cpu_out[1] shape", cpu_out[1].shape)
    else:
        print("\n[Step 2] GOLDEN_MODE=False, skip CPU Golden.")

    print("\n[Step 3] NPU: 执行NPU计算")
    npu_out = npu_fp8_full_quant(q_fp8, k_fp8, v_fp8,
                                 dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                                 ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
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
