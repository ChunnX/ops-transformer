#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""
GPU 后端实现 (基于 flash-attn 库)
支持 BNSD, BSND, TND, PA_BBND, PA_BNBD 多种输入布局。
注意：调用方传入的 q/k/v 始终为 BNSD 格式（由 generate_qkv 生成），
内部根据 input_layout 决定使用哪个 API：
  - BNSD/BSND: flash_attn_func
  - TND: flash_attn_varlen_func
  - PA_*: flash_attn_with_kvcache
"""

import torch
from typing import Optional, List
from utils import (generate_qkv, generate_npu_mask, trans_bnsd_to_layout,
                         data_compare_benchmark_new, Result, gen_block_table)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("[WARN] flash_attn 未安装，GPU 功能不可用。请运行: pip install flash-attn --no-build-isolation")


def _bnsd_to_bshd(tensor: torch.Tensor) -> torch.Tensor:
    """[B, N, S, D] -> [B, S, N, D]"""
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bshd_to_bnsd(tensor: torch.Tensor) -> torch.Tensor:
    """[B, S, N, D] -> [B, N, S, D]"""
    return tensor.permute(0, 2, 1, 3).contiguous()


def flash_attn_gpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    使用 FlashAttention 库进行 GPU 计算。
    输入 q/k/v 形状均为 (B, N, S, D) BNSD 格式。
    输出形状为 (B, N, S_out, Dv) BNSD 格式，上层会调用 trans_bnsd_to_layout 转换。
    k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
        or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        page_block_size must be a multiple of 256.
    v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
        or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
    """
    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError("flash_attn 库未安装，无法使用 GPU 后端")

    input_layout = kwargs.get("input_layout", "BNSD")

    # 1. 拼接 rope（若维度 >0）
    if q_rope is not None and q_rope.shape[-1] > 0:
        q = torch.cat([q, q_rope], dim=-1)
    if k_rope is not None and k_rope.shape[-1] > 0:
        k = torch.cat([k, k_rope], dim=-1)

    # 2. 缩放因子
    softmax_scale = kwargs.get('scale', None)
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** (-0.5)

    # 3. 因果掩码
    mask_mode = kwargs.get('mask_mode', 0)
    causal = int(mask_mode) == 3 or kwargs.get('causal', False)

# ================== Paged Attention 处理 ==================
    layout_kv = kwargs.get("layout_kv", input_layout)
    if layout_kv.startswith("PA_"):
        # PA 格式支持：PA_BBND, PA_BNBD
        # 注意：当前实现为简化版本，将 BNSD reshape 为 paged format
        # 实际 PA 场景中 KV cache 应由专门的 paged tensor 生成（根据 block_table 映射）
        
        block_table = kwargs.get("block_table")
        block_size = kwargs.get("block_size")
        
        if block_table is None or block_size is None:
            raise ValueError(f"Paged Attention ({layout_kv}) 需要提供 block_table 和 block_size")
        
        # 转换 block_table 为 tensor: (batch_size, max_num_blocks_per_seq)
        if isinstance(block_table, list):
            if isinstance(block_table[0], list):
                block_table_tensor = torch.tensor(block_table, dtype=torch.int32, device=q.device)
            else:
                block_table_tensor = torch.tensor([block_table], dtype=torch.int32, device=q.device)
        else:
            block_table_tensor = block_table.to(device=q.device, dtype=torch.int32)
        
        B, N, S, D = k.shape
        
        if layout_kv in ("PA_BBND", "PA_BNBD"):
            # PA_BBND: (num_blocks, block_size, nheads, headdim)
            # PA_BNBD: (num_blocks, nheads, block_size, headdim)
            device = torch.cuda.current_device()
            k_cache = trans_bnsd_to_layout(k, "PA_BBND", **kwargs).contiguous().cuda(device)
            v_cache = trans_bnsd_to_layout(v, "PA_BBND", **kwargs).contiguous().cuda(device)
        else:
            raise ValueError(f"不支持的 PA layout: {layout_kv}")
        
        # Q 转换为 BSHD 格式
        q_bshd = _bnsd_to_bshd(q)
        
        # 提取 seqused_kv 作为 cache_seqlens
        seqused_kv = kwargs.get("_npu_seqused_kv") or kwargs.get("seqused_kv")
        if seqused_kv is None:
            cache_seqlens = torch.full((B,), S, dtype=torch.int32, device=q.device)
        elif isinstance(seqused_kv, list):
            cache_seqlens = torch.tensor(seqused_kv, dtype=torch.int32, device=q.device)
        else:
            cache_seqlens = seqused_kv.to(device=q.device, dtype=torch.int32)
        
        # 调用 flash_attn_with_kvcache
        out_bshd = flash_attn_with_kvcache(
            q_bshd,
            k_cache,
            v_cache,
            k=None,
            v=None,
            cache_seqlens=cache_seqlens,
            block_table=block_table_tensor,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
        )
        
        out_bnsd = _bshd_to_bnsd(out_bshd)
        return out_bnsd
    
    # ================== TND 变长序列处理 ==================
    if input_layout == "TND":
        cu_seqlens_q = kwargs.get("cu_seqlens_q") or kwargs.get("_npu_cu_q")
        cu_seqlens_kv = kwargs.get("cu_seqlens_kv") or kwargs.get("_npu_cu_kv")
        if not cu_seqlens_q or not cu_seqlens_kv:
            raise ValueError("TND layout 需要提供 cu_seqlens_q 和 cu_seqlens_kv")

        # 将 BNSD (B=1, N, total_tokens, D) 转换为 TND: (total_tokens, N, D)
        # q 的形状：(1, N, total_q_tokens, D) -> (total_q_tokens, N, D)
        q_tnd = q.squeeze(0).permute(1, 0, 2).contiguous()  # [1, N, T, D] -> [N, T, D] -> [T, N, D]
        k_tnd = k.squeeze(0).permute(1, 0, 2).contiguous()
        v_tnd = v.squeeze(0).permute(1, 0, 2).contiguous()

        # 计算 max_seqlen（用于 flash_attn_varlen_func 参数）
        seqlens_q = [cu_seqlens_q[i+1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q)-1)]
        seqlens_kv = [cu_seqlens_kv[i+1] - cu_seqlens_kv[i] for i in range(len(cu_seqlens_kv)-1)]
        max_seqlen_q = max(seqlens_q) if seqlens_q else 0
        max_seqlen_kv = max(seqlens_kv) if seqlens_kv else 0

        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=q.device)
        cu_kv = torch.tensor(cu_seqlens_kv, dtype=torch.int32, device=q.device)

        # 调用 flash_attn_varlen_func（变长序列版本）
        out_tnd = flash_attn_varlen_func(
            q_tnd, k_tnd, v_tnd,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
        )  # 输出形状 (total_tokens, N, Dv)

        # 将 TND 输出 (total_tokens, N, Dv) 转换为统一格式
        # 方案：保持 B=1（类似 generate_qkv 生成的输入格式），便于 trans_bnsd_to_layout 处理
        # 输出格式：(1, N, total_tokens, Dv) - 注意：第一维固定为 1
        total_tokens = out_tnd.size(0)
        N = out_tnd.size(1)
        Dv = out_tnd.size(2)
        
        # 将 (total_tokens, N, Dv) 转换为 (1, N, total_tokens, Dv)
        out_bnsd = out_tnd.unsqueeze(0).permute(0, 2, 1, 3)  # [T, N, D] -> [1, T, N, D] -> [1, N, T, D]
        
        return out_bnsd.contiguous()

    # ================== BNSD / BSND 固定长度处理 ==================
    # 输入已经是 BNSD，直接转置为 (B, S, N, D) 后调用 flash_attn_func
    q_bshd = _bnsd_to_bshd(q)
    k_bshd = _bnsd_to_bshd(k)
    v_bshd = _bnsd_to_bshd(v)

    out_bshd = flash_attn_func(
        q_bshd, k_bshd, v_bshd,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    out_bnsd = _bshd_to_bnsd(out_bshd)
    return out_bnsd