#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Input customization for FlashMlaWithKvcache TTK cases (m0070 接口，无 v / 无独立 q_rope/k_rope)。

E2E customize_inputs contract: modify tensors in-place via x.copy_(value).
Rule: only fill tensors that are provided (not None).
布局限定 TND/NZ/NTD（R1）：q=[T,N1,576]，k_cache=PA_NZ [num_blocks,1,36,block_size,16]。
"""

import torch


def _to_int_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def _fill_block_table(block_table, cache_seqlens, block_size):
    if block_table is None or not torch.is_tensor(block_table):
        return
    if cache_seqlens is None:
        return
    if block_size is None or int(block_size) <= 0:
        return

    shape = tuple(block_table.shape)
    if len(shape) < 2:
        return

    bt = torch.full(shape, -1, dtype=torch.int32)
    seq_list = _to_int_list(cache_seqlens)
    if not seq_list:
        return

    bs = int(block_size)
    block_idx = 0
    for b in range(shape[0]):
        kv_len = seq_list[b] if b < len(seq_list) else seq_list[-1]
        num_blocks_b = (kv_len + bs - 1) // bs
        for j in range(min(num_blocks_b, shape[1])):
            bt[b][j] = block_idx
            block_idx += 1
    block_table.copy_(bt.to(dtype=block_table.dtype, device=block_table.device))


def _fill_attn_mask(attn_mask, mask_mode):
    if attn_mask is None or not torch.is_tensor(attn_mask):
        return
    if mask_mode != 3:
        return
    # mask_mode=3 (CAUSAL/RIGHT_DOWN)：attn_mask 固定 2048x2048 上三角，对角线全 0
    mask = torch.triu(torch.ones(2048, 2048), diagonal=1)
    attn_mask.copy_(mask.to(dtype=attn_mask.dtype, device=attn_mask.device))


def customize_inputs(
    q,
    k_cache,
    *,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    metadata=None,
    attn_mask=None,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="TND",
    layout_kv="NZ",
    layout_out="NTD",
    return_softmax_lse=False,
    **kwargs,
):
    """Fill auxiliary tensors in-place for FlashMlaWithKvcache TTK cases.

    MLA 适配（对照 flash_attn 的 assets/impl/inputs.py）：
      - 无 v / 无 q_rope/k_rope 输入（rope 合并进 q/k_cache 576 宽最后维）
      - block_table 由 cache_seqlens（每 batch KV cache 长度）+ block_size 生成
      - KV 侧无 cu_seqlens_kv/seqused_kv（连续 KV 不在 R1 路由内）
    """
    block_size = kwargs.get("block_size")

    # 小整数张量（cu_seqlens_q/cache_seqlens/seqused_q）的值经 *_values attr 传入
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cache_seqlens", cache_seqlens),
        ("seqused_q", seqused_q),
    ):
        values = kwargs.get(f"{name}_values")
        if tensor is not None and values is not None:
            t = torch.tensor(list(values), dtype=tensor.dtype)
            tensor.copy_(t.to(device=tensor.device))

    if torch.is_tensor(block_table):
        _fill_block_table(block_table, cache_seqlens, block_size)

    if mask_mode == 3:
        _fill_attn_mask(attn_mask, mask_mode)