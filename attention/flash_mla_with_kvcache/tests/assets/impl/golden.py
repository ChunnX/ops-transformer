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

"""FlashMlaWithKvcache CPU golden（uniform attention golden 的 MLA 适配）。

S = (Q_nope ‖ Q_rope) · (K_nope ‖ K_rope)^T * softmax_scale（q/k_cache 最后维 576 全量参与 mm1）
P = softmax(S, dim=-1)（mask_mode=3 时右下对齐因果）
O = P · V_nope（k_cache[..., :512]）
LSE = log(sum_k exp(S - max)) + max，形状 (N, T)（flash_attn TND 族约定）

与 pytests 的 core/backends/cpu_golden.py 同源（本文件为 TTK assets 入口的薄封装）。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pytests"))
from core.backends.cpu_golden import mla_forward  # noqa: E402

import torch  # noqa: E402


def flash_mla_with_kvcache_golden(tensors, params):
    """从 TTK customize_inputs 的 tensors 字典 + 参数 dict 计算 golden。

    Args:
        tensors: {q, k_cache, block_table?, cu_seqlens_q?, cache_seqlens?, seqused_q?, attn_mask?, metadata?}
        params:  case 参数字典（含 layout/mask_mode/scale/block_size ...）

    Returns:
        torch.Tensor out = [N1, T, 512]（NTD，与 q 同 dtype）
    """
    layout_q = params.get("layout_q", "TND")
    layout_kv = params.get("layout_kv", "NZ")
    layout_out = params.get("layout_out", "NTD")
    assert layout_q == "TND" and layout_kv == "NZ" and layout_out == "NTD", \
        f"MLA golden 仅支持 TND/NZ/NTD（R1），got {layout_q}/{layout_kv}/{layout_out}"

    cu = tensors.get("cu_seqlens_q")
    cache_lens = tensors.get("cache_seqlens")
    bt = tensors.get("block_table")
    if cu is None or cache_lens is None or bt is None:
        raise ValueError("flash_mla_with_kvcache golden 需要 cu_seqlens_q / cache_seqlens / block_table")

    q = tensors["q"]
    k_cache = tensors["k_cache"]
    mask_mode = params.get("mask_mode", 0)
    d = params.get("D", 512)
    scale = params.get("scale", 1 / (d ** 0.5))

    out, lse = mla_forward(
        q, k_cache,
        cu_seqlens_q=[int(x) for x in cu.detach().cpu().reshape(-1).tolist()],
        cache_seqlens=[int(x) for x in cache_lens.detach().cpu().reshape(-1).tolist()],
        block_table=bt.detach().cpu(),
        block_size=params.get("block_size", 128),
        seqused_q=None if tensors.get("seqused_q") is None else
        [int(x) for x in tensors["seqused_q"].detach().cpu().reshape(-1).tolist()],
        mask_mode=mask_mode,
        scale=scale,
        N1=params.get("N1", q.shape[1]),
    )
    lse[lse.isinf() & (lse > 0)] = -float('inf')
    return out, lse