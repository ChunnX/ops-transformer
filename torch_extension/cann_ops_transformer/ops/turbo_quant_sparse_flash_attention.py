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

from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

INT64_MAX = 9223372036854775807


class TurboQuantSparseFlashAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super(TurboQuantSparseFlashAttentionOpBuilder, self).__init__(
            "turbo_quant_sparse_flash_attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/turbo_quant_sparse_flash_attention.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return (
            "turbo_quant_sparse_flash_attention(Tensor query, Tensor key, Tensor value, "
            "Tensor sparse_indices, *, "
            "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, "
            "Tensor block_table, Tensor actual_seq_lengths_query, "
            "Tensor actual_seq_lengths_kv, "
            "float scale_value=1.0, int key_quant_mode=3, int value_quant_mode=3, "
            "int sparse_block_size=1, "
            'str layout_query="TND", str layout_kv="PA_BSND", '
            "int sparse_mode=3, int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, int attention_mode=2, "
            "int quant_scale_repo_mode=1, int tile_size=128, int rope_head_dim=64, "
            "bool return_softmax_lse=False) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(AS_LIBRARY, self.name, "Meta")
        def turbo_quant_sparse_flash_attention_meta(
            query,
            key,
            value,
            sparse_indices,
            *,
            key_dequant_scale=None,
            value_dequant_scale=None,
            block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            scale_value=1.0,
            key_quant_mode=3,
            value_quant_mode=3,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=INT64_MAX,
            next_tokens=INT64_MAX,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
            return_softmax_lse=False,
        ):
            # Meta 与 runtime 必须使用同一套入参校验：缺校验时非法的 rope_head_dim
            # 会算出错误甚至为负的输出尾维，仅由 torch.empty 兜住，报错也与 runtime 不同。
            torch._check(
                query.dim() == 3,
                lambda: "query dim num[%d] should be 3 (TND)" % query.dim(),
            )
            torch._check(
                0 <= rope_head_dim < query.size(2),
                lambda: "rope_head_dim[%d] should be in [0, query last dim)"
                % rope_head_dim,
            )
            attention_out = torch.empty(
                (query.size(0), query.size(1), query.size(2) - rope_head_dim),
                dtype=query.dtype,
                device="meta",
            )
            torch._check(
                layout_kv == "PA_BSND",
                lambda: "layout_kv[%s] is not supported, only PA_BSND" % layout_kv,
            )
            if return_softmax_lse:
                kv_head_dim = key.size(2)
                softmax_size = (
                    kv_head_dim,
                    query.size(0),
                    query.size(1) // kv_head_dim,
                )
            else:
                softmax_size = (0,)
            softmax_max = torch.empty(softmax_size, dtype=torch.float32, device="meta")
            softmax_sum = torch.empty(softmax_size, dtype=torch.float32, device="meta")
            return attention_out, softmax_max, softmax_sum


turbo_quant_sparse_flash_attention_op_builder = (
    TurboQuantSparseFlashAttentionOpBuilder()
)


@impl(AS_LIBRARY, turbo_quant_sparse_flash_attention_op_builder.name, "PrivateUse1")
def turbo_quant_sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    *,
    key_dequant_scale: Optional[torch.Tensor] = None,
    value_dequant_scale: Optional[torch.Tensor] = None,
    # 当前唯一支持的 TND + PA_BSND 路径下三者均为必需，故不提供默认值
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    scale_value: float = 1.0,
    key_quant_mode: int = 3,  # 3: TQ4 码本量化，host 侧唯一合法值
    value_quant_mode: int = 3,
    sparse_block_size: int = 1,
    layout_query: str = "TND",  # 仅支持 TND：launcher 强制 query 为 3D
    layout_kv: str = "PA_BSND",  # 当前仅支持 PA_BSND
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
    attention_mode: int = 2,  # 2: MLA-absorb，host 侧唯一合法值
    quant_scale_repo_mode: int = 1,
    tile_size: int = 128,
    rope_head_dim: int = 64,
    return_softmax_lse: bool = False,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = turbo_quant_sparse_flash_attention_op_builder.load()
    return op_module.turbo_quant_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        return_softmax_lse,
    )
