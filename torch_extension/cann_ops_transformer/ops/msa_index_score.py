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

SCORE_STRIDE_ALIGN = 16
DEFAULT_BLOCK_SIZE = 128


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


class MsaIndexScoreOpBuilder(OpBuilder):
    def __init__(self):
        super(MsaIndexScoreOpBuilder, self).__init__("msa_index_score")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/msa_index_score.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "msa_index_score(Tensor query, Tensor key_cache, Tensor block_table, "
            "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, Tensor start_loc, *, "
            "Tensor? scale=None, int block_size=128, bool fill_invalid_tail=True) -> Tensor"
        )

    def register_meta(self):
        """Registers the Meta implementation (Shape/Dtype inference)."""

        @impl(AS_LIBRARY, self.name, "Meta")
        def msa_index_score_meta(
            query: torch.Tensor,
            key_cache: torch.Tensor,
            block_table: torch.Tensor,
            actual_seq_lengths_query: torch.Tensor,
            actual_seq_lengths_key: torch.Tensor,
            start_loc: torch.Tensor,
            *,
            scale: Optional[torch.Tensor] = None,
            block_size: int = DEFAULT_BLOCK_SIZE,
            fill_invalid_tail: bool = True,
        ):
            torch._check(
                query.dim() == 3,
                lambda: f"query must be 3D [T, Hq, D], but got dim={query.dim()}",
            )
            torch._check(
                block_table.dim() == 2,
                lambda: f"block_table must be 2D [B, MB], but got dim={block_table.dim()}",
            )
            total_q = query.size(0)
            num_q_heads = query.size(1)
            max_blocks = block_table.size(1)
            score_stride = _round_up(int(max_blocks), SCORE_STRIDE_ALIGN)
            return torch.empty(
                (num_q_heads, total_q, score_stride),
                dtype=torch.float32,
                device="meta",
            )


_msa_index_score_op_builder = MsaIndexScoreOpBuilder()
_op_module = None


def _get_op_module():
    global _op_module
    if _op_module is None:
        _op_module = _msa_index_score_op_builder.load()
    return _op_module


@impl(AS_LIBRARY, _msa_index_score_op_builder.name, "PrivateUse1")
def _msa_index_score(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    start_loc: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    fill_invalid_tail: bool = True,
) -> torch.Tensor:
    """Dispatcher implementation for NPU (PrivateUse1)."""
    return _get_op_module().msa_index_score(
        query,
        key_cache,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        start_loc,
        scale,
        block_size,
        fill_invalid_tail,
    )


def msa_index_score(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    start_loc: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    fill_invalid_tail: bool = True,
) -> torch.Tensor:
    """MSA Lightning Indexer 的 block score 计算，封装 aclnnMsaIndexScore。

    Prefill / Decode 归一到同一接口。非量化路径不得传入 ``scale``；
    int8 前融合路径 ``key_cache`` 为 int8，且必须传入
    ``scale``，shape 为 ``[num_pages, num_kv_heads, block_size]``。

    Args:
        query (Tensor): index query，shape ``[T, Hq, D]``，dtype float16 / bfloat16。
        key_cache (Tensor): 分页 index key cache，shape ``[NP, P, Hkv, D]``。
            非量化与 query 同 dtype；量化为 int8（当前要求 query 为 float16）。
        block_table (Tensor): 逻辑 block → 物理 page，shape ``[B, MB]``，dtype int32。
        actual_seq_lengths_query (Tensor): query 长度前缀和，shape ``[B+1]``，dtype int32。
        actual_seq_lengths_key (Tensor): 每请求可见 kv 长度，shape ``[B]``，dtype int32。
        start_loc (Tensor): 首个 query token 在自身 kv 序列中的位置，shape ``[B]``，dtype int32。
        scale (Tensor, optional): int8 反量化系数 ``[NP, N_kv, P]`` float32；非量化必须为 None。
        block_size (int): sparse block / page token 数，当前仅支持 128。
        fill_invalid_tail (bool): 不可见 block 是否写 ``-inf``，默认 True。

    Returns:
        Tensor: block score，shape ``[Hq, T, RoundUp(MB, 16)]``，dtype float32。

    Example:
        >>> score = cann_ops_transformer.msa_index_score(
        ...     query, key_cache, block_table, cu_seq_q, kv_len, start_loc,
        ...     block_size=128, fill_invalid_tail=True)
    """
    return torch.ops.cann_ops_transformer.msa_index_score(
        query,
        key_cache,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        start_loc,
        scale=scale,
        block_size=block_size,
        fill_invalid_tail=fill_invalid_tail,
    )
