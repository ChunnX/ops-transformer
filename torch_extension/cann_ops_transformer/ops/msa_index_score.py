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
DEFAULT_SPARSE_MODE = 3


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


class MsaIndexScoreOpBuilder(OpBuilder):
    def __init__(self):
        super(MsaIndexScoreOpBuilder, self).__init__("msa_index_score")

    def sources(self):
        return ["ops/csrc/msa_index_score.cpp"]

    def schema(self) -> str:
        return (
            "msa_index_score(Tensor query, Tensor key, Tensor? block_table, "
            "Tensor? scale, Tensor? atten_mask, Tensor? actual_seq_qlen, "
            "Tensor? actual_seq_klen, Tensor start_loc, *, int sparse_mode=3) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def msa_index_score_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            block_table: Optional[torch.Tensor],
            scale: Optional[torch.Tensor],
            atten_mask: Optional[torch.Tensor],
            actual_seq_qlen: Optional[torch.Tensor],
            actual_seq_klen: Optional[torch.Tensor],
            start_loc: torch.Tensor,
            *,
            sparse_mode: int = DEFAULT_SPARSE_MODE,
        ):
            torch._check(query.dim() == 3, lambda: "query must be 3D [T, Hq, D]")
            torch._check(block_table is not None and block_table.dim() == 2,
                         lambda: "block_table must be 2D [B, MB]")
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
    key: torch.Tensor,
    block_table: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    atten_mask: Optional[torch.Tensor],
    actual_seq_qlen: Optional[torch.Tensor],
    actual_seq_klen: Optional[torch.Tensor],
    start_loc: torch.Tensor,
    *,
    sparse_mode: int = DEFAULT_SPARSE_MODE,
) -> torch.Tensor:
    return _get_op_module().msa_index_score(
        query,
        key,
        block_table,
        scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        start_loc,
        sparse_mode,
    )


def msa_index_score(
    query: torch.Tensor,
    key: torch.Tensor,
    start_loc: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_klen: Optional[torch.Tensor] = None,
    sparse_mode: int = DEFAULT_SPARSE_MODE,
) -> torch.Tensor:
    """MSA Index Branch block score，封装 aclnnMsaIndexScore。

    Args:
        query: ``[T1, N1, D]`` float16 / bfloat16
        key: PA BBND ``[NP, P, N2, D]``；非量化同 query，量化为 int8
        start_loc: ``[B]``，当前 query 所在逻辑 block 索引（local_mask）
        block_table: ``[B, MB]``，PA 必选
        scale: int8 反量化 ``[NP, N2, P]``；非量化为 None
        atten_mask: sparse_mode=3 时 ``[2048, 2048]`` int8
        actual_seq_qlen / actual_seq_klen: TND/PA 长度元数据
        sparse_mode: 0 或 3

    Returns:
        ``[N1, T1, RoundUp(MB, 16)]`` float32
    """
    return torch.ops.cann_ops_transformer.msa_index_score(
        query,
        key,
        block_table,
        scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        start_loc,
        sparse_mode=sparse_mode,
    )
