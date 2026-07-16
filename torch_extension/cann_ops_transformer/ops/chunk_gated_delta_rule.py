# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class ChunkGatedDeltaRuleOpBuilder(OpBuilder):
    def __init__(self):
        super(ChunkGatedDeltaRuleOpBuilder, self).__init__("chunk_gated_delta_rule")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/chunk_gated_delta_rule.cpp"]

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "chunk_gated_delta_rule("
            "Tensor query, Tensor key, Tensor value, Tensor beta, "
            "Tensor initial_state, Tensor actual_seq_lengths, "
            "*, "
            "Tensor? g=None, "
            "float scale=1.0"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        """Register Meta implementation for shape/dtype inference."""

        @impl(AS_LIBRARY, self.name, "Meta")
        def chunk_gated_delta_rule_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            actual_seq_lengths: torch.Tensor,
            *,
            g: Optional[torch.Tensor] = None,
            scale: float = 1.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            out = torch.empty_like(value)
            final_state = torch.empty_like(initial_state)
            return (out, final_state)


_chunk_gated_delta_rule_op_builder = ChunkGatedDeltaRuleOpBuilder()


@impl(AS_LIBRARY, _chunk_gated_delta_rule_op_builder.name, "PrivateUse1")
def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    *,
    g: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _op_module = _chunk_gated_delta_rule_op_builder.load()
    return _op_module.chunk_gated_delta_rule(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        g,
        scale,
    )


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    *,
    g: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunk Gated Delta Rule 前向计算，封装 aclnnChunkGatedDeltaRule。

    基于 chunk-wise 算法计算门控增量规则（Gated Delta Rule），支持变长序列。
    对 query/key 做 L2 归一化后，按 chunk 分块计算注意力输出与递归状态更新，
    可选的门控因子 g 控制状态衰减。

    Args:
        query (Tensor): 查询张量，shape 为 [T, Nk, Dk]，dtype 支持 bfloat16。
        key (Tensor): 键张量，shape 为 [T, Nk, Dk]，dtype 与 query 一致。
        value (Tensor): 值张量，shape 为 [T, Nv, Dv]，dtype 与 query 一致。
        beta (Tensor): beta 因子，shape 为 [T, Nv]，dtype 与 query 一致。
        initial_state (Tensor): 初始递归状态，shape 为 [B, Nv, Dv, Dk]，dtype 与 query 一致。
        actual_seq_lengths (Tensor): 各 batch 的实际序列长度，shape 为 [B]，dtype 为 int32。
        g (Tensor, optional): 门控因子，shape 为 [T, Nv]，dtype 为 float32，None 表示不使用门控，默认 None。
        scale (float): 缩放系数，默认 1.0，通常取 1/sqrt(Dk)。

    Returns:
        Tuple[Tensor, Tensor]:
            out: 注意力输出，shape 为 [T, Nv, Dv]，dtype 与 value 一致。
            final_state: 最终递归状态，shape 为 [B, Nv, Dv, Dk]，dtype 与 initial_state 一致。
    """
    return _chunk_gated_delta_rule(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        g=g,
        scale=scale,
    )
