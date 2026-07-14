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
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class ChunkGatedDeltaRuleOpBuilder(OpBuilder):
    def __init__(self):
        super(ChunkGatedDeltaRuleOpBuilder, self).__init__("chunk_gated_delta_rule")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/chunk_gated_delta_rule.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "chunk_gated_delta_rule("
            "Tensor query, Tensor key, Tensor value, Tensor beta, "
            "Tensor initial_state, Tensor actual_seq_lengths, "
            "*, Tensor? g=None, float scale=1.0"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
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
            out = torch.empty(value.shape[:3], dtype=value.dtype, device='meta')
            final_state = torch.empty(initial_state.shape, dtype=initial_state.dtype, device='meta')
            return out, final_state


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
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    op_module = _chunk_gated_delta_rule_op_builder.load()
    return op_module.chunk_gated_delta_rule(query, key, value, beta, initial_state, actual_seq_lengths, g, scale)


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
    """Chunk 门控 Delta Rule 前向计算，封装 aclnnChunkGatedDeltaRule。

    基于 chunk-wise 策略计算门控 DeltaNet 线性注意力。对 query/key 做 L2 归一化后按
    chunk 切分，在 chunk 内通过 Delta Rule（带 beta 门控的递归状态更新）累积历史状态，
    chunk 间通过 gamma（g）门控做指数衰减传递状态；输出注意力结果 out 及最终递归状态
    final_state。

    Args:
        query (Tensor): query，shape 为 [T, Nk, Dk]，dtype 为 bfloat16。
        key (Tensor): key，shape 为 [T, Nk, Dk]，dtype 为 bfloat16。
        value (Tensor): value，shape 为 [T, Nv, Dv]，dtype 为 bfloat16。
        beta (Tensor): beta 门控，shape 为 [T, Nv]，dtype 为 bfloat16。
        initial_state (Tensor): 初始递归状态，shape 为 [B, Nv, Dv, Dk]，dtype 为 bfloat16。
        actual_seq_lengths (Tensor): 每个 batch 的实际序列长度，shape 为 [B]，dtype 为 int32。
        g (Tensor, optional): gamma 门控，shape 为 [T, Nv]，dtype 为 float32，None 表示不使用门控衰减。默认 None。
        scale (float): 缩放系数，通常为 1/sqrt(Dk)。默认值为 1.0。

    Returns:
        Tuple[Tensor, Tensor]:
            - out: 注意力输出，shape 为 [T, Nv, Dv]，dtype 为 bfloat16。
            - final_state: 最终递归状态，shape 与 initial_state 一致 [B, Nv, Dv, Dk]，dtype 为 bfloat16。
    """
    return _chunk_gated_delta_rule(
        query, key, value, beta, initial_state, actual_seq_lengths,
        g=g, scale=scale,
    )
