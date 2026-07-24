# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class ChunkGatedDeltaRuleOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("chunk_gated_delta_rule")

    def sources(self):
        """C++ 源码路径（相对包根）。"""
        return ["ops/csrc/chunk_gated_delta_rule.cpp"]

    def schema(self) -> str:
        """torch op schema: 与 aclnn/aclnnChunkGatedDeltaRule 入参一致, 返回 (out, final_state)。

        query/key/value/beta: bf16; initial_state: bf16 或 fp32; actual_seq_lengths: int32;
        g: 可选 fp32; scale: float(IR attr scale_value)。harness 以 kwargs 调用, 形参名须对应。
        """
        return (
            "chunk_gated_delta_rule("
            "Tensor query, Tensor key, Tensor value, "
            "Tensor beta, Tensor initial_state, Tensor actual_seq_lengths, "
            "Tensor? g=None, float scale=1.0"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def chunk_gated_delta_rule_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            actual_seq_lengths: torch.Tensor,
            g: torch.Tensor = None,
            scale: float = 1.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # out: bf16, shape 同 value (T, Nv, Dv)
            out = torch.empty(value.size(), dtype=torch.bfloat16, device="meta")
            # final_state: dtype 跟随 initial_state(bf16/fp32), shape 同 initial_state (B, Nv, Dv, Dk)
            final_state = torch.empty(
                initial_state.size(), dtype=initial_state.dtype, device="meta"
            )
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
    g: torch.Tensor = None,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    op_module = _chunk_gated_delta_rule_op_builder.load()
    return op_module.chunk_gated_delta_rule(
        query, key, value, beta, initial_state, actual_seq_lengths, g, scale
    )


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    g: torch.Tensor = None,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """封装 aclnnChunkGatedDeltaRule。

    q/k/v/beta 为 bfloat16; initial_state 与 final_state 同 dtype(bf16 或 fp32);
    actual_seq_lengths 为 int32; g 可选 float32; scale 为 float(对应 IR attr scale_value)。
    """
    return _chunk_gated_delta_rule(
        query, key, value, beta, initial_state, actual_seq_lengths, g, scale
    )
