# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class CausalConv1dUpdateOpBuilder(OpBuilder):
    def __init__(self):
        super(CausalConv1dUpdateOpBuilder, self).__init__("npu_causal_conv1d_update")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/causal_conv1d_update.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "npu_causal_conv1d_update("
            "Tensor x, Tensor conv_state, Tensor weight, "
            "Tensor? bias, "
            "str? activation, "
            "Tensor? conv_state_indices, "
            "Tensor? num_accepted_tokens, "
            "Tensor? query_start_loc, "
            "int max_query_len, "
            "int null_block_id, "
            "Tensor? block_idx_last_scheduled_token, "
            "Tensor? initial_state_idx"
            ") -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_causal_conv1d_update_meta(
            x, conv_state, weight,
            bias=None, activation=None,
            conv_state_indices=None,
            num_accepted_tokens=None,
            query_start_loc=None,
            max_query_len=-1, null_block_id=0,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
        ):
            y = torch.empty_like(x)
            return y


# Instantiate the builder
_causal_conv1d_update_op_builder = CausalConv1dUpdateOpBuilder()
_op_module = _causal_conv1d_update_op_builder.load()


@impl(AS_LIBRARY, _causal_conv1d_update_op_builder.name, "PrivateUse1")
def npu_causal_conv1d_update(
    x, conv_state, weight,
    bias=None, activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1, null_block_id=0,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
):
    """
    Decode / update mode.

    Args:
        x: input tensor [B, D] or [T, D]
        conv_state: convolution hidden states [numSlots, kW-1, D] (in/out)
        weight: convolution weight [kW, D]
        bias: optional bias [D]
        activation: "silu" or None
        conv_state_indices: optional token→conv_state slot mapping
        num_accepted_tokens: optional accepted token count (speculative decode)
        query_start_loc: optional query start locations for varlen
        max_query_len: max query length, -1=unlimited
        null_block_id: null block id
        block_idx_last_scheduled_token: optional
        initial_state_idx: optional index of sequences with initial state

    Returns:
        y: output and updated hidden states
    """
    return _op_module.npu_causal_conv1d_update(
        x, conv_state, weight, bias,
        activation if activation is not None else "none",
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len, null_block_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )


# Public API — mirrors causal_conv1d_update from vLLM/SGLang
causal_conv1d_update = npu_causal_conv1d_update
