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


class CausalConv1dFnOpBuilder(OpBuilder):
    def __init__(self):
        super(CausalConv1dFnOpBuilder, self).__init__("npu_causal_conv1d_fn")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/causal_conv1d_fn.cpp']

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "npu_causal_conv1d_fn("
            "Tensor x, Tensor weight, Tensor? bias, "
            "Tensor conv_states, "
            "Tensor query_start_loc, "
            "Tensor? cache_indices, "
            "Tensor? has_initial_state, "
            "str? activation, "
            "int pad_slot_id, "
            "int null_block_id, "
            "Tensor? block_idx_first_scheduled_token, "
            "Tensor? block_idx_last_scheduled_token, "
            "Tensor? initial_state_idx, "
            "Tensor? num_computed_tokens, "
            "int block_size_to_align"
            ") -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_causal_conv1d_fn_meta(
            x, weight, bias, conv_states, query_start_loc,
            cache_indices=None, has_initial_state=None,
            activation="silu", pad_slot_id=-1, null_block_id=0,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None, num_computed_tokens=None,
            block_size_to_align=0,
        ):
            y = torch.empty_like(x)
            return y


# Instantiate the builder
_causal_conv1d_fn_op_builder = CausalConv1dFnOpBuilder()
_op_module = _causal_conv1d_fn_op_builder.load()


@impl(AS_LIBRARY, _causal_conv1d_fn_op_builder.name, "PrivateUse1")
def npu_causal_conv1d_fn(
    x, weight, bias, conv_states, query_start_loc,
    cache_indices=None, has_initial_state=None,
    activation="silu", pad_slot_id=-1, null_block_id=0,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None, num_computed_tokens=None,
    block_size_to_align=0,
):
    """
    Prefill / chunk-prefill mode.

    Args:
        x: input tensor [B, L, D] or [T, D]
        weight: convolution weight [kW, D]
        bias: optional bias [D]
        conv_states: convolution hidden states [B, kW-1, D] (in/out)
        query_start_loc: query start locations for varlen
        cache_indices: optional cache line index per batch
        has_initial_state: optional flag per sequence
        activation: "silu" or None
        pad_slot_id: padding slot id
        null_block_id: null block id
        block_idx_first_scheduled_token: optional
        block_idx_last_scheduled_token: optional
        initial_state_idx: optional index of sequences with initial state
        num_computed_tokens: optional computed token count
        block_size_to_align: block alignment hint

    Returns:
        y: output tensor
    """
    return _op_module.npu_causal_conv1d_fn(
        x, weight, bias, conv_states,
        query_start_loc, cache_indices,
        has_initial_state,
        activation, null_block_id,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        block_size_to_align,
    )


# Public API — mirrors causal_conv1d_fn from vLLM/SGLang
causal_conv1d_fn = npu_causal_conv1d_fn
