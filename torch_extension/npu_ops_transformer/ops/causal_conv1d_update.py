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
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


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
            "Tensor x, Tensor weight, Tensor? bias, "
            "Tensor conv_states, "
            "Tensor? conv_state_indices, Tensor? query_start_loc, "
            "Tensor? initial_state_idx, "
            "Tensor? num_accepted_tokens, "
            "Tensor? block_idx_last_scheduled_token, "
            "int activation_mode, int null_block_id, "
            "int max_query_len"
            ") -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_causal_conv1d_update_meta(
            x, weight, bias, conv_states,
            conv_state_indices=None, query_start_loc=None,
            initial_state_idx=None,
            num_accepted_tokens=None,
            block_idx_last_scheduled_token=None,
            activation_mode=1, null_block_id=-1,
            max_query_len=-1,
        ):
            y = torch.empty_like(x)
            return y


# Instantiate the builder
_causal_conv1d_update_op_builder = CausalConv1dUpdateOpBuilder()
_op_module = _causal_conv1d_update_op_builder.load()


@impl(AS_LIBRARY, _causal_conv1d_update_op_builder.name, "PrivateUse1")
def npu_causal_conv1d_update(
    x, weight, bias, conv_states,
    conv_state_indices=None, query_start_loc=None,
    initial_state_idx=None,
    num_accepted_tokens=None,
    block_idx_last_scheduled_token=None,
    activation=None, null_block_id=-1,
    max_query_len=-1,
):
    """
    Decode / update mode.

    Args:
        x: input tensor [B, D] or [T, D]
        weight: convolution weight [kW, D]
        bias: optional bias [D]
        conv_states: convolution hidden states [numSlots, kW-1, D] (in/out)
        conv_state_indices: optional token→conv_state slot mapping   (NEW, not yet wired)
        query_start_loc: optional query start locations for varlen
        initial_state_idx: optional index of sequences with initial state  (NEW, not yet wired)
        num_accepted_tokens: optional accepted token count (speculative decode)
        block_idx_last_scheduled_token: optional                        (NEW, not yet wired)
        activation: "silu" or None
        null_block_id: null block id                                    (NEW, not yet wired)
        max_query_len: max query length, -1=unlimited                   (NEW, not yet wired)

    Returns:
        y: output and updated hidden states
    """
    activation_mode = 1 if activation == "silu" else 0
    return _op_module.npu_causal_conv1d_update(
        x, weight, bias, conv_states,
        conv_state_indices, query_start_loc,
        initial_state_idx,
        num_accepted_tokens,
        block_idx_last_scheduled_token,
        activation_mode, null_block_id,
        max_query_len,
    )


# Public API — mirrors causal_conv1d_update from vLLM/SGLang
causal_conv1d_update = npu_causal_conv1d_update
