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
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class _DispatchLayoutOpBuilder(OpBuilder):
    def __init__(self):
        super(_DispatchLayoutOpBuilder, self).__init__("npu_dispatch_layout")

    def sources(self):
        return ['ops/csrc/dispatch_layout.cpp']

    def schema(self) -> str:
        return "npu_dispatch_layout(Tensor topkIdx, int num_tokens, int num_ranks, int num_experts, " \
               "int num_topk, int local_ranksize) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_dispatch_layout_meta(topkIdx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize):
            torch._check(
                num_ranks > 0,
                lambda: (
                    f"num_ranks should be greater than 0, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                num_experts > 0,
                lambda: (
                    f"num_experts should be greater than 0, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            numTokensPerRank = topkIdx.new_empty((num_ranks,), dtype=torch.int32)
            numTokensPerExpert = topkIdx.new_empty((num_experts,), dtype=torch.int32)
            isTokenInRank = topkIdx.new_empty((num_tokens, num_ranks), dtype=torch.int32)
            notifySendData = topkIdx.new_empty((num_tokens, num_experts), dtype=torch.int32)
            sendTokenIdxSmall = topkIdx.new_empty((num_tokens, num_topk), dtype=torch.int32)
            return (numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData, sendTokenIdxSmall)


_dispatch_layout_op_builder = _DispatchLayoutOpBuilder()
_op_module = _dispatch_layout_op_builder.load()


@impl(AS_LIBRARY, _dispatch_layout_op_builder.name, "PrivateUse1")
def _npu_dispatch_layout(topkIdx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize):
    return _op_module.npu_dispatch_layout(
        topkIdx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize)


def dispatch_layout(
    topkIdx: torch.Tensor,
    num_tokens: int,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    local_ranksize: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute dispatch layout information for MoE distributed dispatch.
    
    Args:
        topkIdx: Top-k expert indices tensor [num_tokens, num_topk] (int64)
        num_tokens: Number of tokens
        num_ranks: Number of ranks in distributed group
        num_experts: Total number of experts
        num_topk: Number of top-k experts selected per token
        local_ranksize: Local rank size for multi-machine scenario
    
    Returns:
        Tuple of:
            numTokensPerRank: Token count per rank [num_ranks] (int32)
            numTokensPerExpert: Token count per expert [num_experts] (int32)
            isTokenInRank: Token-to-rank mapping [num_tokens, num_ranks] (int32)
            notifySendData: Temporary buffer [num_tokens, num_experts] (int32)
            sendTokenIdxSmall: Token indices for sending [num_tokens, num_topk] (int32)
    """
    return _npu_dispatch_layout(
        topkIdx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize)