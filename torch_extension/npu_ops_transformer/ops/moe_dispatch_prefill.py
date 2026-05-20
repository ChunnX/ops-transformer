# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
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


class MoeDispatchPrefillOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDispatchPrefillOpBuilder, self).__init__("npu_moe_dispatch_prefill")

    def sources(self):
        return ['ops/csrc/moe_dispatch_prefill.cpp']

    def schema(self) -> str:
        return ("npu_moe_dispatch_prefill(Tensor x, Tensor topkIdx, Tensor topkWeights, "
                "Tensor numTokensPerExpert, Tensor sendTokenIdxSmall, str groupEp, int rank, "
                "int numRanks, bool useQuant) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_dispatch_prefill_meta(x, topkIdx, topkWeights, numTokensPerExpert,
                                          sendTokenIdxSmall, groupEp, rank, numRanks, useQuant):
            torch._check(
                x.dim() == 2,
                lambda: (
                    f"x should be 2D tensor, got {x.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                topkIdx.dim() == 2,
                lambda: (
                    f"topkIdx should be 2D tensor, got {topkIdx.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                numTokensPerExpert.dim() == 1,
                lambda: (
                    f"numTokensPerExpert should be 1D tensor, got {numTokensPerExpert.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            numTokens = x.size(0)
            hidden = x.size(1)
            numExperts = numTokensPerExpert.size(0)
            numRecvTokens = numTokens
            expandxOut = x.new_empty((numRecvTokens, hidden), dtype=torch.int8 if useQuant else x.dtype)
            dynamicScalesOut = x.new_empty((numRecvTokens,), dtype=torch.float)
            expandIdxOut = x.new_empty((numRecvTokens * 3,), dtype=torch.int32)
            recvCount = x.new_empty((numExperts,), dtype=torch.int32)
            recvTokensPerExpert = x.new_empty((numExperts // numRanks,), dtype=torch.int64)
            return (expandxOut, dynamicScalesOut, expandIdxOut, recvCount, recvTokensPerExpert)


_moe_dispatch_prefill_op_builder = MoeDispatchPrefillOpBuilder()
_op_module = _moe_dispatch_prefill_op_builder.load()


@impl(AS_LIBRARY, _moe_dispatch_prefill_op_builder.name, "PrivateUse1")
def _npu_moe_dispatch_prefill(x, topkIdx, topkWeights, numTokensPerExpert, sendTokenIdxSmall,
                              groupEp, rank, numRanks, useQuant):
    return _op_module.npu_moe_dispatch_prefill(x, topkIdx, topkWeights, numTokensPerExpert,
                                               sendTokenIdxSmall, groupEp, rank, numRanks, useQuant)


def moe_dispatch_prefill(
    x: torch.Tensor,
    topkIdx: torch.Tensor,
    topkWeights: torch.Tensor,
    numTokensPerExpert: torch.Tensor,
    sendTokenIdxSmall: torch.Tensor,
    groupEp: str,
    rank: int,
    numRanks: int,
    useQuant: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MoE dispatch prefill: combines notify_dispatch and moe_dispatch_normal.
    
    Args:
        x: Input tensor [numTokens, hidden] (float16/bfloat16)
        topkIdx: Top-k expert indices [numTokens, numTopk] (int64/int32)
        topkWeights: Top-k weights [numTokens, numTopk] (float16/float32)
        numTokensPerExpert: Token count per expert [numExperts] (int32)
        sendTokenIdxSmall: Send token indices [numTokens, numTopk] (int32)
        groupEp: Expert parallel group name
        rank: Current rank ID
        numRanks: Number of ranks
        useQuant: Whether to use quantization
    
    Returns:
        Tuple of:
            expandxOut: Expanded tokens [numRecvTokens, hidden] (int8 if useQuant else input dtype)
            dynamicScalesOut: Dynamic scales [numRecvTokens] (float32)
            expandIdxOut: Expanded indices [numRecvTokens * 3] (int32)
            recvCount: Receive counts [numExperts] (int32)
            recvTokensPerExpert: Tokens per expert [numLocalExperts] (int64)
    """
    return _npu_moe_dispatch_prefill(x, topkIdx, topkWeights, numTokensPerExpert, sendTokenIdxSmall,
                                     groupEp, rank, numRanks, useQuant)