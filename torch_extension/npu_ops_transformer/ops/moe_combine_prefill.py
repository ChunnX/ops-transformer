# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeCombinePrefillOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeCombinePrefillOpBuilder, self).__init__("npu_moe_combine_prefill")

    def sources(self):
        return ['ops/csrc/moe_combine_prefill.cpp']

    def schema(self) -> str:
        return ("npu_moe_combine_prefill(Tensor x, Tensor topkIdx, Tensor topkWeights, "
                "Tensor srcIdx, Tensor sendHead, str groupEp, int rank, int numRanks) -> Tensor")

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_combine_prefill_meta(x, topkIdx, topkWeights, srcIdx, sendHead,
                                        groupEp, rank, numRanks):
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
            numTokens = topkIdx.size(0)
            hidden = x.size(1)
            combinedX = x.new_empty((numTokens, hidden), dtype=x.dtype)
            return combinedX


_moe_combine_prefill_op_builder = MoeCombinePrefillOpBuilder()
_op_module = _moe_combine_prefill_op_builder.load()


@impl(AS_LIBRARY, _moe_combine_prefill_op_builder.name, "PrivateUse1")
def _npu_moe_combine_prefill(x, topkIdx, topkWeights, srcIdx, sendHead, groupEp, rank, numRanks):
    return _op_module.npu_moe_combine_prefill(x, topkIdx, topkWeights, srcIdx, sendHead, groupEp, rank, numRanks)


def moe_combine_prefill(
    x: torch.Tensor,
    topkIdx: torch.Tensor,
    topkWeights: torch.Tensor,
    srcIdx: torch.Tensor,
    sendHead: torch.Tensor,
    groupEp: str,
    rank: int,
    numRanks: int
) -> torch.Tensor:
    """
    MoE combine prefill: wraps moe_combine_normal for prefill scenario.
    
    Args:
        x: Input tensor [numRecvTokens, hidden] (float16/bfloat16)
        topkIdx: Top-k expert indices [numTokens, numTopk] (int64/int32)
        topkWeights: Top-k weights [numTokens, numTopk] (float16/float32)
        srcIdx: Source indices from dispatch [numRecvTokens * 3] (int32)
        sendHead: Send head counts [numExperts] (int32)
        groupEp: Expert parallel group name
        rank: Current rank ID
        numRanks: Number of ranks
    
    Returns:
        combinedX: Combined tensor [numTokens, hidden] (same dtype as input)
    """
    return _npu_moe_combine_prefill(x, topkIdx, topkWeights, srcIdx, sendHead, groupEp, rank, numRanks)