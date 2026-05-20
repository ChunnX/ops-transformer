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


class GetDispatchLayoutOpBuilder(OpBuilder):
    def __init__(self):
        super(GetDispatchLayoutOpBuilder, self).__init__("npu_get_dispatch_layout")

    def sources(self):
        return ['ops/csrc/get_dispatch_layout.cpp']

    def schema(self) -> str:
        return "npu_get_dispatch_layout(Tensor topkIdx, int numExperts, int numRanks) -> (Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_get_dispatch_layout_meta(topkIdx, numExperts, numRanks):
            torch._check(
                topkIdx.dim() == 2,
                lambda: (
                    f"topkIdx should be 2D tensor, got {topkIdx.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                numExperts > 0,
                lambda: (
                    f"numExperts should be greater than 0, got {numExperts}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            numTokensPerExpert = topkIdx.new_empty((numExperts,), dtype=torch.int32)
            sendTokenIdxSmall = topkIdx.new_empty((topkIdx.size(0), topkIdx.size(1)), dtype=torch.int32)
            return (numTokensPerExpert, sendTokenIdxSmall)


_get_dispatch_layout_op_builder = GetDispatchLayoutOpBuilder()
_op_module = _get_dispatch_layout_op_builder.load()


@impl(AS_LIBRARY, _get_dispatch_layout_op_builder.name, "PrivateUse1")
def _npu_get_dispatch_layout(topkIdx, numExperts, numRanks):
    return _op_module.npu_get_dispatch_layout(topkIdx, numExperts, numRanks)


def get_dispatch_layout(
    topkIdx: torch.Tensor,
    numExperts: int,
    numRanks: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dispatch layout for MoE prefill scenario.
    
    Args:
        topkIdx: Top-k expert indices [numTokens, numTopk] (int64)
        numExperts: Number of experts
        numRanks: Number of ranks
    
    Returns:
        Tuple of:
            numTokensPerExpert: Token count per expert [numExperts] (int32)
            sendTokenIdxSmall: Send token indices [numTokens, numTopk] (int32)
    """
    return _npu_get_dispatch_layout(topkIdx, numExperts, numRanks)