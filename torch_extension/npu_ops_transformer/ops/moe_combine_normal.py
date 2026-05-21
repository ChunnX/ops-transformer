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
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeCombineNormalOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeCombineNormalOpBuilder, self).__init__("npu_moe_combine_normal")

    def sources(self):
        return ['ops/csrc/moe_combine_normal.cpp']

    def schema(self) -> str:
        return "npu_moe_combine_normal(Tensor recvX, Tensor tokenSrcInfo, Tensor epRecvCount, " \
            "Tensor topkWeights, *, Tensor? XOut=None, Tensor? sendCostStatsOut=None) -> (Tensor, Tensor?)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_combine_normal_meta(recvX, tokenSrcInfo, epRecvCount, topkWeights,
                                        XOut=None, sendCostStatsOut=None):
            torch._check(
                recvX.dim() == 2,
                lambda: (
                    f"recvX should be 2D tensor, got {recvX.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                tokenSrcInfo.dim() == 2,
                lambda: (
                    f"tokenSrcInfo should be 2D tensor, got {tokenSrcInfo.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                epRecvCount.dim() == 1,
                lambda: (
                    f"epRecvCount should be 1D tensor, got {epRecvCount.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                topkWeights.dim() == 2,
                lambda: (
                    f"topkWeights should be 2D tensor, got {topkWeights.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            bs = topkWeights.size(0)
            h = recvX.size(1)
            if XOut is not None:
                x_out = XOut
            else:
                x_out = recvX.new_empty(tuple([bs, h]), dtype=recvX.dtype)
            cost_stats = sendCostStatsOut
            return (x_out, cost_stats)


_moe_combine_normal_op_builder = MoeCombineNormalOpBuilder()
_op_module = _moe_combine_normal_op_builder.load()


@impl(AS_LIBRARY, _moe_combine_normal_op_builder.name, "PrivateUse1")
def _npu_moe_combine_normal(recvX, tokenSrcInfo, epRecvCount, topkWeights,
                            XOut=None, sendCostStatsOut=None):
    return _op_module.npu_moe_combine_normal(
        recvX, tokenSrcInfo, epRecvCount, topkWeights, XOut, sendCostStatsOut)


class MoeCombineBuffer:
    def __init__(
        self,
        num_tokens: int,
        hidden: int,
        num_experts: int,
        topk: int,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.num_experts = num_experts
        self.topk = topk
        self.dtype = dtype
        self.recvX = torch.empty(num_tokens * topk, hidden, dtype=dtype).npu()
        self.tokenSrcInfo = torch.empty(num_tokens * topk, 3, dtype=torch.int32).npu()
        self.epRecvCount = torch.empty(num_experts, dtype=torch.int32).npu()
        self.topkWeights = torch.empty(num_tokens, topk, dtype=torch.float32).npu()
        self.XOut = torch.empty(num_tokens, hidden, dtype=dtype).npu()

    def reset(self):
        self.recvX.zero_()
        self.tokenSrcInfo.zero_()
        self.epRecvCount.zero_()
        self.topkWeights.zero_()


def get_moe_combine_buffer(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16
) -> MoeCombineBuffer:
    return MoeCombineBuffer(num_tokens, hidden, num_experts, topk, dtype)


def moe_combine_normal(
    recvX: torch.Tensor,
    tokenSrcInfo: torch.Tensor,
    epRecvCount: torch.Tensor,
    topkWeights: torch.Tensor,
    XOut: Optional[torch.Tensor] = None,
    sendCostStatsOut: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return _npu_moe_combine_normal(
        recvX, tokenSrcInfo, epRecvCount, topkWeights, XOut, sendCostStatsOut)