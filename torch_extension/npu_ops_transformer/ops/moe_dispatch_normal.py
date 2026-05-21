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


class MoeDispatchNormalOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDispatchNormalOpBuilder, self).__init__("npu_moe_dispatch_normal")

    def sources(self):
        return ['ops/csrc/moe_dispatch_normal.cpp']

    def schema(self) -> str:
        return "npu_moe_dispatch_normal(Tensor x, Tensor topkIdx, Tensor sendOffset, Tensor sendTokenIdx, " \
               "Tensor recvOffset, Tensor recvCount, str groupEp, int epWorldSize, int epRankId, " \
               "str? groupTp=None, int tpWorldSize=0, int tpRankId=0, int moeExpertNum, int quantMode=0, " \
               "int globalBs=0, *, Tensor? recvX=None, Tensor? recvXScales=None, " \
               "Tensor? assistInfoForCombine=None, Tensor? waitRecvCostStats=None) -> " \
               "(Tensor, Tensor, Tensor, Tensor?)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_dispatch_normal_meta(
            x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount,
            groupEp, epWorldSize, epRankId, groupTp=None, tpWorldSize=0, tpRankId=0,
            moeExpertNum, quantMode=0, globalBs=0,
            recvX=None, recvXScales=None, assistInfoForCombine=None, waitRecvCostStats=None
        ):
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
                sendOffset.dim() == 1,
                lambda: (
                    f"sendOffset should be 1D tensor, got {sendOffset.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                sendTokenIdx.dim() == 1,
                lambda: (
                    f"sendTokenIdx should be 1D tensor, got {sendTokenIdx.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                recvOffset.dim() == 1,
                lambda: (
                    f"recvOffset should be 1D tensor, got {recvOffset.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                recvCount.dim() == 1,
                lambda: (
                    f"recvCount should be 1D tensor, got {recvCount.dim()}D, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            
            bs = x.size(0)
            h = x.size(1)
            k = topkIdx.size(1)
            total_recv_tokens = recvOffset.size(0)
            
            if recvX is not None:
                recv_x = recvX
            else:
                recv_x = x.new_empty(tuple([total_recv_tokens, h]), dtype=x.dtype)
            
            if recvXScales is not None:
                recv_x_scales = recvXScales
            else:
                recv_x_scales = x.new_empty(tuple([total_recv_tokens]), dtype=torch.float32)
            
            if assistInfoForCombine is not None:
                assist_info = assistInfoForCombine
            else:
                assist_info = x.new_empty(tuple([total_recv_tokens, 3]), dtype=torch.int32)
            
            cost_stats = waitRecvCostStats
            
            return (recv_x, recv_x_scales, assist_info, cost_stats)


_moe_dispatch_normal_op_builder = MoeDispatchNormalOpBuilder()
_op_module = _moe_dispatch_normal_op_builder.load()


@impl(AS_LIBRARY, _moe_dispatch_normal_op_builder.name, "PrivateUse1")
def _npu_moe_dispatch_normal(
    x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount,
    groupEp, epWorldSize, epRankId, groupTp=None, tpWorldSize=0, tpRankId=0,
    moeExpertNum, quantMode=0, globalBs=0,
    recvX=None, recvXScales=None, assistInfoForCombine=None, waitRecvCostStats=None
):
    return _op_module.npu_moe_dispatch_normal(
        x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount,
        groupEp, epWorldSize, epRankId, groupTp, tpWorldSize, tpRankId,
        moeExpertNum, quantMode, globalBs,
        recvX, recvXScales, assistInfoForCombine, waitRecvCostStats)


def moe_dispatch_normal(
    x: torch.Tensor,
    topkIdx: torch.Tensor,
    sendOffset: torch.Tensor,
    sendTokenIdx: torch.Tensor,
    recvOffset: torch.Tensor,
    recvCount: torch.Tensor,
    groupEp: str,
    epWorldSize: int,
    epRankId: int,
    moeExpertNum: int,
    groupTp: Optional[str] = None,
    tpWorldSize: int = 0,
    tpRankId: int = 0,
    quantMode: int = 0,
    globalBs: int = 0,
    recvX: Optional[torch.Tensor] = None,
    recvXScales: Optional[torch.Tensor] = None,
    assistInfoForCombine: Optional[torch.Tensor] = None,
    waitRecvCostStats: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    MoE dispatch normal operation for distributed expert dispatch.
    
    Args:
        x: Input token features [bs, h]
        topkIdx: Top-k expert indices [bs, k]
        sendOffset: Send offset for each rank [epWorldSize]
        sendTokenIdx: Send token indices
        recvOffset: Receive offset for each rank [epWorldSize]
        recvCount: Receive count for each rank [epWorldSize]
        groupEp: EP communication group name
        epWorldSize: EP world size
        epRankId: EP rank ID
        moeExpertNum: Total number of experts
        groupTp: TP communication group name (optional)
        tpWorldSize: TP world size (optional)
        tpRankId: TP rank ID (optional)
        quantMode: Quantization mode (optional)
        globalBs: Global batch size (optional)
        recvX: Output tensor for received features (optional)
        recvXScales: Output tensor for received scales (optional)
        assistInfoForCombine: Output tensor for assist info (optional)
        waitRecvCostStats: Output tensor for cost statistics (optional)
    
    Returns:
        Tuple of:
            recvX: Received features [total_recv_tokens, h]
            recvXScales: Received scales [total_recv_tokens]
            assistInfoForCombine: Assist info for combine [total_recv_tokens, 3]
            waitRecvCostStats: Cost statistics (optional)
    """
    return _npu_moe_dispatch_normal(
        x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount,
        groupEp, epWorldSize, epRankId, groupTp, tpWorldSize, tpRankId,
        moeExpertNum, quantMode, globalBs,
        recvX, recvXScales, assistInfoForCombine, waitRecvCostStats)