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


class NotifyDispatchOpBuilder(OpBuilder):
    def __init__(self):
        super(NotifyDispatchOpBuilder, self).__init__("npu_notify_dispatch")

    def sources(self):
        return ['ops/csrc/notify_dispatch.cpp']

    def schema(self) -> str:
        return "npu_notify_dispatch(Tensor sendData, Tensor tokenPerExpertData, int sendCount, " \
               "int numTokens, str commGroup, int rankSize, int rankId, int localRankSize, " \
               "int localRankId) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_notify_dispatch_meta(
            sendData, tokenPerExpertData, sendCount, numTokens,
            commGroup, rankSize, rankId, localRankSize, localRankId
        ):
            torch._check(
                rankSize > 0,
                lambda: (
                    f"rankSize should be greater than 0, got {rankSize}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                localRankSize > 0,
                lambda: (
                    f"localRankSize should be greater than 0, got {localRankSize}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                sendCount >= 0,
                lambda: (
                    f"sendCount should be >= 0, got {sendCount}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                numTokens >= 0,
                lambda: (
                    f"numTokens should be >= 0, got {numTokens}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            
            sendDataOffset = sendData.new_empty(tuple([rankSize]), dtype=torch.int64)
            recvData = sendData.new_empty(tuple([sendCount]), dtype=sendData.dtype)
            totalRecvTokens = sendData.new_empty(tuple([1]), dtype=torch.int64)
            recvCount = sendData.new_empty(tuple([rankSize]), dtype=torch.int64)
            recvOffset = sendData.new_empty(tuple([rankSize]), dtype=torch.int64)
            maxBs = sendData.new_empty(tuple([1]), dtype=torch.int64)
            recvTokensPerExpert = sendData.new_empty(tuple([rankSize]), dtype=torch.int64)
            
            return (sendDataOffset, recvData, totalRecvTokens, recvCount, recvOffset, maxBs, recvTokensPerExpert)


_notify_dispatch_op_builder = NotifyDispatchOpBuilder()
_op_module = _notify_dispatch_op_builder.load()


@impl(AS_LIBRARY, _notify_dispatch_op_builder.name, "PrivateUse1")
def _npu_notify_dispatch(
    sendData, tokenPerExpertData, sendCount, numTokens,
    commGroup, rankSize, rankId, localRankSize, localRankId
):
    return _op_module.npu_notify_dispatch(
        sendData, tokenPerExpertData, sendCount, numTokens,
        commGroup, rankSize, rankId, localRankSize, localRankId)


def notify_dispatch(
    sendData: torch.Tensor,
    tokenPerExpertData: torch.Tensor,
    sendCount: int,
    numTokens: int,
    commGroup: str,
    rankSize: int,
    rankId: int,
    localRankSize: int,
    localRankId: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Notify dispatch operation for all-to-all communication.
    
    Args:
        sendData: Data to send [sendCount]
        tokenPerExpertData: Token per expert data
        sendCount: Number of tokens to send
        numTokens: Total number of tokens
        commGroup: Communication group name
        rankSize: Total number of ranks
        rankId: Current rank ID
        localRankSize: Local rank size
        localRankId: Local rank ID
    
    Returns:
        Tuple of:
            sendDataOffset: Send data offset [rankSize] (int64)
            recvData: Received data [sendCount]
            totalRecvTokens: Total received tokens [1] (int64)
            recvCount: Receive count per rank [rankSize] (int64)
            recvOffset: Receive offset per rank [rankSize] (int64)
            maxBs: Max batch size [1] (int64)
            recvTokensPerExpert: Received tokens per expert [rankSize] (int64)
    """
    return _npu_notify_dispatch(
        sendData, tokenPerExpertData, sendCount, numTokens,
        commGroup, rankSize, rankId, localRankSize, localRankId)