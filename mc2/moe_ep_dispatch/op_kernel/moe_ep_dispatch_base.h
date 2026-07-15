/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_dispatch_base.h
 * \brief Helper functions shared by MoE EP dispatch kernels.
 */

#ifndef MOE_EP_DISPATCH_BASE_H
#define MOE_EP_DISPATCH_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "adv_api/hcomm/hcomm.h"

#if __has_include("../common/mc2_moe_context.h")
#include "../common/mc2_moe_context.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace MoeEpDispatchBase {

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t DISPATCH_WIN_ADDR_ALIGN = 512U;
constexpr uint64_t DISPATCH_CQE_MAX_WRITE_SIZE = 256UL * 1024UL * 1024UL;

constexpr uint32_t DEBUG_OFFSET_PATH_MARKER = 0U;
constexpr uint32_t DEBUG_OFFSET_NETWORK_MODE = 1U;
constexpr uint32_t DEBUG_OFFSET_SUPER_NODE_COUNT = 2U;
constexpr uint32_t DEBUG_OFFSET_RANK_SIZE_PER_SERVER = 3U;
constexpr uint32_t DEBUG_OFFSET_CURRENT_SERVER_INDEX = 4U;
constexpr uint32_t DEBUG_OFFSET_REMOTE_ROUTE_COUNT = 8U;
constexpr uint32_t DEBUG_OFFSET_LOCAL_SEND_COUNT = 16U;
constexpr uint32_t DEBUG_OFFSET_ERROR_FLAGS = 24U;

constexpr uint8_t DEBUG_PATH_DIRECT = 0xD0U;
constexpr uint8_t DEBUG_PATH_HYBRID = 0xA1U;
constexpr uint32_t DEBUG_FLAG_REMOTE_ROUTE_UNSUPPORTED = 1U;

__aicore__ inline GM_ADDR GetWindowAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *context, uint32_t rankId,
                                                uint64_t offset)
{
    return (GM_ADDR)context->epHcclBuffer[rankId] + offset;
}

__aicore__ inline uint64_t GetCommHandle(__gm__ Mc2Aclnn::MoeCommContext *context, uint32_t localRankId,
                                         uint32_t remoteRankId)
{
    uint32_t commHandleIndex = remoteRankId > localRankId ? remoteRankId - 1U : remoteRankId;
    return context->hcommHandle[commHandleIndex];
}

__aicore__ inline uint32_t GetCurrentServerIndex(uint32_t epRankId, uint32_t rankSizePerServer)
{
    return rankSizePerServer == 0U ? 0U : epRankId / rankSizePerServer;
}

__aicore__ inline uint32_t GetServerStartRank(uint32_t epRankId, uint32_t rankSizePerServer)
{
    return GetCurrentServerIndex(epRankId, rankSizePerServer) * rankSizePerServer;
}

__aicore__ inline uint32_t GetServerEndRank(uint32_t epRankId, uint32_t rankSizePerServer, uint32_t epWorldSize)
{
    uint32_t serverEndRank = GetServerStartRank(epRankId, rankSizePerServer) + rankSizePerServer;
    return serverEndRank > epWorldSize ? epWorldSize : serverEndRank;
}

__aicore__ inline bool IsRankInSameServer(uint32_t sourceRankId, uint32_t targetRankId, uint32_t rankSizePerServer)
{
    return GetCurrentServerIndex(sourceRankId, rankSizePerServer) ==
           GetCurrentServerIndex(targetRankId, rankSizePerServer);
}

__aicore__ inline uint64_t GetScaleupSlotOffset(uint32_t rankId, uint32_t slotIndex, uint32_t maxTokenCount,
                                                uint32_t scaleupSlotBytes)
{
    return (static_cast<uint64_t>(rankId) * static_cast<uint64_t>(maxTokenCount) +
            static_cast<uint64_t>(slotIndex)) * static_cast<uint64_t>(scaleupSlotBytes);
}

__aicore__ inline GM_ADDR GetScaleupSendSlotAddr(GM_ADDR scaleupSendWorkspaceAddr, uint32_t targetRankId,
                                                 uint32_t slotIndex, uint32_t maxTokenCount,
                                                 uint32_t scaleupSlotBytes)
{
    return scaleupSendWorkspaceAddr + GetScaleupSlotOffset(targetRankId, slotIndex, maxTokenCount, scaleupSlotBytes);
}

__aicore__ inline GM_ADDR GetScaleupFinalRecvSlotAddr(GM_ADDR scaleupFinalRecvBaseAddr, uint32_t sourceRankId,
                                                      uint32_t slotIndex, uint32_t maxTokenCount,
                                                      uint32_t scaleupSlotBytes)
{
    return scaleupFinalRecvBaseAddr + GetScaleupSlotOffset(sourceRankId, slotIndex, maxTokenCount, scaleupSlotBytes);
}

__aicore__ inline void WriteUint32ToDebug(GlobalTensor<uint8_t> &debugOutTensor, uint32_t byteOffset,
                                          uint32_t value)
{
    debugOutTensor.SetValue(byteOffset, static_cast<uint8_t>(value & 0xFFU));
    debugOutTensor.SetValue(byteOffset + 1U, static_cast<uint8_t>((value >> 8U) & 0xFFU));
    debugOutTensor.SetValue(byteOffset + 2U, static_cast<uint8_t>((value >> 16U) & 0xFFU));
    debugOutTensor.SetValue(byteOffset + 3U, static_cast<uint8_t>((value >> 24U) & 0xFFU));
}

__aicore__ inline void IncrementDebugUint32(GlobalTensor<uint8_t> &debugOutTensor, uint32_t byteOffset)
{
    uint32_t value = static_cast<uint32_t>(debugOutTensor.GetValue(byteOffset)) |
                     (static_cast<uint32_t>(debugOutTensor.GetValue(byteOffset + 1U)) << 8U) |
                     (static_cast<uint32_t>(debugOutTensor.GetValue(byteOffset + 2U)) << 16U) |
                     (static_cast<uint32_t>(debugOutTensor.GetValue(byteOffset + 3U)) << 24U);
    WriteUint32ToDebug(debugOutTensor, byteOffset, value + 1U);
}

template <typename HcommType>
__aicore__ inline void WriteRemoteWindowByChunks(HcommType &hcomm, uint64_t commHandle, GM_ADDR remoteWindowAddr,
                                                 GM_ADDR localWorkspaceAddr, uint64_t sendBytes)
{
    uint64_t splitCount = (sendBytes + DISPATCH_CQE_MAX_WRITE_SIZE - 1UL) / DISPATCH_CQE_MAX_WRITE_SIZE;
    uint64_t dataBytes = DISPATCH_CQE_MAX_WRITE_SIZE;
    for (uint64_t splitIndex = 0UL; splitIndex < splitCount; splitIndex++) {
        if (splitIndex == splitCount - 1UL) {
            dataBytes = sendBytes - splitIndex * DISPATCH_CQE_MAX_WRITE_SIZE;
        }
        hcomm.WriteNbi(commHandle, remoteWindowAddr, localWorkspaceAddr, dataBytes);
        localWorkspaceAddr += dataBytes;
        remoteWindowAddr += dataBytes;
    }
    hcomm.Drain(commHandle);
}

}  // namespace MoeEpDispatchBase

#endif  // MOE_EP_DISPATCH_BASE_H
