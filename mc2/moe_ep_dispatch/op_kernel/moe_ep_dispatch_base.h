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
 * \brief
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

constexpr uint32_t DISPATCH_WIN_ADDR_ALIGN = 512U;
constexpr uint64_t DISPATCH_CQE_MAX_WRITE_SIZE = 256UL * 1024UL * 1024UL;

// debug_out中各字段的字节偏移
constexpr uint32_t DEBUG_OFFSET_PATH_MARKER = 0U;
constexpr uint32_t DEBUG_OFFSET_NETWORK_MODE = 1U;
constexpr uint32_t DEBUG_OFFSET_SUPER_NODE_COUNT = 2U;
constexpr uint32_t DEBUG_OFFSET_RANK_SIZE_PER_SERVER = 3U;
constexpr uint32_t DEBUG_OFFSET_CURRENT_SERVER_INDEX = 4U;
constexpr uint32_t DEBUG_OFFSET_TOTAL_CYCLES = 8U;
constexpr uint32_t DEBUG_OFFSET_PREPARE_CYCLES = 16U;
constexpr uint32_t DEBUG_OFFSET_TRANSFER_CYCLES = 24U;
constexpr uint32_t DEBUG_OFFSET_CAL_SEND_COUNT_CYCLES = 32U;
constexpr uint32_t DEBUG_OFFSET_EXCHANGE_COUNT_CYCLES = 40U;
constexpr uint32_t DEBUG_OFFSET_RECV_COUNT_WAIT_CYCLES = 48U;
constexpr uint32_t DEBUG_OFFSET_ROUTE_SLOT_CYCLES = 56U;
constexpr uint32_t DEBUG_OFFSET_PAYLOAD_BUILD_CYCLES = 64U;
constexpr uint32_t DEBUG_OFFSET_SEND_PHASE_CYCLES = 72U;
constexpr uint32_t DEBUG_OFFSET_LOCAL_SEND_CYCLES = 80U;
constexpr uint32_t DEBUG_OFFSET_SCALEOUT_SEND_CYCLES = 88U;
constexpr uint32_t DEBUG_OFFSET_INITIAL_SEND_CYCLES = 96U;
constexpr uint32_t DEBUG_OFFSET_READY_WAIT_CYCLES = 104U;
constexpr uint32_t DEBUG_OFFSET_FANOUT_STAGE_CYCLES = 112U;
constexpr uint32_t DEBUG_OFFSET_FANOUT_SEND_CYCLES = 120U;
constexpr uint32_t DEBUG_OFFSET_ACK_WAIT_CYCLES = 128U;
constexpr uint32_t DEBUG_OFFSET_READY_POLL_COUNT = 136U;
constexpr uint32_t DEBUG_OFFSET_ACK_POLL_COUNT = 144U;
constexpr uint32_t DEBUG_OFFSET_SCALEOUT_SLOT_COUNT = 152U;
constexpr uint32_t DEBUG_OFFSET_FANOUT_PAYLOAD_COUNT = 156U;
constexpr uint32_t DEBUG_OFFSET_FANOUT_ACTIVE_RANK_COUNT = 160U;
constexpr uint32_t DEBUG_OFFSET_FANOUT_NBI_COUNT = 164U;
constexpr uint32_t DEBUG_OFFSET_FANOUT_DRAIN_COUNT = 168U;
constexpr uint32_t DEBUG_OFFSET_SCALEOUT_PAYLOAD_SUBMIT_CYCLES = 176U;
constexpr uint32_t DEBUG_OFFSET_SCALEOUT_PAYLOAD_DRAIN_CYCLES = 184U;
constexpr uint32_t DEBUG_OFFSET_SCALEOUT_NOTIFY_SUBMIT_CYCLES = 192U;
constexpr uint32_t DEBUG_OFFSET_SCALEOUT_NOTIFY_DRAIN_CYCLES = 200U;
constexpr uint32_t DEBUG_OFFSET_SCALEOUT_PAYLOAD_BYTES = 208U;

constexpr uint8_t DEBUG_PATH_DIRECT = 0xD0U;
constexpr uint8_t DEBUG_PATH_HYBRID = 0xA1U;

// PerExpertCnt Calculation
template <Reg::HistogramsType htype, typename T, typename U>
__simd_vf__ __aicore__ inline void HistogramsVf(__ubuf__ U *dst, __ubuf__ T *src, uint16_t repeatElm,
                                           uint16_t halfRepeat, uint32_t totalElm, uint16_t repeatTimes)
{
    Reg::RegTensor<T> srcReg;
    Reg::RegTensor<U> dst0Reg;
    Reg::RegTensor<U> dst1Reg;
    Reg::MaskReg pregOut = Reg::CreateMask<U>();
    Reg::Duplicate(dst0Reg, 0);
    Reg::Duplicate(dst1Reg, 0);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t remaining = totalElm - static_cast<uint32_t>(i) * repeatElm;
        uint32_t valid = (remaining > repeatElm) ? repeatElm : remaining;
        Reg::MaskReg preg = Reg::UpdateMask<T>(valid);
        Reg::LoadAlign(srcReg, src + repeatElm * i);
        Reg::Histograms<T, U, Reg::HistogramsBinType::BIN0, htype>(dst0Reg, srcReg, preg);
        Reg::Histograms<T, U, Reg::HistogramsBinType::BIN1, htype>(dst1Reg, srcReg, preg);
    }
    Reg::StoreAlign(dst, dst0Reg, pregOut);
    Reg::StoreAlign(dst + halfRepeat, dst1Reg, pregOut);
}

__aicore__ inline void GetExpertFreq(LocalTensor<uint16_t> &dstLocal, LocalTensor<uint8_t> &srcLocal, uint32_t totalElm)
{
    uint32_t repeatElm = GetVecLen();
    uint16_t repeatTimes = Ceil(totalElm, repeatElm);
    __ubuf__ uint8_t *src = (__ubuf__ uint8_t *)srcLocal.GetPhyAddr();
    __ubuf__ uint16_t *dst = (__ubuf__ uint16_t *)dstLocal.GetPhyAddr();
    asc_vf_call<HistogramsVf<Reg::HistogramsType::FREQUENCY, uint8_t, uint16_t>>(dst, src, repeatElm, repeatElm >> 1,
                                                                                 totalElm, repeatTimes);
    PipeBarrier<PIPE_V>();
}

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

// For Hybrid
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
    return (static_cast<uint64_t>(rankId) * static_cast<uint64_t>(maxTokenCount) + static_cast<uint64_t>(slotIndex)) *
           static_cast<uint64_t>(scaleupSlotBytes);
}

__aicore__ inline GM_ADDR GetScaleupSendSlotAddr(GM_ADDR scaleupSendWorkspaceAddr, uint32_t targetRankId,
                                                 uint32_t slotIndex, uint32_t maxTokenCount, uint32_t scaleupSlotBytes)
{
    return scaleupSendWorkspaceAddr + GetScaleupSlotOffset(targetRankId, slotIndex, maxTokenCount, scaleupSlotBytes);
}

__aicore__ inline GM_ADDR GetScaleupFinalRecvSlotAddr(GM_ADDR scaleupFinalRecvBaseAddr, uint32_t sourceRankId,
                                                      uint32_t slotIndex, uint32_t maxTokenCount,
                                                      uint32_t scaleupSlotBytes)
{
    return scaleupFinalRecvBaseAddr + GetScaleupSlotOffset(sourceRankId, slotIndex, maxTokenCount, scaleupSlotBytes);
}

__aicore__ inline void WriteUint32ToDebug(GlobalTensor<uint8_t> &debugOutTensor, uint32_t byteOffset, uint32_t value)
{
    debugOutTensor.SetValue(byteOffset, static_cast<uint8_t>(value & 0xFFU));
    debugOutTensor.SetValue(byteOffset + 1U, static_cast<uint8_t>((value >> 8U) & 0xFFU));
    debugOutTensor.SetValue(byteOffset + 2U, static_cast<uint8_t>((value >> 16U) & 0xFFU));
    debugOutTensor.SetValue(byteOffset + 3U, static_cast<uint8_t>((value >> 24U) & 0xFFU));
}

__aicore__ inline void WriteUint64ToDebug(GlobalTensor<uint8_t> &debugOutTensor, uint32_t byteOffset, uint64_t value)
{
    for (uint32_t byteIndex = 0U; byteIndex < sizeof(uint64_t); byteIndex++) {
        debugOutTensor.SetValue(byteOffset + byteIndex,
                                static_cast<uint8_t>((value >> (byteIndex * 8U)) & 0xFFUL));
    }
}

template <typename HcommType>
__aicore__ inline void WriteRemoteWindowByChunks(HcommType &hcomm, uint64_t commHandle, GM_ADDR remoteWindowAddr,
                                                 GM_ADDR localWorkspaceAddr, uint64_t sendBytes)
{
    // HCCL单次写有CQE大小限制，保持原分片写逻辑
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


} // namespace MoeEpDispatchBase

#endif // MOE_EP_DISPATCH_BASE_H
