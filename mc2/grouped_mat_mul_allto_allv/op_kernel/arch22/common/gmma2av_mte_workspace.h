/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Arch22 MTE workspace operations.
#ifndef COMMON_GMMA2AV_MTE_WORKSPACE_H
#define COMMON_GMMA2AV_MTE_WORKSPACE_H

#include "gmma2av_mte_context.h"

namespace MC2KernelTemplate {

template <typename T>
__aicore__ inline bool GmmA2avMteOp<T>::IsStageAiv() const
{
    return subBlockIdx_ == 0U;
}

template <typename T>
__aicore__ inline bool GmmA2avMteOp<T>::IsCommAiv() const
{
    return subBlockIdx_ != 0U;
}

template <typename T>
__aicore__ inline bool GmmA2avMteOp<T>::HasExpertWork(uint32_t expertIdx) const
{
    if (rankWorkerCount_ == 0U) {
        return false;
    }
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    uint64_t expertRows = 0UL;
    for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
        expertRows += static_cast<uint64_t>(taskTilingInfo_->sendCnt[dstRank * expertNum + expertIdx]);
    }
    if (expertRows == 0UL) {
        return false;
    }
    const uint32_t tileRows = Gmma2avMteTiling::L1_TILE_M;
    const uint32_t loopsM = static_cast<uint32_t>((expertRows + tileRows - 1UL) / tileRows);
    const uint32_t startMIdx = (logicalCoreIdx_ + rankWorkerCount_ - expertStartCoreIdx_) % rankWorkerCount_;
    return startMIdx < loopsM;
}

template <typename T>
__aicore__ inline GM_ADDR GmmA2avMteOp<T>::GetWindow(uint32_t rank) const
{
    if (isA3_) {
        if (rank == rank_) {
            return (GM_ADDR)(a3WinContext_->localWindowsIn);
        }
        auto relation =
            reinterpret_cast<__gm__ HcclRankRelationResV2 *>(a3WinContext_->remoteRes[rank].nextDevicePtr);
        return (GM_ADDR)(relation->windowsIn);
    }
    if (a2WinContext_->multiFlag == 0U) {
        return (GM_ADDR)(a2WinContext_->windowsIn[rank]);
    }
    return rank == rank_ ? (GM_ADDR)(a2WinContext_->data[rank].localInput.addr) :
                           (GM_ADDR)(a2WinContext_->data[rank].remoteInput.addr);
}

template <typename T>
__aicore__ inline uint64_t GmmA2avMteOp<T>::GetBufferIndex() const
{
    // Every kernel invocation enters buffer 0. Tiling can be cached, so
    // a host tiling counter must never determine the collective's phase.
    // End's COUNT_READY=0 then RECV_DONE=0 acknowledgement protects
    // immediate reuse before any next invocation publishes COUNT_READY=1.
    return 0UL;
}

template <typename T>
__aicore__ inline __gm__ int32_t *GmmA2avMteOp<T>::GetCountTable(uint32_t rank) const
{
    return reinterpret_cast<__gm__ int32_t *>(GetWindow(rank) + winSize_ -
                                              Gmma2avMteTiling::SEND_COUNT_STAGING_FROM_TAIL +
                                              GetBufferIndex() * Gmma2avMteTiling::COUNT_BUFFER_STRIDE);
}

template <typename T>
__aicore__ inline __gm__ T *GmmA2avMteOp<T>::GetOutputData(uint32_t rank) const
{
    return reinterpret_cast<__gm__ T *>(GetWindow(rank) +
        GetBufferIndex() * Gmma2avMteTiling::PayloadBufferBytes(winSize_));
}

template <typename T>
__aicore__ inline __gm__ int64_t *GmmA2avMteOp<T>::GetStageProgressSlot(uint32_t producerCore) const
{
    return stageProgressBase_ +
           static_cast<uint64_t>(producerCore) *
               (Gmma2avMteTiling::STAGE_PROGRESS_SLOT_BYTES / sizeof(int64_t));
}

template <typename T>
__aicore__ inline __gm__ int64_t *GmmA2avMteOp<T>::GetComputeProgressSlot(uint32_t producerCore) const
{
    return computeProgressBase_ + static_cast<uint64_t>(producerCore) *
           (Gmma2avMteTiling::STAGE_PROGRESS_SLOT_BYTES / sizeof(int64_t));
}

template <typename T>
__aicore__ inline AscendC::LocalTensor<int32_t> GmmA2avMteOp<T>::GetControlTensor()
{
    return ubBuffer_.template Get<int32_t>()[GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int32_t)];
}

template <typename T>
__aicore__ inline AscendC::LocalTensor<int64_t> GmmA2avMteOp<T>::GetSyncTensor()
{
    return ubBuffer_.template Get<int64_t>()[GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int64_t)];
}

template <typename T>
__aicore__ inline AscendC::LocalTensor<uint32_t> GmmA2avMteOp<T>::GetDestinationOffsetTensor()
{
    return ubBuffer_
        .template Get<uint32_t>()[GmmA2avMteDetail::DESTINATION_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
}

template <typename T>
__aicore__ inline AscendC::LocalTensor<uint32_t> GmmA2avMteOp<T>::GetSourceOffsetTensor()
{
    return ubBuffer_.template Get<uint32_t>()[GmmA2avMteDetail::SOURCE_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
}

template <typename T>
__aicore__ inline bool GmmA2avMteOp<T>::SafeAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
    if (lhs > maxValue - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

template <typename T>
__aicore__ inline bool GmmA2avMteOp<T>::SafeMulU32(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxU32 = 0xffffffffUL;
    if (lhs > maxU32 || rhs > maxU32) {
        return false;
    }
    // A sanitizer-instrumented 64 x 64 multiply lowers to compiler-rt's
    // __multi3, which is unavailable to arch22 AICore linking. Split the
    // bounded 32 x 32 product into native 16-bit partial products.
    const uint32_t lhs32 = static_cast<uint32_t>(lhs);
    const uint32_t rhs32 = static_cast<uint32_t>(rhs);
    const uint32_t lhsLow = lhs32 & 0xffffU;
    const uint32_t lhsHigh = lhs32 >> 16U;
    const uint32_t rhsLow = rhs32 & 0xffffU;
    const uint32_t rhsHigh = rhs32 >> 16U;
    const uint64_t low = static_cast<uint64_t>(lhsLow * rhsLow);
    const uint64_t middle = static_cast<uint64_t>(lhsLow * rhsHigh) + static_cast<uint64_t>(lhsHigh * rhsLow);
    const uint64_t high = static_cast<uint64_t>(lhsHigh * rhsHigh);
    result = low + (middle << 16U) + (high << 32U);
    return true;
}

template <typename T>
__aicore__ inline __gm__ int64_t *GmmA2avMteOp<T>::GetFixedSyncSlot(uint32_t windowRank, uint64_t phaseBase, uint64_t index) const
{
    const uint64_t syncOffset =
        winSize_ - Gmma2avMteTiling::SYNC_REGION_FROM_TAIL +
        GetBufferIndex() * Gmma2avMteTiling::FIXED_SYNC_BYTES + phaseBase + index * Gmma2avMteTiling::SYNC_SLOT_BYTES;
    return reinterpret_cast<__gm__ int64_t *>(GetWindow(windowRank) + syncOffset);
}

template <typename T>
__aicore__ inline __gm__ int64_t *GmmA2avMteOp<T>::GetExpertReadySlot(uint32_t windowRank, uint32_t sourceRank,
                                                     uint32_t expertIdx) const
{
    const uint64_t expertSlot = static_cast<uint64_t>(sourceRank) * taskTilingInfo_->e + expertIdx;
    return GetFixedSyncSlot(windowRank, Gmma2avMteTiling::EXPERT_READY_BASE, expertSlot);
}

template <typename T>
__aicore__ inline bool GmmA2avMteOp<T>::IsPrimaryAiv() const
{
    // The Comm subblock of logical core 0 publishes this rank's count
    // table and the rank-level phase flags.
    return logicalCoreIdx_ == 0U && IsCommAiv();
}

template <typename T>
__aicore__ inline uint32_t GmmA2avMteOp<T>::AlignControlElements(uint32_t elementCount) const
{
    constexpr uint32_t alignment = GmmA2avMteDetail::CONTROL_MTE_ELEMENTS;
    return (elementCount + alignment - 1U) / alignment * alignment;
}

} // namespace MC2KernelTemplate

#endif // COMMON_GMMA2AV_MTE_WORKSPACE_H
