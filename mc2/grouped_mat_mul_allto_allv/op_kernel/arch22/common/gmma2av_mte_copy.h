/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Arch22 MTE copy operations.
#ifndef COMMON_GMMA2AV_MTE_COPY_H
#define COMMON_GMMA2AV_MTE_COPY_H

#include "gmma2av_mte_context.h"

namespace MC2KernelTemplate {

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::LoadInt32Row(__gm__ int32_t *source, const AscendC::LocalTensor<int32_t> &local,
                                    uint32_t elementCount)
{
    const uint32_t alignedElementCount = AlignControlElements(elementCount);
    AscendC::GlobalTensor<int32_t> sourceGlobal;
    sourceGlobal.SetGlobalBuffer(source);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::DataCopy(local, sourceGlobal, alignedElementCount);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::StoreInt32Row(__gm__ int32_t *destination, const AscendC::LocalTensor<int32_t> &local,
                                     uint32_t elementCount)
{
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::GlobalTensor<int32_t> destinationGlobal;
    destinationGlobal.SetGlobalBuffer(destination);
    AscendC::DataCopy(destinationGlobal, local, elementCount);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::CopyGmToGm(__gm__ T *source, __gm__ T *destination, uint64_t elementCount)
{
    uint32_t moveIdx = 0U;
    BeginGmToGmCopy();
    CopyGmToGmSegment(source, destination, elementCount, moveIdx);
    EndGmToGmCopy();
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::BeginGmToGmCopy()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::CopyGmToGmSegment(__gm__ T *source, __gm__ T *destination, uint64_t elementCount,
                                         uint32_t &moveIdx)
{
    AscendC::LocalTensor<T> local = ubBuffer_.template Get<T>();
    AscendC::LocalTensor<T> ping = local;
    AscendC::LocalTensor<T> pong = local[GmmA2avMteDetail::PONG_UB_OFFSET_BYTES / sizeof(T)];
    uint64_t copied = 0UL;
    // Segmentation stays inside the caller's contiguous [source,
    // source + elementCount) range. Interior boundaries are multiples of
    // ubMoveElements_; only the final segment may carry a non-32B-aligned
    // tail length, which DataCopyPad supports (GM addresses have no
    // alignment constraint, blockLen must merely be sizeof(T) multiples).
    // Zero-length ranges never enter the loop.
    while (copied < elementCount) {
        const uint32_t current = static_cast<uint32_t>(
            elementCount - copied > ubMoveElements_ ? ubMoveElements_ : elementCount - copied);
        const auto eventId = (moveIdx & 1U) != 0U ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> copyTensor = (moveIdx & 1U) != 0U ? ping : pong;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        AscendC::GlobalTensor<T> sourceGlobal;
        sourceGlobal.SetGlobalBuffer(source + copied);
        AscendC::DataCopyExtParams copyParams(1U, current * sizeof(T), 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(copyTensor, sourceGlobal, copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::GlobalTensor<T> destinationGlobal;
        destinationGlobal.SetGlobalBuffer(destination + copied);
        AscendC::DataCopyPad(destinationGlobal, copyTensor, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        copied += current;
        ++moveIdx;
    }
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::EndGmToGmCopy()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

} // namespace MC2KernelTemplate

#endif // COMMON_GMMA2AV_MTE_COPY_H
