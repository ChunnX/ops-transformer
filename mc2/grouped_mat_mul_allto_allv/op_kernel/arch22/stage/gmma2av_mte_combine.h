/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Arch22 MTE combine operations.
#ifndef STAGE_GMMA2AV_MTE_COMBINE_H
#define STAGE_GMMA2AV_MTE_COMBINE_H

#include "../common/gmma2av_mte_context.h"

namespace MC2KernelTemplate {

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::CommExpertChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (IsStageAiv() || rankWorkerCount_ == 0U) {
        return;
    }
    WaitLocalStageProgress(startExpertIdx, endExpertIdx);
    PublishChunkReady(startExpertIdx, endExpertIdx);
    WaitChunkReady(startExpertIdx, endExpertIdx);
    PullExpertChunk(startExpertIdx, endExpertIdx);
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::PublishChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx)
{
    // Comm role only. The old all-AIV drain barrier is replaced by the
    // per-expert stage-progress wait in CommExpertChunk: ExpertReady for
    // expert e is published only after every producer of e's M tiles
    // has made its payload visible, at expert granularity.
    if (rankWorkerCount_ == 0U) {
        return;
    }
    for (uint32_t dstRank = logicalCoreIdx_; dstRank < rankSize_; dstRank += rankWorkerCount_) {
        if (PublishExpertReadyRange(dstRank, startExpertIdx, endExpertIdx, 1)) {
            continue;
        }
        PublishExpertReadyPerFlag(dstRank, startExpertIdx, endExpertIdx, 1);
    }
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::WaitChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx)
{
    // Comm role only. CommExpertChunk orders this after the worker's own
    // outbound publication for the same expert, so a worker never waits
    // on a slot whose publisher is itself blocked on a waiter.
    if (rankWorkerCount_ == 0U) {
        return;
    }
    for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
        if (WaitExpertReadyRange(srcRank, startExpertIdx, endExpertIdx, 1)) {
            continue;
        }
        WaitExpertReadyPerFlag(srcRank, startExpertIdx, endExpertIdx, 1);
    }
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::PullExpertChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (startExpertIdx == 0U) {
        // The same worker that acquired each source's ready flag loads
        // that source's count table and remains its payload-pull owner.
        BuildPeerSourceOffsets();
    }
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    AscendC::LocalTensor<uint32_t> destinationOffsets = GetDestinationOffsetTensor();
    AscendC::LocalTensor<uint32_t> sourceOffsets = GetSourceOffsetTensor();
    for (uint32_t srcRank = logicalCoreIdx_; rankWorkerCount_ != 0U && srcRank < rankSize_;
         srcRank += rankWorkerCount_) {
        // Source rows are expert-major/destination-major, whereas output
        // rows are source-major/expert-major. Other destinations can
        // leave gaps between this destination's expert slices. Merge
        // only when both sets of offsets prove the range contiguous.
        const uint64_t baseIndex = static_cast<uint64_t>(srcRank) * expertNum;
        const uint32_t firstSourceOffset = sourceOffsets.GetValue(baseIndex + startExpertIdx);
        const uint32_t firstDestinationOffset = destinationOffsets.GetValue(baseIndex + startExpertIdx);
        uint64_t chunkRows = 0UL;
        bool chunkContiguous = firstSourceOffset != GmmA2avMteDetail::INVALID_TOKEN_OFFSET &&
                               firstDestinationOffset != GmmA2avMteDetail::INVALID_TOKEN_OFFSET;
        for (uint32_t expertIdx = startExpertIdx; chunkContiguous && expertIdx < endExpertIdx; ++expertIdx) {
            const int32_t recvCount = taskTilingInfo_->recvCnt[baseIndex + expertIdx];
            if (recvCount < 0 ||
                sourceOffsets.GetValue(baseIndex + expertIdx) == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                destinationOffsets.GetValue(baseIndex + expertIdx) == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                static_cast<uint64_t>(sourceOffsets.GetValue(baseIndex + expertIdx)) !=
                    static_cast<uint64_t>(firstSourceOffset) + chunkRows ||
                static_cast<uint64_t>(destinationOffsets.GetValue(baseIndex + expertIdx)) !=
                    static_cast<uint64_t>(firstDestinationOffset) + chunkRows) {
                chunkContiguous = false;
                break;
            }
            chunkRows += static_cast<uint64_t>(recvCount);
        }
        if (chunkContiguous) {
            if (chunkRows > 0UL) {
                CopyGmToGm(GetOutputData(srcRank) + static_cast<uint64_t>(firstSourceOffset) * taskTilingInfo_->N1,
                           recvBuffer_ + static_cast<uint64_t>(firstDestinationOffset) * taskTilingInfo_->N1,
                           chunkRows * taskTilingInfo_->N1);
            }
            continue;
        }
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
            PullPeerExpertFromSource(srcRank, expertIdx);
        }
    }
    // End's all-AIV rendezvous orders the final pull before RECV_DONE.
    // No expert slot is overwritten while a peer may still be waiting.

}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::PullPeerExpertFromSource(uint32_t srcRank, uint32_t expertIdx)
{
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    AscendC::LocalTensor<uint32_t> destinationOffsets = GetDestinationOffsetTensor();
    AscendC::LocalTensor<uint32_t> sourceOffsets = GetSourceOffsetTensor();
    const uint64_t index = static_cast<uint64_t>(srcRank) * expertNum + expertIdx;
    const int32_t recvCount = taskTilingInfo_->recvCnt[index];
    const uint32_t sourceOffset = sourceOffsets.GetValue(index);
    const uint32_t destinationOffset = destinationOffsets.GetValue(index);
    if (recvCount <= 0 || sourceOffset == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
        destinationOffset == GmmA2avMteDetail::INVALID_TOKEN_OFFSET) {
        return;
    }
    CopyGmToGm(GetOutputData(srcRank) + static_cast<uint64_t>(sourceOffset) * taskTilingInfo_->N1,
               recvBuffer_ + static_cast<uint64_t>(destinationOffset) * taskTilingInfo_->N1,
               static_cast<uint64_t>(recvCount) * taskTilingInfo_->N1);
}

} // namespace MC2KernelTemplate

#endif // STAGE_GMMA2AV_MTE_COMBINE_H
