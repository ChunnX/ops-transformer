/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Arch22 MTE prepare operations.
#ifndef STAGE_GMMA2AV_MTE_PREPARE_H
#define STAGE_GMMA2AV_MTE_PREPARE_H

#include "../common/gmma2av_mte_context.h"

namespace MC2KernelTemplate {

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::Init(const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                            GM_ADDR recvBuffer, GM_ADDR cumsumBuffer, AscendC::TPipe *pipe,
                            const GmmA2avCoCTiling *cocTiling, uint64_t commBufferSize,
                            uint32_t isA3)
{
    taskTilingInfo_ = taskTilingInfo;
    sendBuffer_ = reinterpret_cast<__gm__ T *>(sendBuffer);
    recvBuffer_ = reinterpret_cast<__gm__ T *>(recvBuffer);
    if ASCEND_IS_AIC {
        return;
    }

    auto context = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    isA3_ = isA3 != 0U;
    if (isA3_) {
        // The established A3 package uses the full MC2 resource context.
        // Its peer windows are referenced by remoteRes[].
        a3WinContext_ = reinterpret_cast<__gm__ HcclOpResParam *>(context);
        rank_ = a3WinContext_->localUsrRankId;
        rankSize_ = a3WinContext_->rankSize;
        winSize_ = a3WinContext_->winSize == 0UL ? commBufferSize : a3WinContext_->winSize;
    } else {
        // Ascend 910B MultiPut exposes HcclCombineOpParam. For larger
        // communication domains HCCL supplies the dynamic data[] table.
        a2WinContext_ = reinterpret_cast<__gm__ AscendC::HcclCombineOpParam *>(context);
        rank_ = a2WinContext_->rankId;
        rankSize_ = a2WinContext_->rankNum;
        winSize_ = a2WinContext_->winSize == 0UL ? commBufferSize : a2WinContext_->winSize;
    }
    // The serialized layout places the (unused) cumsum region directly
    // after the GMM output; the local stage-progress slots are appended
    // after it. Host allocation and this derivation share
    // CumsumRegionBytes(), so the two can never drift.
    stageProgressBase_ = reinterpret_cast<__gm__ int64_t *>(
        cumsumBuffer +
        MC2KernelTemplate::Gmma2avMteTiling::CumsumRegionBytes(
            rankSize_, static_cast<uint64_t>(taskTilingInfo_->e)));
    computeProgressBase_ = reinterpret_cast<__gm__ int64_t *>(
        cumsumBuffer + Gmma2avMteTiling::ComputeProgressOffset(
            rankSize_, taskTilingInfo_->e, taskTilingInfo_->aicCoreNum));
    coreIdx_ = AscendC::GetBlockIdx();
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    logicalCoreIdx_ = coreIdx_ / AscendC::GetTaskRation();
    rankWorkerCount_ = static_cast<uint32_t>(taskTilingInfo_->aicCoreNum);
    AscendC::SetAtomicNone();
    if (cocTiling != nullptr && cocTiling->ubMoveNum > 0) {
        ubMoveElements_ = static_cast<uint32_t>(cocTiling->ubMoveNum);
    }
    // A segment can never exceed one payload ping/pong region, whatever
    // the host passed: DataCopyPad's blockLen ceiling (2 MiB) is far
    // above the 64 KiB region, so UB capacity is the only hard bound.
    const uint32_t payloadRegionElements =
        static_cast<uint32_t>(Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES / sizeof(T));
    if (ubMoveElements_ > payloadRegionElements) {
        ubMoveElements_ = payloadRegionElements;
    }

    // The first 64 KiB is the payload ping buffer. Count-table scratch is
    // staged at 64 KiB, destination offsets at 72 KiB, source offsets at
    // 80 KiB, and payload pong starts at 96 KiB.
    pipe->InitBuffer(ubBuffer_, GmmA2avMteDetail::USED_UB_BYTES);
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::BeginExpertPipeline()
{
    if ASCEND_IS_AIC {
        return;
    }
    // Role split: subblock 0 stages workspace→window and publishes a
    // local GM progress slot; subblock 1 owns the cross-rank protocol.
    // Both roles enter the common-init rendezvous below, which orders
    // every progress-slot clear before the first Comm-side progress
    // read in the expert loop and the count-table publication before
    // the COUNT_READY phase.
    if (IsStageAiv()) {
        // Clear this producer's stage-progress slot (slot value is
        // expertIdx + 1, monotonic within the invocation). Clearing it
        // before the common-init barrier below guarantees the Comm role
        // cannot satisfy an early wait with a stale value from a
        // previous invocation.
        StoreSyncFlag(GetStageProgressSlot(logicalCoreIdx_), 0);
        StoreSyncFlag(GetComputeProgressSlot(logicalCoreIdx_), 0);
    } else {
        // Clear all flags in this invocation's selected ring buffer.
        ResetSyncRegion();
        PublishLocalCountTable();
    }
    // All peers must finish clearing their expert mailboxes before any
    // producer can publish into a destination's window this round.
    PublishAndWaitRankPhase(Gmma2avMteTiling::COUNT_READY_BASE, 1);
    if (!IsStageAiv()) {
    BuildDestinationOffsets();
    }
    LocalAivSync();
    localExpertBaseRows_ = 0UL;
    // Mirror the AIC scheduler's initial cross-expert rotation state.
    expertStartCoreIdx_ = 0U;
    computeWaveSequence_ = 0;
    // All local progress clears must finish before any AIC publishes a
    // wave. AIC joins this once, before constructing its CATLASS pipeline.
    AscendC::SyncAll<false>();
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::Wait(uint32_t startExpertIdx)
{
    (void)startExpertIdx;
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::End()
{
    if ASCEND_IS_AIV {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        PublishAndWaitRankPhase(Gmma2avMteTiling::RECV_DONE_BASE, 2);
        // All payload pulls have finished. Clear the entry flag and wait
        // for every peer to clear it before leaving this buffer. The final
        // RECV_DONE=0 acknowledgement below protects immediate reuse.
        PublishAndWaitRankPhase(Gmma2avMteTiling::COUNT_READY_BASE, 0);
        // Acknowledge that every peer has observed the cleared entry flag.
        // A peer cannot republish DONE until all ranks enter its next round,
        // so this cleared acknowledgement cannot be lost during exit skew.
        PublishAndWaitRankPhase(Gmma2avMteTiling::RECV_DONE_BASE, 0);
    }
    AscendC::SyncAll<false>();
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::PublishLocalCountTable()
{
    if (!IsPrimaryAiv()) {
        return;
    }
    const uint32_t countNum = rankSize_ * static_cast<uint32_t>(taskTilingInfo_->e);
    const uint32_t alignedCountNum = AlignControlElements(countNum);
    AscendC::LocalTensor<int32_t> local = GetControlTensor();
    for (uint32_t index = 0U; index < countNum; ++index) {
        local.SetValue(index, taskTilingInfo_->sendCnt[index]);
    }
    for (uint32_t index = countNum; index < alignedCountNum; ++index) {
        local.SetValue(index, 0);
    }
    StoreInt32Row(GetCountTable(rank_), local, alignedCountNum);
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::BuildDestinationOffsets()
{
    if (IsStageAiv()) {
        return;
    }
    AscendC::LocalTensor<uint32_t> offsets = GetDestinationOffsetTensor();
    const uint32_t countNum = rankSize_ * static_cast<uint32_t>(taskTilingInfo_->e);
    uint64_t runningOffset = 0UL;
    bool prefixValid = true;
    for (uint32_t index = 0U; index < countNum; ++index) {
        const int32_t count = taskTilingInfo_->recvCnt[index];
        const bool currentValid = prefixValid && count >= 0 &&
                                  runningOffset <= GmmA2avMteDetail::INVALID_TOKEN_OFFSET &&
                                  runningOffset <= taskTilingInfo_->BSK &&
                                  static_cast<uint64_t>(count) <= taskTilingInfo_->BSK - runningOffset;
        offsets.SetValue(
            index, currentValid ? static_cast<uint32_t>(runningOffset) : GmmA2avMteDetail::INVALID_TOKEN_OFFSET);
        if (!currentValid) {
            prefixValid = false;
        } else {
            runningOffset += static_cast<uint64_t>(count);
        }
    }
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::BuildPeerSourceOffsets()
{
    if (IsStageAiv() || rankWorkerCount_ == 0U) {
        return;
    }
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    const uint32_t countNum = rankSize_ * expertNum;
    const uint64_t payloadCapacityBytes = Gmma2avMteTiling::PayloadBufferBytes(winSize_);
    AscendC::LocalTensor<int32_t> peerCounts = GetControlTensor();
    AscendC::LocalTensor<uint32_t> destinationOffsets = GetDestinationOffsetTensor();
    AscendC::LocalTensor<uint32_t> sourceOffsets = GetSourceOffsetTensor();
    for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
        if (srcRank == rank_) {
            for (uint32_t index = 0U; index < countNum; ++index) {
                peerCounts.SetValue(index, taskTilingInfo_->sendCnt[index]);
            }
        } else {
            LoadInt32Row(GetCountTable(srcRank), peerCounts, countNum);
        }
        uint64_t expertSourceBase = 0UL;
        bool sourcePrefixValid = true;
        for (uint32_t expert = 0U; expert < expertNum; ++expert) {
            uint64_t sourceOffsetInExpert = 0UL;
            uint64_t expertTotal = 0UL;
            bool expertLayoutValid = sourcePrefixValid;
            for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                const int32_t count = peerCounts.GetValue(static_cast<uint64_t>(dstRank) * expertNum + expert);
                if (count < 0) {
                    expertLayoutValid = false;
                    continue;
                }
                uint64_t updated = 0UL;
                if (dstRank < rank_ && !SafeAdd(sourceOffsetInExpert, static_cast<uint64_t>(count), updated)) {
                    expertLayoutValid = false;
                } else if (dstRank < rank_) {
                    sourceOffsetInExpert = updated;
                }
                if (!SafeAdd(expertTotal, static_cast<uint64_t>(count), updated)) {
                    expertLayoutValid = false;
                } else {
                    expertTotal = updated;
                }
            }
            const uint64_t localIndex = static_cast<uint64_t>(srcRank) * expertNum + expert;
            const int32_t peerSendCount = peerCounts.GetValue(static_cast<uint64_t>(rank_) * expertNum + expert);
            const int32_t localRecvCount = taskTilingInfo_->recvCnt[localIndex];
            uint64_t sourceOffset = 0UL;
            bool expertValid = expertLayoutValid && SafeAdd(expertSourceBase, sourceOffsetInExpert, sourceOffset);
            uint64_t sourceEndRows = 0UL;
            uint64_t sourceEndElements = 0UL;
            if (peerSendCount < 0 || localRecvCount < 0 || peerSendCount != localRecvCount ||
                sourceOffset >= GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                destinationOffsets.GetValue(localIndex) == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                !SafeAdd(sourceOffset, static_cast<uint64_t>(peerSendCount), sourceEndRows) ||
                !SafeMulU32(sourceEndRows, static_cast<uint64_t>(taskTilingInfo_->N1), sourceEndElements) ||
                sourceEndElements > payloadCapacityBytes / sizeof(T)) {
                expertValid = false;
            }
            // A source rank is built and consumed by the same AIV worker.
            // Keep its offsets owner-local instead of routing private
            // state through a reused shared-GM scratch whose cross-core
            // visibility and cache lifecycle provide no useful sharing.
            sourceOffsets.SetValue(localIndex, expertValid ? static_cast<uint32_t>(sourceOffset) :
                                                             GmmA2avMteDetail::INVALID_TOKEN_OFFSET);
            if (!expertLayoutValid || !SafeAdd(expertSourceBase, expertTotal, expertSourceBase)) {
                sourcePrefixValid = false;
            }
        }
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

} // namespace MC2KernelTemplate

#endif // STAGE_GMMA2AV_MTE_PREPARE_H
