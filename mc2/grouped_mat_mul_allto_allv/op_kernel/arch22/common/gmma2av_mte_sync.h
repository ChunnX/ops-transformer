/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Arch22 MTE sync operations.
#ifndef COMMON_GMMA2AV_MTE_SYNC_H
#define COMMON_GMMA2AV_MTE_SYNC_H

#include "gmma2av_mte_context.h"

namespace MC2KernelTemplate {

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::PublishStageProgress(uint32_t expertIdx)
{
    GmmA2avMteTimerScope<TimerType> trace(timer_, 26, expertIdx);
    if (!IsStageAiv()) {
        return;
    }
    StoreSyncFlag(GetStageProgressSlot(logicalCoreIdx_), static_cast<int64_t>(expertIdx) + 1);
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::WaitComputeProgress(int64_t sequence)
{
    // AIC releases its scalar store with DCCI. Read through the normal
    // local memory path, matching the GM-ready poll in mega_moe.
    WaitLocalProgress(GetComputeProgressSlot(logicalCoreIdx_), sequence, false);
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::WaitStageProgress(uint32_t producerCore, int64_t boundary)
{
    WaitLocalProgress(GetStageProgressSlot(producerCore), boundary);
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::WaitLocalProgress(__gm__ int64_t *slot, int64_t boundary, bool bypassL2)
{
    AscendC::LocalTensor<int64_t> local = GetSyncTensor();
    AscendC::GlobalTensor<int64_t> global;
    global.SetGlobalBuffer(slot);
    // Stage->Comm keeps the established MTE publication/polling path.
    // AIC->Stage uses normal local loads after the scalar writer's DCCI.
    if (bypassL2) {
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    // MTE updates this UB storage asynchronously; each poll must issue a
    // fresh scalar load even when the loop has no other scalar side effects.
    while (true) {
        AscendC::DataCopy(local, global, GmmA2avMteDetail::SYNC_MTE_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        // >= rather than ==: the producer publishes monotonic boundaries,
        // so a later wave/expert can overwrite the boundary being waited
        // for while still proving all its preceding writes are complete.
        if (local.GetValue(0) >= boundary) {
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            break;
        }
    }
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::WaitLocalStageProgress(uint32_t startExpertIdx, uint32_t endExpertIdx)
{
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    const uint32_t aicCoreNum = rankWorkerCount_;
    const uint32_t tileRows = Gmma2avMteTiling::L1_TILE_M;
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
        uint64_t expertRows = 0UL;
        for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
            expertRows += static_cast<uint64_t>(taskTilingInfo_->sendCnt[dstRank * expertNum + expertIdx]);
        }
        const uint32_t loopsM = static_cast<uint32_t>((expertRows + tileRows - 1UL) / tileRows);
        if (aicCoreNum != 0U && expertRows > 0UL) {
            const uint32_t producerCount = loopsM < aicCoreNum ? loopsM : aicCoreNum;
            for (uint32_t mIdx = 0U; mIdx < producerCount; ++mIdx) {
                WaitStageProgress((expertStartCoreIdx_ + mIdx) % aicCoreNum, static_cast<int64_t>(expertIdx) + 1);
            }
        }
        expertStartCoreIdx_ = (expertStartCoreIdx_ + loopsM) % aicCoreNum;
    }
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::LocalAivSync()
{
    // On arch22 SyncAll<true> drains local pipelines and performs an
    // AIV-only inter-core rendezvous. After the Stage/Comm role split
    // this is used ONLY at common initialization (BeginExpertPipeline)
    // and common exit (End), which both roles reach; the per-expert
    // loops contain no all-AIV rendezvous.
    AscendC::SyncAll<true>();
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::PublishAndWaitRankPhase(uint64_t phaseBase, int64_t flag)
{
    LocalAivSync();
    if (IsCommAiv() && logicalCoreIdx_ == 0U) {
        StoreSyncFlag(GetFixedSyncSlot(rank_, phaseBase, 0UL), flag);
    }
    // Do not allow a worker to wait until this rank's own state is
    // published; the slot must be visible before anyone polls peers.
    LocalAivSync();
    if (IsCommAiv() && rankWorkerCount_ != 0U) {
        for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
            // Peer windows must be polled without the L2-bypass alias:
            // the alias is only valid for local GM addresses.
            WaitSyncFlag(GetFixedSyncSlot(srcRank, phaseBase, 0UL), flag, srcRank == rank_);
        }
    }
    LocalAivSync();
}

template <typename T, typename TimerType>
__aicore__ inline bool GmmA2avMteOp<T, TimerType>::IsExpertSlotActive(const int32_t *countTable, uint32_t peerRank, uint32_t expertIdx,
                                          uint32_t expertNum) const
{
    if (expertIdx == 0U || peerRank == rank_) {
        return true;
    }
    return countTable[static_cast<uint64_t>(peerRank) * expertNum + expertIdx] > 0;
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::ResetSyncRegion()
{
    if (IsStageAiv() || rankWorkerCount_ == 0U) {
        return;
    }
    AscendC::LocalTensor<int64_t> staging = GetSyncTensor();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    for (uint32_t i = 0U; i < GmmA2avMteDetail::FLAG_STAGING_ELEMENTS; ++i) {
        staging.SetValue(i, 0);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);

    const uint32_t slotsPerBatch =
        GmmA2avMteDetail::FLAG_STAGING_ELEMENTS / GmmA2avMteDetail::SYNC_SLOT_ELEMENTS;
    // Clear every flag in the selected ring buffer before COUNT_READY.
    const uint64_t resetBase = 0UL;
    const uint32_t totalSlots = static_cast<uint32_t>(
        Gmma2avMteTiling::FIXED_SYNC_BYTES / Gmma2avMteTiling::SYNC_SLOT_BYTES);
    // Batch ownership is split round-robin across the Comm workers: the
    // c-th worker clears batches c, c + rankWorkerCount_, ... (batch b
    // spans slots [b * slotsPerBatch, min((b+1) * slotsPerBatch,
    // totalSlots))), so every slot has exactly one writer and the union
    // covers the chosen reset region, including all expert slots when
    // the CCL window is reused across different shapes. Workers
    // without a batch fall straight through to the common rendezvous.
    // Ordering: each batch's MTE3 store is drained by WaitFlag + DDR
    // barrier before this worker reaches PublishAndWaitRankPhase, whose
    // leading LocalAivSync makes every AIV (including the COUNT_READY
    // publisher) wait until all workers finished their share; no
    // new-round flag can be observed before the entire clear is
    // globally visible.
    for (uint32_t batch = logicalCoreIdx_; batch * slotsPerBatch < totalSlots;
         batch += rankWorkerCount_) {
        const uint32_t offset = batch * slotsPerBatch;
        const uint32_t batchSlots = (totalSlots - offset > slotsPerBatch) ?
                                        slotsPerBatch : (totalSlots - offset);
        const uint32_t batchElements = batchSlots * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS;
        AscendC::GlobalTensor<int64_t> global;
        global.SetGlobalBuffer(GetFixedSyncSlot(rank_, resetBase, offset));
        AscendC::DataCopy(global, staging, batchElements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
}

template <typename T, typename TimerType>
__aicore__ inline bool GmmA2avMteOp<T, TimerType>::PublishExpertReadyRange(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                               int64_t flag)
{
    const uint32_t slotCount = endExpertIdx - startExpertIdx;
    const uint32_t totalElements = slotCount * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS;
    if (totalElements > GmmA2avMteDetail::FLAG_STAGING_ELEMENTS) {
        return false;
    }
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    AscendC::LocalTensor<int64_t> staging = GetSyncTensor();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    // The batched copy transfers complete 64-byte slots, including padding
    // and inactive flags. Initialize every element read by that transfer.
    for (uint32_t index = 0U; index < totalElements; ++index) {
        staging.SetValue(index, 0);
    }
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
        if (!IsExpertSlotActive(taskTilingInfo_->sendCnt, dstRank, expertIdx, expertNum)) {
            continue;
        }
        staging.SetValue((expertIdx - startExpertIdx) * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS, flag);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::GlobalTensor<int64_t> global;
    global.SetGlobalBuffer(GetExpertReadySlot(dstRank, rank_, startExpertIdx));
    AscendC::DataCopy(global, staging, totalElements);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return true;
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::PublishExpertReadyPerFlag(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                 int64_t flag)
{
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
        if (!IsExpertSlotActive(taskTilingInfo_->sendCnt, dstRank, expertIdx, expertNum)) {
            continue;
        }
        StoreSyncFlag(GetExpertReadySlot(dstRank, rank_, expertIdx), flag);
    }
}

template <typename T, typename TimerType>
__aicore__ inline bool GmmA2avMteOp<T, TimerType>::WaitExpertReadyRange(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                            int64_t expected)
{
    const uint32_t slotCount = endExpertIdx - startExpertIdx;
    const uint32_t totalElements = slotCount * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS;
    if (totalElements > GmmA2avMteDetail::FLAG_STAGING_ELEMENTS) {
        return false;
    }
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    AscendC::LocalTensor<int64_t> staging = GetSyncTensor();
    AscendC::GlobalTensor<int64_t> global;
    // Sync slots are always polled through this rank's local window. The
    // bypass alias is valid for local GM and prevents a stale cache line
    // after a peer updates the slot; never apply it to encoded peer VAs.
    global.SetGlobalBuffer(GetExpertReadySlot(rank_, srcRank, startExpertIdx));
    global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    // MTE updates this UB storage asynchronously; each poll must issue a
    // fresh scalar load even when the loop has no other scalar side effects.
    while (true) {
        AscendC::DataCopy(staging, global, totalElements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        bool allReady = true;
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
            if (!IsExpertSlotActive(taskTilingInfo_->recvCnt, srcRank, expertIdx, expertNum)) {
                continue;
            }
            if (staging.GetValue((expertIdx - startExpertIdx) * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS) != expected) {
                allReady = false;
                break;
            }
        }
        if (allReady) {
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            return true;
        }
    }
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::WaitExpertReadyPerFlag(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                              int64_t expected)
{
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
        if (!IsExpertSlotActive(taskTilingInfo_->recvCnt, srcRank, expertIdx, expertNum)) {
            continue;
        }
        WaitSyncFlag(GetExpertReadySlot(rank_, srcRank, expertIdx), expected, true);
    }
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::StoreSyncFlag(__gm__ int64_t *destination, int64_t flag)
{
    AscendC::LocalTensor<int64_t> local = GetSyncTensor();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    local.SetValue(0U, flag);
    for (uint32_t index = 1U; index < GmmA2avMteDetail::SYNC_MTE_ELEMENTS; ++index) {
        local.SetValue(index, 0);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::GlobalTensor<int64_t> global;
    global.SetGlobalBuffer(destination);
    // A3's established remote-state protocols publish one complete 32-byte
    // block.  Keeping the transfer naturally aligned also avoids issuing a
    // short MTE transaction against a peer mapping.
    AscendC::DataCopy(global, local, GmmA2avMteDetail::SYNC_MTE_ELEMENTS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

template <typename T, typename TimerType>
__aicore__ inline void GmmA2avMteOp<T, TimerType>::WaitSyncFlag(__gm__ int64_t *source, int64_t expected, bool bypassL2)
{
    AscendC::LocalTensor<int64_t> local = GetSyncTensor();
    AscendC::GlobalTensor<int64_t> global;
    global.SetGlobalBuffer(source);
    // The L2-bypass alias is only valid for this rank's local window, where
    // it prevents a stale cache line after a peer updates the slot. Applying
    // it to an encoded peer VA makes MTE raise an invalid-GM-address error
    // on the cross-device access, so peer windows are polled with plain
    // MTE2 loads like the mature arch22 reference kernels.
    if (bypassL2) {
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    // MTE updates this UB storage asynchronously; each poll must issue a
    // fresh scalar load even when the loop has no other scalar side effects.
    while (true) {
        AscendC::DataCopy(local, global, GmmA2avMteDetail::SYNC_MTE_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        if (local.GetValue(0) == expected) {
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            break;
        }
    }
}

} // namespace MC2KernelTemplate

#endif // COMMON_GMMA2AV_MTE_SYNC_H
