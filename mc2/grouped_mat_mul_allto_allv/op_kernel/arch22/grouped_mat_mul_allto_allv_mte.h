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
 * \file grouped_mat_mul_allto_allv_mte.h
 * \brief AIV-driven MTE communication for GroupedMatMulAlltoAllvV2 on Ascend 910B/910_93.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "grouped_mat_mul_allto_allv_mte_tiling.h"

namespace MC2KernelTemplate {

namespace GmmA2avMteDetail {
constexpr uint32_t DEFAULT_UB_MOVE_ELEMENTS = Gmma2avMteTiling::UB_MOVE_ELEMENTS;
constexpr uint32_t USED_UB_BYTES = 192U * 1024U;
constexpr uint32_t CONTROL_UB_OFFSET_BYTES = 64U * 1024U;
constexpr uint32_t DESTINATION_OFFSET_UB_OFFSET_BYTES = 72U * 1024U;
constexpr uint32_t SOURCE_OFFSET_UB_OFFSET_BYTES = 80U * 1024U;
constexpr uint32_t PONG_UB_OFFSET_BYTES = 96U * 1024U;
constexpr uint32_t CONTROL_MTE_BYTES = 32U;
constexpr uint32_t CONTROL_MTE_ELEMENTS = CONTROL_MTE_BYTES / sizeof(int32_t);
constexpr uint32_t SYNC_MTE_ELEMENTS = CONTROL_MTE_BYTES / sizeof(int64_t);
// The control scratch spans [CONTROL_UB_OFFSET_BYTES, DESTINATION_OFFSET_UB_
// OFFSET_BYTES) = 8 KiB. Expert-ready staging reuses it in a disjoint phase
// from the peer-count staging, so the full region is available; one batched
// flag transaction may cover up to 128 consecutive 64-byte slots.
constexpr uint32_t FLAG_STAGING_ELEMENTS =
    (DESTINATION_OFFSET_UB_OFFSET_BYTES - CONTROL_UB_OFFSET_BYTES) / sizeof(int64_t);
constexpr uint32_t SYNC_SLOT_ELEMENTS = Gmma2avMteTiling::SYNC_SLOT_BYTES / sizeof(int64_t);
static_assert(CONTROL_MTE_BYTES % sizeof(int64_t) == 0U,
              "The synchronization transaction must contain complete int64 elements");
constexpr uint32_t INVALID_TOKEN_OFFSET = 0xffffffffU;
static_assert(CONTROL_MTE_BYTES % sizeof(int32_t) == 0U,
              "The control transaction must contain complete int32 elements");
static_assert(Gmma2avMteTiling::SYNC_SLOT_BYTES >= CONTROL_MTE_BYTES,
              "Every synchronization slot must hold one control transaction");
static_assert(Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES <= CONTROL_UB_OFFSET_BYTES,
              "The payload ping buffer must fit before the control scratch");
static_assert(SOURCE_OFFSET_UB_OFFSET_BYTES % CONTROL_MTE_BYTES == 0U,
              "The source-offset buffer must be control-transaction aligned");
static_assert(CONTROL_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(int32_t) <=
                  DESTINATION_OFFSET_UB_OFFSET_BYTES,
              "The peer count table must fit before the destination-offset buffer");
static_assert(DESTINATION_OFFSET_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(uint32_t) <=
                  SOURCE_OFFSET_UB_OFFSET_BYTES,
              "The destination-offset table must fit before the source-offset buffer");
static_assert(SOURCE_OFFSET_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(uint32_t) <=
                  PONG_UB_OFFSET_BYTES,
              "The source-offset table must fit before the payload pong buffer");
static_assert(PONG_UB_OFFSET_BYTES + Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES <= USED_UB_BYTES,
              "The payload pong buffer must fit in the allocated UB");
} // namespace GmmA2avMteDetail

template <typename T>
class GmmA2avMteOp {
public:
    __aicore__ inline GmmA2avMteOp() {}

    __aicore__ inline void Init(const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                                GM_ADDR recvBuffer, GM_ADDR cumsumBuffer, AscendC::TPipe *pipe,
                                const GmmA2avCoCTiling *cocTiling = nullptr, uint64_t commBufferSize = 0UL,
                                uint32_t isA3 = 0U)
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
        coreIdx_ = AscendC::GetBlockIdx();
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        logicalCoreIdx_ = coreIdx_ / AscendC::GetTaskRation();
        rankWorkerCount_ = static_cast<uint32_t>(taskTilingInfo_->aicCoreNum);
        AscendC::SetAtomicNone();
        if (cocTiling != nullptr && cocTiling->ubMoveNum > 0) {
            ubMoveElements_ = static_cast<uint32_t>(cocTiling->ubMoveNum);
        }

        // The first 64 KiB is the payload ping buffer. Count-table scratch is
        // staged at 64 KiB, destination offsets at 72 KiB, source offsets at
        // 80 KiB, and payload pong starts at 96 KiB.
        pipe->InitBuffer(ubBuffer_, GmmA2avMteDetail::USED_UB_BYTES);
    }

    __aicore__ inline void BeginExpertPipeline()
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
        } else {
            // 0/1/2 protocol, step 1: clear every sync slot in this rank's
            // own window before any peer can observe a new-round value. The
            // previous End's RECV_DONE rendezvous guarantees every peer
            // finished reading this window last round, so the clear cannot
            // race a peer's read. Starting from 0 makes each round's value-1
            // publish unambiguous (no stale 1/2 can be mistaken for this
            // round's 1) and also sanitises uninitialised window memory on
            // the first round.
            ResetSyncRegion();
            PublishLocalCountTable();
        }
        // Write COUNT_READY = 1 to all peers (send done), wait for all peers' 1.
        // A slot's 1 is never overwritten within the round: writing the
        // consume-done 2 into the same slot would let a slow peer miss the
        // transient 1 and spin forever.
        PublishAndWaitRankPhase(Gmma2avMteTiling::COUNT_READY_BASE, 1);
        if (!IsStageAiv()) {
            BuildDestinationOffsets();
        }
        LocalAivSync();
        localExpertBaseRows_ = 0UL;
        // Mirror the AIC scheduler's initial cross-expert rotation state.
        expertStartCoreIdx_ = 0U;
    }

    // Stage path: copy one expert's locally owned M tiles from the GMM
    // workspace into this rank's window. The C2V notification for this
    // expert has already been consumed by the scheduler, so every row read
    // here is visible. This path never waits on a peer, pulls payload, or
    // joins any rendezvous that requires the Comm subblock.
    __aicore__ inline void StageExpertPayload(uint32_t startExpertIdx, uint32_t endExpertIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        PublishLocalChunk(startExpertIdx, endExpertIdx);
    }

    // Publish the Stage→Comm completion progress for this producer core.
    // Called only for experts whose M tiles this core actually staged
    // (scheduler-gated on HasExpertWork). Release ordering: CopyGmToGm
    // completes the payload MTE3 writes with a DDR barrier before returning,
    // and StoreSyncFlag ends with its own DDR barrier, so the payload is
    // globally visible before the progress value is.
    __aicore__ inline void PublishStageProgress(uint32_t expertIdx)
    {
        if (!IsStageAiv()) {
            return;
        }
        StoreSyncFlag(GetStageProgressSlot(logicalCoreIdx_), static_cast<int64_t>(expertIdx) + 1);
    }

    // Cross-rank protocol for one expert, executed by the Comm subblocks
    // only. Order per expert: wait every local producer's stage progress,
    // publish ExpertReady to all owned destinations, then wait owned
    // sources and pull. Every wait has an independent publisher:
    // ExpertReady publication depends only on the publisher rank's local
    // stage progress, so worker progress skew cannot form a publish/wait
    // cycle. No all-AIV rendezvous exists in this path.
    __aicore__ inline void CommExpertChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
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

    __aicore__ inline void PublishChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx)
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

    __aicore__ inline void WaitChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx)
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

    __aicore__ inline void PullExpertChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
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
        // No consume-done write-back here: overwriting the source's 1 with 2
        // in the source's window would let a slow source miss the transient
        // 1 and spin forever. Consumption is confirmed once for the whole
        // round by End's RECV_DONE rendezvous instead, which every peer
        // waits for before the next invocation reuses any window data.
        // End's leading all-AIV barrier is also what orders this rank's last
        // pull before the RECV_DONE publication, so no per-expert trailing
        // rendezvous is needed.
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        (void)startExpertIdx;
    }

    // Round completion. RECV_DONE is the one remaining value-2 write and it
    // has a reader: every peer waits for it, so the 2 can never overwrite a
    // value someone still polls for. After this rendezvous every peer has
    // consumed this round's data, which is what makes the next round's
    // ResetSyncRegion (and the count-table/payload reuse) safe.
    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            PublishAndWaitRankPhase(Gmma2avMteTiling::RECV_DONE_BASE, 2);
        }
        AscendC::SyncAll<false>();
    }

public:
    // Role split inside the 1:2 AIC:AIV pairing. mode=0x2 C2V
    // notifications are delivered to BOTH AIV subblocks of the AI core, so
    // both must consume every notification to keep the FFTS counters
    // aligned; the split therefore happens in the work performed after the
    // notification, not in the notification itself.
    __aicore__ inline bool IsStageAiv() const
    {
        return subBlockIdx_ == 0U;
    }

    __aicore__ inline bool IsCommAiv() const
    {
        return subBlockIdx_ != 0U;
    }

    // Mirror of the AIC-side ownership decision: true when the AIC paired
    // with this AIV logical core owns at least one M tile of the expert.
    // Reads the same rotation state PublishLocalChunk advances, so it must
    // be queried before staging the expert it describes. Both subblocks of
    // a logical core derive the same answer and thus keep their flag waits
    // aligned.
    __aicore__ inline bool HasExpertWork(uint32_t expertIdx) const
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

private:
    __aicore__ inline GM_ADDR GetWindow(uint32_t rank) const
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

    __aicore__ inline __gm__ int32_t *GetCountTable(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ int32_t *>(GetWindow(rank) + winSize_ -
                                                  Gmma2avMteTiling::SEND_COUNT_STAGING_FROM_TAIL);
    }

    __aicore__ inline __gm__ T *GetOutputData(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ T *>(GetWindow(rank));
    }

    __aicore__ inline void PublishLocalCountTable()
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

    // Stage one expert's rows from the GMM output workspace into this rank's
    // window. Ownership follows the AIC-side M-row striped scheduling: the
    // AIC paired with logical core c computed exactly the tiles mIdx with
    // (mIdx - startCoreIdx_e) mod aicCoreNum == c, so this AIV copies only
    // those tiles. Both sides replay the same cross-expert rotation that
    // advances by ceil(rows_e / L1_TILE_M), keeping tile ownership identical
    // without any tiling-table extension.
    __aicore__ inline void PublishLocalChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        const uint32_t aicCoreNum = rankWorkerCount_;
        const uint32_t tileRows = Gmma2avMteTiling::L1_TILE_M;
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
            // Mirror the AIC's GetExpertTokens exactly (raw sum, no filter):
            // the tile count and the rotation must match the producer side.
            uint64_t expertRows = 0UL;
            for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                expertRows += static_cast<uint64_t>(taskTilingInfo_->sendCnt[dstRank * expertNum + expertIdx]);
            }
            const uint32_t loopsM = static_cast<uint32_t>((expertRows + tileRows - 1UL) / tileRows);
            if (aicCoreNum != 0U && expertRows > 0UL) {
                const uint32_t startMIdx = (logicalCoreIdx_ + aicCoreNum - expertStartCoreIdx_) % aicCoreNum;
                for (uint32_t mIdx = startMIdx; mIdx < loopsM; mIdx += aicCoreNum) {
                    const uint64_t tileRowStart = static_cast<uint64_t>(mIdx) * tileRows;
                    const uint64_t remaining = expertRows - tileRowStart;
                    const uint64_t tileRowCount = remaining > tileRows ? tileRows : remaining;
                    const uint64_t sourceRow = localExpertBaseRows_ + tileRowStart;
                    CopyGmToGm(sendBuffer_ + sourceRow * taskTilingInfo_->N1,
                               GetOutputData(rank_) + sourceRow * taskTilingInfo_->N1,
                               tileRowCount * taskTilingInfo_->N1);
                }
            }
            localExpertBaseRows_ += expertRows;
            expertStartCoreIdx_ = (expertStartCoreIdx_ + loopsM) % aicCoreNum;
        }
    }

    __aicore__ inline void BuildDestinationOffsets()
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

    __aicore__ inline void BuildPeerSourceOffsets()
    {
        if (IsStageAiv() || rankWorkerCount_ == 0U) {
            return;
        }
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        const uint32_t countNum = rankSize_ * expertNum;
        const uint64_t payloadCapacityBytes = winSize_ - Gmma2avMteTiling::CONTROL_REGION_BYTES;
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

    // Local (non-CCL) stage-progress protocol. Slot c is written ONLY by
    // the Stage subblock of logical core c through a 32-byte MTE3
    // transaction that ends with a DDR barrier; readers use 32-byte MTE2
    // transactions with the L2 bypass hint (see WaitStageProgress). Both
    // widths are naturally aligned inside one 64-byte cache line, and the
    // producer's DDR barrier after the payload copy plus the reader's DDR
    // barrier after observing the value provide the release/acquire
    // ordering.
    __aicore__ inline __gm__ int64_t *GetStageProgressSlot(uint32_t producerCore) const
    {
        return stageProgressBase_ +
               static_cast<uint64_t>(producerCore) *
                   (Gmma2avMteTiling::STAGE_PROGRESS_SLOT_BYTES / sizeof(int64_t));
    }

    __aicore__ inline void WaitStageProgress(uint32_t producerCore, int64_t boundary)
    {
        AscendC::LocalTensor<int64_t> local = GetSyncTensor();
        AscendC::GlobalTensor<int64_t> global;
        global.SetGlobalBuffer(GetStageProgressSlot(producerCore));
        // The slot is local GM written by the Stage subblock of another core
        // (possibly on another die). A plain MTE2 poll can keep hitting a
        // stale line in this die's L2 that the producer's DDR-barriered write
        // never invalidates, hanging the Comm role intermittently. Use the
        // same L2-bypass policy as every other local-slot poll in this file
        // (valid for local GM VAs only, never for encoded peer VAs).
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (true) {
            AscendC::DataCopy(local, global, GmmA2avMteDetail::SYNC_MTE_ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            // >= rather than ==: the producer publishes monotonic boundaries,
            // so a later expert's value can legitimately overwrite the one
            // being waited for and still proves this expert is staged.
            if (local.GetValue(0U) >= boundary) {
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                break;
            }
        }
    }

    // Wait until every producer that owns M tiles of the expert range has
    // staged them, replaying the same cross-expert rotation the AIC
    // scheduler and PublishLocalChunk use. Cores without tiles for an
    // expert never write progress for it and are never waited on. The
    // rotation advance below keeps this instance's HasExpertWork answers
    // identical to the Stage subblock's.
    __aicore__ inline void WaitLocalStageProgress(uint32_t startExpertIdx, uint32_t endExpertIdx)
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

    __aicore__ inline void PullPeerExpertFromSource(uint32_t srcRank, uint32_t expertIdx)
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

    __aicore__ inline AscendC::LocalTensor<int32_t> GetControlTensor()
    {
        return ubBuffer_.template Get<int32_t>()[GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int32_t)];
    }

    __aicore__ inline AscendC::LocalTensor<int64_t> GetSyncTensor()
    {
        return ubBuffer_.template Get<int64_t>()[GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int64_t)];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetDestinationOffsetTensor()
    {
        return ubBuffer_
            .template Get<uint32_t>()[GmmA2avMteDetail::DESTINATION_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetSourceOffsetTensor()
    {
        return ubBuffer_.template Get<uint32_t>()[GmmA2avMteDetail::SOURCE_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
    }

    __aicore__ static inline bool SafeAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
    {
        constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
        if (lhs > maxValue - rhs) {
            return false;
        }
        result = lhs + rhs;
        return true;
    }

    __aicore__ static inline bool SafeMulU32(uint64_t lhs, uint64_t rhs, uint64_t &result)
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

    __aicore__ inline void LoadInt32Row(__gm__ int32_t *source, const AscendC::LocalTensor<int32_t> &local,
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

    __aicore__ inline void StoreInt32Row(__gm__ int32_t *destination, const AscendC::LocalTensor<int32_t> &local,
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

    __aicore__ inline void LocalAivSync()
    {
        // On arch22 SyncAll<true> drains local pipelines and performs an
        // AIV-only inter-core rendezvous. After the Stage/Comm role split
        // this is used ONLY at common initialization (BeginExpertPipeline)
        // and common exit (End), which both roles reach; the per-expert
        // loops contain no all-AIV rendezvous.
        AscendC::SyncAll<true>();
    }

    // Single-writer rendezvous (mirrors matmul_allto_all's CrossRankSyncV1):
    // each rank publishes ONE state slot in its OWN window and every peer
    // polls that slot in the publisher's window. Because a window's slots
    // are written only by their owner, the Begin-of-round ResetSyncRegion can
    // never race a peer's publish, and no published value is ever overwritten
    // while a peer still polls for it.
    __aicore__ inline void PublishAndWaitRankPhase(uint64_t phaseBase, int64_t flag)
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

    // Publish-only variant removed: writing the consume-done value 2 into the
    // same slot that peers poll for 1 let slow peers miss the transient 1.
    // The 0/1/2 round now relies on Begin's ResetSyncRegion plus End's
    // RECV_DONE rendezvous instead of an in-round consume-done write-back.

    __aicore__ inline __gm__ int64_t *GetFixedSyncSlot(uint32_t windowRank, uint64_t phaseBase, uint64_t index) const
    {
        const uint64_t syncOffset =
            winSize_ - Gmma2avMteTiling::SYNC_REGION_FROM_TAIL + phaseBase + index * Gmma2avMteTiling::SYNC_SLOT_BYTES;
        return reinterpret_cast<__gm__ int64_t *>(GetWindow(windowRank) + syncOffset);
    }

    __aicore__ inline __gm__ int64_t *GetExpertReadySlot(uint32_t windowRank, uint32_t sourceRank,
                                                         uint32_t expertIdx) const
    {
        const uint64_t expertSlot = static_cast<uint64_t>(sourceRank) * taskTilingInfo_->e + expertIdx;
        return GetFixedSyncSlot(windowRank, Gmma2avMteTiling::EXPERT_READY_BASE, expertSlot);
    }

    // sendCnt/recvCnt come from the same tiling computation on both ends, so
    // a skipped publish slot is exactly the slot the peer skips waiting on:
    // publish is needed only when this rank actually sends expert e to D, and
    // waiting is needed only when this rank actually receives expert e from
    // S. Expert 0 is always kept unmasked on both sides: its flag is the
    // release/acquire handshake for this rank's count table, which
    // BuildPeerSourceOffsets loads even when no expert-0 rows are exchanged.
    // The self slot is always kept because the local wait doubles as the
    // all-AIV rendezvous for this expert.
    __aicore__ inline bool IsExpertSlotActive(const int32_t *countTable, uint32_t peerRank, uint32_t expertIdx,
                                              uint32_t expertNum) const
    {
        if (expertIdx == 0U || peerRank == rank_) {
            return true;
        }
        return countTable[static_cast<uint64_t>(peerRank) * expertNum + expertIdx] > 0;
    }

    // Consume-done write-back removed: writing 2 over a source's published 1
    // in the source's window could let a slow source miss the transient 1
    // and spin forever. Round-level consumption is confirmed by End's
    // RECV_DONE rendezvous, which precedes any window reuse.

    // Clear the whole fixed sync region (COUNT_READY + EXPERT_READY +
    // RECV_DONE) in this rank's own window at the start of every round.
    // Uses batched MTE3 zero-writes; a full clear covers persistent windows
    // that may have been dirtied by previous rounds with different shapes.
    __aicore__ inline void ResetSyncRegion()
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
        const uint32_t totalSlots = static_cast<uint32_t>(Gmma2avMteTiling::FIXED_SYNC_BYTES /
                                                          Gmma2avMteTiling::SYNC_SLOT_BYTES);
        for (uint32_t offset = 0U; offset < totalSlots; offset += slotsPerBatch) {
            const uint32_t batchSlots = (totalSlots - offset > slotsPerBatch) ?
                                            slotsPerBatch : (totalSlots - offset);
            const uint32_t batchElements = batchSlots * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS;
            AscendC::GlobalTensor<int64_t> global;
            global.SetGlobalBuffer(GetFixedSyncSlot(rank_, 0UL, offset));
            AscendC::DataCopy(global, staging, batchElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
    }

    // One batched MTE3 transaction publishes every expert slot of one chunk
    // for one destination rank, replacing one 32-byte store plus DDR barrier
    // per active expert. Inactive slots are written as zero; they are never
    // polled this invocation (the peer's wait mask is identical), and the
    // next invocation's publisher overwrites them before any waiter observes.
    __aicore__ inline bool PublishExpertReadyRange(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
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

    __aicore__ inline void PublishExpertReadyPerFlag(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
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

    // One batched MTE2 transaction polls every expert slot of one chunk from
    // one source rank. A single DDR barrier after all active slots hold the
    // expected value replaces the per-flag barrier, so a ready chunk costs
    // one MTE round trip instead of one per active (expert, source) pair.
    __aicore__ inline bool WaitExpertReadyRange(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
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

    __aicore__ inline void WaitExpertReadyPerFlag(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
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

    __aicore__ inline void StoreSyncFlag(__gm__ int64_t *destination, int64_t flag)
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

    __aicore__ inline bool IsPrimaryAiv() const
    {
        // The Comm subblock of logical core 0 publishes this rank's count
        // table and the rank-level phase flags.
        return logicalCoreIdx_ == 0U && IsCommAiv();
    }

    __aicore__ inline void WaitSyncFlag(__gm__ int64_t *source, int64_t expected, bool bypassL2)
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
        while (true) {
            AscendC::DataCopy(local, global, GmmA2avMteDetail::SYNC_MTE_ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            if (local.GetValue(0U) == expected) {
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                break;
            }
        }
    }

    __aicore__ inline uint32_t AlignControlElements(uint32_t elementCount) const
    {
        constexpr uint32_t alignment = GmmA2avMteDetail::CONTROL_MTE_ELEMENTS;
        return (elementCount + alignment - 1U) / alignment * alignment;
    }

    __aicore__ inline void CopyGmToGm(__gm__ T *source, __gm__ T *destination, uint64_t elementCount)
    {
        uint32_t moveIdx = 0U;
        BeginGmToGmCopy();
        CopyGmToGmSegment(source, destination, elementCount, moveIdx);
        EndGmToGmCopy();
    }

    __aicore__ inline void BeginGmToGmCopy()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void CopyGmToGmSegment(__gm__ T *source, __gm__ T *destination, uint64_t elementCount,
                                             uint32_t &moveIdx)
    {
        AscendC::LocalTensor<T> local = ubBuffer_.template Get<T>();
        AscendC::LocalTensor<T> ping = local;
        AscendC::LocalTensor<T> pong = local[GmmA2avMteDetail::PONG_UB_OFFSET_BYTES / sizeof(T)];
        uint64_t copied = 0UL;
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

    __aicore__ inline void EndGmToGmCopy()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

private:
    const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_{nullptr};
    __gm__ HcclOpResParam *a3WinContext_{nullptr};
    __gm__ AscendC::HcclCombineOpParam *a2WinContext_{nullptr};
    __gm__ T *sendBuffer_{nullptr};
    __gm__ T *recvBuffer_{nullptr};
    // Base of the local stage-progress region (one 64-byte slot per AIC
    // logical core, appended after the cumsum region in user workspace).
    // Never overlaps the cross-card sync mailbox in the CCL window.
    __gm__ int64_t *stageProgressBase_{nullptr};
    uint64_t winSize_{0UL};
    uint64_t localExpertBaseRows_{0UL};
    // AIV-side replay of the AIC scheduler's cross-expert first-core
    // rotation; both sides advance it by ceil(rows_e / L1_TILE_M).
    uint32_t expertStartCoreIdx_{0U};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    bool isA3_{false};
    uint32_t coreIdx_{0U};
    uint32_t subBlockIdx_{0U};
    uint32_t logicalCoreIdx_{0U};
    uint32_t rankWorkerCount_{0U};
    uint32_t ubMoveElements_{GmmA2avMteDetail::DEFAULT_UB_MOVE_ELEMENTS};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuffer_;
};

template <typename CommOpType, typename ComputationOpType, typename SharedComputationOpType, bool IsNeedMM>
class GmmA2avMteScheduler {
private:
    // A3 CrossCore set-count allows at most 15 outstanding notifications.
    // Keep this explicit: the generic CATLASS default is 16 on this branch.
    using ExpertReadyFlag = Catlass::Arch::CrossCoreFlagWithReverse<15U>;

    struct AicExpertChunkReadyCallback {
        __aicore__ inline AicExpertChunkReadyCallback(ExpertReadyFlag &flagEven, ExpertReadyFlag &flagOdd,
                                                      const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
            : flagEven_(flagEven),
              flagOdd_(flagOdd),
              taskTilingInfo_(taskTilingInfo)
        {}

        // Every expert is a notification boundary: the paired AIV can start
        // staging this expert's output while the AIC keeps computing the
        // next one. The workspace holds the whole grouped-matmul result, so
        // the AIC never has to wait for the AIV to finish copying. Only
        // experts whose tiles this AIC actually computed produce a
        // notification; the paired AIV skips the matching wait.
        __aicore__ inline bool operator()(uint32_t, bool hasWork) const
        {
            return hasWork;
        }

        // Direct C2V notification with no all-AIC rendezvous. Ping-pong by
        // this core's valid-notification sequence number: after skipping
        // experts without work, consecutive valid notifications may all come
        // from even or all from odd expert indices, so expertIdx parity is
        // not a valid channel selector. Both sides start from zero and
        // advance only on hasWork, keeping the channel sequence identical.
        __aicore__ inline void NotifyExpertReady()
        {
            if (validNotifyCount_ & 1U) {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagOdd_);
            } else {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagEven_);
            }
            ++validNotifyCount_;
        }

        ExpertReadyFlag &flagEven_;
        ExpertReadyFlag &flagOdd_;
        const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_;
        uint32_t validNotifyCount_{0U};
    };

public:
    static __aicore__ inline void ProcessAic(ComputationOpType &computeOp, SharedComputationOpType &sharedComputeOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
    {
        if ASCEND_IS_AIC {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            ExpertReadyFlag flagEven(EXPERT_READY_FLAG_EVEN, EXPERT_REVERSE_FLAG_EVEN);
            ExpertReadyFlag flagOdd(EXPERT_READY_FLAG_ODD, EXPERT_REVERSE_FLAG_ODD);
            AicExpertChunkReadyCallback callback(flagEven, flagOdd, taskTilingInfo);
            computeOp.ProcessExperts(0U, expertNum, callback);

            if constexpr (IsNeedMM) {
                sharedComputeOp.Process(0U, 1U);
            }
            AscendC::SyncAll<false>();
        }
    }

    static __aicore__ inline void ProcessAiv(CommOpType &commOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
    {
        if ASCEND_IS_AIV {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            ExpertReadyFlag flagEven(EXPERT_READY_FLAG_EVEN, EXPERT_REVERSE_FLAG_EVEN);
            ExpertReadyFlag flagOdd(EXPERT_READY_FLAG_ODD, EXPERT_REVERSE_FLAG_ODD);
            commOp.BeginExpertPipeline();
            // Stage/Comm role split. mode=0x2 delivers every AIC notification
            // to BOTH AIV subblocks of the AI core, so both consume the same
            // notification sequence (skipping either side's wait would
            // desynchronise the FFTS counters and stall the AIC). The split
            // is in the work after the notification: subblock 0 (Stage)
            // consumes the notification, copies its M tiles into the window
            // and publishes its local progress — its path never waits on a
            // peer. Subblock 1 (Comm) consumes the notification and drives
            // the cross-rank protocol from that local progress. HasExpertWork
            // is derived from the same rotation state on both subblocks, so
            // the wait sequences stay identical.
            uint32_t validNotifyCount = 0U;
            for (uint32_t expertIdx = 0U; expertIdx < expertNum; ++expertIdx) {
                const bool hasWork = commOp.HasExpertWork(expertIdx);
                if (hasWork) {
                    if (validNotifyCount & 1U) {
                        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagOdd);
                    } else {
                        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagEven);
                    }
                    ++validNotifyCount;
                }
                if (commOp.IsStageAiv()) {
                    commOp.StageExpertPayload(expertIdx, expertIdx + 1U);
                    if (hasWork) {
                        commOp.PublishStageProgress(expertIdx);
                    }
                } else {
                    commOp.CommExpertChunk(expertIdx, expertIdx + 1U);
                }
            }

            commOp.End();
        }
    }

private:
    // Ping-pong flag IDs: even experts use IDs {0,1}, odd experts use {2,3}.
    // This prevents FFTS double-set traps when notifying per-expert.
    static constexpr Catlass::Arch::FlagID EXPERT_READY_FLAG_EVEN = 0U;
    static constexpr Catlass::Arch::FlagID EXPERT_REVERSE_FLAG_EVEN = 1U;
    static constexpr Catlass::Arch::FlagID EXPERT_READY_FLAG_ODD = 2U;
    static constexpr Catlass::Arch::FlagID EXPERT_REVERSE_FLAG_ODD = 3U;
    static_assert(EXPERT_READY_FLAG_EVEN <= 15U && EXPERT_REVERSE_FLAG_EVEN <= 15U &&
                  EXPERT_READY_FLAG_ODD <= 15U && EXPERT_REVERSE_FLAG_ODD <= 15U,
                  "arch22 local CrossCore flag IDs must be in [0, 15]");
    static_assert(EXPERT_READY_FLAG_EVEN != EXPERT_REVERSE_FLAG_EVEN &&
                  EXPERT_READY_FLAG_ODD != EXPERT_REVERSE_FLAG_ODD &&
                  EXPERT_READY_FLAG_EVEN != EXPERT_READY_FLAG_ODD,
                  "expert ready and reverse-credit flags must be distinct");
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
