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

    __aicore__ inline void Init(const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                GM_ADDR sendBuffer, GM_ADDR recvBuffer, GM_ADDR cumsumBuffer,
                                AscendC::TPipe *pipe,
                                const GmmA2avCoCTiling *cocTiling = nullptr,
                                uint64_t commBufferSize = 0UL,
                                uint32_t isA3 = 0U)
    {
        taskTilingInfo_ = taskTilingInfo;
        sendBuffer_ = reinterpret_cast<__gm__ T *>(sendBuffer);
        recvBuffer_ = reinterpret_cast<__gm__ T *>(recvBuffer);
        // Kept in the common MTE Init signature for serialized-tiling ABI
        // compatibility. Expert-overlap offsets are owner-local in UB.
        (void)cumsumBuffer;
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

    __aicore__ inline void LaunchCommData(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        BeginInvocation();
        PublishLocalCountTable();
        PublishLocalPayload();

        // Bulk-ready protects both the complete payload and the count table.
        // Its fixed slot region is deliberately independent of runtime W/E.
        PublishAndWaitRankPhase(Gmma2avMteTiling::BULK_READY_BASE);
        PullPeerPayloadBulk(startExpertIdx, expertNum);
    }

    __aicore__ inline void BeginExpertPipeline()
    {
        if ASCEND_IS_AIC {
            return;
        }
        BeginInvocation();

        // Counts do not depend on any GMM output, so publish them while AIC is
        // computing expert 0. ExpertReady(..., 0) supplies their remote
        // release/acquire handshake before a receiver loads the table.
        PublishLocalCountTable();
        LocalAivSync();
        BuildDestinationOffsets();
        LocalAivSync();
        localExpertBaseRows_ = 0UL;
    }

    __aicore__ inline void LaunchExpert(uint32_t expertIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }

        // Each AIV only acquires completion from its paired AIC.  The CATLASS
        // scheduler rotates expert tiles across AICs, while payload rows are
        // redistributed independently across AIVs; consequently the AIV that
        // copies a row is not necessarily paired with the AIC that produced
        // it.  Match mega_moe A3's C2V hand-off: after every paired wait, all
        // AIVs rendezvous before any of them reads the shared GMM workspace.
        LocalAivSync();
        PublishLocalExpert(expertIdx);
        PublishAndWaitExpertReady(expertIdx);
        if (expertIdx == 0U) {
            // The same worker that acquired each source's ready flag loads
            // that source's count table and remains its payload-pull owner.
            BuildPeerSourceOffsets();
            LocalAivSync();
        }
        PullPeerExpert(expertIdx);
        LocalAivSync();
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        (void)startExpertIdx;
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            // Every payload-pull worker has drained its MTE2/MTE3 pipeline
            // before the entry barrier below. Completion protects source
            // windows; acknowledgement protects all persistent phase slots
            // against reuse by the next invocation.
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            PublishAndWaitRankPhase(Gmma2avMteTiling::COMPLETION_BASE);
            PublishAndWaitRankPhase(Gmma2avMteTiling::ACK_BASE);
        }
        AscendC::SyncAll<false>();
    }

private:
    __aicore__ inline GM_ADDR GetWindow(uint32_t rank) const
    {
        if (isA3_) {
            if (rank == rank_) {
                return (GM_ADDR)(a3WinContext_->localWindowsIn);
            }
            auto relation = reinterpret_cast<__gm__ HcclRankRelationResV2 *>(
                a3WinContext_->remoteRes[rank].nextDevicePtr);
            return (GM_ADDR)(relation->windowsIn);
        }
        if (a2WinContext_->multiFlag == 0U) {
            return (GM_ADDR)(a2WinContext_->windowsIn[rank]);
        }
        return rank == rank_ ?
            (GM_ADDR)(a2WinContext_->data[rank].localInput.addr) :
            (GM_ADDR)(a2WinContext_->data[rank].remoteInput.addr);
    }

    __aicore__ inline __gm__ int32_t *GetCountTable(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ int32_t *>(
            GetWindow(rank) + winSize_ - Gmma2avMteTiling::SEND_COUNT_STAGING_FROM_TAIL);
    }

    __aicore__ inline __gm__ T *GetOutputData(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ T *>(GetWindow(rank));
    }

    __aicore__ inline void BeginInvocation()
    {
        // HCCL windows persist across cases. The control plane must provide a
        // zero-initialized fixed-layout region on first use and serialize all
        // invocations on this communicator. Under that contract the complete
        // acknowledgement vector is the ownership proof for advancing epoch.
        LocalAivSync();
        if (IsPrimaryAiv()) {
            const int32_t previousEpoch = AscendC::ReadGmByPassDCache(GetEpochSlot(rank_));
            for (uint32_t participant = 0U; participant < rankSize_; ++participant) {
                WaitSyncFlag(GetAckSlot(rank_, participant), previousEpoch);
            }
            // Admission guarantees 0 <= previousEpoch < MAX_SYNC_EPOCH. This
            // hot kernel intentionally has no unilateral wrap/recovery path;
            // the executor must quiesce and reinitialize before exhaustion.
            syncEpoch_ = previousEpoch == 0 ? 1 : previousEpoch + 1;
            StoreSyncFlag(GetEpochSlot(rank_), syncEpoch_);
        }
        LocalAivSync();
        if (!IsPrimaryAiv()) {
            syncEpoch_ = AscendC::ReadGmByPassDCache(GetEpochSlot(rank_));
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
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

    __aicore__ inline void PublishLocalPayload()
    {
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
            return;
        }
        uint64_t totalRows = 0UL;
        const uint32_t countNum = rankSize_ * static_cast<uint32_t>(taskTilingInfo_->e);
        for (uint32_t index = 0U; index < countNum; ++index) {
            const int32_t count = taskTilingInfo_->sendCnt[index];
            if (count > 0) {
                totalRows += static_cast<uint64_t>(count);
            }
        }
        const uint64_t rowsPerWorker = (totalRows + rankWorkerCount_ - 1UL) / rankWorkerCount_;
        const uint64_t rowStart = static_cast<uint64_t>(logicalCoreIdx_) * rowsPerWorker;
        if (rowStart < totalRows) {
            const uint64_t rowCount = totalRows - rowStart > rowsPerWorker ?
                rowsPerWorker : totalRows - rowStart;
            CopyGmToGm(sendBuffer_ + rowStart * taskTilingInfo_->N1,
                       GetOutputData(rank_) + rowStart * taskTilingInfo_->N1,
                       rowCount * taskTilingInfo_->N1);
        }
    }

    __aicore__ inline void PublishLocalExpert(uint32_t expertIdx)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        uint64_t expertRows = 0UL;
        for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
            const int32_t count = taskTilingInfo_->sendCnt[static_cast<uint64_t>(dstRank) * expertNum + expertIdx];
            if (count > 0) {
                expertRows += static_cast<uint64_t>(count);
            }
        }

        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            const uint64_t rowsPerWorker = (expertRows + rankWorkerCount_ - 1UL) / rankWorkerCount_;
            const uint64_t workerRowStart = static_cast<uint64_t>(logicalCoreIdx_) * rowsPerWorker;
            if (workerRowStart < expertRows) {
                const uint64_t rowCount =
                    expertRows - workerRowStart > rowsPerWorker ? rowsPerWorker : expertRows - workerRowStart;
                const uint64_t sourceRow = localExpertBaseRows_ + workerRowStart;
                CopyGmToGm(sendBuffer_ + sourceRow * taskTilingInfo_->N1,
                           GetOutputData(rank_) + sourceRow * taskTilingInfo_->N1, rowCount * taskTilingInfo_->N1);
            }
        }
        // All AIVs execute the same expert sequence, including empty experts.
        localExpertBaseRows_ += expertRows;
    }

    __aicore__ inline void BuildDestinationOffsets()
    {
        if (subBlockIdx_ != 0U) {
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
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
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

    __aicore__ inline void PullPeerExpert(uint32_t expertIdx)
    {
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
            return;
        }
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        AscendC::LocalTensor<uint32_t> destinationOffsets = GetDestinationOffsetTensor();
        AscendC::LocalTensor<uint32_t> sourceOffsets = GetSourceOffsetTensor();
        for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
            const uint64_t index = static_cast<uint64_t>(srcRank) * expertNum + expertIdx;
            const int32_t recvCount = taskTilingInfo_->recvCnt[index];
            const uint32_t sourceOffset = sourceOffsets.GetValue(index);
            const uint32_t destinationOffset = destinationOffsets.GetValue(index);
            if (recvCount <= 0 || sourceOffset == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                destinationOffset == GmmA2avMteDetail::INVALID_TOKEN_OFFSET) {
                continue;
            }
            CopyGmToGm(GetOutputData(srcRank) + static_cast<uint64_t>(sourceOffset) * taskTilingInfo_->N1,
                       recvBuffer_ + static_cast<uint64_t>(destinationOffset) * taskTilingInfo_->N1,
                       static_cast<uint64_t>(recvCount) * taskTilingInfo_->N1);
        }
    }

    __aicore__ inline void PullPeerPayloadBulk(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
            return;
        }
        const uint32_t localExpertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        const uint32_t remainingExpertNum =
            startExpertIdx < localExpertNum ? localExpertNum - startExpertIdx : 0U;
        const uint32_t endExpertIdx =
            startExpertIdx + (expertNum > remainingExpertNum ? remainingExpertNum : expertNum);
        const uint32_t countNum = rankSize_ * localExpertNum;
        AscendC::LocalTensor<int32_t> peerCounts = GetControlTensor();
        for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
            if (srcRank == rank_) {
                for (uint32_t index = 0U; index < countNum; ++index) {
                    peerCounts.SetValue(index, taskTilingInfo_->sendCnt[index]);
                }
            } else {
                LoadInt32Row(GetCountTable(srcRank), peerCounts, countNum);
            }

            uint64_t destinationTokenOffset = 0UL;
            bool destinationLayoutValid = true;
            for (uint32_t priorSource = 0U; priorSource < srcRank; ++priorSource) {
                for (uint32_t expert = 0U; expert < localExpertNum; ++expert) {
                    const int32_t count = taskTilingInfo_->recvCnt[
                        static_cast<uint64_t>(priorSource) * localExpertNum + expert];
                    uint64_t updated = 0UL;
                    if (count < 0 || !SafeAdd(destinationTokenOffset, static_cast<uint64_t>(count), updated)) {
                        destinationLayoutValid = false;
                    } else {
                        destinationTokenOffset = updated;
                    }
                }
            }

            uint64_t expertSourceBase = 0UL;
            bool sourceLayoutValid = true;
            for (uint32_t expert = 0U; expert < localExpertNum; ++expert) {
                uint64_t sourceOffsetInExpert = 0UL;
                uint64_t expertTotal = 0UL;
                bool expertLayoutValid = sourceLayoutValid && destinationLayoutValid;
                for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                    const int32_t count = peerCounts.GetValue(
                        static_cast<uint64_t>(dstRank) * localExpertNum + expert);
                    uint64_t updated = 0UL;
                    if (count < 0) {
                        expertLayoutValid = false;
                        continue;
                    }
                    if (dstRank < rank_ &&
                        !SafeAdd(sourceOffsetInExpert, static_cast<uint64_t>(count), updated)) {
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

                const uint64_t countIndex = static_cast<uint64_t>(rank_) * localExpertNum + expert;
                const int32_t peerSendCount = peerCounts.GetValue(countIndex);
                const int32_t localRecvCount = taskTilingInfo_->recvCnt[
                    static_cast<uint64_t>(srcRank) * localExpertNum + expert];
                uint64_t sourceOffset = 0UL;
                uint64_t sourceEndRows = 0UL;
                uint64_t sourceEndElements = 0UL;
                uint64_t destinationEndRows = 0UL;
                bool copyValid = expertLayoutValid && peerSendCount >= 0 && localRecvCount >= 0 &&
                                 peerSendCount == localRecvCount &&
                                 SafeAdd(expertSourceBase, sourceOffsetInExpert, sourceOffset) &&
                                 SafeAdd(sourceOffset, static_cast<uint64_t>(peerSendCount), sourceEndRows) &&
                                 SafeMulU32(sourceEndRows, static_cast<uint64_t>(taskTilingInfo_->N1),
                                            sourceEndElements) &&
                                 sourceEndElements <=
                                     (winSize_ - Gmma2avMteTiling::CONTROL_REGION_BYTES) / sizeof(T) &&
                                 SafeAdd(destinationTokenOffset, static_cast<uint64_t>(localRecvCount),
                                         destinationEndRows) &&
                                 destinationEndRows <= taskTilingInfo_->BSK;
                if (expert >= startExpertIdx && expert < endExpertIdx && peerSendCount > 0 && copyValid) {
                    // Preserve the published BULK baseline: self-rank data is
                    // read directly from the GMM workspace, while remote data
                    // is pulled from the peer HCCL window.
                    __gm__ T *source = srcRank == rank_ ? sendBuffer_ : GetOutputData(srcRank);
                    CopyGmToGm(source + sourceOffset * taskTilingInfo_->N1,
                               recvBuffer_ + destinationTokenOffset * taskTilingInfo_->N1,
                               static_cast<uint64_t>(peerSendCount) * taskTilingInfo_->N1);
                }

                if (!expertLayoutValid || !SafeAdd(expertSourceBase, expertTotal, expertSourceBase)) {
                    sourceLayoutValid = false;
                }
                if (localRecvCount < 0 ||
                    !SafeAdd(destinationTokenOffset, static_cast<uint64_t>(localRecvCount), destinationTokenOffset)) {
                    destinationLayoutValid = false;
                }
            }
        }
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> GetControlTensor()
    {
        return ubBuffer_.template Get<int32_t>()[
            GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int32_t)];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetDestinationOffsetTensor()
    {
        return ubBuffer_
            .template Get<uint32_t>()[GmmA2avMteDetail::DESTINATION_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetSourceOffsetTensor()
    {
        return ubBuffer_
            .template Get<uint32_t>()[GmmA2avMteDetail::SOURCE_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
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

    __aicore__ inline void LoadInt32Row(__gm__ int32_t *source,
                                        const AscendC::LocalTensor<int32_t> &local,
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

    __aicore__ inline void StoreInt32Row(__gm__ int32_t *destination,
                                         const AscendC::LocalTensor<int32_t> &local,
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
        // AIV-only inter-core rendezvous. Every AIV, including non-primary
        // subblocks, enters every protocol phase in the same order.
        AscendC::SyncAll<true>();
    }

    __aicore__ inline void PublishAndWaitRankPhase(uint64_t phaseBase)
    {
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            for (uint32_t dstRank = logicalCoreIdx_; dstRank < rankSize_; dstRank += rankWorkerCount_) {
                StoreSyncFlag(GetPhaseSlot(dstRank, phaseBase, rank_), syncEpoch_);
            }
        }
        // Do not allow a worker to wait until every outbound slot from this
        // rank has been published; otherwise ranks can form a wait cycle.
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
                WaitSyncFlag(GetPhaseSlot(rank_, phaseBase, srcRank), syncEpoch_);
            }
        }
        LocalAivSync();
    }

    __aicore__ inline void PublishAndWaitExpertReady(uint32_t expertIdx)
    {
        // Match mega_moe/combine A3's destination-local mailbox protocol. All
        // payload writers drain before this barrier, then source rank S writes
        // ExpertReady(D, S, E) into every destination D's local window. A
        // destination only polls its own window through the valid L2-bypass
        // alias; it never polls an encoded peer VA.
        // sendCnt/recvCnt come from the same tiling computation on both ends,
        // so a skipped publish slot is exactly the slot the peer skips
        // waiting on: publish is needed only when this rank actually sends
        // expert e to D, and waiting is needed only when this rank actually
        // receives expert e from S. The self slot is always kept because the
        // local wait doubles as the all-AIV rendezvous for this expert.
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
            for (uint32_t dstRank = logicalCoreIdx_; dstRank < rankSize_; dstRank += rankWorkerCount_) {
                if (dstRank != rank_ &&
                    taskTilingInfo_->sendCnt[static_cast<uint64_t>(dstRank) * expertNum + expertIdx] <= 0) {
                    continue;
                }
                StoreSyncFlag(GetExpertReadySlot(dstRank, rank_, expertIdx), syncEpoch_);
            }
        }
        // Finish every outbound publication before allowing any worker to
        // wait, preventing a rank-level publish/wait cycle.
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
            for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
                if (srcRank != rank_ &&
                    taskTilingInfo_->recvCnt[static_cast<uint64_t>(srcRank) * expertNum + expertIdx] <= 0) {
                    continue;
                }
                WaitSyncFlag(GetExpertReadySlot(rank_, srcRank, expertIdx), syncEpoch_);
            }
        }
        LocalAivSync();
    }

    __aicore__ inline __gm__ int32_t *GetFixedSyncSlot(uint32_t windowRank, uint64_t phaseBase, uint64_t index) const
    {
        const uint64_t syncOffset =
            winSize_ - Gmma2avMteTiling::SYNC_REGION_FROM_TAIL + phaseBase + index * Gmma2avMteTiling::SYNC_SLOT_BYTES;
        return reinterpret_cast<__gm__ int32_t *>(GetWindow(windowRank) + syncOffset);
    }

    __aicore__ inline __gm__ int32_t *GetEpochSlot(uint32_t windowRank) const
    {
        return GetFixedSyncSlot(windowRank, Gmma2avMteTiling::EPOCH_BASE, 0UL);
    }

    __aicore__ inline __gm__ int32_t *GetPhaseSlot(uint32_t windowRank, uint64_t phaseBase,
                                                   uint32_t participantRank) const
    {
        return GetFixedSyncSlot(windowRank, phaseBase, participantRank);
    }

    __aicore__ inline __gm__ int32_t *GetExpertReadySlot(uint32_t windowRank, uint32_t sourceRank,
                                                         uint32_t expertIdx) const
    {
        const uint64_t expertSlot = static_cast<uint64_t>(sourceRank) * taskTilingInfo_->e + expertIdx;
        return GetFixedSyncSlot(windowRank, Gmma2avMteTiling::EXPERT_READY_BASE, expertSlot);
    }

    __aicore__ inline __gm__ int32_t *GetAckSlot(uint32_t windowRank, uint32_t participantRank) const
    {
        return GetPhaseSlot(windowRank, Gmma2avMteTiling::ACK_BASE, participantRank);
    }

    __aicore__ inline void StoreSyncFlag(__gm__ int32_t *destination, int32_t flag)
    {
        AscendC::LocalTensor<int32_t> local = GetControlTensor();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        local.SetValue(0U, flag);
        for (uint32_t index = 1U; index < GmmA2avMteDetail::CONTROL_MTE_ELEMENTS; ++index) {
            local.SetValue(index, 0);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(destination);
        // A3's established remote-state protocols publish one complete 32-byte
        // block.  Keeping the transfer naturally aligned also avoids issuing a
        // short MTE transaction against a peer mapping.
        AscendC::DataCopy(global, local, GmmA2avMteDetail::CONTROL_MTE_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    __aicore__ inline bool IsPrimaryAiv() const
    {
        return logicalCoreIdx_ == 0U && subBlockIdx_ == 0U;
    }

    __aicore__ inline void WaitSyncFlag(__gm__ int32_t *source, int32_t expected)
    {
        AscendC::LocalTensor<int32_t> local = GetControlTensor();
        AscendC::GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(source);
        // Sync slots are always polled through this rank's local window.  The
        // bypass alias is valid for local GM and prevents a stale cache line
        // after a peer updates the slot; never apply it to encoded peer VAs.
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (true) {
            AscendC::DataCopy(local, global, GmmA2avMteDetail::CONTROL_MTE_ELEMENTS);
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

    __aicore__ inline void CopyGmToGm(__gm__ T *source, __gm__ T *destination,
                                      uint64_t elementCount)
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

    __aicore__ inline void CopyGmToGmSegment(__gm__ T *source, __gm__ T *destination,
                                             uint64_t elementCount, uint32_t &moveIdx)
    {
        AscendC::LocalTensor<T> local = ubBuffer_.template Get<T>();
        AscendC::LocalTensor<T> ping = local;
        AscendC::LocalTensor<T> pong = local[
            GmmA2avMteDetail::PONG_UB_OFFSET_BYTES / sizeof(T)];
        uint64_t copied = 0UL;
        while (copied < elementCount) {
            const uint32_t current = static_cast<uint32_t>(elementCount - copied > ubMoveElements_ ?
                ubMoveElements_ : elementCount - copied);
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
    uint64_t winSize_{0UL};
    uint64_t localExpertBaseRows_{0UL};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    bool isA3_{false};
    uint32_t coreIdx_{0U};
    uint32_t subBlockIdx_{0U};
    int32_t syncEpoch_{1};
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

    struct AicExpertReadyCallback {
        __aicore__ inline explicit AicExpertReadyCallback(ExpertReadyFlag &readyFlag)
            : readyFlag_(readyFlag)
        {}

        __aicore__ inline void operator()(uint32_t expertIdx)
        {
            (void)expertIdx;
            // Every AIC first drains the same expert and rendezvouses with all
            // other AICs. Only then does each AIC notify its paired AIV.
            AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AIC_ALL_READY_FLAG_IDX);
            AscendC::CrossCoreWaitFlag(AIC_ALL_READY_FLAG_IDX);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
        }

        ExpertReadyFlag &readyFlag_;
    };

public:
    static __aicore__ inline void ProcessAic(ComputationOpType &computeOp, SharedComputationOpType &sharedComputeOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                             uint32_t expertPipelineMode)
    {
        if ASCEND_IS_AIC {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            if (expertPipelineMode == static_cast<uint32_t>(ExpertPipelineMode::EXPERT_OVERLAP)) {
                ExpertReadyFlag readyFlag(EXPERT_READY_FLAG_IDX, EXPERT_REVERSE_FLAG_IDX);
                AicExpertReadyCallback callback(readyFlag);
                computeOp.ProcessExperts(0U, expertNum, callback);
            } else {
                computeOp.Process(0U, expertNum);
                AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AIC_ALL_READY_FLAG_IDX);
                AscendC::CrossCoreWaitFlag(AIC_ALL_READY_FLAG_IDX);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(EXPERT_READY_FLAG_IDX);
            }

            if constexpr (IsNeedMM) {
                sharedComputeOp.Process(0U, 1U);
            }
            AscendC::SyncAll<false>();
        }
    }

    static __aicore__ inline void ProcessAiv(CommOpType &commOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                             uint32_t expertPipelineMode)
    {
        if ASCEND_IS_AIV {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            if (expertPipelineMode == static_cast<uint32_t>(ExpertPipelineMode::EXPERT_OVERLAP)) {
                ExpertReadyFlag readyFlag(EXPERT_READY_FLAG_IDX, EXPERT_REVERSE_FLAG_IDX);
                commOp.BeginExpertPipeline();
                for (uint32_t expertIdx = 0U; expertIdx < expertNum; ++expertIdx) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(readyFlag);
                    commOp.LaunchExpert(expertIdx);
                }
            } else {
                AscendC::CrossCoreWaitFlag(EXPERT_READY_FLAG_IDX);
                commOp.LaunchCommData(0U, expertNum);
            }
            commOp.End();
        }
    }

private:
    static constexpr Catlass::Arch::FlagID EXPERT_READY_FLAG_IDX = 0U;
    static constexpr Catlass::Arch::FlagID EXPERT_REVERSE_FLAG_IDX = 1U;
    static constexpr Catlass::Arch::FlagID AIC_ALL_READY_FLAG_IDX = 9U;
    static_assert(EXPERT_READY_FLAG_IDX <= 15U && EXPERT_REVERSE_FLAG_IDX <= 15U && AIC_ALL_READY_FLAG_IDX <= 15U,
                  "arch22 local CrossCore flag IDs must be in [0, 15]");
    static_assert(EXPERT_READY_FLAG_IDX != EXPERT_REVERSE_FLAG_IDX && EXPERT_READY_FLAG_IDX != AIC_ALL_READY_FLAG_IDX &&
                      EXPERT_REVERSE_FLAG_IDX != AIC_ALL_READY_FLAG_IDX,
                  "expert ready, reverse-credit and all-AIC barrier flags must be distinct");
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
