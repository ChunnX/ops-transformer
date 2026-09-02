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
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "grouped_mat_mul_allto_allv_mte_tiling.h"

namespace MC2KernelTemplate {

namespace GmmA2avMteDetail {
constexpr uint32_t DEFAULT_UB_MOVE_ELEMENTS = Gmma2avMteTiling::UB_MOVE_ELEMENTS;
constexpr uint32_t USED_UB_BYTES = 192U * 1024U;
constexpr uint32_t CONTROL_UB_OFFSET_BYTES = 64U * 1024U;
constexpr uint32_t PONG_UB_OFFSET_BYTES = 96U * 1024U;
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

        // The first 64 KiB is the payload ping buffer, the packed count table
        // is staged at 64 KiB, and the payload pong buffer starts at 96 KiB.
        pipe->InitBuffer(ubBuffer_, GmmA2avMteDetail::USED_UB_BYTES);

        // HCCL windows persist across operator invocations. Reusing 0/1 phase
        // flags permits a fast rank to accept the previous invocation's ready
        // value (ABA) before a slower peer has reset it. The acknowledgement
        // slot records the last completed invocation, so advance it and use the
        // new epoch for every phase of this invocation without resetting slots.
        const int32_t previousEpoch = AscendC::ReadGmByPassDCache(
            GetSyncSlot(rank_, rank_, SYNC_ACK_FLAG_IDX));
        syncEpoch_ = previousEpoch <= 0 || previousEpoch >= MAX_SYNC_EPOCH ? 1 : previousEpoch + 1;
    }

    __aicore__ inline void LaunchCommData(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (rankWorkerCount_ == 0U || expertNum == 0U || startExpertIdx >= taskTilingInfo_->e) {
            return;
        }

        // Follow the proven AllToAllvGMM publish/pull protocol: every source
        // publishes its complete payload and send-count table in its own HCCL
        // window.  Receivers pull only their segments after the publish barrier.
        // No peer-written output-offset table is reused across invocations.
        PublishLocalPayload();
        SynchronizeRanks(SYNC_PAYLOAD_FLAG_IDX, syncEpoch_);
        PullPeerPayload(startExpertIdx, expertNum);
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        (void)startExpertIdx;
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            // A source may reuse its persistent window only after every peer
            // has finished pulling this invocation's payload and metadata.
            SynchronizeRanks(SYNC_COMPLETION_FLAG_IDX, syncEpoch_);
            SynchronizeRanks(SYNC_ACK_FLAG_IDX, syncEpoch_);
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

    __aicore__ inline void PublishLocalPayload()
    {
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
            return;
        }
        uint64_t totalRows = 0UL;
        const uint32_t countNum = rankSize_ * static_cast<uint32_t>(taskTilingInfo_->e);
        for (uint32_t index = 0U; index < countNum; ++index) {
            totalRows += static_cast<uint64_t>(taskTilingInfo_->sendCnt[index]);
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
        if (IsPrimaryAiv()) {
            AscendC::LocalTensor<int32_t> local = GetControlTensor();
            for (uint32_t index = 0U; index < countNum; ++index) {
                local.SetValue(index, taskTilingInfo_->sendCnt[index]);
            }
            // This is a local-window publication.  Keep the complete table
            // packed, matching the AllToAllvGMM peer-pull protocol.
            StoreInt32Row(GetCountTable(rank_), local, countNum);
        }
    }

    __aicore__ inline void PullPeerPayload(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
            return;
        }
        const uint32_t localExpertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        const uint32_t endExpertIdx = startExpertIdx + expertNum > localExpertNum ?
            localExpertNum : startExpertIdx + expertNum;
        const uint32_t countNum = rankSize_ * localExpertNum;
        AscendC::LocalTensor<int32_t> peerCounts = GetControlTensor();
        for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
            if (srcRank == rank_) {
                for (uint32_t index = 0U; index < countNum; ++index) {
                    peerCounts.SetValue(index, taskTilingInfo_->sendCnt[index]);
                }
            } else {
                // The publish/release handshake makes an ordinary peer-window
                // read safe across invocations; bypass-cache scalar access is
                // not supported for every remote-window mapping.
                LoadInt32Row(GetCountTable(srcRank), peerCounts, countNum);
            }

            uint64_t destinationTokenOffset = 0UL;
            for (uint32_t priorSource = 0U; priorSource < srcRank; ++priorSource) {
                for (uint32_t expert = 0U; expert < localExpertNum; ++expert) {
                    destinationTokenOffset += static_cast<uint64_t>(taskTilingInfo_->recvCnt[
                        static_cast<uint64_t>(priorSource) * localExpertNum + expert]);
                }
            }

            uint64_t expertSourceBase = 0UL;
            for (uint32_t expert = 0U; expert < localExpertNum; ++expert) {
                uint64_t sourceOffsetInExpert = 0UL;
                for (uint32_t dstRank = 0U; dstRank < rank_; ++dstRank) {
                    sourceOffsetInExpert += static_cast<uint64_t>(
                        peerCounts.GetValue(static_cast<uint64_t>(dstRank) * localExpertNum + expert));
                }
                const uint64_t countIndex = static_cast<uint64_t>(rank_) * localExpertNum + expert;
                const uint64_t sendTokenCount = static_cast<uint64_t>(peerCounts.GetValue(countIndex));
                const uint64_t expectedTokenCount = static_cast<uint64_t>(taskTilingInfo_->recvCnt[
                    static_cast<uint64_t>(srcRank) * localExpertNum + expert]);
                if (sendTokenCount != expectedTokenCount) {
                    return;
                }
                if (expert >= startExpertIdx && expert < endExpertIdx && sendTokenCount != 0UL) {
                    __gm__ T *source = srcRank == rank_ ? sendBuffer_ : GetOutputData(srcRank);
                    CopyGmToGm(source + (expertSourceBase + sourceOffsetInExpert) * taskTilingInfo_->N1,
                               recvBuffer_ + destinationTokenOffset * taskTilingInfo_->N1,
                               sendTokenCount * taskTilingInfo_->N1);
                }
                for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                    expertSourceBase += static_cast<uint64_t>(
                        peerCounts.GetValue(static_cast<uint64_t>(dstRank) * localExpertNum + expert));
                }
                destinationTokenOffset += expectedTokenCount;
            }
        }
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> GetControlTensor()
    {
        return ubBuffer_.template Get<int32_t>()[
            GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int32_t)];
    }

    __aicore__ inline void LoadInt32Row(__gm__ int32_t *source,
                                        const AscendC::LocalTensor<int32_t> &local,
                                        uint32_t elementCount)
    {
        AscendC::GlobalTensor<int32_t> sourceGlobal;
        sourceGlobal.SetGlobalBuffer(source);
        AscendC::DataCopyExtParams copyParams(1U, elementCount * sizeof(int32_t), 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<int32_t> padParams;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::DataCopyPad(local, sourceGlobal, copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    }

    __aicore__ inline void StoreInt32Row(__gm__ int32_t *destination,
                                         const AscendC::LocalTensor<int32_t> &local,
                                         uint32_t elementCount)
    {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::GlobalTensor<int32_t> destinationGlobal;
        destinationGlobal.SetGlobalBuffer(destination);
        AscendC::DataCopyExtParams copyParams(1U, elementCount * sizeof(int32_t), 0U, 0U, 0U);
        AscendC::DataCopyPad(destinationGlobal, local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    }

    __aicore__ inline void LocalAivSync(uint32_t flagIdx)
    {
        SetAndWaitAivSync(flagIdx);
    }

    __aicore__ inline void SynchronizeRanks(uint32_t flagIdx, int32_t flagValue)
    {
        LocalAivSync(flagIdx);
        CrossRankSyncV1(flagIdx, flagValue);
        LocalAivSync(flagIdx);
    }

    __aicore__ inline void SetAndWaitAivSync(uint64_t flagIdx)
    {
        // Keep local flag IDs 2-6 disjoint from the AIC/AIV events 8/9. This is the
        // exact local synchronization primitive used around CrossRankSyncV1
        // by alltoallmatmul and matmulalltoall arch22.
        constexpr uint64_t pipeDepth = 2UL;
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(flagIdx + pipeDepth);
        AscendC::CrossCoreWaitFlag(flagIdx + pipeDepth);
    }

    __aicore__ inline void CrossRankSyncV1(uint32_t flagIdx, int32_t flagValue)
    {
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
            return;
        }
        // Publish every outbound flag first. The flag is written into the same
        // destination window as the preceding payload/control data, so the
        // receiver observes completion through a local read only after the
        // peer-memory stores ahead of it have reached that destination.
        for (uint32_t dstRank = logicalCoreIdx_; dstRank < rankSize_; dstRank += rankWorkerCount_) {
            StoreSyncFlag(GetSyncSlot(dstRank, rank_, flagIdx), flagValue);
        }
        for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
            WaitSyncFlag(GetSyncSlot(rank_, srcRank, flagIdx), flagValue);
        }
    }

    __aicore__ inline __gm__ int32_t *GetSyncSlot(uint32_t windowRank, uint32_t sourceRank,
                                                  uint32_t flagIdx) const
    {
        const uint64_t syncOffset = winSize_ - Gmma2avMteTiling::SYNC_REGION_FROM_TAIL +
                                    (static_cast<uint64_t>(flagIdx) * rankSize_ + sourceRank) *
                                        Gmma2avMteTiling::SYNC_SLOT_BYTES;
        return reinterpret_cast<__gm__ int32_t *>(GetWindow(windowRank) + syncOffset);
    }

    __aicore__ inline void StoreSyncFlag(__gm__ int32_t *destination, int32_t flag)
    {
        AscendC::LocalTensor<int32_t> local = GetControlTensor();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        local.SetValue(0U, flag);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(destination);
        AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
        AscendC::DataCopyPad(global, local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
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
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<int32_t> padParams;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (true) {
            AscendC::DataCopyPad(local, global, copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            if (local.GetValue(0U) == expected) {
                break;
            }
        }
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
    }

private:
    static constexpr uint32_t SYNC_PAYLOAD_FLAG_IDX = 2U;
    static constexpr uint32_t SYNC_COMPLETION_FLAG_IDX = 3U;
    static constexpr uint32_t SYNC_ACK_FLAG_IDX = 4U;
    static constexpr int32_t MAX_SYNC_EPOCH = 0x7ffffffe;
    const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_{nullptr};
    __gm__ HcclOpResParam *a3WinContext_{nullptr};
    __gm__ AscendC::HcclCombineOpParam *a2WinContext_{nullptr};
    __gm__ T *sendBuffer_{nullptr};
    __gm__ T *recvBuffer_{nullptr};
    uint64_t winSize_{0UL};
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
public:
    __aicore__ inline GmmA2avMteScheduler(CommOpType &commOp, ComputationOpType &computeOp,
                                          SharedComputationOpType &sharedComputeOp,
                                          const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
        : commOp_(commOp), computeOp_(computeOp), sharedComputeOp_(sharedComputeOp),
          taskTilingInfo_(taskTilingInfo) {}

    __aicore__ inline void Process()
    {
        ProcessAic(computeOp_, sharedComputeOp_, taskTilingInfo_);
        ProcessAiv(commOp_, taskTilingInfo_);
    }

    static __aicore__ inline void ProcessAic(ComputationOpType &computeOp,
                                             SharedComputationOpType &sharedComputeOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
    {
        if ASCEND_IS_AIC {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            computeOp.Process(0U, expertNum);
            // A source rank's expert output is striped over all AICs. Drain the
            // AIC-only FIX barrier before every AIC notifies its paired AIV;
            // only then may the AIV write the workspace result to peer memory.
            AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AIC_ALL_READY_FLAG_IDX);
            AscendC::CrossCoreWaitFlag(AIC_ALL_READY_FLAG_IDX);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(AIC_TO_AIV_READY_FLAG_IDX);

            if constexpr (IsNeedMM) {
                sharedComputeOp.Process(0U, 1U);
            }
            AscendC::SyncAll<false>();
        }
    }

    static __aicore__ inline void ProcessAiv(
        CommOpType &commOp, const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
    {
        if ASCEND_IS_AIV {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            AscendC::CrossCoreWaitFlag(AIC_TO_AIV_READY_FLAG_IDX);
            commOp.LaunchCommData(0U, expertNum);
            commOp.End();
        }
    }

private:
    static constexpr uint64_t AIC_ALL_READY_FLAG_IDX = 8UL;
    static constexpr uint64_t AIC_TO_AIV_READY_FLAG_IDX = 9UL;
    CommOpType &commOp_;
    ComputationOpType &computeOp_;
    SharedComputationOpType &sharedComputeOp_;
    const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_;
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
