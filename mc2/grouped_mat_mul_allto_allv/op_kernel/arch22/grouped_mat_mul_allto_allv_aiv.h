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
 * \file grouped_mat_mul_allto_allv_aiv.h
 * \brief AIV memory-semantics communication for GroupedMatMulAlltoAllvV2 on Ascend 910_93.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_AIV_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_AIV_H

#include "kernel_operator.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "grouped_mat_mul_allto_allv_aiv_tiling.h"

namespace MC2KernelTemplate {

namespace GmmA2avAivDetail {
constexpr uint32_t DEFAULT_UB_MOVE_ELEMENTS = Gmma2avAivTiling::UB_MOVE_ELEMENTS;

template <typename T>
__aicore__ inline void GmStore(__gm__ T *address, T value)
{
    AscendC::WriteGmByPassDCache(address, value);
}

template <typename T>
__aicore__ inline T GmLoad(__gm__ T *address)
{
    return AscendC::ReadGmByPassDCache(address);
}

template <typename T>
__aicore__ inline void Dcci(__gm__ T *address)
{
    (void)address;
}
} // namespace GmmA2avAivDetail

template <typename T>
class GmmA2avAivOp {
public:
    __aicore__ inline GmmA2avAivOp() {}

    __aicore__ inline void Init(const GroupedMatMulAlltoAllvAivTaskTilingInfo *taskTilingInfo,
                                GM_ADDR sendBuffer, GM_ADDR recvBuffer, GM_ADDR cumsumBuffer,
                                AscendC::TPipe *pipe,
                                const GmmA2avCoCTiling *cocTiling = nullptr,
                                uint64_t commBufferSize = 0UL)
    {
        taskTilingInfo_ = taskTilingInfo;
        sendBuffer_ = reinterpret_cast<__gm__ T *>(sendBuffer);
        recvBuffer_ = reinterpret_cast<__gm__ T *>(recvBuffer);
        cumsumBuffer_ = reinterpret_cast<__gm__ int32_t *>(cumsumBuffer);
        aivCoreNum_ = static_cast<uint32_t>(taskTilingInfo_->aivCoreNum);
        if ASCEND_IS_AIC {
            return;
        }

        auto context = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        winContext_ = reinterpret_cast<__gm__ HcclOpResParam *>(context);
        rank_ = winContext_->localUsrRankId;
        rankSize_ = winContext_->rankSize;
        // winSize can be zero in eager/static-graph contexts. The host tiling
        // obtains the effective per-rank CCL window size from the communicator.
        winSize_ = winContext_->winSize == 0UL ? commBufferSize : winContext_->winSize;
        coreIdx_ = AscendC::GetBlockIdx();
        if (cocTiling != nullptr && cocTiling->ubMoveNum > 0) {
            ubMoveElements_ = static_cast<uint32_t>(cocTiling->ubMoveNum);
        }

        pipe->InitBuffer(ubBuffer_, 2U * ubMoveElements_ * sizeof(T));
        PrepareSyncEpoch();
        AscendC::SyncAll<true>();
        syncEpochBase_ = GmmA2avAivDetail::GmLoad(GetSyncEpochMetadata() + SYNC_EPOCH_PUBLISHED_INDEX);
        StageLocalSendCounts();
        AscendC::SyncAll<true>();
        CrossRankSync();
        AscendC::SyncAll<true>();
        ExchangeCountMatrix();
        AscendC::SyncAll<true>();
        CrossRankSync();
        AscendC::SyncAll<true>();
        BuildCumSum();
        AscendC::SyncAll<true>();
        ExchangeRecvOffsets();
        AscendC::SyncAll<true>();
        CrossRankSync();
        AscendC::SyncAll<true>();
    }

    __aicore__ inline void LaunchCommData(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (startExpertIdx >= taskTilingInfo_->e || expertNum == 0U) {
            return;
        }
        uint32_t endExpertIdx = startExpertIdx + expertNum;
        if (endExpertIdx > taskTilingInfo_->e) {
            endExpertIdx = static_cast<uint32_t>(taskTilingInfo_->e);
        }
        uint64_t expertSourceBase = 0UL;
        for (uint32_t expert = 0U; expert < startExpertIdx; ++expert) {
            for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                expertSourceBase += static_cast<uint64_t>(
                    taskTilingInfo_->sendCnt[dstRank * taskTilingInfo_->e + expert]);
            }
        }
        for (uint32_t expert = startExpertIdx; expert < endExpertIdx; ++expert) {
            uint64_t sourceOffsetInExpert = 0UL;
            for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                uint64_t sendTokenCount = static_cast<uint64_t>(
                    taskTilingInfo_->sendCnt[dstRank * taskTilingInfo_->e + expert]);
                if ((dstRank % aivCoreNum_) == coreIdx_ && sendTokenCount != 0UL) {
                    uint64_t remoteTokenOffset = GetLocalSendTokenOffset(dstRank, expert);
                    __gm__ T *remoteOutput = reinterpret_cast<__gm__ T *>(GetWindow(dstRank));
                    CopyGmToGm(sendBuffer_ + (expertSourceBase + sourceOffsetInExpert) * taskTilingInfo_->N1,
                               remoteOutput + remoteTokenOffset * taskTilingInfo_->N1,
                               sendTokenCount * taskTilingInfo_->N1);
                }
                sourceOffsetInExpert += sendTokenCount;
            }
            expertSourceBase += sourceOffsetInExpert;
        }
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        (void)startExpertIdx;
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<true>();
            CrossRankSync();
            AscendC::SyncAll<true>();
            CopyLocalResult();
            // Do not let a faster rank return and reuse its CCL window while a
            // slower rank is still copying the result produced by this call.
            AscendC::SyncAll<true>();
            CrossRankSync();
            AscendC::SyncAll<true>();
        }
        AscendC::SyncAll<false>();
    }

private:
    __aicore__ inline GM_ADDR GetWindow(uint32_t rank) const
    {
        if (rank == rank_) {
            return (GM_ADDR)(winContext_->localWindowsIn);
        }
        auto relation = reinterpret_cast<__gm__ HcclRankRelationResV2 *>(
            winContext_->remoteRes[rank].nextDevicePtr);
        return (GM_ADDR)(relation->windowsIn);
    }

    __aicore__ inline __gm__ int32_t *GetCountMatrix(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ int32_t *>(
            GetWindow(rank) + winSize_ - Gmma2avAivTiling::CONTROL_REGION_BYTES - GetCountMatrixBytes());
    }

    __aicore__ inline __gm__ int32_t *GetSendCounts(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ int32_t *>(
            GetWindow(rank) + winSize_ - Gmma2avAivTiling::SEND_COUNT_STAGING_FROM_TAIL);
    }

    __aicore__ inline __gm__ int32_t *GetSyncEpochMetadata() const
    {
        return reinterpret_cast<__gm__ int32_t *>(
            GetWindow(rank_) + winSize_ - Gmma2avAivTiling::SYNC_EPOCH_FROM_TAIL);
    }

    __aicore__ inline uint64_t GetCountNum() const
    {
        return static_cast<uint64_t>(rankSize_) * taskTilingInfo_->e;
    }

    __aicore__ inline uint64_t GetCountMatrixBytes() const
    {
        return GetCountNum() * sizeof(int32_t);
    }

    __aicore__ inline void StageLocalSendCounts()
    {
        if (coreIdx_ != 0U) {
            return;
        }
        __gm__ int32_t *staging = GetSendCounts(rank_);
        for (uint64_t idx = 0UL; idx < GetCountNum(); ++idx) {
            GmmA2avAivDetail::GmStore(staging + idx, taskTilingInfo_->sendCnt[idx]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ExchangeCountMatrix()
    {
        // AllGather is implemented with peer-memory stores. Scalar loads from a
        // remote CCL window are not a reliable data path on A3; each rank
        // therefore publishes its row directly into every destination's local
        // receive-count matrix, followed by a cross-rank completion barrier.
        for (uint32_t dstRank = coreIdx_; dstRank < rankSize_; dstRank += aivCoreNum_) {
            __gm__ int32_t *remoteCount =
                GetCountMatrix(dstRank) + static_cast<uint64_t>(rank_) * taskTilingInfo_->e;
            const uint64_t sendOffset = static_cast<uint64_t>(dstRank) * taskTilingInfo_->e;
            AscendC::LocalTensor<int32_t> local = ubBuffer_.template Get<int32_t>();
            for (uint32_t expert = 0U; expert < taskTilingInfo_->e; ++expert) {
                local.SetValue(expert, taskTilingInfo_->sendCnt[sendOffset + expert]);
            }
            StorePeerInt32Row(remoteCount, local, static_cast<uint32_t>(taskTilingInfo_->e));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareSyncEpoch()
    {
        if (coreIdx_ != 0U) {
            return;
        }
        __gm__ int32_t *metadata = GetSyncEpochMetadata();
        int32_t marker = GmmA2avAivDetail::GmLoad(metadata + SYNC_EPOCH_MARKER_INDEX);
        int32_t completedEpoch = 0;
        if (marker == SYNC_EPOCH_MARKER) {
            completedEpoch = GmmA2avAivDetail::GmLoad(metadata + SYNC_EPOCH_COUNTER_INDEX);
        } else {
            GmmA2avAivDetail::GmStore(metadata + SYNC_EPOCH_MARKER_INDEX, SYNC_EPOCH_MARKER);
        }
        if (completedEpoch > INT32_MAX - static_cast<int32_t>(Gmma2avAivTiling::CROSS_RANK_SYNC_COUNT)) {
            completedEpoch = 0;
        }
        const int32_t firstEpoch = completedEpoch + 1;
        GmmA2avAivDetail::GmStore(
            metadata + SYNC_EPOCH_COUNTER_INDEX,
            completedEpoch + static_cast<int32_t>(Gmma2avAivTiling::CROSS_RANK_SYNC_COUNT));
        GmmA2avAivDetail::GmStore(metadata + SYNC_EPOCH_PUBLISHED_INDEX, firstEpoch);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildCumSum()
    {
        if (coreIdx_ != 0U) {
            return;
        }
        AscendC::LocalTensor<int32_t> local = ubBuffer_.template Get<int32_t>();
        __gm__ int32_t *countMatrix = GetCountMatrix(rank_);
        uint64_t tokenOffset = 0UL;
        for (uint64_t idx = 0UL; idx < GetCountNum(); ++idx) {
            GmmA2avAivDetail::Dcci(countMatrix + idx);
            int32_t count = GmmA2avAivDetail::GmLoad(countMatrix + idx);
            local.SetValue(static_cast<uint32_t>(idx), static_cast<int32_t>(tokenOffset));
            int32_t prefix = local.GetValue(static_cast<uint32_t>(idx));
            GmmA2avAivDetail::GmStore(cumsumBuffer_ + idx, prefix);
            GmmA2avAivDetail::GmStore(countMatrix + idx, prefix);
            tokenOffset += static_cast<uint64_t>(count);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ExchangeRecvOffsets()
    {
        // Remote scalar loads are avoided for the data path. Each destination
        // pushes the prefix slice computed for a source back to that source's
        // local table. BuildCumSum leaves the compact prefix table in core 0's
        // UB, so publish from that UB directly. Going through GM here would mix
        // scalar bypass-cache stores with MTE2 loads and is not reliably visible
        // across cache lines on A3.
        if (coreIdx_ != 0U) {
            return;
        }
        AscendC::LocalTensor<int32_t> localPrefix = ubBuffer_.template Get<int32_t>();
        // MTE3 requires the UB source address to be 32-byte aligned. A compact
        // row starts at srcRank * e and therefore is not necessarily aligned.
        // Keep the complete prefix table intact and stage one row at the fixed,
        // aligned offset immediately following its maximum reserved range.
        constexpr uint32_t prefixStagingOffset = GMMA2AV_AIV_MAX_COUNT_NUM;
        AscendC::LocalTensor<int32_t> alignedRow = localPrefix[prefixStagingOffset];
        for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
            const uint32_t sourceOffset = static_cast<uint32_t>(srcRank * taskTilingInfo_->e);
            for (uint32_t expert = 0U; expert < taskTilingInfo_->e; ++expert) {
                alignedRow.SetValue(expert, localPrefix.GetValue(sourceOffset + expert));
            }
            __gm__ int32_t *remoteOffset =
                GetCountMatrix(srcRank) + static_cast<uint64_t>(rank_) * taskTilingInfo_->e;
            StorePeerInt32Row(remoteOffset, alignedRow, static_cast<uint32_t>(taskTilingInfo_->e));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint64_t GetLocalSendTokenOffset(uint32_t dstRank, uint32_t expertIdx) const
    {
        __gm__ int32_t *localOffset =
            GetCountMatrix(rank_) + static_cast<uint64_t>(dstRank) * taskTilingInfo_->e + expertIdx;
        return static_cast<uint64_t>(GmmA2avAivDetail::GmLoad(localOffset));
    }

    __aicore__ inline void StorePeerInt32Row(__gm__ int32_t *destination,
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

    __aicore__ inline void CrossRankSync()
    {
        // Complete every local AIV write before publishing the cross-rank
        // completion flag. The persistent epoch prevents a later invocation
        // from accepting a flag left in the CCL window by an earlier one.
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t epoch = syncEpochBase_ + static_cast<int32_t>(syncPhase_++);
        uint64_t syncOffset = winSize_ - Gmma2avAivTiling::SYNC_REGION_FROM_TAIL;
        for (uint32_t dstRank = coreIdx_; dstRank < rankSize_; dstRank += aivCoreNum_) {
            __gm__ int32_t *remoteSlot = reinterpret_cast<__gm__ int32_t *>(
                GetWindow(dstRank) + syncOffset +
                static_cast<uint64_t>(rank_) * Gmma2avAivTiling::SYNC_SLOT_BYTES);
            StoreSyncFlag(remoteSlot, epoch);
            __gm__ int32_t *localSlot = reinterpret_cast<__gm__ int32_t *>(
                GetWindow(rank_) + syncOffset +
                static_cast<uint64_t>(dstRank) * Gmma2avAivTiling::SYNC_SLOT_BYTES);
            WaitSyncFlag(localSlot, epoch);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreSyncFlag(__gm__ int32_t *destination, int32_t flag)
    {
        AscendC::LocalTensor<int32_t> local = ubBuffer_.template Get<int32_t>();
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

    __aicore__ inline void WaitSyncFlag(__gm__ int32_t *source, int32_t expected)
    {
        AscendC::LocalTensor<int32_t> local = ubBuffer_.template Get<int32_t>();
        AscendC::GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(source);
        AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<int32_t> padParams;
        while (true) {
            AscendC::DataCopyPad(local, global, copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            if (local.GetValue(0U) == expected) {
                break;
            }
        }
    }

    __aicore__ inline void CopyLocalResult()
    {
        uint64_t totalRows = taskTilingInfo_->BSK;
        uint64_t rowsPerCore = (totalRows + aivCoreNum_ - 1UL) / aivCoreNum_;
        uint64_t rowStart = static_cast<uint64_t>(coreIdx_) * rowsPerCore;
        if (rowStart >= totalRows) {
            return;
        }
        uint64_t rowCount = totalRows - rowStart;
        if (rowCount > rowsPerCore) {
            rowCount = rowsPerCore;
        }
        __gm__ T *localOutput = reinterpret_cast<__gm__ T *>(GetWindow(rank_));
        CopyGmToGm(localOutput + rowStart * taskTilingInfo_->N1,
                   recvBuffer_ + rowStart * taskTilingInfo_->N1, rowCount * taskTilingInfo_->N1);
    }

    __aicore__ inline void CopyGmToGm(__gm__ T *source, __gm__ T *destination, uint64_t elementCount)
    {
        AscendC::LocalTensor<T> local = ubBuffer_.template Get<T>();
        AscendC::LocalTensor<T> ping = local;
        AscendC::LocalTensor<T> pong = local[ubMoveElements_];
        uint64_t copied = 0UL;
        uint32_t moveIdx = 0U;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (copied < elementCount) {
            uint32_t current = static_cast<uint32_t>(elementCount - copied >
                ubMoveElements_ ? ubMoveElements_ :
                elementCount - copied);
            event_t eventId = (moveIdx & 1U) == 0U ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> currentLocal = (moveIdx & 1U) == 0U ? ping : pong;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::GlobalTensor<T> sourceGlobal;
            sourceGlobal.SetGlobalBuffer(source + copied);
            AscendC::DataCopyExtParams copyParams(1U, current * sizeof(T), 0U, 0U, 0U);
            AscendC::DataCopyPadExtParams<T> padParams;
            AscendC::DataCopyPad(currentLocal, sourceGlobal, copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::GlobalTensor<T> destinationGlobal;
            destinationGlobal.SetGlobalBuffer(destination + copied);
            AscendC::DataCopyPad(destinationGlobal, currentLocal, copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            copied += current;
            ++moveIdx;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

private:
    static constexpr int32_t SYNC_EPOCH_MARKER = 0x47563241;
    static constexpr uint32_t SYNC_EPOCH_MARKER_INDEX = 0U;
    static constexpr uint32_t SYNC_EPOCH_COUNTER_INDEX = 1U;
    static constexpr uint32_t SYNC_EPOCH_PUBLISHED_INDEX = 2U;

    const GroupedMatMulAlltoAllvAivTaskTilingInfo *taskTilingInfo_{nullptr};
    __gm__ HcclOpResParam *winContext_{nullptr};
    __gm__ T *sendBuffer_{nullptr};
    __gm__ T *recvBuffer_{nullptr};
    __gm__ int32_t *cumsumBuffer_{nullptr};
    uint64_t winSize_{0UL};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    uint32_t coreIdx_{0U};
    uint32_t aivCoreNum_{1U};
    uint32_t syncPhase_{0U};
    int32_t syncEpochBase_{0};
    uint32_t ubMoveElements_{GmmA2avAivDetail::DEFAULT_UB_MOVE_ELEMENTS};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuffer_;
};

template <typename CommOpType, typename ComputationOpType, typename SharedComputationOpType, bool IsNeedMM>
class GmmA2avAivScheduler {
public:
    __aicore__ inline GmmA2avAivScheduler(CommOpType &commOp, ComputationOpType &computeOp,
                                          SharedComputationOpType &sharedComputeOp,
                                          const GroupedMatMulAlltoAllvAivTaskTilingInfo *taskTilingInfo)
        : commOp_(commOp), computeOp_(computeOp), sharedComputeOp_(sharedComputeOp),
          taskTilingInfo_(taskTilingInfo) {}

    __aicore__ inline void Process()
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        constexpr uint16_t READY_FLAG_ID = 10U;
        constexpr uint16_t AIC_BARRIER_FLAG_ID = 8U;
        constexpr uint16_t EXPERT_READY_FLAG_ID = 9U;
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<2, PIPE_V>(READY_FLAG_ID);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(READY_FLAG_ID);
        }
        for (uint32_t expert = 0U; expert < expertNum; ++expert) {
            if ASCEND_IS_AIC {
                computeOp_.Process(expert, 1U);
                AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_BARRIER_FLAG_ID);
                AscendC::CrossCoreWaitFlag(AIC_BARRIER_FLAG_ID);
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(EXPERT_READY_FLAG_ID);
            }
            if ASCEND_IS_AIV {
                AscendC::CrossCoreWaitFlag(EXPERT_READY_FLAG_ID);
                commOp_.LaunchCommData(expert, 1U);
            }
        }
        if constexpr (IsNeedMM) {
            sharedComputeOp_.Process(0U, 1U);
        }
        commOp_.End();
    }

private:
    CommOpType &commOp_;
    ComputationOpType &computeOp_;
    SharedComputationOpType &sharedComputeOp_;
    const GroupedMatMulAlltoAllvAivTaskTilingInfo *taskTilingInfo_;
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_AIV_H
