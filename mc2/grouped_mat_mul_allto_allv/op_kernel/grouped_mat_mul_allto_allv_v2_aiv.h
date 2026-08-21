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
 * \file grouped_mat_mul_allto_allv_v2_aiv.h
 * \brief Correctness-first AIV communication for GroupedMatMulAlltoAllvV2 on Ascend 910_93.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_V2_AIV_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_V2_AIV_H

#include "kernel_operator.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "grouped_mat_mul_allto_allv_v2_tiling.h"

namespace MC2KernelTemplate {

namespace GmmA2avAivDetail {
constexpr uint64_t COUNT_REGION_FROM_TAIL = 16UL * 1024UL * 1024UL;
constexpr uint64_t SYNC_REGION_FROM_TAIL = 512UL * 1024UL;
constexpr uint64_t SYNC_SLOT_BYTES = 64UL;
constexpr uint32_t DEFAULT_UB_MOVE_ELEMENTS = Gmma2avV2Tiling::UB_MOVE_ELEMENTS;

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

    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
                                const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                                GM_ADDR recvBuffer, uint32_t aivCoreNum = 1U,
                                const GmmA2avCoCTiling *cocTiling = nullptr,
                                uint64_t commBufferSize = 0UL)
    {
        (void)hcclInitTiling;
        (void)hcclCcTilingOffset;
        taskTilingInfo_ = taskTilingInfo;
        sendBuffer_ = reinterpret_cast<__gm__ T *>(sendBuffer);
        recvBuffer_ = reinterpret_cast<__gm__ T *>(recvBuffer);
        aivCoreNum_ = aivCoreNum;
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
        // Correctness-first implementation: one AIV serializes the complete
        // communication phase. SyncAll<true> cannot be used with the current
        // MIX block layout, while the kernel-wide final synchronization remains.
        if (coreIdx_ != 0U) {
            return;
        }
        aivCoreNum_ = 1U;
        if (cocTiling != nullptr && cocTiling->ubMoveNum > 0) {
            ubMoveElements_ = static_cast<uint32_t>(cocTiling->ubMoveNum);
        }

        GetTPipePtr()->InitBuffer(ubBuffer_, ubMoveElements_ * sizeof(T));
        ExchangeCountMatrix();
        CrossRankSync();
    }

    __aicore__ inline void LaunchCommData(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (coreIdx_ != 0U) {
            return;
        }
        // CATLASS intentionally emits all local experts in one expert-major batch.
        if (startExpertIdx != 0U || expertNum != taskTilingInfo_->e) {
            return;
        }
        uint64_t expertSourceBase = 0UL;
        for (uint32_t expert = 0U; expert < taskTilingInfo_->e; ++expert) {
            uint64_t sourceOffsetInExpert = 0UL;
            for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                uint64_t sendTokenCount = static_cast<uint64_t>(
                    taskTilingInfo_->sendCnt[dstRank * taskTilingInfo_->e + expert]);
                uint64_t pairIdx = static_cast<uint64_t>(expert) * rankSize_ + dstRank;
                if ((pairIdx % aivCoreNum_) == coreIdx_ && sendTokenCount != 0UL) {
                    uint64_t remoteTokenOffset = GetRemoteRecvTokenOffset(dstRank, expert);
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
            if (coreIdx_ == 0U) {
                AscendC::PipeBarrier<PIPE_ALL>();
                CrossRankSync();
                CopyLocalResult();
            }
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
        return reinterpret_cast<__gm__ int32_t *>(GetWindow(rank) + winSize_ -
                                                  GmmA2avAivDetail::COUNT_REGION_FROM_TAIL);
    }

    __aicore__ inline void ExchangeCountMatrix()
    {
        uint64_t countNum = static_cast<uint64_t>(rankSize_) * taskTilingInfo_->e;
        for (uint64_t linearIdx = coreIdx_; linearIdx < countNum; linearIdx += aivCoreNum_) {
            uint32_t dstRank = static_cast<uint32_t>(linearIdx / taskTilingInfo_->e);
            uint32_t expert = static_cast<uint32_t>(linearIdx % taskTilingInfo_->e);
            __gm__ int32_t *remoteCount = GetCountMatrix(dstRank) + rank_ * taskTilingInfo_->e + expert;
            GmmA2avAivDetail::GmStore(remoteCount, taskTilingInfo_->sendCnt[dstRank * taskTilingInfo_->e + expert]);
            GmmA2avAivDetail::Dcci(remoteCount);
        }
    }

    __aicore__ inline uint64_t GetRemoteRecvTokenOffset(uint32_t dstRank, uint32_t expertIdx) const
    {
        __gm__ int32_t *remoteCountMatrix = GetCountMatrix(dstRank);
        uint64_t tokenOffset = 0UL;
        for (uint32_t sourceRank = 0U; sourceRank < rank_; ++sourceRank) {
            for (uint32_t expert = 0U; expert < taskTilingInfo_->e; ++expert) {
                __gm__ int32_t *count = remoteCountMatrix + sourceRank * taskTilingInfo_->e + expert;
                GmmA2avAivDetail::Dcci(count);
                tokenOffset += static_cast<uint64_t>(GmmA2avAivDetail::GmLoad(count));
            }
        }
        for (uint32_t expert = 0U; expert < expertIdx; ++expert) {
            __gm__ int32_t *count = remoteCountMatrix + rank_ * taskTilingInfo_->e + expert;
            GmmA2avAivDetail::Dcci(count);
            tokenOffset += static_cast<uint64_t>(GmmA2avAivDetail::GmLoad(count));
        }
        return tokenOffset;
    }

    __aicore__ inline void CrossRankSync()
    {
        // Complete every local AIV write before publishing the cross-rank
        // completion flag. CCL window memory is not guaranteed to start at 0,
        // so use deterministic alternating epochs instead of reading a counter.
        AscendC::PipeBarrier<PIPE_ALL>();
        constexpr int32_t SYNC_MAGIC_PHASE0 = 0x13572468;
        constexpr int32_t SYNC_MAGIC_PHASE1 = 0x24681357;
        int32_t epoch = (syncPhase_++ & 1U) == 0U ? SYNC_MAGIC_PHASE0 : SYNC_MAGIC_PHASE1;
        uint64_t syncOffset = winSize_ - GmmA2avAivDetail::SYNC_REGION_FROM_TAIL;
        for (uint32_t dstRank = coreIdx_; dstRank < rankSize_; dstRank += aivCoreNum_) {
            __gm__ int32_t *remoteSlot = reinterpret_cast<__gm__ int32_t *>(
                GetWindow(dstRank) + syncOffset +
                static_cast<uint64_t>(rank_) * GmmA2avAivDetail::SYNC_SLOT_BYTES);
            StoreSyncFlag(remoteSlot, epoch);
            __gm__ int32_t *localSlot = reinterpret_cast<__gm__ int32_t *>(
                GetWindow(rank_) + syncOffset +
                static_cast<uint64_t>(dstRank) * GmmA2avAivDetail::SYNC_SLOT_BYTES);
            WaitSyncFlag(localSlot, epoch);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreSyncFlag(__gm__ int32_t *destination, int32_t flag)
    {
        AscendC::LocalTensor<int32_t> local = ubBuffer_.template Get<int32_t>();
        local.SetValue(0U, flag);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::GlobalTensor<int32_t> global;
        global.SetGlobalBuffer(destination);
        AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
        AscendC::DataCopyPad(global, local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
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
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
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
        uint64_t copied = 0UL;
        while (copied < elementCount) {
            uint32_t current = static_cast<uint32_t>(elementCount - copied >
                ubMoveElements_ ? ubMoveElements_ :
                elementCount - copied);
            AscendC::GlobalTensor<T> sourceGlobal;
            sourceGlobal.SetGlobalBuffer(source + copied);
            AscendC::DataCopyExtParams copyParams(1U, current * sizeof(T), 0U, 0U, 0U);
            AscendC::DataCopyPadExtParams<T> padParams;
            AscendC::DataCopyPad(local, sourceGlobal, copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::GlobalTensor<T> destinationGlobal;
            destinationGlobal.SetGlobalBuffer(destination + copied);
            AscendC::DataCopyPad(destinationGlobal, local, copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            copied += current;
        }
    }

private:
    const TaskTilingInfo *taskTilingInfo_{nullptr};
    __gm__ HcclOpResParam *winContext_{nullptr};
    __gm__ T *sendBuffer_{nullptr};
    __gm__ T *recvBuffer_{nullptr};
    uint64_t winSize_{0UL};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    uint32_t coreIdx_{0U};
    uint32_t aivCoreNum_{1U};
    uint32_t syncPhase_{0U};
    uint32_t ubMoveElements_{GmmA2avAivDetail::DEFAULT_UB_MOVE_ELEMENTS};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuffer_;
};

template <typename T, int CommMode>
struct GmmA2avCommSelector {
    using type = HcclA2avOp<T, false, CommMode>;
};

template <typename T>
struct GmmA2avCommSelector<T, TILINGKEY_TPL_AIV> {
    using type = GmmA2avAivOp<T>;
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_V2_AIV_H
