/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_HCCL_IMPL_H
#define MC2_HCCL_IMPL_H

#include "../a2av_gmm_utils.h"
#include "kernel_operator.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/hccl/hccl.h"
#include "../common/a2av_common_tiling.h"

using namespace AscendC;

namespace MC2KernelTemplate {

template <typename T1, typename T2>
__aicore__ inline T1 Mc2Min(T1 a, T2 b)
{
    return (a < b) ? a : b;
}

template <typename hcclDataType, bool commBeforeComputeFlag>
class HcclA2avOp {
public:
    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
                                const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer, GM_ADDR recvBuffer,
                                GM_ADDR preCommBuffer, GM_ADDR workspaceGM)
    {
        taskTilingInfo_ = taskTilingInfo;
        workspaceGM_ = workspaceGM;
        e_ = taskTilingInfo_->e;
        epWorldSize_ = taskTilingInfo_->epWorldSize;
        H1_ = taskTilingInfo_->H1;
        N1_ = taskTilingInfo_->N1;
        sendGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)sendBuffer);
        recvGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)recvBuffer);
        preCommSendGlobalBuffer_.SetGlobalBuffer((__gm__ uint64_t *)preCommBuffer);
        preCommRecvGlobalBuffer_.SetGlobalBuffer(
            (__gm__ uint64_t *)(preCommBuffer + epWorldSize_ * e_ * epWorldSize_ * sizeof(uint64_t)));

        GM_ADDR hcclContextGm = GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.InitV2(hcclContextGm, hcclInitTiling);
        hccl_.SetCcTilingV2(hcclCcTilingOffset);
        rankId_ = hccl_.GetRankId();
        rankDim_ = hccl_.GetRankDim();

        if constexpr (AscendC::IsSameType<hcclDataType, bfloat16_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_BFP16;
        } else if constexpr (AscendC::IsSameType<hcclDataType, hifloat8_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_HIF8;
        } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e5m2_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_FP8E5M2;
        } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e4m3fn_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_FP8E4M3;
        } else {
            hcclDataType_ = HCCL_DATA_TYPE_FP16;
        }
    }

    __aicore__ inline void InitPreComm()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
            preCommSendGlobalBuffer_.SetValue(i, static_cast<uint64_t>(recvCnt[i]));
        }
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(preCommSendGlobalBuffer_);

        uint64_t sendCntPerRank[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t recvCntPerRank[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t sendOffsetPerRank[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t recvOffsetPerRank[MAX_EP_RANK_SIZE] = {0UL};
        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            sendCntPerRank[i] = e_ * epWorldSize_;
            recvCntPerRank[i] = e_ * epWorldSize_;
        }
        for (uint32_t i = 1U; i < epWorldSize_; i++) {
            sendOffsetPerRank[i] = sendOffsetPerRank[i - 1] + sendCntPerRank[i - 1];
            recvOffsetPerRank[i] = recvOffsetPerRank[i - 1] + recvCntPerRank[i - 1];
        }
        preRecvCountHandle_ = hccl_.AlltoAllV<true>((__gm__ uint8_t *)preCommSendGlobalBuffer_.GetPhyAddr(),
                                                    sendCntPerRank, sendOffsetPerRank, HCCL_DATA_TYPE_UINT64,
                                                    (__gm__ uint8_t *)preCommRecvGlobalBuffer_.GetPhyAddr(),
                                                    recvCntPerRank, recvOffsetPerRank, HCCL_DATA_TYPE_UINT64);
    }

    __aicore__ inline void WaitPreComm()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(preRecvCountHandle_);

        AscendC::printf("rank%u recvCnt from all ranks:\n", rankId_);
        for (uint32_t rankIdx = 0U; rankIdx < epWorldSize_; rankIdx++) {
            uint64_t recvSum = 0;
            AscendC::printf("  rank%u: ", rankIdx);
            for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
                uint64_t val = preCommRecvGlobalBuffer_.GetValue(rankIdx * e_ * epWorldSize_ + i);
                AscendC::printf("%lu ", val);
                recvSum += val;
            }
            AscendC::printf(" sum=%lu\n", recvSum);
        }

        uint64_t myRecvSum = 0;
        uint64_t maxRecvSum = 0;
        for (uint32_t rankIdx = 0U; rankIdx < epWorldSize_; rankIdx++) {
            uint64_t recvSum = 0;
            for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
                recvSum += preCommRecvGlobalBuffer_.GetValue(rankIdx * e_ * epWorldSize_ + i);
            }
            if (rankIdx == rankId_) {
                myRecvSum = recvSum;
            }
            if (recvSum > maxRecvSum) {
                maxRecvSum = recvSum;
            }
        }

        uint32_t globalBaseA = 2048U;
        totalRound_ = static_cast<uint32_t>((maxRecvSum + globalBaseA - 1) / globalBaseA);
        if (totalRound_ == 0) {
            totalRound_ = 1;
        }
        localBaseA_ = static_cast<uint32_t>((myRecvSum + totalRound_ - 1) / totalRound_);
    }

    __aicore__ inline uint32_t GetLocalBaseA() const
    {
        return localBaseA_;
    }
    __aicore__ inline uint32_t GetTotalRound() const
    {
        return totalRound_;
    }

    __aicore__ inline void LaunchMultiRound(uint32_t roundIdx, uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        LaunchCommMultiRound(roundIdx, slotIdx);
    }

    __aicore__ inline void WaitMultiRound(uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvHandleQueue_[slotIdx]);
    }

    __aicore__ inline void Launch(uint32_t startExpertIdx, uint32_t expertNum)
    {
        LaunchMultiRound(0, 0);
    }
    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        WaitMultiRound(0);
    }

    __aicore__ inline void InitScaleBuffer(GM_ADDR sendBuffer, GM_ADDR recvBuffer)
    {
        sendScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)sendBuffer);
        recvScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)recvBuffer);
    }

    __aicore__ inline void LaunchScaleMultiRound(uint32_t roundIdx, uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        uint32_t srcRank = roundIdx % epWorldSize_;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            alltoAllvScaleSendCnt[i] = 0UL;
            alltoAllvScaleRecvCnt[i] = 0UL;
        }

        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * SCALE_MULTIPLIER;

        if (srcRank == rankId_) {
            for (uint32_t dstRank = 0U; dstRank < rankDim_; dstRank++) {
                if (dstRank == rankId_)
                    continue;
                uint64_t sendM = 0;
                for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                    int32_t totalNeed = sendCnt[expertIdx + dstRank * e_];
                    uint32_t alreadySent = roundIdx * localBaseA_;
                    uint32_t remaining = static_cast<uint32_t>(totalNeed) > alreadySent ?
                                             static_cast<uint32_t>(totalNeed) - alreadySent :
                                             0;
                    sendM += Mc2Min(remaining, localBaseA_);
                }
                alltoAllvScaleSendCnt[dstRank] = sendM * axis;
            }
        }

        if (srcRank != rankId_) {
            uint64_t recvM = 0;
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                int32_t totalNeed = recvCnt[expertIdx + srcRank * e_];
                uint32_t alreadyRecv = roundIdx * localBaseA_;
                uint32_t remaining =
                    static_cast<uint32_t>(totalNeed) > alreadyRecv ? static_cast<uint32_t>(totalNeed) - alreadyRecv : 0;
                recvM += Mc2Min(remaining, localBaseA_);
            }
            alltoAllvScaleRecvCnt[srcRank] = recvM * axis;
        }

        alltoAllvScaleSendOffset[0] = 0UL;
        alltoAllvScaleRecvOffset[0] = 0UL;
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvScaleSendOffset[i] = alltoAllvScaleSendOffset[i - 1U] + alltoAllvScaleSendCnt[i - 1U];
            alltoAllvScaleRecvOffset[i] = alltoAllvScaleRecvOffset[i - 1U] + alltoAllvScaleRecvCnt[i - 1U];
        }
        alltoAllvScaleHandleQueue_[slotIdx] = hccl_.AlltoAllV<true>(
            (__gm__ uint8_t *)sendScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleSendCnt, alltoAllvScaleSendOffset,
            HCCL_DATA_TYPE_FP8E8M0, (__gm__ uint8_t *)recvScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleRecvCnt,
            alltoAllvScaleRecvOffset, HCCL_DATA_TYPE_FP8E8M0);
    }

    __aicore__ inline void WaitScaleMultiRound(uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvScaleHandleQueue_[slotIdx]);
    }

    __aicore__ inline void WaitAll(uint32_t allNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        for (uint32_t i = 0U; i < allNum; i++) {
            hccl_.Wait(alltoAllvHandleQueue_[i]);
        }
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            return;
        }
        SyncAll<true>();
        hccl_.Finalize();
    }

private:
    __aicore__ inline void LaunchCommMultiRound(uint32_t roundIdx, uint32_t slotIdx)
    {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        uint32_t srcRank = roundIdx % epWorldSize_;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
        }

        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        uint64_t axis = CeilDiv(H1_, PACK_FACTOR);

        if (srcRank == rankId_) {
            for (uint32_t dstRank = 0U; dstRank < rankDim_; dstRank++) {
                if (dstRank == rankId_)
                    continue;
                uint64_t sendM = 0;
                for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                    int32_t totalNeed = sendCnt[expertIdx + dstRank * e_];
                    uint32_t alreadySent = roundIdx * localBaseA_;
                    uint32_t remaining = static_cast<uint32_t>(totalNeed) > alreadySent ?
                                             static_cast<uint32_t>(totalNeed) - alreadySent :
                                             0;
                    sendM += Mc2Min(remaining, localBaseA_);
                }
                alltoAllvSendCnt[dstRank] = sendM * axis;
            }
        }

        if (srcRank != rankId_) {
            uint64_t recvM = 0;
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                int32_t totalNeed = recvCnt[expertIdx + srcRank * e_];
                uint32_t alreadyRecv = roundIdx * localBaseA_;
                uint32_t remaining =
                    static_cast<uint32_t>(totalNeed) > alreadyRecv ? static_cast<uint32_t>(totalNeed) - alreadyRecv : 0;
                recvM += Mc2Min(remaining, localBaseA_);
            }
            alltoAllvRecvCnt[srcRank] = recvM * axis;
        }

        alltoAllvSendOffset[0] = 0UL;
        alltoAllvRecvOffset[0] = 0UL;
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U] + alltoAllvSendCnt[i - 1U];
            alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1U] + alltoAllvRecvCnt[i - 1U];
        }

        alltoAllvHandleQueue_[slotIdx] = hccl_.AlltoAllV<true>(
            (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
            (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);

        uint64_t totalSend = 0, totalRecv = 0;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            totalSend += alltoAllvSendCnt[i];
            totalRecv += alltoAllvRecvCnt[i];
        }
        AscendC::printf("rank%u round%u src%u send=%lu recv=%lu\n", rankId_, roundIdx, srcRank, totalSend, totalRecv);
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;
#endif

    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
    static constexpr uint64_t HANDLE_QUEUE_SIZE = 64U;
    static constexpr uint64_t SCALE_ALIGNMENT_BLOCK_SIZE = 64U;
    static constexpr uint64_t SCALE_MULTIPLIER = 2U;

    const TaskTilingInfo *taskTilingInfo_;
    GM_ADDR workspaceGM_ = nullptr;

    uint32_t rankId_ = 0U;
    uint32_t rankDim_ = 0U;
    uint32_t e_ = 0U;
    uint32_t epWorldSize_ = 0U;
    uint64_t H1_ = 0UL;
    uint64_t N1_ = 0UL;
    uint32_t localBaseA_ = 0U;
    uint32_t totalRound_ = 0U;

    GlobalTensor<hcclDataType> sendGlobalBuffer_;
    GlobalTensor<hcclDataType> recvGlobalBuffer_;

    GlobalTensor<fp8_e8m0_t> sendScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> recvScaleGlobalBuffer_;
    GlobalTensor<uint64_t> preCommSendGlobalBuffer_;
    GlobalTensor<uint64_t> preCommRecvGlobalBuffer_;

    HcclHandle alltoAllvHandleQueue_[HANDLE_QUEUE_SIZE] = {INVALID_HANDLE_ID};
    HcclHandle alltoAllvScaleHandleQueue_[HANDLE_QUEUE_SIZE] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;

    HcclHandle preRecvCountHandle_ = INVALID_HANDLE_ID;

    uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

    uint64_t alltoAllvScaleSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
};
}; // namespace MC2KernelTemplate

#endif