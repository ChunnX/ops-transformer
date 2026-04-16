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

        for (int i = 0; i < e_ * rankDim_; i++) {
            recvCounts_[i] = taskTilingInfo_->recvCnt[i];
            sendCounts_[i] = taskTilingInfo_->sendCnt[i];
        }
        ubSize_ = taskTilingInfo_->ubSize;

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

        progressExpertIdx_ = 0U;
        progressExpertOffset_ = 0;
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

        for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
            preCommSendGlobalBuffer_.SetValue(i, static_cast<uint64_t>(recvCounts_[i]));
        }
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(preCommSendGlobalBuffer_);

        uint64_t sendCntPerRank[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t recvCntPerRank[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t sendOffsetPerRank[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t recvOffsetPerRank[MAX_EP_RANK_SIZE] = {0UL};

        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            recvCntPerRank[i] = e_ * epWorldSize_ * 8;
            sendCntPerRank[i] = e_ * epWorldSize_ * 8;
        }
        for (uint32_t i = 1U; i < epWorldSize_; i++) {
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

        uint64_t myRecvSum = 0;
        uint64_t maxRecvSum = 0;
        uint32_t maxRecvRank = 0;
        for (uint32_t rankIdx = 0U; rankIdx < epWorldSize_; rankIdx++) {
            uint64_t recvSum = 0;
            for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
                recvSum += preCommRecvGlobalBuffer_.GetValue(rankIdx * e_ * epWorldSize_ + i);
            }
            recvSumPerRank_[rankIdx] = recvSum;
            if (rankIdx == rankId_) {
                myRecvSum = recvSum;
            }
            if (recvSum > maxRecvSum || (recvSum == maxRecvSum && rankIdx < maxRecvRank)) {
                maxRecvSum = recvSum;
                maxRecvRank = rankIdx;
            }
        }

        constexpr uint32_t MAX_HANDLE_NUM = 63U;
        constexpr uint32_t MX_MAX_HANDLE_NUM = 31U;
        constexpr uint32_t DEFAULT_BASE_A = 2048U;

        uint32_t maxHandleNum = MAX_HANDLE_NUM;
        if constexpr (IsSameType<hcclDataType, fp8_e8m0_t>::value) {
            maxHandleNum = MX_MAX_HANDLE_NUM;
        }

        uint32_t minBaseA = static_cast<uint32_t>((maxRecvSum + maxHandleNum - 1) / maxHandleNum);
        uint32_t globalBaseA = (minBaseA > DEFAULT_BASE_A) ? minBaseA : DEFAULT_BASE_A;

        totalRound_ = static_cast<uint32_t>((maxRecvSum + globalBaseA - 1) / globalBaseA);
        if (totalRound_ == 0) {
            totalRound_ = 1;
        }
        if (totalRound_ > maxHandleNum) {
            totalRound_ = maxHandleNum;
            globalBaseA = static_cast<uint32_t>((maxRecvSum + totalRound_ - 1) / totalRound_);
        }

        for (uint32_t rankIdx = 0U; rankIdx < epWorldSize_; rankIdx++) {
            baseAPerRank_[rankIdx] = static_cast<uint32_t>((recvSumPerRank_[rankIdx] + totalRound_ - 1) / totalRound_);
            recvProgressExpertIdx_[rankIdx] = 0U;
            recvProgressExpertOffset_[rankIdx] = 0;
        }
        localBaseA_ = baseAPerRank_[rankId_];
    }

    __aicore__ inline uint32_t GetLocalBaseA() const
    {
        return localBaseA_;
    }
    __aicore__ inline uint32_t GetTotalRound() const
    {
        return totalRound_;
    }
    __aicore__ inline uint32_t GetRankId() const
    {
        return rankId_;
    }

    __aicore__ inline void LaunchMultiRound(uint32_t roundIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        LaunchCommMultiRound(roundIdx);
    }

    __aicore__ inline void WaitMultiRound(uint32_t roundIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        // hccl_.Wait(alltoAllvHandleQueue_[roundIdx]);
    }

    __aicore__ inline void Launch(uint32_t startExpertIdx, uint32_t expertNum)
    {
        LaunchMultiRound(0);
    }
    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        WaitMultiRound(0);
    }

    __aicore__ inline void InitScaleBuffer(GM_ADDR sendBuffer, GM_ADDR commOutBuffer, GM_ADDR permuteOutBuffer)
    {
        sendScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)sendBuffer);
        scaleCommOutBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)commOutBuffer);
        scalePermuteOutBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)permuteOutBuffer);
        TPipe *pipe = GetTPipePtr();
        pipe->InitBuffer(scaleTBuf_, taskTilingInfo_->ubSize);
    }

    __aicore__ inline void LaunchScaleAllOnce()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }

        uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * SCALE_MULTIPLIER;

        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvScaleSendCnt[i] = 0UL;
            alltoAllvScaleRecvCnt[i] = 0UL;
            for (uint64_t expertIdx = 0UL; expertIdx < e_; expertIdx++) {
                alltoAllvScaleSendCnt[i] += static_cast<uint64_t>(sendCounts_[expertIdx + i * e_]) * axis;
                alltoAllvScaleRecvCnt[i] += static_cast<uint64_t>(recvCounts_[expertIdx + i * e_]) * axis;
            }
        }

        alltoAllvScaleSendOffset[0] = 0UL;
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvScaleSendOffset[i] = alltoAllvScaleSendOffset[i - 1U];
            for (uint32_t j = 0U; j < e_; j++) {
                alltoAllvScaleSendOffset[i] += static_cast<uint64_t>(sendCounts_[j + (i - 1U) * e_]) * axis;
            }
        }

        alltoAllvScaleRecvOffset[0] = 0UL;
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvScaleRecvOffset[i] = alltoAllvScaleRecvOffset[i - 1U];
            for (uint32_t j = 0U; j < e_; j++) {
                alltoAllvScaleRecvOffset[i] += static_cast<uint64_t>(recvCounts_[(i - 1U) * e_ + j]) * axis;
            }
        }

        // scaleBatchHandle_ = hccl_.AlltoAllV<true>(
        //     (__gm__ uint8_t *)sendScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleSendCnt, alltoAllvScaleSendOffset,
        //     HCCL_DATA_TYPE_FP8E8M0, (__gm__ uint8_t *)scaleCommOutBuffer_.GetPhyAddr(), alltoAllvScaleRecvCnt,
        //     alltoAllvScaleRecvOffset, HCCL_DATA_TYPE_FP8E8M0);
    }

    __aicore__ inline void WaitScaleBatch()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        // hccl_.Wait(scaleBatchHandle_);
    }

    __aicore__ inline void PermuteScale()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }

        uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * SCALE_MULTIPLIER;
        uint64_t scaleUbSize = ubSize_;

        uint64_t recvSumCntBefore[MAX_EXPERT_SIZE] = {0UL};
        uint64_t recvSumCntAfter[MAX_EXPERT_SIZE] = {0UL};

        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                uint32_t posBefore = rankIndex * e_ + expertIdx;
                recvSumCntBefore[posBefore] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]);
                if (posBefore != 0U) {
                    recvSumCntBefore[posBefore] += recvSumCntBefore[posBefore - 1U];
                }
            }
        }

        for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
            for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
                uint32_t posAfter = expertIdx * rankDim_ + rankIndex;
                recvSumCntAfter[posAfter] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]);
                if (posAfter != 0U) {
                    recvSumCntAfter[posAfter] += recvSumCntAfter[posAfter - 1U];
                }
            }
        }

        TPipe *permutePipe = GetTPipePtr();
        LocalTensor<fp8_e8m0_t> scaleUbTensor = scaleTBuf_.Get<fp8_e8m0_t>();
        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                uint32_t posBefore = rankIndex * e_ + expertIdx;
                uint32_t posAfter = expertIdx * rankDim_ + rankIndex;
                uint64_t srcOffset = (posBefore == 0U) ? 0UL : recvSumCntBefore[posBefore - 1U] * axis;
                uint64_t dstOffset = (posAfter == 0U) ? 0UL : recvSumCntAfter[posAfter - 1U] * axis;
                uint64_t totalCnt = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]) * axis;

                uint64_t tileNum = CeilDiv(totalCnt, scaleUbSize);
                for (uint64_t tile = 0UL; tile < tileNum; tile++) {
                    uint64_t realLength = scaleUbSize;
                    if (tile == tileNum - 1UL) {
                        realLength = totalCnt - tile * scaleUbSize;
                    }
                    DataCopyParams dataCopyParams{1U, static_cast<uint16_t>(realLength), 0U, 0U};
                    DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};
                    DataCopyPad(scaleUbTensor, scaleCommOutBuffer_[srcOffset + tile * scaleUbSize], dataCopyParams,
                                dataCopyPadParams);
                    eventID_ = static_cast<int32_t>(permutePipe->FetchEventID<HardEvent::MTE2_MTE3>());
                    SetFlag<HardEvent::MTE2_MTE3>(eventID_);
                    WaitFlag<HardEvent::MTE2_MTE3>(eventID_);
                    DataCopyPad(scalePermuteOutBuffer_[dstOffset + tile * scaleUbSize], scaleUbTensor, dataCopyParams);
                    eventID_ = static_cast<int32_t>(permutePipe->FetchEventID<HardEvent::MTE3_MTE2>());
                    SetFlag<HardEvent::MTE3_MTE2>(eventID_);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventID_);
                }
            }
        }
    }

    __aicore__ inline void LaunchScaleMultiRound(uint32_t roundIdx)
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

        uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * SCALE_MULTIPLIER;

        if (srcRank == rankId_) {
            for (uint32_t dstRank = 0U; dstRank < rankDim_; dstRank++) {
                if (dstRank == rankId_)
                    continue;
                uint64_t sendM = 0;
                for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                    int32_t totalNeed = sendCounts_[expertIdx + dstRank * e_];
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
                int32_t totalNeed = recvCounts_[expertIdx + srcRank * e_];
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

        // alltoAllvScaleHandleQueue_[roundIdx] = hccl_.AlltoAllV<true>(
        //     (__gm__ uint8_t *)sendScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleSendCnt, alltoAllvScaleSendOffset,
        //     HCCL_DATA_TYPE_FP8E8M0, (__gm__ uint8_t *)scalePermuteOutBuffer_.GetPhyAddr(), alltoAllvScaleRecvCnt,
        //     alltoAllvScaleRecvOffset, HCCL_DATA_TYPE_FP8E8M0);
    }

    __aicore__ inline void WaitScaleMultiRound(uint32_t roundIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        // hccl_.Wait(alltoAllvScaleHandleQueue_[roundIdx]);
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
        // for (uint32_t i = 0U; i < allNum; i++) {
        //     hccl_.Wait(alltoAllvHandleQueue_[i]);
        // }
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
    __aicore__ inline void LaunchCommMultiRound(uint32_t roundIdx)
    {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }

        for (uint32_t i = 0U; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
        }

        uint64_t axis = CeilDiv(H1_, PACK_FACTOR);

        int32_t recvAccumulated = 0;

        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0 && roundIdx < 1U) {
                AscendC::printf(
                    "[RecvStart] r%u round%u: expIdx=%u offset=%d baseA=%u epSize=%u e=%u totalExperts=%u\n", rankId_,
                    roundIdx, progressExpertIdx_, progressExpertOffset_, localBaseA_, epWorldSize_, e_,
                    e_ * epWorldSize_);
            }
        }

        uint32_t recvExpertNum = e_ * epWorldSize_ - progressExpertIdx_;
        for (uint32_t recvLoop = 0U; recvLoop < recvExpertNum && recvAccumulated < static_cast<int32_t>(localBaseA_);
             recvLoop++) {
            uint32_t currExpertIdx = progressExpertIdx_;
            int32_t currOffset = progressExpertOffset_;
            int32_t remaining = recvCounts_[currExpertIdx] - currOffset;
            int32_t toAdd = Mc2Min(remaining, static_cast<int32_t>(localBaseA_) - recvAccumulated);

            if ASCEND_IS_AIV {
                if (GetBlockIdx() == 0 && roundIdx < 1U && recvLoop < 5U) {
                    AscendC::printf(
                        "[RecvLoop] r%u loop%u: expIdx=%u offset=%d recvCnt=%d remaining=%d toAdd=%d acc=%d\n", rankId_,
                        recvLoop, currExpertIdx, currOffset, recvCounts_[currExpertIdx], remaining, toAdd,
                        recvAccumulated);
                }
            }

            recvAccumulated += toAdd;

            uint32_t srcRank = static_cast<uint32_t>(currExpertIdx / e_);
            alltoAllvRecvCnt[srcRank] += static_cast<uint64_t>(toAdd) * axis;

            progressExpertOffset_ = currOffset + toAdd;
            if (progressExpertOffset_ >= recvCounts_[currExpertIdx]) {
                progressExpertIdx_++;
                progressExpertOffset_ = 0;
            }
        }

        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0 && roundIdx < 1U) {
                AscendC::printf("[RecvEnd] r%u: finalExpIdx=%u finalOffset=%d finalAcc=%d\n", rankId_,
                                progressExpertIdx_, progressExpertOffset_, recvAccumulated);
            }
        }

        for (uint32_t dstRank = 0U; dstRank < rankDim_; dstRank++) {
            uint32_t dstExpertIdx = recvProgressExpertIdx_[dstRank];
            int32_t dstOffset = recvProgressExpertOffset_[dstRank];
            int32_t dstAccumulated = 0;
            int32_t sendAmount = 0;

            if ASCEND_IS_AIV {
                if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank < 2U) {
                    AscendC::printf("[SendStart] r%u dstRank=%u: expIdx=%u offset=%d baseA=%u totalExperts=%u\n",
                                    rankId_, dstRank, dstExpertIdx, dstOffset, baseAPerRank_[dstRank],
                                    e_ * epWorldSize_);
                }
            }

            uint32_t sendExpertNum = e_ * epWorldSize_ - dstExpertIdx;
            for (uint32_t sendLoop = 0U;
                 sendLoop < sendExpertNum && dstAccumulated < static_cast<int32_t>(baseAPerRank_[dstRank]);
                 sendLoop++) {
                uint32_t currExpertIdx = dstExpertIdx;
                int32_t currOffset = dstOffset;
                uint64_t dstRecvCntValue =
                    preCommRecvGlobalBuffer_.GetValue(dstRank * e_ * epWorldSize_ + currExpertIdx);
                int32_t remaining = static_cast<int32_t>(dstRecvCntValue) - currOffset;
                int32_t toAdd = Mc2Min(remaining, static_cast<int32_t>(baseAPerRank_[dstRank]) - dstAccumulated);

                if ASCEND_IS_AIV {
                    if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank < 2U && sendLoop < 5U) {
                        AscendC::printf("[SendLoop] r%u dstRank=%u loop%u: expIdx=%u offset=%d recvCnt=%lu "
                                        "remaining=%d toAdd=%d acc=%d\n",
                                        rankId_, dstRank, sendLoop, currExpertIdx, currOffset, dstRecvCntValue,
                                        remaining, toAdd, dstAccumulated);
                    }
                }

                uint32_t srcRankOfExpert = static_cast<uint32_t>(currExpertIdx / e_);
                if (srcRankOfExpert == rankId_) {
                    sendAmount += toAdd;
                }

                dstAccumulated += toAdd;
                dstOffset = currOffset + toAdd;
                if (dstOffset >= static_cast<int32_t>(dstRecvCntValue)) {
                    dstExpertIdx++;
                    dstOffset = 0;
                }
            }

            if ASCEND_IS_AIV {
                if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank < 2U) {
                    AscendC::printf("[SendEnd] r%u dstRank=%u: finalExpIdx=%u finalOffset=%d finalAcc=%d sendAmt=%d\n",
                                    rankId_, dstRank, dstExpertIdx, dstOffset, dstAccumulated, sendAmount);
                }
            }

            recvProgressExpertIdx_[dstRank] = dstExpertIdx;
            recvProgressExpertOffset_[dstRank] = dstOffset;
            alltoAllvSendCnt[dstRank] = static_cast<uint64_t>(sendAmount) * axis;
        }

        alltoAllvSendOffset[0] = 0UL;
        alltoAllvRecvOffset[0] = 0UL;
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U] + alltoAllvSendCnt[i - 1U];
            alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1U] + alltoAllvRecvCnt[i - 1U];
        }

        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0 && roundIdx < 2U) {
                uint64_t totalSend = 0, totalRecv = 0;
                for (uint32_t i = 0U; i < rankDim_; i++) {
                    totalSend += alltoAllvSendCnt[i];
                    totalRecv += alltoAllvRecvCnt[i];
                }
                AscendC::printf("[Comm] r%u round%u: sendCnt=[%lu %lu] recvCnt=[%lu %lu] total=%lu/%lu\n", rankId_,
                                roundIdx, alltoAllvSendCnt[0], alltoAllvSendCnt[1], alltoAllvRecvCnt[0],
                                alltoAllvRecvCnt[1], totalSend, totalRecv);
            }
        }

        // alltoAllvHandleQueue_[roundIdx] = hccl_.AlltoAllV<true>(
        //     (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
        //     (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
    }
}

for (uint32_t i = 0U; i < rankDim_; i++)
{
    alltoAllvSendCnt[i] = 0UL;
    alltoAllvRecvCnt[i] = 0UL;
}

uint64_t axis = CeilDiv(H1_, PACK_FACTOR);

int32_t recvAccumulated = 0;
uint32_t recvStartExpert = progressExpertIdx_;
int32_t recvStartOffset = progressExpertOffset_;

if ASCEND_IS_AIV {
    if (GetBlockIdx() == 0 && roundIdx < 1U) {
        AscendC::printf("[RecvStart] r%u round%u: expIdx=%u offset=%d baseA=%u epSize=%u e=%u\n", rankId_, roundIdx,
                        progressExpertIdx_, progressExpertOffset_, localBaseA_, epWorldSize_, e_);
    }
}

while (recvAccumulated < static_cast<int32_t>(localBaseA_) && progressExpertIdx_ < e_ * epWorldSize_) {
    int32_t remaining = recvCounts_[progressExpertIdx_] - progressExpertOffset_;
    int32_t toAdd = Mc2Min(remaining, static_cast<int32_t>(localBaseA_) - recvAccumulated);

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0 && roundIdx < 1U) {
            AscendC::printf("[RecvLoop] r%u: expIdx=%u offset=%d recvCnt=%d remaining=%d toAdd=%d acc=%d\n", rankId_,
                            progressExpertIdx_, progressExpertOffset_, recvCounts_[progressExpertIdx_], remaining,
                            toAdd, recvAccumulated);
        }
    }

    recvAccumulated += toAdd;

    uint32_t srcRank = static_cast<uint32_t>(progressExpertIdx_ / e_);
    alltoAllvRecvCnt[srcRank] += static_cast<uint64_t>(toAdd) * axis;

    progressExpertOffset_ += toAdd;
    if (progressExpertOffset_ >= recvCounts_[progressExpertIdx_]) {
        progressExpertIdx_++;
        progressExpertOffset_ = 0;
    }
}

if ASCEND_IS_AIV {
    if (GetBlockIdx() == 0 && roundIdx < 1U) {
        AscendC::printf("[RecvEnd] r%u: finalExpIdx=%u finalOffset=%d finalAcc=%d\n", rankId_, progressExpertIdx_,
                        progressExpertOffset_, recvAccumulated);
    }
}

for (uint32_t dstRank = 0U; dstRank < rankDim_; dstRank++) {
    uint32_t dstExpertIdx = recvProgressExpertIdx_[dstRank];
    int32_t dstOffset = recvProgressExpertOffset_[dstRank];
    int32_t dstAccumulated = 0;
    int32_t sendAmount = 0;

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank < 2U) {
            AscendC::printf("[SendStart] r%u dstRank=%u: expIdx=%u offset=%d baseA=%u\n", rankId_, dstRank,
                            dstExpertIdx, dstOffset, baseAPerRank_[dstRank]);
        }
    }

    while (dstAccumulated < static_cast<int32_t>(baseAPerRank_[dstRank]) && dstExpertIdx < e_ * epWorldSize_) {
        uint64_t dstRecvCntValue = preCommRecvGlobalBuffer_.GetValue(dstRank * e_ * epWorldSize_ + dstExpertIdx);
        int32_t remaining = static_cast<int32_t>(dstRecvCntValue) - dstOffset;
        int32_t toAdd = Mc2Min(remaining, static_cast<int32_t>(baseAPerRank_[dstRank]) - dstAccumulated);

        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank < 2U) {
                AscendC::printf(
                    "[SendLoop] r%u dstRank=%u: expIdx=%u offset=%d recvCnt=%lu remaining=%d toAdd=%d acc=%d\n",
                    rankId_, dstRank, dstExpertIdx, dstOffset, dstRecvCntValue, remaining, toAdd, dstAccumulated);
            }
        }

        uint32_t srcRankOfExpert = static_cast<uint32_t>(dstExpertIdx / e_);
        if (srcRankOfExpert == rankId_) {
            sendAmount += toAdd;
        }

        dstAccumulated += toAdd;
        dstOffset += toAdd;
        if (dstOffset >= static_cast<int32_t>(dstRecvCntValue)) {
            dstExpertIdx++;
            dstOffset = 0;
        }
    }

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank < 2U) {
            AscendC::printf("[SendEnd] r%u dstRank=%u: finalExpIdx=%u finalOffset=%d finalAcc=%d sendAmt=%d\n", rankId_,
                            dstRank, dstExpertIdx, dstOffset, dstAccumulated, sendAmount);
        }
    }

    recvProgressExpertIdx_[dstRank] = dstExpertIdx;
    recvProgressExpertOffset_[dstRank] = dstOffset;
    alltoAllvSendCnt[dstRank] = static_cast<uint64_t>(sendAmount) * axis;
}

alltoAllvSendOffset[0] = 0UL;
alltoAllvRecvOffset[0] = 0UL;
for (uint32_t i = 1U; i < rankDim_; i++) {
    alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U] + alltoAllvSendCnt[i - 1U];
    alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1U] + alltoAllvRecvCnt[i - 1U];
}

if ASCEND_IS_AIV {
    if (GetBlockIdx() == 0 && roundIdx < 2U) {
        uint64_t totalSend = 0, totalRecv = 0;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            totalSend += alltoAllvSendCnt[i];
            totalRecv += alltoAllvRecvCnt[i];
        }
        AscendC::printf("[Comm] r%u round%u: sendCnt=[%lu %lu] recvCnt=[%lu %lu] total=%lu/%lu\n", rankId_, roundIdx,
                        alltoAllvSendCnt[0], alltoAllvSendCnt[1], alltoAllvRecvCnt[0], alltoAllvRecvCnt[1], totalSend,
                        totalRecv);
    }
}

// alltoAllvHandleQueue_[roundIdx] = hccl_.AlltoAllV<true>(
//     (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
//     (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
}
}

for (uint32_t i = 0U; i < rankDim_; i++) {
    alltoAllvSendCnt[i] = 0UL;
    alltoAllvRecvCnt[i] = 0UL;
}

uint64_t axis = CeilDiv(H1_, PACK_FACTOR);

int32_t recvAccumulated = 0;
uint32_t recvStartExpert = progressExpertIdx_;
int32_t recvStartOffset = progressExpertOffset_;
uint32_t recvEndExpert = progressExpertIdx_;
int32_t recvEndOffset = progressExpertOffset_;

constexpr int32_t MAX_RECV_LOOP = 100;
int32_t recvLoopCount = 0;
for (recvLoopCount = 0; recvLoopCount < MAX_RECV_LOOP; recvLoopCount++) {
    if (recvAccumulated >= static_cast<int32_t>(localBaseA_) || progressExpertIdx_ >= e_ * epWorldSize_) {
        break;
    }
    int32_t remaining = recvCounts_[progressExpertIdx_] - progressExpertOffset_;
    int32_t toAdd = Mc2Min(remaining, static_cast<int32_t>(localBaseA_) - recvAccumulated);

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0 && roundIdx < 1U && recvLoopCount < 3) {
            AscendC::printf("[RecvLoop] r%u loop%d: expIdx=%u offset=%d remaining=%d toAdd=%d acc=%d baseA=%u\n",
                            rankId_, recvLoopCount, progressExpertIdx_, progressExpertOffset_, remaining, toAdd,
                            recvAccumulated, localBaseA_);
        }
    }

    recvAccumulated += toAdd;
    recvEndExpert = progressExpertIdx_;
    recvEndOffset = progressExpertOffset_ + toAdd;

    uint32_t srcRank = static_cast<uint32_t>(progressExpertIdx_ / e_);
    alltoAllvRecvCnt[srcRank] += static_cast<uint64_t>(toAdd) * axis;

    progressExpertOffset_ += toAdd;
    if (progressExpertOffset_ >= recvCounts_[progressExpertIdx_]) {
        progressExpertIdx_++;
        progressExpertOffset_ = 0;
    }
}

if ASCEND_IS_AIV {
    if (GetBlockIdx() == 0 && roundIdx < 1U) {
        AscendC::printf("[RecvLoop] r%u totalLoop=%d finalExpIdx=%u finalOffset=%d finalAcc=%d\n", rankId_,
                        recvLoopCount, progressExpertIdx_, progressExpertOffset_, recvAccumulated);
    }
}

for (uint32_t dstRank = 0U; dstRank < rankDim_; dstRank++) {
    uint32_t dstExpertIdx = recvProgressExpertIdx_[dstRank];
    int32_t dstOffset = recvProgressExpertOffset_[dstRank];
    int32_t dstAccumulated = 0;
    int32_t sendAmount = 0;

    constexpr int32_t MAX_SEND_LOOP = 100;
    int32_t sendLoopCount = 0;
    for (sendLoopCount = 0; sendLoopCount < MAX_SEND_LOOP; sendLoopCount++) {
        if (dstAccumulated >= static_cast<int32_t>(baseAPerRank_[dstRank]) || dstExpertIdx >= e_ * epWorldSize_) {
            break;
        }
        uint64_t dstRecvCntValue = preCommRecvGlobalBuffer_.GetValue(dstRank * e_ * epWorldSize_ + dstExpertIdx);
        int32_t remaining = static_cast<int32_t>(dstRecvCntValue) - dstOffset;
        int32_t toAdd = Mc2Min(remaining, static_cast<int32_t>(baseAPerRank_[dstRank]) - dstAccumulated);

        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank == 0U && sendLoopCount < 3) {
                AscendC::printf("[SendLoop] r%u dstRank=%u loop%d: expIdx=%u offset=%d recvCnt=%lu "
                                "remaining=%d toAdd=%d acc=%d baseA=%u\n",
                                rankId_, dstRank, sendLoopCount, dstExpertIdx, dstOffset, dstRecvCntValue, remaining,
                                toAdd, dstAccumulated, baseAPerRank_[dstRank]);
            }
        }

        uint32_t srcRankOfExpert = static_cast<uint32_t>(dstExpertIdx / e_);
        if (srcRankOfExpert == rankId_) {
            sendAmount += toAdd;
        }

        dstAccumulated += toAdd;
        dstOffset += toAdd;
        if (dstOffset >= static_cast<int32_t>(dstRecvCntValue)) {
            dstExpertIdx++;
            dstOffset = 0;
        }
    }

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0 && roundIdx < 1U && dstRank < 2U) {
            AscendC::printf(
                "[SendLoop] r%u dstRank=%u totalLoop=%d finalExpIdx=%u finalOffset=%d finalAcc=%d sendAmt=%d\n",
                rankId_, dstRank, sendLoopCount, dstExpertIdx, dstOffset, dstAccumulated, sendAmount);
        }
    }

    recvProgressExpertIdx_[dstRank] = dstExpertIdx;
    recvProgressExpertOffset_[dstRank] = dstOffset;
    alltoAllvSendCnt[dstRank] = static_cast<uint64_t>(sendAmount) * axis;
}

alltoAllvSendOffset[0] = 0UL;
alltoAllvRecvOffset[0] = 0UL;
for (uint32_t i = 1U; i < rankDim_; i++) {
    alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U] + alltoAllvSendCnt[i - 1U];
    alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1U] + alltoAllvRecvCnt[i - 1U];
}

if ASCEND_IS_AIV {
    if (GetBlockIdx() == 0 && roundIdx < 2U) {
        uint64_t totalSend = 0, totalRecv = 0;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            totalSend += alltoAllvSendCnt[i];
            totalRecv += alltoAllvRecvCnt[i];
        }
        AscendC::printf("[Comm] r%u round%u: sendCnt=[%lu %lu] recvCnt=[%lu %lu] total=%lu/%lu\n", rankId_, roundIdx,
                        alltoAllvSendCnt[0], alltoAllvSendCnt[1], alltoAllvRecvCnt[0], alltoAllvRecvCnt[1], totalSend,
                        totalRecv);
    }
}

// alltoAllvHandleQueue_[roundIdx] = hccl_.AlltoAllV<true>(
//     (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
//     (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;
#endif

static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
static constexpr uint64_t HANDLE_QUEUE_SIZE = 64U;
static constexpr uint64_t MAX_EXPERT_SIZE = 64U;
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
uint64_t ubSize_ = 0UL;
int32_t recvCounts_[MAX_EXPERT_SIZE] = {0};
int32_t sendCounts_[MAX_EXPERT_SIZE] = {0};

uint32_t progressExpertIdx_ = 0U;
int32_t progressExpertOffset_ = 0;

uint64_t recvSumPerRank_[MAX_EP_RANK_SIZE] = {0UL};
uint32_t baseAPerRank_[MAX_EP_RANK_SIZE] = {0U};
uint32_t recvProgressExpertIdx_[MAX_EP_RANK_SIZE] = {0U};
int32_t recvProgressExpertOffset_[MAX_EP_RANK_SIZE] = {0};

GlobalTensor<hcclDataType> sendGlobalBuffer_;
GlobalTensor<hcclDataType> recvGlobalBuffer_;

GlobalTensor<fp8_e8m0_t> sendScaleGlobalBuffer_;
GlobalTensor<fp8_e8m0_t> scaleCommOutBuffer_;
GlobalTensor<fp8_e8m0_t> scalePermuteOutBuffer_;

TBuf<QuePosition::VECIN> scaleTBuf_;
int32_t eventID_ = 0;

GlobalTensor<uint64_t> preCommSendGlobalBuffer_;
GlobalTensor<uint64_t> preCommRecvGlobalBuffer_;

HcclHandle alltoAllvHandleQueue_[HANDLE_QUEUE_SIZE] = {INVALID_HANDLE_ID};
HcclHandle alltoAllvScaleHandleQueue_[HANDLE_QUEUE_SIZE] = {INVALID_HANDLE_ID};
HcclHandle scaleBatchHandle_ = INVALID_HANDLE_ID;
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
}
;
} // namespace MC2KernelTemplate

#endif