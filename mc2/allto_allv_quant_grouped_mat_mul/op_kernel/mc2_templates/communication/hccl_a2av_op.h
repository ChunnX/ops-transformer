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

#ifndef TILINGKEY_TPL_CCU
#define TILINGKEY_TPL_CCU 0
#endif
#ifndef TILINGKEY_TPL_AICPU
#define TILINGKEY_TPL_AICPU 1
#endif

namespace MC2KernelTemplate {

template<int commMode = TILINGKEY_TPL_CCU>
struct HcclTypeSelector {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template<>
struct HcclTypeSelector<TILINGKEY_TPL_AICPU> {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode = TILINGKEY_TPL_CCU>
class HcclA2avOp {
public:
    __aicore__ inline HcclA2avOp() {}
    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset, const TaskTilingInfo *taskTilingInfo,
                                GM_ADDR sendBuffer, GM_ADDR recvBuffer, GM_ADDR permuteBuffer);
    __aicore__ inline void InitScaleBuffer(GM_ADDR sendBuffer, GM_ADDR commOutBuffer, GM_ADDR permuteOutBuffer);
    __aicore__ inline void LaunchCommScale(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void LaunchCommData(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void PermuteScale(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void PermuteData(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void WaitScale(uint32_t startExpertIdx);
    __aicore__ inline void Wait(uint32_t startExpertIdx);
    __aicore__ inline void WaitAll(uint32_t allNum);
    __aicore__ inline void End();

private:
    __aicore__ inline void LaunchCommScaleBeforeCompute(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void LaunchCommDataBeforeCompute(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void LaunchCommAfterCompute(uint32_t startExpertIdx, uint32_t expertNum);

private:
    typename HcclTypeSelector<commMode>::type hccl_;

    const TaskTilingInfo *taskTilingInfo_;

    uint32_t rankId_ = 0U;
    uint32_t rankDim_ = 0U;
    uint32_t e_ = 0U;
    uint64_t H1_ = 0UL;
    uint64_t N1_ = 0UL;
    uint64_t bufferLen_ = 0UL;

    GlobalTensor<hcclDataType> sendGlobalBuffer_;
    GlobalTensor<hcclDataType> recvGlobalBuffer_;
    GlobalTensor<hcclDataType> permuteOutBuffer_;

    GlobalTensor<fp8_e8m0_t> sendScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> recvScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> scalePermuteOutBuffer_;

    TPipe *pipe_ = nullptr;

    int32_t eventID_ = 0;

    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclHandle alltoAllvScaleHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;

    uint64_t alltoAllvRecvOffsetLastSum = 0UL;
    uint64_t alltoAllvScaleRecvOffsetLastSum = 0UL;

    uint64_t alltoAllvScaleSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

    uint64_t alltoAllvDataSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvDataSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvDataRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvDataRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

    uint64_t recvCounts_[MAX_EXPERT_SIZE] = {0UL};

    uint64_t permuteBaseOffset_ = 0UL;
    uint64_t scalePermuteBaseOffset_ = 0UL;
};

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::Init(const void *hcclInitTiling,
    uint64_t hcclCcTilingOffset, const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer, GM_ADDR recvBuffer, GM_ADDR permuteBuffer)
{
    taskTilingInfo_ = taskTilingInfo;
    GM_ADDR hcclContextGm = GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.InitV2(hcclContextGm, hcclInitTiling);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();
    e_ = taskTilingInfo_->e;
    H1_ = taskTilingInfo_->H1;
    N1_ = taskTilingInfo_->N1;
    bufferLen_ = MAX_BUFFER_SIZE;
    sendGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)sendBuffer);
    recvGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)recvBuffer);
    permuteOutBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)permuteBuffer);
    for (int i = 0; i < e_ * rankDim_; i++) {
        recvCounts_[i] = taskTilingInfo_->recvCnt[i];
    }
    if constexpr (AscendC::IsSameType<hcclDataType, bfloat16_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_BFP16;
    } else if constexpr (AscendC::IsSameType<hcclDataType, hifloat8_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_HIF8;
    } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e5m2_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e4m3fn_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_FP8E4M3;
    } else {
        hcclDataType_ = HCCL_DATA_TYPE_INT8;
    }
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::InitScaleBuffer(
    GM_ADDR sendBuffer, GM_ADDR commOutBuffer, GM_ADDR permuteOutBuffer)
{
    sendScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)sendBuffer);
    recvScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)commOutBuffer);
    scalePermuteOutBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)permuteOutBuffer);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::LaunchCommScale(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
    }
    LaunchCommScaleBeforeCompute(startExpertIdx, expertNum);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::LaunchCommData(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
    }
    if constexpr (commBeforeComputeFlag) {
        LaunchCommDataBeforeCompute(startExpertIdx, expertNum);
    } else {
        LaunchCommAfterCompute(startExpertIdx, expertNum);
    }
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::PermuteScale(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
    }
    pipe_ = GetTPipePtr();
    pipe_->Reset();
    TBuf<QuePosition::VECIN> scaleTBuf;
    pipe_->InitBuffer(scaleTBuf, bufferLen_);

    uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
    uint64_t baseOffset = scalePermuteBaseOffset_;
    uint64_t recvSumCntBefore[MAX_EXPERT_SIZE] = {0UL};
    uint64_t recvSumCntAfter[MAX_EXPERT_SIZE] = {0UL};
    LocalTensor<fp8_e8m0_t> ubTensor = scaleTBuf.Get<fp8_e8m0_t>();

    for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
        for (uint32_t localExpertIdx = 0U; localExpertIdx < expertNum; localExpertIdx++) {
            uint32_t localPosBefore = rankIndex * expertNum + localExpertIdx;
            uint32_t globalExpertIdx = startExpertIdx + localExpertIdx;
            recvSumCntBefore[localPosBefore] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + globalExpertIdx]);
            if (localPosBefore != 0U) {
                recvSumCntBefore[localPosBefore] += recvSumCntBefore[localPosBefore - 1U];
            }
        }
    }

    for (uint32_t localExpertIdx = 0U; localExpertIdx < expertNum; localExpertIdx++) {
        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            uint32_t localPosAfter = localExpertIdx * rankDim_ + rankIndex;
            uint32_t globalExpertIdx = startExpertIdx + localExpertIdx;
            recvSumCntAfter[localPosAfter] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + globalExpertIdx]);
            if (localPosAfter != 0U) {
                recvSumCntAfter[localPosAfter] += recvSumCntAfter[localPosAfter - 1U];
            }
        }
    }

    uint32_t endExpertIdx = startExpertIdx + expertNum;
    for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
            uint32_t localExpertIdx = expertIdx - startExpertIdx;
            uint32_t localPosBefore = rankIndex * expertNum + localExpertIdx;
            uint32_t localPosAfter = localExpertIdx * rankDim_ + rankIndex;

            uint64_t srcOffset = baseOffset;
            if (localPosBefore != 0U) {
                srcOffset += recvSumCntBefore[localPosBefore - 1U] * axis;
            }

            uint64_t dstOffset = baseOffset;
            if (localPosAfter != 0U) {
                dstOffset += recvSumCntAfter[localPosAfter - 1U] * axis;
            }

            uint64_t totalCnt = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]) * axis;
            uint64_t tileNum = CeilDiv(totalCnt, bufferLen_);
            for (uint64_t tile = 0UL; tile < tileNum; tile++) {
                uint64_t realLength = bufferLen_;
                if (tile == tileNum - 1UL) {
                    realLength = totalCnt - tile * bufferLen_;
                }
                uint64_t realLenBytes = realLength * sizeof(fp8_e8m0_t);
                DataCopyParams dataCopyParams{1U, static_cast<uint16_t>(realLenBytes), 0U, 0U};
                DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};
                DataCopyPad(ubTensor, recvScaleGlobalBuffer_[srcOffset + tile * bufferLen_], dataCopyParams,
                            dataCopyPadParams);
                eventID_ = static_cast<int32_t>(pipe_->FetchEventID<HardEvent::MTE2_MTE3>());
                SetFlag<HardEvent::MTE2_MTE3>(eventID_);
                WaitFlag<HardEvent::MTE2_MTE3>(eventID_);
                DataCopyPad(scalePermuteOutBuffer_[dstOffset + tile * bufferLen_], ubTensor, dataCopyParams);
                eventID_ = static_cast<int32_t>(pipe_->FetchEventID<HardEvent::MTE3_MTE2>());
                SetFlag<HardEvent::MTE3_MTE2>(eventID_);
                WaitFlag<HardEvent::MTE3_MTE2>(eventID_);
            }
        }
    }

    uint64_t currentDataCount = 0UL;
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            currentDataCount += recvCounts_[rankIndex * e_ + expertIdx];
        }
    }
    scalePermuteBaseOffset_ += currentDataCount * axis;
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::PermuteData(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
    }
    pipe_ = GetTPipePtr();
    pipe_->Reset();
    TBuf<QuePosition::VECIN> dataTBuff;
    pipe_->InitBuffer(dataTBuff, bufferLen_);

    uint64_t axis = CeilDiv(H1_, PACK_FACTOR);
    uint64_t baseOffset = permuteBaseOffset_;
    uint64_t recvSumCntBefore[MAX_EXPERT_SIZE] = {0UL};
    uint64_t recvSumCntAfter[MAX_EXPERT_SIZE] = {0UL};
    LocalTensor<hcclDataType> ubTensor = dataTBuff.Get<hcclDataType>();

    for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
        for (uint32_t localExpertIdx = 0U; localExpertIdx < expertNum; localExpertIdx++) {
            uint32_t localPosBefore = rankIndex * expertNum + localExpertIdx;
            uint32_t globalExpertIdx = startExpertIdx + localExpertIdx;
            recvSumCntBefore[localPosBefore] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + globalExpertIdx]);
            if (localPosBefore != 0U) {
                recvSumCntBefore[localPosBefore] += recvSumCntBefore[localPosBefore - 1U];
            }
        }
    }

    for (uint32_t localExpertIdx = 0U; localExpertIdx < expertNum; localExpertIdx++) {
        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            uint32_t localPosAfter = localExpertIdx * rankDim_ + rankIndex;
            uint32_t globalExpertIdx = startExpertIdx + localExpertIdx;
            recvSumCntAfter[localPosAfter] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + globalExpertIdx]);
            if (localPosAfter != 0U) {
                recvSumCntAfter[localPosAfter] += recvSumCntAfter[localPosAfter - 1U];
            }
        }
    }

    uint32_t endExpertIdx = startExpertIdx + expertNum;
    for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
            uint32_t localExpertIdx = expertIdx - startExpertIdx;
            uint32_t localPosBefore = rankIndex * expertNum + localExpertIdx;
            uint32_t localPosAfter = localExpertIdx * rankDim_ + rankIndex;

            uint64_t srcOffset = baseOffset;
            if (localPosBefore != 0U) {
                srcOffset += recvSumCntBefore[localPosBefore - 1U] * axis;
            }

            uint64_t dstOffset = baseOffset;
            if (localPosAfter != 0U) {
                dstOffset += recvSumCntAfter[localPosAfter - 1U] * axis;
            }

            uint64_t totalCnt = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]) * axis;

            uint64_t tileNum = CeilDiv(totalCnt, bufferLen_);
            for (uint64_t tile = 0UL; tile < tileNum; tile++) {
                uint64_t realLength = bufferLen_;
                if (tile == tileNum - 1UL) {
                    realLength = totalCnt - tile * bufferLen_;
                }
                uint64_t realLenBytes = realLength * sizeof(hcclDataType) * PACK_FACTOR;
                DataCopyParams dataCopyParams{1U, static_cast<uint16_t>(realLenBytes), 0U, 0U};
                DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};
                DataCopyPad(ubTensor, recvGlobalBuffer_[srcOffset + tile * bufferLen_], dataCopyParams,
                            dataCopyPadParams);
                eventID_ = static_cast<int32_t>(pipe_->FetchEventID<HardEvent::MTE2_MTE3>());
                SetFlag<HardEvent::MTE2_MTE3>(eventID_);
                WaitFlag<HardEvent::MTE2_MTE3>(eventID_);
                DataCopyPad(permuteOutBuffer_[dstOffset + tile * bufferLen_], ubTensor, dataCopyParams);
                eventID_ = static_cast<int32_t>(pipe_->FetchEventID<HardEvent::MTE3_MTE2>());
                SetFlag<HardEvent::MTE3_MTE2>(eventID_);
                WaitFlag<HardEvent::MTE3_MTE2>(eventID_);
            }
        }
    }

    uint64_t currentDataCount = 0UL;
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            currentDataCount += recvCounts_[rankIndex * e_ + expertIdx];
        }
    }
    permuteBaseOffset_ += currentDataCount * axis;
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::LaunchCommScaleBeforeCompute(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
    uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
    for (uint64_t i = 0UL; i < rankDim_; i++) {
        alltoAllvScaleSendCnt[i] = 0UL;
        alltoAllvScaleRecvCnt[i] = 0UL;
        for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
            alltoAllvScaleSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
            alltoAllvScaleRecvCnt[i] += static_cast<uint64_t>(recvCounts_[expertIdx + i * e_]) * axis;
        }
    }

    alltoAllvScaleSendOffset[0] = 0UL;
    for (uint32_t j = 0U; j < startExpertIdx; j++) {
        alltoAllvScaleSendOffset[0] += static_cast<uint64_t>(sendCnt[j]) * axis;
    }
    for (uint32_t i = 1U; i < rankDim_; i++) {
        alltoAllvScaleSendOffset[i] = alltoAllvScaleSendOffset[i - 1U];
        for (uint32_t j = 0U; j < e_; j++) {
            alltoAllvScaleSendOffset[i] +=
                static_cast<uint64_t>(sendCnt[startExpertIdx + (i - 1U) * e_ + j]) * axis;
        }
    }

    for (uint32_t i = 0U; i < rankDim_; i++) {
        if ((startExpertIdx == 0U) && (i == 0U)) {
            alltoAllvScaleRecvOffset[i] = 0UL;
            alltoAllvScaleRecvOffsetLastSum += alltoAllvScaleRecvCnt[0];
        } else {
            alltoAllvScaleRecvOffset[i] = alltoAllvScaleRecvOffsetLastSum;
            alltoAllvScaleRecvOffsetLastSum += alltoAllvScaleRecvCnt[i];
        }
    }

    alltoAllvScaleHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
        (__gm__ uint8_t *)sendScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleSendCnt, alltoAllvScaleSendOffset, HCCL_DATA_TYPE_FP8E8M0,
        (__gm__ uint8_t *)recvScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleRecvCnt, alltoAllvScaleRecvOffset, HCCL_DATA_TYPE_FP8E8M0);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::LaunchCommDataBeforeCompute(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
    uint64_t axis = CeilDiv(H1_, PACK_FACTOR);
    for (uint64_t i = 0UL; i < rankDim_; i++) {
        alltoAllvDataSendCnt[i] = 0UL;
        alltoAllvDataRecvCnt[i] = 0UL;
        for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
            alltoAllvDataSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
            alltoAllvDataRecvCnt[i] += static_cast<uint64_t>(recvCounts_[expertIdx + i * e_]) * axis;
        }
    }

    alltoAllvDataSendOffset[0] = 0UL;
    for (uint32_t j = 0U; j < startExpertIdx; j++) {
        alltoAllvDataSendOffset[0] += static_cast<uint64_t>(sendCnt[j]) * axis;
    }
    for (uint32_t i = 1U; i < rankDim_; i++) {
        alltoAllvDataSendOffset[i] = alltoAllvDataSendOffset[i - 1U];
        for (uint32_t j = 0U; j < e_; j++) {
            alltoAllvDataSendOffset[i] +=
                static_cast<uint64_t>(sendCnt[startExpertIdx + (i - 1U) * e_ + j]) * axis;
        }
    }

    for (uint32_t i = 0U; i < rankDim_; i++) {
        if ((startExpertIdx == 0U) && (i == 0U)) {
            alltoAllvDataRecvOffset[i] = 0UL;
            alltoAllvRecvOffsetLastSum += alltoAllvDataRecvCnt[0];
        } else {
            alltoAllvDataRecvOffset[i] = alltoAllvRecvOffsetLastSum;
            alltoAllvRecvOffsetLastSum += alltoAllvDataRecvCnt[i];
        }
    }

    alltoAllvHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
        (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvDataSendCnt, alltoAllvDataSendOffset, hcclDataType_,
        (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvDataRecvCnt, alltoAllvDataRecvOffset, hcclDataType_);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::WaitScale(uint32_t startExpertIdx)
{
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
    }
    hccl_.Wait(alltoAllvScaleHandleId_[startExpertIdx]);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::Wait(uint32_t startExpertIdx)
{
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
    }
    hccl_.Wait(alltoAllvHandleId_[startExpertIdx]);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::WaitAll(uint32_t allNum)
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
        hccl_.Wait(alltoAllvHandleId_[i]);
    }
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::End()
{
    if ASCEND_IS_AIC {
        return;
    }
    hccl_.Finalize();
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode>::LaunchCommAfterCompute(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
    uint64_t axis = N1_;
    for (uint64_t i = 0UL; i < rankDim_; i++) {
        alltoAllvDataSendCnt[i] = 0UL;
        alltoAllvDataRecvCnt[i] = 0UL;
        for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
            alltoAllvDataSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
            alltoAllvDataRecvCnt[i] += static_cast<uint64_t>(recvCounts_[expertIdx + i * e_]) * axis;
        }
    }
    uint64_t expertOffset = 0UL;
    for (uint64_t i = 0UL; i < startExpertIdx; i++) {
        for (uint64_t j = 0UL; j < rankDim_; j++) {
            expertOffset += static_cast<uint64_t>(sendCnt[i + j * e_]);
        }
    }
    alltoAllvDataSendOffset[0] = expertOffset * N1_;
    for (uint64_t i = 1UL; i < rankDim_; i++) {
        alltoAllvDataSendOffset[i] = alltoAllvDataSendOffset[i - 1];
        for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
            alltoAllvDataSendOffset[i] += static_cast<uint64_t>(sendCnt[expertIdx + (i - 1) * e_]) * N1_;
        }
    }
    alltoAllvDataRecvOffset[0] = 0UL;
    for (uint32_t i = 0U; i < startExpertIdx; i++) {
        alltoAllvDataRecvOffset[0] += static_cast<uint64_t>(recvCounts_[i]) * N1_;
    }
    for (uint32_t i = 1U; i < rankDim_; i++) {
        alltoAllvDataRecvOffset[i] = alltoAllvDataRecvOffset[i - 1];
        for (uint32_t j = 0U; j < e_; j++) {
            alltoAllvDataRecvOffset[i] += static_cast<uint64_t>(recvCounts_[startExpertIdx + (i - 1) * e_ + j]) * N1_;
        }
    }
    alltoAllvHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
        (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvDataSendCnt, alltoAllvDataSendOffset, hcclDataType_,
        (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvDataRecvCnt, alltoAllvDataRecvOffset, hcclDataType_);
}

}; // namespace MC2KernelTemplate

#endif