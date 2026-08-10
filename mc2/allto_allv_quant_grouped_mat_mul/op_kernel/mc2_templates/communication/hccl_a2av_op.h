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
#include "a2av_comm_guard.h"
#include "a2av_comm_params.h"

using namespace AscendC;

#ifndef TILINGKEY_TPL_CCU
#define TILINGKEY_TPL_CCU 0
#endif
#ifndef TILINGKEY_TPL_AICPU
#define TILINGKEY_TPL_AICPU 1
#endif

namespace MC2KernelTemplate {

template <typename T>
struct CommBufferType {
    using type = T;
};

template <>
struct CommBufferType<fp4x2_e2m1_t> {
    using type = int8_t;
};

template <int commMode = TILINGKEY_TPL_CCU>
struct HcclTypeSelector {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template <>
struct HcclTypeSelector<TILINGKEY_TPL_AICPU> {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode = TILINGKEY_TPL_CCU,
          bool CountsFromTensor = false>
class HcclA2avOp {
public:
    using CommBufType = typename CommBufferType<hcclDataType>::type;

    __aicore__ inline HcclA2avOp() {}
    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
                                const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                                GM_ADDR recvBuffer, uint32_t aivCoreNum = 1U, TPipe *tPipe = nullptr,
                                GM_ADDR sendCountsTensorGM = nullptr, GM_ADDR recvCountsTensorGM = nullptr);
    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
                                const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                                GM_ADDR recvBuffer, GM_ADDR permuteBuffer,
                                uint32_t aivCoreNum);
    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
                                const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                                GM_ADDR recvBuffer, GM_ADDR permuteBuffer,
                                uint32_t aivCoreNum, TPipe *tPipe,
                                GM_ADDR sendCountsTensorGM = nullptr, GM_ADDR recvCountsTensorGM = nullptr);
    __aicore__ inline void InitScaleBuffer(GM_ADDR sendBuffer, GM_ADDR commOutBuffer, GM_ADDR permuteOutBuffer);
    __aicore__ inline void LaunchCommScale(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void LaunchCommData(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void Launch(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void LaunchScaleBeforeCompute(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void WaitScale(uint32_t startExpertIdx);
    __aicore__ inline void Wait(uint32_t startExpertIdx);
    __aicore__ inline void WaitAll(uint32_t allNum, uint32_t step = 1U);
    __aicore__ inline void PermuteData(uint32_t startExpertIdx, uint32_t expertNum, GM_ADDR dstBuffer);
    __aicore__ inline void End();

private:
    typename HcclTypeSelector<commMode>::type hccl_;
    const TaskTilingInfo *taskTilingInfo_;

    uint32_t rankDim_ = 0U;
    uint32_t e_ = 0U;
    uint64_t H1_ = 0UL;
    uint64_t N1_ = 0UL;
    uint64_t bufferLen_ = 0UL;

    GlobalTensor<CommBufType> sendGlobalBuffer_;
    GlobalTensor<CommBufType> recvGlobalBuffer_;
    GlobalTensor<CommBufType> permuteOutBuffer_;
    GlobalTensor<fp8_e8m0_t> sendScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> recvScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> scalePermuteOutBuffer_;
    GlobalTensor<int64_t> sendCountsGlobal_;
    GlobalTensor<int64_t> recvCountsGlobal_;

    A2avCommParams dataCommParams_;
    A2avCommParams scaleCommParams_;

    uint64_t dataSendOffsetLastSum_ = 0UL;
    uint64_t dataRecvOffsetLastSum_ = 0UL;
    uint64_t scaleSendOffsetLastSum_ = 0UL;
    uint64_t scaleRecvOffsetLastSum_ = 0UL;

    uint64_t recvCounts_[MAX_EXPERT_SIZE] = {0UL};
    uint64_t sendCounts_[MAX_EXPERT_SIZE] = {0UL};
    TBuf<QuePosition::VECCALC> sendCountsUB_;
    TBuf<QuePosition::VECCALC> recvCountsUB_;
    LocalTensor<int64_t> sendCountsLt_;
    LocalTensor<int64_t> recvCountsLt_;
    uint32_t countsTensorLen_ = 0U;

    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclHandle alltoAllvScaleHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;

    uint32_t aivCoreNum_ = 40U;
    TBuf<QuePosition::VECIN> permuteTBuf_;
    TBuf<QuePosition::VECIN> permuteTBuf2_;
    bool permuteBufInitialized_ = false;
    uint64_t permuteBaseOffset_ = 0UL;
    int32_t eventID_ = 0;
};

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::Init(
    const void *hcclInitTiling, uint64_t hcclCcTilingOffset, const TaskTilingInfo *taskTilingInfo,
    GM_ADDR sendBuffer, GM_ADDR recvBuffer, uint32_t aivCoreNum, TPipe *tPipe,
    GM_ADDR sendCountsTensorGM, GM_ADDR recvCountsTensorGM)
{
    Init(hcclInitTiling, hcclCcTilingOffset, taskTilingInfo, sendBuffer, recvBuffer, nullptr, aivCoreNum, tPipe,
         sendCountsTensorGM, recvCountsTensorGM);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::Init(
    const void *hcclInitTiling, uint64_t hcclCcTilingOffset, const TaskTilingInfo *taskTilingInfo,
    GM_ADDR sendBuffer, GM_ADDR recvBuffer, GM_ADDR permuteBuffer,
    uint32_t aivCoreNum)
{
    Init(hcclInitTiling, hcclCcTilingOffset, taskTilingInfo, sendBuffer, recvBuffer, permuteBuffer, aivCoreNum,
         nullptr, nullptr, nullptr);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::Init(
    const void *hcclInitTiling, uint64_t hcclCcTilingOffset, const TaskTilingInfo *taskTilingInfo,
    GM_ADDR sendBuffer, GM_ADDR recvBuffer, GM_ADDR permuteBuffer,
    uint32_t aivCoreNum, TPipe *tPipe, GM_ADDR sendCountsTensorGM, GM_ADDR recvCountsTensorGM)
{
    taskTilingInfo_ = taskTilingInfo;
    aivCoreNum_ = aivCoreNum;
    GM_ADDR hcclContextGm = GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.InitV2(hcclContextGm, hcclInitTiling);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);
    rankDim_ = hccl_.GetRankDim();
    e_ = taskTilingInfo_->e;
    H1_ = taskTilingInfo_->H1;
    N1_ = taskTilingInfo_->N1;
    bufferLen_ = PERMUTE_BUF_SIZE;
    sendGlobalBuffer_.SetGlobalBuffer((__gm__ CommBufType *)sendBuffer);
    recvGlobalBuffer_.SetGlobalBuffer((__gm__ CommBufType *)recvBuffer);
    permuteOutBuffer_.SetGlobalBuffer((__gm__ CommBufType *)permuteBuffer);
    countsTensorLen_ = static_cast<uint32_t>(e_ * rankDim_);
    if constexpr (CountsFromTensor) {
        ASCENDC_ASSERT(tPipe != nullptr, { KERNEL_LOG(KERNEL_ERROR, "tPipe must not be null for tensor counts path"); });
        ASCENDC_ASSERT(sendCountsTensorGM != nullptr && recvCountsTensorGM != nullptr,
                       { KERNEL_LOG(KERNEL_ERROR, "counts tensor GM addr must not be null for tensor counts path"); });
        // UB / DataCopyPad / MTE2_V 仅在 AIV 核上可用，AIC 核无 UB 和 V 流水线
        if ASCEND_IS_AIV {
            constexpr uint32_t COUNTS_UB_ALIGN = 32U;
            uint32_t countsBytes = countsTensorLen_ * sizeof(int64_t);
            uint32_t countsUbSize = (countsBytes + COUNTS_UB_ALIGN - 1U) & ~(COUNTS_UB_ALIGN - 1U);
            tPipe->InitBuffer(sendCountsUB_, countsUbSize);
            tPipe->InitBuffer(recvCountsUB_, countsUbSize);
            sendCountsGlobal_.SetGlobalBuffer((__gm__ int64_t *)sendCountsTensorGM);
            recvCountsGlobal_.SetGlobalBuffer((__gm__ int64_t *)recvCountsTensorGM);
            DataCopyParams countsCopyParams{1U, static_cast<uint16_t>(countsTensorLen_ * sizeof(int64_t)), 0U, 0U};
            DataCopyPadParams countsPadParams{false, 0U, 0U, 0U};
            sendCountsLt_ = sendCountsUB_.Get<int64_t>();
            recvCountsLt_ = recvCountsUB_.Get<int64_t>();
            DataCopyPad(sendCountsLt_, sendCountsGlobal_, countsCopyParams, countsPadParams);
            DataCopyPad(recvCountsLt_, recvCountsGlobal_, countsCopyParams, countsPadParams);
            int32_t mte2vEventID = static_cast<int32_t>(tPipe->FetchEventID<HardEvent::MTE2_V>());
            SetFlag<HardEvent::MTE2_V>(mte2vEventID);
            WaitFlag<HardEvent::MTE2_V>(mte2vEventID);
            tPipe->ReleaseEventID<HardEvent::MTE2_V>(mte2vEventID);
        }
    } else {
        for (int i = 0; i < e_ * rankDim_; i++) {
            recvCounts_[i] = taskTilingInfo_->recvCnt[i];
            sendCounts_[i] = taskTilingInfo_->sendCnt[i];
        }
    }
    if constexpr (AscendC::IsSameType<hcclDataType, bfloat16_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_BFP16;
    } else if constexpr (AscendC::IsSameType<hcclDataType, half>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_FP16;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    } else if constexpr (AscendC::IsSameType<hcclDataType, hifloat8_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_HIF8;
    } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e5m2_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e4m3fn_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_FP8E4M3;
#endif
    } else {
        hcclDataType_ = HCCL_DATA_TYPE_INT8;
    }
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::InitScaleBuffer(
    GM_ADDR sendBuffer, GM_ADDR commOutBuffer, GM_ADDR permuteOutBuffer)
{
    sendScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)sendBuffer);
    recvScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)commOutBuffer);
    scalePermuteOutBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)permuteOutBuffer);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::LaunchCommScale(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    A2AV_AIV_ONLY();
    uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
    struct __aicore__ GetCountFn {
        HcclA2avOp *self;
        __aicore__ inline uint64_t operator()(bool isSend, uint64_t idx) const
        {
            if constexpr (CountsFromTensor) {
                return static_cast<uint64_t>(isSend ? self->sendCountsLt_.GetValue(idx) : self->recvCountsLt_.GetValue(idx));
            }
            return isSend ? self->sendCounts_[idx] : self->recvCounts_[idx];
        }
    } getCount{this};
    CalcA2avCommBeforeParams(scaleCommParams_, getCount, rankDim_, e_, startExpertIdx, expertNum,
                             axis, scaleSendOffsetLastSum_, scaleRecvOffsetLastSum_);
    alltoAllvScaleHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
        (__gm__ uint8_t *)sendScaleGlobalBuffer_.GetPhyAddr(), scaleCommParams_.sendCnt,
        scaleCommParams_.sendOffset, HCCL_DATA_TYPE_FP8E8M0,
        (__gm__ uint8_t *)recvScaleGlobalBuffer_.GetPhyAddr(), scaleCommParams_.recvCnt,
        scaleCommParams_.recvOffset, HCCL_DATA_TYPE_FP8E8M0);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::LaunchCommData(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    A2AV_AIV_ONLY();
    struct __aicore__ GetCountFn {
        HcclA2avOp *self;
        __aicore__ inline uint64_t operator()(bool isSend, uint64_t idx) const
        {
            if constexpr (CountsFromTensor) {
                return static_cast<uint64_t>(isSend ? self->sendCountsLt_.GetValue(idx) : self->recvCountsLt_.GetValue(idx));
            }
            return isSend ? self->sendCounts_[idx] : self->recvCounts_[idx];
        }
    } getCount{this};
    constexpr uint64_t MAX_SINGLE_COMM_BYTES = 256UL * 1024UL * 1024UL;
    uint64_t singleCommSendBytes = 0UL;
    uint64_t singleCommRecvBytes = 0UL;
    uint64_t axisBytes = CeilDiv(H1_, PACK_FACTOR) * sizeof(CommBufType);
    for (uint32_t r = 0; r < rankDim_; r++) {
        for (uint32_t e = startExpertIdx; e < startExpertIdx + expertNum; e++) {
            singleCommSendBytes += getCount(true, e + r * e_) * axisBytes;
            singleCommRecvBytes += getCount(false, e + r * e_) * axisBytes;
        }
    }
    uint64_t maxSingleCommBytes = (singleCommSendBytes > singleCommRecvBytes) ? singleCommSendBytes : singleCommRecvBytes;
    ASCENDC_ASSERT(maxSingleCommBytes <= MAX_SINGLE_COMM_BYTES, {
        KERNEL_LOG(KERNEL_ERROR,
                   "single AlltoAllV launch comm size %lu exceeds 256MB, batch=%u, expertNum=%u, H1=%lu",
                   maxSingleCommBytes, startExpertIdx, expertNum, H1_);
    });
    if constexpr (commBeforeComputeFlag) {
        uint64_t axis = CeilDiv(H1_, PACK_FACTOR);
        CalcA2avCommBeforeParams(dataCommParams_, getCount, rankDim_, e_, startExpertIdx, expertNum,
                                 axis, dataSendOffsetLastSum_, dataRecvOffsetLastSum_);
    } else {
        CalcA2avCommAfterParams(dataCommParams_, getCount, rankDim_, e_, startExpertIdx, expertNum,
                                N1_, dataSendOffsetLastSum_, dataRecvOffsetLastSum_);
    }
    alltoAllvHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
        (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), dataCommParams_.sendCnt,
        dataCommParams_.sendOffset, hcclDataType_,
        (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), dataCommParams_.recvCnt,
        dataCommParams_.recvOffset, hcclDataType_);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::Launch(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    LaunchCommData(startExpertIdx, expertNum);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::LaunchScaleBeforeCompute(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    LaunchCommScale(startExpertIdx, expertNum);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::WaitScale(uint32_t startExpertIdx)
{
    A2AV_AIV_ONLY();
    hccl_.Wait(alltoAllvScaleHandleId_[startExpertIdx]);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::Wait(uint32_t startExpertIdx)
{
    A2AV_AIV_ONLY();
    hccl_.Wait(alltoAllvHandleId_[startExpertIdx]);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::WaitAll(
    uint32_t allNum, uint32_t step)
{
    A2AV_AIV_ONLY();
    for (uint32_t i = 0U; i < allNum; i += step) {
        hccl_.Wait(alltoAllvHandleId_[i]);
    }
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::PermuteData(
    uint32_t startExpertIdx, uint32_t expertNum, GM_ADDR dstBuffer)
{
    A2AV_AIV_ALL();
    if (!permuteBufInitialized_) {
        TPipe *pipe = GetTPipePtr();
        pipe->InitBuffer(permuteTBuf_, PERMUTE_BUF_SIZE);
        pipe->InitBuffer(permuteTBuf2_, PERMUTE_BUF_SIZE);
        permuteBufInitialized_ = true;
    }
    GlobalTensor<CommBufType> dstGlobalBuffer;
    dstGlobalBuffer.SetGlobalBuffer((__gm__ CommBufType *)dstBuffer);
    uint64_t axis = CeilDiv(H1_, PACK_FACTOR);
    struct __aicore__ GetRecvCountFn {
        HcclA2avOp *self;
        __aicore__ inline uint64_t operator()(uint64_t idx) const
        {
            if constexpr (CountsFromTensor) {
                return static_cast<uint64_t>(self->recvCountsLt_.GetValue(idx));
            }
            return self->recvCounts_[idx];
        }
    } getCount{this};
    PermuteImplParallel<CommBufType, false, true>(
        recvGlobalBuffer_, dstGlobalBuffer,
        getCount, e_, rankDim_, startExpertIdx, expertNum,
        axis, permuteBaseOffset_, permuteTBuf_, permuteTBuf2_, bufferLen_, eventID_,
        aivCoreNum_);
}

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode, bool CountsFromTensor>
__aicore__ inline void HcclA2avOp<hcclDataType, commBeforeComputeFlag, commMode, CountsFromTensor>::End()
{
    SyncAll<false>();
    if ASCEND_IS_AIC {
        return;
    }
    hccl_.Finalize();
}

}; // namespace MC2KernelTemplate

#endif
