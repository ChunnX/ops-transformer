/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_fused_topk.h
 * \brief
 */

#ifndef ASCENDC_MOE_FUSED_TOPK_H_
#define ASCENDC_MOE_FUSED_TOPK_H_

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

constexpr static uint32_t BASE_COUNT = 256;
constexpr static uint32_t REPEAT_BYTES = 256;
constexpr static uint32_t BLOCK_BYTES = 32;
constexpr static uint32_t SORT_UNIT = 32;
constexpr static uint32_t ADD_COUNT_THIRTY_TWO = 32;
constexpr static uint32_t ADD_COUNT_SIXTY_FOUR = 64;
constexpr static uint32_t ADD_COUNT_ONE_TWENTY_EIGHT = 128;
constexpr static uint32_t BUFFER_NUM = 1;
constexpr static uint32_t BUFFER_NUM_ONE = 1;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t BROADCAST_DIM = 2;
constexpr static uint32_t BROADCAST_AXIS = 1;
constexpr static uint32_t SORTED_COEF = 2;
constexpr static int32_t FLOAT_BYTES = 4;
constexpr static uint8_t REPEAT_STRIDE_EIGHT = 8;

template <typename inputT, typename calT, uint32_t enableExpertMapping>
class MoeFusedTopk {
public:
    __aicore__ inline MoeFusedTopk(){};
    template <typename tilingT>
    __aicore__ inline void InitTilingData(tilingT *tilingData, GM_ADDR x, GM_ADDR addNum, GM_ADDR mappingNum,
                                          GM_ADDR mappingTable, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInAddNum();
    __aicore__ inline void CopyInX(const int32_t loop);
    __aicore__ inline void ActivateAndAdd();
    __aicore__ inline void GroupTopkImpl();
    __aicore__ inline void GroupReduceSumInternelImpl();
    __aicore__ inline void GatherSigmoidImpl();
    __aicore__ inline void NormImpl();
    __aicore__ inline void CopyFromWorkspace();
    __aicore__ inline void CopyToWorkspace();
    __aicore__ inline void CopyOut(const int32_t loop);
    __aicore__ inline void CopyInMappingNum();

    __aicore__ inline void ProcessSortAlign();

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    };

private:
    TPipe *pipe_;
    // create queues for input, in this case depth is equal to buffer num
    TBuf<TPosition::VECIN> xInBuf_;
    TBuf<TPosition::VECCALC> addNumInBuf_;
    TBuf<TPosition::VECOUT> yOutBuf_;
    TBuf<TPosition::VECOUT> indicesOutBuf_;
    TBuf<TPosition::VECIN> assistBuf_;
    TBuf<TPosition::VECCALC> sigmoidBuf_;
    TBuf<TPosition::VECIN> sigmoidAddBuf_;
    TBuf<TPosition::VECIN> sortedBuf_;
    TBuf<TPosition::VECIN> topkValueBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;

    TBuf<TPosition::VECIN> mappingNumBuf_;

    uint32_t secondDimSize_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupTopk_ = 1;
    uint32_t n_ = 1;
    uint32_t k_ = 0;
    uint32_t activateType_ = 0;
    uint32_t isNorm_ = 1;
    float scale_ = 1.0;
    uint32_t groupEles_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t tableDim_ = 0;
    int64_t outBatchStride_ = 0;
    int64_t batchOffset_ = 0;
    uint32_t loopBatch_ = 0;

    uint32_t groupElesAlignBlockCountFp32_ = 0;
    uint32_t groupElesAlignSortCount_ = 0;
    uint32_t secondAlignBlockCountFp32_ = 0;
    uint32_t secondAlignBlockCountFp16_ = 0;
    int64_t wsOffset_ = 0;
    uint32_t sortRepeatTimes_ = 1;
    uint32_t wholeSortNum_ = 1;
    uint32_t topKSortRepeatTimes_ = 1;

    uint32_t topnPad_ = 0;
    uint32_t topkPad_ = 0;
    uint32_t topkMaxValue_ = 0;
    uint32_t topkMinValue_ = 0;
    float floatNegativeInf_ = -3.4e38;

    GlobalTensor<inputT> mGmX_;
    GlobalTensor<inputT> mGmAddNum_;
    GlobalTensor<float> mGmY_;
    GlobalTensor<int32_t> mGmIndices_;
    GlobalTensor<float> mGmWorkspace_;

    GlobalTensor<int32_t> mappingNumGm_;
    GlobalTensor<int32_t> mappingTableGm_;

    TopkTiling topkTilingData_;
};

template <typename inputT, typename calT, uint32_t enableExpertMapping>
template <typename tilingT>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::InitTilingData(
    tilingT *tilingData, GM_ADDR x, GM_ADDR addNum, GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y,
    GM_ADDR indices, GM_ADDR workspace)
{
    secondDimSize_ = tilingData->secondDimSize;
    groupNum_ = tilingData->groupNum;
    groupTopk_ = tilingData->groupTopk;
    n_ = tilingData->topN;
    k_ = tilingData->topK;
    activateType_ = tilingData->activateType;
    isNorm_ = tilingData->isNorm;
    scale_ = tilingData->scale;
    groupEles_ = tilingData->groupEles;
    expertNum_ = tilingData->expertNum;
    tableDim_ = tilingData->tableDim;
    topkMaxValue_ = tilingData->topkMaxValue;
    topkMinValue_ = tilingData->topkMinValue;
    topkTilingData_ = tilingData->topkTilingData;

    uint32_t batchPerCore = tilingData->batchPerCore;
    uint32_t tailBatch = tilingData->tailBatch;
    uint32_t blockIdx = GetBlockIdx();
    uint64_t workspacePerCore = tilingData->workspacePerCore / sizeof(float);
    uint32_t perBlockCountFp32 = BLOCK_BYTES / sizeof(float);
    uint32_t perBlockCountFp16 = BLOCK_BYTES / sizeof(half);
    if (blockIdx < tailBatch) {
        loopBatch_ = batchPerCore + 1;
        batchOffset_ = blockIdx * loopBatch_;
    } else {
        loopBatch_ = batchPerCore;
        batchOffset_ = blockIdx * batchPerCore + tailBatch;
    }
    topkPad_ = CeilAlign(k_, BLOCK_BYTES / sizeof(calT));
    topnPad_ = CeilAlign(n_, BLOCK_BYTES / sizeof(calT));

    uint32_t tmpNegativeInf = NEGATIVE_MIN_VAULE_FP32;
    floatNegativeInf_ = *((float *)&tmpNegativeInf);

    outBatchStride_ = k_ * batchOffset_;
    mGmX_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(x));
    mGmAddNum_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(addNum));
    mGmY_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + outBatchStride_);
    mGmIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices) + outBatchStride_);
    mGmWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    mappingNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mappingNum));
    mappingTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mappingTable));

    groupElesAlignBlockCountFp32_ = CeilAlign(groupEles_, perBlockCountFp32);
    groupElesAlignSortCount_ = CeilAlign(groupEles_, SORT_UNIT);
    secondAlignBlockCountFp32_ = CeilAlign(secondDimSize_, perBlockCountFp32);
    secondAlignBlockCountFp16_ = CeilAlign(secondDimSize_, perBlockCountFp16);
    wsOffset_ = blockIdx * workspacePerCore;
    sortRepeatTimes_ = CeilDiv(secondDimSize_, SORT_UNIT);
    topKSortRepeatTimes_ = CeilDiv(k_, SORT_UNIT);
    wholeSortNum_ = sortRepeatTimes_ * SORT_UNIT;
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::CopyInAddNum()
{
    LocalTensor<float> addNumLocal = addNumInBuf_.Get<float>();
    LocalTensor<int32_t> assistLocal = assistBuf_.Get<int32_t>();
    uint32_t secondDimSizeInputBytes = secondDimSize_ * sizeof(inputT);

    // 初始化assistLocal
    ArithProgression(assistLocal, 0, 1, wholeSortNum_);

    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyPad(addNumLocal, mGmAddNum_, {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        LocalTensor<inputT> addNumLocalInputT = addNumLocal.template ReinterpretCast<inputT>();
        DataCopyPad(addNumLocalInputT[secondAlignBlockCountFp16_], mGmAddNum_, {1, secondDimSizeInputBytes, 0, 0, 0},
                    {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(addNumLocal, addNumLocalInputT[secondAlignBlockCountFp16_], RoundMode::CAST_NONE, secondDimSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::CopyInX(const int32_t loop)
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    int64_t xOffset = loop * secondDimSize_ + batchOffset_ * static_cast<int64_t>(secondDimSize_);
    uint32_t secondDimSizeInputBytes = secondDimSize_ * sizeof(inputT);
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyPad(xLocal, mGmX_[xOffset], {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        LocalTensor<inputT> xLocalInputT = xLocal.template ReinterpretCast<inputT>();
        DataCopyPad(xLocalInputT[secondAlignBlockCountFp16_], mGmX_[xOffset], {1, secondDimSizeInputBytes, 0, 0, 0},
                    {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(xLocal, xLocalInputT[secondAlignBlockCountFp16_], RoundMode::CAST_NONE, secondDimSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::ActivateAndAdd()
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    LocalTensor<float> addNumLocal = addNumInBuf_.Get<float>();
    LocalTensor<uint8_t> sharedTmpBuffer = tempBuf_.Get<uint8_t>();
    LocalTensor<float> sigmoidTensor = sigmoidBuf_.Get<float>();
    LocalTensor<float> sigmoidAddTensor = sigmoidAddBuf_.Get<float>();

    Sigmoid(sigmoidTensor, xLocal, sharedTmpBuffer, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    Add(sigmoidAddTensor, sigmoidTensor, addNumLocal, secondDimSize_);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::CopyToWorkspace()
{
    LocalTensor<float> sigmoidAddTensor = sigmoidAddBuf_.Get<float>();
    DataCopyPad(mGmWorkspace_[wsOffset_], sigmoidAddTensor, {1, (uint32_t)(secondDimSize_ * sizeof(float)), 0, 0, 0});
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::CopyFromWorkspace()
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    DataCopyExtParams xWorkspaceGroupCopyParams{(uint16_t)1, (uint32_t)(groupEles_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> xWorkspaceGroupPadParams{true, 0, (uint8_t)(groupElesAlignBlockCountFp32_ - groupEles_),
                                                         floatNegativeInf_};
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    Duplicate<float>(xLocal, floatNegativeInf_, groupElesAlignSortCount_ * groupNum_);
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    for (size_t i = 0; i < groupNum_; i++) {
        DataCopyPad(xLocal[groupElesAlignSortCount_ * i], mGmWorkspace_[wsOffset_ + groupEles_ * i],
                    xWorkspaceGroupCopyParams, xWorkspaceGroupPadParams);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::CopyOut(const int32_t loop)
{
    int64_t offset = loop * k_;
    LocalTensor<float> yLocal = yOutBuf_.Get<float>();

    DataCopyExtParams copyParams{1, (uint32_t)(k_ * sizeof(float)), 0, 0, 0};
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(mGmY_[offset], yLocal, copyParams);

    LocalTensor<int32_t> indicesLocal = indicesOutBuf_.Get<int32_t>();
    if constexpr (enableExpertMapping) {
        LocalTensor<int32_t> mappingNumLocal = mappingNumBuf_.Get<int32_t>();

        // Update Indices
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        for (size_t kI = 0; kI < k_; kI++) {
            uint32_t expertId = indicesLocal.GetValue(kI);
            uint32_t expertMappingNum = mappingNumLocal.GetValue(expertId);
            uint32_t redundantOffset = expertMappingNum == 0 ? 0 : (batchOffset_ + loop) % expertMappingNum;
            uint32_t tableOffset = expertId * tableDim_ + redundantOffset;
            indicesLocal.SetValue(kI, mappingTableGm_[tableOffset].GetValue(0));
        }
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    }
    DataCopyPad(mGmIndices_[offset], indicesLocal, copyParams);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::Process()
{
    CopyInAddNum();
    if constexpr (enableExpertMapping) {
        // CopyIn mappingNum
        CopyInMappingNum();
    }
    for (size_t loop = 0; loop < loopBatch_; loop++) {
        CopyInX(loop);
        ActivateAndAdd();
        CopyToWorkspace();
        CopyFromWorkspace();
        GroupReduceSumInternelImpl();
        GroupTopkImpl();
        GatherSigmoidImpl();

        if (isNorm_ == 1) {
            NormImpl();
        }
        CopyOut(loop);
    }
}

#endif // ASCENDC_MOE_FUSED_TOPK_H_
