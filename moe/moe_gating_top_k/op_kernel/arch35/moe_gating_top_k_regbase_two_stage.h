/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_GATING_TOP_K_REGBASE_TWO_STAGE_H
#define MOE_GATING_TOP_K_REGBASE_TWO_STAGE_H

#include "kernel_operator.h"
#include "basic_api/kernel_vec_intf.h"
#include "simt_api/asc_simt.h"
#include "common.h"

namespace MoeGatingTopK {
using namespace AscendC;

constexpr uint32_t TWO_STAGE_THREAD_NUM = 1024;
constexpr SortConfig TWO_STAGE_SORT_CONFIG{SortType::RADIX_SORT, true};

__simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__ inline void BuildTopKIndexByRankInverse(
    uint32_t totalElems, uint32_t segmentLen, uint32_t k, uint32_t topKStride, __ubuf__ uint32_t *stage1Order,
    __ubuf__ uint32_t *rankInverse, __ubuf__ int32_t *finalIdx)
{
    for (uint32_t rank = static_cast<uint32_t>(threadIdx.x); rank < totalElems; rank += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = stage1Order[rank];
        rankInverse[flatIdx] = rank;
    }
    asc_syncthreads();

    for (uint32_t flat = static_cast<uint32_t>(threadIdx.x); flat < totalElems; flat += TWO_STAGE_THREAD_NUM) {
        uint32_t segId = flat / segmentLen;
        uint32_t laneId = flat - segId * segmentLen;
        uint32_t rank = rankInverse[flat];
        uint32_t segBase = segId * segmentLen;
        uint32_t pos = 0;
        for (uint32_t lane = 0; lane < segmentLen; ++lane) {
            pos += (rankInverse[segBase + lane] < rank) ? 1U : 0U;
        }
        if (pos < k) {
            finalIdx[segId * topKStride + pos] = static_cast<int32_t>(laneId);
        }
    }
}

template <typename T>
class MoeGatingTopKRegbaseTwoStage {
public:
    __aicore__ inline MoeGatingTopKRegbaseTwoStage() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyBias();
    __aicore__ inline uint32_t GetValidRows(uint32_t rowStart) const;
    __aicore__ inline void ComputeBatch(uint32_t rowStart, uint32_t validRows);
    __aicore__ inline void SortBatch(uint32_t validRows);
    __aicore__ inline void CopyOutBatch(uint32_t rowStart, uint32_t validRows);
    __aicore__ inline void CopyOutOneRow(uint32_t rowStart, uint32_t rowInBatch);

private:
    TPipe *pipe_ = nullptr;
    const MoeGatingTopKRegbaseTilingData *tilingData_ = nullptr;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;

    TBuf<TPosition::VECCALC> inputRawBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> softmaxSrcBuf_;
    TBuf<TPosition::VECCALC> rowNormBuf_;
    TBuf<TPosition::VECCALC> xNormBuf_;
    TBuf<TPosition::VECCALC> xBiasBuf_;
    TBuf<TPosition::VECCALC> stage1ValueBuf_;
    TBuf<TPosition::VECCALC> stage1OrderBuf_;
    TBuf<TPosition::VECCALC> rankInverseBuf_;
    TBuf<TPosition::VECCALC> finalIdxBuf_;
    TBuf<TPosition::VECCALC> yFloatBuf_;
    TBuf<TPosition::VECCALC> yCastBuf_;
    TBuf<TPosition::VECCALC> expertOutBuf_;
    TBuf<TPosition::VECCALC> gatherOffsetBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    uint32_t blockIdx_ = 0;
    uint32_t curCoreRows_ = 0;
    uint32_t perCoreRows_ = 0;
    uint32_t expertCount_ = 0;
    uint32_t expertStride_ = 0;
    uint32_t inputStride_ = 0;
    uint32_t softmaxSrcStride_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t maxBatchElems_ = 0;
    uint32_t k_ = 0;
    uint32_t kAlign_ = 0;
    bool hasBias_ = false;
    uint32_t normType_ = 0;
    uint32_t renorm_ = 0;
    float eps_ = 0.0f;
    float routedScalingFactor_ = 1.0f;
};

template <typename T>
__aicore__ inline void MoeGatingTopKRegbaseTwoStage<T>::Init(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
    const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *pipe)
{
    (void)out;
    (void)workspace;
    tilingData_ = tilingData;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    perCoreRows_ = static_cast<uint32_t>(tilingData_->perCoreRowCount);
    curCoreRows_ = static_cast<uint32_t>(
        blockIdx_ == GetBlockNum() - 1 ? tilingData_->lastCoreRowCount : tilingData_->perCoreRowCount);
    expertCount_ = static_cast<uint32_t>(tilingData_->expertCount);
    expertStride_ = static_cast<uint32_t>(tilingData_->perGroupExpertCountAlign);
    inputStride_ = static_cast<uint32_t>(Align(expertCount_, sizeof(T)));
    k_ = static_cast<uint32_t>(tilingData_->k);
    kAlign_ = static_cast<uint32_t>(Align(k_, sizeof(float)));
    batchSize_ = static_cast<uint32_t>(tilingData_->twoStageBatchSize);
    maxBatchElems_ = batchSize_ * expertStride_;
    normType_ = static_cast<uint32_t>(tilingData_->normType);
    uint32_t softmaxSrcFactor = (normType_ == 0 && !IsSameType<T, float>::value) ?
        static_cast<uint32_t>(sizeof(float) / sizeof(T)) : 1U;
    softmaxSrcStride_ = expertStride_ * softmaxSrcFactor;
    renorm_ = static_cast<uint32_t>(tilingData_->renorm);
    eps_ = tilingData_->eps;
    routedScalingFactor_ = tilingData_->routedScalingFactor;

    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRows_ * expertCount_ * blockIdx_, expertCount_);
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRows_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRows_ * k_ * blockIdx_, k_);
    if (bias != nullptr) {
        hasBias_ = true;
        biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    }

    pipe_->InitBuffer(inputRawBuf_, AlignBytes(inputStride_, sizeof(T)));
    pipe_->InitBuffer(biasBuf_, AlignBytes(expertCount_, sizeof(float)));
    pipe_->InitBuffer(softmaxSrcBuf_, AlignBytes(softmaxSrcStride_, sizeof(float)));
    pipe_->InitBuffer(rowNormBuf_, AlignBytes(expertStride_, sizeof(float)));
    pipe_->InitBuffer(xNormBuf_, AlignBytes(maxBatchElems_, sizeof(float)));
    pipe_->InitBuffer(xBiasBuf_, AlignBytes(maxBatchElems_, sizeof(float)));
    pipe_->InitBuffer(stage1ValueBuf_, AlignBytes(maxBatchElems_, sizeof(float)));
    pipe_->InitBuffer(stage1OrderBuf_, AlignBytes(maxBatchElems_, sizeof(uint32_t)));
    pipe_->InitBuffer(rankInverseBuf_, AlignBytes(maxBatchElems_, sizeof(uint32_t)));
    pipe_->InitBuffer(finalIdxBuf_, AlignBytes(batchSize_ * kAlign_, sizeof(int32_t)));
    pipe_->InitBuffer(yFloatBuf_, AlignBytes(kAlign_, sizeof(float)));
    pipe_->InitBuffer(yCastBuf_, AlignBytes(kAlign_, sizeof(T)));
    pipe_->InitBuffer(expertOutBuf_, AlignBytes(kAlign_, sizeof(int32_t)));
    pipe_->InitBuffer(gatherOffsetBuf_, AlignBytes(kAlign_, sizeof(uint32_t)));
    pipe_->InitBuffer(tmpBuf_, static_cast<uint32_t>(tilingData_->twoStageTmpUbSize));
}

template <typename T>
__aicore__ inline void MoeGatingTopKRegbaseTwoStage<T>::CopyBias()
{
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();
    if (!hasBias_) {
        Duplicate(biasTensor, static_cast<float>(0), expertCount_);
        PipeBarrier<PIPE_V>();
        return;
    }
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(biasTensor, biasGm_, copyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    } else {
        LocalTensor<T> rawTensor = inputRawBuf_.Get<T>();
        DataCopyPad(rawTensor, biasGm_, copyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Cast(biasTensor, rawTensor, RoundMode::CAST_NONE, expertCount_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline uint32_t MoeGatingTopKRegbaseTwoStage<T>::GetValidRows(uint32_t rowStart) const
{
    uint32_t remainRows = curCoreRows_ - rowStart;
    return remainRows > batchSize_ ? batchSize_ : remainRows;
}

template <typename T>
__aicore__ inline void MoeGatingTopKRegbaseTwoStage<T>::ComputeBatch(uint32_t rowStart, uint32_t validRows)
{
    LocalTensor<T> rawTensor = inputRawBuf_.Get<T>();
    LocalTensor<float> softmaxSrcTensor = softmaxSrcBuf_.Get<float>();
    LocalTensor<float> rowNormTensor = rowNormBuf_.Get<float>();
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();
    LocalTensor<uint8_t> tmpTensor = tmpBuf_.Get<uint8_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    for (uint32_t row = 0; row < validRows; ++row) {
        uint32_t elemOffset = row * expertStride_;
        Duplicate(xBiasTensor[elemOffset], *((float *)&MIN_FP32), expertStride_);
        if (normType_ == 0) {
            Duplicate(softmaxSrcTensor, *((float *)&MIN_FP32), softmaxSrcStride_);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPadExtParams<float> floatPadParams{true, 0,
                                                       static_cast<uint8_t>(expertStride_ - expertCount_),
                                                       *((float *)&MIN_FP32)};
            if (normType_ == 0) {
                DataCopyPad(softmaxSrcTensor, xGm_[(rowStart + row) * expertCount_], copyParams, floatPadParams);
            } else {
                DataCopyPad(xBiasTensor[elemOffset], xGm_[(rowStart + row) * expertCount_], copyParams,
                            floatPadParams);
            }
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else {
            DataCopyPad(rawTensor, xGm_[(rowStart + row) * expertCount_], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            if (normType_ == 0) {
                Cast(softmaxSrcTensor, rawTensor, RoundMode::CAST_NONE, expertCount_);
            } else {
                Cast(xBiasTensor[elemOffset], rawTensor, RoundMode::CAST_NONE, expertCount_);
            }
            PipeBarrier<PIPE_V>();
        }
        if (normType_ == 0) {
            SoftMaxShapeInfo softmaxShapeInfo;
            softmaxShapeInfo.srcM = 1;
            softmaxShapeInfo.srcK = expertStride_;
            softmaxShapeInfo.oriSrcM = 1;
            softmaxShapeInfo.oriSrcK = expertStride_;
            SoftMax<float, true, false>(rowNormTensor, softmaxSrcTensor, tmpTensor,
                                        tilingData_->softmaxTilingData, softmaxShapeInfo);
        } else {
            Sigmoid(rowNormTensor, xBiasTensor[elemOffset], tmpTensor, expertCount_);
            PipeBarrier<PIPE_V>();
        }
        Adds(xNormTensor[elemOffset], rowNormTensor, static_cast<float>(0), expertStride_);
        PipeBarrier<PIPE_V>();
        Add(xBiasTensor[elemOffset], rowNormTensor, biasTensor, expertCount_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKRegbaseTwoStage<T>::SortBatch(uint32_t validRows)
{
    uint32_t totalElems = validRows * expertStride_;
    LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
    LocalTensor<float> stage1Values = stage1ValueBuf_.Get<float>();
    LocalTensor<uint32_t> stage1Order = stage1OrderBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> rankInverse = rankInverseBuf_.Get<uint32_t>();
    LocalTensor<int32_t> finalIdx = finalIdxBuf_.Get<int32_t>();
    LocalTensor<uint8_t> tmpTensor = tmpBuf_.Get<uint8_t>();

    ArithProgression(stage1Order.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), static_cast<int32_t>(1),
                     totalElems);
    PipeBarrier<PIPE_V>();
    AscendC::Sort<float, false, TWO_STAGE_SORT_CONFIG>(stage1Values, stage1Order, xBiasTensor, tmpTensor, totalElems);
    PipeBarrier<PIPE_V>();
    asc_vf_call<BuildTopKIndexByRankInverse>(
        dim3(TWO_STAGE_THREAD_NUM), totalElems, expertStride_, k_, kAlign_,
        (__ubuf__ uint32_t *)stage1Order.GetPhyAddr(), (__ubuf__ uint32_t *)rankInverse.GetPhyAddr(),
        (__ubuf__ int32_t *)finalIdx.GetPhyAddr());
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
}

template <typename T>
__aicore__ inline void MoeGatingTopKRegbaseTwoStage<T>::CopyOutOneRow(uint32_t rowStart, uint32_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<int32_t> finalIdx = finalIdxBuf_.Get<int32_t>();
    LocalTensor<int32_t> expertOut = expertOutBuf_.Get<int32_t>();
    LocalTensor<int32_t> gatherOffset = gatherOffsetBuf_.Get<int32_t>();
    LocalTensor<float> yFloat = yFloatBuf_.Get<float>();
    LocalTensor<T> yCast = yCastBuf_.Get<T>();
    uint32_t rowOffset = rowInBatch * expertStride_;
    uint32_t topKOffset = rowInBatch * kAlign_;

    DataCopy(expertOut, finalIdx[topKOffset], Align(k_, sizeof(int32_t)));
    PipeBarrier<PIPE_V>();
    Muls(gatherOffset, expertOut, static_cast<int32_t>(sizeof(float)), k_);
    PipeBarrier<PIPE_V>();
    Gather(yFloat, xNormTensor[rowOffset], gatherOffset.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), k_);
    PipeBarrier<PIPE_V>();

    if (!(normType_ == 0 && renorm_ == 0)) {
        LocalTensor<float> sumTensor = tmpBuf_.Get<float>();
        LocalTensor<float> divTensor = tmpBuf_.Get<float>()[BLOCK_BYTES];
        ReduceSum(sumTensor, yFloat, divTensor, k_);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        float sumValue = sumTensor.GetValue(0) + eps_;
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        Duplicate(divTensor, sumValue, k_);
        PipeBarrier<PIPE_V>();
        Div(yFloat, yFloat, divTensor, k_);
        PipeBarrier<PIPE_V>();
    }
    Muls(yFloat, yFloat, routedScalingFactor_, k_);
    PipeBarrier<PIPE_V>();

    DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(yGm_[(rowStart + rowInBatch) * k_], yFloat, yCopyParams);
    } else {
        Cast(yCast, yFloat, RoundMode::CAST_RINT, k_);
        PipeBarrier<PIPE_V>();
        DataCopyPad(yGm_[(rowStart + rowInBatch) * k_], yCast, yCopyParams);
    }
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(k_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(expertIdxGm_[(rowStart + rowInBatch) * k_], expertOut, idxCopyParams);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}

template <typename T>
__aicore__ inline void MoeGatingTopKRegbaseTwoStage<T>::CopyOutBatch(uint32_t rowStart, uint32_t validRows)
{
    for (uint32_t row = 0; row < validRows; ++row) {
        CopyOutOneRow(rowStart, row);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKRegbaseTwoStage<T>::Process()
{
    if (batchSize_ == 0 || curCoreRows_ == 0 || expertCount_ == 0 || k_ == 0) {
        return;
    }
    CopyBias();
    for (uint32_t rowStart = 0; rowStart < curCoreRows_; rowStart += batchSize_) {
        uint32_t validRows = GetValidRows(rowStart);
        ComputeBatch(rowStart, validRows);
        SortBatch(validRows);
        CopyOutBatch(rowStart, validRows);
    }
}

} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_REGBASE_TWO_STAGE_H
