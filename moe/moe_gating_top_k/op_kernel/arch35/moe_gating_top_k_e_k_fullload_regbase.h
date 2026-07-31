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
 * \file moe_gating_top_k_e_k_fullload_regbase.h
 * \brief Arch35 register-based kernel for MoE Gating TopK DeepSeekV3 high-performance scenario.
 *        Conditions: expertCount==256, groupCount==8, kGroup==4, k<=32, addBias,
 *        groupSelectMode==SUM, renorm==NO, normType==SIGMOID, !outFlag, !hashFlag.
 *        Optimizations:
 *        1. perGroupExpertCount==32 → single Sort32 per group.
 *        2. kGroup==4 → single MrgSort4 for final merge.
 *        3. k<=32 → MicroAPI single-register score gather.
 *        4. MicroAPI Sigmoid + AddBias normalization.
 */
#ifndef MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_H
#define MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_H

#include <cmath>
#include "common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"

namespace MoeGatingTopK {
using namespace AscendC;
using MicroAPI::RegTensor;

constexpr uint32_t EK_VL_FLOAT_SIZE = VECTOR_REG_WIDTH / sizeof(float);
constexpr MicroAPI::DivSpecificMode EK_DIV_MODE = {MicroAPI::MaskMergeMode::ZEROING, true};

template <typename T>
class MoeGatingTopKEKFullloadRegbase {
public:
    __aicore__ inline MoeGatingTopKEKFullloadRegbase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out,
                                GM_ADDR workspace, const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBias();
    __aicore__ inline void CopyInX(int64_t row);
    __aicore__ inline void ComputeX();
    __aicore__ inline void SortInGroup();
    __aicore__ inline void SelectTopKGroupIndex();
    __aicore__ inline void SelectTopKExpertIdx();
    __aicore__ inline void SelectTopKExpertScore();
    __aicore__ inline void CopyOut(int64_t row);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 2> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;

    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> xSigmoidBuf_;
    TBuf<TPosition::VECCALC> xBiasBuf_;
    TBuf<TPosition::VECCALC> sortedInGroupBuf_;
    TBuf<TPosition::VECCALC> sortedGroupBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<float> outGm_;

    int64_t blockIdx_;
    int64_t perCoreRowCount_;
    int64_t curCoreRowCount_;
    int64_t expertCount_;
    int64_t expertCountAlign_;
    int64_t k_;
    int64_t kGroup_;
    int64_t groupCount_;
    int64_t perGroupExpertCount_;
    float routedScalingFactor_;
    float eps_;
    bool hasBias_;

    const MoeGatingTopKRegbaseTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::CopyInBias()
{
    if (!hasBias_) {
        return;
    }
    LocalTensor<T> biasTensor = biasBuf_.Get<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(biasTensor, biasGm_, dataCopyParams, dataCopyPadParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::CopyInX(int64_t row)
{
    LocalTensor<T> xInLocalTensor = xInQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xInLocalTensor, xGm_[row * expertCount_], dataCopyParams, dataCopyPadParams);
    xInQueue_.EnQue(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::ComputeX()
{
    LocalTensor<T> xInLocalTensor = xInQueue_.DeQue<T>();
    LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
    LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
    LocalTensor<T> biasTensor = biasBuf_.Get<T>();

    uint32_t size = static_cast<uint32_t>(expertCount_);
    uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, EK_VL_FLOAT_SIZE));

    __local_mem__ T *inputAddr = (__local_mem__ T *)xInLocalTensor.GetPhyAddr();
    __local_mem__ float *sigmoidOutAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
    __local_mem__ float *addBiasOutAddr = (__local_mem__ float *)xBiasTensor.GetPhyAddr();
    __local_mem__ T *biasAddr = hasBias_ ? (__local_mem__ T *)biasTensor.GetPhyAddr() : nullptr;

    __VEC_SCOPE__
    {
        RegTensor<float> vregBiasFp32;
        RegTensor<float> vregSigmoidResult;
        RegTensor<float> vregBiasResult;
        RegTensor<float> vregOne;
        RegTensor<float> vregInFp32;
        RegTensor<float> vreg1;
        RegTensor<float> vreg2;
        RegTensor<float> vreg3;
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
        MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<float>(size);
            if (hasBias_) {
                ops::LoadTwoTensorForDtypeT<T>(inputAddr, biasAddr, vregInFp32, vregBiasFp32, preg0, preg0,
                                               i * EK_VL_FLOAT_SIZE, i * EK_VL_FLOAT_SIZE);
            } else {
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregInFp32, preg0, i * EK_VL_FLOAT_SIZE);
            }
            MicroAPI::Muls(vreg1, vregInFp32, static_cast<float>(-1), preg0);
            MicroAPI::Exp(vreg2, vreg1, preg0);
            MicroAPI::Adds(vreg3, vreg2, static_cast<float>(1), preg0);
            MicroAPI::Div<float, &EK_DIV_MODE>(vregSigmoidResult, vregOne, vreg3, preg0);
            if (hasBias_) {
                MicroAPI::Add(vregBiasResult, vregSigmoidResult, vregBiasFp32, preg0);
            } else {
                vregBiasResult = vregSigmoidResult;
            }
            MicroAPI::DataCopy(sigmoidOutAddr + i * EK_VL_FLOAT_SIZE, vregSigmoidResult, preg0);
            MicroAPI::DataCopy(addBiasOutAddr + i * EK_VL_FLOAT_SIZE, vregBiasResult, preg0);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE>();
    }

    xInQueue_.FreeTensor(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SortInGroup()
{
    LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
    LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
    LocalTensor<uint32_t> indexTensor = indexBuf_.Get<uint32_t>();

    ArithProgression(indexTensor.ReinterpretCast<int32_t>(), 0, 1, expertCount_);
    PipeBarrier<PIPE_V>();
    Sort32(sortedInGroupTensor, xBiasTensor, indexTensor, expertCount_ / ONE_REPEAT_SORT_NUM);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SelectTopKGroupIndex()
{
    LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
    LocalTensor<float> top2ValueInGroupTensor = calcTmpBuf_.Get<float>();
    LocalTensor<uint32_t> indexTensor = indexBuf_.Get<uint32_t>();
    LocalTensor<float> sortedGroupTensor = sortedGroupBuf_.Get<float>();

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    indexTensor.SetValue(0, static_cast<uint32_t>(5));
    indexTensor.SetValue(1, static_cast<uint32_t>(0));
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    uint64_t rsvdCnt = 0;
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 8;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8;
    gatherMaskParams.src1RepeatStride = 0;
    GatherMask(top2ValueInGroupTensor, sortedInGroupTensor, indexTensor, true, static_cast<uint32_t>(64),
               gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> groupTop2SumTensor = top2ValueInGroupTensor;
    PairReduceSum(groupTop2SumTensor, top2ValueInGroupTensor, 1, groupCount_ * 2, 1, 1, 1);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> groupIndexTensor = indexTensor;
    ArithProgression(groupIndexTensor.ReinterpretCast<int32_t>(), 0, 1, groupCount_);
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = ONE_REPEAT_SORT_NUM - groupCount_;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX << groupCount_;
        uint64_t mask[2] = {mask0, 0};
        Duplicate(groupTop2SumTensor, MIN_FP32, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    Sort32(sortedGroupTensor, groupTop2SumTensor, groupIndexTensor, 1);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> sortedGroupIndexTensor = indexTensor.ReinterpretCast<int32_t>();
    uint8_t src1Pattern = 2;
    GatherMask(sortedGroupIndexTensor, sortedGroupTensor.template ReinterpretCast<int32_t>(), src1Pattern, false,
               static_cast<uint32_t>(0), {1, 1, 0, 0}, rsvdCnt);

    Cast(sortedGroupTensor, sortedGroupIndexTensor, RoundMode::CAST_ROUND, kGroup_);
    PipeBarrier<PIPE_V>();
    duplicateNum = ONE_REPEAT_SORT_NUM - kGroup_;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX << kGroup_;
        uint64_t mask[2] = {mask0, 0};
        Duplicate(sortedGroupTensor, MIN_FP32, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    Sort32(top2ValueInGroupTensor, sortedGroupTensor, sortedGroupIndexTensor.template ReinterpretCast<uint32_t>(), 1);
    PipeBarrier<PIPE_V>();
    src1Pattern = 1;
    GatherMask(sortedGroupTensor, top2ValueInGroupTensor, src1Pattern, false, static_cast<uint32_t>(0),
               {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    Cast(sortedGroupIndexTensor, sortedGroupTensor, RoundMode::CAST_ROUND, kGroup_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SelectTopKExpertIdx()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> topKGroupIndexTensor = calcTmpBuf_.Get<int32_t>();
    LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
    LocalTensor<float> sortedExpertTensor = xInQueue_.AllocTensor<float>();

    AscendC::MrgSort4Info params;
    params.elementLengths[0] = k_;
    params.elementLengths[1] = k_;
    params.elementLengths[2] = k_;
    params.elementLengths[3] = k_;
    params.ifExhaustedSuspension = true;
    params.validBit = 0b1111;
    params.repeatTimes = 1;

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    int64_t listOffset1 = topKGroupIndexTensor.GetValue(3) * perGroupExpertCount_ * 2;
    int64_t listOffset2 = topKGroupIndexTensor.GetValue(2) * perGroupExpertCount_ * 2;
    int64_t listOffset3 = topKGroupIndexTensor.GetValue(1) * perGroupExpertCount_ * 2;
    int64_t listOffset4 = topKGroupIndexTensor.GetValue(0) * perGroupExpertCount_ * 2;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = sortedInGroupTensor[listOffset1];
    srcList.src2 = sortedInGroupTensor[listOffset2];
    srcList.src3 = sortedInGroupTensor[listOffset3];
    srcList.src4 = sortedInGroupTensor[listOffset4];
    MrgSort<float>(sortedExpertTensor, srcList, params);
    PipeBarrier<PIPE_V>();

    uint64_t rsvdCnt = 0;
    uint8_t src1Pattern = 2;
    GatherMask(expertIdxTensor, sortedExpertTensor.template ReinterpretCast<int32_t>(), src1Pattern, false,
               static_cast<uint32_t>(0), {1, 1, 0, 0}, rsvdCnt);

    xInQueue_.FreeTensor(sortedExpertTensor);
    expertIdxOutQueue_.EnQue(expertIdxTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SelectTopKExpertScore()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
    LocalTensor<T> yTensor = yOutQueue_.AllocTensor<T>();

    __local_mem__ float *inputAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
    __local_mem__ uint32_t *expertIdxAddr = (__local_mem__ uint32_t *)expertIdxTensor.GetPhyAddr();
    __local_mem__ T *outputAddr = (__local_mem__ T *)yTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vreg2;
        RegTensor<float> vreg3;
        RegTensor<float> vreg4;
        RegTensor<float> vregOutput;
        RegTensor<uint32_t> vreg0;
        RegTensor<uint32_t> vreg1;

        uint32_t kU32 = static_cast<uint32_t>(k_);
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(kU32);

        MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1, expertIdxAddr);
        MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);

        MicroAPI::ReduceSum(vreg3, vreg2, preg0);
        MicroAPI::Adds(vreg3, vreg3, eps_, preg0);
        MicroAPI::Duplicate(vreg4, vreg3, preg0);
        MicroAPI::Div(vreg4, vreg2, vreg4, preg0);
        MicroAPI::Muls(vregOutput, vreg4, routedScalingFactor_, preg0);

        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg0, 0);
    }

    yOutQueue_.EnQue(yTensor);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::CopyOut(int64_t row)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[row * k_], yOutTensor, dataCopyParams);
    dataCopyParams.blockLen = k_ * sizeof(int32_t);
    DataCopyPad(expertIdxGm_[row * k_], expertIdxTensor, dataCopyParams);
    expertIdxOutQueue_.FreeTensor(expertIdxTensor);
    yOutQueue_.FreeTensor(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::Init(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
    const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    k_ = tilingData_->k;
    kGroup_ = tilingData_->kGroup;
    groupCount_ = tilingData_->groupCount;
    perGroupExpertCount_ = tilingData_->perGroupExpertCount;
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;
    hasBias_ = (tilingData_->addBias == 1) && (bias != nullptr);

    expertCountAlign_ = Align(expertCount_, sizeof(float));

    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    if (hasBias_) {
        biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    }
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ float *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    int64_t groupCountAlign = static_cast<int64_t>(CeilAlign(groupCount_, ONE_REPEAT_SORT_NUM));
    pipe_->InitBuffer(xInQueue_, CONSTANT_TWO, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, 1, AlignBytes(k_, sizeof(T)));
    pipe_->InitBuffer(expertIdxOutQueue_, 1, AlignBytes(k_, sizeof(int32_t)));

    pipe_->InitBuffer(biasBuf_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(xSigmoidBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(xBiasBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(sortedInGroupBuf_, expertCountAlign_ * sizeof(float) * CONSTANT_TWO);
    pipe_->InitBuffer(sortedGroupBuf_, groupCountAlign * sizeof(float) * CONSTANT_TWO);
    pipe_->InitBuffer(indexBuf_, expertCountAlign_ * sizeof(int32_t));
    pipe_->InitBuffer(calcTmpBuf_, expertCountAlign_ * sizeof(float));
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::Process()
{
    CopyInBias();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeX();
        SortInGroup();
        SelectTopKGroupIndex();
        SelectTopKExpertIdx();
        SelectTopKExpertScore();
        CopyOut(row);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_H
