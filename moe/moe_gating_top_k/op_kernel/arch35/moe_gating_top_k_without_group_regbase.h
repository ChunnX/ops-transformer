/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_without_group_regbase.h
 * \brief Arch35 register-based kernel for MoE Gating TopK without group.
 */
#ifndef MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H
#define MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H

#include <cmath>
#include "common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "basic_api/kernel_operator_utils_intf.h"
#include "moe_gating_top_k_regbase.h"

namespace MoeGatingTopK {
using namespace AscendC;
using MicroAPI::RegTensor;

constexpr int32_t WG_CONSTANT_TWO = 2;
constexpr int32_t WG_CONSTANT_TEN = 10;
constexpr uint32_t WG_VL_FLOAT_SIZE = VECTOR_REG_WIDTH / sizeof(float);
constexpr int64_t DEFAULT_BATCH_ROWS = 4;
constexpr int64_t SINGLE_EXPERT_FALLBACK_BATCH = 512;
constexpr MicroAPI::DivSpecificMode WG_DIV_MODE = {MicroAPI::MaskMergeMode::ZEROING, true};

template <typename T, typename U1 = int32_t, typename U2 = int32_t>
class MoeGatingTopKWithoutGroupRegbase : public MoeGatingTopKRegbase<T, U1, U2> {
public:
    __aicore__ inline MoeGatingTopKWithoutGroupRegbase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid, GM_ADDR y,
                                GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    using MoeGatingTopKRegbase<T, U1, U2>::pipe_;
    using MoeGatingTopKRegbase<T, U1, U2>::xInQueue_;
    using MoeGatingTopKRegbase<T, U1, U2>::yOutQueue_;
    using MoeGatingTopKRegbase<T, U1, U2>::expertIdxOutQueue_;
    using MoeGatingTopKRegbase<T, U1, U2>::outOutQueue_;
    using MoeGatingTopKRegbase<T, U1, U2>::biasBuf_;
    using MoeGatingTopKRegbase<T, U1, U2>::xSigmoidBuf_;
    using MoeGatingTopKRegbase<T, U1, U2>::xBiasBuf_;
    using MoeGatingTopKRegbase<T, U1, U2>::indexBuffer_;
    using MoeGatingTopKRegbase<T, U1, U2>::sortedInGroupBuf_;
    using MoeGatingTopKRegbase<T, U1, U2>::finalSortBuffer_;
    using MoeGatingTopKRegbase<T, U1, U2>::xGm_;
    using MoeGatingTopKRegbase<T, U1, U2>::biasGm_;
    using MoeGatingTopKRegbase<T, U1, U2>::yGm_;
    using MoeGatingTopKRegbase<T, U1, U2>::expertIdxGm_;
    using MoeGatingTopKRegbase<T, U1, U2>::outGm_;
    using MoeGatingTopKRegbase<T, U1, U2>::inputIdsGm_;
    using MoeGatingTopKRegbase<T, U1, U2>::tid2eidGm_;
    using MoeGatingTopKRegbase<T, U1, U2>::indexTensor;
    using MoeGatingTopKRegbase<T, U1, U2>::blockIdx_;
    using MoeGatingTopKRegbase<T, U1, U2>::curCoreRowCount_;
    using MoeGatingTopKRegbase<T, U1, U2>::expertCount_;
    using MoeGatingTopKRegbase<T, U1, U2>::k_;
    using MoeGatingTopKRegbase<T, U1, U2>::kGroup_;
    using MoeGatingTopKRegbase<T, U1, U2>::groupCount_;
    using MoeGatingTopKRegbase<T, U1, U2>::perGroupExpertCount_;
    using MoeGatingTopKRegbase<T, U1, U2>::perGroupExpertCountAlign_;
    using MoeGatingTopKRegbase<T, U1, U2>::expertCountAlign_;
    using MoeGatingTopKRegbase<T, U1, U2>::routedScalingFactor_;
    using MoeGatingTopKRegbase<T, U1, U2>::eps_;
    using MoeGatingTopKRegbase<T, U1, U2>::hasBias_;
    using MoeGatingTopKRegbase<T, U1, U2>::hashFlag_;
    using MoeGatingTopKRegbase<T, U1, U2>::noNorm_;
    using MoeGatingTopKRegbase<T, U1, U2>::normType_;
    using MoeGatingTopKRegbase<T, U1, U2>::renorm_;
    using MoeGatingTopKRegbase<T, U1, U2>::outFlag_;
    using MoeGatingTopKRegbase<T, U1, U2>::groupSelectMode_;
    using MoeGatingTopKRegbase<T, U1, U2>::tilingData_;
    using MoeGatingTopKRegbase<T, U1, U2>::CopyInBias;
    using MoeGatingTopKRegbase<T, U1, U2>::CopyOutXNorm;
    using MoeGatingTopKRegbase<T, U1, U2>::TopKComputeCore;
    using MoeGatingTopKRegbase<T, U1, U2>::HashComputeCore;
    using MoeGatingTopKRegbase<T, U1, U2>::InitBase;
    using MoeGatingTopKRegbase<T, U1, U2>::PadWithMinFp32;

    __aicore__ inline void CopyInXBatch(int64_t batchIdx, int64_t rowsInBatch);
    __aicore__ inline void ComputeNormForTop1(LocalTensor<float> xInTensor, int64_t rowInBatch);
    __aicore__ inline void CopyOutXNormTop1(int64_t globalRow);
    __aicore__ inline void ComputeXForBatchRow(LocalTensor<float> xInTensor, int64_t rowOffset);
    __aicore__ inline void ComputeOneRow(LocalTensor<float> xInTensor, int64_t r, int64_t localRow,
                                          LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut);
    __aicore__ inline void SelectTop1AndScore(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                              int64_t rowInBatch);
    __aicore__ inline void CopyOutBatch(int64_t globalBaseRow, int64_t rowsInBatch);
    __aicore__ inline void ProcessSingleExpert();
    __aicore__ inline void SelectTop1AndScoreNoNorm(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                                    int64_t rowInBatch);
    __aicore__ inline void SelectTop1AndScoreWithNorm(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                                      int64_t rowInBatch);
    __aicore__ inline void ProcessSingleExpertSigmoid(int64_t batchSize, int64_t numBatches);
    __aicore__ inline void ProcessSingleExpertSoftMaxRenorm(int64_t batchSize, int64_t numBatches);
    __aicore__ inline void ProcessSingleExpertSoftMaxNoRenorm(int64_t batchSize, int64_t numBatches);
    __aicore__ inline void ProcessSingleExpertSoftplus(int64_t batchSize, int64_t numBatches);

private:
    TBuf<TPosition::VECCALC> xNormBuf_;
    TBuf<TPosition::VECCALC> xNormWithBiasBuf_;
    TBuf<TPosition::VECCALC> sortedBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;

    int64_t batchRows_ = 0;
    int64_t kStride_ = 0;
};

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::CopyInXBatch(int64_t batchIdx, int64_t rowsInBatch)
{
    LocalTensor<float> xInTensor = xInQueue_.AllocTensor<float>();

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = static_cast<uint16_t>(rowsInBatch * groupCount_);
    dataCopyParams.blockLen = static_cast<uint32_t>(perGroupExpertCount_ * sizeof(T));
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = static_cast<uint32_t>((perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(T) / BLOCK_BYTES);
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

    int64_t gmOffset = batchIdx * batchRows_ * expertCount_;
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(xInTensor, xGm_[gmOffset], dataCopyParams, dataCopyPadParams);
    } else {
        LocalTensor<T> rawTensor = xInTensor[batchRows_ * expertCountAlign_].ReinterpretCast<T>();
        DataCopyPad(rawTensor, xGm_[gmOffset], dataCopyParams, dataCopyPadParams);
    }
    xInQueue_.EnQue(xInTensor);
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ComputeNormForTop1(LocalTensor<float> xInTensor,
                                                                                       int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();

    int64_t rowOffset = rowInBatch * expertCountAlign_;

    if constexpr (!IsSameType<T, float>::value) {
        LocalTensor<T> rawTensor = xInTensor[batchRows_ * expertCountAlign_].ReinterpretCast<T>();
        Cast(xInTensor[rowOffset], rawTensor[rowOffset], RoundMode::CAST_NONE, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> xRow = xInTensor[rowOffset];

    int64_t duplicateNum = expertCount_ % ONE_REPEAT_SORT_NUM;
    int64_t duplicateIndex = expertCount_ - duplicateNum;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(xRow.ReinterpretCast<int32_t>()[duplicateIndex], MIN_FP32, mask, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<uint8_t> calcNormTmpTensor = calcTmpBuf_.Get<uint8_t>();
    if (normType_ == 1) {
        Sigmoid(xNormTensor, xRow, calcNormTmpTensor, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    } else if (normType_ == 0) {
        uint32_t size = static_cast<uint32_t>(expertCountAlign_);
        __ubuf__ float *xRowAddr = (__ubuf__ float *)xRow.GetPhyAddr();
        __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
        __ubuf__ float *xNormWithBiasAddr = (__ubuf__ float *)xNormWithBiasTensor.GetPhyAddr();
        __ubuf__ float *biasAddr = (__ubuf__ float *)biasTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> vregX;
            RegTensor<float> vregMax;
            RegTensor<float> vregMaxBcast;
            RegTensor<float> vregExp;
            RegTensor<float> vregSum;
            RegTensor<float> vregSumBcast;
            RegTensor<float> vregResult;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);

            MicroAPI::LoadAlign(vregX, xRowAddr);
            Reg::Reduce<Reg::ReduceType::MAX>(vregMax, vregX, preg0);
            MicroAPI::Duplicate(vregMaxBcast, vregMax, preg0);
            MicroAPI::Sub(vregExp, vregX, vregMaxBcast, preg0);
            MicroAPI::Exp(vregExp, vregExp, preg0);
            Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregExp, preg0);
            MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
            MicroAPI::Div(vregResult, vregExp, vregSumBcast, preg0);
            MicroAPI::StoreAlign(xNormAddr, vregResult, preg0);
        }
        if (hasBias_) {
            __VEC_SCOPE__
            {
                RegTensor<float> vregResult;
                RegTensor<float> vregBias;
                RegTensor<float> vregBiasResult;
                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);
                MicroAPI::LoadAlign(vregResult, xNormAddr);
                MicroAPI::LoadAlign(vregBias, biasAddr);
                MicroAPI::Add(vregBiasResult, vregResult, vregBias, preg0);
                MicroAPI::StoreAlign(xNormWithBiasAddr, vregBiasResult, preg0);
            }
        } else {
            __VEC_SCOPE__
            {
                RegTensor<float> vregResult;
                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);
                MicroAPI::LoadAlign(vregResult, xNormAddr);
                MicroAPI::StoreAlign(xNormWithBiasAddr, vregResult, preg0);
            }
        }
        if (duplicateNum > 0) {
            __VEC_SCOPE__
            {
                RegTensor<float> vregPad;
                MicroAPI::UnalignRegForStore u0;
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                uint32_t padCount = static_cast<uint32_t>(expertCountAlign_ - expertCount_);
                auto padAddr = xNormWithBiasAddr + expertCount_;
                MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(padAddr, vregPad, u0,
                                                                                           padCount);
                MicroAPI::StoreUnAlignPost(padAddr, u0, 0);
            }
        }
    } else {
        Exp(xNormTensor, xRow, expertCountAlign_);
        PipeBarrier<PIPE_V>();
        Adds(xNormTensor, xNormTensor, static_cast<float>(1), expertCountAlign_);
        PipeBarrier<PIPE_V>();
        Ln(xNormTensor, xNormTensor, expertCountAlign_);
        PipeBarrier<PIPE_V>();
        Sqrt(xNormTensor, xNormTensor, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }

    if (normType_ != 0) {
        if (hasBias_) {
            Add(xNormWithBiasTensor, xNormTensor, biasTensor, expertCountAlign_);
        } else {
            DataCopy(xNormWithBiasTensor, xNormTensor, expertCountAlign_);
        }
        PipeBarrier<PIPE_V>();

        if (duplicateNum > 0) {
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
            uint64_t mask[2] = {mask0, 0};
            Duplicate(xNormWithBiasTensor.ReinterpretCast<int32_t>()[duplicateIndex], MIN_FP32, mask, 1, 1, 1);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::CopyOutXNormTop1(int64_t globalRow)
{
    LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    DataCopy(outOutTensor, xNormTensor, expertCountAlign_);
    outOutQueue_.EnQue<float>(outOutTensor);
    outOutTensor = outOutQueue_.DeQue<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(outGm_[globalRow * expertCount_], outOutTensor, dataCopyParams);
    outOutQueue_.FreeTensor(outOutTensor);
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::SelectTop1AndScoreNoNorm(LocalTensor<T> yOutTensor,
                                                                                      LocalTensor<int32_t> expertIdxOut,
                                                                                      int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> sortedScore = sortedBuf_.Get<float>();

    uint32_t expertCountU32 = static_cast<uint32_t>(expertCount_);
    __ubuf__ float *xNormWithBiasAddr = (__ubuf__ float *)xNormWithBiasTensor.GetPhyAddr();
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
    __ubuf__ float *sortedAddr = (__ubuf__ float *)sortedScore.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr() + rowInBatch * kStride_;
    __ubuf__ uint32_t *expertIdxAddr = (__ubuf__ uint32_t *)expertIdxOut.GetPhyAddr() + rowInBatch * kStride_;

    __VEC_SCOPE__
    {
        RegTensor<float> valueAndIndexReg;
        MicroAPI::MaskReg maskForVL2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL2>();
        MicroAPI::MaskReg maskForExpertCount = MicroAPI::UpdateMask<uint32_t>(expertCountU32);

        MicroAPI::LoadAlign(valueAndIndexReg, xNormWithBiasAddr);
        Reg::Reduce<Reg::ReduceType::MAX>(valueAndIndexReg, valueAndIndexReg, maskForExpertCount);
        MicroAPI::StoreAlign(sortedAddr, valueAndIndexReg, maskForVL2);
    }

    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValueDiscard;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGatheredScore;
        RegTensor<float> vregOutput;

        uint32_t kU32 = static_cast<uint32_t>(k_);
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(kU32);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValueDiscard, vregExpertIdx,
                                                                          (__ubuf__ uint32_t *)sortedAddr);
        MicroAPI::Gather(vregGatheredScore, xNormAddr, vregExpertIdx, preg0);
        MicroAPI::Muls(vregOutput, vregGatheredScore, routedScalingFactor_, preg0);

        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg0, 0);
        MicroAPI::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::SelectTop1AndScoreWithNorm(
    LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut, int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> sortedScore = sortedBuf_.Get<float>();

    uint32_t expertCountU32 = static_cast<uint32_t>(expertCount_);
    __ubuf__ float *xNormWithBiasAddr = (__ubuf__ float *)xNormWithBiasTensor.GetPhyAddr();
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
    __ubuf__ float *sortedAddr = (__ubuf__ float *)sortedScore.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr() + rowInBatch * kStride_;
    __ubuf__ uint32_t *expertIdxAddr = (__ubuf__ uint32_t *)expertIdxOut.GetPhyAddr() + rowInBatch * kStride_;

    __VEC_SCOPE__
    {
        RegTensor<float> valueAndIndexReg;
        MicroAPI::MaskReg maskForVL2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL2>();
        MicroAPI::MaskReg maskForExpertCount = MicroAPI::UpdateMask<uint32_t>(expertCountU32);

        MicroAPI::LoadAlign(valueAndIndexReg, xNormWithBiasAddr);
        Reg::Reduce<Reg::ReduceType::MAX>(valueAndIndexReg, valueAndIndexReg, maskForExpertCount);
        MicroAPI::StoreAlign(sortedAddr, valueAndIndexReg, maskForVL2);
    }

    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValueDiscard;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGatheredScore;
        RegTensor<float> vregScoreSumEps;
        RegTensor<float> vregRenormScore;
        RegTensor<float> vregOutput;

        uint32_t kU32 = static_cast<uint32_t>(k_);
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(kU32);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValueDiscard, vregExpertIdx,
                                                                          (__ubuf__ uint32_t *)sortedAddr);
        MicroAPI::Gather(vregGatheredScore, xNormAddr, vregExpertIdx, preg0);
        Reg::Reduce<Reg::ReduceType::SUM>(vregScoreSumEps, vregGatheredScore, preg0);
        MicroAPI::Adds(vregScoreSumEps, vregScoreSumEps, eps_, preg0);
        MicroAPI::Duplicate(vregRenormScore, vregScoreSumEps, preg0);
        MicroAPI::Div(vregRenormScore, vregGatheredScore, vregRenormScore, preg0);
        MicroAPI::Muls(vregOutput, vregRenormScore, routedScalingFactor_, preg0);

        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg0, 0);
        MicroAPI::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::SelectTop1AndScore(LocalTensor<T> yOutTensor,
                                                                                LocalTensor<int32_t> expertIdxOut,
                                                                                int64_t rowInBatch)
{
    if (noNorm_) {
        SelectTop1AndScoreNoNorm(yOutTensor, expertIdxOut, rowInBatch);
    } else {
        SelectTop1AndScoreWithNorm(yOutTensor, expertIdxOut, rowInBatch);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::CopyOutBatch(int64_t globalBaseRow, int64_t rowsInBatch)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams yCopyParams{static_cast<uint16_t>(rowsInBatch), static_cast<uint32_t>(k_ * sizeof(T)),
                                  static_cast<uint32_t>((kStride_ - k_) * sizeof(T) / BLOCK_BYTES), 0, 0};
    DataCopyPad(yGm_[globalBaseRow * k_], yOutTensor, yCopyParams);
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(rowsInBatch), static_cast<uint32_t>(k_ * sizeof(int32_t)),
                                    static_cast<uint32_t>((kStride_ - k_) * sizeof(int32_t) / BLOCK_BYTES), 0, 0};
    DataCopyPad(expertIdxGm_[globalBaseRow * k_], expertIdxOut, idxCopyParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOut);
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ProcessSingleExpertSigmoid(int64_t batchSize,
                                                                                        int64_t numBatches)
{
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        if (batch + 1 < numBatches) {
            int64_t nextRows = Min(batchSize, curCoreRowCount_ - (batch + 1) * batchSize);
            LocalTensor<T> xNextTensor = xInQueue_.AllocTensor<T>();
            DataCopyExtParams nextCopyParams{1, static_cast<uint32_t>(nextRows * sizeof(T)), 0, 0, 0};
            DataCopyPad(xNextTensor, xGm_[(batch + 1) * batchSize], nextCopyParams, dataCopyPadParams);
            xInQueue_.EnQue(xNextTensor);
        }

        LocalTensor<T> xInTensor = xInQueue_.DeQue<T>();
        LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
        LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, WG_VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<float> vregOne;
            RegTensor<float> vregTmp1;
            RegTensor<float> vregTmp2;
            RegTensor<float> vregTmp3;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
            MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::Muls(vregTmp1, vregIn, static_cast<float>(-1), preg0);
                MicroAPI::Exp(vregTmp2, vregTmp1, preg0);
                MicroAPI::Adds(vregTmp3, vregTmp2, static_cast<float>(1), preg0);
                MicroAPI::Div<float, &WG_DIV_MODE>(vregNorm, vregOne, vregTmp3, preg0);
                RegTensor<float> vregSum;
                Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregNorm, preg0);
                MicroAPI::Adds(vregSum, vregSum, eps_, preg0);
                RegTensor<float> vregSumBcast;
                MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
                MicroAPI::Div(vregOut, vregNorm, vregSumBcast, preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * WG_VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * WG_VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        yOutQueue_.EnQue(yOutTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
        yOutTensor = yOutQueue_.DeQue<T>();
        expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[gmOffset], yOutTensor, yCopyParams);
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expertIdxGm_[gmOffset], expertIdxOut, idxCopyParams);
        yOutQueue_.FreeTensor(yOutTensor);
        expertIdxOutQueue_.FreeTensor(expertIdxOut);

        if (outFlag_) {
            outOutQueue_.EnQue<float>(outOutTensor);
            outOutTensor = outOutQueue_.DeQue<float>();
            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(float)), 0, 0, 0};
            DataCopyPad(outGm_[gmOffset], outOutTensor, outCopyParams);
        }
        outOutQueue_.FreeTensor(outOutTensor);
        xInQueue_.FreeTensor(xInTensor);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ProcessSingleExpertSoftMaxRenorm(int64_t batchSize,
                                                                                              int64_t numBatches)
{
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        if (batch + 1 < numBatches) {
            int64_t nextRows = Min(batchSize, curCoreRowCount_ - (batch + 1) * batchSize);
            LocalTensor<T> xNextTensor = xInQueue_.AllocTensor<T>();
            DataCopyExtParams nextCopyParams{1, static_cast<uint32_t>(nextRows * sizeof(T)), 0, 0, 0};
            DataCopyPad(xNextTensor, xGm_[(batch + 1) * batchSize], nextCopyParams, dataCopyPadParams);
            xInQueue_.EnQue(xNextTensor);
        }

        LocalTensor<T> xInTensor = xInQueue_.DeQue<T>();
        LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
        LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, WG_VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::Duplicate(vregNorm, static_cast<float>(1), preg0);
                RegTensor<float> vregSum;
                Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregNorm, preg0);
                MicroAPI::Adds(vregSum, vregSum, eps_, preg0);
                RegTensor<float> vregSumBcast;
                MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
                MicroAPI::Div(vregOut, vregNorm, vregSumBcast, preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * WG_VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * WG_VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        yOutQueue_.EnQue(yOutTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
        yOutTensor = yOutQueue_.DeQue<T>();
        expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[gmOffset], yOutTensor, yCopyParams);
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expertIdxGm_[gmOffset], expertIdxOut, idxCopyParams);
        yOutQueue_.FreeTensor(yOutTensor);
        expertIdxOutQueue_.FreeTensor(expertIdxOut);

        if (outFlag_) {
            outOutQueue_.EnQue<float>(outOutTensor);
            outOutTensor = outOutQueue_.DeQue<float>();
            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(float)), 0, 0, 0};
            DataCopyPad(outGm_[gmOffset], outOutTensor, outCopyParams);
        }
        outOutQueue_.FreeTensor(outOutTensor);
        xInQueue_.FreeTensor(xInTensor);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ProcessSingleExpertSoftMaxNoRenorm(int64_t batchSize,
                                                                                                int64_t numBatches)
{
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        if (batch + 1 < numBatches) {
            int64_t nextRows = Min(batchSize, curCoreRowCount_ - (batch + 1) * batchSize);
            LocalTensor<T> xNextTensor = xInQueue_.AllocTensor<T>();
            DataCopyExtParams nextCopyParams{1, static_cast<uint32_t>(nextRows * sizeof(T)), 0, 0, 0};
            DataCopyPad(xNextTensor, xGm_[(batch + 1) * batchSize], nextCopyParams, dataCopyPadParams);
            xInQueue_.EnQue(xNextTensor);
        }

        LocalTensor<T> xInTensor = xInQueue_.DeQue<T>();
        LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
        LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, WG_VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::Duplicate(vregNorm, static_cast<float>(1), preg0);
                MicroAPI::Muls(vregOut, vregNorm, static_cast<float>(1), preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * WG_VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * WG_VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        yOutQueue_.EnQue(yOutTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
        yOutTensor = yOutQueue_.DeQue<T>();
        expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[gmOffset], yOutTensor, yCopyParams);
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expertIdxGm_[gmOffset], expertIdxOut, idxCopyParams);
        yOutQueue_.FreeTensor(yOutTensor);
        expertIdxOutQueue_.FreeTensor(expertIdxOut);

        if (outFlag_) {
            outOutQueue_.EnQue<float>(outOutTensor);
            outOutTensor = outOutQueue_.DeQue<float>();
            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(float)), 0, 0, 0};
            DataCopyPad(outGm_[gmOffset], outOutTensor, outCopyParams);
        }
        outOutQueue_.FreeTensor(outOutTensor);
        xInQueue_.FreeTensor(xInTensor);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ProcessSingleExpertSoftplus(int64_t batchSize,
                                                                                         int64_t numBatches)
{
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        if (batch + 1 < numBatches) {
            int64_t nextRows = Min(batchSize, curCoreRowCount_ - (batch + 1) * batchSize);
            LocalTensor<T> xNextTensor = xInQueue_.AllocTensor<T>();
            DataCopyExtParams nextCopyParams{1, static_cast<uint32_t>(nextRows * sizeof(T)), 0, 0, 0};
            DataCopyPad(xNextTensor, xGm_[(batch + 1) * batchSize], nextCopyParams, dataCopyPadParams);
            xInQueue_.EnQue(xNextTensor);
        }

        LocalTensor<T> xInTensor = xInQueue_.DeQue<T>();
        LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
        LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, WG_VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<float> vregTmp1;
            RegTensor<float> vregTmp2;
            RegTensor<float> vregTmp3;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::Exp(vregTmp1, vregIn, preg0);
                MicroAPI::Adds(vregTmp2, vregTmp1, static_cast<float>(1), preg0);
                MicroAPI::Ln(vregTmp3, vregTmp2, preg0);
                MicroAPI::Sqrt(vregNorm, vregTmp3, preg0);
                MicroAPI::Muls(vregOut, vregNorm, static_cast<float>(1), preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * WG_VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * WG_VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        yOutQueue_.EnQue(yOutTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
        yOutTensor = yOutQueue_.DeQue<T>();
        expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[gmOffset], yOutTensor, yCopyParams);
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expertIdxGm_[gmOffset], expertIdxOut, idxCopyParams);
        yOutQueue_.FreeTensor(yOutTensor);
        expertIdxOutQueue_.FreeTensor(expertIdxOut);

        if (outFlag_) {
            outOutQueue_.EnQue<float>(outOutTensor);
            outOutTensor = outOutQueue_.DeQue<float>();
            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(float)), 0, 0, 0};
            DataCopyPad(outGm_[gmOffset], outOutTensor, outCopyParams);
        }
        outOutQueue_.FreeTensor(outOutTensor);
        xInQueue_.FreeTensor(xInTensor);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ProcessSingleExpert()
{
    int64_t batchSize = Min(batchRows_, curCoreRowCount_);
    int64_t numBatches = CeilDiv(curCoreRowCount_, batchSize);

    int64_t firstRows = Min(batchSize, curCoreRowCount_);
    LocalTensor<T> xInTensor = xInQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(firstRows * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xInTensor, xGm_[0], dataCopyParams, dataCopyPadParams);
    xInQueue_.EnQue(xInTensor);

    if (normType_ == 1) {
        ProcessSingleExpertSigmoid(batchSize, numBatches);
    } else if (normType_ == 0) {
        if (renorm_ == 0) {
            ProcessSingleExpertSoftMaxNoRenorm(batchSize, numBatches);
        } else {
            ProcessSingleExpertSoftMaxRenorm(batchSize, numBatches);
        }
    } else {
        ProcessSingleExpertSoftplus(batchSize, numBatches);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ComputeXForBatchRow(LocalTensor<float> xInTensor,
                                                                                         int64_t rowOffset)
{
    LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
    LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
    indexTensor = indexBuffer_.Get<uint32_t>();

    if constexpr (!IsSameType<T, float>::value) {
        LocalTensor<T> rawTensor = xInTensor[batchRows_ * expertCountAlign_].ReinterpretCast<T>();
        Cast(xInTensor[rowOffset], rawTensor[rowOffset], RoundMode::CAST_NONE, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> xRow = xInTensor[rowOffset];

    int64_t duplicateNum = perGroupExpertCount_ % ONE_REPEAT_SORT_NUM;
    int64_t duplicateIndex = perGroupExpertCount_ - duplicateNum;
    if (perGroupExpertCount_ != perGroupExpertCountAlign_ && duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(xRow.ReinterpretCast<int32_t>()[duplicateIndex], MIN_FP32, mask, groupCount_, 1,
                  (perGroupExpertCountAlign_ * sizeof(float)) / BLOCK_BYTES);
    }

    LocalTensor<uint8_t> calcNormTmpTensor = calcTmpBuf_.Get<uint8_t>();
    if (normType_ == 1) {
        Sigmoid(xSigmoidTensor, xRow, calcNormTmpTensor, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    } else if (normType_ == 0) {
        uint32_t size = static_cast<uint32_t>(expertCountAlign_);
        __ubuf__ float *xRowAddr = (__ubuf__ float *)xRow.GetPhyAddr();
        __ubuf__ float *xSigmoidAddr = (__ubuf__ float *)xSigmoidTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> vregX;
            RegTensor<float> vregMax;
            RegTensor<float> vregMaxBcast;
            RegTensor<float> vregExp;
            RegTensor<float> vregSum;
            RegTensor<float> vregSumBcast;
            RegTensor<float> vregResult;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);

            MicroAPI::LoadAlign(vregX, xRowAddr);
            Reg::Reduce<Reg::ReduceType::MAX>(vregMax, vregX, preg0);
            MicroAPI::Duplicate(vregMaxBcast, vregMax, preg0);
            MicroAPI::Sub(vregExp, vregX, vregMaxBcast, preg0);
            MicroAPI::Exp(vregExp, vregExp, preg0);
            Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregExp, preg0);
            MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
            MicroAPI::Div(vregResult, vregExp, vregSumBcast, preg0);
            MicroAPI::StoreAlign(xSigmoidAddr, vregResult, preg0);
        }
    } else {
        Exp(xSigmoidTensor, xRow, expertCountAlign_);
        PipeBarrier<PIPE_V>();
        Adds(xSigmoidTensor, xSigmoidTensor, static_cast<float>(1), expertCountAlign_);
        PipeBarrier<PIPE_V>();
        Ln(xSigmoidTensor, xSigmoidTensor, expertCountAlign_);
        PipeBarrier<PIPE_V>();
        Sqrt(xSigmoidTensor, xSigmoidTensor, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }

    __ubuf__ float *xSigmoidAddr = (__ubuf__ float *)xSigmoidTensor.GetPhyAddr();
    __ubuf__ float *xBiasAddr = (__ubuf__ float *)xBiasTensor.GetPhyAddr();
    __ubuf__ float *biasAddr = (__ubuf__ float *)biasBuf_.Get<float>().GetPhyAddr();
    __ubuf__ int32_t *indexOutAddr = (__ubuf__ int32_t *)indexTensor.GetPhyAddr();
    uint32_t size = static_cast<uint32_t>(expertCountAlign_);
    uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, WG_VL_FLOAT_SIZE));

    __VEC_SCOPE__
    {
        RegTensor<float> vregSigmoid;
        RegTensor<float> vregBias;
        RegTensor<float> vregBiasResult;
        RegTensor<int32_t> vregIndex;
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<float>(size);
            MicroAPI::LoadAlign(vregSigmoid, xSigmoidAddr + i * WG_VL_FLOAT_SIZE);
            if (hasBias_) {
                MicroAPI::LoadAlign(vregBias, biasAddr + i * WG_VL_FLOAT_SIZE);
                MicroAPI::Add(vregBiasResult, vregSigmoid, vregBias, preg0);
            }
            MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * WG_VL_FLOAT_SIZE));
            if (hasBias_) {
                MicroAPI::StoreAlign(xBiasAddr + i * WG_VL_FLOAT_SIZE, vregBiasResult, preg0);
            } else {
                MicroAPI::StoreAlign(xBiasAddr + i * WG_VL_FLOAT_SIZE, vregSigmoid, preg0);
            }
            MicroAPI::StoreAlign(indexOutAddr + i * WG_VL_FLOAT_SIZE, vregIndex, preg0);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
        if (perGroupExpertCount_ != perGroupExpertCountAlign_ && duplicateNum > 0) {
            PadWithMinFp32(xBiasAddr, groupCount_, perGroupExpertCount_, perGroupExpertCountAlign_);
        }
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds,
                                                                          GM_ADDR tid2eid, GM_ADDR y, GM_ADDR expertIdx,
                                                                          GM_ADDR out, GM_ADDR workspace,
                                                                          const MoeGatingTopKRegbaseTilingData *tilingData,
                                                                          TPipe *tPipe)
{
    InitBase(x, bias, inputIds, tid2eid, y, expertIdx, out, tilingData, tPipe);

    batchRows_ = tilingData->batchRows > 0 ? tilingData->batchRows : DEFAULT_BATCH_ROWS;
    kStride_ = Ceil(k_, static_cast<int64_t>(BLOCK_BYTES) / static_cast<int64_t>(sizeof(T))) *
               static_cast<int64_t>(BLOCK_BYTES) / static_cast<int64_t>(sizeof(T));

    if (expertCount_ == 1) {
        int64_t singleExpertBatch = tilingData->batchRows > 0 ? tilingData->batchRows : SINGLE_EXPERT_FALLBACK_BATCH;
        batchRows_ = singleExpertBatch;
        pipe_->InitBuffer(xInQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(T));
        pipe_->InitBuffer(yOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(T));
        pipe_->InitBuffer(expertIdxOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(int32_t));
        pipe_->InitBuffer(outOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(float));
        return;
    }

    int64_t batchExpertAlign = batchRows_ * expertCountAlign_;
    pipe_->InitBuffer(xInQueue_, WG_CONSTANT_TWO, batchExpertAlign * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, 1, batchRows_ * kStride_ * sizeof(T));
    pipe_->InitBuffer(expertIdxOutQueue_, 1, batchRows_ * kStride_ * sizeof(int32_t));
    pipe_->InitBuffer(outOutQueue_, 1, expertCountAlign_ * sizeof(float));

    pipe_->InitBuffer(xNormBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(xNormWithBiasBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(sortedBuf_, expertCountAlign_ * sizeof(float) * WG_CONSTANT_TWO);
    pipe_->InitBuffer(calcTmpBuf_, expertCountAlign_ * sizeof(float) * WG_CONSTANT_TEN);
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::ComputeOneRow(
    LocalTensor<float> xInTensor, int64_t r, int64_t localRow, LocalTensor<T> yOutTensor,
    LocalTensor<int32_t> expertIdxOut)
{
    int64_t globalRow = localRow + tilingData_->perCoreRowCount * blockIdx_;

    if (hashFlag_) {
        ComputeXForBatchRow(xInTensor, r * expertCountAlign_);
        if (outFlag_) {
            CopyOutXNorm(localRow);
        }
        HashComputeCore(yOutTensor[r * kStride_], expertIdxOut[r * kStride_], globalRow);
    } else if (k_ == 1) {
        ComputeNormForTop1(xInTensor, r);
        if (outFlag_) {
            CopyOutXNormTop1(localRow);
        }
        SelectTop1AndScore(yOutTensor, expertIdxOut, r);
    } else {
        ComputeXForBatchRow(xInTensor, r * expertCountAlign_);
        if (outFlag_) {
            CopyOutXNorm(localRow);
        }
        TopKComputeCore(yOutTensor[r * kStride_], expertIdxOut[r * kStride_]);
    }
}

template <typename T, typename U1, typename U2>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T, U1, U2>::Process()
{
    if (expertCount_ == 1) {
        ProcessSingleExpert();
        return;
    }

    CopyInBias();

    int64_t numBatches = CeilDiv(curCoreRowCount_, batchRows_);
    int64_t firstRows = Min(batchRows_, curCoreRowCount_);
    CopyInXBatch(0, firstRows);

    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchRows_, curCoreRowCount_ - batch * batchRows_);

        if (batch + 1 < numBatches) {
            int64_t nextRows = Min(batchRows_, curCoreRowCount_ - (batch + 1) * batchRows_);
            CopyInXBatch(batch + 1, nextRows);
        }

        LocalTensor<float> xInTensor = xInQueue_.DeQue<float>();
        LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();

        for (int64_t r = 0; r < rowsInBatch; r++) {
            int64_t localRow = batch * batchRows_ + r;
            ComputeOneRow(xInTensor, r, localRow, yOutTensor, expertIdxOut);
        }

        yOutQueue_.EnQue<T>(yOutTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
        CopyOutBatch(batch * batchRows_, rowsInBatch);
        xInQueue_.FreeTensor(xInTensor);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H
