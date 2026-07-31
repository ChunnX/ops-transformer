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
 * \file moe_gating_top_k_without_group_regbase.h
 * \brief Arch35 register-based kernel for MoE Gating TopK without group.
 *        Optimizations:
 *        1. Multi-row batch copy-in to amortize DMA overhead.
 *        2. k==1: ReduceMax replaces Sort for O(n) top-1 selection.
 *        3. expertCount==1: skip 32-byte alignment, direct scalar/vector compute.
 */
#ifndef MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H
#define MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H

#include <cmath>
#include "common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "basic_api/kernel_operator_utils_intf.h"

namespace MoeGatingTopK {
using namespace AscendC;
using MicroAPI::RegTensor;

constexpr int32_t WG_CONSTANT_TWO = 2;
constexpr int32_t WG_CONSTANT_TEN = 10;
constexpr uint32_t WG_VL_FLOAT_SIZE = VECTOR_REG_WIDTH / sizeof(float);
constexpr int64_t DEFAULT_BATCH_ROWS = 4;
constexpr int64_t SINGLE_EXPERT_FALLBACK_BATCH = 512;
constexpr MicroAPI::DivSpecificMode WG_DIV_MODE = {MicroAPI::MaskMergeMode::ZEROING, true};

template <typename T>
class MoeGatingTopKWithoutGroupRegbase {
public:
    __aicore__ inline MoeGatingTopKWithoutGroupRegbase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out,
                                GM_ADDR workspace, const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBias();
    __aicore__ inline void CopyInXBatch(int64_t batchIdx, int64_t rowsInBatch);
    __aicore__ inline void ComputeNorm(LocalTensor<float> xInTensor, int64_t rowInBatch);
    __aicore__ inline void CopyOutXNorm(int64_t globalRow);
    __aicore__ inline void SelectTop1AndScore(LocalTensor<T> yOutTensor,
                                              LocalTensor<int32_t> expertIdxOut, int64_t rowInBatch);
    __aicore__ inline void CopyOutBatch(int64_t globalBaseRow, int64_t rowsInBatch);

    __aicore__ inline void ProcessSingleExpert();

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 2> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> outOutQueue_;

    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> xNormBuf_;
    TBuf<TPosition::VECCALC> xNormWithBiasBuf_;
    TBuf<TPosition::VECCALC> sortedBuf_;
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
    int64_t kStride_;
    int64_t batchRows_;
    int64_t normType_;
    int64_t renorm_;
    bool hasBias_;
    bool outFlag_;
    float routedScalingFactor_;
    float eps_;

    const MoeGatingTopKRegbaseTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyInBias()
{
    if (!hasBias_) {
        return;
    }
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(biasTensor, biasGm_, dataCopyParams, dataCopyPadParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        DataCopyPad(biasTensor[expertCountAlign_].ReinterpretCast<T>(), biasGm_, dataCopyParams, dataCopyPadParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(biasTensor, biasTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyInXBatch(int64_t batchIdx, int64_t rowsInBatch)
{
    LocalTensor<float> xInTensor = xInQueue_.AllocTensor<float>();

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = static_cast<uint16_t>(rowsInBatch);
    dataCopyParams.blockLen = static_cast<uint32_t>(expertCount_ * sizeof(T));
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = static_cast<uint32_t>((expertCountAlign_ - expertCount_) * sizeof(T) / BLOCK_BYTES);
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

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ComputeNorm(LocalTensor<float> xInTensor, int64_t rowInBatch)
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
        __local_mem__ float *xRowAddr = (__local_mem__ float *)xRow.GetPhyAddr();
        __local_mem__ float *xNormAddr = (__local_mem__ float *)xNormTensor.GetPhyAddr();
        __local_mem__ float *xNormWithBiasAddr = (__local_mem__ float *)xNormWithBiasTensor.GetPhyAddr();
        __local_mem__ float *biasAddr = hasBias_ ? (__local_mem__ float *)biasTensor.GetPhyAddr() : nullptr;

        __VEC_SCOPE__
        {
            RegTensor<float> vregX;
            RegTensor<float> vregMax;
            RegTensor<float> vregMaxBcast;
            RegTensor<float> vregExp;
            RegTensor<float> vregSum;
            RegTensor<float> vregSumBcast;
            RegTensor<float> vregResult;
            RegTensor<float> vregBias;
            RegTensor<float> vregBiasResult;
            RegTensor<float> vregPad;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);

            MicroAPI::DataCopy(vregX, xRowAddr);
            MicroAPI::ReduceMax(vregMax, vregX, preg0);
            MicroAPI::Duplicate(vregMaxBcast, vregMax, preg0);
            MicroAPI::Sub(vregExp, vregX, vregMaxBcast, preg0);
            MicroAPI::Exp(vregExp, vregExp, preg0);
            MicroAPI::ReduceSum(vregSum, vregExp, preg0);
            MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
            MicroAPI::Div(vregResult, vregExp, vregSumBcast, preg0);
            MicroAPI::DataCopy(xNormAddr, vregResult, preg0);

            if (hasBias_) {
                MicroAPI::DataCopy(vregBias, biasAddr);
                MicroAPI::Add(vregBiasResult, vregResult, vregBias, preg0);
            } else {
                vregBiasResult = vregResult;
            }
            MicroAPI::DataCopy(xNormWithBiasAddr, vregBiasResult, preg0);

            if (duplicateNum > 0) {
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                MicroAPI::UnalignReg u0;
                uint32_t padCount = static_cast<uint32_t>(expertCountAlign_ - expertCount_);
                auto padAddr = xNormWithBiasAddr + expertCount_;
                MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(padAddr, vregPad, u0, padCount);
                MicroAPI::DataCopyUnAlignPost(padAddr, u0, 0);
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

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyOutXNorm(int64_t globalRow)
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

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::SelectTop1AndScore(LocalTensor<T> yOutTensor,
                                                                               LocalTensor<int32_t> expertIdxOut,
                                                                               int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> sortedScore = sortedBuf_.Get<float>();

    uint32_t expertCountU32 = static_cast<uint32_t>(expertCount_);
    __local_mem__ float *xNormWithBiasAddr = (__local_mem__ float *)xNormWithBiasTensor.GetPhyAddr();
    __local_mem__ float *xNormAddr = (__local_mem__ float *)xNormTensor.GetPhyAddr();
    __local_mem__ float *sortedAddr = (__local_mem__ float *)sortedScore.GetPhyAddr();
    __local_mem__ T *outputAddr = (__local_mem__ T *)yOutTensor.GetPhyAddr() + rowInBatch * kStride_;
    __local_mem__ uint32_t *expertIdxAddr =
        (__local_mem__ uint32_t *)expertIdxOut.GetPhyAddr() + rowInBatch * kStride_;

    __VEC_SCOPE__
    {
        RegTensor<float> valueAndIndexReg;
        MicroAPI::MaskReg maskForVL2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL2>();
        MicroAPI::MaskReg maskForExpertCount = MicroAPI::UpdateMask<uint32_t>(expertCountU32);

        MicroAPI::DataCopy(valueAndIndexReg, xNormWithBiasAddr);
        MicroAPI::ReduceMax(valueAndIndexReg, valueAndIndexReg, maskForExpertCount);
        MicroAPI::DataCopy(sortedAddr, valueAndIndexReg, maskForVL2);
    }

    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vreg0;
        RegTensor<uint32_t> vreg1;
        RegTensor<float> vreg2;
        RegTensor<float> vreg3;
        RegTensor<float> vreg4;
        RegTensor<float> vregOutput;

        uint32_t kU32 = static_cast<uint32_t>(k_);
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(kU32);

        MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                           (__local_mem__ uint32_t *)sortedAddr);
        MicroAPI::DataCopyGather(vreg2, xNormAddr, vreg1, preg0);

        bool needRenorm = !(normType_ == 0 && renorm_ == 0);
        if (needRenorm) {
            MicroAPI::ReduceSum(vreg3, vreg2, preg0);
            MicroAPI::Adds(vreg3, vreg3, eps_, preg0);
            MicroAPI::Duplicate(vreg4, vreg3, preg0);
            MicroAPI::Div(vreg4, vreg2, vreg4, preg0);
            MicroAPI::Muls(vregOutput, vreg4, routedScalingFactor_, preg0);
        } else {
            MicroAPI::Muls(vregOutput, vreg2, routedScalingFactor_, preg0);
        }

        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg0, 0);
        MicroAPI::DataCopy(expertIdxAddr, vreg1, preg0);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyOutBatch(int64_t globalBaseRow, int64_t rowsInBatch)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams yCopyParams{
        static_cast<uint16_t>(rowsInBatch),
        static_cast<uint32_t>(k_ * sizeof(T)),
        static_cast<uint32_t>((kStride_ - k_) * sizeof(T) / BLOCK_BYTES),
        0, 0};
    DataCopyPad(yGm_[globalBaseRow * k_], yOutTensor, yCopyParams);
    DataCopyExtParams idxCopyParams{
        static_cast<uint16_t>(rowsInBatch),
        static_cast<uint32_t>(k_ * sizeof(int32_t)),
        static_cast<uint32_t>((kStride_ - k_) * sizeof(int32_t) / BLOCK_BYTES),
        0, 0};
    DataCopyPad(expertIdxGm_[globalBaseRow * k_], expertIdxOut, idxCopyParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ProcessSingleExpert()
{
    int64_t batchSize = Min(batchRows_, curCoreRowCount_);
    int64_t numBatches = CeilDiv(curCoreRowCount_, batchSize);

    int64_t firstRows = Min(batchSize, curCoreRowCount_);
    LocalTensor<T> xInTensor = xInQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(firstRows * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xInTensor, xGm_[0], dataCopyParams, dataCopyPadParams);
    xInQueue_.EnQue(xInTensor);

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

        xInTensor = xInQueue_.DeQue<T>();

        LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
        LocalTensor<float> outOutTensor;
        if (outFlag_) {
            outOutTensor = outOutQueue_.AllocTensor<float>();
        }

        __local_mem__ T *inputAddr = (__local_mem__ T *)xInTensor.GetPhyAddr();
        __local_mem__ T *outputAddr = (__local_mem__ T *)yOutTensor.GetPhyAddr();
        __local_mem__ int32_t *expertIdxAddr = (__local_mem__ int32_t *)expertIdxOut.GetPhyAddr();
        __local_mem__ float *outAddr = nullptr;
        if (outFlag_) {
            outAddr = (__local_mem__ float *)outOutTensor.GetPhyAddr();
        }

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

                if (normType_ == 1) {
                    MicroAPI::Muls(vregTmp1, vregIn, static_cast<float>(-1), preg0);
                    MicroAPI::Exp(vregTmp2, vregTmp1, preg0);
                    MicroAPI::Adds(vregTmp3, vregTmp2, static_cast<float>(1), preg0);
                    MicroAPI::Div<float, &WG_DIV_MODE>(vregNorm, vregOne, vregTmp3, preg0);
                } else if (normType_ == 0) {
                    MicroAPI::Duplicate(vregNorm, static_cast<float>(1), preg0);
                } else {
                    MicroAPI::Exp(vregTmp1, vregIn, preg0);
                    MicroAPI::Adds(vregTmp2, vregTmp1, static_cast<float>(1), preg0);
                    MicroAPI::Ln(vregTmp3, vregTmp2, preg0);
                    MicroAPI::Sqrt(vregNorm, vregTmp3, preg0);
                }

                bool needRenorm = (normType_ == 1) || (normType_ == 0 && renorm_ == 1);
                if (needRenorm) {
                    RegTensor<float> vregSum;
                    MicroAPI::ReduceSum(vregSum, vregNorm, preg0);
                    MicroAPI::Adds(vregSum, vregSum, eps_, preg0);
                    RegTensor<float> vregSumBcast;
                    MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
                    MicroAPI::Div(vregOut, vregNorm, vregSumBcast, preg0);
                } else {
                    MicroAPI::Muls(vregOut, vregNorm, static_cast<float>(1), preg0);
                }
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);

                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);

                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * WG_VL_FLOAT_SIZE);
                MicroAPI::DataCopy(expertIdxAddr + i * WG_VL_FLOAT_SIZE, vregZeroIdx, preg0);

                if (outFlag_) {
                    MicroAPI::DataCopy(outAddr + i * WG_VL_FLOAT_SIZE, vregNorm, preg0);
                }
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
            outOutQueue_.FreeTensor(outOutTensor);
        }

        xInQueue_.FreeTensor(xInTensor);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::Init(
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
    hasBias_ = (tilingData_->addBias == 1) && (bias != nullptr);
    k_ = tilingData_->k;
    normType_ = tilingData_->normType;
    renorm_ = tilingData_->renorm;
    outFlag_ = tilingData_->outFlag == 1;
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;
    batchRows_ = tilingData_->batchRows > 0 ? tilingData_->batchRows : DEFAULT_BATCH_ROWS;

    expertCountAlign_ = Ceil(expertCount_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    kStride_ = Ceil(k_, static_cast<int64_t>(BLOCK_BYTES) / static_cast<int64_t>(sizeof(T))) *
               static_cast<int64_t>(BLOCK_BYTES) / static_cast<int64_t>(sizeof(T));

    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    if (hasBias_) {
        biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    }
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ float *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    if (expertCount_ == 1) {
        int64_t singleExpertBatch = tilingData_->batchRows > 0 ? tilingData_->batchRows : SINGLE_EXPERT_FALLBACK_BATCH;
        batchRows_ = singleExpertBatch;
        pipe_->InitBuffer(xInQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(T));
        pipe_->InitBuffer(yOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(T));
        pipe_->InitBuffer(expertIdxOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(int32_t));
        pipe_->InitBuffer(outOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(float));
        return;
    }

    int64_t batchExpertAlign = batchRows_ * expertCountAlign_;
    pipe_->InitBuffer(xInQueue_, WG_CONSTANT_TWO, batchExpertAlign * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, 1, batchRows_ * kStride_ * sizeof(float));
    pipe_->InitBuffer(expertIdxOutQueue_, 1, batchRows_ * kStride_ * sizeof(int32_t));
    pipe_->InitBuffer(outOutQueue_, 1, expertCountAlign_ * sizeof(float));

    pipe_->InitBuffer(biasBuf_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(xNormBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(xNormWithBiasBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(sortedBuf_, expertCountAlign_ * sizeof(float) * WG_CONSTANT_TWO);
    pipe_->InitBuffer(calcTmpBuf_, expertCountAlign_ * sizeof(float) * WG_CONSTANT_TEN);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::Process()
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
            int64_t globalRow = batch * batchRows_ + r;
            ComputeNorm(xInTensor, r);
            if (outFlag_) {
                CopyOutXNorm(globalRow);
            }
            SelectTop1AndScore(yOutTensor, expertIdxOut, r);
        }
        yOutQueue_.EnQue<T>(yOutTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
        CopyOutBatch(batch * batchRows_, rowsInBatch);
        xInQueue_.FreeTensor(xInTensor);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H
