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
 * \file moe_v3_full_load_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_DYNAMIC_QUANT_H
#define MOE_V3_FULL_LOAD_DYNAMIC_QUANT_H

#include "moe_v3_full_load_base.h"
#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM = 1;

__aicore__ inline void CopySmallFullLoadQuantOutToBatch(LocalTensor<int8_t> &dstLocal, LocalTensor<int8_t> &srcLocal,
                                                        int64_t dstOffset, int64_t copyBytes)
{
    for (int64_t i = 0; i < copyBytes; i++) {
        dstLocal.SetValue(dstOffset + i, srcLocal.GetValue(i));
    }
}

template <typename T, typename QuantT = int8_t>
class MoeV3FullLoadDynamicQuant : public MoeV3FullLoadBase<T> {
public:
    __aicore__ inline MoeV3FullLoadDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    __aicore__ inline void ScatterOutXDynamicQuant();
    __aicore__ inline void GatherOutXDynamicQuant();

private:
    TQue<QuePosition::VECIN, 2> inputXInQueue_;
    TQue<QuePosition::VECIN, 1> smoothInQueue_;
    TQue<QuePosition::VECOUT, 1> calcQueue_;
    TQue<QuePosition::VECOUT, 2> outBatchQueue_;
    TQue<QuePosition::VECOUT, 2> scaleBatchQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> expandedScaleGm_;

    int64_t colsAlign_;
    int64_t colsAsInt8_;

    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF32ToI16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static MicroAPI::CastTrait castTraitI16ToF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T, typename QuantT>
__aicore__ inline void
MoeV3FullLoadDynamicQuant<T, QuantT>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                           GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                           GM_ADDR expandedScale, GM_ADDR workspace,
                                           const MoeInitRoutingV3Arch35TilingData *tilingData,
                                           TPipe *tPipe)
{
    MoeV3FullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);

    colsAlign_ = Align(this->cols_, sizeof(T));
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        colsAsInt8_ = Ceil(this->cols_, static_cast<int64_t>(2)) * sizeof(int8_t);
    } else {
        colsAsInt8_ = this->cols_ * sizeof(int8_t);
    }

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    if (this->isInputScale_) {
        quantSmoothGm_.SetGlobalBuffer((__gm__ float *)scale);
    }
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    if constexpr (IsSameType<T, float>::value) {
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            this->pipe_->InitBuffer(inputXInQueue_, 2, AlignBytes(this->cols_, sizeof(float)));
        } else {
            this->pipe_->InitBuffer(inputXInQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM,
                                    AlignBytes(this->cols_, sizeof(float)));
        }
    } else {
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            this->pipe_->InitBuffer(inputXInQueue_, 2, 2 * AlignBytes(this->cols_, sizeof(T)));
        } else {
            this->pipe_->InitBuffer(inputXInQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM,
                                    2 * AlignBytes(this->cols_, sizeof(T)));
        }
    }
    this->pipe_->InitBuffer(smoothInQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM, AlignBytes(this->cols_, sizeof(float)));
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        this->pipe_->InitBuffer(calcQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM, AlignBytes(this->cols_, sizeof(float)));
        this->pipe_->InitBuffer(outBatchQueue_, 2, AlignBytes(this->cols_, sizeof(float)));
        this->pipe_->InitBuffer(scaleBatchQueue_, 2, BLOCK_BYTES);
    }
    this->pipe_->InitBuffer(inputXOutQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM,
                            AlignBytes(colsAsInt8_, sizeof(int8_t)));
    this->pipe_->InitBuffer(scaleOutQueue_, FULLLOAD_DYNAMIC_QUANT_BUFFER_NUM, BLOCK_BYTES + BLOCK_BYTES);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::Compute(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();
    LocalTensor<float> quantTempLocal;

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleLocal.GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)outLocal.GetPhyAddr();
    __local_mem__ float *quantTempUbAddr;
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        quantTempLocal = calcQueue_.AllocTensor<float>();
        quantTempUbAddr = (__local_mem__ float *)quantTempLocal.GetPhyAddr();
    }
    __local_mem__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__local_mem__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + colsAlign_;
    }
    __local_mem__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__local_mem__ float *)smoothLocal.GetPhyAddr();
    }

    uint16_t repeatTimes = Ceil(this->cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t maskLength = static_cast<uint32_t>(this->cols_);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, smoothReg, scaleValueReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;

        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegVL8 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL8>();

        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                MicroAPI::DataCopy(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            }
            if constexpr (IS_INPUT_SCALE) {
                MicroAPI::DataCopy(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
            }
            if constexpr (!IsSameType<T, float>::value || IS_INPUT_SCALE) {
                MicroAPI::DataCopy(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop);
            }
            MicroAPI::Abs(inReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, inReg, maskRegAll);
        }
        MicroAPI::ReduceMax(scaleValueReg, scaleValueReg, maskRegAll);
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        } else {
            MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / 127.0f, maskRegVL1);
        }
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::DataCopy(scaleUbAddr, scaleValueReg, maskRegVL8);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::DataCopy(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
            if constexpr (IsSameType<QuantT, int4b_t>::value) {
                MicroAPI::DataCopy(quantTempUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop);
            } else {
                MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
                MicroAPI::Cast<int8_t, half, castTraitF16ToI8>(outRegI8, outRegF16, maskRegInLoop);
                MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                                outRegI8, maskRegInLoop);
            }
        }
    }

    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        auto outLocalInt4 = outLocal.template ReinterpretCast<int4b_t>();
        LocalTensor<int32_t> tempInt32 = quantTempLocal.template ReinterpretCast<int32_t>();
        LocalTensor<half> tempHalf = quantTempLocal.template ReinterpretCast<half>();

        PipeBarrier<PIPE_V>();
        Cast(tempInt32, quantTempLocal, RoundMode::CAST_RINT, this->cols_);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0));
        PipeBarrier<PIPE_V>();
        Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, this->cols_);
        PipeBarrier<PIPE_V>();
        Cast(outLocalInt4, tempHalf, RoundMode::CAST_TRUNC, this->cols_);
        calcQueue_.FreeTensor(quantTempLocal);
    }

    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::ScatterOutXDynamicQuant()
{
    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = Min(startRowIdx + this->coreIndicesElements_, this->actualExpertIdxNum_);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->cols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsAsInt8_), 0, 0, 0};

    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        if (colsAsInt8_ < BLOCK_BYTES && startRowIdx < endRowIdx && startRowIdx < this->activeNum_) {
            LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
            int64_t rowEnd = Min(endRowIdx, this->activeNum_);

            if (this->isInputScale_) {
                DataCopyPad(smoothLocal, quantSmoothGm_[0], smoothParams, {false, 0, 0, 0});
                smoothInQueue_.EnQue(smoothLocal);
                smoothLocal = smoothInQueue_.DeQue<float>();
            }

            int32_t srcIdx = sortedRowIdx.GetValue(startRowIdx);
            LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
            if constexpr (IsSameType<T, float>::value) {
                DataCopyPad(inLocal, xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
            } else {
                DataCopyPad(inLocal[colsAlign_], xGm_[srcIdx / this->k_ * this->cols_], copyInParams,
                            {false, 0, 0, 0});
            }
            inputXInQueue_.EnQue<T>(inLocal);

            LocalTensor<T> nextInLocal;
            bool hasNext = startRowIdx + 1 < rowEnd;
            if (hasNext) {
                int32_t nextSrcIdx = sortedRowIdx.GetValue(startRowIdx + 1);
                nextInLocal = inputXInQueue_.AllocTensor<T>();
                if constexpr (IsSameType<T, float>::value) {
                    DataCopyPad(nextInLocal, xGm_[nextSrcIdx / this->k_ * this->cols_], copyInParams,
                                {false, 0, 0, 0});
                } else {
                    DataCopyPad(nextInLocal[colsAlign_], xGm_[nextSrcIdx / this->k_ * this->cols_], copyInParams,
                                {false, 0, 0, 0});
                }
                inputXInQueue_.EnQue<T>(nextInLocal);
            }

            for (int64_t i = startRowIdx; i < rowEnd; i++) {
                if (this->isInputScale_) {
                    Compute<true>(smoothLocal);
                } else {
                    Compute<false>(smoothLocal);
                }
                inputXInQueue_.FreeTensor(inLocal);

                LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
                DataCopyPad(expandedScaleGm_[i], scaleLocal, {1, 4, 0, 0, 0});

                LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
                DataCopyPad(expandedXGm_[i * colsAsInt8_], outLocal, copyOutParams);

                LocalTensor<T> prefetchInLocal;
                bool hasPrefetch = i + 2 < rowEnd;
                if (hasPrefetch) {
                    int32_t prefetchSrcIdx = sortedRowIdx.GetValue(i + 2);
                    prefetchInLocal = inputXInQueue_.AllocTensor<T>();
                    if constexpr (IsSameType<T, float>::value) {
                        DataCopyPad(prefetchInLocal, xGm_[prefetchSrcIdx / this->k_ * this->cols_], copyInParams,
                                    {false, 0, 0, 0});
                    } else {
                        DataCopyPad(prefetchInLocal[colsAlign_], xGm_[prefetchSrcIdx / this->k_ * this->cols_],
                                    copyInParams, {false, 0, 0, 0});
                    }
                    inputXInQueue_.EnQue<T>(prefetchInLocal);
                }

                inputXOutQueue_.FreeTensor(outLocal);
                scaleOutQueue_.FreeTensor(scaleLocal);
                if (hasNext) {
                    inLocal = nextInLocal;
                }
                if (hasPrefetch) {
                    nextInLocal = prefetchInLocal;
                }
                hasNext = hasPrefetch;
            }

            smoothInQueue_.FreeTensor(smoothLocal);
            this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
            this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
            return;
        }
    }

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    LocalTensor<int8_t> outBatchLocal;
    int64_t outBatchRows = 1;
    int64_t outBatchCount = 0;
    int64_t outBatchStart = 0;
    LocalTensor<float> scaleBatchLocal;
    int64_t scaleBatchRows = BLOCK_BYTES / static_cast<int64_t>(sizeof(float));
    int64_t scaleBatchCount = 0;
    int64_t scaleBatchStart = 0;
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        if (colsAsInt8_ < BLOCK_BYTES) {
            outBatchLocal = outBatchQueue_.AllocTensor<int8_t>();
            scaleBatchLocal = scaleBatchQueue_.AllocTensor<float>();
            outBatchRows = Max(static_cast<int64_t>(1), AlignBytes(this->cols_, sizeof(float)) / colsAsInt8_);
        }
    }
    int32_t lastExpertIdx = -1;

    for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        int32_t srcIdx = sortedRowIdx.GetValue(i);
        int32_t expertIdx = curExpertId - this->expertStart_;
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[colsAlign_], xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        if (this->isInputScale_) {
            if constexpr (IsSameType<QuantT, int4b_t>::value) {
                if (lastExpertIdx == -1) {
                    DataCopyPad(smoothLocal, quantSmoothGm_[0], smoothParams, {false, 0, 0, 0});
                    smoothInQueue_.EnQue(smoothLocal);
                    smoothLocal = smoothInQueue_.DeQue<float>();
                    lastExpertIdx = 0;
                }
            } else {
                if (expertIdx != lastExpertIdx) {
                    int64_t smoothOffset = expertIdx * this->cols_;
                    DataCopyPad(smoothLocal, quantSmoothGm_[smoothOffset], smoothParams, {false, 0, 0, 0});
                    smoothInQueue_.EnQue(smoothLocal);
                    smoothLocal = smoothInQueue_.DeQue<float>();
                    lastExpertIdx = expertIdx;
                }
            }
        }

        if (this->isInputScale_) {
            Compute<true>(smoothLocal);
        } else {
            Compute<false>(smoothLocal);
        }
        inputXInQueue_.FreeTensor(inLocal);

        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            if (colsAsInt8_ < BLOCK_BYTES) {
                if (scaleBatchCount == 0) {
                    scaleBatchStart = i;
                }
                scaleBatchLocal.SetValue(scaleBatchCount, scaleLocal.GetValue(0));
                scaleBatchCount++;
                if (scaleBatchCount == scaleBatchRows || i == endRowIdx - 1 || i == this->activeNum_ - 1) {
                    DataCopyExtParams copyScaleBatchParams{
                        1, static_cast<uint32_t>(scaleBatchCount * sizeof(float)), 0, 0, 0};
                    DataCopyPad(expandedScaleGm_[scaleBatchStart], scaleBatchLocal, copyScaleBatchParams);
                    scaleBatchCount = 0;
                }
            } else {
                DataCopyPad(expandedScaleGm_[i], scaleLocal, {1, 4, 0, 0, 0});
            }
        } else {
            DataCopyPad(expandedScaleGm_[i], scaleLocal, {1, 4, 0, 0, 0});
        }
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            if (colsAsInt8_ < BLOCK_BYTES) {
                if (outBatchCount == 0) {
                    outBatchStart = i;
                }
                CopySmallFullLoadQuantOutToBatch(outBatchLocal, outLocal, outBatchCount * colsAsInt8_, colsAsInt8_);
                outBatchCount++;
                if (outBatchCount == outBatchRows || i == endRowIdx - 1 || i == this->activeNum_ - 1) {
                    DataCopyExtParams copyOutBatchParams{
                        1, static_cast<uint32_t>(outBatchCount * colsAsInt8_), 0, 0, 0};
                    DataCopyPad(expandedXGm_[outBatchStart * colsAsInt8_], outBatchLocal, copyOutBatchParams);
                    outBatchCount = 0;
                }
            } else {
                DataCopyPad(expandedXGm_[i * colsAsInt8_], outLocal, copyOutParams);
            }
        } else {
            DataCopyPad(expandedXGm_[i * colsAsInt8_], outLocal, copyOutParams);
        }

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        if (colsAsInt8_ < BLOCK_BYTES) {
            outBatchQueue_.FreeTensor(outBatchLocal);
            scaleBatchQueue_.FreeTensor(scaleBatchLocal);
        }
    }
    smoothInQueue_.FreeTensor(smoothLocal);
    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::GatherOutXDynamicQuant()
{
    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsAsInt8_), 0, 0, 0};

    int64_t curIndex = startRowIdx;
    int64_t k = 0;
    int64_t outputRows = Min(this->actualExpertIdxNum_, this->activeNum_);

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();

    for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, xGm_[i * this->cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[colsAlign_], xGm_[i * this->cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        Compute<false>(smoothLocal);

        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();

        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex >= 0 && outIndex < outputRows) {
                DataCopyPad(expandedXGm_[outIndex * colsAsInt8_], outLocal, copyOutParams);
                DataCopyPad(expandedScaleGm_[outIndex], scaleLocal, {1, 4, 0, 0, 0});
            }
        }

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
        inputXInQueue_.FreeTensor(inLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeV3FullLoadDynamicQuant<T, QuantT>::Process()
{
    if (this->blockIdx_ < this->needCoreNum_) {
        this->CopyIn();
        this->SortCompute();

        if (this->blockIdx_ == 0) {
            this->CopyOutRowIdx();
        }

        if (this->blockIdx_ == this->needCoreNum_ - 1 && this->expertTokensNumFlag_ == 1) {
            this->ComputeExpertTokenCount();
            this->CopyExpertCountToOutput();
        }

        if (this->epFullload_ || this->isInputScale_) {
            ScatterOutXDynamicQuant();
        } else {
            GatherOutXDynamicQuant();
        }

        this->FreeLocalTensor();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_DYNAMIC_QUANT_H
