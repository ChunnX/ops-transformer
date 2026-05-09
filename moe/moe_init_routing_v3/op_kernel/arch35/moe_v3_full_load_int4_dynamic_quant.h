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
 * \file moe_v3_full_load_int4_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_INT4_DYNAMIC_QUANT_H
#define MOE_V3_FULL_LOAD_INT4_DYNAMIC_QUANT_H

#include "moe_v3_full_load_base.h"
#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t FULLLOAD_INT4_DYNAMIC_QUANT_BUFFER_NUM = 1;

template <typename T>
class MoeV3FullLoadInt4DynamicQuant : public MoeV3FullLoadBase<T> {
public:
    __aicore__ inline MoeV3FullLoadInt4DynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    __aicore__ inline void ScatterOutXInt4DynamicQuant();
    __aicore__ inline void GatherOutXInt4DynamicQuant();

private:
    TQue<QuePosition::VECIN, 1> inputXInQueue_;
    TQue<QuePosition::VECIN, 1> smoothInQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int4b_t> expandedXGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> expandedScaleGm_;

    int64_t colsAlign_;

    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF16ToI4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT_SYM,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void
MoeV3FullLoadInt4DynamicQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                       GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                       GM_ADDR expandedScale, GM_ADDR workspace,
                                       const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    MoeV3FullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);

    colsAlign_ = Align(this->cols_, sizeof(T));

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ int4b_t *)expandedX);

    if (this->isInputScale_) {
        quantSmoothGm_.SetGlobalBuffer((__gm__ float *)scale);
    }
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    if constexpr (IsSameType<T, float>::value) {
        this->pipe_->InitBuffer(inputXInQueue_, FULLLOAD_INT4_DYNAMIC_QUANT_BUFFER_NUM,
                                AlignBytes(this->cols_, sizeof(float)));
    } else {
        this->pipe_->InitBuffer(inputXInQueue_, FULLLOAD_INT4_DYNAMIC_QUANT_BUFFER_NUM,
                                2 * AlignBytes(this->cols_, sizeof(T)));
    }
    this->pipe_->InitBuffer(smoothInQueue_, FULLLOAD_INT4_DYNAMIC_QUANT_BUFFER_NUM,
                             AlignBytes(this->cols_, sizeof(float)));
    this->pipe_->InitBuffer(inputXOutQueue_, FULLLOAD_INT4_DYNAMIC_QUANT_BUFFER_NUM,
                            AlignBytes(this->cols_ / ONE_BYTE_INT4_NUM, sizeof(int4b_t)));
    this->pipe_->InitBuffer(scaleOutQueue_, FULLLOAD_INT4_DYNAMIC_QUANT_BUFFER_NUM, BLOCK_BYTES + BLOCK_BYTES);
}

template <typename T>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeV3FullLoadInt4DynamicQuant<T>::Compute(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int4b_t> outLocal = inputXOutQueue_.AllocTensor<int4b_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleLocal.GetPhyAddr();
    __local_mem__ int4b_t *outUbAddr = (__local_mem__ int4b_t *)outLocal.GetPhyAddr();
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
        MicroAPI::RegTensor<int4b_t> outRegI4;

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
        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::DataCopy(scaleUbAddr, scaleValueReg, maskRegVL8);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::DataCopy(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
            MicroAPI::Cast<int4b_t, half, castTraitF16ToI4>(outRegI4, outRegF16, maskRegInLoop);
            MicroAPI::DataCopy<int4b_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                             outRegI4, maskRegInLoop);
        }
    }

    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadInt4DynamicQuant<T>::ScatterOutXInt4DynamicQuant()
{
    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->cols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1,
                                        static_cast<uint32_t>(this->cols_ / ONE_BYTE_INT4_NUM * sizeof(int4b_t)),
                                        0, 0, 0};

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    int32_t lastExpertIdx = -1;

    for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
            break;
        }

        int32_t srcIdx = sortedRowIdx.GetValue(i);
        int32_t expertIdx = curExpertId - this->expertStart_;
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[colsAlign_], xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        if (this->isInputScale_ && expertIdx != lastExpertIdx) {
            DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * this->cols_], smoothParams, {false, 0, 0, 0});
            smoothInQueue_.EnQue(smoothLocal);
            smoothLocal = smoothInQueue_.DeQue<float>();
            lastExpertIdx = expertIdx;
        }

        if (this->isInputScale_) {
            Compute<true>(smoothLocal);
        } else {
            Compute<false>(smoothLocal);
        }
        inputXInQueue_.FreeTensor(inLocal);

        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(expandedScaleGm_[i], scaleLocal, {1, 4, 0, 0, 0});
        LocalTensor<int4b_t> outLocal = inputXOutQueue_.DeQue<int4b_t>();
        DataCopyPad(expandedXGm_[i * this->cols_], outLocal, copyOutParams);

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadInt4DynamicQuant<T>::GatherOutXInt4DynamicQuant()
{
    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1,
                                        static_cast<uint32_t>(this->cols_ / ONE_BYTE_INT4_NUM * sizeof(int4b_t)),
                                        0, 0, 0};

    int64_t curIndex = startRowIdx;
    int64_t k = 0;

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

        LocalTensor<int4b_t> outLocal = inputXOutQueue_.DeQue<int4b_t>();
        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();

        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex < this->activeNum_) {
                DataCopyPad(expandedXGm_[outIndex * this->cols_], outLocal, copyOutParams);
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

template <typename T>
__aicore__ inline void MoeV3FullLoadInt4DynamicQuant<T>::Process()
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
            ScatterOutXInt4DynamicQuant();
        } else {
            GatherOutXInt4DynamicQuant();
        }

        this->FreeLocalTensor();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_INT4_DYNAMIC_QUANT_H
