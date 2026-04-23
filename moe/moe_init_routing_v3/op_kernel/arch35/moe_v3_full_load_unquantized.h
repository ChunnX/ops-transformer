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
 * \file moe_v3_full_load_unquantized.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_UNQUANTIZED_H
#define MOE_V3_FULL_LOAD_UNQUANTIZED_H

#include "moe_v3_full_load_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3FullLoadUnquantized : public MoeV3FullLoadBase<T> {
public:
    __aicore__ inline MoeV3FullLoadUnquantized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ScatterOutX();
    __aicore__ inline void GatherOutX();
    __aicore__ inline void ScatterOutScale();
    __aicore__ inline void GatherOutScale();

private:
    TQue<QuePosition::VECIN, 1> xCopyInQueue_;
    TQue<QuePosition::VECIN, 1> scaleCopyInQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<uint8_t> xUint8tGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<float> expandedScaleGm_;
};

template <typename T>
__aicore__ inline void
MoeV3FullLoadUnquantized<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                  GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                  GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    MoeV3FullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        xUint8tGm_.SetGlobalBuffer((__gm__ uint8_t *)x);
    } else {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
    }
    if (this->isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }

    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);

    int64_t rowLength = this->endXRow_ - this->startXRow_ + 1;

    if (this->epFullload_) {
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_, sizeof(uint8_t)));
        } else {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_, sizeof(T)));
        }
    } else {
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_,
                                    AlignBytes(this->cols_ * rowLength, sizeof(uint8_t)));
        } else {
            this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_ * rowLength, sizeof(T)));
        }
    }
    this->pipe_->InitBuffer(scaleCopyInQueue_, 1, AlignBytes(1, sizeof(float)));
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::Process()
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

        if (this->epFullload_) {
            this->ScatterOutX();
            if (this->isInputScale_) {
                this->ScatterOutScale();
            }
        } else {
            this->GatherOutX();
            if (this->isInputScale_) {
                this->GatherOutScale();
            }
        }

        this->FreeLocalTensor();
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::ScatterOutX()
{
    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();

    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        DataCopyExtParams hif8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(uint8_t)),
                                         0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        LocalTensor<uint8_t> xLocal = xCopyInQueue_.AllocTensor<uint8_t>();
        for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
            int32_t curExpertId = sortedExpertIdx.GetValue(i);
            if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
                break;
            }
            int64_t rowIdx = sortedRowIdx.GetValue(i);
            int64_t srcOffset = rowIdx / this->k_ * this->cols_;
            int64_t dstOffset = i * this->cols_;
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(xLocal, xUint8tGm_[srcOffset], hif8CopyParams, padParams);
            xCopyInQueue_.EnQue(xLocal);
            LocalTensor<T> xOutLocal = xCopyInQueue_.DeQue<T>();
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(expandedXGm_[dstOffset], xOutLocal, copyParams);
            xCopyInQueue_.FreeTensor(xOutLocal);
        }
    } else {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
            int32_t curExpertId = sortedExpertIdx.GetValue(i);
            if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
                break;
            }
            int64_t rowIdx = sortedRowIdx.GetValue(i);
            int64_t srcOffset = rowIdx / this->k_ * this->cols_;
            int64_t dstOffset = i * this->cols_;
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(expandedXGm_[dstOffset], xLocal, copyParams);
        }
        xCopyInQueue_.FreeTensor(xLocal);
    }

    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::GatherOutX()
{
    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t rowLength = this->endXRow_ - this->startXRow_ + 1;
    int64_t inFactor = Align(this->cols_, sizeof(T));

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        DataCopyExtParams hif8CopyParams{static_cast<uint16_t>(rowLength),
                                         static_cast<uint32_t>(this->cols_ * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        LocalTensor<uint8_t> xLocal = xCopyInQueue_.AllocTensor<uint8_t>();
        DataCopyPad(xLocal, xUint8tGm_[this->startXRow_ * this->cols_], hif8CopyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        xCopyInQueue_.EnQue(xLocal);
        LocalTensor<T> xOutLocal = xCopyInQueue_.DeQue<T>();
        int64_t curIndex = startRowIdx;
        int64_t k = 0;
        for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
            for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
                int32_t outIndex = expandedRowIdx.GetValue(curIndex);
                if (outIndex < this->activeNum_) {
                    DataCopyPad(expandedXGm_[outIndex * this->cols_], xOutLocal[(i - this->startXRow_) * inFactor],
                                copyParams);
                }
            }
        }
        xCopyInQueue_.FreeTensor(xOutLocal);
    } else {
        DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(rowLength),
                                          static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        DataCopyPad(xLocal, xGm_[this->startXRow_ * this->cols_], dataXCopyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        int64_t curIndex = startRowIdx;
        int64_t k = 0;
        for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
            for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
                int32_t outIndex = expandedRowIdx.GetValue(curIndex);
                if (outIndex < this->activeNum_) {
                    DataCopyPad(expandedXGm_[outIndex * this->cols_], xLocal[(i - this->startXRow_) * inFactor],
                                copyParams);
                }
            }
        }
        xCopyInQueue_.FreeTensor(xLocal);
    }

    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::ScatterOutScale()
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;

    for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
            break;
        }
        int64_t rowIdx = sortedRowIdx.GetValue(i);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        DataCopyPad(scaleLocal, scaleGm_[rowIdx / this->k_], copyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyPad(expandedScaleGm_[i], scaleLocal, copyParams);
    }

    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadUnquantized<T>::GatherOutScale()
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t curIndex = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t k = 0;

    for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        DataCopyPad(scaleLocal, scaleGm_[i], copyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex < this->activeNum_) {
                DataCopyPad(expandedScaleGm_[outIndex], scaleLocal, copyParams);
            }
        }
    }

    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_UNQUANTIZED_H