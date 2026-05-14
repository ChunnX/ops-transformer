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
 * \file moe_v3_full_load_mxfp8_quant.h
 * \brief MXFP8全载模板，支持E5M2和E4M3FN两种FP8格式
 */
#ifndef MOE_V3_FULL_LOAD_MXFP8_QUANT_H
#define MOE_V3_FULL_LOAD_MXFP8_QUANT_H

#include "moe_v3_full_load_base.h"
#include "moe_v3_gather_mxfp8_quant.h"
#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t FULLLOAD_MXFP8_QUANT_BUFFER_NUM = 1;

template <typename T, typename U>
class MoeV3FullLoadMxfp8Quant : public MoeV3FullLoadBase<T> {
public:
    __aicore__ inline MoeV3FullLoadMxfp8Quant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeMxQuant(int64_t rowLength);
    __aicore__ inline void ScatterMxQuantProcess();
    __aicore__ inline void GatherMxQuantProcess();

private:
    TQue<QuePosition::VECIN, 1> xInputQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, 1> mxScaleOutQueue_;
    TBuf<TPosition::VECCALC> maxExpBuf_;
    TBuf<TPosition::VECCALC> invScaleBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int8_t> expandedScaleGm_;

    uint16_t lowerBoundOfB16MaxExp_;
    int64_t mxQuantBlockNum_;
    int64_t scaleCols_;
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    const uint32_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    const uint32_t vlForB16_ = vRegSize_ / sizeof(T);
    const uint32_t numUbBlocksPerVReg_ = vRegSize_ / ubBlockSize_;
};

template <typename T, typename U>
__aicore__ inline void
MoeV3FullLoadMxfp8Quant<T, U>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                    const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    MoeV3FullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);

    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3;
    } else {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2;
    }

    mxQuantBlockNum_ = Ops::Base::CeilDiv<int64_t>(this->cols_, MX_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(mxQuantBlockNum_, 2);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    expandedScaleGm_.SetGlobalBuffer((__gm__ int8_t *)expandedScale);

    this->pipe_->InitBuffer(xInputQueue_, FULLLOAD_MXFP8_QUANT_BUFFER_NUM, AlignBytes(this->cols_, sizeof(T)));
    this->pipe_->InitBuffer(xQuantOutQueue_, FULLLOAD_MXFP8_QUANT_BUFFER_NUM, AlignBytes(this->cols_, sizeof(int8_t)));
    this->pipe_->InitBuffer(mxScaleOutQueue_, FULLLOAD_MXFP8_QUANT_BUFFER_NUM, AlignBytes(scaleCols_, sizeof(int8_t)));
    this->pipe_->InitBuffer(maxExpBuf_, AlignBytes(scaleCols_, sizeof(T)));
    this->pipe_->InitBuffer(invScaleBuf_, AlignBytes(scaleCols_, sizeof(T)));
}

template <typename T, typename U>
__aicore__ inline void MoeV3FullLoadMxfp8Quant<T, U>::ComputeMxQuant(int64_t rowLength)
{
    LocalTensor<T> xInputLocal = xInputQueue_.DeQue<T>();
    LocalTensor<int8_t> xQuantLocal = xQuantOutQueue_.AllocTensor<int8_t>();
    LocalTensor<uint16_t> mxScaleLocal = mxScaleOutQueue_.AllocTensor<uint16_t>();
    LocalTensor<uint16_t> maxExpLocal = maxExpBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> invScaleLocal = invScaleBuf_.Get<uint16_t>();

    __ubuf__ T *xAddr = reinterpret_cast<__ubuf__ T *>(xInputLocal.GetPhyAddr());
    __ubuf__ int8_t *xQuantAddr = reinterpret_cast<__ubuf__ int8_t *>(xQuantLocal.GetPhyAddr());
    __ubuf__ uint16_t *mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxScaleLocal.GetPhyAddr());
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
    __ubuf__ uint16_t *invScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(invScaleLocal.GetPhyAddr());

    uint32_t xElemNum = static_cast<uint32_t>(rowLength);
    uint32_t scaleElemNum = static_cast<uint32_t>(scaleCols_);
    uint32_t validScaleElemNum = static_cast<uint32_t>(mxQuantBlockNum_);
    uint16_t vfLoopNumForX = (xElemNum + vlForB16_ * 2 - 1) / (vlForB16_ * 2);
    uint16_t vfLoopNumForScale = (scaleElemNum + vlForB16_ - 1) / vlForB16_;

    VF_CALL<vfComputeMaxExp<T, U>>(xAddr, maxExpAddr, xElemNum, vfLoopNumForX, vlForB16_, numUbBlocksPerVReg_);
    VF_CALL<vfComputeScale<T, U>>(maxExpAddr, mxScaleAddr, invScaleAddr, scaleElemNum, validScaleElemNum,
                                  vfLoopNumForScale, vlForB16_, lowerBoundOfB16MaxExp_);
    VF_CALL<vfComputeData<T, U>>(xAddr, invScaleAddr, xQuantAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                 numUbBlocksPerVReg_);

    xInputQueue_.FreeTensor(xInputLocal);
    xQuantOutQueue_.EnQue<int8_t>(xQuantLocal);
    mxScaleOutQueue_.EnQue<uint16_t>(mxScaleLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeV3FullLoadMxfp8Quant<T, U>::ScatterMxQuantProcess()
{
    LocalTensor<int32_t> sortedRowIdx = this->sortedRowIdxQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertIdx = this->sortedExpertIdxQueue_.template DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyExtParams copyScaleParams{1, static_cast<uint32_t>(scaleCols_ * sizeof(int8_t)), 0, 0, 0};

    for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
        int32_t curExpertId = sortedExpertIdx.GetValue(i);
        if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
            break;
        }

        int32_t srcIdx = sortedRowIdx.GetValue(i);
        LocalTensor<T> inLocal = xInputQueue_.AllocTensor<T>();
        DataCopyPad(inLocal, xGm_[srcIdx / this->k_ * this->cols_], copyInParams, {false, 0, 0, 0});
        xInputQueue_.EnQue<T>(inLocal);

        ComputeMxQuant(this->cols_);

        LocalTensor<int8_t> outLocal = xQuantOutQueue_.DeQue<int8_t>();
        LocalTensor<uint8_t> scaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();

        DataCopyPad(expandedXGm_[i * this->cols_], outLocal, copyOutParams);
        DataCopyPad<uint8_t>(expandedScaleGm_[i * scaleCols_], scaleLocal, copyScaleParams);

        xQuantOutQueue_.FreeTensor(outLocal);
        mxScaleOutQueue_.FreeTensor(scaleLocal);
    }

    this->sortedRowIdxQueue_.template EnQue<int32_t>(sortedRowIdx);
    this->sortedExpertIdxQueue_.template EnQue<int32_t>(sortedExpertIdx);
}

template <typename T, typename U>
__aicore__ inline void MoeV3FullLoadMxfp8Quant<T, U>::GatherMxQuantProcess()
{
    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxQueue_.template DeQue<int32_t>();

    int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t curIndex = startRowIdx;
    int64_t k = 0;

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyExtParams copyScaleParams{1, static_cast<uint32_t>(scaleCols_ * sizeof(int8_t)), 0, 0, 0};

    for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
        LocalTensor<T> inLocal = xInputQueue_.AllocTensor<T>();
        DataCopyPad(inLocal, xGm_[i * this->cols_], copyInParams, {false, 0, 0, 0});
        xInputQueue_.EnQue<T>(inLocal);

        ComputeMxQuant(this->cols_);

        LocalTensor<int8_t> outLocal = xQuantOutQueue_.DeQue<int8_t>();
        LocalTensor<uint8_t> scaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();

        for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            if (outIndex < this->activeNum_) {
                DataCopyPad(expandedXGm_[outIndex * this->cols_], outLocal, copyOutParams);
                DataCopyPad<uint8_t>(expandedScaleGm_[outIndex * scaleCols_], scaleLocal, copyScaleParams);
            }
        }

        xQuantOutQueue_.FreeTensor(outLocal);
        mxScaleOutQueue_.FreeTensor(scaleLocal);
        xInputQueue_.FreeTensor(inLocal);
    }

    this->expandedRowIdxQueue_.template EnQue<int32_t>(expandedRowIdx);
}

template <typename T, typename U>
__aicore__ inline void MoeV3FullLoadMxfp8Quant<T, U>::Process()
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
            ScatterMxQuantProcess();
        } else {
            GatherMxQuantProcess();
        }

        this->FreeLocalTensor();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_MXFP8_QUANT_H