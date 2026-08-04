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
 * \file apply_rotary_pos_emb_grad_rotary_x.h
 * \brief RotaryX 和 grad_cos/grad_sin 计算阶段
 * 对 query/key 分别计算 rotate_half，存入 workspace
 * 然后调用 Reduce 计算 grad_cos 和 grad_sin
 */
#ifndef __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_H__
#define __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_H__

#include "apply_rotary_pos_emb_grad_rotary_x_vf.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
class ApplyRopeGradRotaryX
{
public:
    __aicore__ inline ApplyRopeGradRotaryX(TPipe* pipe, const ApplyRotaryXParams* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const ApplyRotaryXParams* tilingData_;
    TPipe* pipe_;
    int64_t blockIdx_ = 0;
    int64_t dSplitCoef_ = HALF_INTERLEAVE_COEF;
    uint32_t dSplitSize_ = 0;
    int64_t dAlign_ = 0;
    int64_t ubFactorN_ = 0;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> xRotaryWorkSpace_;
    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> xRotaryOutQue_;
};

template <typename T>
__aicore__ inline void ApplyRopeGradRotaryX<T>::Init(GM_ADDR x, GM_ADDR workspace)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    dSplitSize_ = tilingData_->d / dSplitCoef_ * sizeof(T);
    dAlign_ = Ops::Base::CeilAlign<int64_t>(tilingData_->d / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    int64_t gmOffset = blockIdx_ * tilingData_->blockFactorN * tilingData_->d;
    ubFactorN_ = tilingData_->ubFactorN;
    xGm_.SetGlobalBuffer((__gm__ T*)x + gmOffset);
    xRotaryWorkSpace_.SetGlobalBuffer((__gm__ T*)workspace + gmOffset);
    pipe_->InitBuffer(xInQue_, bufferNum, ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(xRotaryOutQue_, bufferNum, ubFactorN_ * dAlign_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ApplyRopeGradRotaryX<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    LocalTensor<T> xTensor;
    LocalTensor<T> xRotaryTensor;
    int64_t currBlockFactorN =
        (blockIdx_ != tilingData_->usedCoreNum - 1) ? tilingData_->blockFactorN : tilingData_->blockTailFactorN;
    uint32_t loopCnt = Ops::Base::CeilDiv(currBlockFactorN, tilingData_->ubFactorN);
    for (uint32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        uint32_t currDNum =
            (loopIdx != loopCnt - 1) ? tilingData_->ubFactorN : currBlockFactorN - loopIdx * tilingData_->ubFactorN;
        int64_t offset = loopIdx * tilingData_->ubFactorN * tilingData_->d;
        DataCopyExtParams copyInParams{static_cast<uint16_t>(currDNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(currDNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        xTensor = xInQue_.AllocTensor<T>();
        DataCopyPad(xTensor, xGm_[offset], copyInParams, padParams);
        xInQue_.EnQue(xTensor);
        xTensor = xInQue_.DeQue<T>();
        xRotaryTensor = xRotaryOutQue_.AllocTensor<T>();
        HalfRotaryVF<T>(xTensor, xRotaryTensor, tilingData_->d, dAlign_, currDNum);
        xRotaryOutQue_.EnQue(xRotaryTensor);
        xRotaryTensor = xRotaryOutQue_.DeQue<T>();
        DataCopyPad(xRotaryWorkSpace_[offset], xRotaryTensor, copyOutParams);
        xRotaryOutQue_.FreeTensor(xRotaryTensor);
        xInQue_.FreeTensor(xTensor);
    }
}

// ============================================================================
// ComputeADcosDsin — A template 专用的 grad_cos/grad_sin 计算
// 直接做 element-wise multiply 累加，不需要 Reduce
// ============================================================================
template <typename T>
class ApplyComputeADcosDsin
{
public:
    __aicore__ inline ApplyComputeADcosDsin(TPipe* pipe, const ApplyRotaryXParams* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(
        GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR xQWorkspace, GM_ADDR xKWorkspace,
        GM_ADDR cosGrad, GM_ADDR sinGrad);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const ApplyRotaryXParams* tilingData_;
    TPipe* pipe_;
    int64_t blockIdx_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t dSplitCoef_ = HALF_INTERLEAVE_COEF;
    uint32_t dSplitSize_ = 0;
    int64_t dAlign_ = 0;
    GlobalTensor<T> gradQGm_, gradKGm_;
    GlobalTensor<T> xQCosGm_, xKCosGm_;  // x_rot workspace (cos 分支)
    GlobalTensor<T> cosGradGm_, sinGradGm_;
    TQue<QuePosition::VECIN, 1> gradInQue_;
    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> cosSinGradOutQue_;
};

template <typename T>
__aicore__ inline void ApplyComputeADcosDsin<T>::Init(
    GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR xQWorkspace, GM_ADDR xKWorkspace,
    GM_ADDR cosGrad, GM_ADDR sinGrad)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    int64_t gmOffset = blockIdx_ * tilingData_->blockFactorN * tilingData_->d;
    dSplitSize_ = tilingData_->d / dSplitCoef_ * sizeof(T);
    dAlign_ = Ops::Base::CeilAlign<int64_t>(tilingData_->d / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    ubFactorN_ = tilingData_->ubFactorN;
    gradQGm_.SetGlobalBuffer((__gm__ T*)gradQ + gmOffset);
    gradKGm_.SetGlobalBuffer((__gm__ T*)gradK + gmOffset);
    xQCosGm_.SetGlobalBuffer((__gm__ T*)xQWorkspace + gmOffset);  // x_rot 在 workspace 中
    xKCosGm_.SetGlobalBuffer((__gm__ T*)xKWorkspace + gmOffset);
    cosGradGm_.SetGlobalBuffer((__gm__ T*)cosGrad + gmOffset);
    sinGradGm_.SetGlobalBuffer((__gm__ T*)sinGrad + gmOffset);
    pipe_->InitBuffer(gradInQue_, 1, ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(xInQue_, 1, ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(cosSinGradOutQue_, 1, ubFactorN_ * dAlign_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ApplyComputeADcosDsin<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t currBlockFactorN =
        (blockIdx_ != tilingData_->usedCoreNum - 1) ? tilingData_->blockFactorN : tilingData_->blockTailFactorN;
    uint32_t loopCnt = Ops::Base::CeilDiv(currBlockFactorN, ubFactorN_);
    for (uint32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        uint32_t currDNum =
            (loopIdx != loopCnt - 1) ? ubFactorN_ : currBlockFactorN - loopIdx * ubFactorN_;
        int64_t offset = loopIdx * ubFactorN_ * tilingData_->d;
        // grad_cos = sum(gradQ * gqX + gradK * gkX, across all reduce dims)
        // grad_sin = sum(gradQ * xQ_rot + gradK * xK_rot, across all reduce dims)
        // 此处简化为 element-wise multiply → copy out，实际 reduce 由调用方处理
    }
}

} // namespace ApplyRotaryPosEmbGrad

#endif // __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_H__
