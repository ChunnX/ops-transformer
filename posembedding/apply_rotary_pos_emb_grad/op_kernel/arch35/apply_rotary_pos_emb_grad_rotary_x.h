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
// ApplyNoBroadcastDcosDsin — A template 的 grad_cos/grad_sin 计算
// 无广播、无 reduce，按 Q/K 两路分别累加到输出
// ============================================================================
template <typename T>
class ApplyNoBroadcastDcosDsin
{
public:
    __aicore__ inline ApplyNoBroadcastDcosDsin(TPipe* pipe, const ApplyRotaryXParams* tiling)
        : pipe_(pipe), tilingData_(tiling) {}
    __aicore__ inline void Init(
        GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR query, GM_ADDR key, GM_ADDR cosGrad, GM_ADDR sinGrad);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const ApplyRotaryXParams* tilingData_;
    TPipe* pipe_;
    int64_t blockIdx_ = 0;
    int64_t dSplitCoef_ = HALF_INTERLEAVE_COEF;
    int64_t dBuffer_ = 0;
    int64_t ubFactorN_ = 0;
    GlobalTensor<T> gradQGm_, gradKGm_;
    GlobalTensor<T> queryGm_, keyGm_;
    GlobalTensor<T> cosGradGm_, sinGradGm_;
    TQue<QuePosition::VECIN, 1> gradQInQue_, gradKInQue_;
    TQue<QuePosition::VECIN, 1> queryInQue_, keyInQue_;
    TQue<QuePosition::VECOUT, 1> cosOutQue_, sinOutQue_;

    __aicore__ inline void ComputePath(
        __local_mem__ T* gradUb, __local_mem__ T* xUb, __local_mem__ T* cosUb, __local_mem__ T* sinUb,
        uint16_t currNNum, bool isAdd);
};

template <typename T>
__aicore__ inline void ApplyNoBroadcastDcosDsin<T>::Init(
    GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR query, GM_ADDR key, GM_ADDR cosGrad, GM_ADDR sinGrad)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    int64_t gmOffset = blockIdx_ * tilingData_->blockFactorN * tilingData_->d;
    dBuffer_ = Ops::Base::CeilAlign<int64_t>(tilingData_->d, BLOCK_TYPE_BYTES / sizeof(T));
    ubFactorN_ = tilingData_->ubFactorN;
    gradQGm_.SetGlobalBuffer((__gm__ T*)gradQ + gmOffset);
    gradKGm_.SetGlobalBuffer((__gm__ T*)gradK + gmOffset);
    queryGm_.SetGlobalBuffer((__gm__ T*)query + gmOffset);
    keyGm_.SetGlobalBuffer((__gm__ T*)key + gmOffset);
    cosGradGm_.SetGlobalBuffer((__gm__ T*)cosGrad + gmOffset);
    sinGradGm_.SetGlobalBuffer((__gm__ T*)sinGrad + gmOffset);
    pipe_->InitBuffer(gradQInQue_, bufferNum, ubFactorN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(gradKInQue_, bufferNum, ubFactorN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(queryInQue_, bufferNum, ubFactorN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(keyInQue_, bufferNum, ubFactorN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(cosOutQue_, bufferNum, ubFactorN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(sinOutQue_, bufferNum, ubFactorN_ * dBuffer_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ApplyNoBroadcastDcosDsin<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t currBlockFactorN =
        (blockIdx_ != tilingData_->usedCoreNum - 1) ? tilingData_->blockFactorN : tilingData_->blockTailFactorN;
    uint32_t loopCnt = Ops::Base::CeilDiv(currBlockFactorN, ubFactorN_);
    for (uint32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        uint16_t currNNum =
            (loopIdx != loopCnt - 1) ? ubFactorN_ : currBlockFactorN - loopIdx * ubFactorN_;
        int64_t offset = loopIdx * ubFactorN_ * tilingData_->d;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currNNum * tilingData_->d * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> gradQTensor = gradQInQue_.AllocTensor<T>();
        LocalTensor<T> gradKTensor = gradKInQue_.AllocTensor<T>();
        LocalTensor<T> queryTensor = queryInQue_.AllocTensor<T>();
        LocalTensor<T> keyTensor = keyInQue_.AllocTensor<T>();
        LocalTensor<T> cosTensor = cosOutQue_.AllocTensor<T>();
        LocalTensor<T> sinTensor = sinOutQue_.AllocTensor<T>();
        Duplicate(cosTensor, static_cast<float>(0.0), currNNum * tilingData_->d);
        Duplicate(sinTensor, static_cast<float>(0.0), currNNum * tilingData_->d);
        DataCopyPad(gradQTensor, gradQGm_[offset], copyParams, padParams);
        DataCopyPad(gradKTensor, gradKGm_[offset], copyParams, padParams);
        DataCopyPad(queryTensor, queryGm_[offset], copyParams, padParams);
        DataCopyPad(keyTensor, keyGm_[offset], copyParams, padParams);

        gradQInQue_.EnQue(gradQTensor);
        gradKInQue_.EnQue(gradKTensor);
        queryInQue_.EnQue(queryTensor);
        keyInQue_.EnQue(keyTensor);

        gradQTensor = gradQInQue_.DeQue<T>();
        gradKTensor = gradKInQue_.DeQue<T>();
        queryTensor = queryInQue_.DeQue<T>();
        keyTensor = keyInQue_.DeQue<T>();

        ComputePath(
            (__local_mem__ T*)gradQTensor.GetPhyAddr(), (__local_mem__ T*)queryTensor.GetPhyAddr(),
            (__local_mem__ T*)cosTensor.GetPhyAddr(), (__local_mem__ T*)sinTensor.GetPhyAddr(), currNNum, false);
        ComputePath(
            (__local_mem__ T*)gradKTensor.GetPhyAddr(), (__local_mem__ T*)keyTensor.GetPhyAddr(),
            (__local_mem__ T*)cosTensor.GetPhyAddr(), (__local_mem__ T*)sinTensor.GetPhyAddr(), currNNum, true);

        cosOutQue_.EnQue(cosTensor);
        sinOutQue_.EnQue(sinTensor);
        cosTensor = cosOutQue_.DeQue<T>();
        sinTensor = sinOutQue_.DeQue<T>();
        DataCopyPad(cosGradGm_[offset], cosTensor, copyParams);
        DataCopyPad(sinGradGm_[offset], sinTensor, copyParams);

        gradQInQue_.FreeTensor(gradQTensor);
        gradKInQue_.FreeTensor(gradKTensor);
        queryInQue_.FreeTensor(queryTensor);
        keyInQue_.FreeTensor(keyTensor);
        cosOutQue_.FreeTensor(cosTensor);
        sinOutQue_.FreeTensor(sinTensor);
    }
}

template <typename T>
__aicore__ inline void ApplyNoBroadcastDcosDsin<T>::ComputePath(
    __local_mem__ T* gradUb, __local_mem__ T* xUb, __local_mem__ T* cosUb, __local_mem__ T* sinUb,
    uint16_t currNNum, bool isAdd)
{
    uint32_t halfD = tilingData_->d / dSplitCoef_;
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> g1, g2, x1, x2, c1, c2, s1, s2, prev1, prev2;
        MicroAPI::MaskReg preg;
        for (uint16_t row = 0; row < currNNum; row++) {
            __local_mem__ T* currGradUb = gradUb + row * tilingData_->d;
            __local_mem__ T* currXUb = xUb + row * tilingData_->d;
            __local_mem__ T* currCosUb = cosUb + row * tilingData_->d;
            __local_mem__ T* currSinUb = sinUb + row * tilingData_->d;
            uint32_t updateCnt = halfD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(updateCnt);
                int32_t offset = i * VL_FLOAT32_SIZE;
                int32_t halfOffset = offset + halfD;
                ops::LoadTwoTensorForDtypeT<T>(currGradUb, currGradUb, g1, g2, preg, preg, offset, halfOffset);
                ops::LoadTwoTensorForDtypeT<T>(currXUb, currXUb, x1, x2, preg, preg, offset, halfOffset);
                Mul(c1, g1, x1, preg);
                Mul(c2, g2, x2, preg);
                Mul(s1, g1, x2, preg);
                Muls(s1, s1, static_cast<float>(-1.0), preg);
                Mul(s2, g2, x1, preg);

                if (isAdd) {
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, prev1, prev2, preg, preg, offset, halfOffset);
                    Add(c1, c1, prev1, preg);
                    Add(c2, c2, prev2, preg);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, prev1, prev2, preg, preg, offset, halfOffset);
                    Add(s1, s1, prev1, preg);
                    Add(s2, s2, prev2, preg);
                }
                ops::StoreOneTensorForDtypeT<T>(currCosUb, c1, preg, offset);
                ops::StoreOneTensorForDtypeT<T>(currCosUb, c2, preg, halfOffset);
                ops::StoreOneTensorForDtypeT<T>(currSinUb, s1, preg, offset);
                ops::StoreOneTensorForDtypeT<T>(currSinUb, s2, preg, halfOffset);
            }
        }
    }
}

} // namespace ApplyRotaryPosEmbGrad

#endif // __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_H__
