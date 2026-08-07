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
 * \brief A 模板 grad_cos/grad_sin 计算阶段
 * 无广播场景下按 Q/K 两路直接累加，无需 workspace 中转和 Reduce
 */
#ifndef __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_H__
#define __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_H__

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

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

    __aicore__ inline void ComputeDcosDsin(
        __local_mem__ T* gradQUb, __local_mem__ T* queryUb, __local_mem__ T* gradKUb, __local_mem__ T* keyUb,
        __local_mem__ T* cosUb, __local_mem__ T* sinUb, uint16_t currNNum);
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
    dBuffer_ = Ops::Base::CeilAlign<int64_t>(tilingData_->d, BLOCK_TYPE_SIZE / sizeof(T));
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

        ComputeDcosDsin(
            (__local_mem__ T*)gradQTensor.GetPhyAddr(), (__local_mem__ T*)queryTensor.GetPhyAddr(),
            (__local_mem__ T*)gradKTensor.GetPhyAddr(), (__local_mem__ T*)keyTensor.GetPhyAddr(),
            (__local_mem__ T*)cosTensor.GetPhyAddr(), (__local_mem__ T*)sinTensor.GetPhyAddr(), currNNum);

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
__aicore__ inline void ApplyNoBroadcastDcosDsin<T>::ComputeDcosDsin(
    __local_mem__ T* gradQUb, __local_mem__ T* queryUb, __local_mem__ T* gradKUb, __local_mem__ T* keyUb,
    __local_mem__ T* cosUb, __local_mem__ T* sinUb, uint16_t currNNum)
{
    uint32_t halfD = tilingData_->d / dSplitCoef_;
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> gq1, gq2, q1, q2, gk1, gk2, k1, k2;
        MicroAPI::RegTensor<float> c1, c2, s1, s2, t1, t2;
        MicroAPI::MaskReg preg;
        for (uint16_t row = 0; row < currNNum; row++) {
            __local_mem__ T* currGradQUb = gradQUb + row * tilingData_->d;
            __local_mem__ T* currQueryUb = queryUb + row * tilingData_->d;
            __local_mem__ T* currGradKUb = gradKUb + row * tilingData_->d;
            __local_mem__ T* currKeyUb = keyUb + row * tilingData_->d;
            __local_mem__ T* currCosUb = cosUb + row * tilingData_->d;
            __local_mem__ T* currSinUb = sinUb + row * tilingData_->d;
            uint32_t updateCnt = halfD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(updateCnt);
                int32_t offset = i * VL_FLOAT32_SIZE;
                int32_t halfOffset = offset + halfD;
                ops::LoadTwoTensorForDtypeT<T>(currGradQUb, currGradQUb, gq1, gq2, preg, preg, offset, halfOffset);
                ops::LoadTwoTensorForDtypeT<T>(currQueryUb, currQueryUb, q1, q2, preg, preg, offset, halfOffset);
                ops::LoadTwoTensorForDtypeT<T>(currGradKUb, currGradKUb, gk1, gk2, preg, preg, offset, halfOffset);
                ops::LoadTwoTensorForDtypeT<T>(currKeyUb, currKeyUb, k1, k2, preg, preg, offset, halfOffset);

                // grad_cos = gradQ * query + gradK * key
                Mul(c1, gq1, q1, preg);
                Mul(t1, gk1, k1, preg);
                Add(c1, c1, t1, preg);
                Mul(c2, gq2, q2, preg);
                Mul(t2, gk2, k2, preg);
                Add(c2, c2, t2, preg);

                // grad_sin = -(gradQ * rotate(query)) + -(gradK * rotate(key))
                Mul(s1, gq1, q2, preg);
                Mul(t1, gk1, k2, preg);
                Add(s1, s1, t1, preg);
                Muls(s1, s1, static_cast<float>(-1.0), preg);
                Mul(s2, gq2, q1, preg);
                Mul(t2, gk2, k1, preg);
                Add(s2, s2, t2, preg);

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
