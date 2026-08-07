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
 * \file apply_rotary_pos_emb_grad_bab.h
 * \brief
 *
 * 核间: B × S 二维网格分核
 * 核内: 外层 S 循环搬 cos/sin（被 Q/K 两路复用）
 *       内层 N 循环同时搬入 gradQueryEmbed + gradKeyEmbed
 *       在同一帧内完成 Q/K 两路的梯度计算和 cos/sin 部分积累加
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_BAB_H
#define APPLY_ROTARY_POS_EMB_GRAD_BAB_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbGradBAB
{
public:
    __aicore__ inline ApplyRotaryPosEmbGradBAB(TPipe* pipe, const ApplyRopeGradRegbaseParams* tiling)
        : pipe_(pipe), tilingData_(tiling) {}
    __aicore__ inline void Init(
        GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed,
        GM_ADDR cos, GM_ADDR sin,
        GM_ADDR query, GM_ADDR key,
        GM_ADDR gradQueryOut, GM_ADDR gradKeyOut,
        GM_ADDR gradCosOut, GM_ADDR gradSinOut);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;

    const ApplyRopeGradRegbaseParams* tilingData_;
    TPipe* pipe_;

    GlobalTensor<T> gradQGm_;
    GlobalTensor<T> gradKGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> keyGm_;
    GlobalTensor<T> gradQOutGm_;
    GlobalTensor<T> gradKOutGm_;
    GlobalTensor<T> gradCosOutGm_;
    GlobalTensor<T> gradSinOutGm_;

    TQue<QuePosition::VECIN, bufferNum> gradQInQue_;
    TQue<QuePosition::VECIN, bufferNum> gradKInQue_;
    TQue<QuePosition::VECIN, bufferNum> cosInQue_;
    TQue<QuePosition::VECIN, bufferNum> sinInQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradQOutQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradKOutQue_;

    TQue<QuePosition::VECIN, bufferNum> queryInQue_;
    TQue<QuePosition::VECIN, bufferNum> keyInQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradCosOutQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradSinOutQue_;

    int64_t blockIdx_ = 0;
    int64_t dAlign_ = 0;
    int64_t bIdx_ = 0;
    int64_t sIdx_ = 0;
    int64_t bNum_ = 0;
    int64_t sNum_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;

private:
    __aicore__ inline void PrePareParams();
    __aicore__ inline void ProcessNLoop(uint32_t bIdx, uint32_t sIdx, uint32_t currSNum);
    __aicore__ inline void ProcessN(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        uint32_t bIdx, uint32_t sIdx, uint32_t currSNum);
    __aicore__ inline void ComputeOnePath(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        const LocalTensor<T>& inTensor, const LocalTensor<T>& outTensor,
        uint32_t currSNum, uint32_t currDNum);
};

// =================================================================
// Init
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::Init(
    GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed,
    GM_ADDR cos, GM_ADDR sin,
    GM_ADDR query, GM_ADDR key,
    GM_ADDR gradQueryOut, GM_ADDR gradKeyOut,
    GM_ADDR gradCosOut, GM_ADDR gradSinOut)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->dAlign_ = Ops::Base::CeilAlign<int64_t>(tilingData_->d, BLOCK_TYPE_SIZE / sizeof(T));
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorS_ = tilingData_->ubFactorS;

    this->gradQGm_.SetGlobalBuffer((__gm__ T*)gradQueryEmbed);
    this->gradKGm_.SetGlobalBuffer((__gm__ T*)gradKeyEmbed);
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    this->queryGm_.SetGlobalBuffer((__gm__ T*)query);
    this->keyGm_.SetGlobalBuffer((__gm__ T*)key);
    this->gradQOutGm_.SetGlobalBuffer((__gm__ T*)gradQueryOut);
    this->gradKOutGm_.SetGlobalBuffer((__gm__ T*)gradKeyOut);
    this->gradCosOutGm_.SetGlobalBuffer((__gm__ T*)gradCosOut);
    this->gradSinOutGm_.SetGlobalBuffer((__gm__ T*)gradSinOut);

    this->pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradQInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradKInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradQOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    this->pipe_->InitBuffer(gradKOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));

    if (tilingData_->dCosFlag) {
        this->pipe_->InitBuffer(queryInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(keyInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(gradCosOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(gradSinOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    }
}

// =================================================================
// PrePareParams
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::PrePareParams()
{
    bIdx_ = blockIdx_ % tilingData_->blockNumB;
    sIdx_ = blockIdx_ / tilingData_->blockNumB;
    bNum_ = tilingData_->blockFactorB;
    sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bNum_ = tilingData_->b % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0) {
        sNum_ = tilingData_->s % tilingData_->blockFactorS;
    }
}

// =================================================================
// Process
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    PrePareParams();
    uint32_t bIdxStart = bIdx_ * tilingData_->blockFactorB;
    uint32_t sIdxStart = sIdx_ * tilingData_->blockFactorS;
    uint32_t sLoopCnt = Ops::Base::CeilDiv(sNum_, ubFactorS_);
    for (uint32_t bIdx = bIdxStart; bIdx < bIdxStart + bNum_; bIdx++) {
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? ubFactorS_ : sNum_ - loopIdx * ubFactorS_;
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum);
        }
    }
}

// =================================================================
// ProcessNLoop — 搬 cos/sin（整行搬），然后 Q/K 同时计算
//
// cos(N=1): S×D 全部连续 → blockCount=1, blockLen=currSNum*D, 单次 MTE 请求
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessNLoop(uint32_t bIdx, uint32_t sIdx, uint32_t currSNum)
{
    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    int64_t cosOffset = sIdx * tilingData_->d;
    uint32_t cosCopyLen = currSNum * tilingData_->d * sizeof(T);
    DataCopyExtParams copyCosParams{1, cosCopyLen, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[cosOffset], copyCosParams, padParams);
    DataCopyPad(cosTensor, cosGm_[cosOffset], copyCosParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();

    ProcessN(sinTensor, cosTensor, bIdx, sIdx, currSNum);

    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

// =================================================================
// ProcessN — 同一 N 循环内: Q/K 同时搬入 → 分别计算 → 同时搬出
//
// Q/K 各自独立 GM 偏移：Q stride = nQ * D, K stride = nK * D
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessN(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    uint32_t bIdx, uint32_t sIdx, uint32_t currSNum)
{
    LocalTensor<T> gradQIn;
    LocalTensor<T> gradKIn;
    LocalTensor<T> gradQOut;
    LocalTensor<T> gradKOut;
    LocalTensor<T> queryIn;
    LocalTensor<T> keyIn;

    int64_t dByteSize = tilingData_->d * sizeof(T);
    int64_t bSStride = bIdx * tilingData_->s + sIdx;

    // Q/K 各自独立 base offset：nQ 和 nK 可能不同
    int64_t qBaseOffset = bSStride * tilingData_->nQ * tilingData_->d;
    int64_t kBaseOffset = bSStride * tilingData_->nK * tilingData_->d;

    for (uint32_t idxN = 0; idxN < tilingData_->ubLoopNumN; idxN++) {
        int64_t currQN = (idxN == tilingData_->ubLoopNumN - 1) ?
            std::min(static_cast<int64_t>(tilingData_->ubTailFactorN), tilingData_->nQ - static_cast<int64_t>(idxN) * ubFactorN_) :
            std::min(ubFactorN_, tilingData_->nQ - static_cast<int64_t>(idxN) * ubFactorN_);
        int64_t currKN = (idxN == tilingData_->ubLoopNumN - 1) ?
            std::min(static_cast<int64_t>(tilingData_->ubTailFactorN), tilingData_->nK - static_cast<int64_t>(idxN) * ubFactorN_) :
            std::min(ubFactorN_, tilingData_->nK - static_cast<int64_t>(idxN) * ubFactorN_);
        if (currQN <= 0 && currKN <= 0) {
            break;
        }

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // ========== 同时搬入 Q 和 K 两路的 grad ==========
        if (currQN > 0) {
            gradQIn = gradQInQue_.AllocTensor<T>();
            int64_t qOffset = qBaseOffset + idxN * ubFactorN_ * tilingData_->d;
            uint32_t qBlockLen = currQN * dByteSize;
            uint32_t qSrcStride = (tilingData_->nQ - currQN) * dByteSize;
            DataCopyExtParams copyQParams{static_cast<uint16_t>(currSNum), qBlockLen, qSrcStride, 0, 0};
            DataCopyPad(gradQIn, gradQGm_[qOffset], copyQParams, padParams);
            gradQInQue_.EnQue(gradQIn);
        }
        if (currKN > 0) {
            gradKIn = gradKInQue_.AllocTensor<T>();
            int64_t kOffset = kBaseOffset + idxN * ubFactorN_ * tilingData_->d;
            uint32_t kBlockLen = currKN * dByteSize;
            uint32_t kSrcStride = (tilingData_->nK - currKN) * dByteSize;
            DataCopyExtParams copyKParams{static_cast<uint16_t>(currSNum), kBlockLen, kSrcStride, 0, 0};
            DataCopyPad(gradKIn, gradKGm_[kOffset], copyKParams, padParams);
            gradKInQue_.EnQue(gradKIn);
        }

        if (tilingData_->dCosFlag) {
            if (currQN > 0) {
                queryIn = queryInQue_.AllocTensor<T>();
                int64_t qOptOffset = qBaseOffset + idxN * ubFactorN_ * tilingData_->d;
                uint32_t qOptBlockLen = currQN * dByteSize;
                uint32_t qOptSrcStride = (tilingData_->nQ - currQN) * dByteSize;
                DataCopyExtParams copyQOptParams{static_cast<uint16_t>(currSNum), qOptBlockLen, qOptSrcStride, 0, 0};
                DataCopyPad(queryIn, queryGm_[qOptOffset], copyQOptParams, padParams);
                queryInQue_.EnQue(queryIn);
            }
            if (currKN > 0) {
                keyIn = keyInQue_.AllocTensor<T>();
                int64_t kOptOffset = kBaseOffset + idxN * ubFactorN_ * tilingData_->d;
                uint32_t kOptBlockLen = currKN * dByteSize;
                uint32_t kOptSrcStride = (tilingData_->nK - currKN) * dByteSize;
                DataCopyExtParams copyKOptParams{static_cast<uint16_t>(currSNum), kOptBlockLen, kOptSrcStride, 0, 0};
                DataCopyPad(keyIn, keyGm_[kOptOffset], copyKOptParams, padParams);
                keyInQue_.EnQue(keyIn);
            }
        }
                DataCopyExtParams copyKOptParams{static_cast<uint16_t>(currSNum), kOptBlockLen, kOptSrcStride, 0, 0};
                DataCopyPad(keyIn, keyGm_[nOffset], copyKOptParams, padParams);
                keyInQue_.EnQue(keyIn);
            }
        }

        // ========== DeQue ==========
        if (currQN > 0) gradQIn = gradQInQue_.DeQue<T>();
        if (currKN > 0) gradKIn = gradKInQue_.DeQue<T>();
        if (tilingData_->dCosFlag) {
            if (currQN > 0) queryIn = queryInQue_.DeQue<T>();
            if (currKN > 0) keyIn = keyInQue_.DeQue<T>();
        }

        // ========== Q 路 计算 + 搬出 ==========
        if (currQN > 0) {
            gradQOut = gradQOutQue_.AllocTensor<T>();
            ComputeOnePath(sinTensor, cosTensor, gradQIn, gradQOut, currSNum, currQN);
            gradQInQue_.FreeTensor(gradQIn);
            if (tilingData_->dCosFlag) queryInQue_.FreeTensor(queryIn);

            gradQOutQue_.EnQue(gradQOut);
            gradQOut = gradQOutQue_.DeQue<T>();
            int64_t qOffset = qBaseOffset + idxN * ubFactorN_ * tilingData_->d;
            uint32_t qOutBlockLen = currQN * dByteSize;
            uint32_t qOutDstStride = (tilingData_->nQ - currQN) * dByteSize;
            DataCopyExtParams copyOutQParams{static_cast<uint16_t>(currSNum), qOutBlockLen, 0, qOutDstStride, 0};
            DataCopyPad(gradQOutGm_[qOffset], gradQOut, copyOutQParams);
            gradQOutQue_.FreeTensor(gradQOut);
        }

        // ========== K 路 计算 + 搬出 ==========
        if (currKN > 0) {
            gradKOut = gradKOutQue_.AllocTensor<T>();
            ComputeOnePath(sinTensor, cosTensor, gradKIn, gradKOut, currSNum, currKN);
            gradKInQue_.FreeTensor(gradKIn);
            if (tilingData_->dCosFlag) keyInQue_.FreeTensor(keyIn);

            gradKOutQue_.EnQue(gradKOut);
            gradKOut = gradKOutQue_.DeQue<T>();
            int64_t kOffset = kBaseOffset + idxN * ubFactorN_ * tilingData_->d;
            uint32_t kOutBlockLen = currKN * dByteSize;
            uint32_t kOutDstStride = (tilingData_->nK - currKN) * dByteSize;
            DataCopyExtParams copyOutKParams{static_cast<uint16_t>(currSNum), kOutBlockLen, 0, kOutDstStride, 0};
            DataCopyPad(gradKOutGm_[kOffset], gradKOut, copyOutKParams);
            gradKOutQue_.FreeTensor(gradKOut);
        }
            DataCopyPad(gradKOutGm_[nOffset], gradKOut, copyOutKParams);
            gradKOutQue_.FreeTensor(gradKOut);
        }
    }
}

// =================================================================
// ComputeOnePath
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ComputeOnePath(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    const LocalTensor<T>& inTensor, const LocalTensor<T>& outTensor,
    uint32_t currSNum, uint32_t currDNum)
{
    HalfGradAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor,
                       tilingData_->d, dAlign_, currSNum, currDNum);
}

} // namespace ApplyRotaryPosEmbGrad
#endif // APPLY_ROTARY_POS_EMB_GRAD_BAB_H
