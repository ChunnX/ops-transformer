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
 * 核内: tile-interleaved Q/K 交叉处理, cos/sin 部分积 UB 内累加
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
        int64_t gmOffset, int64_t nPath, int64_t currXN, uint32_t currSNum,
        LocalTensor<T>& cosPartial, LocalTensor<T>& sinPartial, bool isQPath);
    __aicore__ inline void ComputeOnePath(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        const LocalTensor<T>& inTensor, const LocalTensor<T>& outTensor,
        uint32_t currSNum, uint32_t currDNum);
    __aicore__ inline void ComputeGradWithCosSinVF(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        const LocalTensor<T>& gradIn, const LocalTensor<T>& gradOut,
        const LocalTensor<T>& xIn,
        const LocalTensor<T>& cosPartial, const LocalTensor<T>& sinPartial,
        uint32_t dLen, uint32_t dAlign, uint16_t currSNum, uint16_t currDNum, bool isQPath);
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
// ProcessNLoop — 搬 cos/sin, tile-interleaved Q/K
//
// 同一 tile 内 Q 算完 cosPartial→UB → K 直接在 UB 累加 → 写一次 GM
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessNLoop(uint32_t bIdx, uint32_t sIdx, uint32_t currSNum)
{
    int64_t dByteSize = tilingData_->d * sizeof(T);
    int64_t bSStride = bIdx * tilingData_->s + sIdx;
    int64_t maxN = std::max(tilingData_->nQ, tilingData_->nK);
    int64_t cosWsBase = bSStride * maxN * tilingData_->d;

    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    DataCopyExtParams copyCosParams;
    if (tilingData_->d % (BLOCK_TYPE_SIZE / sizeof(T)) == 0) {
        copyCosParams = {1, currSNum * static_cast<uint32_t>(tilingData_->d * sizeof(T)), 0, 0, 0};
    } else {
        copyCosParams = {currSNum, static_cast<uint32_t>(tilingData_->d * sizeof(T)), 0, 0, 0};
    }
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[sIdx * tilingData_->d], copyCosParams, padParams);
    DataCopyPad(cosTensor, cosGm_[sIdx * tilingData_->d], copyCosParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();

    for (uint32_t idxN = 0; idxN < tilingData_->ubLoopNumN; idxN++) {
        int64_t currQN = (idxN == tilingData_->ubLoopNumN - 1) ?
            AscendC::Std::min(static_cast<int64_t>(tilingData_->ubTailFactorN), tilingData_->nQ - static_cast<int64_t>(idxN) * ubFactorN_) :
            AscendC::Std::min(ubFactorN_, tilingData_->nQ - static_cast<int64_t>(idxN) * ubFactorN_);
        int64_t currKN = (idxN == tilingData_->ubLoopNumN - 1) ?
            AscendC::Std::min(static_cast<int64_t>(tilingData_->ubTailFactorN), tilingData_->nK - static_cast<int64_t>(idxN) * ubFactorN_) :
            AscendC::Std::min(ubFactorN_, tilingData_->nK - static_cast<int64_t>(idxN) * ubFactorN_);
        if (currQN <= 0 && currKN <= 0) break;

        LocalTensor<T> cosPartial, sinPartial;
        if (tilingData_->dCosFlag) {
            cosPartial = gradCosOutQue_.AllocTensor<T>();
            sinPartial = gradSinOutQue_.AllocTensor<T>();
        }

        // Q tile
        if (currQN > 0) {
            int64_t qBaseOffset = bSStride * tilingData_->nQ * tilingData_->d + idxN * ubFactorN_ * tilingData_->d;
            ProcessN(sinTensor, cosTensor, qBaseOffset, tilingData_->nQ, currQN, currSNum,
                     cosPartial, sinPartial, true);
        }

        // K tile
        if (currKN > 0) {
            int64_t kBaseOffset = bSStride * tilingData_->nK * tilingData_->d + idxN * ubFactorN_ * tilingData_->d;
            ProcessN(sinTensor, cosTensor, kBaseOffset, tilingData_->nK, currKN, currSNum,
                     cosPartial, sinPartial, false);
        }

        // Write cos/sin to workspace (Q+K already accumulated in UB)
        if (tilingData_->dCosFlag && (currQN > 0 || currKN > 0)) {
            int64_t wsOffset = cosWsBase + idxN * ubFactorN_ * tilingData_->d;
            uint32_t wsCurrN = (currQN > 0) ? currQN : currKN;
            DataCopyExtParams copyWsParams;
            if (tilingData_->d % (BLOCK_TYPE_SIZE / sizeof(T)) == 0) {
                copyWsParams = {static_cast<uint16_t>(currSNum),
                                static_cast<uint32_t>(wsCurrN * dByteSize),
                                0, static_cast<uint32_t>((maxN - wsCurrN) * dByteSize), 0};
            } else {
                copyWsParams = {static_cast<uint16_t>(currSNum * wsCurrN),
                                static_cast<uint32_t>(dByteSize),
                                0, static_cast<uint32_t>((maxN - wsCurrN) * dByteSize), 0};
            }
            DataCopyPad(gradCosOutGm_[wsOffset], cosPartial, copyWsParams);
            DataCopyPad(gradSinOutGm_[wsOffset], sinPartial, copyWsParams);
            gradCosOutQue_.FreeTensor(cosPartial);
            gradSinOutQue_.FreeTensor(sinPartial);
        }
    }

    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

// =================================================================
// ProcessN — 单 tile 处理, CopyIn → Compute → CopyOut
//
// Q (isQPath=true):  cosPartial/sinPartial 为 SET 模式 (初始化)
// K (isQPath=false): cosPartial/sinPartial 为 ACCUMULATE 模式 (读取 UB 中 Q 值累加)
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessN(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    int64_t gmOffset, int64_t nPath, int64_t currXN, uint32_t currSNum,
    LocalTensor<T>& cosPartial, LocalTensor<T>& sinPartial, bool isQPath)
{
    int64_t dByteSize = tilingData_->d * sizeof(T);
    uint32_t stride = (nPath - currXN) * dByteSize;
    uint32_t blockLen = currXN * dByteSize;

    auto& gradXGm = isQPath ? gradQGm_ : gradKGm_;
    auto& gradXOutGm = isQPath ? gradQOutGm_ : gradKOutGm_;
    auto& gradXInQue = isQPath ? gradQInQue_ : gradKInQue_;
    auto& gradXOutQue = isQPath ? gradQOutQue_ : gradKOutQue_;
    auto& xGm = isQPath ? queryGm_ : keyGm_;
    auto& xInQue = isQPath ? queryInQue_ : keyInQue_;

    // CopyIn: gradX
    LocalTensor<T> gradXIn = gradXInQue.AllocTensor<T>();
    DataCopyExtParams copyParams;
    if (tilingData_->d % (BLOCK_TYPE_SIZE / sizeof(T)) == 0) {
        copyParams = {static_cast<uint16_t>(currSNum), blockLen, stride, 0, 0};
    } else {
        copyParams = {static_cast<uint16_t>(currSNum * currXN), dByteSize, stride, 0, 0};
    }
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(gradXIn, gradXGm[gmOffset], copyParams, padParams);
    gradXInQue.EnQue(gradXIn);
    gradXIn = gradXInQue.DeQue<T>();
    LocalTensor<T> gradXOut = gradXOutQue.AllocTensor<T>();

    // CopyIn: x (query/key, dCosFlag only) and Compute
    LocalTensor<T> xIn;
    if (tilingData_->dCosFlag) {
        xIn = xInQue.AllocTensor<T>();
        DataCopyExtParams copyXParams;
        if (tilingData_->d % (BLOCK_TYPE_SIZE / sizeof(T)) == 0) {
            copyXParams = {static_cast<uint16_t>(currSNum), blockLen, stride, 0, 0};
        } else {
            copyXParams = {static_cast<uint16_t>(currSNum * currXN), dByteSize, stride, 0, 0};
        }
        DataCopyPad(xIn, xGm[gmOffset], copyXParams, padParams);
        xInQue.EnQue(xIn);
        xIn = xInQue.DeQue<T>();
        ComputeGradWithCosSinVF(sinTensor, cosTensor, gradXIn, gradXOut, xIn, cosPartial, sinPartial,
            tilingData_->d, dAlign_, currSNum, currXN, isQPath);
    } else {
        ComputeOnePath(sinTensor, cosTensor, gradXIn, gradXOut, currSNum, currXN);
    }

    // CopyOut gradX
    gradXOutQue.EnQue(gradXOut);
    gradXOut = gradXOutQue.DeQue<T>();
    DataCopyExtParams copyOutParams;
    if (tilingData_->d % (BLOCK_TYPE_SIZE / sizeof(T)) == 0) {
        copyOutParams = {static_cast<uint16_t>(currSNum), blockLen, 0, stride, 0};
    } else {
        copyOutParams = {static_cast<uint16_t>(currSNum * currXN), dByteSize, 0, stride, 0};
    }
    DataCopyPad(gradXOutGm[gmOffset], gradXOut, copyOutParams);
    gradXOutQue.FreeTensor(gradXOut);

    // Free inputs (cosPartial/sinPartial stay in UB, managed by caller)
    gradXInQue.FreeTensor(gradXIn);
    if (tilingData_->dCosFlag) xInQue.FreeTensor(xIn);
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
    HalfGradAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->d, dAlign_, currSNum, currDNum);
}

// =================================================================
// ComputeGradWithCosSinVF — 合并 grad + cos/sin 计算
//
// isQPath=true (SET):   cosPartial = grad * x,  sinPartial = grad * rot(x)
// isQPath=false (ADD):  cosPartial += grad * x,  sinPartial += grad * rot(x)
// =================================================================
template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ComputeGradWithCosSinVF(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor, const LocalTensor<T>& gradIn, const LocalTensor<T>& gradOut,
    const LocalTensor<T>& xIn, const LocalTensor<T>& cosPartial, const LocalTensor<T>& sinPartial,
    uint32_t dLen, uint32_t dAlign, uint16_t currSNum, uint16_t currDNum, bool isQPath)
{
    __local_mem__ T* sinUb = (__local_mem__ T*)sinTensor.GetPhyAddr();
    __local_mem__ T* cosUb = (__local_mem__ T*)cosTensor.GetPhyAddr();
    __local_mem__ T* gradUb = (__local_mem__ T*)gradIn.GetPhyAddr();
    __local_mem__ T* gradOutUb = (__local_mem__ T*)gradOut.GetPhyAddr();
    __local_mem__ T* xUb = (__local_mem__ T*)xIn.GetPhyAddr();
    __local_mem__ T* cosPartialUb = (__local_mem__ T*)cosPartial.GetPhyAddr();
    __local_mem__ T* sinPartialUb = (__local_mem__ T*)sinPartial.GetPhyAddr();
    uint32_t halfD = dLen / HALF_COEF;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __local_mem__ T *currGradUb, *currGradOutUb, *currSinUb, *currCosUb, *currXUb, *currCosPartialUb, *currSinPartialUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregGrad1, vregGrad2, vregCos1, vregCos2, vregSin1, vregSin2, vregX1, vregX2;
        Reg::RegTensor<float> vregTmp1, vregTmp2, vregTmp3, vregTmp4, vregTmp5, vregTmp6, vregTmp7, vregTmp8;
        Reg::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currGradUb = gradUb + (sIdx * currDNum + row) * dAlign;
                currGradOutUb = gradOutUb + (sIdx * currDNum + row) * dAlign;
                currXUb = xUb + (sIdx * currDNum + row) * dAlign;
                currCosPartialUb = cosPartialUb + (sIdx * currDNum + row) * dAlign;
                currSinPartialUb = sinPartialUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = Reg::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlign;

                    ops::LoadTwoTensorForDtypeT<T>(
                        currGradUb, currGradUb, vregGrad1, vregGrad2, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos1, vregCos2, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin1, vregSin2, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currXUb, currXUb, vregX1, vregX2, preg, preg, offset, halfOffset);

                    // === gradOut: tmp1-4 ===
                    Mul(vregTmp1, vregCos1, vregGrad1, preg);
                    Mul(vregTmp2, vregSin2, vregGrad2, preg);
                    Add(vregTmp1, vregTmp1, vregTmp2, preg);
                    Mul(vregTmp3, vregCos2, vregGrad2, preg);
                    Mul(vregTmp4, vregSin1, vregGrad1, preg);
                    Sub(vregTmp3, vregTmp3, vregTmp4, preg);

                    // === cos/sin partial: tmp5-8 ===
                    Mul(vregTmp5, vregGrad1, vregX1, preg);
                    Mul(vregTmp6, vregGrad2, vregX2, preg);
                    Mul(vregTmp7, vregGrad1, vregX2, preg);
                    Muls(vregTmp7, vregTmp7, -1.0f, preg);
                    Mul(vregTmp8, vregGrad2, vregX1, preg);

                    if (isQPath) {
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp3, preg, halfOffset);
                        ops::StoreOneTensorForDtypeT<T>(currCosPartialUb, vregTmp5, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currCosPartialUb, vregTmp6, preg, halfOffset);
                        ops::StoreOneTensorForDtypeT<T>(currSinPartialUb, vregTmp7, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currSinPartialUb, vregTmp8, preg, halfOffset);
                    } else {
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp3, preg, halfOffset);

                        ops::LoadTwoTensorForDtypeT<T>(
                            currCosPartialUb, currCosPartialUb, vregCos1, vregCos2, preg, preg, offset, halfOffset);
                        Add(vregCos1, vregCos1, vregTmp5, preg);
                        Add(vregCos2, vregCos2, vregTmp6, preg);
                        ops::StoreOneTensorForDtypeT<T>(currCosPartialUb, vregCos1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currCosPartialUb, vregCos2, preg, halfOffset);

                        ops::LoadTwoTensorForDtypeT<T>(
                            currSinPartialUb, currSinPartialUb, vregSin1, vregSin2, preg, preg, offset, halfOffset);
                        Add(vregSin1, vregSin1, vregTmp7, preg);
                        Add(vregSin2, vregSin2, vregTmp8, preg);
                        ops::StoreOneTensorForDtypeT<T>(currSinPartialUb, vregSin1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currSinPartialUb, vregSin2, preg, halfOffset);
                    }
                }
            }
        }
    }
}
} // namespace ApplyRotaryPosEmbGrad
#endif // APPLY_ROTARY_POS_EMB_GRAD_BAB_H
