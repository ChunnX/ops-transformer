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
 * \file apply_rotary_pos_emb_grad_ab.h
 * \brief AB split kernel（SBND layout）
 * 双路反向：同时处理 Q 和 K
 * 参照 rotary_position_embedding_grad_ab.h
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_AB_H
#define APPLY_ROTARY_POS_EMB_GRAD_AB_H

#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbGradAB
{
public:
    __aicore__ inline ApplyRotaryPosEmbGradAB(TPipe* pipe, const ApplyRopeGradRegbaseABParams* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(
        GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR dxQOut, GM_ADDR dxKOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(
        int64_t gradGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer,
        int64_t ubIdx, int64_t bsCount, int64_t nCount,
        GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> dxOutQueue_;

    GlobalTensor<T> gradQGm_, gradKGm_;
    GlobalTensor<T> cosGm_, sinGm_;
    GlobalTensor<T> dxQGm_, dxKGm_;
    const ApplyRopeGradRegbaseABParams* tilingData_;
    DataCopyPadExtParams<T> padParams_{false, 0, 0, static_cast<T>(0)};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSize_ = 0;
    int64_t bsBlockCount_ = 0;
    int64_t nBlockCount_ = 0;
    int64_t maxN_ = 0;
};

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T>::Init(
    GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dxQOut, GM_ADDR dxKOut)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    dSplitSize_ = tilingData_->d / tilingData_->dSplitCoef * sizeof(T);
    maxN_ = (tilingData_->nQ > tilingData_->nK) ? tilingData_->nQ : tilingData_->nK;
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    bsBlockCount_ = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->d;
    cosGm_.SetGlobalBuffer((__gm__ T*)cos + cosOffset);
    sinGm_.SetGlobalBuffer((__gm__ T*)sin + cosOffset);
    // grad 的 Q/K 路径共享相同的 BS offset，但 N 维度不同
    int64_t gradQOffset = cosOffset * tilingData_->nQ + blockDimN * tilingData_->blockFactorN * tilingData_->d;
    int64_t gradKOffset = cosOffset * tilingData_->nK + blockDimN * tilingData_->blockFactorN * tilingData_->d;
    gradQGm_.SetGlobalBuffer((__gm__ T*)gradQ + gradQOffset);
    gradKGm_.SetGlobalBuffer((__gm__ T*)gradK + gradKOffset);
    dxQGm_.SetGlobalBuffer((__gm__ T*)dxQOut + gradQOffset);
    dxKGm_.SetGlobalBuffer((__gm__ T*)dxKOut + gradKOffset);

    int64_t bufferSize = tilingData_->dAlign * sizeof(T) * tilingData_->ubFactorBS;
    pipe_->InitBuffer(gradInQueue_, DB_FLAG, bufferSize * tilingData_->ubFactorN);
    pipe_->InitBuffer(cosInQueue_, DB_FLAG, bufferSize);
    pipe_->InitBuffer(sinInQueue_, DB_FLAG, bufferSize);
    pipe_->InitBuffer(dxOutQueue_, DB_FLAG, bufferSize * tilingData_->ubFactorN);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    uint32_t bsLoopCnt = Ops::Base::CeilDiv(bsBlockCount_, tilingData_->ubFactorBS);
    uint32_t nLoopCnt = Ops::Base::CeilDiv(nBlockCount_, tilingData_->ubFactorN);
    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        int64_t xGmOffset = bsLoopIdx * tilingData_->ubFactorBS * maxN_ * tilingData_->d;
        uint32_t currBSNum = (bsLoopIdx != bsLoopCnt - 1) ? tilingData_->ubFactorBS :
                                                            bsBlockCount_ - (bsLoopIdx * tilingData_->ubFactorBS);

        DataCopyExtParams cosParams = {static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};
        LocalTensor<T> cosBuffer = cosInQueue_.AllocTensor<T>();
        LocalTensor<T> sinBuffer = sinInQueue_.AllocTensor<T>();
        DataCopyPad(cosBuffer, cosGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->d], cosParams, padParams_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.DeQue<T>();
        DataCopyPad(sinBuffer, sinGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->d], cosParams, padParams_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.DeQue<T>();

        for (int64_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
            int64_t currNNum = (nLoopIdx != nLoopCnt - 1) ? tilingData_->ubFactorN :
                                                            nBlockCount_ - (nLoopIdx * tilingData_->ubFactorN);
            // === Q Path ===
            int64_t actualNQ = Ops::Base::Min(currNNum, tilingData_->nQ - nLoopIdx * tilingData_->ubFactorN);
            if (actualNQ > 0) {
                ProcessLoop(xGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, actualNQ, gradQGm_, dxQGm_, tilingData_->nQ);
            }
            // === K Path ===
            int64_t actualNK = Ops::Base::Min(currNNum, tilingData_->nK - nLoopIdx * tilingData_->ubFactorN);
            if (actualNK > 0) {
                ProcessLoop(xGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, actualNK, gradKGm_, dxKGm_, tilingData_->nK);
            }
        }

        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T>::ProcessLoop(
    int64_t gradGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer,
    int64_t ubIdx, int64_t bsCount, int64_t nCount,
    GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize)
{
    int64_t totalCount = bsCount * nCount;
    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};

    LocalTensor<T> inBuffer = gradInQueue_.AllocTensor<T>();
    LocalTensor<T> outBuffer = dxOutQueue_.AllocTensor<T>();
    DataCopyPad(inBuffer, gradGm[gradGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->d], inParams, padParams_);
    gradInQueue_.EnQue(inBuffer);
    inBuffer = gradInQueue_.DeQue<T>();

    HalfGradAlignVF(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->d, tilingData_->dAlign, bsCount, nCount);

    dxOutQueue_.EnQue(outBuffer);
    outBuffer = dxOutQueue_.DeQue<T>();
    gradInQueue_.FreeTensor(inBuffer);
    DataCopyPad(dxGm[gradGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->d], outBuffer, outParams);
    dxOutQueue_.FreeTensor(outBuffer);
}

} // namespace ApplyRotaryPosEmbGrad

#endif // APPLY_ROTARY_POS_EMB_GRAD_AB_H
