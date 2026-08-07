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
 * \file apply_rotary_pos_emb_grad_a.h
 * \brief A template kernel (NO_BROADCAST, N=1, cos==dy 全同)
 * MergeDim 将 B×N×S 合并为 b_, 逐 b_ 分核处理 Q 和 K
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_A_H
#define APPLY_ROTARY_POS_EMB_GRAD_A_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbGradA
{
public:
    __aicore__ inline ApplyRotaryPosEmbGradA(TPipe* pipe, const ApplyRopeGradRegbaseParams* td)
        : pipe_(pipe), tilingData_(td) {}
    __aicore__ inline ~ApplyRotaryPosEmbGradA() {}

    __aicore__ inline void Init(
        GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR grad_query, GM_ADDR grad_key);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(
        GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR grad_query, GM_ADDR grad_key);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void ProcessInLoop(LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t pos, int64_t len,
        GlobalTensor<T>& gGm, GlobalTensor<T>& oGm, int64_t nSize);

    __aicore__ inline void CopyInCosAndSin(int64_t pos, int64_t len);
    __aicore__ inline void CopyIn(GlobalTensor<T>& src, int64_t pos, int64_t len, int64_t nSize);
    __aicore__ inline void CopyOut(GlobalTensor<T>& dst, int64_t pos, int64_t len, int64_t nSize);
    __aicore__ inline void Compute(LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bL, int64_t nL);

private:
    TPipe* pipe_;
    const ApplyRopeGradRegbaseParams* tilingData_;
    GlobalTensor<T> gqGm_, gkGm_, cosGm_, sinGm_, oqGm_, okGm_;
    TQue<QuePosition::VECIN, 1> gInQ_, cosInQ_, sinInQ_;
    TQue<QuePosition::VECOUT, 1> oOutQ_;
    int64_t blkIdx_, bStart_, bLen_, ubB_, ubN_, D_, dBuffer_, dSplit_;
};

template <typename T>
void ApplyRotaryPosEmbGradA<T>::Init(
    GM_ADDR gq, GM_ADDR gk, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR oq, GM_ADDR ok)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) return;
    InitAllGlobalBuffer(gq, gk, cos, sin, oq, ok);
    InitAllBuffer();
    blkIdx_ = GetBlockIdx();
    bLen_ = tilingData_->blockFactorB;
    bStart_ = blkIdx_ * tilingData_->blockFactorB;
    if (blkIdx_ == static_cast<int64_t>(tilingData_->blockNumB) - 1 &&
        tilingData_->b % tilingData_->blockFactorB != 0)
        bLen_ = tilingData_->b % tilingData_->blockFactorB;
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::InitAllGlobalBuffer(
    GM_ADDR gq, GM_ADDR gk, GM_ADDR cos, GM_ADDR sin, GM_ADDR oq, GM_ADDR ok)
{
    gqGm_.SetGlobalBuffer((__gm__ T*)gq);
    gkGm_.SetGlobalBuffer((__gm__ T*)gk);
    cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    oqGm_.SetGlobalBuffer((__gm__ T*)oq);
    okGm_.SetGlobalBuffer((__gm__ T*)ok);
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::InitAllBuffer()
{
    ubB_ = tilingData_->ubFactorS; ubN_ = tilingData_->ubFactorN;
    D_ = tilingData_->d; dSplit_ = HALF_INTERLEAVE_COEF;
    dBuffer_ = CeilAlign<int64_t>(D_, BLOCK_TYPE_SIZE / sizeof(T));
    pipe_->InitBuffer(gInQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(oOutQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(cosInQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(sinInQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) return;
    int64_t ulc = CeilDiv(bLen_, ubB_);
    for (int64_t i = 0; i < ulc; i++) {
        int64_t pos = bStart_ + i * ubB_;
        int64_t len = (i != ulc - 1) ? ubB_ : bLen_ - i * ubB_;

        CopyInCosAndSin(pos, len);
        LocalTensor<T> cU = cosInQ_.DeQue<T>();
        LocalTensor<T> sU = sinInQ_.DeQue<T>();

        ProcessInLoop(cU, sU, pos, len, gqGm_, oqGm_, tilingData_->nQ);
        ProcessInLoop(cU, sU, pos, len, gkGm_, okGm_, tilingData_->nK);

        cosInQ_.FreeTensor(cU);
        sinInQ_.FreeTensor(sU);
    }
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::ProcessInLoop(
    LocalTensor<T>& cU, LocalTensor<T>& sU, int64_t pos, int64_t len,
    GlobalTensor<T>& gGm, GlobalTensor<T>& oGm, int64_t nSize)
{
    CopyIn(gGm, pos, len, nSize);
    Compute(cU, sU, len, nSize);
    CopyOut(oGm, pos, len, nSize);
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::CopyInCosAndSin(int64_t pos, int64_t len)
{
    LocalTensor<T> cU = cosInQ_.AllocTensor<T>();
    LocalTensor<T> sU = sinInQ_.AllocTensor<T>();
    DataCopyExtParams ep{1, static_cast<uint32_t>(len * D_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(cU, cosGm_[pos * D_], ep, pp);
    DataCopyPad(sU, sinGm_[pos * D_], ep, pp);
    cosInQ_.EnQue(cU); sinInQ_.EnQue(sU);
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::CopyIn(GlobalTensor<T>& src, int64_t pos, int64_t len, int64_t nSize)
{
    LocalTensor<T> t = gInQ_.AllocTensor<T>();
    DataCopyExtParams ep{1, static_cast<uint32_t>(len * nSize * D_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(t, src[pos * nSize * D_], ep, pp);
    gInQ_.EnQue(t);
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::CopyOut(GlobalTensor<T>& dst, int64_t pos, int64_t len, int64_t nSize)
{
    LocalTensor<T> s = oOutQ_.DeQue<T>();
    DataCopyExtParams ep{1, static_cast<uint32_t>(len * nSize * D_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[pos * nSize * D_], s, ep);
    oOutQ_.FreeTensor(s);
}

template <typename T>
void ApplyRotaryPosEmbGradA<T>::Compute(LocalTensor<T>& cU, LocalTensor<T>& sU, int64_t bL, int64_t nL)
{
    LocalTensor<T> in = gInQ_.DeQue<T>();
    LocalTensor<T> out = oOutQ_.AllocTensor<T>();
    BatchHalfGradContiguousVF<T, false>(
        (__local_mem__ T*)in.GetPhyAddr(), (__local_mem__ T*)cU.GetPhyAddr(),
        (__local_mem__ T*)sU.GetPhyAddr(), (__local_mem__ T*)out.GetPhyAddr(),
        1, bL, nL, D_, ubB_, ubN_);
    gInQ_.FreeTensor(in);
    oOutQ_.EnQue(out);
}

} // namespace ApplyRotaryPosEmbGrad
#endif
