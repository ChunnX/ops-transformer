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
 * \file moe_tutel_combine_gates_regbase.h
 * \brief MoeTutelCombineGates implementation for Ascend 950.
 */
#ifndef MOE_TUTEL_COMBINE_GATES_REGBASE_H_
#define MOE_TUTEL_COMBINE_GATES_REGBASE_H_

#include "moe_tutel_combine_gates_common.h"

namespace MoeTutelCombineGates {
using namespace AscendC;

template <typename T, bool CUT_H>
class MoeTutelCombineGatesRegbase {
public:
    __aicore__ inline MoeTutelCombineGatesRegbase() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR yGrad, GM_ADDR indices, GM_ADDR locations, GM_ADDR gatesGrad,
                                const MoeTutelCombineGatesTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInX(int64_t sample, int64_t hOffset, uint32_t len);
    __aicore__ inline void CopyInY(int64_t row, int64_t hOffset, uint32_t len);
    __aicore__ inline void ComputeOnePair(int64_t sample, int64_t sInTile, int64_t kInTile, int64_t routeOffset,
                                          LocalTensor<T> &xUb);
    __aicore__ inline void ComputeTile(int64_t kBase, int64_t curK, int64_t sBase, int64_t curS);
    __aicore__ inline void CopyOut(int64_t kBase, int64_t curK, int64_t sBase, int64_t curS);
    __aicore__ inline uint32_t ChunkLen(int64_t chunk) const;

private:
    TPipe *pipe_ = nullptr;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGradGm_;
    GlobalTensor<T> gatesGradGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> locationsGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> yInQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> outFp32Buf_;
    TBuf<TPosition::VECCALC> vlPartialBuf_;
    TBuf<TPosition::VECCALC> chunkPartialBuf_;

    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t experts_ = 0;
    int64_t capacity_ = 0;
    int64_t routes_ = 0;
    int64_t sBlock_ = 0;
    int64_t sBlockAlign_ = 0;
    int64_t kBlock_ = 0;
    int64_t hFactor_ = 0;
    int64_t hFactorAlign_ = 0;
    int64_t numChunks_ = 0;
    int64_t hTail_ = 0;
    int64_t vlPartialAlign_ = 0;
    int64_t chunkPartialAlign_ = 0;

    CoreRange sRange_ = {0, 0};
    CoreRange kRange_ = {0, 0};
};

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::Init(GM_ADDR x, GM_ADDR yGrad, GM_ADDR indices,
                                                                   GM_ADDR locations, GM_ADDR gatesGrad,
                                                                   const MoeTutelCombineGatesTilingData *tilingData,
                                                                   TPipe *pipe)
{
    pipe_ = pipe;
    samples_ = static_cast<int64_t>(tilingData->samples);
    hidden_ = static_cast<int64_t>(tilingData->hidden);
    experts_ = static_cast<int64_t>(tilingData->experts);
    capacity_ = static_cast<int64_t>(tilingData->capacity);
    routes_ = static_cast<int64_t>(tilingData->routes);
    sBlock_ = static_cast<int64_t>(tilingData->sBlock);
    sBlockAlign_ = static_cast<int64_t>(tilingData->sBlockAlign);
    kBlock_ = static_cast<int64_t>(tilingData->kBlock);
    hFactor_ = static_cast<int64_t>(tilingData->hFactor);
    hFactorAlign_ = static_cast<int64_t>(tilingData->hFactorAlign);
    numChunks_ = static_cast<int64_t>(tilingData->numChunks);
    hTail_ = static_cast<int64_t>(tilingData->hTail);
    vlPartialAlign_ = static_cast<int64_t>(tilingData->vlPartialAlign);
    chunkPartialAlign_ = static_cast<int64_t>(tilingData->chunkPartialAlign);

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t sCoreNum = static_cast<int64_t>(tilingData->sCoreNum);
    sRange_ = SolveCoreRange(blockIdx % sCoreNum, static_cast<int64_t>(tilingData->sPerCoreBase),
                             static_cast<int64_t>(tilingData->sCoreRemain));
    kRange_ = SolveCoreRange(blockIdx / sCoreNum, static_cast<int64_t>(tilingData->kPerCoreBase),
                             static_cast<int64_t>(tilingData->kCoreRemain));

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), samples_ * hidden_);
    yGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(yGrad), experts_ * capacity_ * hidden_);
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), routes_ * samples_);
    locationsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(locations), routes_ * samples_);
    gatesGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gatesGrad), routes_ * samples_);

    if (hidden_ > 0) {
        const uint32_t rowBytes = static_cast<uint32_t>(hFactorAlign_ * static_cast<int64_t>(sizeof(T)));
        pipe_->InitBuffer(xInQueue_, DOUBLE_BUFFER, rowBytes);
        pipe_->InitBuffer(yInQueue_, DOUBLE_BUFFER, rowBytes);
        pipe_->InitBuffer(vlPartialBuf_, static_cast<uint32_t>(vlPartialAlign_ * sizeof(float)));
        if constexpr (CUT_H) {
            pipe_->InitBuffer(chunkPartialBuf_, static_cast<uint32_t>(chunkPartialAlign_ * sizeof(float)));
        }
    }
    pipe_->InitBuffer(outQueue_, 1, static_cast<uint32_t>(kBlock_ * sBlockAlign_ * static_cast<int64_t>(sizeof(T))));
    pipe_->InitBuffer(outFp32Buf_, static_cast<uint32_t>(kBlock_ * sBlockAlign_ * static_cast<int64_t>(sizeof(float))));
}

template <typename T, bool CUT_H>
__aicore__ inline uint32_t MoeTutelCombineGatesRegbase<T, CUT_H>::ChunkLen(int64_t chunk) const
{
    if constexpr (CUT_H) {
        return static_cast<uint32_t>((chunk + 1 == numChunks_) ? hTail_ : hFactor_);
    }
    return static_cast<uint32_t>(hidden_);
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::CopyInX(int64_t sample, int64_t hOffset, uint32_t len)
{
    LocalTensor<T> xUb = xInQueue_.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[sample * hidden_ + hOffset], copyParams, padParams);
    xInQueue_.template EnQue(xUb);
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::CopyInY(int64_t row, int64_t hOffset, uint32_t len)
{
    LocalTensor<T> yUb = yInQueue_.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(yUb, yGradGm_[row * hidden_ + hOffset], copyParams, padParams);
    yInQueue_.template EnQue(yUb);
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::ComputeOnePair(int64_t sample, int64_t sInTile,
                                                                             int64_t kInTile, int64_t routeOffset,
                                                                             LocalTensor<T> &xUb)
{
    const int32_t expertId = indicesGm_.GetValue(routeOffset);
    const int32_t location = locationsGm_.GetValue(routeOffset);
    const bool valid = expertId >= 0 && static_cast<int64_t>(expertId) < experts_ && location >= 0 &&
                       static_cast<int64_t>(location) < capacity_;
    if (!valid || hidden_ == 0) {
        return;
    }

    __ubuf__ float *outFp32 = reinterpret_cast<__ubuf__ float *>(outFp32Buf_.Get<float>().GetPhyAddr());
    const uint32_t outOffset = static_cast<uint32_t>(kInTile * sBlockAlign_ + sInTile);
    const int64_t row = static_cast<int64_t>(expertId) * capacity_ + static_cast<int64_t>(location);
    LocalTensor<float> vlPartial = vlPartialBuf_.Get<float>();
    __ubuf__ float *vlPartialPtr = reinterpret_cast<__ubuf__ float *>(vlPartial.GetPhyAddr());

    if constexpr (CUT_H) {
        LocalTensor<float> chunkPartial = chunkPartialBuf_.Get<float>();
        Duplicate(chunkPartial, 0.0f, static_cast<int32_t>(chunkPartialAlign_));
        __ubuf__ float *chunkPartialPtr = reinterpret_cast<__ubuf__ float *>(chunkPartial.GetPhyAddr());
        for (int64_t chunk = 0; chunk < numChunks_; chunk++) {
            Duplicate(vlPartial, 0.0f, static_cast<int32_t>(vlPartialAlign_));
            const uint32_t len = ChunkLen(chunk);
            const int64_t hOffset = chunk * hFactor_;
            CopyInX(sample, hOffset, len);
            CopyInY(row, hOffset, len);
            LocalTensor<T> xChunk = xInQueue_.template DeQue<T>();
            LocalTensor<T> yChunk = yInQueue_.template DeQue<T>();
            BuildDotPartialsVF<T>(vlPartialPtr, reinterpret_cast<__ubuf__ T *>(xChunk.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ T *>(yChunk.GetPhyAddr()), len);
            xInQueue_.FreeTensor(xChunk);
            yInQueue_.FreeTensor(yChunk);
            ReducePartialTreeVF(vlPartialPtr, static_cast<uint32_t>(vlPartialAlign_), chunkPartialPtr,
                                static_cast<uint32_t>(chunk));
        }
        ReducePartialTreeVF(chunkPartialPtr, static_cast<uint32_t>(chunkPartialAlign_), outFp32, outOffset);
    } else {
        Duplicate(vlPartial, 0.0f, static_cast<int32_t>(vlPartialAlign_));
        CopyInY(row, 0, static_cast<uint32_t>(hidden_));
        LocalTensor<T> yUb = yInQueue_.template DeQue<T>();
        BuildDotPartialsVF<T>(vlPartialPtr, reinterpret_cast<__ubuf__ T *>(xUb.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ T *>(yUb.GetPhyAddr()), static_cast<uint32_t>(hidden_));
        yInQueue_.FreeTensor(yUb);
        ReducePartialTreeVF(vlPartialPtr, static_cast<uint32_t>(vlPartialAlign_), outFp32, outOffset);
    }
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::ComputeTile(int64_t kBase, int64_t curK, int64_t sBase,
                                                                          int64_t curS)
{
    LocalTensor<float> outFp32 = outFp32Buf_.Get<float>();
    Duplicate(outFp32, 0.0f, static_cast<int32_t>(kBlock_ * sBlockAlign_));

    for (int64_t i = 0; i < curS; i++) {
        const int64_t sample = sBase + i;
        LocalTensor<T> xUb;
        if constexpr (!CUT_H) {
            if (hidden_ > 0) {
                CopyInX(sample, 0, static_cast<uint32_t>(hidden_));
                xUb = xInQueue_.template DeQue<T>();
            }
        }
        for (int64_t kk = 0; kk < curK; kk++) {
            const int64_t k = kBase + kk;
            ComputeOnePair(sample, i, kk, k * samples_ + sample, xUb);
        }
        if constexpr (!CUT_H) {
            if (hidden_ > 0) {
                xInQueue_.FreeTensor(xUb);
            }
        }
    }
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::CopyOut(int64_t kBase, int64_t curK, int64_t sBase,
                                                                      int64_t curS)
{
    __ubuf__ float *outFp32 = reinterpret_cast<__ubuf__ float *>(outFp32Buf_.Get<float>().GetPhyAddr());
    LocalTensor<T> outUb = outQueue_.template AllocTensor<T>();
    __ubuf__ T *outPtr = reinterpret_cast<__ubuf__ T *>(outUb.GetPhyAddr());
    for (int64_t kk = 0; kk < curK; kk++) {
        CastRowVF<T>(outPtr + kk * sBlockAlign_, outFp32 + kk * sBlockAlign_, static_cast<uint32_t>(curS));
    }
    outQueue_.template EnQue(outUb);

    LocalTensor<T> outLocal = outQueue_.template DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curS * sizeof(T)), 0, 0, 0};
    for (int64_t kk = 0; kk < curK; kk++) {
        const int64_t k = kBase + kk;
        DataCopyPad(gatesGradGm_[k * samples_ + sBase], outLocal[kk * sBlockAlign_], copyParams);
    }
    outQueue_.FreeTensor(outLocal);
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::Process()
{
    if (sRange_.len <= 0 || kRange_.len <= 0) {
        return;
    }
    const int64_t sEnd = sRange_.start + sRange_.len;
    const int64_t kEnd = kRange_.start + kRange_.len;
    for (int64_t kBase = kRange_.start; kBase < kEnd; kBase += kBlock_) {
        const int64_t curK = (kBase + kBlock_ <= kEnd) ? kBlock_ : (kEnd - kBase);
        for (int64_t sBase = sRange_.start; sBase < sEnd; sBase += sBlock_) {
            const int64_t curS = (sBase + sBlock_ <= sEnd) ? sBlock_ : (sEnd - sBase);
            ComputeTile(kBase, curK, sBase, curS);
            CopyOut(kBase, curK, sBase, curS);
        }
    }
}
} // namespace MoeTutelCombineGates

#endif // MOE_TUTEL_COMBINE_GATES_REGBASE_H_
