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
 * \brief MoeTutelCombineGates arch35 实现（NOT_CUT_H / CUT_H 两条模板由 CUT_H 参数区分）
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
    // 单个 (s, k) 的输出：合法则点积，非法则保持 0（tile 已整片清零）。
    __aicore__ inline void ComputeOnePair(int64_t sample, int64_t sInTile, int64_t kInTile, int64_t routeIdx,
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
    TBuf<TPosition::VECCALC> partialBuf_;

    // TilingData 展开
    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t experts_ = 0;
    int64_t capacity_ = 0;
    int64_t sBlock_ = 0;
    int64_t sBlockAlign_ = 0;
    int64_t kBlock_ = 0;
    int64_t hFactor_ = 0;
    int64_t hFactorAlign_ = 0;
    int64_t numChunks_ = 0;
    int64_t hTail_ = 0;

    // 本核负责的区间
    CoreRange sRange_ = {0, 0};
    CoreRange kRange_ = {0, 0};
    int64_t partialAlign_ = 0; // CUT_H 的部分和缓冲长度（VL 对齐）
};

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::Init(
    GM_ADDR x, GM_ADDR yGrad, GM_ADDR indices, GM_ADDR locations, GM_ADDR gatesGrad,
    const MoeTutelCombineGatesTilingData *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    samples_ = static_cast<int64_t>(tilingData->samples);
    hidden_ = static_cast<int64_t>(tilingData->hidden);
    experts_ = static_cast<int64_t>(tilingData->experts);
    capacity_ = static_cast<int64_t>(tilingData->capacity);
    sBlock_ = static_cast<int64_t>(tilingData->sBlock);
    sBlockAlign_ = static_cast<int64_t>(tilingData->sBlockAlign);
    kBlock_ = static_cast<int64_t>(tilingData->kBlock);
    hFactor_ = static_cast<int64_t>(tilingData->hFactor);
    hFactorAlign_ = static_cast<int64_t>(tilingData->hFactorAlign);
    numChunks_ = static_cast<int64_t>(tilingData->numChunks);
    hTail_ = static_cast<int64_t>(tilingData->hTail);

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t sCoreNum = static_cast<int64_t>(tilingData->sCoreNum);
    sRange_ = SolveCoreRange(blockIdx % sCoreNum, static_cast<int64_t>(tilingData->sPerCoreBase),
                             static_cast<int64_t>(tilingData->sCoreRemain));
    kRange_ = SolveCoreRange(blockIdx / sCoreNum, static_cast<int64_t>(tilingData->kPerCoreBase),
                             static_cast<int64_t>(tilingData->kCoreRemain));

    // GM 视图全部按 int64 计算元素数，E*C*H 大时不溢出（INV-8）。
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), samples_ * hidden_);
    yGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(yGrad), experts_ * capacity_ * hidden_);
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), experts_ * samples_);
    locationsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(locations), experts_ * samples_);
    gatesGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gatesGrad), experts_ * samples_);

    const int64_t rowBytes = hFactorAlign_ * static_cast<int64_t>(sizeof(T));
    if (rowBytes > 0) {
        pipe_->InitBuffer(xInQueue_, DOUBLE_BUFFER, rowBytes);
        pipe_->InitBuffer(yInQueue_, DOUBLE_BUFFER, rowBytes);
    }
    pipe_->InitBuffer(outQueue_, 1, kBlock_ * sBlockAlign_ * sizeof(T));
    pipe_->InitBuffer(outFp32Buf_, kBlock_ * sBlockAlign_ * sizeof(float));
    if constexpr (CUT_H) {
        partialAlign_ = (numChunks_ + VL_FP32 - 1) / VL_FP32 * VL_FP32;
        pipe_->InitBuffer(partialBuf_, partialAlign_ * sizeof(float));
    }
}

template <typename T, bool CUT_H>
__aicore__ inline uint32_t MoeTutelCombineGatesRegbase<T, CUT_H>::ChunkLen(int64_t chunk) const
{
    if constexpr (CUT_H) {
        return static_cast<uint32_t>((chunk + 1 == numChunks_) ? hTail_ : hFactor_);
    } else {
        return static_cast<uint32_t>(hidden_);
    }
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::CopyInX(int64_t sample, int64_t hOffset, uint32_t len)
{
    LocalTensor<T> xUb = xInQueue_.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    copyParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[sample * hidden_ + hOffset], copyParams, padParams);
    xInQueue_.template EnQue(xUb);
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::CopyInY(int64_t row, int64_t hOffset, uint32_t len)
{
    LocalTensor<T> yUb = yInQueue_.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    copyParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(yUb, yGradGm_[row * hidden_ + hOffset], copyParams, padParams);
    yInQueue_.template EnQue(yUb);
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::ComputeOnePair(int64_t sample, int64_t sInTile,
                                                                             int64_t kInTile, int64_t routeIdx,
                                                                             LocalTensor<T> &xUb)
{
    // INV-2：上下界四条边全判，缺一不可。canndev 只判 idx0 >= 0 && idx1 < C，会越界读 GM。
    const int32_t expertId = indicesGm_.GetValue(routeIdx);
    const int32_t location = locationsGm_.GetValue(routeIdx);
    const bool valid = (expertId >= 0) && (static_cast<int64_t>(expertId) < experts_) && (location >= 0) &&
                       (static_cast<int64_t>(location) < capacity_);
    // INV-3：非法路由（含 H == 0 的空内积）保持 0；tile 已整片清零，且清零本身就是"显式写出"。
    if (!valid || hidden_ == 0) {
        return;
    }

    __local_mem__ float *outFp32 = reinterpret_cast<__local_mem__ float *>(outFp32Buf_.Get<float>().GetPhyAddr());
    const uint32_t outOffset = static_cast<uint32_t>(kInTile * sBlockAlign_ + sInTile);
    const int64_t row = static_cast<int64_t>(expertId) * capacity_ + static_cast<int64_t>(location);

    if constexpr (CUT_H) {
        LocalTensor<float> partial = partialBuf_.Get<float>();
        // SumVF 无掩码整块累加，故先把整段（含 VL 对齐补位）清零。
        Duplicate(partial, 0.0f, static_cast<int32_t>(partialAlign_));
        __local_mem__ float *partialPtr = reinterpret_cast<__local_mem__ float *>(partial.GetPhyAddr());
        for (int64_t chunk = 0; chunk < numChunks_; chunk++) {
            const uint32_t len = ChunkLen(chunk);
            const int64_t hOffset = chunk * hFactor_;
            CopyInX(sample, hOffset, len);
            CopyInY(row, hOffset, len);
            LocalTensor<T> xChunk = xInQueue_.template DeQue<T>();
            LocalTensor<T> yChunk = yInQueue_.template DeQue<T>();
            DotProductVF<T>(partialPtr, static_cast<uint32_t>(chunk),
                            reinterpret_cast<__local_mem__ T *>(xChunk.GetPhyAddr()),
                            reinterpret_cast<__local_mem__ T *>(yChunk.GetPhyAddr()), len);
            xInQueue_.FreeTensor(xChunk);
            yInQueue_.FreeTensor(yChunk);
        }
        SumVF(outFp32, outOffset, partialPtr, static_cast<uint32_t>(numChunks_));
    } else {
        CopyInY(row, 0, static_cast<uint32_t>(hidden_));
        LocalTensor<T> yUb = yInQueue_.template DeQue<T>();
        DotProductVF<T>(outFp32, outOffset, reinterpret_cast<__local_mem__ T *>(xUb.GetPhyAddr()),
                        reinterpret_cast<__local_mem__ T *>(yUb.GetPhyAddr()), static_cast<uint32_t>(hidden_));
        yInQueue_.FreeTensor(yUb);
    }
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::ComputeTile(int64_t kBase, int64_t curK, int64_t sBase, int64_t curS)
{
    // INV-3：先把整片 tile（含对齐补位）清零，非法路由与 H == 0 直接落在 0 上，
    // 不依赖 GM 预清零，也不会出现 canndev「整组全非法就一次都不写」的未定义输出。
    LocalTensor<float> outFp32 = outFp32Buf_.Get<float>();
    Duplicate(outFp32, 0.0f, static_cast<int32_t>(kBlock_ * sBlockAlign_));

    for (int64_t i = 0; i < curS; i++) {
        const int64_t sample = sBase + i;
        LocalTensor<T> xUb;
        if constexpr (!CUT_H) {
            // NOT_CUT_H：x 的一行在本 k 窗口内只搬一次，这是相对 canndev 的主要 MTE2 收益。
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
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::CopyOut(int64_t kBase, int64_t curK, int64_t sBase, int64_t curS)
{
    __local_mem__ float *outFp32 = reinterpret_cast<__local_mem__ float *>(outFp32Buf_.Get<float>().GetPhyAddr());
    LocalTensor<T> outUb = outQueue_.template AllocTensor<T>();
    __local_mem__ T *outPtr = reinterpret_cast<__local_mem__ T *>(outUb.GetPhyAddr());
    for (int64_t kk = 0; kk < curK; kk++) {
        CastRowVF<T>(outPtr + kk * sBlockAlign_, outFp32 + kk * sBlockAlign_, static_cast<uint32_t>(curS));
    }
    outQueue_.template EnQue(outUb);

    LocalTensor<T> outLocal = outQueue_.template DeQue<T>();
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    copyParams.blockLen = static_cast<uint32_t>(curS * sizeof(T));
    for (int64_t kk = 0; kk < curK; kk++) {
        const int64_t k = kBase + kk;
        // INV-4：按实际元素数搬出，绝不像 canndev 那样定长 16 个元素写到输出末尾之外。
        DataCopyPad(gatesGradGm_[k * samples_ + sBase], outLocal[kk * sBlockAlign_], copyParams);
    }
    outQueue_.FreeTensor(outLocal);
}

template <typename T, bool CUT_H>
__aicore__ inline void MoeTutelCombineGatesRegbase<T, CUT_H>::Process()
{
    // S == 0 或 E == 0：输出元素数为 0，空进空出；分核区间长度自然为 0。
    if (sRange_.len <= 0 || kRange_.len <= 0) {
        return;
    }
    const int64_t sEnd = sRange_.start + sRange_.len;
    const int64_t kEnd = kRange_.start + kRange_.len;
    // k 窗口在外、s 分块在内：窗口内 x 行可复用；窗口高度由 tiling 按 UB 预算封顶。
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
