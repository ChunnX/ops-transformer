/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file ai_infra_sinkhorn_grad_generalized.cpp
\brief
*/

#ifndef AI_INFRA_SINKHORN_GRAD_GENERALIZED_H
#define AI_INFRA_SINKHORN_GRAD_GENERALIZED_H
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AiInfraSinkhornGrad
{
    using namespace AscendC;

    constexpr int64_t BLOCK_BYTES = 32;
    constexpr int64_t TRANSPOSE_BUFFER_FACTOR = 128;
    constexpr int64_t CONSTANT_TWO = 2;
    constexpr int64_t BUFFER_TWO = 2;

    __aicore__ inline int64_t Ceil(int64_t a, int64_t b)
    {
        if (b == 0)
        {
            return 0;
        }
        return (a + b - 1) / b;
    }

    __aicore__ inline int64_t Min(int64_t a, int64_t b)
    {
        return (a < b) ? a : b;
    }

    __aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
    {
        if (bytes == 0)
        {
            return 0;
        }
        return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
    }

    __aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
    {
        return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    }

    class AiInfraSinkhornGradGeneralized
    {
    public:
        __aicore__ inline AiInfraSinkhornGradGeneralized(){};
        __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR normOut, GM_ADDR sumOut, GM_ADDR gradInput,
                                    GM_ADDR workspace, const AiInfraSinkhornGradTilingData *__restrict__ tilingData, TPipe *tPipe);
        __aicore__ inline void Process();

    private:
        __aicore__ inline void CopyInX(int64_t offset);
        __aicore__ inline void TransposeXIn();
        __aicore__ inline void CopyInNormOut(int64_t offset);
        __aicore__ inline void CopyInSumOut(int64_t offset);
        __aicore__ inline void colNormGrad();
        __aicore__ inline void rowNormGrad();
        __aicore__ inline void softmaxGrad();
        __aicore__ inline void TransposeXOut();
        __aicore__ inline void CopyOut(int64_t offset);

    private:
        TPipe *pipe_;
        TQue<QuePosition::VECIN, 1> gradInQueue_;
        TQue<QuePosition::VECIN, 2> normInQueue_;
        TQue<QuePosition::VECIN, 2> sumInQueue_;
        TQue<QuePosition::VECOUT, 1> gradOutQueue_;
        TBuf<TPosition::VECCALC> calcTmpBuf_;
        TBuf<TPosition::VECCALC> gradTransposeBuf_;

        GlobalTensor<float> gradOutputGm_;
        GlobalTensor<float> normOutGm_;
        GlobalTensor<float> sumOutGm_;
        GlobalTensor<float> gradInputGm_;

        int64_t totalLength_;
        int64_t n_;
        int64_t numIters_;
        int64_t needCoreNum_;
        int64_t tAlign_;

        // 核间切分
        int64_t perCoreElements_;
        int64_t lastCoreElements_;
        int64_t blockIdx_;

        // 核内切分
        int64_t perCoreLoops_;
        int64_t perCorePerLoopElements_;
        int64_t perCoreLastLoopElements_;
        int64_t lastCoreLoops_;
        int64_t lastCorePerLoopElements_;
        int64_t lastCoreLastLoopElements_;

        // 当前核的tiling策略
        int64_t currentCoreElements_;
        int64_t currentCoreloops_;
        int64_t currentCorePerLoopElements_;
        int64_t currentCoreLastLoopElements_;

        int64_t currentLoopElements_;
        const AiInfraSinkhornGradTilingData *__restrict__ tilingData_;
    };

    __aicore__ inline void
    AiInfraSinkhornGradGeneralized::Init(GM_ADDR gradOutput, GM_ADDR normOut, GM_ADDR sumOut, GM_ADDR gradInput,
                                  GM_ADDR workspace, const AiInfraSinkhornGradTilingData *__restrict__ tilingData, TPipe *tPipe)
    {
        pipe_ = tPipe;
        tilingData_ = tilingData;
        totalLength_ = tilingData->totalLength;
        n_ = tilingData->n;
        numIters_ = tilingData->numIters;
        needCoreNum_ = tilingData->needCoreNum;
        perCoreElements_ = tilingData->perCoreElements;
        lastCoreElements_ = tilingData->lastCoreElements;

        perCoreLoops_ = tilingData->perCoreLoops;
        perCorePerLoopElements_ = tilingData->perCorePerLoopElements;
        perCoreLastLoopElements_ = tilingData->perCoreLastLoopElements;
        lastCoreLoops_ = tilingData->lastCoreLoops;
        lastCorePerLoopElements_ = tilingData->lastCorePerLoopElements;
        lastCoreLastLoopElements_ = tilingData->lastCoreLastLoopElements;

        blockIdx_ = GetBlockIdx();
        currentCoreElements_ = (blockIdx_ < needCoreNum_ - 1) ? perCoreElements_ : lastCoreElements_;
        currentCoreloops_ = (blockIdx_ < needCoreNum_ - 1) ? perCoreLoops_ : lastCoreLoops_;
        currentCorePerLoopElements_ = (blockIdx_ < needCoreNum_ - 1) ? perCorePerLoopElements_ : lastCorePerLoopElements_;
        currentCoreLastLoopElements_ = (blockIdx_ < needCoreNum_ - 1) ? perCoreLastLoopElements_ : lastCoreLastLoopElements_;

        tAlign_ = Align(perCorePerLoopElements_, sizeof(float));
        pipe_->InitBuffer(gradInQueue_, 1, tAlign_ * n_ * n_ * sizeof(float));
        pipe_->InitBuffer(normInQueue_, BUFFER_TWO, tAlign_ * n_ * n_ * sizeof(float));
        pipe_->InitBuffer(sumInQueue_, BUFFER_TWO, tAlign_ * n_ * sizeof(float));
        pipe_->InitBuffer(gradOutQueue_, 1, tAlign_ * n_ * n_ * sizeof(float));

        int64_t transposeBuf = AlignBytes((n_ * n_ * CONSTANT_TWO + 1) * TRANSPOSE_BUFFER_FACTOR, sizeof(float));
        int64_t bufferSize = tAlign_ * n_ * n_ * CONSTANT_TWO * sizeof(float) + tAlign_ * n_ * sizeof(float);
        if (bufferSize >= transposeBuf)
        {
            pipe_->InitBuffer(calcTmpBuf_, bufferSize);
        }
        else
        {
            pipe_->InitBuffer(calcTmpBuf_, transposeBuf);
        }
        pipe_->InitBuffer(gradTransposeBuf_, tAlign_ * n_ * n_ * sizeof(float));

        int64_t Offset = blockIdx_ * perCoreElements_ * n_ * n_;
        gradOutputGm_.SetGlobalBuffer((__gm__ float *)gradOutput + Offset, currentCoreElements_ * n_ * n_);
        normOutGm_.SetGlobalBuffer((__gm__ float *)normOut);
        sumOutGm_.SetGlobalBuffer((__gm__ float *)sumOut);
        gradInputGm_.SetGlobalBuffer((__gm__ float *)gradInput + Offset, currentCoreElements_ * n_ * n_);
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::Process()
    {   
        if (blockIdx_ >= needCoreNum_) {
                return;
        }
        // 一次归一化产生的normOut矩阵大小
        int64_t normOutBlockLen = n_ * n_ * totalLength_;
        // 一次归一化产生的sumOut矩阵大小
        int64_t sumOutBlockLen = n_ * totalLength_;

        for (int i = 0; i < currentCoreloops_; ++i)
        {
            // 当前核当前loop处理的token数
            currentLoopElements_ = (i == currentCoreloops_ - 1) ? currentCoreLastLoopElements_ : currentCorePerLoopElements_;
            // 核内grad_output GM核内偏移, 初始化时已增加核间偏移
            int64_t offset = i * currentCorePerLoopElements_ * n_ * n_;
            // normOut与sumOut在GM上最后一个维度上t的偏移
            int64_t tOffset = blockIdx_ * perCoreElements_ + i * currentCorePerLoopElements_;
            int64_t normOffset = 0;
            int64_t sumOffset = 0;

            // 拷入并转置
            CopyInX(offset);
            TransposeXIn();

            for (int j = numIters_ - 1; j > 0; --j)
            {
                int64_t colOffset = 2 * j + 1;
                int64_t rowOffset = 2 * j;

                // 列归一化
                normOffset = colOffset * normOutBlockLen + tOffset;
                sumOffset = colOffset * sumOutBlockLen + tOffset;
                CopyInNormOut(normOffset);
                CopyInSumOut(sumOffset);
                colNormGrad();

                // 行归一化
                normOffset = rowOffset * normOutBlockLen + tOffset;
                sumOffset = rowOffset * sumOutBlockLen + tOffset;
                CopyInNormOut(normOffset);
                CopyInSumOut(sumOffset);
                rowNormGrad();
            }

            // 列最后一次归一化
            normOffset = 1 * normOutBlockLen + tOffset;
            sumOffset = 1 * sumOutBlockLen + tOffset;
            CopyInNormOut(normOffset);
            CopyInSumOut(sumOffset);
            colNormGrad();

            // 行最后归一化
            normOffset = 0 * normOutBlockLen + tOffset;
            CopyInNormOut(normOffset);
            softmaxGrad();

            // 转置搬出
            TransposeXOut();
            CopyOut(offset);
        }
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::CopyInX(int64_t offset)
    {
        // 拷入[t, n, n] grad
        LocalTensor<float> gradLocal = gradInQueue_.AllocTensor<float>();
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(currentLoopElements_ * n_ * n_ * sizeof(float)), 0,
                                         0, 0};
        DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(gradLocal, gradOutputGm_[offset], dataCopyParams, dataCopyPadParams);
        gradInQueue_.EnQue(gradLocal);
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::TransposeXIn()
    {
        // [t, n, n] grad转换为[n, n, t]
        LocalTensor<float> gradLocal = gradInQueue_.DeQue<float>();
        LocalTensor<float> gradTranspose = gradTransposeBuf_.Get<float>();

        auto h0 = 16;
        auto w0 = 32 / sizeof(float);
        auto tmpBufferSize = (n_ * n_ * 2 + 1) * h0 * w0;
        LocalTensor<uint8_t> stackBuffer = calcTmpBuf_.Get<uint8_t>(tmpBufferSize);

        TransposeParamsExt transposeParams;
        transposeParams.nSize = 1;
        transposeParams.cSize = n_ * n_;
        transposeParams.hSize = 1;
        transposeParams.wSize = tAlign_;
        transposeParams.transposeType = TransposeType::TRANSPOSE_NHWC2NCHW;
        Transpose(gradTranspose, gradLocal, stackBuffer, transposeParams);
        PipeBarrier<PIPE_V>();
        gradInQueue_.FreeTensor(gradLocal);
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::TransposeXOut()
    {
        // [n, n, t] grad转换为[t, n, n]
        LocalTensor<float> gradLocal = gradTransposeBuf_.Get<float>();
        LocalTensor<float> gradTranspose = gradOutQueue_.AllocTensor<float>();

        auto h0 = 16;
        auto w0 = 32 / sizeof(float);
        auto tmpBufferSize = (n_ * n_ + 2) * h0 * w0;
        LocalTensor<uint8_t> stackBuffer = calcTmpBuf_.Get<uint8_t>(tmpBufferSize);

        TransposeParamsExt transposeParams;
        transposeParams.nSize = 1;
        transposeParams.cSize = n_ * n_;
        transposeParams.hSize = 1;
        transposeParams.wSize = tAlign_;
        transposeParams.transposeType = TransposeType::TRANSPOSE_NCHW2NHWC;
        Transpose(gradTranspose, gradLocal, stackBuffer, transposeParams);
        PipeBarrier<PIPE_V>();
        gradOutQueue_.EnQue(gradTranspose);
    }

    /**
     * @brief normOut在GM上的排序shape是[2*numIters, n, n, T], 需要尾轴切分拷入
     */
    __aicore__ inline void AiInfraSinkhornGradGeneralized::CopyInNormOut(int64_t offset)
    {
        uint16_t blockCount = static_cast<uint16_t>(n_ * n_);
        uint32_t blockLen = static_cast<uint32_t>(currentLoopElements_ * sizeof(float));
        uint32_t srcStride = static_cast<uint32_t>((totalLength_ - currentLoopElements_) * sizeof(float));
        uint32_t dstStride = static_cast<uint32_t>((tAlign_ - currentLoopElements_) * sizeof(float) / BLOCK_BYTES);
        DataCopyExtParams dataCopyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};

        LocalTensor<float> normLocal = normInQueue_.AllocTensor<float>();
        DataCopyPad(normLocal, normOutGm_[offset], dataCopyParams, dataCopyPadParams);
        normInQueue_.EnQue(normLocal);
    }

    /**
     * @brief SumOut在GM上的排序shape是[2*numIters, n, T], 需要尾轴切分拷入
     */
    __aicore__ inline void AiInfraSinkhornGradGeneralized::CopyInSumOut(int64_t offset)
    {
        uint16_t blockCount = static_cast<uint16_t>(n_);
        uint32_t blockLen = static_cast<uint32_t>(currentLoopElements_ * sizeof(float));
        uint32_t srcStride = static_cast<uint32_t>((totalLength_ - currentLoopElements_) * sizeof(float));
        uint32_t dstStride = static_cast<uint32_t>((tAlign_ - currentLoopElements_) * sizeof(float) / BLOCK_BYTES);
        DataCopyExtParams dataCopyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};

        LocalTensor<float> sumLocal = sumInQueue_.AllocTensor<float>();
        DataCopyPad(sumLocal, sumOutGm_[offset], dataCopyParams, dataCopyPadParams);
        sumInQueue_.EnQue(sumLocal);
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::colNormGrad()
    {
        // 1. grad * normOut
        // 2. dot_prod = reduceSum(grad * normOut)
        // 3. grad - dot_prod
        // 4. (grad - dot_prod) / sumOut
        LocalTensor<float> gradLocal = gradTransposeBuf_.Get<float>();
        LocalTensor<float> normLocal = normInQueue_.DeQue<float>();
        LocalTensor<float> sumLocal = sumInQueue_.DeQue<float>();

        int64_t nt = tAlign_ * n_;
        int64_t nnt = n_ * nt;
        LocalTensor<float> mulResultTensor = calcTmpBuf_.GetWithOffset<float>(nnt, 0);
        LocalTensor<float> dot_prod = calcTmpBuf_.GetWithOffset<float>(nt, nnt * sizeof(float));
        LocalTensor<float> subResultTensor = calcTmpBuf_.GetWithOffset<float>(nnt, nnt * sizeof(float) + nt * sizeof(float));

        Mul(mulResultTensor, gradLocal, normLocal, nnt);
        PipeBarrier<PIPE_V>();
        uint32_t shape[] = {static_cast<uint32_t>(n_), static_cast<uint32_t>(tAlign_ * n_)};
        // 列方向求和, Pattern::Reduce::RA表示按第一个维度求和
        ReduceSum<float, Pattern::Reduce::RA, true>(dot_prod, mulResultTensor, shape, true);
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < n_; ++i)
        {
            Sub(subResultTensor[i * nt], gradLocal[i * nt], dot_prod, nt);
            PipeBarrier<PIPE_V>();
            Div(gradLocal[i * nt], subResultTensor[i * nt], sumLocal, nt);
        }
        PipeBarrier<PIPE_V>();
        normInQueue_.FreeTensor(normLocal);
        sumInQueue_.FreeTensor(sumLocal);
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::rowNormGrad()
    {
        // 1. grad * normOut
        // 2. dot_prod = sum
        // 3. grad - dot_prod
        // 4. (grad - dot_prod) / sumOut
        LocalTensor<float> gradLocal = gradTransposeBuf_.Get<float>();
        LocalTensor<float> normLocal = normInQueue_.DeQue<float>();
        LocalTensor<float> sumLocal = sumInQueue_.DeQue<float>();

        int64_t nt = tAlign_ * n_;
        int64_t nnt = n_ * nt;
        LocalTensor<float> mulResultTensor = calcTmpBuf_.GetWithOffset<float>(nnt, 0);
        LocalTensor<float> dot_prod = calcTmpBuf_.GetWithOffset<float>(nt, nnt * sizeof(float));
        LocalTensor<float> subResultTensor = calcTmpBuf_.GetWithOffset<float>(nnt, nnt * sizeof(float) + nt * sizeof(float));

        Mul(mulResultTensor, gradLocal, normLocal, nnt);
        PipeBarrier<PIPE_V>();
        uint32_t shape[] = {static_cast<uint32_t>(n_), static_cast<uint32_t>(tAlign_)};
        // 行方向求和
        for (int i = 0; i < n_; ++i)
        {
            ReduceSum<float, Pattern::Reduce::RA, true>(dot_prod[i * tAlign_], mulResultTensor[i * tAlign_ * n_], shape, true);
        }
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < n_; ++i)
        {
            int64_t rowOffset = i * nt;
            for (int j = 0; j < n_; ++j)
            {
                int64_t offset = rowOffset + j * tAlign_;
                Sub(subResultTensor[offset], gradLocal[offset], dot_prod[i * tAlign_], tAlign_);
                PipeBarrier<PIPE_V>();
                Div(gradLocal[offset], subResultTensor[offset], sumLocal[i * tAlign_], tAlign_);
            }
        }
        PipeBarrier<PIPE_V>();
        normInQueue_.FreeTensor(normLocal);
        sumInQueue_.FreeTensor(sumLocal);
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::softmaxGrad()
    {
        // 1. grad * normOut
        // 2. dot_prod = sum
        // 3. grad - dot_prod
        // 4. (grad - dot_prod) * normOut
        LocalTensor<float> gradLocal = gradTransposeBuf_.Get<float>();
        LocalTensor<float> normLocal = normInQueue_.DeQue<float>();

        int64_t nt = tAlign_ * n_;
        int64_t nnt = n_ * nt;
        LocalTensor<float> mulResultTensor = calcTmpBuf_.GetWithOffset<float>(nnt, 0);
        LocalTensor<float> dot_prod = calcTmpBuf_.GetWithOffset<float>(nt, nnt * sizeof(float));
        LocalTensor<float> subResultTensor = calcTmpBuf_.GetWithOffset<float>(nnt, nnt * sizeof(float) + nt * sizeof(float));

        Mul(mulResultTensor, gradLocal, normLocal, nnt);
        PipeBarrier<PIPE_V>();
        uint32_t shape[] = {static_cast<uint32_t>(n_), static_cast<uint32_t>(tAlign_)};

        // 行与行求和
        for (int i = 0; i < n_; ++i)
        {
            ReduceSum<float, Pattern::Reduce::RA, true>(dot_prod[i * tAlign_], mulResultTensor[i * tAlign_ * n_], shape, true);
        }
        PipeBarrier<PIPE_V>();

        for (int i = 0; i < n_; ++i)
        {
            int64_t rowOffset = i * nt;
            for (int j = 0; j < n_; ++j)
            {
                int64_t offset = rowOffset + j * tAlign_;
                Sub(subResultTensor[offset], gradLocal[offset], dot_prod[i * tAlign_], tAlign_);
            }
        }
        PipeBarrier<PIPE_V>();
        Mul(gradLocal, subResultTensor, normLocal, nnt);
        PipeBarrier<PIPE_V>();
        normInQueue_.FreeTensor(normLocal);
    }

    __aicore__ inline void AiInfraSinkhornGradGeneralized::CopyOut(int64_t offset)
    {
        LocalTensor<float> gradLocal = gradOutQueue_.DeQue<float>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(currentLoopElements_ * n_ * n_ * sizeof(float)), 0, 0,
                                     0};
        DataCopyPad(gradInputGm_[offset], gradLocal, copyParams);
        gradOutQueue_.FreeTensor(gradLocal);
    }
}
#endif