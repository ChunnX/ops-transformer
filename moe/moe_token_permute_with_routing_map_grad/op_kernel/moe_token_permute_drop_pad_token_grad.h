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
 * \file moe_token_permute_drop_pad_token_grad.h
 * \brief pad模式下tokensGrad计算: tokensGradOut[sortedIndices[i]] += permutedTokenOutputGrad[i]
 */

#ifndef MOE_TOKEN_PERMUTE_DROP_PAD_TOKEN_GRAD_H
#define MOE_TOKEN_PERMUTE_DROP_PAD_TOKEN_GRAD_H

#include "kernel_operator.h"
#if __CCE_AICORE__ == 200
#include "moe_token_unpermute_common.h"
#endif

using namespace AscendC;

template <typename T1, typename T2>
class KernelMoeTokenPermuteDropPadTokenGrad
{
public:
    __aicore__ inline KernelMoeTokenPermuteDropPadTokenGrad()
    {}

    __aicore__ inline void Init(
        GM_ADDR permuted_token_grad, GM_ADDR sorted_indices, GM_ADDR tokens_grad_out,
        const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitData(const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData& tilingData);
    __aicore__ inline void CalSingleOutToken(const int64_t out_token_idx, const int64_t global_token_id);
    __aicore__ inline void CalPartOutToken(
        const int64_t h_index, const int64_t h_length, const int64_t out_token_idx, const int64_t global_token_id);
    __aicore__ inline void CopyTokenIn(const T2 in_token_index, const int64_t h_index, const int64_t h_length);
    __aicore__ inline void CalFirstToken(const int64_t h_length);
    __aicore__ inline void CalToken(const int64_t h_length);
    __aicore__ inline void CopyOut(const int64_t out_token_index, const int64_t h_index, const int64_t h_length);

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> tokensInQue_;
    TBuf<TPosition::VECCALC> tempBuffer0_;
    TBuf<TPosition::VECCALC> tempBuffer1_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    GlobalTensor<T1> permutedTokenGradGM_;
    GlobalTensor<T2> sortedIndicesGM_;
    GlobalTensor<T1> tokensGradOutGM_;
    LocalTensor<float> tokenTensor0_;
    LocalTensor<float> tokenTensor1_;
    DataCopyPadExtParams<T1> extParams1_{false, 0, 0, 0};
    DataCopyExtParams copyParams_{1, 0, 0, 0, 0};

    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t ALIGN_512 = 512;

    int64_t blockIdx_;
    int64_t hiddenSize_;
    int64_t numOutTokens_;
    int64_t hiddenSplitedLength_;
    int64_t hiddenSplitedNum_;
    int64_t hiddenSplitedRemain_;
    int64_t tokensCoreLength_;
    int64_t tokensCoreRemain_;
    int64_t tokenStart_;
};

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::InitData(
    const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData& tilingData)
{
    blockIdx_ = AscendC::GetBlockIdx();
    hiddenSize_ = tilingData.hiddenSize;
    numOutTokens_ = tilingData.numOutTokens;
    hiddenSplitedLength_ = tilingData.hiddenSplitedLength;
    hiddenSplitedNum_ = tilingData.hiddenSplitedNum;
    hiddenSplitedRemain_ = tilingData.hiddenSplitedRemain;
    tokensCoreLength_ = tilingData.tokensCoreLength;
    tokensCoreRemain_ = tilingData.tokensCoreRemain;

    if (tokensCoreRemain_ > 0 && blockIdx_ < tokensCoreRemain_) {
        tokensCoreLength_ += 1;
    }

    if (tokensCoreRemain_ > 0) {
        if (blockIdx_ < tokensCoreRemain_) {
            tokenStart_ = blockIdx_ * tokensCoreLength_;
        } else {
            tokenStart_ = (tokensCoreLength_ + 1) * tokensCoreRemain_ +
                          tokensCoreLength_ * (blockIdx_ - tokensCoreRemain_);
        }
    } else {
        tokenStart_ = tokensCoreLength_ * blockIdx_;
    }

    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    ASSERT(hiddenSize_ != 0 && "hiddenSize can not be zero!");
    ASSERT(numOutTokens_ != 0 && "numOutTokens can not be zero!");
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::Init(
    GM_ADDR permuted_token_grad, GM_ADDR sorted_indices, GM_ADDR tokens_grad_out,
    const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData* __restrict tilingData)
{
    InitData(*tilingData);

    int64_t hiddenSplitedLengthAlign512 = (hiddenSplitedLength_ + ALIGN_512 - 1) & ~(ALIGN_512 - 1);
    int64_t outBlockOffset;
    if (tokensCoreRemain_ > 0) {
        if (blockIdx_ < tokensCoreRemain_) {
            outBlockOffset = tokensCoreLength_ * blockIdx_ * hiddenSize_;
        } else {
            outBlockOffset = (tokensCoreLength_ + 1) * tokensCoreRemain_ +
                             tokensCoreLength_ * (blockIdx_ - tokensCoreRemain_);
            outBlockOffset *= hiddenSize_;
        }
    } else {
        outBlockOffset = tokensCoreLength_ * blockIdx_ * hiddenSize_;
    }

    permutedTokenGradGM_.SetGlobalBuffer((__gm__ T1*)permuted_token_grad);
    sortedIndicesGM_.SetGlobalBuffer((__gm__ T2*)sorted_indices, numOutTokens_);
    tokensGradOutGM_.SetGlobalBuffer(
        (__gm__ T1*)tokens_grad_out + outBlockOffset, tokensCoreLength_ * hiddenSize_);

    pipe_.InitBuffer(tokensInQue_, tilingData->bufferNum, hiddenSplitedLengthAlign512 * sizeof(T1));
    pipe_.InitBuffer(outQue_, 1, hiddenSplitedLengthAlign512 * sizeof(T1));

    if constexpr (!IsSameType<T1, float>::value) {
        pipe_.InitBuffer(tempBuffer0_, hiddenSplitedLengthAlign512 * sizeof(float) + 256);
        pipe_.InitBuffer(tempBuffer1_, hiddenSplitedLengthAlign512 * sizeof(float));
        tokenTensor0_ = tempBuffer0_.template Get<float>();
        tokenTensor1_ = tempBuffer1_.template Get<float>();
    }
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::Process()
{
    for (int64_t outTokenIdx = 0; outTokenIdx < tokensCoreLength_; ++outTokenIdx) {
        CalSingleOutToken(outTokenIdx, tokenStart_ + outTokenIdx);
    }
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::CalSingleOutToken(
    const int64_t out_token_idx, const int64_t global_token_id)
{
    for (int64_t hIndex = 0; hIndex < hiddenSplitedNum_; ++hIndex) {
        CalPartOutToken(hIndex, hiddenSplitedLength_, out_token_idx, global_token_id);
    }
    if (hiddenSplitedRemain_ > 0) {
        CalPartOutToken(hiddenSplitedNum_, hiddenSplitedRemain_, out_token_idx, global_token_id);
    }
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::CalPartOutToken(
    const int64_t h_index, const int64_t h_length, const int64_t out_token_index, const int64_t global_token_id)
{
    if constexpr (IsSameType<T1, float>::value) {
        tokenTensor0_ = outQue_.template AllocTensor<T1>();
    }

    bool hasFirst = false;
    for (int64_t permutedIdx = 0; permutedIdx < numOutTokens_; ++permutedIdx) {
        T2 tokenIdx = sortedIndicesGM_.GetValue(permutedIdx);
        if (tokenIdx != global_token_id) {
            continue;
        }
        if (!hasFirst) {
            CopyTokenIn(permutedIdx, h_index, h_length);
            PipeBarrier<PIPE_V>();
            CalFirstToken(h_length);
            hasFirst = true;
        } else {
            CopyTokenIn(permutedIdx, h_index, h_length);
            PipeBarrier<PIPE_V>();
            CalToken(h_length);
        }
    }

    if (!hasFirst) {
        PipeBarrier<PIPE_V>();
        Duplicate(tokenTensor0_, static_cast<float>(0), h_length);
    }

    CopyOut(out_token_index, h_index, h_length);
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::CopyTokenIn(
    const T2 in_token_index, const int64_t h_index, const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = tokensInQue_.template AllocTensor<T1>();
    int64_t offset = in_token_index * hiddenSize_ + h_index * hiddenSplitedLength_;

    if (likely((h_length * sizeof(T1)) % BLOCK_SIZE == 0)) {
        DataCopy(tokensLocal, permutedTokenGradGM_[offset], h_length);
    } else {
        copyParams_.blockLen = h_length * sizeof(T1);
#if __CCE_AICORE__ == 200
        MoeTokenUnPermute::DataCopyPadCustom(tokensLocal, permutedTokenGradGM_[offset], copyParams_, extParams1_);
#else
        DataCopyPad(tokensLocal, permutedTokenGradGM_[offset], copyParams_, extParams1_);
#endif
    }
    tokensInQue_.template EnQue(tokensLocal);
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::CalFirstToken(const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = tokensInQue_.template DeQue<T1>();

    if constexpr (!IsSameType<T1, float>::value) {
        Cast(tokenTensor0_, tokensLocal, RoundMode::CAST_NONE, h_length);
    } else {
        uint64_t byteAlign32 = (h_length * sizeof(float) + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
        DataCopy(tokenTensor0_, tokensLocal, byteAlign32 / sizeof(float));
    }
    tokensInQue_.FreeTensor(tokensLocal);
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::CalToken(const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = tokensInQue_.template DeQue<T1>();

    if constexpr (!IsSameType<T1, float>::value) {
        Cast(tokenTensor1_, tokensLocal, RoundMode::CAST_NONE, h_length);
        tokensInQue_.FreeTensor(tokensLocal);
        PipeBarrier<PIPE_V>();
        Add(tokenTensor0_, tokenTensor0_, tokenTensor1_, h_length);
    } else {
        Add(tokenTensor0_, tokenTensor0_, tokensLocal, h_length);
        tokensInQue_.FreeTensor(tokensLocal);
    }
}

template <typename T1, typename T2>
__aicore__ inline void KernelMoeTokenPermuteDropPadTokenGrad<T1, T2>::CopyOut(
    const int64_t out_token_index, const int64_t h_index, const int64_t h_length)
{
    LocalTensor<T1> tempOutTensors;
    if constexpr (!IsSameType<T1, float>::value) {
        tempOutTensors = outQue_.template AllocTensor<T1>();
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        Cast(tempOutTensors, tokenTensor0_, RoundMode::CAST_NONE, h_length);
#else
        Cast(tempOutTensors, tokenTensor0_, RoundMode::CAST_RINT, h_length);
#endif
    } else {
        tempOutTensors = tokenTensor0_;
    }

    outQue_.template EnQue<T1>(tempOutTensors);
    tempOutTensors = outQue_.template DeQue<T1>();

    int64_t offset = out_token_index * hiddenSize_ + h_index * hiddenSplitedLength_;
    if (likely((h_length * sizeof(T1)) % BLOCK_SIZE == 0)) {
        DataCopy(tokensGradOutGM_[offset], tempOutTensors, h_length);
    } else {
        copyParams_.blockLen = h_length * sizeof(T1);
#if __CCE_AICORE__ == 200
        MoeTokenUnPermute::DataCopyCustom(tokensGradOutGM_[offset], tempOutTensors, copyParams_.blockCount, copyParams_.blockLen);
#else
        DataCopyPad(tokensGradOutGM_[offset], tempOutTensors, copyParams_);
#endif
    }
    outQue_.FreeTensor(tempOutTensors);
}

#endif // MOE_TOKEN_PERMUTE_DROP_PAD_TOKEN_GRAD_H
