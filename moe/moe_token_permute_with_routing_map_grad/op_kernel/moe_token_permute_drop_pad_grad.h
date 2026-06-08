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
 * \file moe_token_permute_drop_pad_grad.h
 * \brief pad模式下整合tokensGrad与probsGrad计算
 */

#ifndef MOE_TOKEN_PERMUTE_DROP_PAD_GRAD_H
#define MOE_TOKEN_PERMUTE_DROP_PAD_GRAD_H

#include "kernel_operator.h"
#if __CCE_AICORE__ == 200
#include "moe_token_unpermute_common.h"
#endif

using namespace AscendC;

template <typename TokenT, typename ProbT, typename IdxT>
class KernelMoeTokenPermuteDropPadGrad
{
public:
    __aicore__ inline KernelMoeTokenPermuteDropPadGrad()
    {}

    __aicore__ inline void Init(
        GM_ADDR permuted_token_grad, GM_ADDR permuted_probs_grad, GM_ADDR sorted_indices, GM_ADDR tokens_grad_out,
        GM_ADDR probs_grad_out, const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData* __restrict tokenTilingData,
        const MoeTokenpermuteWithRoutingMapDropPadTilingData* __restrict probsTilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTokenData(const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData& tilingData);
    __aicore__ inline void InitProbsData(const MoeTokenpermuteWithRoutingMapDropPadTilingData& tilingData);
    __aicore__ inline void ProcessTokenGrad();
    __aicore__ inline void ProcessProbsGrad();
    __aicore__ inline void CalSingleOutToken(const int64_t out_token_idx, const int64_t global_token_id);
    __aicore__ inline void CalPartOutToken(
        const int64_t h_index, const int64_t h_length, const int64_t out_token_index, const int64_t global_token_id);
    __aicore__ inline void CopyTokenIn(const IdxT in_token_index, const int64_t h_index, const int64_t h_length);
    __aicore__ inline void CalFirstToken(const int64_t h_length);
    __aicore__ inline void CalToken(const int64_t h_length);
    __aicore__ inline void CopyTokenOut(const int64_t out_token_index, const int64_t h_index, const int64_t h_length);

    TPipe tokenPipe_;
    TQue<QuePosition::VECIN, 1> tokensInQue_;
    TBuf<TPosition::VECCALC> tempBuffer0_;
    TBuf<TPosition::VECCALC> tempBuffer1_;
    TQue<QuePosition::VECOUT, 1> tokenOutQue_;
    TPipe probsPipe_;
    TQue<QuePosition::VECOUT, 1> probGradOutQue_;

    GlobalTensor<TokenT> permutedTokenGradGM_;
    GlobalTensor<IdxT> sortedIndicesGM_;
    GlobalTensor<TokenT> tokensGradOutGM_;
    GlobalTensor<ProbT> permutedProbsGM_;
    GlobalTensor<ProbT> probsGradOutGM_;

    LocalTensor<float> tokenTensor0_;
    LocalTensor<float> tokenTensor1_;
    DataCopyPadExtParams<TokenT> tokenExtParams_{false, 0, 0, 0};
    DataCopyExtParams tokenCopyParams_{1, 0, 0, 0, 0};
    DataCopyExtParams probCopyParams_{1, 0, 0, 0, 0};

    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t ALIGN_512 = 512;

    int64_t blockIdx_;
    int64_t haveProbs_;
    int64_t tokenNum_;
    int64_t hiddenSize_;
    int64_t numOutTokens_;
    int64_t expertNum_;
    int64_t capacity_;
    int64_t hiddenSplitedLength_;
    int64_t hiddenSplitedNum_;
    int64_t hiddenSplitedRemain_;
    int64_t tokensCoreLength_;
    int64_t tokensCoreRemain_;
    int64_t tokenStart_;
    int64_t tokenUsedCoreNum_;
    int64_t probsUsedCoreNum_;
    int64_t probsSingleCoreLen_;
    int64_t probsLastCoreLen_;
};

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::InitTokenData(
    const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData& tilingData)
{
    hiddenSize_ = tilingData.hiddenSize;
    numOutTokens_ = tilingData.numOutTokens;
    hiddenSplitedLength_ = tilingData.hiddenSplitedLength;
    hiddenSplitedNum_ = tilingData.hiddenSplitedNum;
    hiddenSplitedRemain_ = tilingData.hiddenSplitedRemain;
    tokensCoreLength_ = tilingData.tokensCoreLength;
    tokensCoreRemain_ = tilingData.tokensCoreRemain;
    haveProbs_ = tilingData.haveProbs;
    tokenUsedCoreNum_ = tilingData.tokenUsedCoreNum;
    probsUsedCoreNum_ = tilingData.probsUsedCoreNum;

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
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::InitProbsData(
    const MoeTokenpermuteWithRoutingMapDropPadTilingData& tilingData)
{
    expertNum_ = tilingData.expertNum;
    capacity_ = tilingData.capacity;
    probsSingleCoreLen_ = tilingData.singleCoreLen;
    probsLastCoreLen_ = tilingData.lastCoreLen;
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::Init(
    GM_ADDR permuted_token_grad, GM_ADDR permuted_probs_grad, GM_ADDR sorted_indices, GM_ADDR tokens_grad_out,
    GM_ADDR probs_grad_out, const MoeTokenpermuteWithRoutingMapDropPadTokenTilingData* __restrict tokenTilingData,
    const MoeTokenpermuteWithRoutingMapDropPadTilingData* __restrict probsTilingData)
{
    blockIdx_ = AscendC::GetBlockIdx();
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    ASSERT(tokenTilingData->hiddenSize != 0 && "hiddenSize can not be zero!");
    ASSERT(tokenTilingData->numOutTokens != 0 && "numOutTokens can not be zero!");

    haveProbs_ = tokenTilingData->haveProbs;
    tokenUsedCoreNum_ = tokenTilingData->tokenUsedCoreNum;
    probsUsedCoreNum_ = tokenTilingData->probsUsedCoreNum;
    tokenNum_ = tokenTilingData->tokenNum;
    hiddenSize_ = tokenTilingData->hiddenSize;
    numOutTokens_ = tokenTilingData->numOutTokens;

    permutedTokenGradGM_.SetGlobalBuffer((__gm__ TokenT*)permuted_token_grad);
    sortedIndicesGM_.SetGlobalBuffer((__gm__ IdxT*)sorted_indices, numOutTokens_);

    if (blockIdx_ < tokenUsedCoreNum_) {
        InitTokenData(*tokenTilingData);

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

        tokensGradOutGM_.SetGlobalBuffer(
            (__gm__ TokenT*)tokens_grad_out + outBlockOffset, tokensCoreLength_ * hiddenSize_);

        tokenPipe_.InitBuffer(tokensInQue_, tokenTilingData->bufferNum, hiddenSplitedLengthAlign512 * sizeof(TokenT));
        tokenPipe_.InitBuffer(tokenOutQue_, 1, hiddenSplitedLengthAlign512 * sizeof(TokenT));
        if constexpr (!IsSameType<TokenT, float>::value) {
            tokenPipe_.InitBuffer(tempBuffer0_, hiddenSplitedLengthAlign512 * sizeof(float) + 256);
            tokenPipe_.InitBuffer(tempBuffer1_, hiddenSplitedLengthAlign512 * sizeof(float));
            tokenTensor0_ = tempBuffer0_.template Get<float>();
            tokenTensor1_ = tempBuffer1_.template Get<float>();
        }
    }

    if (haveProbs_ != 0) {
        InitProbsData(*probsTilingData);
        if (blockIdx_ < probsUsedCoreNum_) {
            permutedProbsGM_.SetGlobalBuffer((__gm__ ProbT*)permuted_probs_grad, numOutTokens_);
            probsGradOutGM_.SetGlobalBuffer((__gm__ ProbT*)probs_grad_out, tokenNum_ * expertNum_);
            probsPipe_.InitBuffer(probGradOutQue_, 1, BLOCK_SIZE);
        }
    }
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::Process()
{
    ProcessTokenGrad();
    if (haveProbs_ != 0) {
        ProcessProbsGrad();
    }
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::ProcessTokenGrad()
{
    if (blockIdx_ >= tokenUsedCoreNum_) {
        return;
    }
    for (int64_t outTokenIdx = 0; outTokenIdx < tokensCoreLength_; ++outTokenIdx) {
        CalSingleOutToken(outTokenIdx, tokenStart_ + outTokenIdx);
    }
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::ProcessProbsGrad()
{
    if (blockIdx_ >= probsUsedCoreNum_) {
        return;
    }
    LocalTensor<ProbT> probGradOutTensor = probGradOutQue_.template AllocTensor<ProbT>();
    int64_t startIdx = blockIdx_ * probsSingleCoreLen_;
    int64_t processNum = probsSingleCoreLen_;
    probCopyParams_.blockLen = sizeof(ProbT);
    if (blockIdx_ + 1 == probsUsedCoreNum_) {
        processNum = probsLastCoreLen_;
    }
    int64_t endIdx = startIdx + processNum;
    event_t eventIDSToMTE3 = static_cast<event_t>(probsPipe_.FetchEventID(HardEvent::S_MTE3));
    ProbT permuteProbsOutputTmp;
    IdxT xIndex;
    int64_t yIndex;
    int64_t outOffset;
    for (int64_t i = startIdx; i < endIdx; i++) {
        if (i >= numOutTokens_) {
            break;
        }
        permuteProbsOutputTmp = permutedProbsGM_.GetValue(i);
        xIndex = sortedIndicesGM_.GetValue(i);
        yIndex = i / capacity_;
        if (xIndex < 0 || xIndex >= tokenNum_ || yIndex < 0 || yIndex >= expertNum_) {
            continue;
        }
        outOffset = xIndex * expertNum_ + yIndex;
        if (outOffset < 0 || outOffset >= tokenNum_ * expertNum_) {
            continue;
        }
        probGradOutTensor.SetValue(0, permuteProbsOutputTmp);
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        DataCopyPad(probsGradOutGM_[outOffset], probGradOutTensor, probCopyParams_);
    }
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::CalSingleOutToken(
    const int64_t out_token_idx, const int64_t global_token_id)
{
    for (int64_t hIndex = 0; hIndex < hiddenSplitedNum_; ++hIndex) {
        CalPartOutToken(hIndex, hiddenSplitedLength_, out_token_idx, global_token_id);
    }
    if (hiddenSplitedRemain_ > 0) {
        CalPartOutToken(hiddenSplitedNum_, hiddenSplitedRemain_, out_token_idx, global_token_id);
    }
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::CalPartOutToken(
    const int64_t h_index, const int64_t h_length, const int64_t out_token_index, const int64_t global_token_id)
{
    if constexpr (IsSameType<TokenT, float>::value) {
        tokenTensor0_ = tokenOutQue_.template AllocTensor<TokenT>();
    }

    bool hasFirst = false;
    for (int64_t permutedIdx = 0; permutedIdx < numOutTokens_; ++permutedIdx) {
        IdxT tokenIdx = sortedIndicesGM_.GetValue(permutedIdx);
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

    CopyTokenOut(out_token_index, h_index, h_length);
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::CopyTokenIn(
    const IdxT in_token_index, const int64_t h_index, const int64_t h_length)
{
    if (in_token_index < 0 || in_token_index >= numOutTokens_) {
        return;
    }
    LocalTensor<TokenT> tokensLocal = tokensInQue_.template AllocTensor<TokenT>();
    int64_t offset = in_token_index * hiddenSize_ + h_index * hiddenSplitedLength_;

    if (likely((h_length * sizeof(TokenT)) % BLOCK_SIZE == 0)) {
        DataCopy(tokensLocal, permutedTokenGradGM_[offset], h_length);
    } else {
        tokenCopyParams_.blockLen = h_length * sizeof(TokenT);
#if __CCE_AICORE__ == 200
        MoeTokenUnPermute::DataCopyPadCustom(tokensLocal, permutedTokenGradGM_[offset], tokenCopyParams_, tokenExtParams_);
#else
        DataCopyPad(tokensLocal, permutedTokenGradGM_[offset], tokenCopyParams_, tokenExtParams_);
#endif
    }
    tokensInQue_.template EnQue(tokensLocal);
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::CalFirstToken(const int64_t h_length)
{
    LocalTensor<TokenT> tokensLocal = tokensInQue_.template DeQue<TokenT>();

    if constexpr (!IsSameType<TokenT, float>::value) {
        Cast(tokenTensor0_, tokensLocal, RoundMode::CAST_NONE, h_length);
    } else {
        uint64_t byteAlign32 = (h_length * sizeof(float) + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
        DataCopy(tokenTensor0_, tokensLocal, byteAlign32 / sizeof(float));
    }
    tokensInQue_.FreeTensor(tokensLocal);
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::CalToken(const int64_t h_length)
{
    LocalTensor<TokenT> tokensLocal = tokensInQue_.template DeQue<TokenT>();

    if constexpr (!IsSameType<TokenT, float>::value) {
        Cast(tokenTensor1_, tokensLocal, RoundMode::CAST_NONE, h_length);
        tokensInQue_.FreeTensor(tokensLocal);
        PipeBarrier<PIPE_V>();
        Add(tokenTensor0_, tokenTensor0_, tokenTensor1_, h_length);
    } else {
        Add(tokenTensor0_, tokenTensor0_, tokensLocal, h_length);
        tokensInQue_.FreeTensor(tokensLocal);
    }
}

template <typename TokenT, typename ProbT, typename IdxT>
__aicore__ inline void KernelMoeTokenPermuteDropPadGrad<TokenT, ProbT, IdxT>::CopyTokenOut(
    const int64_t out_token_index, const int64_t h_index, const int64_t h_length)
{
    LocalTensor<TokenT> tempOutTensors;
    if constexpr (!IsSameType<TokenT, float>::value) {
        tempOutTensors = tokenOutQue_.template AllocTensor<TokenT>();
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        Cast(tempOutTensors, tokenTensor0_, RoundMode::CAST_NONE, h_length);
#else
        Cast(tempOutTensors, tokenTensor0_, RoundMode::CAST_RINT, h_length);
#endif
    } else {
        tempOutTensors = tokenTensor0_;
    }

    tokenOutQue_.template EnQue<TokenT>(tempOutTensors);
    tempOutTensors = tokenOutQue_.template DeQue<TokenT>();

    int64_t offset = out_token_index * hiddenSize_ + h_index * hiddenSplitedLength_;
    if (likely((h_length * sizeof(TokenT)) % BLOCK_SIZE == 0)) {
        DataCopy(tokensGradOutGM_[offset], tempOutTensors, h_length);
    } else {
        tokenCopyParams_.blockLen = h_length * sizeof(TokenT);
#if __CCE_AICORE__ == 200
        MoeTokenUnPermute::DataCopyCustom(
            tokensGradOutGM_[offset], tempOutTensors, tokenCopyParams_.blockCount, tokenCopyParams_.blockLen);
#else
        DataCopyPad(tokensGradOutGM_[offset], tempOutTensors, tokenCopyParams_);
#endif
    }
    tokenOutQue_.FreeTensor(tempOutTensors);
}

#endif // MOE_TOKEN_PERMUTE_DROP_PAD_GRAD_H
