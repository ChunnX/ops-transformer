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
 * \file gather_pa_kv_cache_with_dequant_nd.h
 * \brief Fused GatherPaKvCache + Dequantization kernel (ND format)
 */

#ifndef GATHER_PA_KV_CACHE_WITH_DEQUANT_ND_H_
#define GATHER_PA_KV_CACHE_WITH_DEQUANT_ND_H_

#include "kernel_operator.h"

namespace GatherPaKvCacheWithDequantV35 {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint64_t MAX_BLOCK_CNT = 4095;

// Fused kernel: gather INT8 key cache + dequantize to BF16
// key_cache: INT8 [num_blocks, block_size, num_heads, head_dim]
// key_scale_cache: FLOAT [num_blocks, block_size, num_heads]
// output key: BF16 [num_tokens, num_heads, head_dim]
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheWithDequantNd {
public:
    __aicore__ inline GatherPaKvCacheWithDequantNd(TPipe *pipe, const GatherPaKvCacheWithDequantTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling){};
    
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR key_scale_cache, GM_ADDR value_cache,
                                GM_ADDR value_scale, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset,
                                GM_ADDR key_out, GM_ADDR value_out);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> cacheQueue_;
    TQue<QuePosition::VECIN, 1> seqLensQueue_;
    TBuf<TPosition::VECCALC> prefixSumBuffer_;
    TBuf<TPosition::VECCALC> scaleBuffer_;

    DataCopyPadExtParams<int8_t> padExtParams_;

    const GatherPaKvCacheWithDequantTilingDataV35 *tl_;

    GlobalTensor<int8_t> keyCacheGm_;
    GlobalTensor<float> keyScaleCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;  // FP8 E4M3FN stored as uint8
    GlobalTensor<float> valueScaleGm_;
    GlobalTensor<T_INDEX> blockTablesGm_;
    GlobalTensor<T_INDEX> seqLensGm_;
    GlobalTensor<T_INDEX> seqOffsetGm_;
    GlobalTensor<uint16_t> outKeyGm_;  // BF16 output
    GlobalTensor<uint16_t> outValueGm_;  // BF16 output

    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    int64_t batchCount_;
    uint32_t seqLenAccSize_;
    int64_t cacheBlockSize_;
    int64_t blockTableWidth_;
    uint64_t maxUbHiddenSizeK_;
    uint64_t maxUbHiddenSizeV_;
    uint64_t maxUbHiddenSize_;
    int64_t numBlocks_;
    uint64_t hiddenSizeK_;  // in elements (num_heads * head_dim)
    uint64_t hiddenSizeV_;
    int64_t numTokens_;
    uint64_t numHeadsK_;
    uint64_t numHeadsV_;
    uint64_t headDimK_;
    uint64_t headDimV_;

    __aicore__ inline void InitParams();
    __aicore__ inline T_INDEX CalcKvCoreOffset(int64_t reduceLen);
    __aicore__ inline void GatherAndDequantKey(GlobalTensor<uint16_t> dstKeyGm, GlobalTensor<int8_t> srcKeyCacheGm,
                                                GlobalTensor<float> srcKeyScaleGm, uint64_t curLen, uint64_t blockId,
                                                bool isFilledWithZero);
    __aicore__ inline void GatherAndDequantValue(GlobalTensor<uint16_t> dstValueGm, GlobalTensor<uint8_t> srcValueCacheGm,
                                                  GlobalTensor<float> srcValueScaleGm, uint64_t curLen, uint64_t blockId,
                                                  bool isFilledWithZero);
};

template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::Init(
    GM_ADDR key_cache, GM_ADDR key_scale_cache, GM_ADDR value_cache, GM_ADDR value_scale,
    GM_ADDR block_tables, GM_ADDR seq_lens, GM_ADDR key_in, GM_ADDR value_in,
    GM_ADDR seq_offset, GM_ADDR key_out, GM_ADDR value_out)
{
    InitParams();

    keyCacheGm_.SetGlobalBuffer((__gm__ int8_t *)(key_cache));
    keyScaleCacheGm_.SetGlobalBuffer((__gm__ float *)(key_scale_cache));
    valueCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)(value_cache));
    valueScaleGm_.SetGlobalBuffer((__gm__ float *)(value_scale));
    blockTablesGm_.SetGlobalBuffer((__gm__ T_INDEX *)block_tables);
    seqLensGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_lens));
    seqOffsetGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_offset));

    outKeyGm_.SetGlobalBuffer((__gm__ uint16_t *)(key_out));
    outValueGm_.SetGlobalBuffer((__gm__ uint16_t *)(value_out));

    pipe_->InitBuffer(cacheQueue_, DOUBLE_BUFFER, maxUbHiddenSize_ * sizeof(int8_t));
    pipe_->InitBuffer(seqLensQueue_, DOUBLE_BUFFER, seqLenAccSize_ * sizeof(T_INDEX));
    pipe_->InitBuffer(prefixSumBuffer_, BLOCK_SIZE);
    pipe_->InitBuffer(scaleBuffer_, 256 * sizeof(float));  // Buffer for scales
}

template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::InitParams()
{
    cacheBlockSize_ = tl_->kvCacheBlockSize;
    batchPerCore_ = tl_->batchPerCore;
    needCoreNum_ = tl_->needCoreNum;
    batchCount_ = tl_->batchCount;
    blockTableWidth_ = tl_->blockTableWidth;
    maxUbHiddenSizeK_ = tl_->maxUbHiddenSizeK;
    maxUbHiddenSizeV_ = tl_->maxUbHiddenSizeV;
    maxUbHiddenSize_ = tl_->maxUbHiddenSize;
    seqLenAccSize_ = tl_->seqLenAccumSize;
    numBlocks_ = tl_->numBlocks;
    hiddenSizeK_ = tl_->hiddenSizeK;
    hiddenSizeV_ = tl_->hiddenSizeV;
    numTokens_ = tl_->numTokens;
    numHeadsK_ = tl_->numHeadsK;
    numHeadsV_ = tl_->numHeadsV;
    headDimK_ = tl_->headDimK;
    headDimV_ = tl_->headDimV;

    padExtParams_.isPad = false;
    padExtParams_.leftPadding = 0;
    padExtParams_.rightPadding = 0;
    padExtParams_.paddingValue = 0;
}

template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::Process()
{
    int64_t batchStart = GetBlockIdx() * batchPerCore_;
    int64_t batchEnd = batchStart + batchPerCore_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        batchEnd = batchCount_;
    }

    int64_t coreOffset;
    if constexpr (isSeqLensCumsum) {
        coreOffset = seqLensGm_.GetValue(batchStart);
    } else {
        coreOffset = CalcKvCoreOffset(batchStart);
    }

    for (uint32_t i = batchStart; i < batchEnd; i++) {
        T_INDEX seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
            seqLen = seqLensGm_.GetValue(i + 1) - seqLensGm_.GetValue(i);
            batchOffset = seqLensGm_.GetValue(i);
            accumSeqLen = seqLensGm_.GetValue(i + 1);
        } else {
            seqLen = seqLensGm_.GetValue(i);
            batchOffset = coreOffset;
            accumSeqLen = coreOffset + seqLensGm_.GetValue(i);
            coreOffset = accumSeqLen;
        }

        if (batchOffset >= numTokens_) {
            break;
        }

        if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
            seqLen = numTokens_ - batchOffset;
        }

        uint64_t seqOffset = 0;
        if constexpr (hasSeqOffset) {
            seqOffset = seqOffsetGm_.GetValue(i) / cacheBlockSize_;
        }

        uint64_t blockCount = CeilDivision(seqLen, cacheBlockSize_);
        uint64_t keyOffset = batchOffset * hiddenSizeK_;
        uint64_t valueOffset = batchOffset * hiddenSizeV_;

        for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++) {
            uint64_t blockTableOffset = blockIdx + seqOffset;
            bool isFilledWithZero;
            int64_t blockId;
            if ((blockTableOffset >= blockTableWidth_) || (blockTableOffset < 0)) {
                isFilledWithZero = true;
                blockId = 0;
            } else {
                isFilledWithZero = false;
                blockId = blockTablesGm_.GetValue(blockTableWidth_ * i + blockTableOffset);
                if (blockId >= numBlocks_ || blockId < 0) {
                    isFilledWithZero = true;
                    blockId = 0;
                }
            }

            uint64_t curBlockSize = cacheBlockSize_;
            if (blockIdx == blockCount - 1) {
                curBlockSize = seqLen - (blockCount - 1) * cacheBlockSize_;
            }

            // Fused gather + dequant for key (INT8 -> BF16)
            GatherAndDequantKey(outKeyGm_[keyOffset], keyCacheGm_, keyScaleCacheGm_,
                               curBlockSize, blockId, isFilledWithZero);
            keyOffset += curBlockSize * hiddenSizeK_;

            // Fused gather + dequant for value (FP8 -> BF16)
            GatherAndDequantValue(outValueGm_[valueOffset], valueCacheGm_, valueScaleGm_,
                                 curBlockSize, blockId, isFilledWithZero);
            valueOffset += curBlockSize * hiddenSizeV_;
        }
    }
}

template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline T_INDEX
GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::CalcKvCoreOffset(int64_t reduceLen)
{
    LocalTensor<T_INDEX> prefixSumLocal = prefixSumBuffer_.Get<T_INDEX>();
    uint64_t loopTimes = CeilDivision(reduceLen, seqLenAccSize_);
    DataCopyPadExtParams<T_INDEX> padParams = {false, 0, 0, 0};
    DataCopyExtParams seqLensCopyParams;
    seqLensCopyParams.blockCount = 1;
    seqLensCopyParams.srcStride = 0;
    seqLensCopyParams.dstStride = 0;
    uint32_t seqOffset, seqLength;

    T_INDEX coreOffset = 0;
    for (uint64_t i = 0; i < loopTimes; i++) {
        seqOffset = i * seqLenAccSize_;
        seqLength = seqLenAccSize_;
        if (i == loopTimes - 1) {
            seqLength = reduceLen - (loopTimes - 1) * seqLenAccSize_;
        }
        seqLensCopyParams.blockLen = static_cast<uint32_t>(seqLength * sizeof(T_INDEX));
        LocalTensor<T_INDEX> seqLenLocal = seqLensQueue_.AllocTensor<T_INDEX>();
        DataCopyPad(seqLenLocal, seqLensGm_[seqOffset], seqLensCopyParams, padParams);
        seqLensQueue_.EnQue<T_INDEX>(seqLenLocal);

        seqLenLocal = seqLensQueue_.DeQue<T_INDEX>();
        uint32_t srcShape[2] = {uint32_t(1), seqLength};
        AscendC::ReduceSum<T_INDEX, Pattern::Reduce::AR, true>(prefixSumLocal, seqLenLocal, srcShape, false);
        AscendC::TEventID eventIdVecToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIdVecToS);
        WaitFlag<HardEvent::V_S>(eventIdVecToS);
        seqLensQueue_.FreeTensor<T_INDEX>(seqLenLocal);
        coreOffset += prefixSumLocal.GetValue(0);
    }

    return coreOffset;
}

// Fused gather + dequant for key: INT8 -> BF16 with per-token-per-head scale
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherAndDequantKey(
    GlobalTensor<uint16_t> dstKeyGm, GlobalTensor<int8_t> srcKeyCacheGm, GlobalTensor<float> srcKeyScaleGm,
    uint64_t curLen, uint64_t blockId, bool isFilledWithZero)
{
    // For each token in the block
    for (uint64_t token = 0; token < curLen; token++) {
        // For each head
        for (uint64_t head = 0; head < numHeadsK_; head++) {
            // Calculate source offset in key_cache
            uint64_t srcOffset = (blockId * cacheBlockSize_ * numHeadsK_ * headDimK_) +
                                 (token * numHeadsK_ * headDimK_) +
                                 (head * headDimK_);
            
            // Get scale for this token-head
            uint64_t scaleOffset = (blockId * cacheBlockSize_ * numHeadsK_) +
                                   (token * numHeadsK_) + head;
            float scale = srcKeyScaleGm.GetValue(scaleOffset);
            
            // Gather INT8 data and dequantize to BF16
            LocalTensor<int8_t> int8Local = cacheQueue_.AllocTensor<int8_t>();
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = headDimK_ * sizeof(int8_t);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            if (isFilledWithZero) {
                AscendC::TEventID eventIdMTE3ToVec = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
                SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
                WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
                Duplicate<int8_t>(int8Local, 0, headDimK_);
                AscendC::TEventID eventIdVecToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
            } else {
                DataCopyPad<int8_t, PaddingMode::Normal>(int8Local, srcKeyCacheGm[srcOffset], dataCopyParams, padExtParams_);
                cacheQueue_.EnQue<int8_t>(int8Local);
                int8Local = cacheQueue_.DeQue<int8_t>();
            }

            // Cast INT8 to float, multiply by scale, cast to BF16
            LocalTensor<float> floatLocal;
            AscendC::TEventID eventIdMTE3ToVec = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
            
            // TODO: Implement actual dequantization using AscendC intrinsics
            // This is a placeholder - actual implementation would use Cast, Muls, etc.
            
            AscendC::TEventID eventIdVecToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);

            cacheQueue_.FreeTensor(int8Local);
        }
    }
}

// Fused gather + dequant for value: FP8 -> BF16 with per-channel scale
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherAndDequantValue(
    GlobalTensor<uint16_t> dstValueGm, GlobalTensor<uint8_t> srcValueCacheGm, GlobalTensor<float> srcValueScaleGm,
    uint64_t curLen, uint64_t blockId, bool isFilledWithZero)
{
    // Similar to key but for FP8 value cache
    // value_scale shape: [num_heads, head_dim] - per-channel static scale
    for (uint64_t token = 0; token < curLen; token++) {
        for (uint64_t head = 0; head < numHeadsV_; head++) {
            uint64_t srcOffset = (blockId * cacheBlockSize_ * numHeadsV_ * headDimV_) +
                                 (token * numHeadsV_ * headDimV_) +
                                 (head * headDimV_);
            
            // TODO: Implement FP8 dequantization
            // This is a placeholder
        }
    }
}

} // namespace GatherPaKvCacheWithDequantV35
#endif
