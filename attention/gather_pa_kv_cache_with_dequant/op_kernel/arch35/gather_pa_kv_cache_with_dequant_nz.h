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
 * \file gather_pa_kv_cache_with_dequant_nz.h
 * \brief NZ format kernel for fused GatherPaKvCache + Dequant
 */

#ifndef GATHER_PA_KV_CACHE_WITH_DEQUANT_NZ_H_
#define GATHER_PA_KV_CACHE_WITH_DEQUANT_NZ_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GatherPaKvCacheWithDequantV35 {
using namespace AscendC;

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

/**
 * NZ format kernel for fused gather + dequant
 * Key cache: INT8 FRACTAL_NZ format
 * Value cache: FP8 E4M3FN FRACTAL_NZ format
 * Output: BF16
 */
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheWithDequantNz {
    static constexpr int32_t BLOCK_SIZE = 32;
    static constexpr uint32_t DOUBLE_BUFFER = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;
    static constexpr uint8_t CONST_ONE = 1;
    static constexpr uint8_t DIM_TWO = 2;
    static constexpr uint8_t DIM_THREE = 3;
    static constexpr int32_t MAX_BLOCK_CNT = 4095;
    static constexpr uint32_t VL_NUMS = GetVRegSize() / sizeof(uint8_t);

public:
    __aicore__ inline GatherPaKvCacheWithDequantNz(TPipe *pipe, const GatherPaKvCacheWithDequantTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling){};

    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR key_scale_cache, GM_ADDR value_cache,
                                GM_ADDR value_scale, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset,
                                GM_ADDR key_out, GM_ADDR value_out)
    {
        InitParams();

        keyCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)(key_cache));
        keyScaleCacheGm_.SetGlobalBuffer((__gm__ float *)(key_scale_cache));
        valueCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)(value_cache));
        valueScaleGm_.SetGlobalBuffer((__gm__ float *)(value_scale));
        blockTablesGm_.SetGlobalBuffer((__gm__ T_INDEX *)block_tables);
        seqLensGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_lens));
        seqOffsetGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_offset));

        outKeyGm_.SetGlobalBuffer((__gm__ uint16_t *)(key_out));
        outValueGm_.SetGlobalBuffer((__gm__ uint16_t *)(value_out));

        pipe_->InitBuffer(cacheQueue_, DOUBLE_BUFFER, maxUbHiddenSize_ * sizeof(uint8_t));
        pipe_->InitBuffer(seqLensQueue_, DOUBLE_BUFFER, seqLenAccumSize_ * sizeof(T_INDEX));
        pipe_->InitBuffer(sumBuffer, sizeof(T_INDEX));
        pipe_->InitBuffer(scaleBuffer_, 256 * sizeof(float));
    }

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_DEPTH> cacheQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> seqLensQueue_;
    TBuf<TPosition::VECCALC> sumBuffer;
    TBuf<TPosition::VECCALC> scaleBuffer_;

    const GatherPaKvCacheWithDequantTilingDataV35 *tl_;

    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<float> keyScaleCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    GlobalTensor<float> valueScaleGm_;
    GlobalTensor<T_INDEX> blockTablesGm_;
    GlobalTensor<T_INDEX> seqLensGm_;
    GlobalTensor<T_INDEX> seqOffsetGm_;
    GlobalTensor<uint16_t> outKeyGm_;
    GlobalTensor<uint16_t> outValueGm_;

    uint32_t eleNumPerBlk = BLOCK_SIZE / sizeof(uint8_t);
    uint32_t blockSize_;
    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    uint64_t batchCount_;
    int64_t blockTableWidth_;
    uint64_t maxUbHiddenSizeK_;
    uint64_t maxUbHiddenSizeV_;
    uint64_t maxUbHiddenSize_;
    uint64_t hiddenSizeK_;
    uint64_t hiddenSizeV_;
    int64_t numTokens_;
    int64_t numBlocks_;
    uint32_t seqLenAccumSize_;
    uint64_t numHeadsK_;
    uint64_t numHeadsV_;
    uint64_t headDimK_;
    uint64_t headDimV_;

    __aicore__ inline void InitParams()
    {
        blockSize_ = tl_->kvCacheBlockSize;
        batchPerCore_ = tl_->batchPerCore;
        needCoreNum_ = tl_->needCoreNum;
        batchCount_ = tl_->batchCount;
        blockTableWidth_ = tl_->blockTableWidth;
        maxUbHiddenSizeK_ = tl_->maxUbHiddenSizeK;
        maxUbHiddenSizeV_ = tl_->maxUbHiddenSizeV;
        maxUbHiddenSize_ = tl_->maxUbHiddenSize;
        hiddenSizeK_ = tl_->hiddenSizeK;
        hiddenSizeV_ = tl_->hiddenSizeV;
        numTokens_ = tl_->numTokens;
        numBlocks_ = tl_->numBlocks;
        seqLenAccumSize_ = tl_->seqLenAccumSize;
        numHeadsK_ = tl_->numHeadsK;
        numHeadsV_ = tl_->numHeadsV;
        headDimK_ = tl_->headDimK;
        headDimV_ = tl_->headDimV;
    }

public:
    __aicore__ inline void Process()
    {
        uint32_t batchStart = GetBlockIdx() * batchPerCore_;
        uint32_t batchEnd = batchStart + batchPerCore_;
        if (GetBlockIdx() == needCoreNum_ - 1) {
            batchEnd = batchCount_;
        }

        int64_t coreOffset_;
        if constexpr (isSeqLensCumsum) {
            coreOffset_ = seqLensGm_.GetValue(batchStart);
        } else {
            coreOffset_ = CalcKvCoreOffset(batchStart);
        }

        uint64_t keyCacheOffset = 0;
        uint64_t valueCacheOffset = 0;
        for (uint32_t i = batchStart; i < batchEnd; i++) {
            T_INDEX seqLen;
            uint32_t batchOffset, accumSeqLen;
            if constexpr (isSeqLensCumsum) {
                seqLen = seqLensGm_.GetValue(i + 1) - seqLensGm_.GetValue(i);
                batchOffset = seqLensGm_.GetValue(i);
                accumSeqLen = seqLensGm_.GetValue(i + 1);
            } else {
                seqLen = seqLensGm_.GetValue(i);
                if (i == batchStart) {
                    batchOffset = coreOffset_;
                    accumSeqLen = coreOffset_ + seqLensGm_.GetValue(i);
                } else {
                    batchOffset += seqLensGm_.GetValue(i - 1);
                    accumSeqLen += seqLensGm_.GetValue(i);
                }
            }

            if (batchOffset >= numTokens_) {
                break;
            }

            if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
                seqLen = numTokens_ - batchOffset;
            }

            T_INDEX seqOffset = 0;
            if constexpr (hasSeqOffset) {
                seqOffset = seqOffsetGm_.GetValue(i) / blockSize_;
            }

            uint32_t blockCount = CeilDivision(seqLen, blockSize_);

            // Process key: INT8 NZ -> BF16
            uint64_t keyOffset = batchOffset * hiddenSizeK_;
            for (uint32_t j = 0; j < blockCount; j++) {
                uint32_t curLen = blockSize_;
                if (j == blockCount - 1) {
                    curLen = seqLen - (blockCount - 1) * blockSize_;
                }

                uint64_t blockTableOffset = j + seqOffset;
                bool isFilledWithZero;
                T_INDEX blockId;
                if (blockTableOffset >= blockTableWidth_ || blockTableOffset < 0) {
                    isFilledWithZero = true;
                    blockId = 0;
                } else {
                    isFilledWithZero = false;
                    blockId = blockTablesGm_.GetValue(blockTableWidth_ * i + seqOffset + j);
                    if (blockId >= numBlocks_ || blockId < 0) {
                        isFilledWithZero = true;
                        blockId = 0;
                    }
                }

                keyCacheOffset = blockId * blockSize_ * hiddenSizeK_;
                
                // TODO: Implement NZ format gather + INT8 dequant
                // This is a placeholder - actual implementation would handle NZ format
                // and perform fused gather + dequantization
                
                keyOffset += curLen * hiddenSizeK_;
            }

            // Process value: FP8 NZ -> BF16
            uint64_t valueOffset = batchOffset * hiddenSizeV_;
            for (uint32_t j = 0; j < blockCount; j++) {
                uint32_t curLen = blockSize_;
                if (j == blockCount - 1) {
                    curLen = seqLen - (blockCount - 1) * blockSize_;
                }

                uint64_t blockTableOffset = j + seqOffset;
                bool isFilledWithZero;
                T_INDEX blockId;
                if (blockTableOffset >= blockTableWidth_ || blockTableOffset < 0) {
                    isFilledWithZero = true;
                    blockId = 0;
                } else {
                    isFilledWithZero = false;
                    blockId = blockTablesGm_.GetValue(blockTableWidth_ * i + seqOffset + j);
                    if (blockId >= numBlocks_ || blockId < 0) {
                        isFilledWithZero = true;
                        blockId = 0;
                    }
                }

                valueCacheOffset = blockId * blockSize_ * hiddenSizeV_;
                
                // TODO: Implement NZ format gather + FP8 dequant
                // This is a placeholder - actual implementation would handle NZ format
                // and perform fused gather + dequantization
                
                valueOffset += curLen * hiddenSizeV_;
            }
        }
    }

private:
    __aicore__ inline T_INDEX CalcKvCoreOffset(uint32_t batchStart)
    {
        LocalTensor<T_INDEX> sumLocal = sumBuffer.Get<T_INDEX>();
        uint32_t reduceLoops = CeilDivision(batchStart, seqLenAccumSize_);
        DataCopyPadExtParams<T_INDEX> padParams = {false, 0, 0, 0};
        DataCopyExtParams seqLensCopyParams;
        seqLensCopyParams.blockCount = 1;
        seqLensCopyParams.srcStride = 0;
        seqLensCopyParams.dstStride = 0;
        uint32_t seqOffset, seqLength;
        uint32_t curLen = batchStart;

        T_INDEX coreOffset = 0;
        for (uint32_t i = 0; i < reduceLoops; i++) {
            seqOffset = i * seqLenAccumSize_;
            seqLength = seqLenAccumSize_;
            if (i == reduceLoops - 1) {
                seqLength = batchStart - (reduceLoops - 1) * seqLenAccumSize_;
            }
            seqLensCopyParams.blockLen = static_cast<uint32_t>(seqLength * sizeof(T_INDEX));
            LocalTensor<T_INDEX> accumSeqLenLocal = seqLensQueue_.AllocTensor<T_INDEX>();
            DataCopyPad(accumSeqLenLocal, seqLensGm_[seqOffset], seqLensCopyParams, padParams);
            seqLensQueue_.EnQue<T_INDEX>(accumSeqLenLocal);
            accumSeqLenLocal = seqLensQueue_.DeQue<T_INDEX>();

            uint32_t srcShape[2] = {uint32_t(1), uint32_t(seqLength)};
            AscendC::ReduceSum<T_INDEX, Pattern::Reduce::AR, true>(sumLocal, accumSeqLenLocal, srcShape, false);
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            seqLensQueue_.FreeTensor<T_INDEX>(accumSeqLenLocal);
            coreOffset += sumLocal.GetValue(0);
        }
        return coreOffset;
    }
};

} // namespace GatherPaKvCacheWithDequantV35
#endif
