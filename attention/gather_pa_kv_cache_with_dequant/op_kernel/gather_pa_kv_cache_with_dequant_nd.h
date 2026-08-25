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
 * \file gather_pa_kv_cache_with_dequant_nd.h
 * \brief Fused GatherPaKvCache + Dequant kernel (ND mode).
 *
 * 融合目标: 把 GatherPaKvCache 的数据搬运与反量化合并到一个 kernel，消除中间 tensor 落 GM 的 MTE3 开销。
 *
 * Key 路径 (pertoken-perhead 动态量化):
 *   key_cache INT8 [num_blocks, block_size, num_heads, head_dim]
 *   key_scale_cache FLOAT [num_blocks, block_size, num_heads]  -- 与 cache 同 block 结构, 需同步 gather
 *   → 在 UB 内完成: INT8 → Cast(FP32) → Mul(scale) → Cast(BF16) → 写出 GM
 *
 * Value 路径 (perhead-perchannel 静态量化):
 *   value_cache INT8/FP8 [num_blocks, block_size, num_heads, head_dim]
 *   value_scale FLOAT [num_heads, head_dim]  -- 连续存储, 按 head 索引即可
 *   → 在 UB 内完成: INT8/FP8 → Cast(FP32) → Mul(scale) → Cast(BF16) → 写出 GM
 */

#ifndef GATHER_PA_KV_CACHE_WITH_DEQUANT_ND_H_
#define GATHER_PA_KV_CACHE_WITH_DEQUANT_ND_H_

#include "kernel_operator.h"

namespace GatherPaKvCacheWithDequantV35 {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint64_t MAX_BLOCK_CNT = 4095;

// UB 缓冲区布局:
//   bufSrc:   INT8/FP8 gather 临时区 (maxChunkElems * 1B)
//   bufFp32:  FP32 中间计算区   (maxChunkElems * 4B)
//   bufBf16:  BF16 输出暂存区   (maxChunkElems * 2B)
//   bufScale: scale 加载区      (maxChunkHeads * 4B, 通常远小于数据区)
// 总 UB 消耗 = maxChunkElems * 7B + scale 区

template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheWithDequantNd {
public:
    __aicore__ inline GatherPaKvCacheWithDequantNd(TPipe *pipe,
                                                    const GatherPaKvCacheWithDequantTilingDataV35 *tiling)
        : pipe_(pipe), tl_(tiling) {}

    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache,
                                GM_ADDR key_scale_cache, GM_ADDR value_scale,
                                GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in,
                                GM_ADDR seq_offset,
                                GM_ADDR key_out, GM_ADDR value_out);

    __aicore__ inline void Process();

private:
    // ===== 核心: gather + dequant 一体化 =====
    // 从 key_cache GM 读 INT8 数据到 UB, 同时读 scale, 在 UB 内完成反量化, 以 BF16 写出
    __aicore__ inline void GatherDequantKey(GlobalTensor<bfloat16_t> dstGm,
                                            GlobalTensor<int8_t> srcGm,
                                            GlobalTensor<float> scaleGm,
                                            uint64_t curBlockSize, uint64_t blockId,
                                            bool isFilledWithZero);

    // value 路径: 从 value_cache GM 读 INT8 到 UB, 用 value_scale 完成反量化
    __aicore__ inline void GatherDequantValue(GlobalTensor<bfloat16_t> dstGm,
                                              GlobalTensor<int8_t> srcGm,
                                              GlobalTensor<float> scaleGm,
                                              uint64_t curBlockSize, uint64_t blockId,
                                              bool isFilledWithZero);

    // 对单个 head 内的连续数据做 gather + dequant
    // srcByteOff: 源 GM 字节偏移
    // scaleOff:   scale GM 偏移 (元素粒度)
    // elemCount:  要处理的元素数
    // scaleVal:   当 scale 为 per-token-perhead 时, 每个 token 有独立 scale
    // isKey:      true=key路径(逐token读scale), false=value路径(整块共用scale)
    __aicore__ inline void GatherDequantChunk(GlobalTensor<bfloat16_t> dstGm,
                                              GlobalTensor<int8_t> srcGm,
                                              GlobalTensor<float> scaleGm,
                                              uint64_t dstByteOff,
                                              uint64_t srcByteOff,
                                              uint64_t scaleBaseOff,
                                              uint64_t elemCount,
                                              uint64_t numHeads,
                                              uint64_t headDim,
                                              bool isKey,
                                              bool isFilledWithZero);

    __aicore__ inline void InitParams();

    __aicore__ inline T_INDEX CalcKvCoreOffset(int64_t reduceLen);

private:
    TPipe *pipe_;
    const GatherPaKvCacheWithDequantTilingDataV35 *tl_;

    // QueBind: 输入输出共享一个队列, 先读后写
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQueue_;
    TQue<QuePosition::VECIN, 1> scaleQueue_;
    TBuf<TPosition::VECCALC> fp32Buf_;   // FP32 中间缓冲
    TBuf<TPosition::VECCALC> prefixSumBuffer_;

    DataCopyPadExtParams<int8_t> padExtParams_;

    // GM tensors
    GlobalTensor<int8_t> keyCacheGm_;
    GlobalTensor<int8_t> valueCacheGm_;
    GlobalTensor<float> keyScaleCacheGm_;   // key 动态 scale [num_blocks, block_size, num_heads]
    GlobalTensor<float> valueScaleGm_;       // value 静态 scale [num_heads, head_dim]
    GlobalTensor<T_INDEX> blockTablesGm_;
    GlobalTensor<T_INDEX> seqLensGm_;
    GlobalTensor<T_INDEX> seqOffsetGm_;
    GlobalTensor<bfloat16_t> outKeyGm_;
    GlobalTensor<bfloat16_t> outValueGm_;

    // tiling 参数
    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    int64_t batchCount_;
    uint32_t seqLenAccSize_;
    int64_t cacheBlockSize_;
    int64_t blockTableWidth_;
    int64_t numBlocks_;
    int64_t numTokens_;

    uint64_t hiddenSizeK_;    // bytes: num_heads * head_dim * sizeof(int8_t)
    uint64_t hiddenSizeV_;    // bytes: num_heads * head_dim * sizeof(int8_t)
    uint64_t hiddenSizeKOut_; // bytes: num_heads * head_dim * sizeof(bfloat16_t)
    uint64_t hiddenSizeVOut_;
    uint64_t numHeadsK_;
    uint64_t numHeadsV_;
    uint64_t headDimK_;       // elements
    uint64_t headDimV_;       // elements

    // UB 容量相关
    uint64_t maxUbChunkElems_;   // 单次能处理的最大元素数 (由 UB 容量决定)
    uint64_t scaleTokenStride_;  // key scale 在 token 维的 stride (元素) = numHeadsK

    // 非连续支持 (简化版, 仅处理连续 ND)
    uint32_t nonContiguousFlag_;
    int64_t kCacheStride0_, kCacheStride1_, kCacheStride2_;
    int64_t vCacheStride0_, vCacheStride1_, vCacheStride2_;
    int64_t keyOutStride0_, keyOutStride1_;
    int64_t valueOutStride0_, valueOutStride1_;
};

// ============================================================================
// Init
// ============================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::Init(
    GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR key_scale_cache, GM_ADDR value_scale,
    GM_ADDR block_tables, GM_ADDR seq_lens,
    GM_ADDR key_in, GM_ADDR value_in,
    GM_ADDR seq_offset,
    GM_ADDR key_out, GM_ADDR value_out)
{
    InitParams();
    (void)key_in;    // inplace: 输出直接写 key_out
    (void)value_in;  // inplace: 输出直接写 value_out

    keyCacheGm_.SetGlobalBuffer((__gm__ int8_t *)(key_cache));
    valueCacheGm_.SetGlobalBuffer((__gm__ int8_t *)(value_cache));
    keyScaleCacheGm_.SetGlobalBuffer((__gm__ float *)(key_scale_cache));
    valueScaleGm_.SetGlobalBuffer((__gm__ float *)(value_scale));
    blockTablesGm_.SetGlobalBuffer((__gm__ T_INDEX *)(block_tables));
    seqLensGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_lens));
    seqOffsetGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_offset));
    outKeyGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(key_out));
    outValueGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(value_out));

    // UB 布局:
    //   dataQueue_ (double buffer): maxUbChunkElems_ * sizeof(int8_t) 每个 buffer
    //   fp32Buf_:                   maxUbChunkElems_ * sizeof(float)
    //   scaleQueue_:                cacheBlockSize_ * max(numHeadsK_, numHeadsV_) * sizeof(float)
    //   prefixSumBuffer_:           64 bytes
    pipe_->InitBuffer(dataQueue_, DOUBLE_BUFFER, maxUbChunkElems_ * sizeof(bfloat16_t));
    pipe_->InitBuffer(fp32Buf_, maxUbChunkElems_ * sizeof(float));
    // scale 每次最多加载一个 token 的 scale (numHeads 个 float)
    uint64_t scaleBufSize = cacheBlockSize_ * numHeadsK_ * sizeof(float);
    if (scaleBufSize < cacheBlockSize_ * numHeadsV_ * sizeof(float)) {
        scaleBufSize = cacheBlockSize_ * numHeadsV_ * sizeof(float);
    }
    pipe_->InitBuffer(scaleQueue_, 1, scaleBufSize);
    pipe_->InitBuffer(prefixSumBuffer_, BLOCK_SIZE);

    padExtParams_.isPad = false;
    padExtParams_.leftPadding = 0;
    padExtParams_.rightPadding = 0;
    padExtParams_.paddingValue = 0;
}

// ============================================================================
// InitParams
// ============================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::InitParams()
{
    cacheBlockSize_ = tl_->kvCacheBlockSize;
    batchPerCore_ = tl_->batchPerCore;
    needCoreNum_ = tl_->needCoreNum;
    batchCount_ = tl_->batchCount;
    blockTableWidth_ = tl_->blockTableWidth;
    numBlocks_ = tl_->numBlocks;
    numTokens_ = tl_->numTokens;

    hiddenSizeK_ = tl_->hiddenSizeK;        // bytes (INT8)
    hiddenSizeV_ = tl_->hiddenSizeV;
    hiddenSizeKOut_ = tl_->hiddenSizeKOut;   // bytes (BF16)
    hiddenSizeVOut_ = tl_->hiddenSizeVOut;
    numHeadsK_ = tl_->numHeadsK;
    numHeadsV_ = tl_->numHeadsV;
    headDimK_ = tl_->headDimK;
    headDimV_ = tl_->headDimV;

    maxUbChunkElems_ = tl_->maxUbChunkElems;
    scaleTokenStride_ = numHeadsK_;  // key scale 的 token 维 stride

    seqLenAccSize_ = tl_->seqLenAccumSize;

    // 非连续
    nonContiguousFlag_ = tl_->nonContiguousFlag;
    kCacheStride0_ = tl_->kCacheStride0;
    kCacheStride1_ = tl_->kCacheStride1;
    kCacheStride2_ = tl_->kCacheStride2;
    vCacheStride0_ = tl_->vCacheStride0;
    vCacheStride1_ = tl_->vCacheStride1;
    vCacheStride2_ = tl_->vCacheStride2;
    keyOutStride0_ = tl_->keyOutStride0;
    keyOutStride1_ = tl_->keyOutStride1;
    valueOutStride0_ = tl_->valueOutStride0;
    valueOutStride1_ = tl_->valueOutStride1;
}

// ============================================================================
// Process: 主循环 (与原 GatherPaKvCacheNd::Process 结构一致)
// ============================================================================
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

        if (batchOffset >= numTokens_) break;
        if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
            seqLen = numTokens_ - batchOffset;
        }

        uint64_t seqOff = 0;
        if constexpr (hasSeqOffset) {
            seqOff = seqOffsetGm_.GetValue(i) / cacheBlockSize_;
        }

        uint64_t blockCount = CeilDivision(static_cast<uint64_t>(seqLen),
                                           static_cast<uint64_t>(cacheBlockSize_));
        uint64_t keyOffset = batchOffset * hiddenSizeKOut_;   // 输出用 BF16 字节
        uint64_t valueOffset = batchOffset * hiddenSizeVOut_;

        for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++) {
            uint64_t blockTableOffset = blockIdx + seqOff;
            bool isFilledWithZero;
            int64_t blockId;
            if ((blockTableOffset >= static_cast<uint64_t>(blockTableWidth_)) || (blockTableOffset < 0)) {
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

            uint64_t curBlockSize = static_cast<uint64_t>(cacheBlockSize_);
            if (blockIdx == blockCount - 1) {
                curBlockSize = static_cast<uint64_t>(seqLen) - (blockCount - 1) * cacheBlockSize_;
            }

            // ===== Key: gather + dequant =====
            GatherDequantKey(outKeyGm_[keyOffset / sizeof(bfloat16_t)],
                            keyCacheGm_[blockId * cacheBlockSize_ * hiddenSizeK_],
                            keyScaleCacheGm_[blockId * cacheBlockSize_ * numHeadsK_],
                            curBlockSize, blockId, isFilledWithZero);
            keyOffset += curBlockSize * hiddenSizeKOut_;

            // ===== Value: gather + dequant =====
            GatherDequantValue(outValueGm_[valueOffset / sizeof(bfloat16_t)],
                              valueCacheGm_[blockId * cacheBlockSize_ * hiddenSizeV_],
                              valueScaleGm_[0],   // value scale 按 head 索引, 不按 block
                              curBlockSize, blockId, isFilledWithZero);
            valueOffset += curBlockSize * hiddenSizeVOut_;
        }
    }
}

// ============================================================================
// GatherDequantKey: key 路径 gather + dequant
//   key_cache INT8 [block_size, num_heads, head_dim]
//   key_scale  FLOAT [block_size, num_heads]  -- pertoken-perhead
//   output BF16 [tokens, num_heads, head_dim]
// ============================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherDequantKey(
    GlobalTensor<bfloat16_t> dstGm,
    GlobalTensor<int8_t> srcGm,
    GlobalTensor<float> scaleGm,
    uint64_t curBlockSize, uint64_t blockId,
    bool isFilledWithZero)
{
    // 逐 token 处理, 因为每个 token 有自己的 scale
    for (uint64_t t = 0; t < curBlockSize; t++) {
        // 整个 token 的 INT8 数据: numHeadsK_ * headDimK_ 个元素
        uint64_t tokenElems = numHeadsK_ * headDimK_;
        uint64_t srcOff = t * hiddenSizeK_;   // INT8 字节偏移
        uint64_t scaleOff = t * numHeadsK_;   // scale 元素偏移 (per-token, per-head)

        GatherDequantChunk(dstGm, srcGm, scaleGm,
                          0,  // dst offset 在调用侧已累加
                          srcOff, scaleOff,
                          tokenElems,
                          numHeadsK_,
                          headDimK_,
                          true,        // isKey
                          isFilledWithZero);
    }
}

// ============================================================================
// GatherDequantValue: value 路径 gather + dequant
//   value_cache INT8 [block_size, num_heads, head_dim]
//   value_scale FLOAT [num_heads, head_dim]  -- perhead-perchannel, 连续存储
//   output BF16 [tokens, num_heads, head_dim]
// ============================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherDequantValue(
    GlobalTensor<bfloat16_t> dstGm,
    GlobalTensor<int8_t> srcGm,
    GlobalTensor<float> scaleGm,
    uint64_t curBlockSize, uint64_t blockId,
    bool isFilledWithZero)
{
    // value scale 是静态的, 整个 block 内所有 token 共用同一份 scale
    // 可以按更大粒度搬运
    uint64_t tokenElems = numHeadsV_ * headDimV_;

    // 按 UB 容量分块处理
    uint64_t totalElems = curBlockSize * tokenElems;
    uint64_t chunkElems = maxUbChunkElems_;
    // 确保 chunkElems 对齐到 token 边界 (简化 scale 处理)
    if (chunkElems > tokenElems) {
        chunkElems = (chunkElems / tokenElems) * tokenElems;
    }
    if (chunkElems == 0) chunkElems = tokenElems;  // 至少处理一个 token

    uint64_t handled = 0;
    while (handled < totalElems) {
        uint64_t curElems = totalElems - handled;
        if (curElems > chunkElems) curElems = chunkElems;

        // value scale: perhead-perchannel, scale[h][d] 对应 head h 的第 d 个元素
        // 在 token 内 scale 的排布: [numHeads, headDim], 与数据布局一致
        // 所以可以用 element-wise 乘法, scale 偏移 = (handled % tokenElems) 但需按 head 对齐
        // 简化: 对齐到 token 边界后, scale offset = 0 (每个新 token 从头开始)
        uint64_t scaleBaseOff = handled % tokenElems;  // 在当前 token 内的偏移

        GatherDequantChunk(dstGm, srcGm, scaleGm,
                          handled * sizeof(bfloat16_t),
                          handled * sizeof(int8_t),
                          scaleBaseOff,
                          curElems,
                          numHeadsV_,
                          headDimV_,
                          false,       // isKey=false
                          isFilledWithZero);
        handled += curElems;
    }
}

// ============================================================================
// GatherDequantChunk: 核心计算单元
//   1. DataCopyPad: GM(INT8) → UB
//   2. Cast: INT8 → FP32 (在 fp32Buf_)
//   3. DataCopyPad: GM(scale FLOAT) → UB (仅需要的部分)
//   4. Mul: FP32 × FP32 (in-place on fp32Buf_)
//   5. Cast: FP32 → BF16 (回 dataQueue_)
//   6. DataCopyPad: UB(BF16) → GM
// ============================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherDequantChunk(
    GlobalTensor<bfloat16_t> dstGm,
    GlobalTensor<int8_t> srcGm,
    GlobalTensor<float> scaleGm,
    uint64_t dstByteOff,
    uint64_t srcByteOff,
    uint64_t scaleBaseOff,
    uint64_t elemCount,
    uint64_t numHeads,
    uint64_t headDim,
    bool isKey,
    bool isFilledWithZero)
{
    if (isFilledWithZero) {
        // 填零路径: 直接生成 BF16 零值写出
        LocalTensor<bfloat16_t> outLocal = dataQueue_.AllocTensor<bfloat16_t>();
        Duplicate<bfloat16_t>(outLocal, bfloat16_t(0),
                              static_cast<uint32_t>(elemCount));
        AscendC::TEventID eV2M = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eV2M);
        WaitFlag<HardEvent::V_MTE3>(eV2M);

        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(bfloat16_t));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad<bfloat16_t, PaddingMode::Normal>(dstGm[dstByteOff / sizeof(bfloat16_t)],
                                                      outLocal, outParams);
        AscendC::TEventID eM3M2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(eM3M2);
        WaitFlag<HardEvent::MTE3_MTE2>(eM3M2);
        dataQueue_.FreeTensor(outLocal);
        return;
    }

    // --- Step 1: GM(INT8) → UB ---
    LocalTensor<int8_t> srcLocal = dataQueue_.AllocTensor<int8_t>();
    {
        DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(int8_t));
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        DataCopyPad<int8_t, PaddingMode::Normal>(srcLocal, srcGm[srcByteOff], inParams, padExtParams_);
        dataQueue_.EnQue<int8_t>(srcLocal);
        srcLocal = dataQueue_.DeQue<int8_t>();
    }

    // --- Step 2: INT8 → FP32 ---
    LocalTensor<float> fp32Local = fp32Buf_.Get<float>();
    Cast<float, int8_t>(fp32Local, srcLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
    AscendC::TEventID eV1 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eV1);
    WaitFlag<HardEvent::V_S>(eV1);

    // --- Step 3 & 4: 加载 scale 并乘 ---
    if (isKey) {
        // Key: pertoken-perhead, 逐 head 乘各自的 scale
        // scale 布局: scaleGm[scaleBaseOff + h], 每个 scale 对应整个 head 的 headDim 个元素
        // 需要 broadcast: scale[h] × [headDim 个元素]
        for (uint64_t h = 0; h < numHeads; h++) {
            float scaleVal = scaleGm.GetValue(scaleBaseOff + h);
            uint64_t headOff = h * headDim;
            // Muls: fp32Local[headOff:headOff+headDim] *= scaleVal
            Muls(fp32Local[headOff], fp32Local[headOff], scaleVal,
                 static_cast<uint32_t>(headDim));
        }
    } else {
        // Value: perhead-perchannel, scale[h][d] 与数据布局一致
        // 直接从 GM 加载 scale 到 UB, 做 element-wise Mul
        LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
        {
            DataCopyExtParams scaleParams;
            scaleParams.blockCount = 1;
            scaleParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(float));
            scaleParams.srcStride = 0;
            scaleParams.dstStride = 0;
            DataCopyPad<float, PaddingMode::Normal>(scaleLocal, scaleGm[scaleBaseOff],
                                                     scaleParams, padExtParams_);
            scaleQueue_.EnQue<float>(scaleLocal);
            scaleLocal = scaleQueue_.DeQue<float>();
        }
        Mul(fp32Local, fp32Local, scaleLocal, static_cast<uint32_t>(elemCount));
        scaleQueue_.FreeTensor(scaleLocal);
    }

    // V→S 同步, 确保 Mul 完成
    AscendC::TEventID eV2 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eV2);
    WaitFlag<HardEvent::V_S>(eV2);

    // --- Step 5: FP32 → BF16 ---
    // 复用 srcLocal 的空间 (INT8 1B < BF16 2B < FP32 4B, 但 dataQueue_ 按 INT8 分配)
    // 需要用 reinterpret 或重新 AllocTensor
    // 实际上 BF16 2B > INT8 1B, 不能直接复用 srcLocal 的空间
    // 但 dataQueue_ 分配的大小是 maxUbChunkElems_ * 1B, 而 BF16 需要 maxUbChunkElems_ * 2B
    // 所以我们需要用 fp32Local 的后半部分或者另外处理
    //
    // 方案: 直接在 fp32Local 上原地 Cast → BF16 写回, Cast 的 dst 可以是 srcLocal 的 reinterpret
    // 但 AscendC Cast 要求 dst 和 src 不重叠且类型匹配
    //
    // 更安全的方案: 利用 dataQueue_ 的 double buffer, 当前 srcLocal 在 slot 0,
    // 重新分配 slot 1 存 BF16 (但 AllocTensor 返回的是队列头)
    //
    // 最简方案: 用 bfloat16_t reinterpret srcLocal, 因为我们只需要 maxUbChunkElems * 2B,
    // 而 dataQueue_ 总大小 = maxUbChunkElems * 1B * 2(双缓冲) = maxUbChunkElems * 2B
    // 恰好够一个 BF16 buffer!
    LocalTensor<bfloat16_t> bf16Local = srcLocal.template ReinterpretCast<bfloat16_t>();
    Cast<bfloat16_t, float>(bf16Local, fp32Local, RoundMode::CAST_ROUND,
                            static_cast<uint32_t>(elemCount));
    AscendC::TEventID eV3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eV3);
    WaitFlag<HardEvent::V_MTE3>(eV3);

    // --- Step 6: UB(BF16) → GM ---
    {
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(bfloat16_t));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad<bfloat16_t, PaddingMode::Normal>(dstGm, bf16Local, outParams);
        AscendC::TEventID eM3M2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(eM3M2);
        WaitFlag<HardEvent::MTE3_MTE2>(eM3M2);
    }

    dataQueue_.FreeTensor(srcLocal);
}

// ============================================================================
// CalcKvCoreOffset: 非 cumsum 模式下计算前缀和
// ============================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline T_INDEX
GatherPaKvCacheWithDequantNd<T_INDEX, isSeqLensCumsum, hasSeqOffset>::CalcKvCoreOffset(int64_t reduceLen)
{
    LocalTensor<T_INDEX> prefixSumLocal = prefixSumBuffer_.Get<T_INDEX>();
    uint64_t loopTimes = CeilDivision(reduceLen, static_cast<int64_t>(seqLenAccSize_));
    DataCopyPadExtParams<T_INDEX> padParams = {false, 0, 0, 0};
    DataCopyExtParams seqLensCopyParams;
    seqLensCopyParams.blockCount = 1;
    seqLensCopyParams.srcStride = 0;
    seqLensCopyParams.dstStride = 0;

    T_INDEX coreOffset = 0;
    for (uint64_t i = 0; i < loopTimes; i++) {
        uint32_t seqOffset = i * seqLenAccSize_;
        uint32_t seqLength = seqLenAccSize_;
        if (i == loopTimes - 1) {
            seqLength = reduceLen - (loopTimes - 1) * seqLenAccSize_;
        }
        seqLensCopyParams.blockLen = static_cast<uint32_t>(seqLength * sizeof(T_INDEX));
        LocalTensor<T_INDEX> seqLenLocal = scaleQueue_.AllocTensor<T_INDEX>();  // 复用 scale 队列
        DataCopyPad(seqLenLocal, seqLensGm_[seqOffset], seqLensCopyParams, padParams);
        scaleQueue_.EnQue<T_INDEX>(seqLenLocal);

        seqLenLocal = scaleQueue_.DeQue<T_INDEX>();
        uint32_t srcShape[2] = {uint32_t(1), seqLength};
        AscendC::ReduceSum<T_INDEX, Pattern::Reduce::AR, true>(prefixSumLocal, seqLenLocal, srcShape, false);
        AscendC::TEventID eventIdVecToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIdVecToS);
        WaitFlag<HardEvent::V_S>(eventIdVecToS);
        scaleQueue_.FreeTensor<T_INDEX>(seqLenLocal);
        coreOffset += prefixSumLocal.GetValue(0);
    }
    return coreOffset;
}

} // namespace GatherPaKvCacheWithDequantV35
#endif // GATHER_PA_KV_CACHE_WITH_DEQUANT_ND_H_
