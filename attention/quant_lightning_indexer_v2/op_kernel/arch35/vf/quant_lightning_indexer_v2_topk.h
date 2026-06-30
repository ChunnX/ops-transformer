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
 * \file quant_lightning_indexer_v2_topk.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_TOPK_H
#define QUANT_LIGHTNING_INDEXER_V2_TOPK_H

#include "kernel_operator.h"
#include "vf_topk.h"
#include "vf_topk_16_gather.h"

namespace topk {

// ======================== TopK 算法常量 ========================
// 基于四级基数排序(Histogram-based)的 TopK 选择算法
// 将 key 按 8-bit 分段统计直方图，从高到低逐段确定 topK 的阈值 bin，
// 最后 gather 出所有 >= 阈值的元素。

static constexpr uint32_t RADIX_BINS = 256;       // 基数排序每轮的 bin 数量 (2^8 = 256，每轮处理 8 bit)
static constexpr uint32_t RADIX_BITS_PER_PASS = 8; // 每轮处理 8 bit
static constexpr uint32_t RADIX_PASSES_U32 = 4;    // uint32_t 需要 4 轮 (4 × 8 = 32 bit)
static constexpr uint32_t RADIX_PASSES_U16 = 2;    // uint16_t 需要 2 轮 (2 × 8 = 16 bit)
static constexpr uint32_t NK_VALUE_SIZE = 64;      // next_k 候选 buffer 大小，gather 粒度
static constexpr uint32_t ALIGN_256 = 256;          // UB 对齐基线，histogram 相关 buffer 按 256 元素对齐
static constexpr uint32_t DEFAULT_TOPK = 512;       // topK 默认值
static constexpr uint32_t DEFAULT_TRUNK_LEN = 16384; // 默认 trunk 长度 (16K)，每次处理的 S2 元素数

template<typename T>
class LITopk {
public:
    __aicore__ inline void operator()(LocalTensor<uint32_t>& outputIdxLocal,
                                      LocalTensor<T>& inputLocal,
                                      uint32_t s2SeqLen)
    {
    }
};

template<>
class LITopk<uint32_t> {
public:
    static __aicore__ inline uint32_t GetSharedTmpBufferSize(uint32_t topK)
    {
        // 2 * topK:              tmpIdxLocal + tmpValueLocal (filter 阶段暂存)
        // 5 * 256:               histogramsLocal + idx0~idx3 (直方图 + 4 轮基数排序索引)
        // 64:                    nkValueLocal (next_k 候选)
        // (topK + 64):           outputValueLocal (输出 value，+64 为 Duplicate 刷 -1 的额外空间)
        return 2 * topK * sizeof(uint32_t) + 5 * RADIX_BINS * sizeof(uint32_t) + NK_VALUE_SIZE * sizeof(uint32_t) +
               (topK + NK_VALUE_SIZE) * sizeof(uint32_t);
    }

    static __aicore__ inline uint32_t GetIndexBufferSize(uint32_t topK)
    {
        return (topK + NK_VALUE_SIZE) * sizeof(uint32_t); // +64: Duplicate 刷 -1 的额外空间
    }

    __aicore__ inline void Init(uint32_t topK)
    {
        this->topK = topK;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer)
    {
        tmpIdxLocal = sharedTmpBuffer[0];
        tmpValueLocal = tmpIdxLocal[topK];
        histogramsLocal = tmpValueLocal[topK];
        idx0Local = histogramsLocal[RADIX_BINS];     // 偏移 256: 直方图 bin 数
        idx1Local = idx0Local[RADIX_BINS];           // 偏移 256: 第 1 轮排序索引
        idx2Local = idx1Local[RADIX_BINS];           // 偏移 256: 第 2 轮排序索引
        idx3Local = idx2Local[RADIX_BINS];           // 偏移 256: 第 3 轮排序索引
        nkValueLocal = idx3Local[RADIX_BINS];        // 偏移 256: 第 4 轮排序索引
        outputValueLocal = nkValueLocal[NK_VALUE_SIZE]; // 偏移 64: next_k 候选数
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t>& outputIdxLocal,
                                      LocalTensor<uint32_t>& inputLocal,
                                      uint32_t s2SeqLen)
    {
        topkb32::LiTopKVF(outputIdxLocal, // filter阶段使用输出value Buf topK * 4B
                          outputValueLocal, // filter阶段使用输出 Idx Buf topK * 4B
                          inputLocal, // 输入 s2SeqLen * 4B
                          tmpIdxLocal, // filter阶段使用暂存index Buf topK * 4B
                          tmpValueLocal, // filter阶段使用暂存value Buf topK * 4B
                          histogramsLocal, // 直方图的临时Buf 256 * 4B
                          idx0Local, // 输入数据第1个8位Buf 256 * 4B
                          idx1Local, // 输入数据第2个8位Buf 256 * 4B
                          idx2Local, // 输入数据第3个8位Buf 256 * 4B
                          idx3Local, // 输入数据第4个8位Buf 256 * 4B
                          nkValueLocal, // next_k 暂存Buf 64 * 4B
                          topK,       // topk数量
                          s2SeqLen); // 输入元素总数
    }
private:
    LocalTensor<uint32_t> tmpIdxLocal;     // filter阶段使用暂存index Buf topK * 4B
    LocalTensor<uint32_t> tmpValueLocal;   // filter阶段使用暂存value Buf topK * 4B
    LocalTensor<uint32_t> histogramsLocal; // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idx0Local;       // 基数排序第1轮(最高8位)索引 Buf 256 * 4B
    LocalTensor<uint32_t> idx1Local;       // 基数排序第2轮索引 Buf 256 * 4B
    LocalTensor<uint32_t> idx2Local;       // 基数排序第3轮索引 Buf 256 * 4B
    LocalTensor<uint32_t> idx3Local;       // 基数排序第4轮(最低8位)索引 Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;    // next_k 暂存Buf 64 * 4B
    LocalTensor<uint32_t> outputValueLocal; // 输出value tensor
    uint32_t topK;
};

template<>
class LITopk<uint16_t> {
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        // 2 * Align(topK, 256):  hisIndexLocal[0] + hisIndexLocal[1] (双 buffer ping-pong)
        // 3 * 256:               histogramsLocal + idxHighLocal + idxLowLocal (2轮基数排序)
        // 64:                    nkValueLocal (next_k 候选)
        uint64_t bufferSize1 = (2 * QLIV2Common::Align(topK, (uint32_t)ALIGN_256) + 3 * RADIX_BINS + NK_VALUE_SIZE) * sizeof(uint32_t);
        // Align(topK, 256) + trunkLen: tmpIndexLocal (当前 trunk 的 topK 索引 + trunk 数据)
        uint64_t bufferSize2 = (QLIV2Common::Align(topK, (uint32_t)ALIGN_256) + trunkLen) * sizeof(uint16_t);
        // hisIndexLocal[0] 复用 indicesOutLocal，从 shared buffer 中扣除
        uint64_t reUseBufferSize = QLIV2Common::Align(topK, (uint32_t)ALIGN_256) * sizeof(uint32_t);
        return bufferSize1 + bufferSize2 - reUseBufferSize;
    }

    __aicore__ inline void Init(uint32_t topK, uint32_t trunkLen)
    {
        this->topK = topK;
        this->trunkLen =  trunkLen;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer, LocalTensor<uint32_t>& indicesOutLocal)
    {
        LocalTensor<uint32_t> hisIndexLocal1 = indicesOutLocal; // hisIndexLocal[0] 复用输出 buffer
        // Align(topK, 256): topK 向上对齐到 256 的倍数，作为各 buffer 的分配基线
        LocalTensor<uint32_t> hisIndexLocal2 = sharedTmpBuffer[0];
        hisIndexLocal[0] = hisIndexLocal1;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = hisIndexLocal2[QLIV2Common::Align(topK, (uint32_t)ALIGN_256)]; // 偏移 Align(topK,256): 跳过 hisIndexLocal[1]
        idxHighLocal = histogramsLocal[RADIX_BINS]; // 偏移 256: 直方图 bin 数，高 8 位索引
        idxLowLocal = idxHighLocal[RADIX_BINS];     // 偏移 256: 低 8 位索引
        nkValueLocal = idxLowLocal[RADIX_BINS];     // 偏移 256: next_k 候选
        LocalTensor<uint32_t> tmpIndexLocalTmp = nkValueLocal[NK_VALUE_SIZE]; // 偏移 64: next_k 候选数
        tmpIndexLocal = tmpIndexLocalTmp.template ReinterpretCast<uint16_t>();
    }

    __aicore__ inline void TopK(LocalTensor<uint16_t>& mrgValueLocal, LocalTensor<uint32_t>& indicesOutLocal,
                                LocalTensor<uint16_t>& hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx,
                                uint32_t s2LoopNum, bool isNeedLD, bool returnValueFlag,
                                uint32_t outputIdxOffset)
    {
        // true: 开启返回hisValueLocal
        if (s2LoopNum == 1) {
            if (isNeedLD || returnValueFlag) {
                topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                              idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            } else {
                topkb16gather::LiTopKVF<false>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                               idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            }
            PipeBarrier<PIPE_V>();
            Cast(indicesOutLocal, tmpIndexLocal, RoundMode::CAST_NONE, topK);
            if (outputIdxOffset != 0) {
                topkb16gather::IndicesAddOffset(indicesOutLocal, outputIdxOffset, topK);
            }
            return;
        }
        if (loopIdx == 0 && !isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                          idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            Cast(hisIndexLocal[(loopIdx + 1) % 2], tmpIndexLocal, RoundMode::CAST_NONE, topK); // %2: 双 buffer ping-pong
        } else if (loopIdx != 0 && !isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                          idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            // loopIdx * trunkLen - Align(topK, 256): 当前 trunk 在 mrgValueLocal 中的起始偏移
            topkb16gather::LiTopKGatherVF(hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, mrgValueLocal,
                                          tmpIndexLocal, hisIndexLocal[loopIdx % 2], topK,
                                          loopIdx * trunkLen - QLIV2Common::Align(topK, (uint32_t)ALIGN_256), s2SeqLen);
            if (loopIdx == s2LoopNum - 1) {
                PipeBarrier<PIPE_V>();
                AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2], QLIV2Common::Align(topK,
                    (uint32_t)ALIGN_256));
            }
        }

        if (loopIdx == 0 && isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                          idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            Cast(hisIndexLocal[(loopIdx + 1) % 2], tmpIndexLocal, RoundMode::CAST_NONE, topK);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2],
                              QLIV2Common::Align(topK, (uint32_t)ALIGN_256));
        } else if (loopIdx != 0 && isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                          idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            topkb16gather::LiTopKGatherVF(hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, mrgValueLocal,
                                          tmpIndexLocal, hisIndexLocal[loopIdx % 2],
                                          topK, loopIdx * trunkLen - QLIV2Common::Align(topK, (uint32_t)ALIGN_256), s2SeqLen);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2],
                QLIV2Common::Align(topK, (uint32_t)ALIGN_256));
        }
        if (outputIdxOffset != 0) {
            topkb16gather::IndicesAddOffset(indicesOutLocal, outputIdxOffset, topK);
        }
    }
    __aicore__ inline void LdTopK(LocalTensor<uint16_t>& mrgValueLocal, LocalTensor<uint32_t> indexLocal,
                                  LocalTensor<uint32_t>& indicesOutLocal,
                                  LocalTensor<uint16_t>& hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx,
                                  uint32_t s2LoopNum)
    {
        topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                      idxLowLocal, nkValueLocal, topK, s2SeqLen);
        PipeBarrier<PIPE_V>();
        topkb16gather::LiTopKLDGatherVF(indicesOutLocal, tmpIndexLocal, indexLocal, topK);
    }

private:
    LocalTensor<uint32_t> hisIndexLocal[2];      // 双 buffer ping-pong: 每个 trunk 选出的 topK 索引
    LocalTensor<uint32_t> histogramsLocal;       // 直方图的临时Buf 256 * 4B (256 bins)
    LocalTensor<uint32_t> idxHighLocal;          // 基数排序高 8 位索引 Buf 256 * 4B
    LocalTensor<uint32_t> idxLowLocal;           // 基数排序低 8 位索引 Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;          // next_k 暂存Buf 64 * 4B
    LocalTensor<uint16_t> tmpIndexLocal;         // 每 trunkLen + topK 的临时 index
    uint32_t topK = DEFAULT_TOPK;
    uint32_t trunkLen = DEFAULT_TRUNK_LEN;
};
}
#endif  // QUANT_LIGHTNING_INDEXER_V2_TOPK_H