/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_LIGHTNING_INDEXER_COMMON_H
#define GENERIC_LIGHTNING_INDEXER_COMMON_H

#include <type_traits>
#include "generic_lightning_indexer_tiling_data.h"
#include "gli_policy.h"

using namespace AscendC;
namespace GliCommon {

// mxFP8 scale 布局: 每 32 个 D 元素共享一个 E8M0 scale, 2 个 E8M0 打包成 1 个 bf16 寻址单位
constexpr uint32_t GLI_MX_SCALE_GROUP_SIZE = 32;
constexpr uint32_t GLI_FP8_TWO = 2;

enum class GliLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

template <typename Q_T, typename K_T, typename OUT_T, GliLayout LAYOUT_T = GliLayout::BSND,
          GliLayout K_LAYOUT_T = GliLayout::BSND, bool DT_W_FLAG = false, typename SCALE_T = void,
          typename WEIGHTS_T = void, uint32_t HEAD_REDUCE = 0, uint32_t EXPAND_MODE = 0>
struct GliType {
    static constexpr bool weightsTypeFlag = DT_W_FLAG;
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    using scaleType = SCALE_T;
    // 量化场景的 weights 实际 dtype(半精度, 由 kernel entry 按 ORIG_DTYPE_WEIGHTS
    // 分发传入 half/bfloat16_t); 非量化(WEIGHTS_T=void)沿用 GliTypeTraits 推导
    using weightsType = WEIGHTS_T;
    static constexpr bool pageAttention = (K_LAYOUT_T == GliLayout::PA_BSND);
    static constexpr GliLayout layout = LAYOUT_T;
    static constexpr GliLayout keyLayout = K_LAYOUT_T;
    static constexpr uint32_t headReduce = HEAD_REDUCE; // 0 weighted, 1 sum, 2 none (MSA keep heads)
    static constexpr uint32_t expandMode = EXPAND_MODE; // 0 off, 1 expand, 2 expand+append
    static constexpr bool useWeights = (HEAD_REDUCE == 0);
    static constexpr bool keepHeads = (HEAD_REDUCE == 2);
    static constexpr bool applyRelu = (HEAD_REDUCE != 2);
    static constexpr bool doTopK = (HEAD_REDUCE != 2);
    static constexpr bool doExpand = (EXPAND_MODE != 0);
    // 量化模式判定: FP8 输入 + E8M0 scale => mxFP8 硬件路径;
    // FP8 输入 + float scale => per-token-head vector 融合路径
    static constexpr bool isMxFp8 = std::is_same<Q_T, fp8_e4m3fn_t>::value && std::is_same<K_T, fp8_e4m3fn_t>::value &&
                                    std::is_same<SCALE_T, fp8_e8m0_t>::value;
    static constexpr bool isFp16Scale = std::is_same<SCALE_T, half>::value;
    // 1 字节 Q 或 K + 非 MX scale：vector 反量化。覆盖 FP8/HiFP8/E5M2+float、QLI INT8、MSA FP16×INT8。
    static constexpr bool isFp8PerToken = (sizeof(Q_T) == 1 || sizeof(K_T) == 1) &&
                                          !std::is_same<SCALE_T, void>::value &&
                                          !std::is_same<SCALE_T, fp8_e8m0_t>::value;
    static constexpr bool isQuant = isMxFp8 || isFp8PerToken;
    // 950 不支持 half×int8 Mmad：AIV 先 Cast K 到 fp16 workspace，再走 half×half cube。
    static constexpr bool isMixedInt8Key = (sizeof(Q_T) == 2) && (sizeof(K_T) == 1);
    // 950 Mmad(float, int8, int8) 非法，同构 INT8 走 int32 累加再 vector Cast。
    static constexpr bool isInt8Mmad = std::is_same<Q_T, int8_t>::value && std::is_same<K_T, int8_t>::value;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start = 0; // 本核 s2 循环起始块号(k_descale 乒乓刷新用)

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1; // L_orig (压缩前 token 总数)
    int32_t poolTailK = 0;      // 当前 batch 的尾部 token 数
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    // 量化场景 scale GM 偏移(仅 quantMode>=0 有效)
    uint64_t tensorQScaleOffset = 0;
    uint64_t tensorKeyScaleOffset = 0;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
};

struct ConstInfo {
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t GLI_SYNC_MODE4 = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;
    static constexpr uint32_t CAST_READY_EVENT = 4;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr int INVALID_IDX = -1;
    uint32_t INVALID_VAL = 0xFF800000;

    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    uint32_t mBaseSize = 1ULL;
    uint32_t mBaseSizeAlign = 1ULL;
    uint32_t mBaseSizeMax = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;
    uint32_t trunkLen = 8192;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum = 1ULL;
    uint64_t headDim = 128ULL;
    float qkScale = 1.0f; // score_scale, default 1; PKI maps 1/sqrt(D)
    uint64_t sparseCount = 0ULL;
    uint64_t topk = 0ULL;
    uint64_t poolSize = 1ULL; // cmp_ratio
    uint32_t expandMode = 0;
    uint32_t hasWeightedSum = 1;
    uint32_t actType = 0;
    uint32_t selectMode = 0;
    uint32_t keyReduceMode = 0;
    uint32_t initBlocks = 0;
    uint32_t localBlocks = 0;
    uint64_t kSeqSize = 0ULL;
    uint64_t qSeqSize = 1ULL;

    uint32_t kCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t keyStride0 = 0U;
    uint32_t keyDequantScaleStride0 = 0U;
    uint32_t hasOutputIdxOffset = 0U;

    GliLayout outputLayout = GliLayout::BSND;
    bool attenMaskFlag = false;
    uint32_t maskMode = 0;
    int32_t quantMode = -1;
    bool returnValue = false;
    bool isSparseCountOver2K = false;
    bool isLDOpen = false;
    bool splitMFlag = false;

    uint32_t actualLenQDims = 0U;
    uint32_t actualLenDims = 0U;
    bool isAccumSeqS1 = false;
    bool isAccumSeqS2 = false;
    uint32_t seqQCuFormat = 0U; // 1: LIv2/MSA cu_seqlens [B+1], 0: LI exclusive prefix [B]
    uint32_t seqKCuFormat = 0U;
    uint32_t seqDtypeFlags = 0U;
    uint32_t msaScoreNT = 0U; // TND MSA sparse_values [N1,T1,nOut]

    // workspace offsets
    uint64_t wsOffScore = 0;
    uint64_t wsOffLdScore = 0;
    uint64_t wsOffLdIdx = 0;
    uint64_t wsOffKCast = 0;
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;
    bool isCoreEnable = false;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

__aicore__ inline uint32_t IndicesOutLen(uint32_t sparseCount, uint32_t cmpRatio, uint32_t expandMode)
{
    if (expandMode == 0) {
        return sparseCount;
    }
    uint32_t expanded = sparseCount * cmpRatio;
    return (expandMode >= 2) ? (expanded + cmpRatio - 1) : expanded;
}

constexpr uint32_t GLI_SEQ_Q_I64 = 1U;
constexpr uint32_t GLI_SEQ_K_I64 = 2U;
constexpr uint32_t GLI_SEQUSED_Q_I64 = 4U;
constexpr uint32_t GLI_SEQUSED_K_I64 = 8U;
constexpr uint32_t GLI_RESIDUAL_I64 = 16U;
constexpr uint32_t GLI_START_LOC_I64 = 32U;

__aicore__ inline int64_t LoadSeqValue(const GlobalTensor<int32_t> &i32, const GlobalTensor<int64_t> &i64, uint32_t idx,
                                       bool useI64)
{
    return useI64 ? i64.GetValue(idx) : static_cast<int64_t>(i32.GetValue(idx));
}

__aicore__ inline uint32_t DecodeActualSeq(uint32_t bIdx, uint32_t actualLenDims, bool cuFormat, bool isAccumSeq,
                                           const GlobalTensor<int32_t> &i32, const GlobalTensor<int64_t> &i64,
                                           bool useI64, uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    }
    if (cuFormat) {
        return static_cast<uint32_t>(LoadSeqValue(i32, i64, bIdx + 1, useI64) - LoadSeqValue(i32, i64, bIdx, useI64));
    }
    if (isAccumSeq && bIdx > 0) {
        return static_cast<uint32_t>(LoadSeqValue(i32, i64, bIdx, useI64) - LoadSeqValue(i32, i64, bIdx - 1, useI64));
    }
    return static_cast<uint32_t>(LoadSeqValue(i32, i64, bIdx, useI64));
}

__aicore__ inline uint64_t DecodeSeqPrefix(uint32_t bIdx, bool isTnd, bool cuFormat, const GlobalTensor<int32_t> &i32,
                                           const GlobalTensor<int64_t> &i64, bool useI64, uint64_t seqSize)
{
    if (!isTnd) {
        return (bIdx == 0) ? 0 : static_cast<uint64_t>(bIdx) * seqSize;
    }
    if (cuFormat) {
        return static_cast<uint64_t>(LoadSeqValue(i32, i64, bIdx, useI64));
    }
    return (bIdx == 0) ? 0 : static_cast<uint64_t>(LoadSeqValue(i32, i64, bIdx - 1, useI64));
}
} // namespace GliCommon

#define UB_BLOCK 32
#define UB_BANK_GROUPS 8
#define UB_BANKS 2
#define UB_BANK_DEPTH 512
#define UB_BANK_GROUP_STRIDE UB_BLOCK
#define UB_BANK_STRIDE (UB_BANK_GROUPS * UB_BLOCK)
#define UB_BANK_DEPTH_STRIDE (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)

#endif // GENERIC_LIGHTNING_INDEXER_COMMON_H
