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
 * \file stem_indexer_common.h
 * \brief
 */
#ifndef stem_indexer_COMMON_H
#define stem_indexer_COMMON_H
using namespace AscendC;
namespace SICommon {

// Keep layout values aligned with tiling.
enum class SI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BNSD = 2,
    BNSD = 3
};

// Sync mode between Cube and Vector.
constexpr uint32_t SI_SYNC_MODE4 = 4U;
constexpr uint32_t AIV0_AIV1_OFFSET = 16U;
constexpr uint32_t CROSS_VC_EVENT = 0U;
constexpr uint32_t CROSS_CV_EVENT = 2U;
// Buffer size in bytes.
constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32U;
constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64U;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256U;
constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512U;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024U;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048U;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096U;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192U;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384U;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768U;
// Invalid index.
constexpr int INVALID_IDX = -1;

template <typename Q_T, typename K_T, typename OUT_T, const bool CAUSAL = false, typename... Args>
struct SIType {
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;

    static constexpr bool causalFlag = CAUSAL;
    static constexpr bool pageAttention = false;
    static constexpr SI_LAYOUT layout = SI_LAYOUT::BNSD;
    static constexpr SI_LAYOUT keyLayout = SI_LAYOUT::BNSD;
};
// TempLoopInfo keeps B/N/S1 loop state before RunInfo is filled and avoids repeated calculation.
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // End index of the gS1 loop.
    uint32_t s2LoopEnd = 0U;    // End index of the S2 loop.
    uint32_t actS1Size = 1U;    // Actual S1 size for the current batch.
    uint32_t actS2Size = 0U;
    uint32_t promptLen = 0U;
    uint32_t s2ValidSize = 0U;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // Whether to clear output when actual S1 is shorter than shape S1.
    uint32_t actMBaseSize = 0U;            // Actual M size in the gS1 dimension.
    uint32_t mBasicSizeTail = 0U;          // Tail basic block size in the gS1 dimension.
    uint32_t s2BasicSizeTail = 0U;         // Tail basic block size in the S2 dimension.
    bool isNeedLD = false;                 // Whether this basic block needs LD.
};
struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start;
    uint32_t s2LoopEnd;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorWeightsOffset;
    uint64_t tensorVBiasOffset;
    uint64_t indiceOutOffset;
    uint64_t indiceLenOffset;
    uint32_t promptLen;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isNeedLD = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo {
    // Basic block size.
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t headDim = 0ULL;
    uint64_t maxQb = 0ULL;
    uint64_t maxKb = 0ULL;
    uint64_t sparseCount = 0ULL;       // TopK selection size.
    uint64_t kSeqSize = 0ULL;          // Maximum KV S length.
    uint64_t qSeqSize = 1ULL;          // Maximum Q S length.
    uint32_t usedCoreNum = 0U;
    uint32_t causal = 1U;
    uint32_t stemBlockSize = 128U;
    uint32_t stemStride = 16U;
    uint32_t initialBlocks = 4U;
    uint32_t windowSize = 4U;
    float rSquare = 1.0f / 64.0f;  // 1/(stem_block_size/stem_stride)^2 = 1/64
    float alpha = 1.0f;  // TPD decay factor, configurable at runtime.
    float kBlockNumRateMedium = 0.2f;
    uint32_t kBlockNumBiasMedium = 30U;
    float kBlockNumRateLarge = 0.1f;
    uint32_t kBlockNumBiasLarge = 30U;
    uint32_t kCacheBlockSize = 0;      // PA block size.
    uint32_t maxBlockNumPerBatch = 0;  // Maximum block number per batch in PA.
    SI_LAYOUT outputLayout;            // Output layout.
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1;
    bool batchSupperFlag = false;      // Whether actual_seq length is B + 1.
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;

    uint32_t actualLenQDims = 0U;  // Dimension of query actualSeqLength.
    uint32_t actualLenDims = 0U;   // Dimension of KV actualSeqLength.
    bool isAccumSeqS1 = false;     // Whether S1 uses cumulative mode.
    bool isAccumSeqS2 = false;     // Whether S2 uses cumulative mode.
    bool isLDOpen = false;
};

struct LdSplitCoreInfo {
    bool isLdCoreEnable = false;     // Whether current core participates in LD reduction.
    uint32_t saveWorkSpaceIdx = 0U;  // Address for LD parameters.
    uint32_t bn2Idx = 0U;            // Reduction task index.
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U;      // Reduction task index on current AIV.
    uint32_t workspaceNum = 0U;      // Number of S2 splits for reduction on current AIV.
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0U;  // Initial offset for final TopK output indices.
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;  // Start position of S2.
    uint32_t s2End = 0U;    // Upper index of the S2 loop.
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;  // Whether current task needs decode reduction.
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
}  // namespace SICommon

// Bank conflict optimization.
// 256KB bank layout
// shape  (             bank_depth  (            banks  bank_groups  block))  (512  (  2   8  32))
// stride (banks*bank_groups*block  (bank_groups*block        block      1))  (512  (256  32   1))
#define UB_BLOCK              32   // 32B
#define UB_BANK_GROUPS        8
#define UB_BANKS              2
#define UB_BANK_DEPTH         512

#define UB_BANK_GROUP_STRIDE  UB_BLOCK                                   // 32B
#define UB_BANK_STRIDE        (UB_BANK_GROUPS * UB_BLOCK)               // 256B
#define UB_BANK_DEPTH_STRIDE  (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)    // 512B

#endif  // stem_indexer_COMMON_H
