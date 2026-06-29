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

// 与tiling的layout保持一致
enum class SI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BNSD = 2, 
    BNSD = 3
};

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
// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;    // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0U;
    uint32_t promptLen = 0U;
    uint32_t s2ValidSize = 0U;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
    bool isNeedLD = false;     // 该基本块是否需要LD
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
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t QLI_SYNC_MODE4 = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;
    // BUFFER的字节数
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
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
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
    uint64_t sparseCount = 0ULL;       // topK选取大小
    uint64_t kSeqSize = 0ULL;          // kv最大S长度
    uint64_t qSeqSize = 1ULL;          // q最大S长度
    uint32_t usedCoreNum = 0U;
    uint32_t causal = 1U;
    uint32_t stemBlockSize = 128U;
    uint32_t stemStride = 16U;
    uint32_t initialBlocks = 4U;
    uint32_t windowSize = 4U;
    float rSquare = 1.0f / 64.0f;
    float alpha = 1.0f;
    float kBlockNumRateMedium = 0.2f;
    uint32_t kBlockNumBiasMedium = 30U;
    float kBlockNumRateLarge = 0.1f;
    uint32_t kBlockNumBiasLarge = 30U;
    uint32_t kCacheBlockSize = 0;      // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0;  // PA场景的最大单batch block number
    SI_LAYOUT outputLayout;            // 输出的格式
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1;
    bool batchSupperFlag = false;      // actual_seq长度是否为B+1
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;

    uint32_t actualLenQDims = 0U;  // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;   // KV 的actualSeqLength 的维度
    bool isAccumSeqS1 = false;     // 是否累加模式
    bool isAccumSeqS2 = false;     // 是否累加模式
    bool isLDOpen = false;
};

struct LdSplitCoreInfo {
    bool isLdCoreEnable = false;     // 当前核是否参与规约任务
    uint32_t saveWorkSpaceIdx = 0U;  // 存放LD参数的地址
    uint32_t bn2Idx = 0U;            // 归约任务
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U;      // 当前AIV核上规约任务的索引
    uint32_t workspaceNum = 0U;      // 当前AIV核上规约任务的S2切分数量
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0U;  // 最终输出索引搬出Topk的初始偏移地址
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;  // S2的起始位置
    uint32_t s2End = 0U;    // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;  // 当前核是否需要进行Decode归约任务
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

// bank冲突优化
// david 256KB bank layout
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
