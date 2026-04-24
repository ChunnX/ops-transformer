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
 * \file dense_lightning_indexer_softmax_lse_regbase_common.h
 * \brief Arch35 common definitions for dense_lightning_indexer_softmax_lse
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_REGBASE_COMMON_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_REGBASE_COMMON_H

#include "../../../common/op_kernel/buffers_policy.h"
using namespace fa_base_matmul;

enum class DenseLISoftmaxLseLayout {
    BSND = 0,
    TND = 1
};

constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
constexpr uint32_t UB_MAX_SIZE = 128 * 1024;
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;
constexpr uint32_t MODE_NUM_2 = 2;
constexpr uint32_t ALIGN_NUM_2 = 2;
constexpr uint32_t BLOCK_SINGLE_LEN = 16;

constexpr uint32_t VEC_BASESIZE = 256;
constexpr uint32_t VEC_REDUCE_BASESIZE = 64;

struct DenseLISoftmaxLseConstInfo {
    static constexpr uint32_t BUFFER_SIZE_BYTE_32 = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64 = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_128 = 128;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256 = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512 = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 64 * 1024;

    uint32_t aicIdx;
    uint32_t aivIdx;
    uint32_t subBlockIdx;

    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t dSize;
    uint32_t kSize;
    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t mBaseSize;

    int32_t sparseMode;
    int64_t preTokens;
    int64_t nextTokens;
    float scaleValue;

    int64_t totalCost;
    DenseLISoftmaxLseLayout layoutType;
};

struct DenseLISoftmaxLseRunInfo {
    uint32_t taskId;
    uint32_t bIdx;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t bN2Idx;

    int64_t accumS1Idx;
    int64_t accumS2Idx;

    uint32_t actS1Size;
    uint32_t actS2Size;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint32_t s2LoopTimes;
    uint32_t s2TailSize;
    uint32_t s2CurSize;
    uint32_t s2BaseSize;

    int64_t queryTensorOffset;
    int64_t keyTensorOffset;
    int64_t weightsTensorOffset;
    int64_t reduceSumGmOutOffset;
    int64_t reduceMaxOutGmOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd;
    uint32_t loop;

    uint32_t s1BlockInnerLoop;
    uint32_t s1Idx;
};

struct DenseLISoftmaxLseKRunInfo {
    uint32_t kTaskId;
    uint32_t kTaskIdMod2;
    uint32_t s2RealBaseSize;
    int32_t kProcessSize;
    int32_t s2Idx;
    int32_t s2SingleLoopTimes;
    int32_t s2SingleIdx;
    int32_t s2SingleCurSize;
    bool isS2end;
    bool isAlign64;
};

constexpr uint8_t SYNC_GATHER_TO_MM1_FLAG[2] = {9, 10};
constexpr uint8_t SYNC_MM1_TO_V1_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_V1_TO_MM2_FLAG = 8;
constexpr uint8_t SYNC_MM2_TO_V2_FLAG[2] = {2, 3};

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X)                                                                               \
    X(INPUT_T)                                                                                                         \
    X(OUT_T)                                                                                                           \
    X(T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X)                                                                              \
    X(LAYOUT_T, DenseLISoftmaxLseLayout, DenseLISoftmaxLseLayout::BSND)

#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type(name) = (default_val),
#define TEMPLATES_DEF                                                                                                  \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>

#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT                                                                                       \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF)                                                   \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS                                                                                                  \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME)                                                                        \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

template <typename T>
__aicore__ inline T DenseLISoftmaxLseAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T>
__aicore__ inline T DenseLISoftmaxLseCeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

template <typename T>
__aicore__ inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_REGBASE_COMMON_H