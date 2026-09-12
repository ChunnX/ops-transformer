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
 * \file lightning_indexer_v2_base_arch35.h
 * \brief Common arch35 data definitions shared by LightningIndexerV2 and QuantLightningIndexerV2.
 */
#ifndef LIGHTNING_INDEXER_V2_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_BASE_ARCH35_H

#include "kernel_operator.h"

namespace LIV2Common {

template <typename Q, typename K, typename QK, typename Score, typename Out, bool PageAttention,
          typename Layout, Layout QLayout, Layout KLayout>
struct LightningIndexerV2DataType {
    using queryType = Q;
    using keyType = K;
    using queryKeyType = QK;
    using scoreType = Score;
    using outputType = Out;
    using layoutType = Layout;

    static constexpr bool pageAttention = PageAttention;
    static constexpr Layout layout = QLayout;
    static constexpr Layout keyLayout = KLayout;
};

struct LightningIndexerV2RunInfoBase {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start = 0;
    uint32_t s2LoopEnd = 0;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    uint32_t curCuSeqlensQ = 0;
    uint32_t curCuSeqlensK = 0;
    uint32_t curSequsedQ = 0;
    uint32_t curSequsedK = 0;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;
    uint64_t outputIdxCoreOffset;

    bool isOutputIdxOffsetValid = false;
    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
    bool isNeedLD = false;
    bool needTndPadding = false;
    uint32_t saveWorkSpaceIdx = 0;
};

template <typename Layout>
struct LightningIndexerV2ConstInfoBase {
    static constexpr uint32_t SYNC_MODE4 = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;

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

    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    uint32_t mBaseSize = 1U;
    uint32_t s1BaseSize = 1U;
    uint32_t s2BaseSize = 1U;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t kSeqSize = 0ULL;
    uint64_t qSeqSize = 1ULL;
    uint32_t kCacheBlockSize = 0U;
    uint32_t maxBlockNumPerBatch = 0U;
    Layout outputLayout;
    bool attenMaskFlag = false;
    uint32_t keyStride0 = 0U;

    uint32_t actualLenQDims = 0U;
    uint32_t actualLenDims = 0U;
    bool isAccumSeqS1 = false;
    bool isAccumSeqS2 = false;
    bool isLDOpen = false;
    bool returnValue = false;
};

struct LightningIndexerV2SplitCoreInfo {
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;
    bool isCoreEnable = false;
};

struct LightningIndexerV2LdSplitCoreInfo {
    bool isLdCoreEnable = false;
    uint32_t saveWorkSpaceIdx = 0U;
    uint32_t bn2Idx = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U;
    uint32_t workspaceNum = 0U;
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0U;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return rnd == 0 ? 0 : (num + rnd - 1) / rnd * rnd;
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return a > b ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return rnd == 0 ? 0 : (num + rnd - 1) / rnd;
}

} // namespace LIV2Common

// UB bank layout: shape (512, (2, 8, 32)), stride (512, (256, 32, 1)).
#define UB_BLOCK 32
#define UB_BANK_GROUPS 8
#define UB_BANKS 2
#define UB_BANK_DEPTH 512
#define UB_BANK_GROUP_STRIDE UB_BLOCK
#define UB_BANK_STRIDE (UB_BANK_GROUPS * UB_BLOCK)
#define UB_BANK_DEPTH_STRIDE (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)

#endif // LIGHTNING_INDEXER_V2_BASE_ARCH35_H
