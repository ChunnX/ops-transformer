/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IFA_META_PUBLIC_DEFINE_H
#define IFA_META_PUBLIC_DEFINE_H

#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <math.h>
#include <numeric>
#include <algorithm>

namespace aicpu::kernels {

inline constexpr uint32_t MAX_CORE_NUM = 50;

enum class Layout {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5,
    BUTT
};

struct IncreFlashAttentionMetadata {
    // FA
    uint32_t usedCoreNum = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint32_t singleProcessSInnerSize = 0U;
    uint32_t singleProcessSInnerSizeTail = 0U;
    uint32_t blockSplitBn2Range = 0U;
    uint32_t tailSplitedBatchRange = 0U;
    uint32_t groupSplitSize = 0U;
    uint32_t s1SplitSize = 0U;
    uint32_t startIdxEachCore[MAX_CORE_NUM];
    // FD
    uint32_t s2 = 0U;
    uint32_t sInnerLoopSize = 0U;
    uint32_t accumOutSize = 0U;
    uint32_t logSumExpSize = 0U;
};

struct IncreFlashAttentionMetadataArgs {
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    uint32_t batchSize;
    uint32_t querySeqSize;
    uint32_t queryHeadNum;
    uint32_t keyHeadNum;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint64_t actSeqKvLenDim;
    int64_t *actSeqKvLen = nullptr;
    Layout layoutQuery = Layout::BUTT;
    int8_t* metaData = nullptr;
};

}

#endif