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
 * \file util_regbase_QBSA.h
 * \brief QBSA sparse runtime extensions to the common RegBase parameters.
 */
#ifndef UTIL_REGBASE_QBSA_H
#define UTIL_REGBASE_QBSA_H

#include "util_regbase.h"

namespace regbaseutil {
struct QbsaRunParamStr : RunParamStr<true> {
    int64_t n1oIdx;
    int32_t actSparseLen;
    int64_t s1LoopStart;
    int64_t s1LoopEnd;
    int64_t qBScalarOffset;
};

struct QbsaRunInfo : RunInfo<true> {
    int64_t qBOffset;
    int64_t qBScalarOffset;
    int32_t s2SparseBlk1RealSize;
    int32_t s2SparseBlk1RealAlignedSize;
    int32_t s2SparseBlk2RealSize;
    int32_t s2SparseBlk2RealAlignedSize;
    int64_t sparseBase = 0;
    int64_t sparseS2TokenOffset = 0;
    int32_t actSparseLen = 0;
    int64_t sparseBlkIdx1;
    int64_t sparseBlkIdx2;
    int64_t phyBlkNumIdx1;
    int64_t phyBlkNumIdx2;
    bool sparseBlk1PartialMask;
    bool sparseBlk2PartialMask;
};

struct QbsaConstInfo : ConstInfo<true, false> {
    int64_t n1Size;
    uint32_t maxQb;
    uint32_t maxKb;
    uint32_t seqUsedQlenSize;
    uint32_t seqUsedKvlenSize;
    uint32_t paBlockStride;
    uint32_t combineDim;
    uint32_t qSparseBlockSize;
    uint32_t kvSparseBlockSize;
    int64_t n2GS1o;
    int64_t gS1o;
    uint32_t maxBlockNumPerBatch;
};

// Keep the original QBSA AIC/AIV transfer layout. The common CVSharedParams
// contains additional dense-attention state and cannot replace this buffer layout.
#define QBSA_CV_SHARED_PARAMS \
    uint32_t bSize; \
    int64_t t1Size; \
    uint32_t n2Size; \
    uint32_t gSize; \
    uint32_t dSize : 16; \
    uint32_t dSizeV : 16; \
    int64_t preTokens; \
    int64_t nextTokens; \
    uint32_t attenMaskS2Size; \
    uint32_t s1OuterSize; \
    uint32_t bandIndex; \
    uint32_t compressMode : 4; \
    uint32_t layoutType : 4; \
    uint32_t dSizeRope : 11; \
    uint32_t splitCoreMode : 1; \
    uint32_t coreNum

template <bool isPa = false>
struct QbsaCVSharedParams;

template <>
struct QbsaCVSharedParams<false> {
    QBSA_CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;
    uint32_t seqUsedQlenSize;
    uint32_t seqUsedKvlenSize;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
};

template <>
struct QbsaCVSharedParams<true> {
    QBSA_CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;
    uint32_t seqUsedQlenSize;
    uint32_t seqUsedKvlenSize;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int32_t qSparseBlockSize;
    int32_t kvSparseBlockSize;
    int32_t blockTableDim2;
    int32_t paBlockNumSum;
    uint32_t paLayoutType;
    uint32_t paBlockStride;
};

#undef QBSA_CV_SHARED_PARAMS

static_assert(sizeof(QbsaCVSharedParams<false>) <= 128, "QBSA shared parameters exceed one cache line");
static_assert(sizeof(QbsaCVSharedParams<true>) <= 128, "QBSA PA shared parameters exceed one cache line");
} // namespace regbaseutil

#endif // UTIL_REGBASE_QBSA_H
