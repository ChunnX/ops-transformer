/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_TILING_FIELDS_H
#define FLASH_MLA_TILING_FIELDS_H

#include <cstddef>
#include <cstdint>

namespace FlashMla {
// Wire mirrors of the complete baseline tilings. Every nested tiling is 8-byte aligned.
struct alignas(8) MlaWireFusedInferAttentionBaseParams {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t headDim;
    uint32_t headDimRope;
    uint32_t actualSeqS1Dims;
    uint32_t actualSeqS2Dims;
    uint32_t accumQSeqFlag;
    uint32_t accumKVSeqFlag;
    float scaleValue;
    uint32_t usedCoreNum;
    uint32_t outputLayout;
    uint32_t batchContinuous;
    uint32_t softmaxLseFlag;
    uint32_t needInit;
    uint32_t slidingFlag;
    uint32_t l2CacheOffFlag;
    uint32_t isLegacyIfa;
    uint8_t isEmptyBatchOverHalf;
};
static_assert(sizeof(MlaWireFusedInferAttentionBaseParams) == 88, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, bSize) == 0,
              "baseline FusedInferAttentionBaseParams.bSize offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, n2Size) == 4,
              "baseline FusedInferAttentionBaseParams.n2Size offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, gSize) == 8,
              "baseline FusedInferAttentionBaseParams.gSize offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, s1Size) == 12,
              "baseline FusedInferAttentionBaseParams.s1Size offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, s2Size) == 16,
              "baseline FusedInferAttentionBaseParams.s2Size offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, headDim) == 20,
              "baseline FusedInferAttentionBaseParams.headDim offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, headDimRope) == 24,
              "baseline FusedInferAttentionBaseParams.headDimRope offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, actualSeqS1Dims) == 28,
              "baseline FusedInferAttentionBaseParams.actualSeqS1Dims offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, actualSeqS2Dims) == 32,
              "baseline FusedInferAttentionBaseParams.actualSeqS2Dims offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, accumQSeqFlag) == 36,
              "baseline FusedInferAttentionBaseParams.accumQSeqFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, accumKVSeqFlag) == 40,
              "baseline FusedInferAttentionBaseParams.accumKVSeqFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, scaleValue) == 44,
              "baseline FusedInferAttentionBaseParams.scaleValue offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, usedCoreNum) == 48,
              "baseline FusedInferAttentionBaseParams.usedCoreNum offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, outputLayout) == 52,
              "baseline FusedInferAttentionBaseParams.outputLayout offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, batchContinuous) == 56,
              "baseline FusedInferAttentionBaseParams.batchContinuous offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, softmaxLseFlag) == 60,
              "baseline FusedInferAttentionBaseParams.softmaxLseFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, needInit) == 64,
              "baseline FusedInferAttentionBaseParams.needInit offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, slidingFlag) == 68,
              "baseline FusedInferAttentionBaseParams.slidingFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, l2CacheOffFlag) == 72,
              "baseline FusedInferAttentionBaseParams.l2CacheOffFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, isLegacyIfa) == 76,
              "baseline FusedInferAttentionBaseParams.isLegacyIfa offset");
static_assert(offsetof(MlaWireFusedInferAttentionBaseParams, isEmptyBatchOverHalf) == 80,
              "baseline FusedInferAttentionBaseParams.isEmptyBatchOverHalf offset");

struct alignas(8) MlaWireFusedInferAttentionPageAttentionParams {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};
static_assert(sizeof(MlaWireFusedInferAttentionPageAttentionParams) == 8, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionPageAttentionParams, blockSize) == 0,
              "baseline FusedInferAttentionPageAttentionParams.blockSize offset");
static_assert(offsetof(MlaWireFusedInferAttentionPageAttentionParams, maxBlockNumPerBatch) == 4,
              "baseline FusedInferAttentionPageAttentionParams.maxBlockNumPerBatch offset");

struct alignas(8) MlaWireFusedInferAttentionMaskParams {
    uint32_t attenMaskFlag;
    uint32_t attenMaskBatchStride;
    uint32_t attenMaskStride;
    int32_t preToken;
    int32_t nextToken;
    uint32_t isRowInvalid;
    uint32_t isExistRowInvalid;
    uint32_t sparseMode;
};
static_assert(sizeof(MlaWireFusedInferAttentionMaskParams) == 32, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, attenMaskFlag) == 0,
              "baseline FusedInferAttentionMaskParams.attenMaskFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, attenMaskBatchStride) == 4,
              "baseline FusedInferAttentionMaskParams.attenMaskBatchStride offset");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, attenMaskStride) == 8,
              "baseline FusedInferAttentionMaskParams.attenMaskStride offset");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, preToken) == 12,
              "baseline FusedInferAttentionMaskParams.preToken offset");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, nextToken) == 16,
              "baseline FusedInferAttentionMaskParams.nextToken offset");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, isRowInvalid) == 20,
              "baseline FusedInferAttentionMaskParams.isRowInvalid offset");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, isExistRowInvalid) == 24,
              "baseline FusedInferAttentionMaskParams.isExistRowInvalid offset");
static_assert(offsetof(MlaWireFusedInferAttentionMaskParams, sparseMode) == 28,
              "baseline FusedInferAttentionMaskParams.sparseMode offset");

struct alignas(8) MlaWireFusedInferAttentionWorkspaceParams {
    uint32_t mm1ResSize;
    uint32_t mm2ResSize;
    uint32_t fdAccumOutSize;
    uint32_t fdLogSumExpSize;
};
static_assert(sizeof(MlaWireFusedInferAttentionWorkspaceParams) == 16, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionWorkspaceParams, mm1ResSize) == 0,
              "baseline FusedInferAttentionWorkspaceParams.mm1ResSize offset");
static_assert(offsetof(MlaWireFusedInferAttentionWorkspaceParams, mm2ResSize) == 4,
              "baseline FusedInferAttentionWorkspaceParams.mm2ResSize offset");
static_assert(offsetof(MlaWireFusedInferAttentionWorkspaceParams, fdAccumOutSize) == 8,
              "baseline FusedInferAttentionWorkspaceParams.fdAccumOutSize offset");
static_assert(offsetof(MlaWireFusedInferAttentionWorkspaceParams, fdLogSumExpSize) == 12,
              "baseline FusedInferAttentionWorkspaceParams.fdLogSumExpSize offset");

struct alignas(8) MlaWireFusedInferAttentionInnerSplitParams {
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
};
static_assert(sizeof(MlaWireFusedInferAttentionInnerSplitParams) == 8, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionInnerSplitParams, mBaseSize) == 0,
              "baseline FusedInferAttentionInnerSplitParams.mBaseSize offset");
static_assert(offsetof(MlaWireFusedInferAttentionInnerSplitParams, s2BaseSize) == 4,
              "baseline FusedInferAttentionInnerSplitParams.s2BaseSize offset");

struct alignas(8) MlaWireFusedInferAttentionOuterSplitParams {
    uint32_t bN2End[26];
    uint32_t gS1End[26];
    uint32_t s2End[26];
};
static_assert(sizeof(MlaWireFusedInferAttentionOuterSplitParams) == 312, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionOuterSplitParams, bN2End) == 0,
              "baseline FusedInferAttentionOuterSplitParams.bN2End offset");
static_assert(offsetof(MlaWireFusedInferAttentionOuterSplitParams, gS1End) == 104,
              "baseline FusedInferAttentionOuterSplitParams.gS1End offset");
static_assert(offsetof(MlaWireFusedInferAttentionOuterSplitParams, s2End) == 208,
              "baseline FusedInferAttentionOuterSplitParams.s2End offset");

struct alignas(8) MlaWireFusedInferAttentionFlashDecodeParams {
    uint32_t numOfFdHead;
    uint32_t reserved;
    uint32_t gS1BaseSizeOfFd;
    uint32_t usedVecNumOfFd;
    uint32_t bN2IdxOfFdHead[26];
    uint32_t gS1IdxOfFdHead[26];
    uint32_t s2SplitNumOfFdHead[26];
    uint32_t s2SplitStartIdxOfCore[26];
    uint32_t gS1SplitNumOfFdHead[26];
    uint32_t gS1LastPartSizeOfFdHead[26];
    uint32_t gS1IdxEndOfFdHead[52];
    uint32_t gS1IdxEndOfFdHeadSplit[52];
};
static_assert(sizeof(MlaWireFusedInferAttentionFlashDecodeParams) == 1056, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, numOfFdHead) == 0,
              "baseline FusedInferAttentionFlashDecodeParams.numOfFdHead offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, reserved) == 4,
              "baseline FusedInferAttentionFlashDecodeParams.reserved offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, gS1BaseSizeOfFd) == 8,
              "baseline FusedInferAttentionFlashDecodeParams.gS1BaseSizeOfFd offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, usedVecNumOfFd) == 12,
              "baseline FusedInferAttentionFlashDecodeParams.usedVecNumOfFd offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, bN2IdxOfFdHead) == 16,
              "baseline FusedInferAttentionFlashDecodeParams.bN2IdxOfFdHead offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, gS1IdxOfFdHead) == 120,
              "baseline FusedInferAttentionFlashDecodeParams.gS1IdxOfFdHead offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, s2SplitNumOfFdHead) == 224,
              "baseline FusedInferAttentionFlashDecodeParams.s2SplitNumOfFdHead offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, s2SplitStartIdxOfCore) == 328,
              "baseline FusedInferAttentionFlashDecodeParams.s2SplitStartIdxOfCore offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, gS1SplitNumOfFdHead) == 432,
              "baseline FusedInferAttentionFlashDecodeParams.gS1SplitNumOfFdHead offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, gS1LastPartSizeOfFdHead) == 536,
              "baseline FusedInferAttentionFlashDecodeParams.gS1LastPartSizeOfFdHead offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, gS1IdxEndOfFdHead) == 640,
              "baseline FusedInferAttentionFlashDecodeParams.gS1IdxEndOfFdHead offset");
static_assert(offsetof(MlaWireFusedInferAttentionFlashDecodeParams, gS1IdxEndOfFdHeadSplit) == 848,
              "baseline FusedInferAttentionFlashDecodeParams.gS1IdxEndOfFdHeadSplit offset");

struct alignas(8) MlaWireFusedInferAttentionPrefixParams {
    uint64_t prefixMaxLen;
    uint64_t prefixLen;
    uint8_t prefixFlag;
};
static_assert(sizeof(MlaWireFusedInferAttentionPrefixParams) == 24, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionPrefixParams, prefixMaxLen) == 0,
              "baseline FusedInferAttentionPrefixParams.prefixMaxLen offset");
static_assert(offsetof(MlaWireFusedInferAttentionPrefixParams, prefixLen) == 8,
              "baseline FusedInferAttentionPrefixParams.prefixLen offset");
static_assert(offsetof(MlaWireFusedInferAttentionPrefixParams, prefixFlag) == 16,
              "baseline FusedInferAttentionPrefixParams.prefixFlag offset");

struct alignas(8) MlaWireFusedInferAttentionPseParams {
    uint32_t pseShiftFlag;
    uint32_t pseShiftByBatch;
    uint32_t pseShiftS1;
    uint32_t pseShiftS2;
};
static_assert(sizeof(MlaWireFusedInferAttentionPseParams) == 16, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionPseParams, pseShiftFlag) == 0,
              "baseline FusedInferAttentionPseParams.pseShiftFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionPseParams, pseShiftByBatch) == 4,
              "baseline FusedInferAttentionPseParams.pseShiftByBatch offset");
static_assert(offsetof(MlaWireFusedInferAttentionPseParams, pseShiftS1) == 8,
              "baseline FusedInferAttentionPseParams.pseShiftS1 offset");
static_assert(offsetof(MlaWireFusedInferAttentionPseParams, pseShiftS2) == 12,
              "baseline FusedInferAttentionPseParams.pseShiftS2 offset");

struct alignas(8) MlaWireFusedInferAttentionLeftPaddingParams {
    uint32_t qPaddingFlag;
    uint32_t kvPaddingFlag;
};
static_assert(sizeof(MlaWireFusedInferAttentionLeftPaddingParams) == 8, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionLeftPaddingParams, qPaddingFlag) == 0,
              "baseline FusedInferAttentionLeftPaddingParams.qPaddingFlag offset");
static_assert(offsetof(MlaWireFusedInferAttentionLeftPaddingParams, kvPaddingFlag) == 4,
              "baseline FusedInferAttentionLeftPaddingParams.kvPaddingFlag offset");

struct alignas(8) MlaWireFusedInferAttentionPostQuantParams {
    uint32_t isPerChnOut;
    uint32_t isOutQuantTypeBf16;
};
static_assert(sizeof(MlaWireFusedInferAttentionPostQuantParams) == 8, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionPostQuantParams, isPerChnOut) == 0,
              "baseline FusedInferAttentionPostQuantParams.isPerChnOut offset");
static_assert(offsetof(MlaWireFusedInferAttentionPostQuantParams, isOutQuantTypeBf16) == 4,
              "baseline FusedInferAttentionPostQuantParams.isOutQuantTypeBf16 offset");

struct alignas(8) MlaWireFusedInferAttentionKvStrideParams {
    uint64_t keyBnStride;
    uint64_t valueBnStride;
    uint64_t keyRopeBnStride;
};
static_assert(sizeof(MlaWireFusedInferAttentionKvStrideParams) == 24, "baseline tiling size");
static_assert(offsetof(MlaWireFusedInferAttentionKvStrideParams, keyBnStride) == 0,
              "baseline FusedInferAttentionKvStrideParams.keyBnStride offset");
static_assert(offsetof(MlaWireFusedInferAttentionKvStrideParams, valueBnStride) == 8,
              "baseline FusedInferAttentionKvStrideParams.valueBnStride offset");
static_assert(offsetof(MlaWireFusedInferAttentionKvStrideParams, keyRopeBnStride) == 16,
              "baseline FusedInferAttentionKvStrideParams.keyRopeBnStride offset");

struct alignas(8) MlaNonQuantTiling {
    MlaWireFusedInferAttentionBaseParams baseParams;
    MlaWireFusedInferAttentionPageAttentionParams pageAttenParams;
    MlaWireFusedInferAttentionMaskParams maskParams;
    MlaWireFusedInferAttentionWorkspaceParams workspaceParams;
    MlaWireFusedInferAttentionInnerSplitParams innerSplitParams;
    MlaWireFusedInferAttentionOuterSplitParams outerSplitParams;
    MlaWireFusedInferAttentionFlashDecodeParams fdParams;
    MlaWireFusedInferAttentionPrefixParams prefixParams;
    MlaWireFusedInferAttentionPseParams pseParams;
    MlaWireFusedInferAttentionLeftPaddingParams leftPaddingParams;
    MlaWireFusedInferAttentionPostQuantParams postquantParams;
    MlaWireFusedInferAttentionKvStrideParams kvStrideParams;
};
static_assert(sizeof(MlaNonQuantTiling) == 1600, "baseline tiling size");
static_assert(offsetof(MlaNonQuantTiling, baseParams) == 0,
              "baseline FusedInferAttentionScoreTilingData.baseParams offset");
static_assert(offsetof(MlaNonQuantTiling, pageAttenParams) == 88,
              "baseline FusedInferAttentionScoreTilingData.pageAttenParams offset");
static_assert(offsetof(MlaNonQuantTiling, maskParams) == 96,
              "baseline FusedInferAttentionScoreTilingData.maskParams offset");
static_assert(offsetof(MlaNonQuantTiling, workspaceParams) == 128,
              "baseline FusedInferAttentionScoreTilingData.workspaceParams offset");
static_assert(offsetof(MlaNonQuantTiling, innerSplitParams) == 144,
              "baseline FusedInferAttentionScoreTilingData.innerSplitParams offset");
static_assert(offsetof(MlaNonQuantTiling, outerSplitParams) == 152,
              "baseline FusedInferAttentionScoreTilingData.outerSplitParams offset");
static_assert(offsetof(MlaNonQuantTiling, fdParams) == 464,
              "baseline FusedInferAttentionScoreTilingData.fdParams offset");
static_assert(offsetof(MlaNonQuantTiling, prefixParams) == 1520,
              "baseline FusedInferAttentionScoreTilingData.prefixParams offset");
static_assert(offsetof(MlaNonQuantTiling, pseParams) == 1544,
              "baseline FusedInferAttentionScoreTilingData.pseParams offset");
static_assert(offsetof(MlaNonQuantTiling, leftPaddingParams) == 1560,
              "baseline FusedInferAttentionScoreTilingData.leftPaddingParams offset");
static_assert(offsetof(MlaNonQuantTiling, postquantParams) == 1568,
              "baseline FusedInferAttentionScoreTilingData.postquantParams offset");
static_assert(offsetof(MlaNonQuantTiling, kvStrideParams) == 1576,
              "baseline FusedInferAttentionScoreTilingData.kvStrideParams offset");

struct alignas(8) MlaWireIncreFlashAttentionBaseParamsMla {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t actualLenQDims;
    uint32_t actualLenDims;
    uint32_t antiquantMode;
    uint32_t attenMaskFlag;
    uint32_t attenMaskSize;
    uint32_t sparseMode;
    uint32_t outputLayout;
    uint32_t softmaxLseFlag;
    uint32_t tSeqSize;
};
static_assert(sizeof(MlaWireIncreFlashAttentionBaseParamsMla) == 64, "baseline tiling size");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, batchSize) == 0,
              "baseline IncreFlashAttentionBaseParamsMla.batchSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, seqSize) == 4,
              "baseline IncreFlashAttentionBaseParamsMla.seqSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, qSeqSize) == 8,
              "baseline IncreFlashAttentionBaseParamsMla.qSeqSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, blockSize) == 12,
              "baseline IncreFlashAttentionBaseParamsMla.blockSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, maxBlockNumPerBatch) == 16,
              "baseline IncreFlashAttentionBaseParamsMla.maxBlockNumPerBatch offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, scaleValue) == 20,
              "baseline IncreFlashAttentionBaseParamsMla.scaleValue offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, nNumOfQInOneGroup) == 24,
              "baseline IncreFlashAttentionBaseParamsMla.nNumOfQInOneGroup offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, actualLenQDims) == 28,
              "baseline IncreFlashAttentionBaseParamsMla.actualLenQDims offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, actualLenDims) == 32,
              "baseline IncreFlashAttentionBaseParamsMla.actualLenDims offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, antiquantMode) == 36,
              "baseline IncreFlashAttentionBaseParamsMla.antiquantMode offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, attenMaskFlag) == 40,
              "baseline IncreFlashAttentionBaseParamsMla.attenMaskFlag offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, attenMaskSize) == 44,
              "baseline IncreFlashAttentionBaseParamsMla.attenMaskSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, sparseMode) == 48,
              "baseline IncreFlashAttentionBaseParamsMla.sparseMode offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, outputLayout) == 52,
              "baseline IncreFlashAttentionBaseParamsMla.outputLayout offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, softmaxLseFlag) == 56,
              "baseline IncreFlashAttentionBaseParamsMla.softmaxLseFlag offset");
static_assert(offsetof(MlaWireIncreFlashAttentionBaseParamsMla, tSeqSize) == 60,
              "baseline IncreFlashAttentionBaseParamsMla.tSeqSize offset");

struct alignas(8) MlaWireIncreFlashAttentionSplitKVParamsMla {
    uint32_t s2;
    uint32_t sInnerLoopSize;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};
static_assert(sizeof(MlaWireIncreFlashAttentionSplitKVParamsMla) == 16, "baseline tiling size");
static_assert(offsetof(MlaWireIncreFlashAttentionSplitKVParamsMla, s2) == 0,
              "baseline IncreFlashAttentionSplitKVParamsMla.s2 offset");
static_assert(offsetof(MlaWireIncreFlashAttentionSplitKVParamsMla, sInnerLoopSize) == 4,
              "baseline IncreFlashAttentionSplitKVParamsMla.sInnerLoopSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionSplitKVParamsMla, accumOutSize) == 8,
              "baseline IncreFlashAttentionSplitKVParamsMla.accumOutSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionSplitKVParamsMla, logSumExpSize) == 12,
              "baseline IncreFlashAttentionSplitKVParamsMla.logSumExpSize offset");

struct alignas(8) MlaWireIncreFlashAttentionTNDSplitCoreParamsMla {
    uint32_t tndFDCoreArrLen;
    uint32_t reserve;
    uint32_t balanceFDCoreBArr[26];
    uint32_t balanceFDCoreS1Arr[26];
    uint32_t balanceFDCoreKVSplitArr[26];
    uint32_t balanceFDCoreStartKVSplitNum[26];
    uint32_t gS1IdxEndOfFdHead[52];
    uint32_t gS1IdxEndOfFdHeadSplit[52];
    uint32_t usedVecNumOfFd;
    uint32_t FdBalanceFlag;
    uint32_t gS1SplitNumOfFdHeadMla[52];
    uint32_t gS1LastPartSizeOfFdHead[52];
};
static_assert(sizeof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla) == 1264, "baseline tiling size");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, tndFDCoreArrLen) == 0,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.tndFDCoreArrLen offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, reserve) == 4,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.reserve offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, balanceFDCoreBArr) == 8,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.balanceFDCoreBArr offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, balanceFDCoreS1Arr) == 112,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.balanceFDCoreS1Arr offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, balanceFDCoreKVSplitArr) == 216,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.balanceFDCoreKVSplitArr offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, balanceFDCoreStartKVSplitNum) == 320,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.balanceFDCoreStartKVSplitNum offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, gS1IdxEndOfFdHead) == 424,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.gS1IdxEndOfFdHead offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, gS1IdxEndOfFdHeadSplit) == 632,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.gS1IdxEndOfFdHeadSplit offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, usedVecNumOfFd) == 840,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.usedVecNumOfFd offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, FdBalanceFlag) == 844,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.FdBalanceFlag offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, gS1SplitNumOfFdHeadMla) == 848,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.gS1SplitNumOfFdHeadMla offset");
static_assert(offsetof(MlaWireIncreFlashAttentionTNDSplitCoreParamsMla, gS1LastPartSizeOfFdHead) == 1056,
              "baseline IncreFlashAttentionTNDSplitCoreParamsMla.gS1LastPartSizeOfFdHead offset");

struct alignas(8) MlaWireIncreFlashAttentionCoreParamsMla {
    uint32_t coreBEnd[26];
    uint32_t coreSidxEnd[26];
    uint32_t coreS1OuterEnd[26];
    uint32_t coreS2End[26];
};
static_assert(sizeof(MlaWireIncreFlashAttentionCoreParamsMla) == 416, "baseline tiling size");
static_assert(offsetof(MlaWireIncreFlashAttentionCoreParamsMla, coreBEnd) == 0,
              "baseline IncreFlashAttentionCoreParamsMla.coreBEnd offset");
static_assert(offsetof(MlaWireIncreFlashAttentionCoreParamsMla, coreSidxEnd) == 104,
              "baseline IncreFlashAttentionCoreParamsMla.coreSidxEnd offset");
static_assert(offsetof(MlaWireIncreFlashAttentionCoreParamsMla, coreS1OuterEnd) == 208,
              "baseline IncreFlashAttentionCoreParamsMla.coreS1OuterEnd offset");
static_assert(offsetof(MlaWireIncreFlashAttentionCoreParamsMla, coreS2End) == 312,
              "baseline IncreFlashAttentionCoreParamsMla.coreS2End offset");

struct alignas(8) MlaWireIncreFlashAttentionSingleCoreParamsMla {
    uint32_t singleProcessSInnerSize;
    uint32_t usedCoreNum;
    uint32_t groupSplitSize;
    uint32_t s1SplitSize;
};
static_assert(sizeof(MlaWireIncreFlashAttentionSingleCoreParamsMla) == 16, "baseline tiling size");
static_assert(offsetof(MlaWireIncreFlashAttentionSingleCoreParamsMla, singleProcessSInnerSize) == 0,
              "baseline IncreFlashAttentionSingleCoreParamsMla.singleProcessSInnerSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionSingleCoreParamsMla, usedCoreNum) == 4,
              "baseline IncreFlashAttentionSingleCoreParamsMla.usedCoreNum offset");
static_assert(offsetof(MlaWireIncreFlashAttentionSingleCoreParamsMla, groupSplitSize) == 8,
              "baseline IncreFlashAttentionSingleCoreParamsMla.groupSplitSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionSingleCoreParamsMla, s1SplitSize) == 12,
              "baseline IncreFlashAttentionSingleCoreParamsMla.s1SplitSize offset");

struct alignas(8) MlaWireIncreFlashAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;
};
static_assert(sizeof(MlaWireIncreFlashAttentionSingleCoreTensorSizeMla) == 8, "baseline tiling size");
static_assert(offsetof(MlaWireIncreFlashAttentionSingleCoreTensorSizeMla, mmResUbSize) == 0,
              "baseline IncreFlashAttentionSingleCoreTensorSizeMla.mmResUbSize offset");
static_assert(offsetof(MlaWireIncreFlashAttentionSingleCoreTensorSizeMla, bmm2ResUbSize) == 4,
              "baseline IncreFlashAttentionSingleCoreTensorSizeMla.bmm2ResUbSize offset");

struct alignas(8) MlaLegacyTiling {
    MlaWireIncreFlashAttentionBaseParamsMla baseParams;
    MlaWireIncreFlashAttentionSplitKVParamsMla splitKVParams;
    MlaWireIncreFlashAttentionTNDSplitCoreParamsMla tndSplitCoreParams;
    MlaWireIncreFlashAttentionCoreParamsMla increFlashAttentionCoreParams;
    MlaWireIncreFlashAttentionSingleCoreParamsMla increFlashAttentionSingleCoreParams;
    MlaWireIncreFlashAttentionSingleCoreTensorSizeMla increFlashAttentionSingleCoreTensorSize;
};
static_assert(sizeof(MlaLegacyTiling) == 1784, "baseline tiling size");
static_assert(offsetof(MlaLegacyTiling, baseParams) == 0,
              "baseline IncreFlashAttentionTilingDataMla.baseParams offset");
static_assert(offsetof(MlaLegacyTiling, splitKVParams) == 64,
              "baseline IncreFlashAttentionTilingDataMla.splitKVParams offset");
static_assert(offsetof(MlaLegacyTiling, tndSplitCoreParams) == 80,
              "baseline IncreFlashAttentionTilingDataMla.tndSplitCoreParams offset");
static_assert(offsetof(MlaLegacyTiling, increFlashAttentionCoreParams) == 1344,
              "baseline IncreFlashAttentionTilingDataMla.increFlashAttentionCoreParams offset");
static_assert(offsetof(MlaLegacyTiling, increFlashAttentionSingleCoreParams) == 1760,
              "baseline IncreFlashAttentionTilingDataMla.increFlashAttentionSingleCoreParams offset");
static_assert(offsetof(MlaLegacyTiling, increFlashAttentionSingleCoreTensorSize) == 1776,
              "baseline IncreFlashAttentionTilingDataMla.increFlashAttentionSingleCoreTensorSize offset");

enum class MlaFieldRole : uint8_t {
    STATIC,
    DYNAMIC,
    COMBINED
};
enum class MlaFieldUnit : uint8_t {
    SCALAR,
    ELEMENTS,
    BYTES,
    INDEX
};
enum class MlaFieldType : uint8_t {
    U8,
    U32,
    I32,
    U64,
    F32
};
struct MlaTilingField {
    const char *name;
    uint32_t offset;
    uint32_t count;
    uint32_t elementBytes;
    MlaFieldRole role;
    MlaFieldUnit unit;
    MlaFieldType type;
};

// name, byte offset, element count, element size, ownership, unit, scalar type.
constexpr MlaTilingField kMlaNonQuantFields[] = {
    {"baseParams.bSize", 0, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.n2Size", 4, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.gSize", 8, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.s1Size", 12, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
    {"baseParams.s2Size", 16, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
    {"baseParams.headDim", 20, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
    {"baseParams.headDimRope", 24, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"baseParams.actualSeqS1Dims", 28, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.actualSeqS2Dims", 32, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.accumQSeqFlag", 36, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.accumKVSeqFlag", 40, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.scaleValue", 44, 1, sizeof(float), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::F32},
    {"baseParams.usedCoreNum", 48, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.outputLayout", 52, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.batchContinuous", 56, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.softmaxLseFlag", 60, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.needInit", 64, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.slidingFlag", 68, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.l2CacheOffFlag", 72, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.isLegacyIfa", 76, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.isEmptyBatchOverHalf", 80, 1, sizeof(uint8_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U8},
    {"pageAttenParams.blockSize", 88, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"pageAttenParams.maxBlockNumPerBatch", 92, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"maskParams.attenMaskFlag", 96, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"maskParams.attenMaskBatchStride", 100, 1, sizeof(uint32_t), MlaFieldRole::COMBINED, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"maskParams.attenMaskStride", 104, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"maskParams.preToken", 108, 1, sizeof(int32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::I32},
    {"maskParams.nextToken", 112, 1, sizeof(int32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::I32},
    {"maskParams.isRowInvalid", 116, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"maskParams.isExistRowInvalid", 120, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"maskParams.sparseMode", 124, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"workspaceParams.mm1ResSize", 128, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"workspaceParams.mm2ResSize", 132, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"workspaceParams.fdAccumOutSize", 136, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"workspaceParams.fdLogSumExpSize", 140, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"innerSplitParams.mBaseSize", 144, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"innerSplitParams.s2BaseSize", 148, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"outerSplitParams.bN2End", 152, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"outerSplitParams.gS1End", 256, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"outerSplitParams.s2End", 360, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.numOfFdHead", 464, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"fdParams.reserved", 468, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"fdParams.gS1BaseSizeOfFd", 472, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"fdParams.usedVecNumOfFd", 476, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"fdParams.bN2IdxOfFdHead", 480, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.gS1IdxOfFdHead", 584, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.s2SplitNumOfFdHead", 688, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.s2SplitStartIdxOfCore", 792, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.gS1SplitNumOfFdHead", 896, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.gS1LastPartSizeOfFdHead", 1000, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.gS1IdxEndOfFdHead", 1104, 52, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"fdParams.gS1IdxEndOfFdHeadSplit", 1312, 52, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"prefixParams.prefixMaxLen", 1520, 1, sizeof(uint64_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U64},
    {"prefixParams.prefixLen", 1528, 1, sizeof(uint64_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U64},
    {"prefixParams.prefixFlag", 1536, 1, sizeof(uint8_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U8},
    {"pseParams.pseShiftFlag", 1544, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"pseParams.pseShiftByBatch", 1548, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"pseParams.pseShiftS1", 1552, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"pseParams.pseShiftS2", 1556, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"leftPaddingParams.qPaddingFlag", 1560, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"leftPaddingParams.kvPaddingFlag", 1564, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"postquantParams.isPerChnOut", 1568, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"postquantParams.isOutQuantTypeBf16", 1572, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"kvStrideParams.keyBnStride", 1576, 1, sizeof(uint64_t), MlaFieldRole::STATIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U64},
    {"kvStrideParams.valueBnStride", 1584, 1, sizeof(uint64_t), MlaFieldRole::STATIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U64},
    {"kvStrideParams.keyRopeBnStride", 1592, 1, sizeof(uint64_t), MlaFieldRole::STATIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U64},
};

// name, byte offset, element count, element size, ownership, unit, scalar type.
constexpr MlaTilingField kMlaLegacyFields[] = {
    {"baseParams.batchSize", 0, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.seqSize", 4, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
    {"baseParams.qSeqSize", 8, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
    {"baseParams.blockSize", 12, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.maxBlockNumPerBatch", 16, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.scaleValue", 20, 1, sizeof(float), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::F32},
    {"baseParams.nNumOfQInOneGroup", 24, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.actualLenQDims", 28, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.actualLenDims", 32, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.antiquantMode", 36, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.attenMaskFlag", 40, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.attenMaskSize", 44, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.sparseMode", 48, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.outputLayout", 52, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"baseParams.softmaxLseFlag", 56, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"baseParams.tSeqSize", 60, 1, sizeof(uint32_t), MlaFieldRole::STATIC, MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
    {"splitKVParams.s2", 64, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"splitKVParams.sInnerLoopSize", 68, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"splitKVParams.accumOutSize", 72, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"splitKVParams.logSumExpSize", 76, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::ELEMENTS,
     MlaFieldType::U32},
    {"tndSplitCoreParams.tndFDCoreArrLen", 80, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"tndSplitCoreParams.reserve", 84, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"tndSplitCoreParams.balanceFDCoreBArr", 88, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"tndSplitCoreParams.balanceFDCoreS1Arr", 192, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"tndSplitCoreParams.balanceFDCoreKVSplitArr", 296, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::INDEX, MlaFieldType::U32},
    {"tndSplitCoreParams.balanceFDCoreStartKVSplitNum", 400, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::INDEX, MlaFieldType::U32},
    {"tndSplitCoreParams.gS1IdxEndOfFdHead", 504, 52, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"tndSplitCoreParams.gS1IdxEndOfFdHeadSplit", 712, 52, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"tndSplitCoreParams.usedVecNumOfFd", 920, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"tndSplitCoreParams.FdBalanceFlag", 924, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::SCALAR,
     MlaFieldType::U32},
    {"tndSplitCoreParams.gS1SplitNumOfFdHeadMla", 928, 52, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"tndSplitCoreParams.gS1LastPartSizeOfFdHead", 1136, 52, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::INDEX, MlaFieldType::U32},
    {"increFlashAttentionCoreParams.coreBEnd", 1344, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"increFlashAttentionCoreParams.coreSidxEnd", 1448, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::INDEX, MlaFieldType::U32},
    {"increFlashAttentionCoreParams.coreS1OuterEnd", 1552, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::INDEX, MlaFieldType::U32},
    {"increFlashAttentionCoreParams.coreS2End", 1656, 26, sizeof(uint32_t), MlaFieldRole::DYNAMIC, MlaFieldUnit::INDEX,
     MlaFieldType::U32},
    {"increFlashAttentionSingleCoreParams.singleProcessSInnerSize", 1760, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"increFlashAttentionSingleCoreParams.usedCoreNum", 1764, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"increFlashAttentionSingleCoreParams.groupSplitSize", 1768, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"increFlashAttentionSingleCoreParams.s1SplitSize", 1772, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::SCALAR, MlaFieldType::U32},
    {"increFlashAttentionSingleCoreTensorSize.mmResUbSize", 1776, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
    {"increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize", 1780, 1, sizeof(uint32_t), MlaFieldRole::DYNAMIC,
     MlaFieldUnit::ELEMENTS, MlaFieldType::U32},
};

// Only these byte ranges travel from AICPU to AICore. The full field tables
// above and test_flash_mla_contract check their ownership byte by byte.
// Each range is [begin, end); static fields, padding and metadata tail are excluded.
struct MlaDynamicRange {
    uint32_t begin;
    uint32_t end;
};
constexpr MlaDynamicRange kMlaNonQuantDynamicRanges[] = {
    {12, 20},    // baseParams.s1Size, s2Size
    {48, 52},    // baseParams.usedCoreNum
    {64, 68},    // baseParams.needInit
    {80, 81},    // baseParams.isEmptyBatchOverHalf (one byte, excluding padding)
    {128, 1520}, // workspaceParams, innerSplitParams, outerSplitParams, fdParams
};
constexpr MlaDynamicRange kMlaLegacyDynamicRanges[] = {
    {4, 12},    // baseParams.seqSize, qSeqSize
    {64, 1784}, // splitKVParams through increFlashAttentionSingleCoreTensorSize
};

} // namespace FlashMla
#endif
