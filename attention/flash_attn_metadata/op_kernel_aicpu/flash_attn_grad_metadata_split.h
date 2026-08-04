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
 * \file flash_attn_grad_metadata_split.h
 * \brief FAG (Flash Attn Grad) metadata splitting logic, migrated from
 *        flash_attention_score_grad tiling DoSparse() and sub-functions.
 *        Covers 3 scenarios: Dense, TND, Sparse (causal/band).
 *        BN2 optimization is not included (to be added later).
 *
 * Migration source mapping:
 *   InitFagParams        <- DoSplit() + SetSparseParams() + ProcessTokensInfo() simplified
 *   DoFagDenseSplit      <- normal_regbase.cpp:701-723 (else branch)
 *   DoFagTndSplit        <- varlen_regbase.cpp:953-1044 (GetSparseUnpadBlockInfo)
 *   DoFagSparseBlockInfo <- normal_regbase.cpp:1553-1622 (GetSparseBlockInfo)
 *   GetFagParseS1S2OuterInfo <- normal_regbase.cpp:1509-1551
 *   FagCheckSparseLeftAndRight <- normal_regbase.cpp:734-771
 *   FagFillBlockInfoLoadBalance <- normal_regbase.cpp:1685-1750
 */
#ifndef FLASH_ATTN_GRAD_METADATA_SPLIT_H
#define FLASH_ATTN_GRAD_METADATA_SPLIT_H

#include "flash_attn_metadata_aicpu.h"
#include <algorithm>
#include <cmath>
#include <climits>

using namespace optiling;

namespace aicpu {

// ===== FAG metadata start =====

// Helper: align value up to alignment
static inline int64_t FagAlignTo(int64_t value, int64_t alignment)
{
    if (alignment <= 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

// Helper: calcle actual token for TND/unpad sparse modes
// Source: common_regbase.cpp:914-936 CalcleActualToken
static inline void FagCalcleActualToken(uint32_t sparseMode, int64_t s1Token, int64_t s2Token,
    int64_t actualS1Len, int64_t actualS2Len, int64_t &actualCalcS1Token, int64_t &actualCalcS2Token)
{
    actualCalcS1Token = s1Token;
    actualCalcS2Token = s2Token;
    // RIGHT_DOWN_CASUAL_BAND(7) / BAND_LEFT_UP_CASUAL(8) with non-bandIdx batch: full mask
    // Not applicable for FAG (mask_mode only 0/3/4), but kept for completeness
    if (sparseMode == FAG_SPARSE_RIGHT_DOWN_CAUSAL || sparseMode == FAG_SPARSE_BAND) {
        actualCalcS1Token = actualCalcS1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = actualCalcS2Token - actualS1Len + actualS2Len;
    }
}

// ===== InitFagParams: initialize split parameters =====
// Source: normal_regbase.cpp:478-509 DoSplit + common_regbase.cpp:1256-1281 ProcessTokensInfo
void FlashAttnMetadataCpuKernel::InitFagParams()
{
    // s1Inner: always 64 (FP32 large-d scenario not supported yet)
    fagS1Inner_ = 64;
    fagS2Inner_ = 128;

    // s1CvInner = min(s1Inner * S1CV_RATIO_DEFAULT, s1)
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);
    fagS1CvInner_ = (fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT > s1) ? s1 : fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT;
    fagCvS2Inner_ = (fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT > s2) ? s2 : fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT;

    fagS1Outer_ = (s1 + fagS1CvInner_ - 1) / fagS1CvInner_;
    fagS2Outer_ = (s2 + fagCvS2Inner_ - 1) / fagCvS2Inner_;

    fagG_ = static_cast<int64_t>(numHeadsQ_) / static_cast<int64_t>(numHeadsKv_);

    // Layout type mapping
    if (layoutQ_ == "TND") {
        fagLayoutType_ = FAG_INPUT_FORMAT_TND;
    } else if (layoutQ_ == "BNSD") {
        fagLayoutType_ = FAG_INPUT_FORMAT_BN2GS2D;
    } else {
        fagLayoutType_ = FAG_INPUT_FORMAT_BS2N2GD;
    }

    // Sparse mode and token mapping
    // Source: common_regbase.cpp:1256-1281 ProcessTokensInfo + common_regbase.cpp:1538-1578 SetSparseParams
    fagSparseMode_ = FAG_SPARSE_NO_MASK;
    fagIsSparse_ = false;
    fagS1Token_ = winLeft_;
    fagS2Token_ = winRight_;

    if (maskMode_ == 0) {
        // NO_MASK: always full compute (Dense split), no sparse optimization
        fagIsSparse_ = false;
    } else if (maskMode_ == 3) {
        // RIGHT_DOWN_CAUSAL
        // Source: common_regbase.cpp:1262-1265: s1Token=INT32_MAX, s2Token=0
        fagSparseMode_ = FAG_SPARSE_RIGHT_DOWN_CAUSAL;
        fagIsSparse_ = true;
        fagS1Token_ = INT32_MAX;
        fagS2Token_ = 0;
        // Source: common_regbase.cpp:1270-1275: "sparse_mode==3(RIGHT_DOWN_CAUSAL)时，
        // token以右下角为基准，需要校正"。pad场景(非TND)需校正token以将causal对角线
        // 从左上角偏移到右下角。TND场景由CalcleActualToken在运行时按per-batch校正。
        if (fagLayoutType_ != FAG_INPUT_FORMAT_TND) {
            fagS1Token_ = fagS1Token_ + s1 - s2;
            fagS2Token_ = fagS2Token_ - s1 + s2;
        }
    } else if (maskMode_ == 4) {
        // BAND
        // Source: common_regbase.cpp:1270-1275: pad scenario token correction for non-TND
        fagSparseMode_ = FAG_SPARSE_BAND;
        if (s1 > fagS1Token_ || s2 > fagS2Token_) {
            fagIsSparse_ = true;
        }
        if (fagLayoutType_ != FAG_INPUT_FORMAT_TND) {
            fagS1Token_ = fagS1Token_ + s1 - s2;
            fagS2Token_ = fagS2Token_ - s1 + s2;
        }
    }

    // For TND: if no actual seq lens available, use maxSeqlen as single-batch
    if (fagLayoutType_ == FAG_INPUT_FORMAT_TND && actualSeqlenQ_.empty()) {
        actualSeqlenQ_.push_back(s1);
        actualSeqlenKv_.push_back(s2);
    }

    fagSplitAxis_ = FAG_SPLIT_AXIS_BN2GS1S2;
}

// ===== DoFagSparse: dispatch to 3 scenarios =====
// Source: normal_regbase.cpp:660-731 DoSparse (BN2 paths removed)
void FlashAttnMetadataCpuKernel::DoFagSparse()
{
    fagSplitAxis_ = FAG_SPLIT_AXIS_BN2GS1S2;
    if (fagLayoutType_ == FAG_INPUT_FORMAT_TND) {
        DoFagTndSplit();
    } else if (fagIsSparse_) {
        DoFagSparseBlockInfo();
    } else {
        DoFagDenseSplit();
    }
}

// ===== DoFagDenseSplit: full compute, linear uniform split =====
// Source: normal_regbase.cpp:701-723
void FlashAttnMetadataCpuKernel::DoFagDenseSplit()
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t fusedOuter = b * n2 * g * fagS1Outer_ * fagS2Outer_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    for (int64_t i = 0; i < blockOuter; i++) {
        fagBlockStarts_[i] = blockFactor * i;
        fagBlockEnds_[i] = std::min(blockFactor * (i + 1), fusedOuter);
    }
    for (uint32_t i = static_cast<uint32_t>(blockOuter); i < FAG_CORE_LIST_NUM; i++) {
        fagBlockStarts_[i] = 0;
        fagBlockEnds_[i] = 0;
    }
}

// ===== DoFagSparseBlockInfo: non-TND sparse split (causal/band) =====
// Source: normal_regbase.cpp:1553-1622 GetSparseBlockInfo
void FlashAttnMetadataCpuKernel::DoFagSparseBlockInfo()
{
    int64_t parseInfo[FAG_MAX_S2_OUTER][FAG_ARRAY_LENGTH];
    std::memset(parseInfo, 0, sizeof(parseInfo));
    GetFagParseS1S2OuterInfo(parseInfo);

    int64_t s1s2oCount = parseInfo[fagS2Outer_ - 1][2]; // LENGTH_IDX = 2

    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    int64_t fusedOuter = b * n2 * g * s1s2oCount;
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    int64_t n2gs1s2o = n2 * g * s1s2oCount;
    int64_t gs1s2o = g * s1s2oCount;

    fagBlockStarts_[0] = 0;
    fagBlockEnds_[blockOuter - 1] = b * n2 * g * fagS1Outer_ * fagS2Outer_;

    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        int64_t bIdx = currentIdx / n2gs1s2o;
        int64_t bTail = currentIdx % n2gs1s2o;
        int64_t n2Idx = bTail / gs1s2o;
        int64_t n2Tail = bTail % gs1s2o;
        int64_t gIdx = n2Tail / s1s2oCount;
        int64_t gTail = n2Tail % s1s2oCount;

        // Reverse map: linear valid-block index -> (s1oIdx, s2oIdx)
        // Source: common_regbase.cpp:894-912 GetCommonS1S2OuterIndex
        int64_t s1oIdx = 0;
        int64_t s2oIdx = 0;
        int64_t preSize = 0;
        for (int64_t i = 0; i < fagS2Outer_; i++) {
            int64_t nextSize = parseInfo[i][2]; // LENGTH_IDX
            if (gTail >= preSize && gTail < nextSize) {
                s2oIdx = i;
                s1oIdx = parseInfo[i][0] + gTail - preSize - 1; // BEGIN_IDX=0
                break;
            }
            preSize = nextSize;
        }

        // Encode as flat BNGS1S2 index
        // Source: normal_regbase.cpp:1606-1608
        fagBlockStarts_[c] = (((bIdx * n2 + n2Idx) * g + gIdx) * fagS2Outer_ + s2oIdx) *
                             fagS1Outer_ + s1oIdx + 1;
        fagBlockEnds_[c - 1] = fagBlockStarts_[c];
    }
    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < FAG_CORE_LIST_NUM; c++) {
        fagBlockStarts_[c] = 0;
        fagBlockEnds_[c] = 0;
    }
}

// ===== GetFagParseS1S2OuterInfo: build sparse valid S1xS2 mapping table =====
// Source: normal_regbase.cpp:1509-1551
void FlashAttnMetadataCpuKernel::GetFagParseS1S2OuterInfo(int64_t (*parseInfo)[FAG_ARRAY_LENGTH])
{
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);
    for (int64_t i = 0; i < fagS2Outer_; i++) {
        int64_t leftIntersectionPoint = std::max(static_cast<int64_t>(0),
            fagCvS2Inner_ * i - fagS2Token_);
        if (leftIntersectionPoint > s1) {
            parseInfo[i][0] = (s1 + fagS1CvInner_ - 1) / fagS1CvInner_; // BEGIN_IDX=0
        } else {
            parseInfo[i][0] = leftIntersectionPoint / fagS1CvInner_;
        }
        // Source: normal_regbase.cpp:1519: cvBlockTail = s2CvTail for last block, cvS2Inner otherwise
        // s2CvTail = (s2 % cvS2Inner == 0) ? cvS2Inner : s2 % cvS2Inner
        int64_t cvBlockTail = (i == fagS2Outer_ - 1) ?
            (s2 % fagCvS2Inner_ == 0 ? fagCvS2Inner_ : s2 % fagCvS2Inner_) : fagCvS2Inner_;
        int64_t rightPoint = std::min(std::max(static_cast<int64_t>(0),
            fagCvS2Inner_ * i + cvBlockTail + fagS1Token_), s1);
        parseInfo[i][1] = (rightPoint + fagS1CvInner_ - 1) / fagS1CvInner_; // END_IDX=1

        int64_t tmpSize = (parseInfo[i][1] > parseInfo[i][0]) ?
            parseInfo[i][1] - parseInfo[i][0] : 0;
        if (i == 0) {
            parseInfo[i][2] = tmpSize; // LENGTH_IDX=2
        } else {
            parseInfo[i][2] = parseInfo[i - 1][2] + tmpSize;
        }
    }
}

// ===== FagCheckSparseLeftAndRight: check sparse validity for non-TND =====
// Source: normal_regbase.cpp:734-771
bool FlashAttnMetadataCpuKernel::FagCheckSparseLeftAndRight(int64_t s1oDimIdx,
    int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx)
{
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);

    if (fagSparseMode_ == FAG_SPARSE_RIGHT_DOWN_CAUSAL) {
        int64_t s2IgnoredEndLen = s1 - fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * (s1oDimIdx + 1);
        int64_t s2EndLen = (s2 > s2IgnoredEndLen) ? (s2 - s2IgnoredEndLen) : 0;
        return s2IdxLeft < s2EndLen;
    } else {
        // BAND mode
        int64_t s2SparseLeft = std::max(
            fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * s1oDimIdx - fagS1Token_, static_cast<int64_t>(0));
        s2SparseLeft = FagAlignTo(s2SparseLeft, FAG_ALIGN64);
        int64_t s2SparseRight = FagAlignTo(
            std::min(fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), s1) + fagS2Token_, FAG_ALIGN64);
        s2SparseRight = std::min(s2SparseRight, s2);
        return s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
    }
}

// ===== FagIsValid: check block validity for non-TND =====
// Source: normal_regbase.cpp:773-788
bool FlashAttnMetadataCpuKernel::FagIsValid(int64_t blockIdx)
{
    int64_t gDimTail = blockIdx % (fagS1Outer_ * fagS2Outer_);
    int64_t s2oDimIdx = gDimTail / fagS1Outer_;
    int64_t s1oDimIdx = gDimTail % fagS1Outer_;
    int64_t s2IdxLeft = s2oDimIdx * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT;
    int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT,
        static_cast<int64_t>(maxSeqlenKv_));
    return FagCheckSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, 0);
}

// ===== DoFagTndSplit: TND variable-length split =====
// Source: varlen_regbase.cpp:953-1044 GetSparseUnpadBlockInfo
void FlashAttnMetadataCpuKernel::DoFagTndSplit()
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    // Build calculatedBlockInfo and totalBlockInfo
    // Source: varlen_regbase.cpp:1046-1075 FillBlockInfo
    // calculatedBlockInfo[b][s2Outer][4]: BEGIN, END, SUM_S1S2, SUM_ALL
    // totalBlockInfo[b][2]: blockCount, cumulativePrefixSum
    int64_t calculatedBlockInfo[FAG_BATCH_MAX_SIZE][FAG_MAX_S2_OUTER][4];
    int64_t totalBlockInfo[FAG_BATCH_MAX_SIZE][2];
    std::memset(calculatedBlockInfo, 0, sizeof(calculatedBlockInfo));
    std::memset(totalBlockInfo, 0, sizeof(totalBlockInfo));

    for (int64_t i = 0; i < b; i++) {
        int64_t actualS1Len = actualSeqlenQ_[i];
        int64_t actualS2Len = actualSeqlenKv_[i];
        auto actualS1Outer = (actualS1Len + fagS1CvInner_ - 1) / fagS1CvInner_;
        auto actualS2Outer = (actualS2Len + fagCvS2Inner_ - 1) / fagCvS2Inner_;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        // CalValidUnpadBlockInfo
        // Source: varlen_regbase.cpp:1077-1125
        int64_t actualCalcS1Token, actualCalcS2Token;
        FagCalcleActualToken(fagSparseMode_, fagS1Token_, fagS2Token_,
            actualS1Len, actualS2Len, actualCalcS1Token, actualCalcS2Token);

        for (int64_t j = 0; j < fagS2Outer_; j++) {
            if (fagCvS2Inner_ * j >= actualS2Len) {
                calculatedBlockInfo[i][j][0] = 0; // BEGIN
                calculatedBlockInfo[i][j][1] = 0; // END
            } else {
                int64_t leftIntersectionPoint = std::max(
                    fagCvS2Inner_ * j - actualCalcS2Token, static_cast<int64_t>(0));
                if (leftIntersectionPoint > actualS1Len) {
                    calculatedBlockInfo[i][j][0] = (actualS1Len + fagS1CvInner_ - 1) / fagS1CvInner_;
                } else {
                    calculatedBlockInfo[i][j][0] = leftIntersectionPoint / fagS1CvInner_;
                }
                int64_t cvBlockTail = (fagCvS2Inner_ * (j + 1) > actualS2Len) ?
                    actualS2Len - fagCvS2Inner_ * j : fagCvS2Inner_;
                calculatedBlockInfo[i][j][1] = (std::min(actualS1Len,
                    std::max(fagCvS2Inner_ * j + cvBlockTail + actualCalcS1Token, static_cast<int64_t>(0))) +
                    fagS1CvInner_ - 1) / fagS1CvInner_;
            }
            int64_t tmpLength = (calculatedBlockInfo[i][j][1] > calculatedBlockInfo[i][j][0]) ?
                calculatedBlockInfo[i][j][1] - calculatedBlockInfo[i][j][0] : 0;
            if (j == 0) {
                calculatedBlockInfo[i][j][2] = tmpLength; // SUM_S1S2
            } else {
                calculatedBlockInfo[i][j][2] = calculatedBlockInfo[i][j - 1][2] + tmpLength;
            }
            calculatedBlockInfo[i][j][3] = 0; // SUM_ALL init
        }

        if (i == 0) {
            calculatedBlockInfo[0][0][3] = n2 * g * calculatedBlockInfo[0][fagS2Outer_ - 1][2];
            totalBlockInfo[0][1] = n2 * g * totalBlockInfo[0][0];
        } else {
            calculatedBlockInfo[i][0][3] = n2 * g * calculatedBlockInfo[i][fagS2Outer_ - 1][2] +
                calculatedBlockInfo[i - 1][0][3];
            totalBlockInfo[i][1] = n2 * g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }

    // Block split
    // Source: varlen_regbase.cpp:961-964
    int64_t fusedOuter = calculatedBlockInfo[b - 1][0][3]; // SUM_ALL
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    fagBlockStarts_[0] = 0;
    fagBlockEnds_[blockOuter - 1] = totalBlockInfo[b - 1][1];

    // Source: varlen_regbase.cpp:985-1029
    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        uint64_t tndS1S2PrefixSumTmp = 0;
        uint64_t tndPrefixSumTmp = 0;
        int64_t bIdx = 0, bTail = 0, n2Idx = 0, n2Tail = 0, gIdx = 0, gTail = 0;
        int64_t s1oIdx = 0, s2oIdx = 0;
        int64_t s1OuterTmp = 0;

        for (int64_t bb = 0; bb < b; bb++) {
            if (calculatedBlockInfo[bb][0][3] > currentIdx) {
                bIdx = bb;
                auto s1os2o = calculatedBlockInfo[bb][fagS2Outer_ - 1][2];
                auto gs1os2o = s1os2o * g;
                bTail = (bb == 0) ? currentIdx : currentIdx - calculatedBlockInfo[bb - 1][0][3];
                n2Idx = bTail / gs1os2o;
                n2Tail = bTail % gs1os2o;
                gIdx = n2Tail / s1os2o;
                gTail = n2Tail % s1os2o;

                // GetUnpadS1S2OuterIndex
                // Source: varlen_regbase.cpp:1127-1139
                for (int64_t i = 0; i < fagS2Outer_; i++) {
                    if (calculatedBlockInfo[bb][i][2] > gTail) {
                        s2oIdx = i;
                        int64_t s1oTail = (i == 0) ? gTail : gTail - calculatedBlockInfo[bb][i - 1][2];
                        s1oIdx = calculatedBlockInfo[bb][i][0] + s1oTail;
                        break;
                    }
                }
                s1OuterTmp = (actualSeqlenQ_[bb] + fagS1CvInner_ - 1) / fagS1CvInner_;

                fagTndStartBIdx_[c] = static_cast<int64_t>(bb);
                break;
            } else {
                tndS1S2PrefixSumTmp += static_cast<uint64_t>(actualSeqlenQ_[bb] * actualSeqlenKv_[bb]);
                int64_t s1Ot = (actualSeqlenQ_[bb] + fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT - 1) /
                    (fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT);
                int64_t s2Ot = (actualSeqlenKv_[bb] + fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT - 1) /
                    (fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT);
                tndPrefixSumTmp += static_cast<uint64_t>(s1Ot * s2Ot);
            }
        }

        if (bIdx == 0) {
            fagBlockStarts_[c] = (n2Idx * g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp + s1oIdx;
        } else {
            fagBlockStarts_[c] = totalBlockInfo[bIdx - 1][1] +
                (n2Idx * g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp + s1oIdx;
        }
        fagBlockEnds_[c - 1] = fagBlockStarts_[c];
    }

    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < FAG_CORE_LIST_NUM; c++) {
        fagBlockStarts_[c] = 0;
        fagBlockEnds_[c] = 0;
    }
}

// ===== FagIsValidUnpad: check block validity for TND =====
// Source: varlen_regbase.cpp:1141-1166
bool FlashAttnMetadataCpuKernel::FagIsValidUnpad(int64_t blockIdx)
{
    int64_t resbaseIdx = blockIdx;
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    for (int64_t bIdx = 0; bIdx < b; bIdx++) {
        int64_t actualS1Len = actualSeqlenQ_[bIdx];
        int64_t actualS2Len = actualSeqlenKv_[bIdx];
        int64_t s1OuterTmp = (actualS1Len + fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT - 1) /
            (fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT);
        int64_t s2OuterTmp = (actualS2Len + fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT - 1) /
            (fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT);
        int64_t totalBaseIdx = n2 * g * s1OuterTmp * s2OuterTmp;
        if (resbaseIdx < totalBaseIdx) {
            int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
            int64_t s2oDimIdx = gDimTail / s1OuterTmp;
            int64_t s1oDimIdx = gDimTail % s1OuterTmp;
            int64_t s2IdxLeft = s2oDimIdx * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT;
            int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT, actualS2Len);
            if (fagIsSparse_) {
                return FagCheckUnpadSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, bIdx);
            }
            return true;
        }
        resbaseIdx -= totalBaseIdx;
    }
    return false;
}

// ===== FagCheckUnpadSparseLeftAndRight: check sparse validity for TND =====
// Source: varlen_regbase.cpp:1168-1214
bool FlashAttnMetadataCpuKernel::FagCheckUnpadSparseLeftAndRight(int64_t s1oDimIdx,
    int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx)
{
    int64_t actualS1Len = actualSeqlenQ_[bIdx];
    int64_t actualS2Len = actualSeqlenKv_[bIdx];
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);

    int64_t actualCalcS1Token = fagS1Token_;
    int64_t actualCalcS2Token = fagS2Token_;
    // Token correction for RIGHT_DOWN_CAUSAL / BAND
    if (fagSparseMode_ == FAG_SPARSE_RIGHT_DOWN_CAUSAL || fagSparseMode_ == FAG_SPARSE_BAND) {
        actualCalcS1Token = fagS1Token_ + actualS1Len - actualS2Len;
        actualCalcS2Token = fagS2Token_ - actualS1Len + actualS2Len;
    }

    int64_t s2SparseLeft = std::max(
        fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * s1oDimIdx - actualCalcS1Token, static_cast<int64_t>(0));
    s2SparseLeft = FagAlignTo(s2SparseLeft, FAG_ALIGN64);
    int64_t s2SparseRight = FagAlignTo(
        std::min(fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), s1) + actualCalcS2Token, FAG_ALIGN64);
    s2SparseRight = std::min(s2SparseRight, actualS2Len);
    return s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
}

// ===== FagFillBlockInfoLoadBalance: build load balance table for TND =====
// Source: normal_regbase.cpp:1685-1750
void FlashAttnMetadataCpuKernel::FagFillBlockInfoLoadBalance(
    int64_t (*totalBlockInfo)[2],
    float (*acturalBlockInfo)[FAG_MAX_S2_OUTER + 2])
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;

    acturalBlockInfo[b][0] = 0;
    acturalBlockInfo[b + 1][0] = 0;

    for (int64_t i = 0; i < b; i++) {
        int64_t actualS1Len = actualSeqlenQ_[i];
        int64_t actualS2Len = actualSeqlenKv_[i];
        auto actualS1Outer = (actualS1Len + fagS1CvInner_ - 1) / fagS1CvInner_;
        auto actualS2Outer = (actualS2Len + fagCvS2Inner_ - 1) / fagCvS2Inner_;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        int64_t actualCalcS1Token, actualCalcS2Token;
        FagCalcleActualToken(fagSparseMode_, fagS1Token_, fagS2Token_,
            actualS1Len, actualS2Len, actualCalcS1Token, actualCalcS2Token);

        for (int64_t j = 0; j < fagS2Outer_; j++) {
            if (fagCvS2Inner_ * j >= actualS2Len) {
                acturalBlockInfo[i][j] = 0;
            } else {
                int64_t leftIntersectionPoint = std::max(
                    fagCvS2Inner_ * j - actualCalcS2Token, static_cast<int64_t>(0));
                int64_t cvBlockTail = (fagCvS2Inner_ * (j + 1) > actualS2Len) ?
                    actualS2Len - fagCvS2Inner_ * j : fagCvS2Inner_;
                float acturalS1Begin = static_cast<float>(
                    leftIntersectionPoint > actualS1Len ? actualS1Len : leftIntersectionPoint);
                float acturalS1End = static_cast<float>(std::min(actualS1Len,
                    std::max(fagCvS2Inner_ * j + cvBlockTail + actualCalcS1Token, static_cast<int64_t>(0))));
                float acturalS1Num = acturalS1Begin > acturalS1End ? 0 : acturalS1End - acturalS1Begin;
                float acturalS2Num = static_cast<float>(cvBlockTail);
                acturalBlockInfo[i][j] = acturalS1Num / static_cast<float>(fagS1CvInner_) +
                    acturalS2Num / static_cast<float>(fagCvS2Inner_);
                acturalBlockInfo[b][0] += acturalBlockInfo[i][j] * n2 * g;
                acturalBlockInfo[b + 1][0] = (acturalBlockInfo[b + 1][0] < acturalBlockInfo[i][j]) ?
                    acturalBlockInfo[i][j] : acturalBlockInfo[b + 1][0];
            }
        }
        if (i == 0) {
            totalBlockInfo[0][1] = n2 * g * totalBlockInfo[0][0];
        } else {
            totalBlockInfo[i][1] = n2 * g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }
}

// ===== GenFagMetadata: serialize FAG split results to metadata tensor =====
// FAG data is appended after FA data: offset = 16 + 1728 * sectionNum
void FlashAttnMetadataCpuKernel::GenFagMetadata(uint32_t sectionNum)
{
    auto *basePtr = static_cast<FA_METADATA_T *>(metadata_->GetData());
    // FAG start offset = Head(16) + FA(sectionNum*36*16) + FD(sectionNum*72*16)
    //                  = 16 + sectionNum * (36*16 + 72*16) = 16 + 1728 * sectionNum
    uint32_t fagOffset = HEAD_METADATA_STRIDE +
        sectionNum * (AIC_CORE_NUM * FA_METADATA_STRIDE + AIV_CORE_NUM * FD_METADATA_STRIDE);

    // Record FAG start offset in head[4]
    basePtr[HEAD_FAG_START_OFFSET_INDEX] = fagOffset;

    auto *fag = basePtr + fagOffset;
    // Zero FAG region
    std::fill_n(fag, FAG_METADATA_SIZE, static_cast<FA_METADATA_T>(0));

    // Write header
    fag[FAG_SPLIT_AXIS_INDEX] = fagSplitAxis_;
    fag[FAG_BLOCK_OUTER_INDEX] = fagBlockOuter_;
    fag[FAG_BLOCK_FACTOR_INDEX] = static_cast<int32_t>(fagBlockFactor_);
    fag[FAG_S1_OUTER_INDEX] = static_cast<int32_t>(fagS1Outer_);
    fag[FAG_S2_OUTER_INDEX] = static_cast<int32_t>(fagS2Outer_);
    fag[FAG_LAYOUT_TYPE_INDEX] = fagLayoutType_;
    fag[FAG_IS_SPARSE_INDEX] = fagIsSparse_ ? 1 : 0;
    fag[FAG_SPARSE_MODE_INDEX] = fagSparseMode_;

    // Write block arrays
    for (uint32_t i = 0; i < FAG_CORE_LIST_NUM; i++) {
        fag[FAG_BLOCK_STARTS_OFFSET + i] = static_cast<int32_t>(fagBlockStarts_[i]);
        fag[FAG_BLOCK_ENDS_OFFSET + i] = static_cast<int32_t>(fagBlockEnds_[i]);
        fag[FAG_TND_START_BIDX_OFFSET + i] = static_cast<int32_t>(fagTndStartBIdx_[i]);
    }
}

// ===== FAG metadata end =====

} // namespace aicpu

#endif // FLASH_ATTN_GRAD_METADATA_SPLIT_H
