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
 * \file copy_ub_to_gm.h
 * \brief
 */
#ifndef COPY_UB_TO_GM_H
#define COPY_UB_TO_GM_H

#include "gm_layout.h"
#include "gm_coord.h"
#include "fa_ub_tensor.h"
#include "offset_calculator_v2.h"
#include "../const_def.h"
// ----------------------------------------------SafeStrideCopy--------------------------------
template <typename OUT_T>
__aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor, const LocalTensor<OUT_T> ubTensor,
                                      uint32_t blockCount, uint32_t blockLen, uint32_t srcStride, uint64_t dstStride)
{
    DataCopyExtParams dataCopyParams;
    // B*S过大时，跳写参数dataCopyParams.dstStride(uint32_t)计算结果将溢出，使用for循环拷贝代替
    if (dstStride > UINT32_MAX) {
        uint64_t gmSingleStride = (dstStride + blockLen) / sizeof(OUT_T);
        uint64_t ubSingleStride = (srcStride * AttentionCommon::BYTE_BLOCK + blockLen) / sizeof(OUT_T);
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0; // 单位为Byte
        for (uint32_t i = 0; i < blockCount; i++) {
            DataCopyPad(gmTensor[i * gmSingleStride], ubTensor[i * ubSingleStride], dataCopyParams);
        }
    } else {
        // dataCopyParams.dstStride(uint32_t)没有溢出时，进行跳写
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = srcStride;
        dataCopyParams.dstStride = dstStride; // 单位为Byte
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }
}

// ----------------------------------------------CopyAttenOutUbToGm--------------------------------
template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm {};

template <typename OUT_T, GmFormat GM_FORMAT>
class CopyAttenOutUbToGm<OUT_T, GM_FORMAT, UbFormat::GS1> {
public:
    template <typename FaGmTensorType, bool ROW_PAD = true>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor, FaUbTensor<OUT_T, ROW_PAD> &srcTensor,
                                      GmCoordGs1Merge &gmCoord)
    {
        uint32_t s1Size = 0;
        auto &offsetCalculator = dstTensor.offsetCalculator;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                s1Size = offsetCalculator.GetDimS1();
            }
        }
        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

        uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, 0);

        // 处理第一个S
        uint32_t headS1 = 0;
        if (gIdxStart == gIdxEnd) {
            headS1 = s1IdxEnd - s1IdxStart;
        } else {
            headS1 = s1Size - s1IdxStart;
        }
        uint64_t gmOffset = attenOutGmbaseOffset + s1IdxStart * offsetCalculator.GetStrideS1();
        uint64_t ubOffset = 0;
        uint32_t blockCount = headS1;
        uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
        uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
        uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
        SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                              dstStride);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideG();
            ubOffset = headS1 * srcTensor.colCount;
            for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                blockCount = s1Size;
                SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                      srcStride, dstStride);
                gmOffset += offsetCalculator.GetStrideG();
                ubOffset += s1Size * srcTensor.colCount;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                blockCount = s1IdxEnd;
                SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                      srcStride, dstStride);
            }
        }
    }
};

template <typename OUT_T, GmFormat GM_FORMAT>
class CopyAttenOutUbToGm<OUT_T, GM_FORMAT, UbFormat::S1G> {
public:
    template <typename FaGmTensorType, bool ROW_PAD = true>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor, FaUbTensor<OUT_T, ROW_PAD> &srcTensor,
                                      GmCoordGs1Merge &gmCoord)
    {
        auto &offsetCalculator = dstTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, 0);
        uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
        uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
        uint64_t dstStride = (offsetCalculator.GetStrideG() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte

        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }
        uint64_t gmOffset = attenOutGmbaseOffset + gIdxStart * offsetCalculator.GetStrideG();
        uint64_t ubOffset = 0;
        uint32_t blockCount = headSize;

        SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                              dstStride);

        if (s1IdxEnd - s1IdxStart >= 1) {
            // 处理中间块
            gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideS1();
            ubOffset = ((uint64_t)headSize) * ((uint64_t)srcTensor.colCount);
            if constexpr ((!ROW_PAD) && ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD))) {
                if (s1IdxEnd > 1 + s1IdxStart) {
                    uint32_t blockCountTmp = s1IdxEnd - s1IdxStart - 1;
                    uint32_t blockLenTmp = gmCoord.dDealSize * sizeof(OUT_T) * offsetCalculator.GetDimG();
                    uint32_t srcStrideTmp = 0;
                    uint64_t dstStrideTmp = offsetCalculator.GetStrideS1() * sizeof(OUT_T) - blockLenTmp; // 单位为Byte

                    // blockCount = offsetCalculator.GetDimG() * (s1IdxEnd - s1IdxStart -1);
                    SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCountTmp,
                                          blockLenTmp, srcStrideTmp, dstStrideTmp);
                    gmOffset += offsetCalculator.GetStrideS1() * blockCountTmp;
                    ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount * blockCountTmp;
                }
            } else {
                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    blockCount = offsetCalculator.GetDimG();
                    SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount,
                                          blockLen, srcStride, dstStride);
                    gmOffset += offsetCalculator.GetStrideS1();
                    ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount;
                }
            }
            // 处理尾块
            if (gIdxEnd > 0) {
                blockCount = gIdxEnd;
                SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                      srcStride, dstStride);
            }
        }
    }
};

template <typename OUT_T, GmFormat GM_FORMAT>
class CopyAttenOutUbToGm<OUT_T, GM_FORMAT, UbFormat::S1_ONLY> {
public:
    template <typename FaGmTensorType, bool ROW_PAD = true>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor, FaUbTensor<OUT_T, ROW_PAD> &srcTensor,
                                      GmCoordS1Only &gmCoord)
    {
        auto &offsetCalculator = dstTensor.offsetCalculator;
        uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.gIdx, gmCoord.s1Idx, 0);
        uint32_t blockCount = gmCoord.s1DealSize;
        uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
        uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
        uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
        SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[0], blockCount, blockLen, srcStride,
                              dstStride);
    }
};

// ----------------------------------------------Copy LSE UB To GM--------------------------------
template <UbFormat UB_FORMAT>
class CopyLseUbToGm {};

// Each UB row occupies one 32-byte block; only its first float is written to GM.
// All offsets/strides below are in float elements. baseOffset includes the batch,
// N2 head group, TND sequence prefix and any left padding supplied by the caller.
// Callers own synchronization and must keep each DMA's count/stride within DataCopyExtParams limits.

// UB rows are ordered as s1 * gSize + g; GM stores each head's sequence contiguously.
// gSize and gmHeadStride must be positive for a nonempty copy.
template <>
class CopyLseUbToGm<UbFormat::S1G> {
public:
    __aicore__ inline void operator()(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
                                      uint64_t baseOffset, uint32_t rowStart, uint32_t rowCount, uint32_t gSize,
                                      uint64_t gmHeadStride)
    {
        if (rowCount == 0) {
            return;
        }
        uint32_t startS1 = rowStart / gSize;
        uint32_t startG = rowStart % gSize;
        uint32_t endS1 = (rowStart + rowCount - 1) / gSize;
        uint32_t endG = (rowStart + rowCount - 1) % gSize;
        uint64_t ubOffset = 0;
        AscendC::DataCopyExtParams params;
        params.blockLen = sizeof(float);
        params.srcStride = 0;
        params.dstStride = (gmHeadStride - 1) * sizeof(float);
        for (uint32_t s1 = startS1; s1 <= endS1; ++s1) {
            uint32_t rows = (s1 == endS1) ? endG + 1 - startG : gSize - startG;
            params.blockCount = rows;
            AscendC::DataCopyPad(dst[baseOffset + startG * gmHeadStride + s1], src[ubOffset], params);
            startG = 0;
            ubOffset += rows * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }
};

// UB rows are ordered as g * actualSeqLen + s1. Split at actual sequence boundaries,
// keeping GM's head stride independent of actualSeqLen (e.g. BNSD with unused sequence capacity).
// actualSeqLen and gmSeqStride must be positive for a nonempty copy.
template <>
class CopyLseUbToGm<UbFormat::GS1> {
public:
    __aicore__ inline void operator()(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
                                      uint64_t baseOffset, uint32_t rowStart, uint32_t rowCount, uint64_t actualSeqLen,
                                      uint64_t gmHeadStride, uint64_t gmSeqStride = 1)
    {
        if (rowCount == 0) {
            return;
        }
        uint64_t headOffset = baseOffset + rowStart / actualSeqLen * gmHeadStride;
        uint64_t seqOffset = rowStart % actualSeqLen;
        uint64_t ubOffset = 0;
        uint64_t pendingRows = rowCount;
        AscendC::DataCopyExtParams params;
        params.blockLen = sizeof(float);
        params.srcStride = 0;
        params.dstStride = (gmSeqStride - 1) * sizeof(float);
        while (pendingRows > 0) {
            uint64_t rows = AttentionCommon::Min(pendingRows, actualSeqLen - seqOffset);
            params.blockCount = rows;
            AscendC::DataCopyPad(dst[headOffset + seqOffset * gmSeqStride], src[ubOffset], params);
            pendingRows -= rows;
            ubOffset += rows * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
            headOffset += gmHeadStride;
            seqOffset = 0;
        }
    }
};

// One head, consecutive sequence positions in GM. No G/S1 boundary splitting is needed.
template <>
class CopyLseUbToGm<UbFormat::S1_ONLY> {
public:
    __aicore__ inline void operator()(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
                                      uint64_t outputOffset, uint32_t rowCount)
    {
        if (rowCount == 0) {
            return;
        }
        AscendC::DataCopyExtParams params;
        params.blockCount = rowCount;
        params.blockLen = sizeof(float);
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopyPad(dst[outputOffset], src, params);
    }
};

#endif
