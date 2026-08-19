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
// ----------------------------------------------CopyAttenOutUbToGm--------------------------------
template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm {
public:
    __aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor, const LocalTensor<OUT_T> ubTensor,
                                          uint32_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                          uint64_t dstStride)
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
    template <typename FaGmTensorType, bool ROW_PAD = true>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor, FaUbTensor<OUT_T, ROW_PAD> &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr (UB_FORMAT == UbFormat::GS1) {
            auto &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1Size = 0;
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
            uint32_t srcStride =
                (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
            uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                           dstStride);

            if (gIdxEnd - gIdxStart >= 1) {
                // 处理中间块
                gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideG();
                ubOffset = headS1 * srcTensor.colCount;
                for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                    blockCount = s1Size;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                   srcStride, dstStride);
                    gmOffset += offsetCalculator.GetStrideG();
                    ubOffset += s1Size * srcTensor.colCount;
                }

                // 处理尾块
                if (s1IdxEnd > 0) {
                    blockCount = s1IdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                   srcStride, dstStride);
                }
            }
        } else if constexpr (UB_FORMAT == UbFormat::S1G) {
            auto &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
            uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
            uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
            uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

            uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, 0);
            uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
            uint32_t srcStride =
                (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
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

            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
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
                        uint64_t dstStrideTmp =
                            offsetCalculator.GetStrideS1() * sizeof(OUT_T) - blockLenTmp; // 单位为Byte

                        // blockCount = offsetCalculator.GetDimG() * (s1IdxEnd - s1IdxStart -1);
                        SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCountTmp,
                                       blockLenTmp, srcStrideTmp, dstStrideTmp);
                        gmOffset += offsetCalculator.GetStrideS1() * blockCountTmp;
                        ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount * blockCountTmp;
                    }
                } else {
                    for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                        blockCount = offsetCalculator.GetDimG();
                        SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                       srcStride, dstStride);
                        gmOffset += offsetCalculator.GetStrideS1();
                        ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount;
                    }
                }
                // 处理尾块
                if (gIdxEnd > 0) {
                    blockCount = gIdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                   srcStride, dstStride);
                }
            }
        }
    }
};

// ----------------------------------------------Copy LSE UB To Gm --------------------------------
enum class SoftmaxLseFormat {
    BSND, // [B, N2, S1, G]: bN2Offset 起点后, 以 (gIdx * s1Size + s1Idx) 步进, dstStride=(s1Size-1)*sizeof(float)
    BNSD, // [B, N2, S1, 1]: 以 qActSeqLens 为单元连续拷贝, 越过 g 边界时按 s1Size 跳步
    TND,  // [T=N2*G*S1, 1]: 以 (s1Idx * n2Size * gSize + gIdx) 步进, dstStride=0
    TNDtoNT, // TND 输入 -> N-major 输出 [N2*G, T]: 以 (gIdx * t1Size + s1Idx) 步进, dstStride=(t1Size-1)*sizeof(float)
    NTD, // [B*N2*G, S1, 1]: 以 (s1Idx * n2Size * gSize + gIdx) 步进, dstStride=(gSize*n2Size-1)*sizeof(float)
};

struct SoftmaxLseDims {
    uint64_t s1Size = 0;            // BSND / BNSD / NTD: S1 维度大小
    uint64_t gSize = 0;             // BSND / TND / TNDtoNT / NTD: G 维度大小
    uint64_t n2Size = 0;            // TND / NTD: N2 维度大小
    uint64_t t1Size = 0;            // TNDtoNT: 每条 N2*G 行的 T 总长度
    uint64_t qActSeqLens = 0;       // BNSD: 当前 batch 实际 S1 长度
    uint64_t s1LeftPaddingSize = 0; // BSND / BNSD: S1 左 padding
};

// 统一对外接口: 依据 SoftmaxLseFormat 模板参数在内部按 layout 分派拷贝逻辑
// 参考 CopyAttenOutUbToGm 的模板参数 + if constexpr 分派方式
template <typename T, SoftmaxLseFormat FORMAT>
__aicore__ inline void DataCopySoftmaxLse(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc, uint64_t bN2Offset,
                                          uint32_t mOffset, uint32_t dealCount, const SoftmaxLseDims &dims)
{
    if constexpr (FORMAT == SoftmaxLseFormat::BSND) {
        // [B, N2, S1, G]: gIdx 外层, s1Idx 内层
        uint64_t s1Size = dims.s1Size;
        uint64_t gSize = dims.gSize;
        uint64_t s1LeftPaddingSize = dims.s1LeftPaddingSize;
        uint32_t startS1Idx = mOffset / gSize;
        uint32_t startGIdx = mOffset % gSize;
        uint32_t endS1Idx = (mOffset + dealCount - 1) / gSize;
        uint32_t endGIdx = (mOffset + dealCount - 1) % gSize;
        uint64_t outOffset = 0;
        uint64_t ubOffset = 0;
        uint32_t curDealRowCount = 0;

        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            outOffset = bN2Offset + startGIdx * s1Size + s1Idx + s1LeftPaddingSize;
            if (s1Idx != endS1Idx) {
                curDealRowCount = gSize - startGIdx;
            } else {
                curDealRowCount = endGIdx + 1 - startGIdx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (s1Size - 1) * sizeof(float);
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    } else if constexpr (FORMAT == SoftmaxLseFormat::BNSD) {
        // [B, N2, S1, 1]: 以 qActSeqLens 为单元连续拷贝, 越过 g 边界时按 s1Size 跳步
        uint64_t qActSeqLens = dims.qActSeqLens;
        uint64_t s1Size = dims.s1Size;
        uint64_t s1LeftPaddingSize = dims.s1LeftPaddingSize;
        uint64_t gOffset = mOffset / qActSeqLens * s1Size;
        uint64_t seqOffset = mOffset % qActSeqLens;

        uint64_t outOffset = bN2Offset + gOffset + seqOffset + s1LeftPaddingSize;
        uint64_t ubOffset = 0;
        // dealCount ≤ 当前actQs剩余部分，则直接搬运全部dealCount
        if ((qActSeqLens - seqOffset) >= dealCount) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = dealCount;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
            return;
        }
        // dealCount > 当前actQs剩余部分，分块搬运dealCount
        // dealCount首块
        uint64_t headActSeq = qActSeqLens - seqOffset;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = headActSeq;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        outOffset += s1Size - qActSeqLens + headActSeq;
        ubOffset += headActSeq * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        // dealCount中间块
        uint64_t pendingCount = dealCount - headActSeq;
        while (pendingCount > qActSeqLens) {
            DataCopyExtParams dataCopyParamsMid;
            dataCopyParamsMid.blockCount = qActSeqLens;
            dataCopyParamsMid.blockLen = sizeof(float);
            dataCopyParamsMid.srcStride = 0;
            dataCopyParamsMid.dstStride = 0;
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParamsMid);
            outOffset += s1Size;
            ubOffset += qActSeqLens * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
            pendingCount -= qActSeqLens;
        }
        // dealCount尾块
        if (pendingCount > 0) {
            DataCopyExtParams dataCopyParamsTail;
            dataCopyParamsTail.blockCount = pendingCount;
            dataCopyParamsTail.blockLen = sizeof(float);
            dataCopyParamsTail.srcStride = 0;
            dataCopyParamsTail.dstStride = 0;
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParamsTail);
        }
    } else if constexpr (FORMAT == SoftmaxLseFormat::TND) {
        // [T=N2*G*S1, 1]: s1Idx 外层, gIdx 内层, dstStride=0
        uint64_t gSize = dims.gSize;
        uint64_t n2Size = dims.n2Size;
        uint32_t startS1Idx = mOffset / gSize;
        uint32_t startGIdx = mOffset % gSize;
        uint32_t endS1Idx = (mOffset + dealCount - 1) / gSize;
        uint32_t endGIdx = (mOffset + dealCount - 1) % gSize;
        uint64_t outOffset = 0;
        uint64_t ubOffset = 0;
        uint32_t curDealRowCount = 0;

        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            outOffset = bN2Offset + s1Idx * n2Size * gSize + startGIdx;
            if (s1Idx != endS1Idx) {
                curDealRowCount = gSize - startGIdx;
            } else {
                curDealRowCount = endGIdx + 1 - startGIdx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    } else if constexpr (FORMAT == SoftmaxLseFormat::TNDtoNT) {
        // TND 输入 -> N-major 输出 [N2*G, T]: gIdx 外层, s1Idx 内层, dstStride=(t1Size-1)*sizeof(float)
        uint64_t gSize = dims.gSize;
        uint64_t t1Size = dims.t1Size;
        uint32_t startS1Idx = mOffset / gSize;
        uint32_t startGIdx = mOffset % gSize;
        uint32_t endS1Idx = (mOffset + dealCount - 1) / gSize;
        uint32_t endGIdx = (mOffset + dealCount - 1) % gSize;
        uint64_t outOffset = 0;
        uint64_t ubOffset = 0;
        uint32_t curDealRowCount = 0;

        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            outOffset = bN2Offset + startGIdx * t1Size + s1Idx;
            if (s1Idx != endS1Idx) {
                curDealRowCount = gSize - startGIdx;
            } else {
                curDealRowCount = endGIdx + 1 - startGIdx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (t1Size - 1) * sizeof(float);
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    } else if constexpr (FORMAT == SoftmaxLseFormat::NTD) {
        // [B*N2*G, S1, 1]: s1Idx 内层, gIdx 外层, dstStride=(gSize*n2Size-1)*sizeof(float)
        uint64_t s1Size = dims.s1Size;
        uint64_t gSize = dims.gSize;
        uint64_t n2Size = dims.n2Size;
        uint32_t startS1Idx = mOffset % s1Size;
        uint32_t startGIdx = mOffset / s1Size;
        uint32_t endS1Idx = (mOffset + dealCount - 1) % s1Size;
        uint32_t endGIdx = (mOffset + dealCount - 1) / s1Size;
        uint64_t outOffset = 0;
        uint64_t ubOffset = 0;
        uint32_t curDealRowCount = 0;

        for (uint32_t gIdx = startGIdx; gIdx <= endGIdx; gIdx++) {
            outOffset = bN2Offset + startS1Idx * n2Size * gSize + gIdx;
            if (gIdx != endGIdx) {
                curDealRowCount = s1Size - startS1Idx;
            } else {
                curDealRowCount = endS1Idx + 1 - startS1Idx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (gSize * n2Size - 1) * sizeof(float);
            DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
            startS1Idx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }
}

#endif
