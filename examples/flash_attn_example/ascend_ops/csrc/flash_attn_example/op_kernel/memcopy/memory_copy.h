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
 * \file memory_copy.h
   GM->L1
   PA
   PARope
 * \brief
 */
#ifndef MEMORY_COPY_H
#define MEMORY_COPY_H
#include "../fa_kernel_public.h"
#include "offset_calculator.h"

// ---------------------------------------Set attention Gm To Zero--------------------------------
template <typename OUT_T>
__aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, const BsngdOffsetInfo &info,
                                           GlobalTensor<OUT_T>& attentionOutGm)
{
    uint32_t s1Size = info.dimS1;
    for (int s1Idx = 0; s1Idx < s1Size; s1Idx++) {
        uint64_t attnOutOffset = GetOffset(info, bIdx, n2Idx, 0, s1Idx, 0);
        matmul::InitOutput<OUT_T>(attentionOutGm[attnOutOffset], info.strideN2, 0);
    }
}

// ----------------------------------------------CopyQueryGmToL1--------------------------------
template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor, uint32_t nValue,
                                              uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    constexpr uint32_t SRCDVALUE_LIMIT = 65536;
    Nd2NzParams nd2nzPara;
    if (unlikely(srcDValue < SRCDVALUE_LIMIT)) {
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = nValue; // nd矩阵的行数
        nd2nzPara.dValue = dValue;       // nd矩阵的列数
        nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    } else {
        constexpr uint32_t BLOCK_ELEMENT_CNT = fa_kernel::hardware::BYTE_BLOCK / sizeof(T);
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < nValue; i++) {
            nd2nzPara.ndNum = 1;
            nd2nzPara.nValue = 1;
            nd2nzPara.dValue = dValue;
            nd2nzPara.srcDValue = dValue;
            nd2nzPara.dstNzC0Stride = dstNzC0Stride;
            nd2nzPara.dstNzNStride = 1;
            nd2nzPara.srcNdMatrixStride = 0;
            nd2nzPara.dstNzMatrixStride = 0;
            DataCopy(l1Tensor[l1Offset], gmTensor[gmOffset], nd2nzPara);

            gmOffset += srcDValue;
            l1Offset += BLOCK_ELEMENT_CNT;
        }
    }
}
template <typename T>
__aicore__ inline void CopyMultiMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor,
                                             uint32_t srcNdMatrixNum, uint32_t srcNdMatrixStride,
                                             uint32_t dstNzMatrixStride, uint32_t nValue, uint32_t dValue,
                                             uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    constexpr uint32_t ND_MATRIX_STRIDE_LIMIT = 65536; // Mutil ND2NZ搬运时，Nd2NzParams支持的srcNdMatrixStride的取值范围为[0, 65536]，单位为元素
    if (unlikely(srcNdMatrixStride > ND_MATRIX_STRIDE_LIMIT)) {
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < srcNdMatrixNum; i++) {
            CopySingleMatrixNDToNZ(l1Tensor[l1Offset], gmTensor[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
            gmOffset += srcNdMatrixStride;
            l1Offset += dstNzMatrixStride;
        }
    } else {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = srcNdMatrixNum;
        nd2nzPara.nValue = nValue; // nd矩阵的行数
        nd2nzPara.dValue = dValue;       // nd矩阵的列数
        nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzPara.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    }
}

template <typename Q_T, GmFormat GM_FORMAT>
__aicore__ inline void CopyQueryGmToL1(FaL1Tensor<Q_T> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
{
    auto &info = srcTensor.offsetInfo;
    uint32_t s1IdxStart = gmCoord.gS1Idx / info.dimG;
    uint32_t gIdxStart = gmCoord.gS1Idx % info.dimG;
    uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / info.dimG;
    uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % info.dimG;

    uint64_t queryGmbaseOffset =
        GetOffset(info, gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, gmCoord.dIdx);


    if (info.dimG == 1) {
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[queryGmbaseOffset], s1IdxEnd - s1IdxStart,
                               gmCoord.dDealSize, info.strideS1, dstTensor.rowCount);
        return;
    }

    // 处理第一个S
    uint32_t headSize = 0;
    if (s1IdxStart == s1IdxEnd) {
        headSize = gIdxEnd - gIdxStart;
    } else {
        headSize = info.dimG - gIdxStart;
    }

    uint64_t offset = queryGmbaseOffset + gIdxStart * info.dimD;
    CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], headSize, gmCoord.dDealSize,
                            info.strideG, dstTensor.rowCount);

    if (s1IdxEnd - s1IdxStart >= 1) {
        // 处理中间块
        uint64_t gmOffset = queryGmbaseOffset + info.strideS1;
        uint32_t blockElementCnt = 32 / sizeof(Q_T);
        uint64_t l1Offset = headSize * blockElementCnt;
        if (s1IdxEnd - s1IdxStart > 1) {
            CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset],
                                  s1IdxEnd - s1IdxStart - 1, info.strideS1,
                                  info.dimG * blockElementCnt, info.dimG,
                                  gmCoord.dDealSize, info.strideG, dstTensor.rowCount);
            gmOffset += (s1IdxEnd - s1IdxStart - 1) * info.strideS1;
            l1Offset += (s1IdxEnd - s1IdxStart - 1) * info.dimG * blockElementCnt;
        }

        // 处理尾块
        if (gIdxEnd > 0) {
            CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], gIdxEnd,
                                    gmCoord.dDealSize, info.strideG, dstTensor.rowCount);
        }
    }
}

// ----------------------------------------------CopyKvGmToL1--------------------------------
struct GmKvCoord {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
    uint32_t s2DealSize;
    uint32_t dDealSize;
};

template <typename KV_T, GmFormat GM_FORMAT>
__aicore__ inline void CopyKvGmToL1(FaL1Tensor<KV_T> &dstTensor, FaGmTensor<KV_T, GM_FORMAT> &srcTensor, GmKvCoord &gmCoord)
{
  // B*N2*GS1*D
        auto &info = srcTensor.offsetInfo;
        uint64_t offset = GetOffset(info, gmCoord.bIdx, gmCoord.n2Idx, gmCoord.s2Idx, gmCoord.dIdx, 0);
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], gmCoord.s2DealSize, gmCoord.dDealSize,
                               info.strideS2, dstTensor.rowCount);
}

// ----------------------------------------------CopyAttnOutUbToGm--------------------------------
template <typename OUT_T>
__aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor, const LocalTensor<OUT_T> ubTensor,
                                      uint32_t blockCount, uint32_t blockLen, uint32_t srcStride, uint64_t dstStride)
{
    DataCopyExtParams dataCopyParams;
    // B*S过大时，跳写参数dataCopyParams.dstStride(uint32_t)计算结果将溢出，使用for循环拷贝代替
    if (dstStride > UINT32_MAX) {
        uint64_t gmSingleStride = (dstStride + blockLen) / sizeof(OUT_T);
        uint64_t ubSingleStride = (srcStride * fa_kernel::hardware::BYTE_BLOCK + blockLen) / sizeof(OUT_T);
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

template <typename OUT_T, GmFormat GM_FORMAT>
__aicore__ inline void CopyAttnOutUbToGm(FaGmTensor<OUT_T, GM_FORMAT> &dstTensor, FaUbTensor<OUT_T> &srcTensor,
                                          GmCoord &gmCoord)
{
    auto &info = dstTensor.offsetInfo;
    uint32_t s1IdxStart = gmCoord.gS1Idx / info.dimG;
    uint32_t gIdxStart = gmCoord.gS1Idx % info.dimG;
    uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / info.dimG;
    uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % info.dimG;

    uint64_t attnOutGmbaseOffset = GetOffset(info, gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, 0);

    // 处理第一个S
    uint32_t headSize = 0;
    if (s1IdxStart == s1IdxEnd) {
        headSize = gIdxEnd - gIdxStart;
    } else {
        headSize = info.dimG - gIdxStart;
    }
    uint64_t gmOffset = attnOutGmbaseOffset + gIdxStart * info.strideG;
    uint64_t ubOffset = 0;
    uint32_t blockCount = headSize;
    uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
    uint32_t srcStride =
        (srcTensor.colCount - gmCoord.dDealSize) / (fa_kernel::hardware::BYTE_BLOCK / sizeof(OUT_T));
    uint64_t dstStride = (info.strideG - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                    dstStride);

    if (s1IdxEnd - s1IdxStart >= 1) {
        // 处理中间块
        gmOffset = attnOutGmbaseOffset + info.strideS1;
        ubOffset = ((uint64_t)headSize) * ((uint64_t)srcTensor.colCount);
        for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
            blockCount = info.dimG;
            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                            srcStride, dstStride);
            gmOffset += info.strideS1;
            ubOffset += info.dimG * srcTensor.colCount;
        }

        // 处理尾块
        if (gIdxEnd > 0) {
            blockCount = gIdxEnd;
            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                            srcStride, dstStride);
        }
    }
}
#endif