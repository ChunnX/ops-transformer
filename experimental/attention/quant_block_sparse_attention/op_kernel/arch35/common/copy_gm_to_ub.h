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
 * \file copy_gm_to_ub.h
 * \brief
 */
#ifndef COPY_GM_TO_UB_H
#define COPY_GM_TO_UB_H

#include "offset_calculator.h"

// GM->UB
//antiquant
// ----------------------------------------------CopyAntiquantGmToUb--------------------------------
struct AntiqGmCoord {
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t firstS2Idx = 0;
    uint32_t nextS2Idx = 0;

    uint32_t firstS2DealSize = 0;
    uint32_t nextS2DealSize = 0;
};

template <typename OUT_T>
struct FaUbTensor {
    LocalTensor<OUT_T> tensor;
    uint32_t rowCount;
    uint32_t colCount;
};

template <typename T, GmFormat GM_FORMAT>
class CopyAntiquantGmToUb {
public:
    __aicore__ inline void operator()(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor,
                                      AntiqGmCoord &antiqGmCoord, ConstInfo &constInfo)
    {
        ProcessAntiqPA(dstTensor, srcTensor, antiqGmCoord, constInfo);
    }

private:
    __aicore__ inline void ProcessAntiqPA(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor,
                                          AntiqGmCoord &antiqGmCoord, ConstInfo &constInfo)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;

        uint64_t dstOffset = 0;
        uint32_t curS2Idx = antiqGmCoord.firstS2Idx;
        uint64_t srcOffset = offsetCalculator.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, curS2Idx, constInfo.dSize);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = antiqGmCoord.firstS2DealSize;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<T> dataCopyPadParams;
        DataCopyPad(dstTensor.tensor[dstOffset], srcTensor.gmTensor[srcOffset / sizeof(float)], dataCopyParams, dataCopyPadParams);

        curS2Idx = antiqGmCoord.nextS2Idx;
        srcOffset = offsetCalculator.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, curS2Idx, constInfo.dSize);
        dataCopyParams.blockCount = antiqGmCoord.nextS2DealSize;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(dstTensor.tensor[dstOffset + antiqGmCoord.firstS2DealSize], srcTensor.gmTensor[srcOffset / sizeof(float)], dataCopyParams, dataCopyPadParams);
    }
};

// ----------------------------------------------CopyQueryScaleGmToUb--------------------------------
template <typename T, BSALayout GM_FORMAT>
class  CopyQueryScaleGmToUb
{
public:
    __aicore__ inline void operator()(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        ProcessGS1(dstTensor, srcTensor, runInfo, constInfo);
    }
private:
    __aicore__ inline void ProcessGS1(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        if (GM_FORMAT == BSALayout::TND) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 64;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = (constInfo.n2Size * constInfo.gSize - 1) * sizeof(float);
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams;
            DataCopyPad(dstTensor, srcTensor[runInfo.qBScalarOffset], dataCopyParams, dataCopyPadParams);
        } else if (GM_FORMAT == BSALayout::BSND) {

        } else if (GM_FORMAT == BSALayout::BNSD) {

        }
    }
};
#endif
