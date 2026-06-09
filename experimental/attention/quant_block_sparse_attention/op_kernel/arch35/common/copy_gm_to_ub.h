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
template <typename T, BSALayout GM_FORMAT>
class CopyAntiquantGmToUb {
public:
    __aicore__ inline void operator()(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        //per per_head + per_token + PA
        ProcessPA(dstTensor, srcTensor, blockTableTensor, runInfo, constInfo);
    }

private:
    __aicore__ inline void ProcessPA(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        int32_t byteOffset1 = 0;
        int32_t byteOffset2 = 0;
        GetPaKScaleOffset(byteOffset1, runInfo.sparseBlkIdx1, blockTableTensor, runInfo, constInfo);
        GetPaKScaleOffset(byteOffset2, runInfo.sparseBlkIdx2, blockTableTensor, runInfo, constInfo);
        if (GM_FORMAT == BSALayout::PA_BNSD) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 128;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = constInfo.dSize + constInfo.dSizeV;
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams;
            DataCopyPad(dstTensor, srcTensor[byteOffset1], dataCopyParams, dataCopyPadParams);
            DataCopyPad(dstTensor[128], srcTensor[byteOffset2], dataCopyParams, dataCopyPadParams);
        } else if (GM_FORMAT == BSALayout::PA_BSND) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 128;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = ((constInfo.n2Size - 1) * constInfo.combineDim) * sizeof(float);
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams;
            DataCopyPad(dstTensor, srcTensor[byteOffset1], dataCopyParams, dataCopyPadParams);
            DataCopyPad(dstTensor[128], srcTensor[byteOffset2], dataCopyParams, dataCopyPadParams);
        }
    }
    __aicore__ inline void GetPaKScaleOffset(int32_t &byteOffset, const int64_t sparseBlkIdx, const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        int32_t physicalBlock = blockTableTensor.GetValue(runInfo.boIdx * constInfo.blockTableDim2 + sparseBlkIdx);
        byteOffset = physicalBlock * constInfo.paBlockStride + runInfo.n2oIdx * constInfo.blockSize * constInfo.combineDim + constInfo.dSize + constInfo.dSizeV;
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
            dataCopyParams.srcStride = (constInfo.n2Size * runInfo.goIdx - 1) * sizeof(float);
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams;
            DataCopyPad(dstTensor, srcTensor[runInfo.qBScalarOffset + runInfo.s1oIdx * constInfo.n2Size * runInfo.goIdx + runInfo.n2oIdx * runInfo.goIdx], dataCopyParams, dataCopyPadParams);
        } else if (GM_FORMAT == BSALayout::BSND) {

        } else if (GM_FORMAT == BSALayout::BNSD) {

        }
        
    }
};
#endif