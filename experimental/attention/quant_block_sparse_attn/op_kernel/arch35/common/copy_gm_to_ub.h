/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
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
// antiquant
// ----------------------------------------------CopyAntiquantGmToUb--------------------------------
template <typename T, BSALayout GM_FORMAT>
class CopyAntiquantGmToUb {
public:
    __aicore__ inline void operator()(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                      const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo,
                                      const ConstInfo &constInfo)
    {
        // per per_head + per_token + PA
        ProcessPA(dstTensor, srcTensor, blockTableTensor, runInfo, constInfo);
    }

private:
    static constexpr uint32_t BSA_BYTE_BLOCK = 32U;
    static constexpr uint32_t BSA_KSCALE_BANK_COPY_NUM = 4U;

    __aicore__ inline uint32_t GetKScaleUbStride(const ConstInfo &constInfo) const
    {
        return static_cast<uint32_t>(constInfo.s2BaseSize) + BSA_BYTE_BLOCK / sizeof(T);
    }

    __aicore__ inline void CopyKScaleBnsdToBankConflictFreeUb(LocalTensor<T> &dstTensor,
                                                              const GlobalTensor<T> &srcTensor, int64_t srcOffset,
                                                              uint32_t dstOffset, uint32_t copyLen,
                                                              const ConstInfo &constInfo)
    {
        if (copyLen == 0U) {
            return;
        }
        const uint32_t blockElemNum = BSA_BYTE_BLOCK / sizeof(T);
        const uint32_t kScaleUbStride = GetKScaleUbStride(constInfo);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(BSA_KSCALE_BANK_COPY_NUM);
        dataCopyParams.blockLen = copyLen * sizeof(T);
        dataCopyParams.srcStride = -static_cast<int64_t>(dataCopyParams.blockLen);
        dataCopyParams.dstStride = (kScaleUbStride + blockElemNum - copyLen) * sizeof(T) / BSA_BYTE_BLOCK;
        DataCopyPadExtParams<T> dataCopyPadParams{};
        DataCopyPad(dstTensor[dstOffset], srcTensor[srcOffset], dataCopyParams, dataCopyPadParams);
    }

    __aicore__ inline void CopyKScaleBsndToBankConflictFreeUb(LocalTensor<T> &dstTensor,
                                                              const GlobalTensor<T> &srcTensor, int64_t srcOffset,
                                                              uint32_t dstOffset, uint32_t copyLen,
                                                              const ConstInfo &constInfo)
    {
        if (copyLen == 0U) {
            return;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(copyLen);
        dataCopyParams.blockLen = sizeof(T);
        dataCopyParams.srcStride = ((constInfo.n2Size - 1) * constInfo.combineDim) * sizeof(T);
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<T> dataCopyPadParams{};
        const uint32_t kScaleUbStride = GetKScaleUbStride(constInfo);
#pragma unroll
        for (uint32_t i = 0; i < BSA_KSCALE_BANK_COPY_NUM; ++i) {
            DataCopyPad(dstTensor[i * kScaleUbStride + dstOffset], srcTensor[srcOffset], dataCopyParams,
                        dataCopyPadParams);
        }
    }

    __aicore__ inline void ProcessPA(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                     const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo,
                                     const ConstInfo &constInfo)
    {
        int64_t byteOffset1 = 0;
        int64_t byteOffset2 = 0;
        if constexpr (GM_FORMAT == BSALayout::PA_BNSD) {
            if (runInfo.sparseBlkIdx1 >= 0) {
                GetPaKScaleOffset(byteOffset1, runInfo.sparseBlkIdx1, blockTableTensor, runInfo, constInfo);
                CopyKScaleBnsdToBankConflictFreeUb(dstTensor, srcTensor, byteOffset1, 0U,
                                                   static_cast<uint32_t>(runInfo.s2SparseBlk1RealSize), constInfo);
            }
            if (runInfo.sparseBlkIdx2 >= 0 && runInfo.s2SparseBlk2RealSize > 0) {
                GetPaKScaleOffset(byteOffset2, runInfo.sparseBlkIdx2, blockTableTensor, runInfo, constInfo);
                CopyKScaleBnsdToBankConflictFreeUb(dstTensor, srcTensor, byteOffset2,
                                                   static_cast<uint32_t>(constInfo.kvSparseBlockSize),
                                                   static_cast<uint32_t>(runInfo.s2SparseBlk2RealSize), constInfo);
            }
        } else if constexpr (GM_FORMAT == BSALayout::PA_BSND) {
            if (runInfo.sparseBlkIdx1 >= 0) {
                GetPaKScaleOffset(byteOffset1, runInfo.sparseBlkIdx1, blockTableTensor, runInfo, constInfo);
                CopyKScaleBsndToBankConflictFreeUb(dstTensor, srcTensor, byteOffset1, 0U,
                                                   static_cast<uint32_t>(runInfo.s2SparseBlk1RealSize), constInfo);
            }
            if (runInfo.sparseBlkIdx2 >= 0 && runInfo.s2SparseBlk2RealSize > 0) {
                GetPaKScaleOffset(byteOffset2, runInfo.sparseBlkIdx2, blockTableTensor, runInfo, constInfo);
                CopyKScaleBsndToBankConflictFreeUb(dstTensor, srcTensor, byteOffset2,
                                                   static_cast<uint32_t>(constInfo.kvSparseBlockSize),
                                                   static_cast<uint32_t>(runInfo.s2SparseBlk2RealSize), constInfo);
            }
        }
    }
    __aicore__ inline void GetPaKScaleOffset(int64_t &byteOffset, const int64_t sparseBlkIdx,
                                             const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo,
                                             const ConstInfo &constInfo)
    {
        int64_t blockTableOffset = runInfo.boIdx * constInfo.blockTableDim2 + sparseBlkIdx;
        int64_t physicalBlock = blockTableTensor.GetValue(blockTableOffset);
        byteOffset = physicalBlock * (constInfo.paBlockStride / sizeof(float)) +
                     runInfo.n2oIdx * constInfo.kvSparseBlockSize;
    }
};

// ----------------------------------------------CopyQueryScaleGmToUb--------------------------------
template <typename T, BSALayout GM_FORMAT>
class CopyQueryScaleGmToUb {
public:
    __aicore__ inline void operator()(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                      const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        ProcessGS1(dstTensor, srcTensor, runInfo, constInfo);
    }

private:
    __aicore__ inline void ProcessGS1(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                      const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        if constexpr (GM_FORMAT == BSALayout::TND) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 64;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = (constInfo.n1Size - 1) * sizeof(float);
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams{};
            int64_t n1Idx = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx;
            int64_t qScaleOffset = runInfo.qBScalarOffset + runInfo.sOuterOffset * constInfo.n1Size + n1Idx;
            DataCopyPad(dstTensor, srcTensor[qScaleOffset], dataCopyParams, dataCopyPadParams);
        } else if constexpr (GM_FORMAT == BSALayout::NTD) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = runInfo.halfS1RealSize * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams{};
            int64_t qScaleOffset = runInfo.qBScalarOffset + runInfo.sOuterOffset;
            DataCopyPad(dstTensor, srcTensor[qScaleOffset], dataCopyParams, dataCopyPadParams);
        }
    }
};
#endif
