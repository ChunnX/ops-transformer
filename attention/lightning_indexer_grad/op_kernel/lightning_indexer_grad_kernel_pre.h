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
* \file lightning_indexer_grad_service_vector_pre.h
* \brief
*/
#ifndef LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_PRE_H
#define LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_PRE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_grad_common.h"
#include "lightning_indexer_grad_tiling.h"

namespace LigKernel {
using namespace LIGCommon;
using namespace AscendC;
using namespace optiling;

template <typename LIGT>
class LIGVectorPre {
public:
    using dataType = typename LIGT::dataType;
    
    __aicore__ inline LIGVectorPre(){};
    __aicore__ inline void Init(TPipe *pipe_in, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dweights,
                                __gm__ uint8_t *workspace, const LIGTilingData *__restrict orgTilingData,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void DoSplitPadding(uint32_t totalLen, uint32_t coreIdx, uint32_t coreNum, uint32_t &outBegin, uint32_t &outLen);
    
    using D_T = typename LIGT::dataType;
protected:
    TPipe *pipe;
    GlobalTensor<float> dkWorkSpaceGm;
    GlobalTensor<float> dkCoreWorkspaceGM;

    const LIGTilingData *__restrict tilingData;

    uint32_t cBlockIdx;
    uint32_t kPreBlockFactor;
    uint32_t kPreBlockTotal;
    uint32_t kPreBlockTail;
    int64_t initdkSize;
    int64_t dkOffset;
    
    __gm__ uint8_t *dqPtr = nullptr;
    __gm__ uint8_t *dkPtr = nullptr;
    __gm__ uint8_t *dweightsPtr = nullptr;
    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGmK;
};

template <typename LIGT>
__aicore__ inline void LIGVectorPre<LIGT>::DoSplitPadding(uint32_t totalLen, uint32_t coreIdx, uint32_t coreNum, uint32_t &outBegin, uint32_t &outLen)
{
    uint64_t base = totalLen / coreNum;
    uint64_t remainder = totalLen % coreNum;
    if (coreIdx < remainder) {
        outLen = base + 1;
        outBegin = coreIdx * (base + 1);
    } else {
        outLen = base;
        outBegin = remainder * (base + 1) + (coreIdx - remainder) * base;
    }

    if (outBegin >= totalLen) {
        outBegin = totalLen;
        outLen = 0;
    } else if (outBegin + outLen > totalLen) {
        outLen = totalLen - outBegin;
    }
}

template <typename LIGT>
__aicore__ inline void LIGVectorPre<LIGT>::Init(TPipe *pipe_in, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dweights,
                                            __gm__ uint8_t *workspace, const LIGTilingData *__restrict orgTilingData,
                                            __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK)
{
    cBlockIdx = GetBlockIdx();
    pipe = pipe_in;
    tilingData = orgTilingData;

    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkWorkSpaceOffset / sizeof(float));

    // Init workspace to store per-core dKey partial results.
    dkCoreWorkspaceGM.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkCoreWorkspaceOffset / sizeof(float));

    uint32_t coreNum = tilingData->usedCoreNum;
    kPreBlockFactor = (tilingData->dkSize + coreNum - 1) / coreNum;
    kPreBlockTotal = (tilingData->dkSize + kPreBlockFactor - 1) / kPreBlockFactor;
    int64_t kPreTailNumTmp = tilingData->dkSize % kPreBlockFactor;
    kPreBlockTail = kPreTailNumTmp == 0 ? kPreBlockFactor : kPreTailNumTmp;

    initdkSize = cBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
    dkOffset = ((int64_t)cBlockIdx) * kPreBlockFactor;

    dqPtr = dq;
    dkPtr = dk;
    dweightsPtr = dweights;

    if (actualSeqLengthsQ != nullptr) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, tilingData->batch);
    }
    if (actualSeqLengthsK != nullptr) {
        actualSeqLengthsGmK.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsK, tilingData->batch);
    }
}

template <typename LIGT> 
__aicore__ inline void LIGVectorPre<LIGT>::Process()
{
    // process clear dk workspace
    if (g_coreType == AIV && cBlockIdx < kPreBlockTotal) {
        InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);
    }    
    // TND layout: process clear dq, dk, dweights's padding
    if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        if (g_coreType == AIV) {
            uint32_t batch = tilingData->batch;
            uint32_t seqlenQ = tilingData->seqlenQ;
            uint32_t seqlenK = tilingData->seqlenK;
            uint32_t actualTotalLenQ = 0;
            uint32_t actualTotalLenK = 0;
            uint32_t headNumQ = tilingData->headNumQ;
            uint32_t headNumK = tilingData->headNumK;
            uint32_t headDim = tilingData->headDim;
            uint32_t usedCoreNum = tilingData->usedCoreNum;
            if (batch > 0) {
                actualTotalLenQ = actualSeqLengthsGmQ.GetValue(batch - 1);
                actualTotalLenK = actualSeqLengthsGmK.GetValue(batch - 1);
            }

            // Output GlobalTensor
            GlobalTensor<D_T> dqGm;
            GlobalTensor<D_T> dkGm;
            GlobalTensor<D_T> dweightsGm;

            // init output global tensor
            dqGm.SetGlobalBuffer((__gm__ D_T *)dqPtr);
            dkGm.SetGlobalBuffer((__gm__ D_T *)dkPtr);
            dweightsGm.SetGlobalBuffer((__gm__ D_T *)dweightsPtr);
            
            // process padding dq, dweights
            if (actualTotalLenQ < seqlenQ) {
                uint32_t padLenQ = seqlenQ - actualTotalLenQ;
                uint32_t beginPadQ = 0;
                uint32_t lenPadQ = 0;
                DoSplitPadding(padLenQ, cBlockIdx, usedCoreNum, beginPadQ, lenPadQ);
                if (lenPadQ > 0) {
                    InitOutput<D_T>(dqGm[(actualTotalLenQ + beginPadQ) * headNumQ * headDim], headNumQ * headDim, 0);
                    InitOutput<D_T>(dweightsGm[(actualTotalLenQ + beginPadQ) * headNumQ], headNumQ, 0);
                }
            }
            // process padding dk
            if (actualTotalLenK < seqlenK) {
                uint32_t padLenK = seqlenK - actualTotalLenK;
                uint32_t beginPadK = 0;
                uint32_t lenPadK = 0;
                DoSplitPadding(padLenK, cBlockIdx, usedCoreNum, beginPadK, lenPadK);
                if (lenPadK > 0) {
                    InitOutput<D_T>(dkGm[(actualTotalLenK + beginPadK) * headNumK * headDim], headNumK * headDim, 0);
                }
            }
        }
    }
}

template <typename LIGT> 
__aicore__ inline void LIGVectorPre<LIGT>::SyncALLCores()
{
    SyncAll();
}
} // namespace LigKernel
#endif