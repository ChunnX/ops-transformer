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
 * \file dense_li_softmax_lse_kernel_base.h
 * \brief Kernel base for A5 (arch35) implementation
 */

#ifndef DENSE_LI_SOFTMAX_LSE_KERNEL_BASE_H
#define DENSE_LI_SOFTMAX_LSE_KERNEL_BASE_H

#include "dense_lightning_indexer_softmax_lse_tiling_regbase.h"
#include "dense_lightning_indexer_softmax_lse_regbase_common.h"
#include "dense_lightning_indexer_softmax_lse_cube_block.h"
#include "dense_lightning_indexer_softmax_lse_vector_block.h"

namespace DenseLISoftmaxLseRegbase {
using T = float;

template <typename CubeBlockType, typename VecBlockType>
class DenseLISoftmaxLseKernelBase {
public:
    __aicore__ inline DenseLISoftmaxLseKernelBase(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR weights,
                                GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengths,
                                GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
                                const optiling::DenseLISoftmaxLseRegBaseTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void FreeBuf();
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetRunInfo(DenseLISoftmaxLseRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx);

    TPipe *pipe = nullptr;
    const optiling::DenseLISoftmaxLseRegBaseTilingData *__restrict tilingData = nullptr;

    GlobalTensor<int64_t> actualSeqLengthsQueryGm;
    GlobalTensor<int64_t> actualSeqLengthsKeyGm;

    GlobalTensor<T> mm1Res;
    GlobalTensor<T> vec1Res;

    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;

    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> mm1Buffers;

    DenseLISoftmaxLseConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLISoftmaxLseKernelBase<CubeBlockType, VecBlockType>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights,
    GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengths,
    GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
    const optiling::DenseLISoftmaxLseRegBaseTilingData *__restrict tiling,
    TPipe *tPipe) {
    pipe = tPipe;
    tilingData = tiling;
    SetConstInfo();

    if (actualSeqLengthsQ != nullptr) {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQ, constInfo.bSize);
    }
    if (actualSeqLengths != nullptr) {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths, constInfo.bSize);
    }

    InitMMResBuf();

    mm1Res.SetGlobalBuffer((__gm__ T *)(workspace));

    if ASCEND_IS_AIV {
        vecBlock.InitParams(constInfo, tilingData);
        vecBlock.InitGlobalBuffer(weights, softmaxMax, softmaxSum, mm1Res, vec1Res);
        vecBlock.InitBuffers(pipe);
    } else if ASCEND_IS_AIC {
        cubeBlock.SetCubeBlockParams(tPipe, &l1BufferManager);
        cubeBlock.InitCubeBuffers();
        cubeBlock.InitGlobalBuffer(query, key, mm1Res);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLISoftmaxLseKernelBase<CubeBlockType, VecBlockType>::SetConstInfo() {
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx();
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % MODE_NUM_2;
    } else {
        constInfo.aicIdx = GetBlockIdx();
    }

    auto &baseInfo = tilingData->baseParams;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.gSizeQuery = baseInfo.gSizeQuery;
    constInfo.gSizeQueryIndex = baseInfo.gSizeQueryIndex;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.kSize = baseInfo.kSize;
    constInfo.dSizeQuery = baseInfo.dSizeQuery;
    constInfo.dSizeQueryIndex = baseInfo.dSizeQueryIndex;
    constInfo.sparseMode = static_cast<DenseLISoftmaxLseSparseMode>(baseInfo.sparseMode);
    constInfo.scaleValue = baseInfo.scaleValue;
    constInfo.syKBaseSize = 16384 / (constInfo.n2Size * constInfo.gSizeQueryIndex);
    constInfo.pKBaseSize = 16384 / (constInfo.n2Size * constInfo.gSizeQuery);
    constInfo.syKBaseSize = Min(1024, constInfo.syKBaseSize);
    constInfo.pKBaseSize = Min(1024, constInfo.pKBaseSize);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLISoftmaxLseKernelBase<CubeBlockType, VecBlockType>::InitMMResBuf() {
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    ubBufferManager.Init(pipe, UB_MAX_SIZE);
    mm1Buffers.Init(ubBufferManager, 32 * 256 * sizeof(T));

    if ASCEND_IS_AIV {
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM1_TO_VEC_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM1_TO_VEC_FLAG[1]);
    } else if ASCEND_IS_AIC {
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_GATHER_TO_MM1_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_GATHER_TO_MM1_FLAG[1]);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLISoftmaxLseKernelBase<CubeBlockType, VecBlockType>::FreeBuf() {
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM1_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM1_FLAG[1]);
    } else if ASCEND_IS_AIC {
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM1_TO_VEC_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM1_TO_VEC_FLAG[1]);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLISoftmaxLseKernelBase<CubeBlockType, VecBlockType>::SetRunInfo(
    DenseLISoftmaxLseRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx) {
    runInfo.taskId = taskId;
    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    runInfo.taskIdMod2 = taskId & 1;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLISoftmaxLseKernelBase<CubeBlockType, VecBlockType>::Process() {
    int64_t bS1TotalSize = tilingData->multiCoreParams.totalSize;
    DenseLISoftmaxLseRunInfo runInfo;

    for (int64_t bS1Idx = constInfo.aicIdx; bS1Idx < bS1TotalSize; bS1Idx += GetBlockNum()) {
        int64_t bIdx = bS1Idx / constInfo.s1Size;
        int64_t s1Idx = bS1Idx - bIdx * constInfo.s1Size;

        SetRunInfo(runInfo, bS1Idx, bIdx, s1Idx);

        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<2, PIPE_MTE1>(SYNC_MM1_TO_VEC_FLAG[runInfo.taskIdMod2]);
            cubeBlock.ComputeMm1(mm1Buffers.Get(), runInfo, constInfo);
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_MM1_TO_VEC_FLAG[runInfo.taskIdMod2]);
        }

        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM1_TO_VEC_FLAG[runInfo.taskIdMod2]);
            vecBlock.CopyInWeight(runInfo);
            vecBlock.ProcessVec0(mm1Buffers.Get(), runInfo);
            vecBlock.ProcessSoftmax(runInfo);
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_MM1_TO_VEC_FLAG[runInfo.taskIdMod2]);
        }
    }

    SyncAll<false>();
    FreeBuf();
}

}

#endif // DENSE_LI_SOFTMAX_LSE_KERNEL_BASE_H