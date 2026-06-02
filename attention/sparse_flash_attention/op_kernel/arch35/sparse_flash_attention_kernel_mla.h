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
 * \file sparse_flash_attention_kernel_mla.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
#define SPARSE_FLASH_ATTENTION_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_service_cube_mla.h"
#include "sparse_flash_attention_service_vector_mla.h"
#include "sparse_flash_attention_common_arch35.h"
#include "sparse_flash_attention_kvcache.h"

#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

using matmul::MatmulType;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType> class SparseFlashAttentionKernelMla {
public:
    ARGS_TRAITS;

    __aicore__ inline SparseFlashAttentionKernelMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
                                const SparseFlashAttentionTilingDataMla *__restrict tiling,
                                __gm__ uint8_t *gmTiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
        const SparseFlashAttentionTilingDataMla *__restrict tiling,
        TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &sfaRunInfo, RunParamStr &sfaRunParam, int64_t sfaTaskId,
        int64_t sfaS2LoopCount, int64_t sfaS2LoopLimit, int64_t sfaMultiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &sfaRunInfo, RunParamStr &sfaRunParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t sfaGS1Index, RunParamStr &sfaRunParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &sfaRunParam, RunInfo &sfaRunInfo);

    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t sfaBIdx);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev, uint32_t s2EndPrev);

    TPipe *pipe;

    const SparseFlashAttentionTilingDataMla *__restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 2;
    /* 核间通道 */
    BufferManager<BufferType::GM> sfaGmBufferManager;

    BufferManager<BufferType::UB> sfaUbBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> sfaBmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> sfaBmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> sfaL1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> sfaL1RightBuffers;
    CVSharedParams sfaSharedParams;
    /* GM信息 */
    __gm__ int32_t *sfaActualSeqKvlenAddr = nullptr;
    __gm__ int32_t *sfaActualSeqQlenAddr = nullptr;

    GlobalTensor<int32_t> sfaActualSeqLengthsQGm;
    uint32_t sfaUsedCoreNum = 0U;

    GlobalTensor<int32_t> oriTopkLengthGm;
    bool hasOriTopkLength = false;
    /* workspace 空间 */
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    /* 核Index信息 */
    int32_t sfaAicIdx;

    /* 切G时最大s2Loop */
    int64_t sfaMaxS2LoopCnt;

    /* 初始化后不变的信息 */
    ConstInfo sfaConstInfo;

    /* 模板库Block */
    CubeBlockType sfaCubeBlock;
    VecBlockType sfaVecBlock;

    uint32_t sfaCrossCoreSyncBufId = 0;
};

template <typename CubeBlockType, typename VecBlockType>
 __aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, const SparseFlashAttentionTilingDataMla *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    sfaConstInfo.sfaSubBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->sfaAicIdx = GetBlockIdx();
        sfaConstInfo.sfaAicIdx = this->sfaAicIdx;
        sfaConstInfo.sfaAivIdx = 0;
    } else {
        sfaConstInfo.sfaAivIdx = GetBlockIdx();
        this->sfaAicIdx = sfaConstInfo.sfaAivIdx >> 1;
        sfaConstInfo.sfaAicIdx = this->sfaAicIdx;
        this->tilingData = tiling;
    }

    sfaConstInfo.sfaS1BaseSize = 64;
    sfaConstInfo.sfaS2BaseSize = 128;

    this->pipe = tPipe;
    sfaVecBlock.InitVecBlock(tPipe, this->tilingData, this->sfaSharedParams,  \
        this->sfaAicIdx, sfaConstInfo.sfaSubBlockIdx, actualSeqLengthsQ, actualSeqLengths);
    if ASCEND_IS_AIV {
        sfaConstInfo.sfaBSize = this->sfaSharedParams.sfaBSize;
        sfaConstInfo.sfaGSize = this->sfaSharedParams.sfaGSize;
        sfaConstInfo.sfaS1Size = this->sfaSharedParams.sfaS1Size;
        sfaConstInfo.sfaDSizeV = 512;
        sfaConstInfo.sfaNeedInit = this->sfaSharedParams.sfaNeedInit;
        sfaConstInfo.sfaReturnSoftmaxLse = this->sfaSharedParams.sfaReturnSoftmaxLse;
    }
    sfaVecBlock.CleanOutput(attentionOut, softmaxMax, softmaxSum, sfaConstInfo);
    /* cube侧不依赖sfaSharedParams的scalar前置 */
    InitMMResBuf(workspace);
    if ASCEND_IS_AIC {
        sfaCubeBlock.InitCubeBlock(pipe, sfaL1BufferManager, query, queryRope);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&sfaSharedParams);
#pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, queryRope, keyRope, sparseIndices, \
        blockTable, actualSeqLengthsQ, actualSeqLengths, softmaxMax, softmaxSum, \
        workspace, tiling, tPipe); // gm设置
    this->InitCalcParamsEach();
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType> __aicore__ inline
void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitCalcParamsEach()
{
    // 计算总的基本块
    sfaMaxS2LoopCnt = 0; // 所有核中最大累计s2Loop
    uint32_t sfaTotalBaseNum = 0;
    uint32_t sfaS1GBaseSize = sfaConstInfo.sfaGSize;
    uint32_t sfaActBatchS2 = 1;
    uint32_t sfaCoreNum = GetBlockNum(); // G128时相邻两个cube核处理一个s1，sfaCoreNum减半
    uint32_t sfaCurrCoreIdx = sfaAicIdx;
    if constexpr (IS_SPLIT_G) {
        sfaCoreNum = sfaCoreNum >> 1;
        sfaCurrCoreIdx = sfaCurrCoreIdx >> 1;
    }
    uint32_t sfaActBatchS1 = 1;
    for (uint32_t sfaBIdx = 0; sfaBIdx < sfaConstInfo.sfaBSize; sfaBIdx++) {
        sfaActBatchS1 = GetBalanceActualSeqLengths(sfaActualSeqLengthsQGm, sfaBIdx); // 不切S2，只关注S1
        if (sfaActBatchS1 < sfaConstInfo.sfaS1Size) {
            sfaConstInfo.sfaNeedInit = true;
        }
        sfaTotalBaseNum += sfaActBatchS1 * sfaActBatchS2;
    }
    uint32_t sfaAvgBaseNum = 1;
    if (sfaTotalBaseNum > sfaCoreNum) {
        sfaAvgBaseNum = (sfaTotalBaseNum + sfaCoreNum - 1) / sfaCoreNum;
        if constexpr (IS_SPLIT_G) {
            sfaUsedCoreNum = (sfaTotalBaseNum + sfaAvgBaseNum - 1) / sfaAvgBaseNum << 1;
        }
    } else {
        if constexpr (IS_SPLIT_G) {
            sfaUsedCoreNum = sfaTotalBaseNum << 1;
        } else {
            sfaUsedCoreNum = sfaTotalBaseNum;
        }
    }

    if constexpr (IS_SPLIT_G) {
        sfaMaxS2LoopCnt = sfaAvgBaseNum * (sfaConstInfo.sfaSparseBlockCount + sfaConstInfo.sfaS2BaseSize - 1) / sfaConstInfo.sfaS2BaseSize;
    }

    if (sfaAicIdx >= sfaUsedCoreNum) {
        return;
    }
	// 计算当前核的基本块
    uint32_t sfaAccumBaseNum = 0; // 当前累积的基本块数
    uint32_t sfaTargetBaseNum = 0;
    uint32_t sfaLastValidBIdx = 0;
    uint32_t sfaLastValidactBatchS1 = 0;
    bool sfaSetStart = false;
    sfaTargetBaseNum = (sfaCurrCoreIdx + 1) * sfaAvgBaseNum; // 计算当前的目标权重
    uint32_t sfaTargetStartBaseNum = sfaTargetBaseNum - sfaAvgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < sfaConstInfo.sfaBSize * sfaConstInfo.sfaN2Size; bN2Idx++) {
        uint32_t sfaBIdx = bN2Idx / sfaConstInfo.sfaN2Size;
        sfaActBatchS1 = GetBalanceActualSeqLengths(sfaActualSeqLengthsQGm, sfaBIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < sfaActBatchS1; s1GIdx++) {
            sfaAccumBaseNum += 1;
            if (!sfaSetStart && sfaAccumBaseNum >= sfaTargetStartBaseNum) {
                sfaConstInfo.sfaBN2Start = bN2Idx;
                sfaConstInfo.sfaGS1Start = s1GIdx;
                sfaSetStart = true;
            }
            if (sfaAccumBaseNum >= sfaTargetBaseNum) {
                // 更新当前核的End分核信息
                sfaConstInfo.sfaBN2End = bN2Idx;
                sfaConstInfo.sfaGS1End = s1GIdx;
                sfaConstInfo.sfaS2End = 0;
                if (sfaCurrCoreIdx != 0) {
                    GetAxisStartIdx(sfaConstInfo.sfaBN2Start, sfaConstInfo.sfaGS1Start, 0);
                }
                return;
            }
        }
        if ((sfaActBatchS1 > 0) && (sfaActBatchS2 > 0)) {
            sfaLastValidBIdx = sfaBIdx;
            sfaLastValidactBatchS1 = sfaActBatchS1;
        }
    }
    if (!sfaSetStart) {
        sfaConstInfo.sfaBN2Start = sfaLastValidBIdx;
        sfaConstInfo.sfaGS1Start = sfaLastValidactBatchS1 - 1;
    }
    if (sfaAccumBaseNum < sfaTargetBaseNum) {
		// 更新最后一个核的End分核信息
        sfaConstInfo.sfaBN2End = sfaLastValidBIdx;
        sfaConstInfo.sfaGS1End = sfaLastValidactBatchS1 - 1;
        sfaConstInfo.sfaS2End = 0;
        if (sfaCurrCoreIdx != 0) {
            GetAxisStartIdx(sfaConstInfo.sfaBN2Start, sfaConstInfo.sfaGS1Start, 0);
        }
        return;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline uint64_t
SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::GetBalanceActualSeqLengths(
    GlobalTensor<int32_t> &actualSeqLengths, uint32_t sfaBIdx)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        if (sfaBIdx > 0) {
            return sfaActualSeqQlenAddr[sfaBIdx] - sfaActualSeqQlenAddr[sfaBIdx - 1];
        } else if (sfaBIdx == 0) {
            return sfaActualSeqQlenAddr[0];
        } else {
            return 0;
        }
    } else {
        if (sfaConstInfo.sfaIsActualLenDimsNull == 1) {
            return sfaConstInfo.sfaS1Size;
        } else {
            return sfaActualSeqQlenAddr[sfaBIdx];
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                uint32_t s1GEndPrev,
                                                                                uint32_t s2EndPrev)
{
    uint32_t sfaBEndPrev = bN2EndPrev / sfaConstInfo.sfaN2Size;
    uint32_t sfaActualSeqQPrev = GetBalanceActualSeqLengths(sfaActualSeqLengthsQGm, sfaBEndPrev);
    uint32_t sfaS1GPrevBaseNum = sfaActualSeqQPrev;
    sfaConstInfo.sfaBN2Start = bN2EndPrev;
    sfaConstInfo.sfaGS1Start = s1GEndPrev;
    
    sfaConstInfo.sfaS2Start = 0;
    if (s1GEndPrev >= sfaS1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        sfaConstInfo.sfaGS1Start = 0;
        sfaConstInfo.sfaBN2Start++;
    } else {
        sfaConstInfo.sfaGS1Start++;
    }
}

template <typename CubeBlockType, typename VecBlockType> __aicore__ inline
void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, const SparseFlashAttentionTilingDataMla *__restrict tiling, TPipe *tPipe)
{
    if (actualSeqLengthsQ != nullptr) {
        sfaActualSeqQlenAddr = (__gm__ int32_t *)actualSeqLengthsQ;
    }
    if (actualSeqLengths != nullptr) {
        sfaActualSeqKvlenAddr = (__gm__ int32_t *)actualSeqLengths;
    }

    sfaVecBlock.InitGlobalBuffer(key, value, keyRope, sparseIndices, blockTable, softmaxMax, softmaxSum);
    sfaCubeBlock.InitCubeInput(key, keyRope, sparseIndices, blockTable, actualSeqLengthsQ, sfaConstInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t *workspace)
{
    uint32_t sfaMm1ResultSize = sfaConstInfo.sfaS1BaseSize / CV_RATIO * sfaConstInfo.sfaS2BaseSize * sizeof(T);
    uint32_t sfaMm2ResultSize = sfaConstInfo.sfaS1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t sfaMm2LeftSize = sfaConstInfo.sfaS1BaseSize * sfaConstInfo.sfaS2BaseSize * sizeof(Q_T);
    uint32_t sfaMm1RightSize = sfaConstInfo.sfaS2BaseSize * 576 * sizeof(Q_T);
    sfaL1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    sfaL1RightBuffers.Init(sfaL1BufferManager, sfaMm1RightSize);
    sfaL1RightBuffers.Get().SetCrossCoreID(sfaCrossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    sfaCrossCoreSyncBufId++;
    sfaL1RightBuffers.Get().SetCrossCoreID(sfaCrossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    sfaCrossCoreSyncBufId++;
    sfaL1RightBuffers.Get().SetCrossCoreID(sfaCrossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    sfaCrossCoreSyncBufId++;
    if ASCEND_IS_AIC {
        sfaL1RightBuffers.Get().SetCrossCore();
        sfaL1RightBuffers.Get().SetCrossCore();
        sfaL1RightBuffers.Get().SetCrossCore();
    }
    sfaUbBufferManager.Init(pipe, sfaMm1ResultSize * 2 + sfaMm2ResultSize);
    sfaBmm2Buffers.Init(sfaUbBufferManager, sfaMm2ResultSize);
    sfaBmm2Buffers.Get().SetCrossCoreID(sfaCrossCoreSyncBufId, sfaCrossCoreSyncBufId);
    sfaCrossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        sfaBmm2Buffers.Get().SetCrossCore();
    }
    sfaBmm1Buffers.Init(sfaUbBufferManager, sfaMm1ResultSize);
    sfaBmm1Buffers.Get().SetCrossCoreID(sfaCrossCoreSyncBufId, sfaCrossCoreSyncBufId);
    sfaCrossCoreSyncBufId++;
    sfaBmm1Buffers.Get().SetCrossCoreID(sfaCrossCoreSyncBufId, sfaCrossCoreSyncBufId);
    sfaCrossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        sfaBmm1Buffers.Get().SetCrossCore();
        sfaBmm1Buffers.Get().SetCrossCore();
    }

    uint32_t sfaV0ResSize = sfaConstInfo.sfaS2BaseSize * 576U * sizeof(Q_T);
    int64_t sfaTotalOffset;
    if constexpr (IS_SPLIT_G) {
        sfaTotalOffset = sfaV0ResSize * 3 * (sfaAicIdx >> 1U);
    } else {
        sfaTotalOffset = sfaV0ResSize * 3 * sfaAicIdx;
    }
    sfaGmBufferManager.Init(workspace + sfaTotalOffset);
    v0ResGmBuffers.Init(sfaGmBufferManager, sfaV0ResSize);
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, sfaCrossCoreSyncBufId);
    sfaCrossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, sfaCrossCoreSyncBufId);
    sfaCrossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, sfaCrossCoreSyncBufId);
    sfaCrossCoreSyncBufId++;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    sfaVecBlock.InitLocalBuffer(pipe, sfaConstInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积
    sfaUsedCoreNum = sfaSharedParams.sfaUsedCoreNum;

    if ASCEND_IS_AIC {
        sfaConstInfo.sfaBSize = this->sfaSharedParams.sfaBSize;
        sfaConstInfo.sfaGSize = this->sfaSharedParams.sfaGSize;
        sfaConstInfo.sfaS1Size = this->sfaSharedParams.sfaS1Size;
        sfaConstInfo.sfaDSizeV = 512;
        sfaConstInfo.sfaNeedInit = this->sfaSharedParams.sfaNeedInit;
    }
    sfaConstInfo.sfaN2Size = sfaSharedParams.sfaN2Size;
    sfaConstInfo.sfaS2Size = sfaSharedParams.sfaS2Size;
    sfaConstInfo.sfaDSize = sfaSharedParams.sfaDSize;
    sfaConstInfo.sfaDSizeVInput = sfaSharedParams.sfaDSizeVInput;
    sfaConstInfo.sfaDSizeRope = 64;
    sfaConstInfo.sfaDSizeNope = 512;
    sfaConstInfo.sfaTileSize = sfaSharedParams.sfaTileSize;
    sfaConstInfo.sfaSparseBlockCount = sfaSharedParams.sfaSparseBlockCount;
    sfaConstInfo.sfaSparseBlockSize = 1;
    sfaConstInfo.sfaCmpRatio = sfaSharedParams.sfaCmpRatio;
    sfaConstInfo.sfaOriWinLeft = sfaSharedParams.sfaOriWinLeft;
    sfaConstInfo.sfaOriWinRight = sfaSharedParams.sfaOriWinRight;
    sfaConstInfo.sfaSparseMode = sfaSharedParams.sfaOriMaskMode;
    sfaConstInfo.sfaS1S2 = sfaConstInfo.sfaS1Size * sfaConstInfo.sfaS2Size;
    sfaConstInfo.sfaGS1 = sfaConstInfo.sfaGSize * sfaConstInfo.sfaS1Size;
    sfaConstInfo.sfaN2G = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaGSize;
    sfaConstInfo.sfaGD = sfaConstInfo.sfaGSize * sfaConstInfo.sfaDSize;
    sfaConstInfo.sfaN2GD = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaGD;
    sfaConstInfo.sfaS1Dv = sfaConstInfo.sfaS1Size * sfaConstInfo.sfaDSizeV;
    sfaConstInfo.sfaS2Dv = sfaConstInfo.sfaS2Size * sfaConstInfo.sfaDSizeV;
    sfaConstInfo.sfaN2Dv = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaDSizeV;
    sfaConstInfo.sfaGDv = sfaConstInfo.sfaGSize * sfaConstInfo.sfaDSizeV;
    sfaConstInfo.sfaGS1Dv = sfaConstInfo.sfaGSize * sfaConstInfo.sfaS1Dv;
    sfaConstInfo.sfaIsActualLenDimsNull = sfaSharedParams.sfaIsActualSeqLengthsNull;
    sfaConstInfo.sfaIsActualLenDimsKVNull = sfaSharedParams.sfaIsActualSeqLengthsKVNull;
    sfaConstInfo.sfaN2S2Dv = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaS2Dv;
    sfaConstInfo.sfaN2GDv = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaGDv;
    sfaConstInfo.sfaS2BaseN2Dv = sfaConstInfo.sfaS2BaseSize * sfaConstInfo.sfaN2Dv;
    sfaConstInfo.sfaN2GS1Dv = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaGS1Dv;
    sfaConstInfo.sfaLayoutType = sfaSharedParams.sfaLayoutType;

    sfaConstInfo.sfaIsActualLenDimsNull = sfaSharedParams.sfaIsActualSeqLengthsNull;
    sfaConstInfo.sfaIsActualLenDimsKVNull = sfaSharedParams.sfaIsActualSeqLengthsKVNull;

    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        // (BS)ND
        sfaConstInfo.sfaS1BaseN2GDv = sfaConstInfo.sfaS1BaseSize * sfaConstInfo.sfaN2GDv;
        sfaConstInfo.sfaMm1Ka = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaDSize;
        if ASCEND_IS_AIV {
            sfaConstInfo.sfaAttentionOutStride = (sfaConstInfo.sfaN2G - sfaConstInfo.sfaGSize) * sfaConstInfo.sfaDSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
        // BSH/BSNGD
        sfaConstInfo.sfaS1BaseN2GDv = sfaConstInfo.sfaS1BaseSize * sfaConstInfo.sfaN2GDv;
        sfaConstInfo.sfaMm1Ka = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaDSize;
        if ASCEND_IS_AIV {
            sfaConstInfo.sfaAttentionOutStride = (sfaConstInfo.sfaN2G - sfaConstInfo.sfaGSize) * sfaConstInfo.sfaDSizeV * sizeof(OUTPUT_T);
        }
    }
    if ASCEND_IS_AIV {
        sfaConstInfo.sfaSoftmaxScale = sfaSharedParams.sfaSoftmaxScale;
        sfaConstInfo.sfaOriBlockSize = sfaSharedParams.sfaOriBlockSize;
        sfaConstInfo.sfaOriMaxBlockNumPerBatch = sfaSharedParams.sfaOriMaxBlockNumPerBatch;
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    // bsize + 1-> bsize
    this->sfaConstInfo.sfaActualSeqLenSize = this->sfaSharedParams.sfaBSize;
    this->sfaConstInfo.sfaActualSeqLenKVSize = this->sfaSharedParams.sfaBSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sfaSharedParams.sfaNeedInit) {
        SyncAll<false>();
    }

    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    bool sfaHasLoad = sfaAicIdx < sfaUsedCoreNum;
    if (!sfaHasLoad) {
        if ASCEND_IS_AIV {
            if constexpr (IS_SPLIT_G) {
                for (int64_t sfaLoopCnt = 0; sfaLoopCnt < sfaMaxS2LoopCnt; sfaLoopCnt++) {
                    CrossCoreSetFlag<SFA_SYNC_MODE0, PIPE_MTE3>(15);
                    CrossCoreWaitFlag<SFA_SYNC_MODE0, PIPE_MTE3>(15);
                }
            }
        }
        return;
    }

    // 适配分核左闭右开
    uint32_t sfaBIdx = sfaConstInfo.sfaBN2End / sfaConstInfo.sfaN2Size;
    uint32_t sfaActS1Size = GetBalanceActualSeqLengths(sfaActualSeqLengthsQGm, sfaBIdx);
    uint32_t sfaGS1max = sfaActS1Size;
    if (sfaConstInfo.sfaGS1End + 1 < sfaGS1max) {
        /* sfaConstInfo.sfaGS1End != sfaGS1max时，gS1End需要往后加一格, bN2End不变 */
        sfaConstInfo.sfaGS1End = sfaConstInfo.sfaGS1End + 1;
    } else {
        /* sfaConstInfo.sfaGS1End == sfaGS1max，bN2End需要往后加一格，bN2End变为0，以代表末尾 */
        sfaConstInfo.sfaBN2End = sfaConstInfo.sfaBN2End + 1;
        sfaConstInfo.sfaGS1End = 0;
    }

    // 分核信息
    uint32_t sfaBN2StartIdx = sfaConstInfo.sfaBN2Start;
    uint32_t sfaBN2EndIdx = sfaConstInfo.sfaBN2End;
    uint32_t sfaGS1StartIdx = sfaConstInfo.sfaGS1Start;
    uint32_t sfaNextGs1Idx = sfaConstInfo.sfaGS1End;
    uint32_t sfaS2StartIdx = 0;
    uint32_t sfaS2EndIdx = 0;
    uint32_t sfaS2LoopLimit = 0;

    if (sfaNextGs1Idx != 0) {
        sfaBN2EndIdx++;
    }

    int64_t sfaTaskId = 0;
    bool sfaNotLast = true;
    RunInfo sfaRunInfo[3];
    RunParamStr sfaRunParam;
    int64_t sfaMultiCoreInnerIdx = 1;

    for (int64_t sfaBnIdx = sfaBN2StartIdx; sfaBnIdx < sfaBN2EndIdx; sfaBnIdx++) {
        bool sfaLastBN = (sfaBnIdx == sfaBN2EndIdx - 1);
        sfaRunParam.sfaBoIdx = sfaBnIdx;
        sfaRunParam.sfaN2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(sfaRunParam, this->sfaConstInfo,
            this->sfaActualSeqQlenAddr, this->sfaActualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(sfaRunParam, this->sfaConstInfo, sfaLastBN, sfaNextGs1Idx, sfaGS1StartIdx);

        int64_t sfaGS1LoopEnd = sfaLastBN ? (sfaRunParam.gs1LoopEndIdx + PRELOAD_NUM) : sfaRunParam.gs1LoopEndIdx;
        for (int64_t sfaGS1Index = sfaRunParam.gs1LoopStartIdx; sfaGS1Index < sfaGS1LoopEnd; sfaGS1Index++) {
            bool sfaNotLastTwoLoop = true;
            if (sfaLastBN) {
                int32_t sfaExtraGS1 = sfaGS1Index - sfaRunParam.gs1LoopEndIdx;
                switch (sfaExtraGS1) {
                    case 0:
                        sfaNotLastTwoLoop = false;
                        break;
                    case 1:
                        sfaNotLast = false;
                        sfaNotLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (sfaNotLastTwoLoop) {
                this->ComputeAxisIdxByBnAndGs1(sfaBnIdx, sfaGS1Index, sfaRunParam);
                bool sfaS1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(
                    sfaRunParam, this->sfaConstInfo, sfaGS1Index, this->sfaActualSeqQlenAddr);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算sfaTaskIdx+2的部分
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(sfaRunParam, this->sfaConstInfo);
                if (sfaS1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                sfaS2LoopLimit = sfaRunParam.sfaS2LoopEndIdx - 1;
                if constexpr (IS_SPLIT_G) {
                    sfaMaxS2LoopCnt -= (sfaS2LoopLimit + 1);
                }
            } else {
                sfaS2LoopLimit = 0;
            }
            for (int64_t sfaS2LoopCount = 0; sfaS2LoopCount <= sfaS2LoopLimit; ++sfaS2LoopCount) {
                if (sfaNotLastTwoLoop) {
                    RunInfo &sfaRunInfo1 = sfaRunInfo[sfaTaskId % 3];
                    this->SetRunInfo(sfaRunInfo1, sfaRunParam, sfaTaskId, sfaS2LoopCount, sfaS2LoopLimit,
                        sfaMultiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->sfaCubeBlock.IterateBmm1(this->sfaBmm1Buffers.Get(),
                            this->sfaL1RightBuffers.Get(), v0ResGmBuffers.Get(), sfaRunInfo1, this->sfaConstInfo);
                    } else {
                        this->sfaVecBlock.ProcessVec0(this->sfaL1RightBuffers.Get(), v0ResGmBuffers.Get(),
                            sfaRunInfo1, this->sfaConstInfo, 0);
                    }
                } else {
                    if ASCEND_IS_AIV {
                        if constexpr (IS_SPLIT_G) {
                            if (sfaMaxS2LoopCnt > 0) {
                                sfaMaxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
                            }
                        }
                    }
                }
                if (sfaTaskId > 0 && sfaNotLast) {
                    auto &sfaRunInfo2 = sfaRunInfo[(sfaTaskId + 2) % 3];
                    if ASCEND_IS_AIV {
                        this->sfaVecBlock.ProcessVec1(this->sfaL1RightBuffers.GetReused(),
                            this->sfaBmm1Buffers.Get(), sfaRunInfo2, this->sfaConstInfo);
                    } else {
                        RunInfo &sfaRunInfo2 = sfaRunInfo[(sfaTaskId + 2) % 3];
                        this->sfaCubeBlock.IterateBmm2(this->sfaBmm2Buffers.Get(), this->sfaL1RightBuffers,
                            this->sfaL1RightBuffers.GetReused(), sfaRunInfo2, this->sfaConstInfo);
                    }
                }
                if (sfaTaskId > 1) {
                    if ASCEND_IS_AIV {
                        RunInfo &sfaRunInfo3 = sfaRunInfo[(sfaTaskId + 1) % 3];
                        this->sfaVecBlock.ProcessVec2(this->sfaBmm2Buffers.Get(), sfaRunInfo3, this->sfaConstInfo);
                    }
                }
                ++sfaTaskId;
            }
            ++sfaMultiCoreInnerIdx;
        }
        sfaGS1StartIdx = 0;
    }
    if ASCEND_IS_AIV {
        if constexpr (IS_SPLIT_G) {
            for (int64_t sfaLoopCnt = 0; sfaLoopCnt < sfaMaxS2LoopCnt; sfaLoopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t sfaGS1Index, RunParamStr &sfaRunParam)
{
    // GS1合轴, 不切G, 只切S1
    sfaRunParam.sfaS1oIdx = sfaGS1Index * sfaRunParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        uint32_t firstHalfG = (sfaConstInfo.sfaGSize + 1) >> 1;
        sfaRunParam.sfaGoIdx = (sfaAicIdx % 2 == 0) ? 0 : firstHalfG; // N1>64场景，相邻cube核处理一个s1，第一个cube核承担前一半，第二个cube核承担后一半
    } else {
        sfaRunParam.sfaGoIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &sfaRunInfo, RunParamStr &sfaRunParam, int64_t sfaTaskId, int64_t sfaS2LoopCount,
    int64_t sfaS2LoopLimit, int64_t sfaMultiCoreInnerIdx)
{
    if (sfaS2LoopCount < sfaRunParam.kvLoopEndIdx) {
        sfaRunInfo.sfaS2StartIdx = sfaRunParam.sfaS2LineStartIdx;
        sfaRunInfo.sfaS2EndIdx = sfaRunParam.sfaS2LineEndIdx;
    }
    sfaRunInfo.sfaS2LoopCount = sfaS2LoopCount;
    if (sfaRunInfo.sfaMultiCoreInnerIdx != sfaMultiCoreInnerIdx) {
        sfaRunInfo.sfaS1oIdx = sfaRunParam.sfaS1oIdx;
        sfaRunInfo.sfaBoIdx = sfaRunParam.sfaBoIdx;
        sfaRunInfo.sfaN2oIdx = sfaRunParam.sfaN2oIdx;
        sfaRunInfo.sfaGoIdx = sfaRunParam.sfaGoIdx;
        sfaRunInfo.sfaMultiCoreInnerIdx = sfaMultiCoreInnerIdx;
        sfaRunInfo.sfaMultiCoreIdxMod2 = sfaMultiCoreInnerIdx & 1;
        sfaRunInfo.sfaMultiCoreIdxMod3 = sfaMultiCoreInnerIdx % 3;
    }

    sfaRunInfo.sfaTaskId = sfaTaskId;
    sfaRunInfo.sfaTaskIdMod2 = sfaTaskId & 1;
    sfaRunInfo.sfaTaskIdMod3 = sfaTaskId % 3;
    sfaRunInfo.sfaS2LoopLimit = sfaS2LoopLimit;

    sfaRunInfo.sfaActualS1Size = sfaRunParam.sfaActualS1Size;
    sfaRunInfo.sfaActualS2Size = sfaRunParam.sfaActualS2Size;
    sfaRunInfo.sfaAttentionOutOffset = sfaRunParam.sfaAttentionOutOffset;
    sfaRunInfo.sfaSOuterOffset = sfaRunParam.sfaSOuterOffset;
    sfaRunInfo.sfaQueryOffset = sfaRunParam.sfaTensorQOffset;
    this->ComputeBmm1Tail(sfaRunInfo, sfaRunParam);
    InitUniqueRunInfo(sfaRunParam, sfaRunInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &sfaRunParam, RunInfo &sfaRunInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(sfaRunParam, sfaRunInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo &sfaRunInfo, RunParamStr &sfaRunParam)
{
    // ------------------------S1 Base Related---------------------------
    sfaRunInfo.sfaS1RealSize = sfaRunParam.sfaS1RealSize;
    sfaRunInfo.sfaHalfS1RealSize = sfaRunParam.sfaHalfS1RealSize;
    sfaRunInfo.sfaFirstHalfS1RealSize = sfaRunParam.sfaFirstHalfS1RealSize;
    sfaRunInfo.sfaMRealSize = sfaRunParam.sfaMRealSize;
    sfaRunInfo.sfaHalfMRealSize = sfaRunParam.sfaHalfMRealSize;
    sfaRunInfo.sfaFirstHalfMRealSize = sfaRunParam.sfaFirstHalfMRealSize;

    sfaRunInfo.sfaVec2S1BaseSize = sfaRunInfo.sfaHalfS1RealSize;
    sfaRunInfo.sfaVec2MBaseSize = sfaRunInfo.sfaHalfMRealSize;

    // ------------------------S2 Base Related----------------------------
    sfaRunInfo.sfaS2RealSize = sfaConstInfo.sfaS2BaseSize;
    sfaRunInfo.sfaS2AlignedSize = sfaRunInfo.sfaS2RealSize;
    int64_t curS2LoopCnt = sfaRunInfo.sfaS2LoopCount;
    if (sfaRunInfo.sfaS2StartIdx + (curS2LoopCnt + 1) * sfaRunInfo.sfaS2RealSize > sfaRunInfo.sfaS2EndIdx) {
        sfaRunInfo.sfaS2RealSize = sfaRunInfo.sfaS2EndIdx - curS2LoopCnt * sfaRunInfo.sfaS2RealSize -
            sfaRunInfo.sfaS2StartIdx;
        sfaRunInfo.sfaS2AlignedSize = Align(sfaRunInfo.sfaS2RealSize);
    }
}
}
#endif // SPARSE_FLASH_ATTENTION_KERNEL_MLA_H