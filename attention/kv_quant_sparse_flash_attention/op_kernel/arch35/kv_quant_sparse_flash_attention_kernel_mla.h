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
 * \file kv_quant_sparse_flash_attention_kernel_mla.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_flash_attention_service_cube_mla.h"
#include "kv_quant_sparse_flash_attention_service_vector_mla.h"
#include "kv_quant_sparse_flash_attention_common_arch35.h"
#include "kv_quant_sparse_flash_attention_kvcache.h"

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
template <typename CubeBlockType, typename VecBlockType> class KvQuantSparseFlashAttentionMla {
public:
    ARGS_TRAITS;

    __aicore__ inline KvQuantSparseFlashAttentionMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale,
                                __gm__ uint8_t* valueScale, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling,
				                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *workspace, const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &qsfaRunInfo, RunParamStr &qsfaRunParam, int64_t qsfaTaskId,
        int64_t qsfaS2LoopCount, int64_t qsfaS2LoopLimit, int64_t qsfaMultiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &qsfaRunInfo, RunParamStr &qsfaRunParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t qsfaGS1Index, RunParamStr &qsfaRunParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &qsfaRunParam, RunInfo &qsfaRunInfo);

    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t qsfaBIdx);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev, uint32_t s2EndPrev);

    TPipe *pipe;

    const KvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 2;
    /* 核间通道 */
    BufferManager<BufferType::GM> gmBufferManager;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1RightBuffers;
    CVSharedParams sharedParams;
    /* GM信息 */
    __gm__ int32_t *actualSeqKvlenAddr = nullptr;
    __gm__ int32_t *actualSeqQlenAddr = nullptr;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    uint32_t usedCoreNum = 0U;

    /* workspace 空间 */
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    
    /* 核Index信息 */
    int32_t aicIdx;

    /* 切G时最大s2Loop */
    int64_t maxS2LoopCnt;

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    uint32_t crossCoreSyncBufId = 0;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale,
    __gm__ uint8_t* valueScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling,
    TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        constInfo.aivIdx = 0;
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }

    constInfo.s1BaseSize = 64;
    constInfo.s2BaseSize = 128;

    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, actualSeqLengthsQ, actualSeqLengths);
    if ASCEND_IS_AIV {
        constInfo.bSize = this->sharedParams.bSize;
        constInfo.gSize = this->sharedParams.gSize;
        constInfo.s1Size = this->sharedParams.s1Size;
        constInfo.needInit = this->sharedParams.needInit;
        constInfo.dSizeV = 512;
    }
    vecBlock.CleanOutput(attentionOut, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
        #pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, sparseIndices, blockTable, actualSeqLengthsQ, actualSeqLengths,
        workspace, tiling, tPipe); // gm设置
    this->InitCalcParamsEach();
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitCalcParamsEach()
{
    // 计算总的基本块
    maxS2LoopCnt = 0; // 所有核中最大累计s2Loop
    uint32_t qsfaTotalBaseNum = 0;
    uint32_t qsfaS1GBaseSize = constInfo.gSize;
    uint32_t qsfaActBatchS2 = 1;
    uint32_t qsfaCoreNum = GetBlockNum(); // G128时相邻两个cube核处理一个s1，qsfaCoreNum减半
    uint32_t qsfaCurrCoreIdx = aicIdx;
    if constexpr (IS_SPLIT_G) {
        qsfaCoreNum = qsfaCoreNum >> 1;
        qsfaCurrCoreIdx = qsfaCurrCoreIdx >> 1;
    }
    uint32_t qsfaActBatchS1 = 1;
    for (uint32_t qsfaBIdx = 0; qsfaBIdx < constInfo.bSize; qsfaBIdx++) {
        uint32_t qsfaActBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBIdx); // 不切S2，只关注S1
        qsfaTotalBaseNum += qsfaActBatchS1 * qsfaActBatchS2;
    }
    uint32_t qsfaAvgBaseNum = 1;
    if (qsfaTotalBaseNum > qsfaCoreNum) {
        qsfaAvgBaseNum = (qsfaTotalBaseNum + qsfaCoreNum - 1) / qsfaCoreNum;
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = ((qsfaTotalBaseNum + qsfaAvgBaseNum - 1) / qsfaAvgBaseNum) << 1;
        }
    } else {
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = qsfaTotalBaseNum << 1;
        } else {
            usedCoreNum = qsfaTotalBaseNum;
        }
    }

    if constexpr (IS_SPLIT_G) {
        maxS2LoopCnt = qsfaAvgBaseNum * (constInfo.sparseBlockCount + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }

    if (aicIdx >= usedCoreNum) {
        return;
    }
	// 计算当前核的基本块
    uint32_t qsfaAccumBaseNum = 0; // 当前累积的基本块数
    uint32_t qsfaTargetBaseNum = 0;
    uint32_t qsfaLastValidBIdx = 0;
    uint32_t qsfaLastValidactBatchS1 = 0;
    bool qsfaSetStart = false;
    qsfaTargetBaseNum = (qsfaCurrCoreIdx + 1) * qsfaAvgBaseNum; // 计算当前的目标权重
    uint32_t qsfaTargetStartBaseNum = qsfaTargetBaseNum - qsfaAvgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.bSize * constInfo.n2Size; bN2Idx++) {
        uint32_t qsfaBIdx = bN2Idx / constInfo.n2Size;
        qsfaActBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < qsfaActBatchS1; s1GIdx++) {
            qsfaAccumBaseNum += 1;
            if (!qsfaSetStart && qsfaAccumBaseNum >= qsfaTargetStartBaseNum) {
                constInfo.bN2Start = bN2Idx;
                constInfo.gS1Start = s1GIdx;
                qsfaSetStart = true;
            }
            if (qsfaAccumBaseNum >= qsfaTargetBaseNum) {
                // 更新当前核的End分核信息
                constInfo.bN2End = bN2Idx;
                constInfo.gS1End = s1GIdx;
                constInfo.s2End = 0;
                if (qsfaCurrCoreIdx != 0) {
                    GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
                }
                return;
            }
        }
        if ((qsfaActBatchS1 > 0) && (qsfaActBatchS2 > 0)) {
            qsfaLastValidBIdx = qsfaBIdx;
            qsfaLastValidactBatchS1 = qsfaActBatchS1;
        }
    }
    if (!qsfaSetStart) {
        constInfo.bN2Start = qsfaLastValidBIdx;
        constInfo.gS1Start = qsfaLastValidactBatchS1 - 1;
    }
    if (qsfaAccumBaseNum < qsfaTargetBaseNum) {
		// 更新最后一个核的End分核信息
        constInfo.bN2End = qsfaLastValidBIdx;
        constInfo.gS1End = qsfaLastValidactBatchS1 - 1;
        constInfo.s2End = 0;
        if (qsfaCurrCoreIdx != 0) {
            GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
        }
        return;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline uint64_t KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::\
    GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t qsfaBIdx)
{
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        if (qsfaBIdx > 0) {
            return actualSeqQlenAddr[qsfaBIdx] - actualSeqQlenAddr[qsfaBIdx - 1];
        } else if (qsfaBIdx == 0) {
            return actualSeqQlenAddr[0];
        } else {
            return 0;
        }
    } else {
        if (constInfo.isActualLenDimsNull == 1) {
            return constInfo.s1Size;
        } else {
            return actualSeqQlenAddr[qsfaBIdx];
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                uint32_t s1GEndPrev,
                                                                                uint32_t s2EndPrev)
{
    uint32_t qsfaBEndPrev = bN2EndPrev / constInfo.n2Size;
    uint32_t qsfaActualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBEndPrev);
    uint32_t qsfaS1GPrevBaseNum = qsfaActualSeqQPrev;
    constInfo.bN2Start = bN2EndPrev;
    constInfo.gS1Start = s1GEndPrev;
    
    constInfo.s2Start = 0;
    if (s1GEndPrev >= qsfaS1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        constInfo.gS1Start = 0;
        constInfo.bN2Start++;
    } else {
        constInfo.gS1Start++;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *workspace, const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling, TPipe *tPipe)
{
    if (actualSeqLengthsQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t *)actualSeqLengthsQ;
    }
    if (actualSeqLengths != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t *)actualSeqLengths;
    }

    vecBlock.InitGlobalBuffer(key, value, sparseIndices, blockTable);
    cubeBlock.InitCubeInput(actualSeqLengthsQ, constInfo);
}


template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitMMResBuf(
    __gm__ uint8_t *workspace)
{
    uint32_t qsfaMm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t qsfaMm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t qsfaMm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t qsfaMm1RightSize = constInfo.s2BaseSize * 576 * sizeof(Q_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    l1RightBuffers.Init(l1BufferManager, qsfaMm1RightSize);
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIC {
        l1RightBuffers.Get().SetCrossCore();
        l1RightBuffers.Get().SetCrossCore();
        l1RightBuffers.Get().SetCrossCore();
    }
    ubBufferManager.Init(pipe, qsfaMm1ResultSize * 2 + qsfaMm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, qsfaMm2ResultSize);
    bmm2Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, qsfaMm1ResultSize);
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }

    if constexpr (IS_SPLIT_G) {
        uint32_t qsfaV0ResSize = constInfo.s2BaseSize * 576U * sizeof(Q_T);
        int64_t qsfaTotalOffset = qsfaV0ResSize * 3 * (aicIdx >> 1U);
        gmBufferManager.Init(workspace + qsfaTotalOffset);
        v0ResGmBuffers.Init(gmBufferManager, qsfaV0ResSize);
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
        crossCoreSyncBufId++;
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
        crossCoreSyncBufId++;
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
        crossCoreSyncBufId++;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积
    usedCoreNum = sharedParams.usedCoreNum;

    if ASCEND_IS_AIC {
        constInfo.bSize = this->sharedParams.bSize;
        constInfo.gSize = this->sharedParams.gSize;
        constInfo.s1Size = this->sharedParams.s1Size;
        constInfo.needInit = this->sharedParams.needInit;
        constInfo.dSizeV = 512;
    }
    constInfo.n2Size = sharedParams.n2Size;
    constInfo.s2Size = sharedParams.s2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeVInput = sharedParams.dSizeVInput;
    constInfo.dSizeRope = sharedParams.dSizeRope;
    constInfo.dSizeNope = constInfo.dSize - constInfo.dSizeRope;
    constInfo.tileSize = sharedParams.tileSize;
    constInfo.sparseBlockCount = sharedParams.sparseBlockCount;
    constInfo.sparseBlockSize = 1;
    constInfo.sparseMode = sharedParams.maskMode;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = constInfo.s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    constInfo.layoutType = sharedParams.layoutType;

    constInfo.isActualLenDimsNull = sharedParams.isActualSeqLengthsNull;
    constInfo.isActualLenDimsKVNull = sharedParams.isActualSeqLengthsKVNull;

    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
    if ASCEND_IS_AIV {
        constInfo.softmaxScale = sharedParams.softmaxScale;
        constInfo.blockSize = sharedParams.blockSize;
        constInfo.maxBlockNumPerBatch = sharedParams.maxBlockNumPerBatch;
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    // bsize + 1-> bsize
    this->constInfo.actualSeqLenSize = this->sharedParams.bSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.bSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }

    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    bool qsfaHasLoad = aicIdx < usedCoreNum;
    if (!qsfaHasLoad) {
        if ASCEND_IS_AIV {
            if constexpr (IS_SPLIT_G) {
                for (int64_t qsfaLoopCnt = 0; qsfaLoopCnt < maxS2LoopCnt; qsfaLoopCnt++) {
                    CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                    CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                }
            }
        }
        return;
    }

    // 适配分核左闭右开
    uint32_t qsfaBIdx = constInfo.bN2End / constInfo.n2Size;
    uint32_t qsfaActS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBIdx);
    uint32_t qsfaGS1max = qsfaActS1Size;
    if (constInfo.gS1End + 1 < qsfaGS1max) {
        /* constInfo.gS1End != qsfaGS1max时，gS1End需要往后加一格, bN2End不变 */
        constInfo.gS1End = constInfo.gS1End + 1;
    } else { 
        /* constInfo.gS1End == qsfaGS1max，bN2End需要往后加一格，bN2End变为0，以代表末尾 */
        constInfo.bN2End = constInfo.bN2End + 1;
        constInfo.gS1End = 0;
    }

    // 分核信息
    uint32_t qsfaBN2StartIdx = constInfo.bN2Start;
    uint32_t qsfaBN2EndIdx = constInfo.bN2End;
    uint32_t qsfaGS1StartIdx = constInfo.gS1Start;
    uint32_t qsfaNextGs1Idx = constInfo.gS1End;
    uint32_t qsfaS2StartIdx = 0;
    uint32_t qsfaS2EndIdx = 0;
    uint32_t qsfaS2LoopLimit = 0;

    if (qsfaNextGs1Idx != 0) {
        qsfaBN2EndIdx++;
    }

    int64_t qsfaTaskId = 0;
    bool qsfaNotLast = true;
    RunInfo qsfaRunInfo[3];
    RunParamStr qsfaRunParam;
    int64_t qsfaMultiCoreInnerIdx = 1;
    for (int64_t qsfaBnIdx = qsfaBN2StartIdx; qsfaBnIdx < qsfaBN2EndIdx; qsfaBnIdx++) {
        bool qsfaLastBN = (qsfaBnIdx == qsfaBN2EndIdx - 1);
        qsfaRunParam.boIdx = qsfaBnIdx;
        qsfaRunParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(qsfaRunParam, this->constInfo,
            this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(qsfaRunParam, this->constInfo, qsfaLastBN,
            qsfaNextGs1Idx, qsfaGS1StartIdx);

        int64_t qsfaGS1LoopEnd = qsfaLastBN ? (qsfaRunParam.gs1LoopEndIdx + PRELOAD_NUM) : qsfaRunParam.gs1LoopEndIdx;
        for (int64_t qsfaGS1Index = qsfaRunParam.gs1LoopStartIdx; qsfaGS1Index < qsfaGS1LoopEnd; qsfaGS1Index++) {
            bool qsfaNotLastTwoLoop = true;
            if (qsfaLastBN) {
                int32_t qsfaExtraGS1 = qsfaGS1Index - qsfaRunParam.gs1LoopEndIdx;
                switch (qsfaExtraGS1) {
                    case 0:
                        qsfaNotLastTwoLoop = false;
                        break;
                    case 1:
                        qsfaNotLast = false;
                        qsfaNotLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (qsfaNotLastTwoLoop) {
                this->ComputeAxisIdxByBnAndGs1(qsfaBnIdx, qsfaGS1Index, qsfaRunParam);
                bool qsfaS1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(
                    qsfaRunParam, this->constInfo, qsfaGS1Index, this->actualSeqQlenAddr);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(qsfaRunParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算qsfaTaskIdx+2的部分
                if (qsfaS1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                if constexpr (IS_SPLIT_G) {
                    maxS2LoopCnt -= qsfaRunParam.s2LoopEndIdx;
                }
                qsfaS2LoopLimit = qsfaRunParam.s2LoopEndIdx - 1;
            } else {
                qsfaS2LoopLimit = 0;
            }

            for (int64_t qsfaS2LoopCount = 0; qsfaS2LoopCount <= qsfaS2LoopLimit; ++qsfaS2LoopCount) {
                if (qsfaNotLastTwoLoop) {
                    RunInfo &qsfaRunInfo1 = qsfaRunInfo[qsfaTaskId % 3];
                    this->SetRunInfo(qsfaRunInfo1, qsfaRunParam, qsfaTaskId, qsfaS2LoopCount, qsfaS2LoopLimit,
                        qsfaMultiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), this->l1RightBuffers.Get(),
                        this->v0ResGmBuffers.Get(), qsfaRunInfo1, this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(), this->v0ResGmBuffers.Get(),
                        qsfaRunInfo1, this->constInfo);
                    }
                } else {
                    if ASCEND_IS_AIV {
                        if constexpr (IS_SPLIT_G) {
                            if (maxS2LoopCnt > 0) {
                                maxS2LoopCnt--;
                                CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                                CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                            }
                        }
                    }
                }
                if (qsfaTaskId > 0 && qsfaNotLast) {
                    auto &qsfaRunInfo2 = qsfaRunInfo[(qsfaTaskId + 2) % 3];
                    if ASCEND_IS_AIV {
                        this->vecBlock.ProcessVec1(this->l1RightBuffers.GetReused(), this->bmm1Buffers.Get(),
                            qsfaRunInfo2, this->constInfo);
                    } else {
                        RunInfo &qsfaRunInfo2 = qsfaRunInfo[(qsfaTaskId + 2) % 3];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1RightBuffers,
                            this->l1RightBuffers.GetReused(), qsfaRunInfo2, this->constInfo);
                    }
                }
                if (qsfaTaskId > 1) {
                    if ASCEND_IS_AIV {
                        RunInfo &qsfaRunInfo3 = qsfaRunInfo[(qsfaTaskId + 1) % 3];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), qsfaRunInfo3, this->constInfo);
                    }
                }
                ++qsfaTaskId;
            }
            ++qsfaMultiCoreInnerIdx;
        }
        qsfaGS1StartIdx = 0;
    }
    if ASCEND_IS_AIV {
        if constexpr (IS_SPLIT_G) {
            for (int64_t qsfaLoopCnt = 0; qsfaLoopCnt < maxS2LoopCnt; qsfaLoopCnt++) {
                CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t qsfaGS1Index, RunParamStr &qsfaRunParam)
{
    // GS1合轴, 不切G, 只切S1
    qsfaRunParam.s1oIdx = qsfaGS1Index * qsfaRunParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        qsfaRunParam.goIdx = (aicIdx % 2 == 0) ? 0 : 64; // N1=128场景，相邻cube核处理一个s1，第一个cube核承担0-63行g，第二个cube核承担后64行g
    } else {
        qsfaRunParam.goIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &qsfaRunInfo, RunParamStr &qsfaRunParam, int64_t qsfaTaskId, int64_t qsfaS2LoopCount,
    int64_t qsfaS2LoopLimit, int64_t qsfaMultiCoreInnerIdx)
{
    if (qsfaS2LoopCount < qsfaRunParam.kvLoopEndIdx) {
        qsfaRunInfo.qsfaS2StartIdx = qsfaRunParam.s2LineStartIdx;
        qsfaRunInfo.qsfaS2EndIdx = qsfaRunParam.s2LineEndIdx;
    }
    qsfaRunInfo.qsfaS2LoopCount = qsfaS2LoopCount;
    if (qsfaRunInfo.qsfaMultiCoreInnerIdx != qsfaMultiCoreInnerIdx) {
        qsfaRunInfo.s1oIdx = qsfaRunParam.s1oIdx;
        qsfaRunInfo.boIdx = qsfaRunParam.boIdx;
        qsfaRunInfo.n2oIdx = qsfaRunParam.n2oIdx;
        qsfaRunInfo.goIdx = qsfaRunParam.goIdx;
        qsfaRunInfo.qsfaMultiCoreInnerIdx = qsfaMultiCoreInnerIdx;
        qsfaRunInfo.multiCoreIdxMod2 = qsfaMultiCoreInnerIdx & 1;
        qsfaRunInfo.multiCoreIdxMod3 = qsfaMultiCoreInnerIdx % 3;
    }

    qsfaRunInfo.qsfaTaskId = qsfaTaskId;
    qsfaRunInfo.qsfaTaskIdMod2 = qsfaTaskId & 1;
    qsfaRunInfo.qsfaTaskIdMod3 = qsfaTaskId % 3;
    qsfaRunInfo.qsfaS2LoopLimit = qsfaS2LoopLimit;

    qsfaRunInfo.actualS1Size = qsfaRunParam.actualS1Size;
    qsfaRunInfo.actualS2Size = qsfaRunParam.actualS2Size;
    qsfaRunInfo.attentionOutOffset = qsfaRunParam.attentionOutOffset;
    qsfaRunInfo.sOuterOffset = qsfaRunParam.sOuterOffset;
    this->ComputeBmm1Tail(qsfaRunInfo, qsfaRunParam);
    InitUniqueRunInfo(qsfaRunParam, qsfaRunInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &qsfaRunParam, RunInfo &qsfaRunInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(qsfaRunParam, qsfaRunInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo &qsfaRunInfo, RunParamStr &qsfaRunParam)
{
    // ------------------------S1 Base Related---------------------------
    qsfaRunInfo.s1RealSize = qsfaRunParam.s1RealSize;
    qsfaRunInfo.halfS1RealSize = qsfaRunParam.halfS1RealSize;
    qsfaRunInfo.firstHalfS1RealSize = qsfaRunParam.firstHalfS1RealSize;
    qsfaRunInfo.mRealSize = qsfaRunParam.mRealSize;
    qsfaRunInfo.halfMRealSize = qsfaRunParam.halfMRealSize;
    qsfaRunInfo.firstHalfMRealSize = qsfaRunParam.firstHalfMRealSize;

    qsfaRunInfo.vec2S1BaseSize = qsfaRunInfo.halfS1RealSize;
    qsfaRunInfo.vec2MBaseSize = qsfaRunInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    qsfaRunInfo.s2RealSize = constInfo.s2BaseSize;
    qsfaRunInfo.s2AlignedSize = qsfaRunInfo.s2RealSize;
    if (qsfaRunInfo.qsfaS2StartIdx + (qsfaRunInfo.qsfaS2LoopCount + 1) * qsfaRunInfo.s2RealSize > \
        qsfaRunInfo.qsfaS2EndIdx) {
        qsfaRunInfo.s2RealSize = qsfaRunInfo.qsfaS2EndIdx - qsfaRunInfo.qsfaS2LoopCount * qsfaRunInfo.s2RealSize -
            qsfaRunInfo.qsfaS2StartIdx;
        qsfaRunInfo.s2AlignedSize = Align(qsfaRunInfo.s2RealSize);
    }
}
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H