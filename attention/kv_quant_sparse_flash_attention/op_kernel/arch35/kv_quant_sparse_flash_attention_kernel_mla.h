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
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ
         __gm__ uint8_t *actualSeqLengths,
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
    BufferManager<BufferType::GM> qsfaGmBufferManager;

    BufferManager<BufferType::UB> qsfaUbBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> qsfaBmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> qsfaBmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> qsfaL1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> qsfaL1RightBuffers;
    CVSharedParams qsfaSharedParams;
    /* GM信息 */
    __gm__ int32_t *qsfaActualSeqKvlenAddr = nullptr;
    __gm__ int32_t *qsfaActualSeqQlenAddr = nullptr;

    GlobalTensor<int32_t> qsfaActualSeqLengthsQGm;
    uint32_t qsfaUsedCoreNum = 0U;

    /* workspace 空间 */
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    
    /* 核Index信息 */
    int32_t qsfaAicIdx;

    /* 切G时最大s2Loop */
    int64_t qsfaMaxS2LoopCnt;

    /* 初始化后不变的信息 */
    ConstInfo qsfaConstInfo;

    /* 模板库Block */
    CubeBlockType qsfaCubeBlock;
    VecBlockType qsfaVecBlock;

    uint32_t qsfaCrossCoreSyncBufId = 0;
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
    qsfaConstInfo.qsfaSubBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->qsfaAicIdx = GetBlockIdx();
        qsfaConstInfo.qsfaAivIdx = 0;
    } else {
        qsfaConstInfo.qsfaAivIdx = GetBlockIdx();
        this->qsfaAicIdx = qsfaConstInfo.qsfaAivIdx >> 1;
        this->tilingData = tiling;
    }

    qsfaConstInfo.qsfaS1BaseSize = 64;
    qsfaConstInfo.qsfaS2BaseSize = 128;

    this->pipe = tPipe;
    qsfaVecBlock.InitVecBlock(
        tPipe, this->tilingData, this->qsfaSharedParams, this->qsfaAicIdx, qsfaConstInfo.qsfaSubBlockIdx,
        actualSeqLengthsQ, actualSeqLengths);
    )
    if ASCEND_IS_AIV {
        qsfaConstInfo.qsfaBSize = this->qsfaSharedParams.qsfaBSize;
        qsfaConstInfo.qsfaGSize = this->qsfaSharedParams.qsfaGSize;
        qsfaConstInfo.qsfaS1Size = this->qsfaSharedParams.qsfaS1Size;
        qsfaConstInfo.qsfaNeedInit = this->qsfaSharedParams.qsfaNeedInit;
        qsfaConstInfo.qsfaDSizeV = 512;
    }
    qsfaVecBlock.CleanOutput(attentionOut, qsfaConstInfo);
    /* cube侧不依赖qsfaSharedParams的scalar前置 */
    InitMMResBuf(workspace);
    if ASCEND_IS_AIC {
        qsfaCubeBlock.InitCubeBlock(pipe, &qsfaL1BufferManager, query);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&qsfaSharedParams);
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
    qsfaMaxS2LoopCnt = 0; // 所有核中最大累计s2Loop
    uint32_t qsfaTotalBaseNum = 0;
    uint32_t qsfaS1GBaseSize = qsfaConstInfo.qsfaGSize;
    uint32_t qsfaActBatchS2 = 1;
    uint32_t qsfaCoreNum = GetBlockNum(); // G128时相邻两个cube核处理一个s1，qsfaCoreNum减半
    uint32_t qsfaCurrCoreIdx = qsfaAicIdx;
    if constexpr (IS_SPLIT_G) {
        qsfaCoreNum = qsfaCoreNum >> 1;
        qsfaCurrCoreIdx = qsfaCurrCoreIdx >> 1;
    }
    uint32_t qsfaActBatchS1 = 1;
    for (uint32_t qsfaBIdx = 0; qsfaBIdx < qsfaConstInfo.qsfaBSize; qsfaBIdx++) {
        uint32_t qsfaActBatchS1 = GetBalanceActualSeqLengths(qsfaActualSeqLengthsQGm, qsfaBIdx); // 不切S2，只关注S1
        qsfaTotalBaseNum += qsfaActBatchS1 * qsfaActBatchS2;
    }
    uint32_t qsfaAvgBaseNum = 1;
    if (qsfaTotalBaseNum > qsfaCoreNum) {
        qsfaAvgBaseNum = (qsfaTotalBaseNum + qsfaCoreNum - 1) / qsfaCoreNum;
        if constexpr (IS_SPLIT_G) {
            qsfaUsedCoreNum = ((qsfaTotalBaseNum + qsfaAvgBaseNum - 1) / qsfaAvgBaseNum) << 1;
        }
    } else {
        if constexpr (IS_SPLIT_G) {
            qsfaUsedCoreNum = qsfaTotalBaseNum << 1;
        } else {
            qsfaUsedCoreNum = qsfaTotalBaseNum;
        }
    }

        qsfaMaxS2LoopCnt = qsfaAvgBaseNum * 
            (qsfaConstInfo.qsfaSparseBlockCount + qsfaConstInfo.qsfaS2BaseSize - 1) / qsfaConstInfo.qsfaS2BaseSize;
        qsfaMaxS2LoopCnt = qsfaAvgBaseNum * 
            (qsfaConstInfo.qsfaSparseBlockCount + qsfaConstInfo.qsfaS2BaseSize - 1) / qsfaConstInfo.qsfaS2BaseSize;
    }

    if (qsfaAicIdx >= qsfaUsedCoreNum) {
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
    for (uint32_t bN2Idx = 0; bN2Idx < qsfaConstInfo.qsfaBSize * qsfaConstInfo.qsfaN2Size; bN2Idx++) {
        uint32_t qsfaBIdx = bN2Idx / qsfaConstInfo.qsfaN2Size;
        qsfaActBatchS1 = GetBalanceActualSeqLengths(qsfaActualSeqLengthsQGm, qsfaBIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < qsfaActBatchS1; s1GIdx++) {
            qsfaAccumBaseNum += 1;
            if (!qsfaSetStart && qsfaAccumBaseNum >= qsfaTargetStartBaseNum) {
                qsfaConstInfo.qsfaBN2Start = bN2Idx;
                qsfaConstInfo.qsfaGS1Start = s1GIdx;
                qsfaSetStart = true;
            }
            if (qsfaAccumBaseNum >= qsfaTargetBaseNum) {
                // 更新当前核的End分核信息
                qsfaConstInfo.qsfaBN2End = bN2Idx;
                qsfaConstInfo.qsfaGS1End = s1GIdx;
                qsfaConstInfo.qsfaS2End = 0;
                if (qsfaCurrCoreIdx != 0) {
                    GetAxisStartIdx(qsfaConstInfo.qsfaBN2Start, qsfaConstInfo.qsfaGS1Start, 0);
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
        qsfaConstInfo.qsfaBN2Start = qsfaLastValidBIdx;
        qsfaConstInfo.qsfaGS1Start = qsfaLastValidactBatchS1 - 1;
    }
    if (qsfaAccumBaseNum < qsfaTargetBaseNum) {
		// 更新最后一个核的End分核信息
        qsfaConstInfo.qsfaBN2End = qsfaLastValidBIdx;
        qsfaConstInfo.qsfaGS1End = qsfaLastValidactBatchS1 - 1;
        qsfaConstInfo.qsfaS2End = 0;
        if (qsfaCurrCoreIdx != 0) {
            GetAxisStartIdx(qsfaConstInfo.qsfaBN2Start, qsfaConstInfo.qsfaGS1Start, 0);
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
            return qsfaActualSeqQlenAddr[qsfaBIdx] - qsfaActualSeqQlenAddr[qsfaBIdx - 1];
        } else if (qsfaBIdx == 0) {
            return qsfaActualSeqQlenAddr[0];
        } else {
            return 0;
        }
    } else {
        if (qsfaConstInfo.qsfaIsActualLenDimsNull == 1) {
            return qsfaConstInfo.qsfaS1Size;
        } else {
            return qsfaActualSeqQlenAddr[qsfaBIdx];
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                uint32_t s1GEndPrev,
                                                                                uint32_t s2EndPrev)
{
    uint32_t qsfaBEndPrev = bN2EndPrev / qsfaConstInfo.qsfaN2Size;
    uint32_t qsfaActualSeqQPrev = GetBalanceActualSeqLengths(qsfaActualSeqLengthsQGm, qsfaBEndPrev);
    uint32_t qsfaS1GPrevBaseNum = qsfaActualSeqQPrev;
    qsfaConstInfo.qsfaBN2Start = bN2EndPrev;
    qsfaConstInfo.qsfaGS1Start = s1GEndPrev;
    
    qsfaConstInfo.qsfaS2Start = 0;
    if (s1GEndPrev >= qsfaS1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        qsfaConstInfo.qsfaGS1Start = 0;
        qsfaConstInfo.qsfaBN2Start++;
    } else {
        qsfaConstInfo.qsfaGS1Start++;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *workspace, const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling, TPipe *tPipe)
{
    if (actualSeqLengthsQ != nullptr) {
        qsfaActualSeqQlenAddr = (__gm__ int32_t *)actualSeqLengthsQ;
    }
    if (actualSeqLengths != nullptr) {
        qsfaActualSeqKvlenAddr = (__gm__ int32_t *)actualSeqLengths;
    }

    qsfaVecBlock.InitGlobalBuffer(key, value, sparseIndices, blockTable);
    qsfaCubeBlock.InitCubeInput(actualSeqLengthsQ, qsfaConstInfo);
}


template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitMMResBuf(
    __gm__ uint8_t *workspace)
{
    uint32_t qsfaMm1ResultSize = qsfaConstInfo.qsfaS1BaseSize / CV_RATIO * qsfaConstInfo.qsfaS2BaseSize * sizeof(T);
    uint32_t qsfaMm2ResultSize = qsfaConstInfo.qsfaS1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t qsfaMm2LeftSize = qsfaConstInfo.qsfaS1BaseSize * qsfaConstInfo.qsfaS2BaseSize * sizeof(Q_T);
    uint32_t qsfaMm1RightSize = qsfaConstInfo.qsfaS2BaseSize * 576 * sizeof(Q_T);
    qsfaL1BufferManager.Init(pipe, 524288); // 512 * 1024
    qsfaL1RightBuffers.Init(qsfaL1BufferManager, qsfaMm1RightSize);
    qsfaL1RightBuffers.Get().SetCrossCoreID(qsfaCrossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    qsfaCrossCoreSyncBufId++;
    qsfaL1RightBuffers.Get().SetCrossCoreID(qsfaCrossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    qsfaCrossCoreSyncBufId++;
    qsfaL1RightBuffers.Get().SetCrossCoreID(qsfaCrossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    qsfaCrossCoreSyncBufId++;
    if ASCEND_IS_AIC {
        qsfaL1RightBuffers.Get().SetCrossCore();
        qsfaL1RightBuffers.Get().SetCrossCore();
        qsfaL1RightBuffers.Get().SetCrossCore();
    }
    qsfaUbBufferManager.Init(pipe, qsfaMm1ResultSize * 2 + qsfaMm2ResultSize);
    qsfaBmm2Buffers.Init(qsfaUbBufferManager, qsfaMm2ResultSize);
    qsfaBmm2Buffers.Get().SetCrossCoreID(qsfaCrossCoreSyncBufId, qsfaCrossCoreSyncBufId);
    qsfaCrossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        qsfaBmm2Buffers.Get().SetCrossCore();
    }
    qsfaBmm1Buffers.Init(qsfaUbBufferManager, qsfaMm1ResultSize);
    qsfaBmm1Buffers.Get().SetCrossCoreID(qsfaCrossCoreSyncBufId, qsfaCrossCoreSyncBufId);
    qsfaCrossCoreSyncBufId++;
    qsfaBmm1Buffers.Get().SetCrossCoreID(qsfaCrossCoreSyncBufId, qsfaCrossCoreSyncBufId);
    qsfaCrossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        qsfaBmm1Buffers.Get().SetCrossCore();
        qsfaBmm1Buffers.Get().SetCrossCore();
    }

    if constexpr (IS_SPLIT_G) {
        uint32_t qsfaV0ResSize = qsfaConstInfo.qsfaS2BaseSize * 576U * sizeof(Q_T);
        int64_t qsfaTotalOffset = qsfaV0ResSize * 3 * (qsfaAicIdx >> 1U);
        qsfaGmBufferManager.Init(workspace + qsfaTotalOffset);
        v0ResGmBuffers.Init(qsfaGmBufferManager, qsfaV0ResSize);
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, qsfaCrossCoreSyncBufId);
        qsfaCrossCoreSyncBufId++;
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, qsfaCrossCoreSyncBufId);
        qsfaCrossCoreSyncBufId++;
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, qsfaCrossCoreSyncBufId);
        qsfaCrossCoreSyncBufId++;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    qsfaVecBlock.InitLocalBuffer(pipe, qsfaConstInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积
    qsfaUsedCoreNum = qsfaSharedParams.qsfaUsedCoreNum;

    if ASCEND_IS_AIC {
        qsfaConstInfo.qsfaBSize = this->qsfaSharedParams.qsfaBSize;
        qsfaConstInfo.qsfaGSize = this->qsfaSharedParams.qsfaGSize;
        qsfaConstInfo.qsfaS1Size = this->qsfaSharedParams.qsfaS1Size;
        qsfaConstInfo.qsfaNeedInit = this->qsfaSharedParams.qsfaNeedInit;
        qsfaConstInfo.qsfaDSizeV = 512;
    }
    qsfaConstInfo.qsfaN2Size = qsfaSharedParams.qsfaN2Size;
    qsfaConstInfo.qsfaS2Size = qsfaSharedParams.qsfaS2Size;
    qsfaConstInfo.qsfaDSize = qsfaSharedParams.qsfaDSize;
    qsfaConstInfo.qsfaDSizeVInput = qsfaSharedParams.qsfaDSizeVInput;
    qsfaConstInfo.qsfaDSizeRope = qsfaSharedParams.qsfaDSizeRope;
    qsfaConstInfo.qsfaDSizeNope = qsfaConstInfo.qsfaDSize - qsfaConstInfo.qsfaDSizeRope;
    qsfaConstInfo.qsfaTileSize = qsfaSharedParams.qsfaTileSize;
    qsfaConstInfo.qsfaSparseBlockCount = qsfaSharedParams.qsfaSparseBlockCount;
    qsfaConstInfo.qsfaSparseBlockSize = 1;
    qsfaConstInfo.qsfaSparseMode = qsfaSharedParams.qsfaMaskMode;
    qsfaConstInfo.qsfaS1S2 = qsfaConstInfo.qsfaS1Size * qsfaConstInfo.qsfaS2Size;
    qsfaConstInfo.qsfaGS1 = qsfaConstInfo.qsfaGSize * qsfaConstInfo.qsfaS1Size;
    qsfaConstInfo.qsfaN2G = qsfaConstInfo.qsfaN2Size * qsfaConstInfo.qsfaGSize;
    qsfaConstInfo.qsfaGD = qsfaConstInfo.qsfaGSize * qsfaConstInfo.qsfaDSize;
    qsfaConstInfo.qsfaN2GD = qsfaConstInfo.qsfaN2Size * qsfaConstInfo.qsfaGD;

    qsfaConstInfo.qsfaS1Dv = qsfaConstInfo.qsfaS1Size * qsfaConstInfo.qsfaDSizeV;
    qsfaConstInfo.qsfaS2Dv = qsfaConstInfo.qsfaS2Size * qsfaConstInfo.qsfaDSizeV;
    qsfaConstInfo.qsfaN2Dv = qsfaConstInfo.qsfaN2Size * qsfaConstInfo.qsfaDSizeV;
    qsfaConstInfo.qsfaGDv = qsfaConstInfo.qsfaGSize * qsfaConstInfo.qsfaDSizeV;
    qsfaConstInfo.qsfaGS1Dv = qsfaConstInfo.qsfaGSize * qsfaConstInfo.qsfaS1Dv;
    qsfaConstInfo.qsfaN2S2Dv = qsfaConstInfo.qsfaN2Size * qsfaConstInfo.qsfaS2Dv;
    qsfaConstInfo.qsfaN2GDv = qsfaConstInfo.qsfaN2Size * qsfaConstInfo.qsfaGDv;
    qsfaConstInfo.qsfaS2BaseN2Dv = qsfaConstInfo.qsfaS2BaseSize * qsfaConstInfo.qsfaN2Dv;
    qsfaConstInfo.qsfaN2GS1Dv = qsfaConstInfo.qsfaN2Size * qsfaConstInfo.qsfaGS1Dv;
    qsfaConstInfo.qsfaLayoutType = qsfaSharedParams.qsfaLayoutType;

    qsfaConstInfo.qsfaIsActualLenDimsNull = qsfaSharedParams.qsfaIsActualSeqLengthsNull;
    qsfaConstInfo.qsfaIsActualLenDimsKVNull = qsfaSharedParams.qsfaIsActualSeqLengthsKVNull;

    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        // (BS)ND
        qsfaConstInfo.qsfaS1BaseN2GDv = qsfaConstInfo.qsfaS1BaseSize * qsfaConstInfo.qsfaN2GDv;
            qsfaConstInfo.qsfaAttentionOutStride = (
                qsfaConstInfo.qsfaN2G - qsfaConstInfo.qsfaGSize) * qsfaConstInfo.qsfaDSizeV * 
                
                sizeof(OUTPUT_T);
        if ASCEND_IS_AIV {
            qsfaConstInfo.qsfaAttentionOutStride = (
                qsfaConstInfo.qsfaN2G - qsfaConstInfo.qsfaGSize) * qsfaConstInfo.qsfaDSizeV * 
                sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
        // BSH/BSNGD
            qsfaConstInfo.qsfaAttentionOutStride = (
                qsfaConstInfo.qsfaN2G - qsfaConstInfo.qsfaGSize) * qsfaConstInfo.qsfaDSizeV * 
                
                sizeof(OUTPUT_T);
        qsfaConstInfo.qsfaMm1Ka = qsfaConstInfo.qsfaN2Size * qsfaConstInfo.qsfaDSize;
        if ASCEND_IS_AIV {
            qsfaConstInfo.qsfaAttentionOutStride = (
                qsfaConstInfo.qsfaN2G - qsfaConstInfo.qsfaGSize) * qsfaConstInfo.qsfaDSizeV * 
                sizeof(OUTPUT_T);
        }
    }
    if ASCEND_IS_AIV {
        qsfaConstInfo.qsfaSoftmaxScale = qsfaSharedParams.qsfaSoftmaxScale;
        qsfaConstInfo.qsfaBlockSize = qsfaSharedParams.qsfaBlockSize;
        qsfaConstInfo.qsfaMaxBlockNumPerBatch = qsfaSharedParams.qsfaMaxBlockNumPerBatch;
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    // bsize + 1-> bsize
    this->qsfaConstInfo.qsfaActualSeqLenSize = this->qsfaSharedParams.qsfaBSize;
    this->qsfaConstInfo.qsfaActualSeqLenKVSize = this->qsfaSharedParams.qsfaBSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->qsfaSharedParams.qsfaNeedInit) {
        SyncAll<false>();
    }

    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    bool qsfaHasLoad = qsfaAicIdx < qsfaUsedCoreNum;
    if (!qsfaHasLoad) {
        if ASCEND_IS_AIV {
            if constexpr (IS_SPLIT_G) {
                for (int64_t qsfaLoopCnt = 0; qsfaLoopCnt < qsfaMaxS2LoopCnt; qsfaLoopCnt++) {
                    CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                    CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                }
            }
        }
        return;
    }

    // 适配分核左闭右开
    uint32_t qsfaBIdx = qsfaConstInfo.qsfaBN2End / qsfaConstInfo.qsfaN2Size;
    uint32_t qsfaActS1Size = GetBalanceActualSeqLengths(qsfaActualSeqLengthsQGm, qsfaBIdx);
    uint32_t qsfaGS1max = qsfaActS1Size;
    if (qsfaConstInfo.qsfaGS1End + 1 < qsfaGS1max) {
        /* qsfaConstInfo.qsfaGS1End != qsfaGS1max时，gS1End需要往后加一格, bN2End不变 */
        qsfaConstInfo.qsfaGS1End = qsfaConstInfo.qsfaGS1End + 1;
    } else { 
        /* qsfaConstInfo.qsfaGS1End == qsfaGS1max，bN2End需要往后加一格，bN2End变为0，以代表末尾 */
        qsfaConstInfo.qsfaBN2End = qsfaConstInfo.qsfaBN2End + 1;
        qsfaConstInfo.qsfaGS1End = 0;
    }

    // 分核信息
    uint32_t qsfaBN2StartIdx = qsfaConstInfo.qsfaBN2Start;
    uint32_t qsfaBN2EndIdx = qsfaConstInfo.qsfaBN2End;
    uint32_t qsfaGS1StartIdx = qsfaConstInfo.qsfaGS1Start;
    uint32_t qsfaNextGs1Idx = qsfaConstInfo.qsfaGS1End;
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
        qsfaRunParam.qsfaBoIdx = qsfaBnIdx;
        qsfaRunParam.qsfaN2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(qsfaRunParam, this->qsfaConstInfo,
            this->qsfaActualSeqQlenAddr, this->qsfaActualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(qsfaRunParam, this->qsfaConstInfo, qsfaLastBN,
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
                    qsfaRunParam, this->qsfaConstInfo, qsfaGS1Index, this->qsfaActualSeqQlenAddr);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(qsfaRunParam, this->qsfaConstInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算qsfaTaskIdx+2的部分
                if (qsfaS1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                if constexpr (IS_SPLIT_G) {
                    qsfaMaxS2LoopCnt -= qsfaRunParam.qsfaS2LoopEndIdx;
                }
                qsfaS2LoopLimit = qsfaRunParam.qsfaS2LoopEndIdx - 1;
            } else {
                qsfaS2LoopLimit = 0;
            }

            for (int64_t qsfaS2LoopCount = 0; qsfaS2LoopCount <= qsfaS2LoopLimit; ++qsfaS2LoopCount) {
                if (qsfaNotLastTwoLoop) {
                    RunInfo &qsfaRunInfo1 = qsfaRunInfo[qsfaTaskId % 3];
                    this->SetRunInfo(qsfaRunInfo1, qsfaRunParam, qsfaTaskId, qsfaS2LoopCount, qsfaS2LoopLimit,
                        qsfaMultiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->qsfaCubeBlock.IterateBmm1(this->qsfaBmm1Buffers.Get(), this->qsfaL1RightBuffers.Get(),
                        this->v0ResGmBuffers.Get(), qsfaRunInfo1, this->qsfaConstInfo);
                    } else {
                        this->qsfaVecBlock.ProcessVec0(this->qsfaL1RightBuffers.Get(), this->v0ResGmBuffers.Get(),
                        qsfaRunInfo1, this->qsfaConstInfo);
                    }
                } else {
                    if ASCEND_IS_AIV {
                        if constexpr (IS_SPLIT_G) {
                            if (qsfaMaxS2LoopCnt > 0) {
                                qsfaMaxS2LoopCnt--;
                                CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                                CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                            }
                        }
                    }
                }
                if (qsfaTaskId > 0 && qsfaNotLast) {
                    auto &qsfaRunInfo2 = qsfaRunInfo[(qsfaTaskId + 2) % 3];
                    if ASCEND_IS_AIV {
                        this->qsfaVecBlock.ProcessVec1(this->qsfaL1RightBuffers.GetReused(
                            ), this->qsfaBmm1Buffers.Get(),
                            qsfaRunInfo2, this->qsfaConstInfo);
                    } else {
                        RunInfo &qsfaRunInfo2 = qsfaRunInfo[(qsfaTaskId + 2) % 3];
                        this->qsfaCubeBlock.IterateBmm2(this->qsfaBmm2Buffers.Get(), this->qsfaL1RightBuffers,
                            this->qsfaL1RightBuffers.GetReused(), qsfaRunInfo2, this->qsfaConstInfo);
                    }
                }
                if (qsfaTaskId > 1) {
                    if ASCEND_IS_AIV {
                        RunInfo &qsfaRunInfo3 = qsfaRunInfo[(qsfaTaskId + 1) % 3];
                        this->qsfaVecBlock.ProcessVec2(this->qsfaBmm2Buffers.Get(), qsfaRunInfo3, this->qsfaConstInfo);
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
            for (int64_t qsfaLoopCnt = 0; qsfaLoopCnt < qsfaMaxS2LoopCnt; qsfaLoopCnt++) {
                CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t qsfaGS1Index, RunParamStr &qsfaRunParam)
        // N1=128场景，相邻cube核处理一个s1，第一个cube核承担0-63行g，第二个cube核承担后64行g
        qsfaRunParam.qsfaGoIdx = (qsfaAicIdx % 2 == 0) ? 0 : 64;
    // GS1合轴, 不切G, 只切S1
    qsfaRunParam.qsfaS1oIdx = qsfaGS1Index * qsfaRunParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        // N1=128场景，相邻cube核处理一个s1，第一个cube核承担0-63行g，第二个cube核承担后64行g
        qsfaRunParam.qsfaGoIdx = (qsfaAicIdx % 2 == 0) ? 0 : 64;
    } else {
        qsfaRunParam.qsfaGoIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseFlashAttentionMla<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &qsfaRunInfo, RunParamStr &qsfaRunParam, int64_t qsfaTaskId, int64_t qsfaS2LoopCount,
    int64_t qsfaS2LoopLimit, int64_t qsfaMultiCoreInnerIdx)
{
    if (qsfaS2LoopCount < qsfaRunParam.kvLoopEndIdx) {
        qsfaRunInfo.qsfaS2StartIdx = qsfaRunParam.qsfaS2LineStartIdx;
        qsfaRunInfo.qsfaS2EndIdx = qsfaRunParam.qsfaS2LineEndIdx;
    }
    qsfaRunInfo.qsfaS2LoopCount = qsfaS2LoopCount;
    if (qsfaRunInfo.qsfaMultiCoreInnerIdx != qsfaMultiCoreInnerIdx) {
        qsfaRunInfo.qsfaS1oIdx = qsfaRunParam.qsfaS1oIdx;
        qsfaRunInfo.qsfaBoIdx = qsfaRunParam.qsfaBoIdx;
        qsfaRunInfo.qsfaN2oIdx = qsfaRunParam.qsfaN2oIdx;
        qsfaRunInfo.qsfaGoIdx = qsfaRunParam.qsfaGoIdx;
        qsfaRunInfo.qsfaMultiCoreInnerIdx = qsfaMultiCoreInnerIdx;
        qsfaRunInfo.qsfaMultiCoreIdxMod2 = qsfaMultiCoreInnerIdx & 1;
        qsfaRunInfo.qsfaMultiCoreIdxMod3 = qsfaMultiCoreInnerIdx % 3;
    }

    qsfaRunInfo.qsfaTaskId = qsfaTaskId;
    qsfaRunInfo.qsfaTaskIdMod2 = qsfaTaskId & 1;
    qsfaRunInfo.qsfaTaskIdMod3 = qsfaTaskId % 3;
    qsfaRunInfo.qsfaS2LoopLimit = qsfaS2LoopLimit;

    qsfaRunInfo.qsfaActualS1Size = qsfaRunParam.qsfaActualS1Size;
    qsfaRunInfo.qsfaActualS2Size = qsfaRunParam.qsfaActualS2Size;
    qsfaRunInfo.qsfaAttentionOutOffset = qsfaRunParam.qsfaAttentionOutOffset;
    qsfaRunInfo.qsfaSOuterOffset = qsfaRunParam.qsfaSOuterOffset;
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
    qsfaRunInfo.qsfaS1RealSize = qsfaRunParam.qsfaS1RealSize;
    qsfaRunInfo.qsfaHalfS1RealSize = qsfaRunParam.qsfaHalfS1RealSize;
    qsfaRunInfo.qsfaFirstHalfS1RealSize = qsfaRunParam.qsfaFirstHalfS1RealSize;
    qsfaRunInfo.qsfaMRealSize = qsfaRunParam.qsfaMRealSize;
    qsfaRunInfo.qsfaHalfMRealSize = qsfaRunParam.qsfaHalfMRealSize;
    qsfaRunInfo.qsfaFirstHalfMRealSize = qsfaRunParam.qsfaFirstHalfMRealSize;

    qsfaRunInfo.qsfaVec2S1BaseSize = qsfaRunInfo.qsfaHalfS1RealSize;
    qsfaRunInfo.qsfaVec2MBaseSize = qsfaRunInfo.qsfaHalfMRealSize;

        qsfaRunInfo.qsfaS2RealSize = qsfaRunInfo.qsfaS2EndIdx - qsfaRunInfo.qsfaS2LoopCount * 
            qsfaRunInfo.qsfaS2RealSize -
    qsfaRunInfo.qsfaS2RealSize = qsfaConstInfo.qsfaS2BaseSize;
    qsfaRunInfo.qsfaS2AlignedSize = qsfaRunInfo.qsfaS2RealSize;
    if (qsfaRunInfo.qsfaS2StartIdx + (qsfaRunInfo.qsfaS2LoopCount + 1) * qsfaRunInfo.qsfaS2RealSize > \
        qsfaRunInfo.qsfaS2EndIdx) {
        qsfaRunInfo.qsfaS2RealSize = qsfaRunInfo.qsfaS2EndIdx - qsfaRunInfo.qsfaS2LoopCount * 
            qsfaRunInfo.qsfaS2RealSize -
            qsfaRunInfo.qsfaS2StartIdx;
        qsfaRunInfo.qsfaS2AlignedSize = Align(qsfaRunInfo.qsfaS2RealSize);
    }
}
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H