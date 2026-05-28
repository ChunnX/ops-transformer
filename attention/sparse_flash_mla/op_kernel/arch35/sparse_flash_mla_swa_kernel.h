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
 * \file sparse_flash_mla_swa_kernel.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_SWA_KERNEL_H
#define SPARSE_FLASH_MLA_SWA_KERNEL_H
#include "sparse_flash_mla_common_arch35.h"
#include "sparse_flash_mla_kvcache.h"
#include "sparse_flash_mla_scfa_block_cube.h"
#include "sparse_flash_mla_scfa_block_vector.h"
#include "kernel_operator.h"
#include "../sparse_flash_mla_metadata.h"

// 线上编包
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"

#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace SMLAKernel {
template <typename CubeBlockType, typename VecBlockType>
class SparseFlashMlaSwa {
public:
    ARGS_TRAITS;
    __aicore__ inline SparseFlashMlaSwa() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
        __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
        __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
        __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
        __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
        const SparseFlashMlaTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void ParseTilingData(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
        __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
        __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedQ,
        __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
        const SparseFlashMlaTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo);
    TPipe *pipe;

    const SparseFlashMlaTilingData *__restrict tilingData;
    /* 编译期常量的基本块信息 */
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 2;
    /* 核间通道 */
    BufferManager<BufferType::GM> gmBufferManager;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::INNER_CORE_SYNC> l1RightBuffers;
    /* GM信息 */
    GlobalTensor<uint32_t> metadataGm;
    BuffersPolicy3buff<BufferType::GM> s2LenGmBuffers;
    __gm__ int32_t *cuSeqlensQAddr = nullptr;
    __gm__ int32_t *cuSeqlensOriKvAddr = nullptr;
    __gm__ int32_t *cuSeqlensCmpKvAddr = nullptr;
    __gm__ int32_t *actualSeqKvlenAddr = nullptr;
    __gm__ int32_t *actualSeqQlenAddr = nullptr;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const SparseFlashMlaTilingData *__restrict tiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        constInfo.aivIdx = 0;
        this->tilingData = tiling;
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }

    if (metadata == nullptr) {
        return;
    }
    this->metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);

    constInfo.s1BaseSize = 64;
    constInfo.s2BaseSize = 128;
    constInfo.hasOriTopkLength = (oriTopkLength != nullptr);
    constInfo.hasCmpTopkLength = (cmpTopkLength != nullptr);

    this->pipe = tPipe;
    this->ParseTilingData(cuSeqlensQ, cuSeqlensOriKv, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    vecBlock.InitVecBlock(tPipe, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    vecBlock.CleanOutput(attentionOut, softmaxLse,constInfo);
    InitMMResBuf();
    cubeBlock.InitCubeBlock(pipe, l1BufferManager, query);
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
        cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv, sinks, workspace, tiling, tPipe); // gm设置
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    auto &sparseFlashMlaBaseParams = this->tilingData->baseParams;
    auto &sparseFlashMlaCmpParams = this->tilingData->cmpParams;
    constInfo.bSize = sparseFlashMlaBaseParams.batchSize;
    constInfo.n2Size = 1;
    constInfo.gSize = sparseFlashMlaBaseParams.nNumOfQInOneGroup; 
    constInfo.s1Size = sparseFlashMlaBaseParams.qSeqSize;
    constInfo.s2Size = sparseFlashMlaBaseParams.kvSeqSize;
    constInfo.cmpS2Size = sparseFlashMlaCmpParams.cmpKvSeqSize;
    constInfo.oriSparseBlockCount = sparseFlashMlaBaseParams.oriSparseBlockCount;
    constInfo.cmpSparseBlockCount = sparseFlashMlaCmpParams.cmpSparseBlockCount;
    constInfo.cmpRatio = sparseFlashMlaCmpParams.cmpRatio;
    constInfo.oriMaskMode = sparseFlashMlaBaseParams.oriMaskMode;
    constInfo.cmpMaskMode = sparseFlashMlaCmpParams.cmpMaskMode;
    constInfo.oriWinLeft = sparseFlashMlaBaseParams.oriWinLeft;
    constInfo.oriWinRight = sparseFlashMlaBaseParams.oriWinRight;
    constInfo.layoutType = sparseFlashMlaBaseParams.outputLayout; 
    constInfo.returnSoftmaxLse = sparseFlashMlaBaseParams.returnSoftmaxLse;
    constInfo.tileSize = 0;
    constInfo.dSizeRope = 64;
    constInfo.softmaxScale = sparseFlashMlaBaseParams.softmaxScale; 
    constInfo.dSize = 512;
    constInfo.dSizeV = constInfo.dSize;
    constInfo.dSizeVInput = constInfo.dSize;
    constInfo.dSizeNope = constInfo.dSize - constInfo.dSizeRope;
    constInfo.sparseBlockSize = 1;
    constInfo.actualSeqLenSize = constInfo.bSize + 1;
    constInfo.actualSeqLenKVSize = constInfo.bSize;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        this->constInfo.isActualLenDimsKVNull = 0U;
    } else {
        this->constInfo.isActualLenDimsKVNull = (sequsedOriKv == nullptr);
    }

    if constexpr (IS_PA) {
        constInfo.oriBlockSize = sparseFlashMlaBaseParams.oriBlockSize;
        constInfo.cmpBlockSize = sparseFlashMlaBaseParams.cmpBlockSize;
        constInfo.oriMaxBlockNumPerBatch = sparseFlashMlaBaseParams.oriMaxBlockNumPerBatch; 
        constInfo.cmpMaxBlockNumPerBatch = sparseFlashMlaCmpParams.cmpMaxBlockNumPerBatch;
    }

    constInfo.needInit = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
        int64_t s2Size;
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BNBD) {
            GlobalTensor<int32_t> actualSeqLengthsKVGm;
            actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)sequsedOriKv);
            s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
        } else if (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
            GlobalTensor<int32_t> cuSeqlensOriKvGm;
            cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            s2Size = cuSeqlensOriKvGm.GetValue(bIdx + 1) - cuSeqlensOriKvGm.GetValue(bIdx);
        } else {
            s2Size = constInfo.s2Size;
        }
        int64_t s1Size;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            GlobalTensor<int32_t> cuSeqlensQGm;
            cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            s1Size = cuSeqlensQGm.GetValue(bIdx + 1) - cuSeqlensQGm.GetValue(bIdx);
        } else {
            s1Size = constInfo.s1Size;
        }
        if (s1Size > s2Size) {
            constInfo.needInit = 1;
            break;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
    __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
    const SparseFlashMlaTilingData *__restrict tiling, TPipe *tPipe)
{
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQAddr = (__gm__ int32_t *)cuSeqlensQ;
    }
    if (cuSeqlensOriKv != nullptr) {
        cuSeqlensOriKvAddr = (__gm__ int32_t *)cuSeqlensOriKv;
    }
    if (cuSeqlensCmpKv != nullptr) {
        cuSeqlensCmpKvAddr = (__gm__ int32_t *)cuSeqlensCmpKv;
    }
    if (sequsedOriKv != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t *)sequsedOriKv;
    }
    if (sequsedQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t *)sequsedQ;
    }

    vecBlock.InitGlobalBuffer(oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ,
                              sinks);
    cubeBlock.InitCubeInput(oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ, cuSeqlensQ,
                              cuSeqlensOriKv, cuSeqlensCmpKv, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t mm1RightSize = constInfo.s2BaseSize * 512 * sizeof(Q_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    if ASCEND_IS_AIC {
        l1RightBuffers.Init(l1BufferManager, mm1RightSize);
        l1PBuffers.Get().SetCrossCore();
        l1PBuffers.Get().SetCrossCore();
    }
    ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = constInfo.s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;

    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->constInfo.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    if (hasLoad == 0) {
        return;
    }

    // 从meta data解析分核信息
    uint32_t bN2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_START_INDEX, false));
    uint32_t gS1StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_START_INDEX, false));
    uint32_t s2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_START_INDEX, false));
    uint32_t bN2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_END_INDEX, false));
    uint32_t nextGs1Idx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_END_INDEX, false));
    uint32_t s2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_END_INDEX, false));
    uint32_t s2LoopLimit = 0;

    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    int64_t taskId = 0;
    bool notLast = true;
    RunInfo runInfo[3];
    RunParamStr runParam;
    int64_t multiCoreInnerIdx = 1;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        runParam.boIdx = bnIdx;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(runParam, this->constInfo,
            this->cuSeqlensQAddr, this->cuSeqlensOriKvAddr, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, lastBN, nextGs1Idx, gS1StartIdx);

        int64_t gS1LoopEnd = lastBN ? (runParam.gs1LoopEndIdx + PRELOAD_NUM) : runParam.gs1LoopEndIdx;
        for (int64_t gS1Index = runParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.gs1LoopEndIdx;
                switch (extraGS1) {
                    case 0:
                        notLastTwoLoop = false;
                        break;
                    case 1:
                        notLast = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastTwoLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(
                    runParam, this->constInfo, gS1Index, this->cuSeqlensQAddr);
                GlobalTensor<int32_t> tmpTensor;
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(bnIdx, gS1Index, this->cuSeqlensQAddr, tmpTensor,
                        false, tmpTensor, false, runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastTwoLoop) {
                    RunInfo &runInfo1 = runInfo[taskId % 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), this->l1RightBuffers.Get(), runInfo1,
                            this->constInfo);
                    }
                }
                if (taskId > 0 && notLast) {
                    auto &runInfo2 = runInfo[(taskId + 2) % 3];
                    if ASCEND_IS_AIV {
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), s2LenGmBuffers.Get(), runInfo2,
                            this->constInfo);
                    } else {
                        RunInfo &runInfo2 = runInfo[(taskId + 2) % 3];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, this->l1RightBuffers.GetReused(), runInfo2,
                            this->constInfo);
                    }
                }
                if (taskId > 1) {
                    if ASCEND_IS_AIV {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 3];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam)
{
    // GS1合轴, 不切G, 只切S1
    runParam.s1oIdx = gS1Index * runParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        runParam.goIdx = (aicIdx % 2 == 0) ? 0 : 64;
    } else {
        runParam.goIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2LineStartIdx;
        runInfo.s2EndIdx = runParam.s2LineEndIdx;
    } else {
        runInfo.s2StartIdx = 0;
        runInfo.s2EndIdx = runParam.s2CmpLineEndIdx;
    }
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseFlashMlaSwa<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaSwa<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo &runInfo, RunParamStr &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    runInfo.firstHalfMRealSize = runParam.firstHalfMRealSize;

    runInfo.vec2MBaseSize = runInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = constInfo.s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    int64_t curS2LoopCnt = (runInfo.s2LoopCount >= runParam.oriKvLoopEndIdx) ? (runInfo.s2LoopCount - runParam.oriKvLoopEndIdx) : runInfo.s2LoopCount;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}
}
#endif // SPARSE_FLASH_MLA_SWA_KERNEL_H