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
 * \file kv_quant_sparse_attn_sharedkv_scfa_kernel.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "kv_quant_sparse_attn_sharedkv_kvcache.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_cube.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_vector.h"
#include "kernel_operator.h"
#include "../kv_quant_sparse_attn_sharedkv_metadata.h"
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

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class KvQuantSparseAttnSharedkvScfa {
public:
    ARGS_TRAITS;
    __aicore__ inline KvQuantSparseAttnSharedkvScfa() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                       __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                       __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
                                       __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
        __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks,
        __gm__ uint8_t *attentionOut);
    __aicore__ inline void InitLocalBuffer(uint32_t goIdx);
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam);

    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 3;
    /* 核间通道 */
    BufferManager<BufferType::GM> v0ResGmBufferManager;

    StaticBuffer<T> bmm1Buffers[2];
    StaticBuffer<T> bmm2Buffers;
    uint32_t bmm1GetFlag = 0;
    uint32_t vUbBase = 0;

    StaticBuffer<Q_T> l1PBuffers[2];
    uint32_t l1PGetFlag = 0;
    uint32_t l1CubeBase = 0;
    /* GM信息 */
    GlobalTensor<uint32_t> metadataGm;
    __gm__ int32_t *cuSeqlensQAddr = nullptr;
    __gm__ int32_t *actualSeqKvlenAddr = nullptr;
    __gm__ int32_t *actualSeqQlenAddr = nullptr;
    /* workspace 空间 */
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    /* 核Index信息 */
    int32_t aicIdx;
    uint32_t bN2StartIdx;
    uint32_t gS1StartIdx;
    uint32_t s2StartIdx;
    uint32_t bN2EndIdx;
    uint32_t nextGs1Idx;
    uint32_t s2EndIdx;
    uint32_t hasLoad;

    /* 核内子块索引 */
    uint8_t subBlockIdx;
    uint32_t aivIdx;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
{
    fa_base_matmul::idCounterNum = 0;
    this->subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        this->aivIdx = 0;
    } else {
        this->aivIdx = GetBlockIdx();
        this->aicIdx = this->aivIdx >> 1;
    }

    if (metadata == nullptr) {
        return;
    }
    this->metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);

    bN2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_START_INDEX, false));
    gS1StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_START_INDEX, false));
    s2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_START_INDEX, false));
    bN2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_END_INDEX, false));
    nextGs1Idx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_END_INDEX, false));
    s2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_END_INDEX, false));
    hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    this->InitGlobalBuffer(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable,
        cuSeqlensQ, sequsedQ, sequsedKv, sinks, attentionOut);
    if ASCEND_IS_AIV {
        if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE && IS_VEC_S2PHYADDR) {
            this->vecBlock.GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx,
                actualSeqQlenAddr, cuSeqlensQAddr, actualSeqKvlenAddr, workspace, this->subBlockIdx, this->aivIdx);
        }
    }
    InitMMResBuf(workspace);
    uint32_t goIdx = 0;
    if constexpr (IS_SPLIT_G) {
        goIdx = (aicIdx % 2 == 0) ? 0 : 64;
    }
    this->InitLocalBuffer(goIdx);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *attentionOut)
{
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQAddr = (__gm__ int32_t *)cuSeqlensQ;
    }
    if (sequsedKv != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t *)sequsedKv;
    }

    if (sequsedQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t *)sequsedQ;
    }

    vecBlock.InitGlobalBuffer(oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable,
        cuSeqlensQ, sequsedKv, sinks, attentionOut);

    cubeBlock.InitGlobalBuffer(query, cuSeqlensQ);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t *workspace)
{
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(Q_T);
    uint32_t l1PAddr = 0;
    l1PBuffers[0] = {LocalTensor<Q_T>(TPosition::A1, l1PAddr, mm2LeftSize / sizeof(Q_T)), 0};
    l1PAddr += mm2LeftSize;
    l1PBuffers[1] = {LocalTensor<Q_T>(TPosition::A1, l1PAddr, mm2LeftSize / sizeof(Q_T)), 1};
    l1PAddr += mm2LeftSize;
    l1CubeBase = l1PAddr;

    uint32_t ubAddr = 0;
    bmm2Buffers = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm2ResultSize / sizeof(T)), 0};
    ubAddr += mm2ResultSize;
    bmm1Buffers[0] = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm1ResultSize / sizeof(T)), 0};
    ubAddr += mm1ResultSize;
    bmm1Buffers[1] = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm1ResultSize / sizeof(T)), 1};
    ubAddr += mm1ResultSize;
    vUbBase = ubAddr;

    if ASCEND_IS_AIV {
        CrossCoreSetFlag<4, PIPE_V>(CROSSCORE_BMM1(bmm1Buffers[0].idx));
        CrossCoreSetFlag<4, PIPE_V>(CROSSCORE_BMM1(bmm1Buffers[1].idx));
        CrossCoreSetFlag<4, PIPE_V>(CROSSCORE_BMM2);
    }

    uint32_t v0ResSize = s2BaseSize * 512U * sizeof(Q_T);
    int64_t totalOffset = 0;
    if constexpr (IS_SPLIT_G) {
        totalOffset = v0ResSize * 3 * (aicIdx >> 1U);
    } else {
        totalOffset = v0ResSize * 3 * aicIdx;
    }
    v0ResGmBufferManager.Init(workspace + totalOffset);
    v0ResGmBuffers.Init(v0ResGmBufferManager, v0ResSize);
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitLocalBuffer(uint32_t goIdx)
{
    cubeBlock.InitLocalBuffer(l1CubeBase, goIdx);
    vecBlock.InitLocalBuffer(vUbBase);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Process()
{
    ICachePreLoad(6);

    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int64_t maxS2LoopCnt = 0;
    if constexpr (IS_SPLIT_G) {
        maxS2LoopCnt = static_cast<int64_t>(metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_MAX_NUM, false)));
    }
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<0, PIPE_MTE2>(10);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(10);
                }
            }
        }
        return;
    }
    uint32_t s2LoopLimit = 0;

    int64_t taskId = 0;
    bool isFirstLoop = true;
    bool notLast = true;
    bool notLastTwoLoop = true;
    RunInfo runInfo[4];
    RunParamStr runParam;
    int64_t multiCoreInnerIdx = 1;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        runParam.boIdx = bnIdx;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(runParam,
            this->cuSeqlensQAddr, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(runParam, lastBN, nextGs1Idx, gS1StartIdx);

        int64_t gS1LoopEnd = lastBN ? (runParam.gs1LoopEndIdx + PRELOAD_NUM) : runParam.gs1LoopEndIdx;
        for (int64_t gS1Index = runParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool notLastThreeLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.gs1LoopEndIdx;
                switch (extraGS1) {
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastTwoLoop = false;
                        notLastThreeLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastTwoLoop = false;
                        notLastThreeLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(
                    runParam, this->subBlockIdx, gS1Index, this->cuSeqlensQAddr);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                if constexpr (IS_SPLIT_G) {
                    maxS2LoopCnt -= runParam.s2LoopEndIdx;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastThreeLoop) {
                    RunInfo &runInfo1 = runInfo[taskId % 4];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                }
                if ASCEND_IS_AIV {
                    if (notLastThreeLoop) {
                        RunInfo &runInfo1 = runInfo[taskId % 4];
                        this->vecBlock.ProcessVec0(
                            v0ResGmBuffers.Get(runInfo1.taskIdMod3), this->aivIdx,
                            runInfo1.s2LoopCount, runInfo1.s2RealSize, runInfo1.s2StartIdx, runInfo1.s1oIdx, runInfo1.oriKvLoopEndIdx);
                    }
                    if (taskId > 1 && notLast) {
                        uint32_t bmm1Slot = bmm1GetFlag;
                        bmm1GetFlag ^= 1;
                        uint32_t l1PSlot = l1PGetFlag;
                        l1PGetFlag ^= 1;
                        auto &runInfo2 = runInfo[(taskId + 2) % 4];
                        this->vecBlock.ProcessVec1(this->l1PBuffers[l1PSlot], this->bmm1Buffers[bmm1Slot],
                            this->subBlockIdx, this->aivIdx,
                            runInfo2.s2LoopCount, runInfo2.s2RealSize,
                            runInfo2.taskIdMod2, runInfo2.multiCoreIdxMod2);
                    }
                    if (taskId > 2) {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 4];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers,
                            runInfo3.s2LoopCount, runInfo3.s2LoopLimit,
                            runInfo3.taskIdMod2, runInfo3.multiCoreIdxMod2, runInfo3.attentionOutOffset);
                    }
                } else {
                    if (taskId > 0 && notLastTwoLoop) {
                        RunInfo &runInfo1 = runInfo[(taskId + 3) % 4];
                        this->cubeBlock.IterateLoadQK(
                            v0ResGmBuffers.Get(runInfo1.taskIdMod3), isFirstLoop,
                            runInfo1.taskIdMod3, runInfo1.s2RealSize, runInfo1.s1oIdx);
                        isFirstLoop = false;
                    } else {
                        if constexpr (IS_SPLIT_G) {
                            if (taskId > 0 && maxS2LoopCnt > 0) {
                                maxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE2>(10);
                                CrossCoreWaitFlag<0, PIPE_MTE2>(10);
                            }
                        }
                    }
                    if (taskId > 1 && notLast) {
                        uint32_t bmm1Slot = bmm1GetFlag;
                        bmm1GetFlag ^= 1;
                        auto &runInfo2 = runInfo[(taskId + 2) % 4];
                        RunInfo &runInfoNext = runInfo[(taskId + 3) % 4];
                        this->cubeBlock.IterateBmm1(
                            this->bmm1Buffers[bmm1Slot],
                                v0ResGmBuffers.Get(runInfo2.taskIdMod3),
                            notLastTwoLoop, runInfoNext.s1oIdx,
                            runInfo2.taskIdMod3, runInfo2.s2RealSize, runInfo2.s2LoopCount, runInfo2.s2LoopLimit);
                    }
                    if (taskId > 2) {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 4];
                        uint32_t l1PSlot = l1PGetFlag;
                        l1PGetFlag ^= 1;
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers, this->l1PBuffers[l1PSlot],
                            runInfo3.taskIdMod3, runInfo3.s2RealSize);
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
    if ASCEND_IS_AIC {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(10);
                CrossCoreWaitFlag<0, PIPE_MTE2>(10);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
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
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    int64_t s2EndIdx = 0;
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2LineStartIdx;
        s2EndIdx = runParam.s2LineEndIdx;
    } else {
        runInfo.s2StartIdx = 0;
        s2EndIdx = runParam.s2CmpLineEndIdx;
    }
    runInfo.s2LoopCount = s2LoopCount;
    runInfo.s2LoopLimit = s2LoopLimit;
    runInfo.s1oIdx = runParam.s1oIdx;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.oriKvLoopEndIdx = runParam.oriKvLoopEndIdx;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;

    runInfo.s2RealSize = s2BaseSize;
    int64_t curS2LoopCnt = (s2LoopCount >= runParam.oriKvLoopEndIdx) ? \
        (s2LoopCount - runParam.oriKvLoopEndIdx) : s2LoopCount;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > s2EndIdx) {
        runInfo.s2RealSize = s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
    }
}
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H