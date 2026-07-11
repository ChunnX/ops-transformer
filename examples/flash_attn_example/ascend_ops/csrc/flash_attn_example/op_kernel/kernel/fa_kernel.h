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
 * \file fa_kernel.h
 * \brief
 */

#ifndef FA_KERNEL_H
#define FA_KERNEL_H

#include "../fa_kernel_public.h"
#include "fa_block_cube.h"
#include "fa_block_vec.h"
#include "../vector/vector.h"
#include "../memcopy/memory_copy.h"
#include "./fa_block_vec_combine.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace fa_kernel::hardware;
using namespace fa_kernel::config;
using namespace fa_kernel::layout;
using namespace fa_kernel;
using namespace fa_kernel::util;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecCombineBlockType>
class FlashAttentionKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool COMBINE = VecFaBlockType::COMBINE;
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT; // V100 只支持一种??
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    BufferManager<BufferType::GM> gmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BufferPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BufferPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BufferPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BufferPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> faMetaDataGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoX constInfo;

    const NoQuantTilingArch35 *__restrict tilingData;
    TPipe *pipe = nullptr;
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecCombineBlockType vecCombineBlock;

    uint32_t coreGS1Loops = 0U;
    uint32_t accGS1Loops = 0U;
    uint32_t varlenCalcTimes = 0U;
    int64_t preTaskIdx = 0;
    int64_t curTaskIdx = 0;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t prevBIdx = 0;
    uint32_t prevBN2Idx = 0;
    uint32_t prevGS1Idx = 0;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionKernel()
        : cubeBlock(constInfo), vecFaBlock(constInfo), vecCombineBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attnMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace, __gm__ uint8_t *faMetaData,
                                const NoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tilingData = tiling;

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)faMetaData,
                                     NPU_AIC_CORE_NUM * FA_METADATA_SIZE + NPU_AIV_CORE_NUM * COMBINE_METADATA_SIZE);

        InitConstInfo();


        keyPtr = key;
        valuePtr = value;

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualSeqLenSize);

        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsKv, constInfo.actualSeqLenKVSize);

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            vecFaBlock.InitVecBlock(tPipe, actualSeqLengths, actualSeqLengthsKv, attnMask, attentionOut, workspace);
            vecFaBlock.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, actualSeqLengths, actualSeqLengthsKv);
        }

        if constexpr (COMBINE) {
            if ASCEND_IS_AIV {
                vecCombineBlock.InitParams();
                vecCombineBlock.InitGlobalTensor(this->vecFaBlock.softmaxCombineMaxGm,
                                                 this->vecFaBlock.softmaxCombineSumGm, this->vecFaBlock.accumOutGm,
                                                 this->vecFaBlock.attentionOutGm, this->actualSeqLengthsGmQ,
                                                 this->actualSeqLengthsGm, keyPtr);
            }
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);

        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager.Init(pipe, 524288); // 512 * 1024
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
        bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        auto faBaseParams = this->tilingData->faBaseParams;
        auto faAttnMaskParams = this->tilingData->faAttnMaskParams;
        auto faWorkspaceParams = this->tilingData->faWorkspaceParams;

        constInfo.bSize = faBaseParams.bSize;
        constInfo.t1Size = faBaseParams.t1Size;
        constInfo.t2Size = faBaseParams.t2Size;
        constInfo.n2Size = faBaseParams.n2Size;
        constInfo.gSize = faBaseParams.gSize;
        constInfo.s1Size = faBaseParams.s1Size;
        constInfo.s2Size = faBaseParams.s2Size;
        constInfo.dSize = faBaseParams.dSize;
        constInfo.dSizeV = faBaseParams.dSizeV;
        constInfo.actualSeqLenSize = faBaseParams.actualSeqLengthsQSize;
        constInfo.actualSeqLenKVSize = faBaseParams.actualSeqLengthsKVSize;
        constInfo.softmaxScale = static_cast<float>(faBaseParams.softmaxScale);
        constInfo.coreNum = faBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FA_LAYOUT>(faBaseParams.outputLayout);

        constInfo.sparseMode = faAttnMaskParams.sparseMode;
        constInfo.preTokens = faAttnMaskParams.preTokens;
        constInfo.nextTokens = faAttnMaskParams.nextTokens;
        constInfo.attnMaskBatch = faAttnMaskParams.attnMaskBatch;
        constInfo.attnMaskS1Size = faAttnMaskParams.attnMaskS1Size;
        constInfo.attnMaskS2Size = faAttnMaskParams.attnMaskS2Size;
        constInfo.isRowInvalidOpen = faAttnMaskParams.isRowInvalidOpen;
        constInfo.isExistRowInvalid = faAttnMaskParams.isExistRowInvalid;

        constInfo.accumOutSize = faWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = faWorkspaceParams.logSumExpSize;

        constInfo.bN2Start = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_BN2_START_INDEX));
        constInfo.gS1OStart = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_M_START_INDEX));
        constInfo.s2OStart = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_S2_START_INDEX));
        constInfo.bN2End = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_BN2_END_INDEX));
        constInfo.gS1OEnd = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_M_END_INDEX));
        constInfo.s2OEnd = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_S2_END_INDEX));
        constInfo.coreFirstTmpOutWsPos =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetCombineMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * NPU_AIC_CORE_NUM + COMBINE_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm2Buffers.Get().SetCrossCore();
            bmm2Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
            bmm2Buffers.Get().WaitCrossCore();
            bmm2Buffers.Get().WaitCrossCore();
        }
    }

    __aicore__ inline void FlashAttention()
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
        uint32_t bN2Cur = constInfo.bN2Start;
        uint32_t gS1Cur = constInfo.gS1OStart;
        uint32_t s2Cur = constInfo.s2OStart;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        uint32_t createdTaskCount = 0U;
        uint32_t executedTaskCount = 0U;
        while (shouldDispatchTask || shouldExecuteTask) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        vecFaBlock.DealZeroActSeqLen(bN2Cur);
                    }
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
            shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
            if (shouldExecuteTask) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != constInfo.bN2End) || (gS1Cur != constInfo.gS1OEnd) || (s2Cur != constInfo.s2OEnd));
    }

    __aicore__ inline bool ShouldExecuteTask(RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        for (uint32_t i = 0; i < PRELOAD_TASK_CACHE_SIZE; i++) {
            if (taskRunInfo[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart) && (s2Cur == constInfo.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            actSeqLensKv = constInfo.s2Size;
            actSeqLensQ = constInfo.s1Size;
        }

        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }

        if (curS2Start >= curS2End) {
            return TASK_DEAL_MODE::SKIP;
        }

        if (s2Cur < curS2Start) {
            return TASK_DEAL_MODE::NOT_START;
        }

        if (s2Cur >= curS2End) {
            return TASK_DEAL_MODE::S2_END;
        }

        if (s2Cur == curS2Start) {
            mloop++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo.preTokens;
        nextTokenLeftUp = constInfo.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo.sparseMode);

        nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
            headS2Split = constInfo.s2OStart != 0U;
            curS2Start = constInfo.s2OStart;
        }

        if ((bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
            tailS2Split = constInfo.s2OEnd != 0U;
            curS2End = constInfo.s2OEnd;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.gSize)) -
                               1;

        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
        s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.gSize);

        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start, curS2End
        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == constInfo.bN2Start && gS1Cur == constInfo.gS1OStart) { // first line
            headS2Split = constInfo.s2OStart > s2StartWithSparse ? true : false;
            curS2Start = Max(s2StartWithSparse, constInfo.s2OStart);
        }
        if (bN2Cur == constInfo.bN2End && gS1Cur == constInfo.gS1OEnd) { // last line
            tailS2Split = constInfo.s2OEnd > 0U ? true : false;
            curS2End = constInfo.s2OEnd > 0U ? Min(s2EndWithSparse, constInfo.s2OEnd) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1(runInfo0);
            } else {
                ComputeVec1(runInfo0);
            }
        }

        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIC {
                    ComputeMm2(runInfoNegN);
                } else {
                    ComputeVec2(runInfoNegN);
                }
                runInfoNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        vecFaBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        this->vecFaBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo);
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop;
        info.bIdx = bN2Cur / constInfo.n2Size;
        info.n2Idx = bN2Cur % constInfo.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                      LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.gSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;
        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }

        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = constInfo.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);

        info.actVecMSize = (info.actMSize + 1) >> 1;
        info.vecMbaseIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((constInfo.bN2Start == constInfo.bN2End && constInfo.gS1OStart == constInfo.gS1OEnd)) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start;
            return;
        } else if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
            s2Cur++;
            return;
        }

        s2Cur = 0;

        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void Combine()
    {
        vecCombineBlock.InitBuffers(this->pipe);
        AscendC::ICachePreLoad(2);
        uint32_t combineCoreEnable =
            faMetaDataGm.GetValue(GetCombineMetaDataIndex(constInfo.aivIdx, COMBINE_CORE_ENABLE_INDEX));
        uint32_t combineBN2Idx =
            faMetaDataGm.GetValue(GetCombineMetaDataIndex(constInfo.aivIdx, COMBINE_BN2_IDX_INDEX));
        uint32_t combineMIdx = faMetaDataGm.GetValue(GetCombineMetaDataIndex(constInfo.aivIdx, COMBINE_M_IDX_INDEX));
        uint32_t combineS2SplitNum =
            faMetaDataGm.GetValue(GetCombineMetaDataIndex(constInfo.aivIdx, COMBINE_S2_SPLIT_NUM_INDEX));
        uint32_t mStart = faMetaDataGm.GetValue(GetCombineMetaDataIndex(constInfo.aivIdx, COMBINE_M_START_INDEX));
        uint32_t mLen = faMetaDataGm.GetValue(GetCombineMetaDataIndex(constInfo.aivIdx, COMBINE_M_NUM_INDEX));
        uint32_t combineWorkspaceIdx =
            faMetaDataGm.GetValue(GetCombineMetaDataIndex(constInfo.aivIdx, COMBINE_WORKSPACE_IDX_INDEX));

        CombineParamsX combineParams = {combineCoreEnable,  combineBN2Idx, combineMIdx, combineS2SplitNum, mStart, mLen,
                                        combineWorkspaceIdx};
        vecCombineBlock.AllocEventID();
        SyncAll();
        vecCombineBlock.Combine(combineParams);
        vecCombineBlock.FreeEventID();
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx < constInfo.coreNum) {
            CrossCoreBufferInit();
            if ASCEND_IS_AIV {
                vecFaBlock.InitBuffers();
                vecFaBlock.AllocEventID();
            } else {
                cubeBlock.InitBuffers();
                cubeBlock.AllocEventID();
            }
            FlashAttention();

            if ASCEND_IS_AIV {
                vecFaBlock.FreeEventID();
            } else {
                cubeBlock.FreeEventID();
            }
            CrossCoreBufferUnInit();
        }

        if constexpr (COMBINE) {
            if ASCEND_IS_AIV {
                Combine();
            }
        }
    }
}; // FlashAttentionKernel

} // namespace BaseApi

#endif // FA_KERNEL_H
