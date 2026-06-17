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
 * \file fia_kernel_fullquant_mx.h
 * \brief
 */

#ifndef FIA_KERNEL_FULLQUANT_MX_MX_H_
#define FIA_KERNEL_FULLQUANT_MX_MX_H_

#include "fia_public_define_arch35.h"
#include "fia_block_cube_fullquant_mx.h"
#include "fia_block_vec_fullquant_mx.h"
#include "../memory_copy_arch35.h"
#include "fia_block_vec_flashdecode_fullquant.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "fia_tiling_data_fullquant_mx.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionFullQuantMxKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;
    static constexpr bool HAS_PREFIX = CubeBlockType::HAS_PREFIX;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    // static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    // static constexpr bool HAS_ROPE = CubeBlockType::HAS_ROPE;
    // // static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;
    // static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT; // V100 只支持一种??
    // static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    // static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    // static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    TPipe tPipe;
    TBuf<TPosition::A1> L1_P0, L1_P1, L1_P2;
    TBuf<TPosition::A1> L1_PScale;
    TBuf<TPosition::VECIN> UB_MM1Res0, UB_MM1Res1;
    TBuf<TPosition::VECIN> UB_MM2Res;
    int bmm1_id{0};
    int bmm2_id{4};

    // GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    // GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> fiaMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<INPUT_T> sinkGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    __gm__ uint64_t *actualSeqLengthsGmQ = nullptr;
    __gm__ uint64_t *actualSeqLengthsGm = nullptr;

    ConstInfoX constInfo;

    // const FullQuantTiling *__restrict tilingData;
    // TPipe *pipe = nullptr;
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    uint32_t coreGS1Loops = 0U;
    uint32_t frontGS1Count = 0U;
    uint32_t invalidGS1Size = 0U;
    uint32_t accGS1Loops = 0U;
    uint32_t totalSize = 0U;
    uint32_t createdTaskCount = 0U;
    uint32_t executedTaskCount = 0U;
    uint32_t varlenCalcTimes = 0U;
    bool enableS1OutSplit = false;

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

    // ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    // ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionFullQuantMxKernel()
        : cubeBlock(constInfo), vecFaBlock(constInfo), vecFdBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace)
    {
        // this->pipe = tPipe;
        // this->tilingData = tiling;

        // fiaMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)fiaMetaData,
        //                               NPU_AIC_CORE_NUM * FA_METADATA_SIZE + NPU_AIV_CORE_NUM * FD_METADATA_SIZE);

        InitConstInfo();

        cubeBlock.SetTPipe(tPipe);
        vecFaBlock.SetTPipe(tPipe);

        keyPtr = key;
        valuePtr = value;
        actualSeqLengthsGmQ = (__gm__ uint64_t *)actualSeqLengths;
        actualSeqLengthsGm = (__gm__ uint64_t *)actualSeqLengthsKv;
        InitMMResBuf();

        if ASCEND_IS_AIV {
            // vecFaBlock.InitVecBlock(actualSeqLengths, actualSeqLengthsKv, pScale, attenMask, softmaxLse,
            //                         attentionOut, workspace);
            vecFaBlock.InitVecBlock(actualSeqLengths, pScale, attenMask, attentionOut);
            // vecFaBlock.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock(query, key, value, blockTable, queryRope, keyRope,
                                    actualSeqLengths, actualSeqLengthsKv, dequantScaleQuery, dequantScaleKey,
                                    dequantScaleValue);
        }

    }

    __aicore__ inline void InitMMResBuf()
    {
        constexpr uint32_t mm2LeftSize = 128 * 512;
        constexpr uint32_t pScaleSize = 128 * 512 / 32;
        // 保存p结果的L1内存，triple buffer
        tPipe.InitBuffer(L1_P0, mm2LeftSize);
        tPipe.InitBuffer(L1_P1, mm2LeftSize);
        tPipe.InitBuffer(L1_P2, mm2LeftSize);
        tPipe.InitBuffer(L1_PScale, pScaleSize);

        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;
        tPipe.InitBuffer(UB_MM1Res0, mm1ResultSize);
        tPipe.InitBuffer(UB_MM1Res1, mm1ResultSize);
        tPipe.InitBuffer(UB_MM2Res, mm2ResultSize);

        // 填充 pscale 为全 1 MX scale（0x7f7f = half 格式的全 1）
        if ASCEND_IS_AIC {
            LocalTensor<int16_t> pscaleTensor = L1_PScale.Get<int16_t>();
            InitConstValue(pscaleTensor, {1, pScaleSize / 32, 0, 0x7f7f});
        }
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

        constInfo.bSize = 2;
        constInfo.t1Size = 192;
        constInfo.t2Size = 0;
        constInfo.n2Size = 8; // fiaBaseParams.n2Size;
        constInfo.gSize = 8;
        constInfo.s1Size = 4;
        constInfo.s2Size = 512;
        constInfo.dSize = 128;
        constInfo.dSizeV = 128;
        constInfo.realN2Size = constInfo.n2Size;
        constInfo.realGSize = constInfo.gSize;

        constInfo.dSizeRope = 0;
        constInfo.actualSeqLenSize = 48;
        constInfo.actualSeqLenKVSize = 48;
        constInfo.scaleValue = 0.08838834764831843;
        constInfo.isKvContinuous = 1;
        constInfo.coreNum = 32;
        constInfo.outputLayout = FIA_LAYOUT::TND;

        constInfo.sparseMode = 3;
        constInfo.preTokens = 2147483647;
        constInfo.nextTokens = 2147483647;
        constInfo.attenMaskBatch = 1;
        constInfo.attenMaskS1Size = 2048;
        constInfo.attenMaskS2Size = 2048;
        constInfo.isRowInvalidOpen = 0;
        constInfo.isExistRowInvalid = 0;

        constInfo.accumOutSize = 0;
        constInfo.logSumExpSize = 0;

        constInfo.maxBlockNumPerBatch = 1;//修改
        constInfo.blockSize = 512;
        constInfo.paLayoutType = 2;

        constInfo.isSoftmaxLseEnable = 0;

        constInfo.bN2Start = 1 * constInfo.aicIdx;//修改
        constInfo.gS1OStart = 0;
        constInfo.s2OStart = 0;
        constInfo.bN2End = 1 * (constInfo.aicIdx + 1);//修改
        constInfo.gS1OEnd = 0;
        constInfo.s2OEnd = 0;
        constInfo.coreFirstTmpOutWsPos = 0;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * NPU_AIC_CORE_NUM + FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    // __aicore__ inline bool SkipForS1OutSplit()
    // {
    //     if (!enableS1OutSplit) {
    //         return false;
    //     }

    //     if (coreGS1Loops % 2 == 0) {
    //         if (coreGS1Loops * constInfo.coreNum + constInfo.aicIdx != accGS1Loops) {
    //             return true;
    //         }
    //     } else {
    //         if ((coreGS1Loops + 1) * constInfo.coreNum - constInfo.aicIdx - 1 != accGS1Loops) {
    //             return true;
    //         }
    //     }

    //     return false;
    // }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            // bmm2Buffers.Get().SetCrossCore();
            // set_intra_block(PIPE_V, 0 + 0 + 5);
            set_intra_block(PIPE_V, bmm2_id + 1);
        }
        
        if ASCEND_IS_AIV {
            // bmm1Buffers.Get().SetCrossCore();
            // bmm1Buffers.Get().SetCrossCore();
            // set_intra_block(PIPE_V, 0 + 0 + 9);
            // set_intra_block(PIPE_V, 0 + 0 + 7);
            set_intra_block(PIPE_V, bmm2_id + 5);
            set_intra_block(PIPE_V, bmm2_id + 3);
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            // bmm1Buffers.Get().WaitCrossCore();
            // bmm1Buffers.Get().WaitCrossCore();
            // wait_intra_block(PIPE_FIX, 9);
            // wait_intra_block(PIPE_FIX, 9 + 16);

            // wait_intra_block(PIPE_FIX, 7 );
            // wait_intra_block(PIPE_FIX, 7 + 16);

            wait_intra_block(PIPE_FIX, bmm2_id + 5);
            wait_intra_block(PIPE_FIX, bmm2_id + 5 + 16);

            wait_intra_block(PIPE_FIX, bmm2_id + 3);
            wait_intra_block(PIPE_FIX, bmm2_id + 3 + 16);
        }
        if ASCEND_IS_AIC {
            // bmm2Buffers.Get().WaitCrossCore();
            // wait_intra_block(PIPE_FIX, 5);
            // wait_intra_block(PIPE_FIX, 5 + 16);

            wait_intra_block(PIPE_FIX, bmm2_id + 1);
            wait_intra_block(PIPE_FIX, bmm2_id + 1 + 16);
        }
        
    }

        __aicore__ inline void prefree_kv_l1()
    {
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)0);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)1);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)2);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)3);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)4);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)5);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)6);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)7);
    }

    __aicore__ inline void prefree_l0c(){
        set_flag(PIPE_FIX, PIPE_M, (event_t)0);
        set_flag(PIPE_FIX, PIPE_M, (event_t)1);
    }

    __aicore__ inline void prefree_l0ab(){
        set_flag(PIPE_M, PIPE_MTE1, (event_t)4);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)5);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)6);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)7);
    }

    __aicore__ inline void postalloc_l0ab(){
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)4);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)5);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)6);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)7);
    }

    __aicore__ inline void postalloc_l0c(){
        wait_flag(PIPE_FIX, PIPE_M, (event_t)0);
        wait_flag(PIPE_FIX, PIPE_M, (event_t)1);
    }

        __aicore__ inline void postalloc_kv_l1()
    {
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)3);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)4);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)5);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)6);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)7);
    }

    __aicore__ inline void FlashAttention()
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        RunInfoX taskRunInfo[3];
        uint32_t bN2Cur = constInfo.bN2Start;
        uint32_t gS1Cur = constInfo.gS1OStart;
        uint32_t s2Cur = constInfo.s2OStart;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        // uint32_t bIdx = 0, n2Idx = 0, gS1Idx = 0, s2Idx = 0, actMSize = 0, loopIdx = 0, mloopIdx = 0;
        uint32_t gS1Idxp[3], s2Idxp[3], actMSizexp[3], loopIdx[3], mloopIdx[3];
        uint32_t bIdxp[3], n2Idxp[3];
        bool runInfo[3] = {false, false, false};

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        bool shouldExecuteTask_MX = false;
        uint32_t inner = 0, outer = 0;


        while (shouldDispatchTask || shouldExecuteTask) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            // //set_mark(0x988773);
            // //set_mark(shouldDispatchTask);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);

                    bIdxp[createdTaskCount % 3] = bN2Cur / constInfo.n2Size;
                    n2Idxp[createdTaskCount % 3] = bN2Cur  % constInfo.n2Size;
                    gS1Idxp[createdTaskCount % 3] = gS1Cur * mBaseSize;
                    s2Idxp[createdTaskCount % 3] = s2Cur * s2BaseSize;
                    actMSizexp[createdTaskCount % 3] = actSeqLensQ * constInfo.gSize - gS1Cur * mBaseSize;
                    loopIdx[createdTaskCount % 3] = createdTaskCount;
                    mloopIdx[createdTaskCount % 3] = mloop;
                    runInfo[createdTaskCount % 3] = true;
                    inner = s2Cur;
                    outer = gS1Cur;
                    
                    createdTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                }
            }
            // 执行任务
            shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
            shouldExecuteTask_MX = ShouldExecuteTask_MX(runInfo);
            if (shouldExecuteTask) {
                // bIdxp[executedTaskCount % 3] = bIdx;
                // n2Idxp[executedTaskCount % 3] = n2Idx;
                // gS1Idxp[executedTaskCount % 3] = gS1Idx;
                // s2Idxp[executedTaskCount % 3] = s2Idx;
                // actMSizexp[executedTaskCount % 3] = actMSize;
                // ipid ++;
                // if (runInfo[executedTaskCount % 3] == true) {
                //     for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
                //         bmm1_id ++;
                //         if ASCEND_IS_AIC {
                //             cubeBlock.IterateBmm1(mm1Res[subLoop], bmm1_id % 2, bIdxp[executedTaskCount % 3], n2Idxp[executedTaskCount % 3], 
                //                             gS1Idxp[executedTaskCount % 3], s2Idxp[executedTaskCount % 3], actMSizexp[executedTaskCount % 3], inner, subLoop);
                //         } else {
                //             vecFaBlock.ProcessVec1(p1[executedTaskCount % 3], mm1Res[subLoop], bmm1_id % 2, executedTaskCount % 3, inner,
                //                                     loopIdx[executedTaskCount % 3], mloopIdx[executedTaskCount % 3], subLoop);
                //         }
                //     }
                // }
                // if (executedTaskCount >= 2) {
                //     if (runInfo[(executedTaskCount - 2) % 3] == true) {
                //         if ASCEND_IS_AIC {
                //             cubeBlock.IterateBmm2((executedTaskCount - 2) % 3, mm2Res, p1[(executedTaskCount - 2) % 3], bmm2_id, pscale, runInfoNegN);
                //         } else {
                //             vecFaBlock.ProcessVec2(mm2Res, bmm2_id, runInfoNegN);
                //         }
                //         runInfo[(executedTaskCount - 2) % 3] = false;
                //     }
                // }
                // for(int i=0;i<3;i++){
                //     printf("start____ ,runinfo[%d] = %d\n",i,runInfo[i]);
                //     printf("ExecuteTaskend,taskRunInfo[%d].isValid = %d\n",i,taskRunInfo[i].isValid);
                // }
                // printf("\n");
                ExecuteTask_MX(executedTaskCount, taskRunInfo, runInfo, bIdxp, n2Idxp, 
                                            gS1Idxp, s2Idxp, actMSizexp,
                                                    loopIdx, mloopIdx, inner);
                executedTaskCount++;//loop
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
    __aicore__ inline bool ShouldExecuteTask_MX(bool runInfo[3])
    {
        for (uint32_t i = 0; i < 3; i++) {
            if (runInfo[i] == true) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart) && (s2Cur == constInfo.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo.realN2Size;
         if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if (constInfo.actualSeqLenKVSize == 0) {
                actSeqLensKv =  constInfo.s2Size;
            }
            if (constInfo.actualSeqLenKVSize == 1) {
                actSeqLensKv = actualSeqLengthsGm[0];
            }
            actSeqLensKv = actualSeqLengthsGm[bIdx];
            if (bIdx == 0) {
                actSeqLensQ = actualSeqLengthsGmQ[0];
            } else {
                actSeqLensQ = actualSeqLengthsGmQ[bIdx] - actualSeqLengthsGmQ[bIdx - 1];
            }
            
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
       
        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            // if constexpr (!HAS_MASK) {
            //     CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            // } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            // }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
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

        if (constInfo.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL || constInfo.sparseMode == fa_base_vector::TREE) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    // __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    // {
    //     curS2Start = 0U;
    //     curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;
    //     if ((bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
    //         headS2Split = constInfo.s2OStart != 0U;
    //         curS2Start = constInfo.s2OStart;
    //     }

    //     if (!enableS1OutSplit && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
    //         tailS2Split = constInfo.s2OEnd != 0U;
    //         curS2End = constInfo.s2OEnd;
    //     }
    // }

    __aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
    {
        sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
        sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
        return sInnerToken;
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
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.realGSize)) -
                               1;

        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        // if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.realGSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.realGSize);
        // } 
        // else {
        //     if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
        //         // start and end locate in one G
        //         s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
        //         s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
        //     } else {
        //         // start and end locate in tow or more G, but working same as crossing one complete block
        //         s1FirstToken = 0;
        //         s1LastToken = static_cast<int64_t>(actSeqLensQ);
        //     }
        // }

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

    __aicore__ inline void ExecuteTask_MX(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE], bool runInfo[3], uint32_t bIdxp[3], uint32_t n2Idxp[3], 
                                            uint32_t gS1Idxp[3], uint32_t s2Idxp[3], uint32_t actMSizexp[3],
                                                    uint32_t loopIdx[3], uint32_t mloopIdx[3], uint32_t inner)
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
        // if (runInfo0.isValid) {
        //     // uint32_t c1v1Loop = CeilDiv(runInfo0.actSingleLoopS2Size, 256);
        //     for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
        //         bmm1_id ++;
        //         if ASCEND_IS_AIC {
        //             // //printf("mm1 = %d\n",loop);
        //             cubeBlock.IterateBmm1(mm1Res[subLoop], bmm1_id % 2, runInfo0, subLoop);
        //         } else {
        //             vecFaBlock.ProcessVec1(p1[loop % 3], mm1Res[subLoop], bmm1_id % 2, loop % 3, runInfo0, subLoop);
        //         }
        //     }
        // }
        // for(int i=0;i<3;i++){
        //     printf("ExecuteTask ,runinfo[%d] = %d\n",i,runInfo[i]);
        //     printf("ExecuteTask ,taskRunInfo[%d].isValid = %d\n",i,taskRunInfo[i].isValid);
        // }
        // printf("\n");
        if (runInfo[loop % 3]) {
            for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
                bmm1_id ++;
                if ASCEND_IS_AIC {
                    // 用 TBuf.Get<float>() 构造 LocalTensor 传入
                    LocalTensor<float> mm1ResTensor = (subLoop == 0) ? UB_MM1Res0.Get<float>() : UB_MM1Res1.Get<float>();
                    cubeBlock.IterateBmm1(mm1ResTensor, bmm1_id % 2, bIdxp[loop % 3], n2Idxp[loop % 3],
                                    gS1Idxp[loop % 3], s2Idxp[loop % 3], actMSizexp[loop % 3], inner, subLoop);
                } else {
                    // AIV: 传 LocalTensor<float> 和 p1 LocalTensor
                    LocalTensor<float> mm1ResTensor = (subLoop == 0) ? UB_MM1Res0.Get<float>() : UB_MM1Res1.Get<float>();
                    LocalTensor<float8_e4m3_t> p1Tensor = (loop % 3 == 0) ? L1_P0.Get<float8_e4m3_t>() : ((loop % 3 == 1) ? L1_P1.Get<float8_e4m3_t>() : L1_P2.Get<float8_e4m3_t>());
                    vecFaBlock.ProcessVec1(mm1ResTensor, p1Tensor, bmm1_id % 2, loop % 3, inner,
                                            loopIdx[loop % 3], mloopIdx[loop % 3], subLoop);
                }
            }
        }
        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIC {
                    LocalTensor<float> mm2ResTensor = UB_MM2Res.Get<float>();
                    LocalTensor<float8_e4m3_t> p1Tensor = ((loop - 2) % 3 == 0) ? L1_P0.Get<float8_e4m3_t>() : (((loop - 2) % 3 == 1) ? L1_P1.Get<float8_e4m3_t>() : L1_P2.Get<float8_e4m3_t>());
                    LocalTensor<float8_e8m0_t> pscaleTensor = L1_PScale.Get<float8_e8m0_t>();
                    cubeBlock.IterateBmm2((loop - 2) % 3, mm2ResTensor, p1Tensor, bmm2_id, pscaleTensor,
                                            bIdxp[(loop - 2) % 3], n2Idxp[(loop - 2) % 3], s2Idxp[(loop - 2) % 3], actMSizexp[(loop - 2) % 3]);
                } else {
                    vecFaBlock.ProcessVec2(UB_MM2Res.Get<float>(), bmm2_id, runInfoNegN);
                }
                runInfoNegN.isValid = false;
                runInfo[(loop - 2) % 3] = false;
            }
        }
        //  for(int i=0;i<3;i++){
        //     printf("ExecuteTaskend ,runinfo[%d] = %d\n",i,runInfo[i]);
        //     printf("ExecuteTaskend,taskRunInfo[%d].isValid = %d\n",i,taskRunInfo[i].isValid);
        // }
        // printf("\n");
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
        info.bIdx = bN2Cur / (constInfo.realN2Size);
        info.n2Idx = (bN2Cur / (constInfo.realN2Size / constInfo.n2Size)) % constInfo.n2Size;
        info.realN2Idx = bN2Cur % constInfo.realN2Size;
        info.gS1Idx = gS1Cur * mBaseSize;

        // if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
        //               LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
        //     // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.realGSize;
        // } else {
        //     // GS1 layout
        //     info.s1Idx = info.gS1Idx % actSeqLensQ;
        // }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.realGSize;//4 *8
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

        if (constInfo.isKvContinuous) {
            info.isChangeBatch = false;
        } else {
            // for tensor-list
            if (loop == 0) { // 第一个有效任务才需要重置KV的tensor
                info.isChangeBatch = true;
            } else {
                info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
            }
        }

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);


        info.actVecMSize = (info.actMSize + 1) >> 1;
        
        info.vecMbaseIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        // 当前S2未处理完
        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }

        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }


        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
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
                // cubeBlock.AllocEventID();
                prefree_kv_l1();  // 4buffer
                prefree_l0ab();
                prefree_l0c();
            }
            FlashAttention();

            if ASCEND_IS_AIV {
                vecFaBlock.FreeEventID();
            } else {
                // cubeBlock.FreeEventID();
                postalloc_kv_l1();
                postalloc_l0ab();
                postalloc_l0c();
            }
            CrossCoreBufferUnInit();
        }

        // if constexpr (FLASH_DECODE) {
        //     if ASCEND_IS_AIV {
        //         FlashDecode();
        //     }
        // }
    }
}; // FlashAttentionFullQuantMxKernel

} // namespace BaseApi

#endif // FIA_KERNEL_FULLQUANT_MX_H_