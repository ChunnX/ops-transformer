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

#ifndef FIA_KERNEL_FULLQUANT_PA_WHILE_LOOP_P_L1_H_
#define FIA_KERNEL_FULLQUANT_PA_WHILE_LOOP_P_L1_H_

// #include "fia_public_define_arch35.h"
#include "fia_block_cube_fullquant_mx.h"
#include "fia_block_vec_fullquant_mx.h"
// #include "../memory_copy_arch35.h"
// #include "fia_block_vec_flashdecode_fullquant.h"

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
template <typename CubeBlockType, typename VecFaBlockType>
class FlashAttentionFullQuantMxKernel {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)128;
    static constexpr uint32_t s2BaseSize = (uint32_t)512;
    static constexpr uint32_t dBaseSize = (uint32_t)128;
    static constexpr uint32_t dVBaseSize = (uint32_t)128;
    static constexpr uint32_t bSize = (uint32_t)48;//修改
    static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t realN2Size = (uint32_t)8;
    static constexpr uint32_t gSize = (uint32_t)8;
    static constexpr uint32_t s1Size = (uint32_t)4;
    static constexpr uint32_t s2Size = (uint32_t)512;
    static constexpr uint32_t realGSize = (uint32_t)8;
    static constexpr uint32_t maxBlockNumPerBatch = (uint32_t)20;//修改
    static constexpr uint32_t gS1OStart = (uint32_t)0;
    static constexpr uint32_t s2OStart = (uint32_t)0;
    static constexpr uint32_t coreNum = (uint32_t)32;
    static constexpr uint32_t gS1OEnd = (uint32_t)0;
    static constexpr uint32_t s2OEnd = (uint32_t)0;
    static constexpr int64_t preTokens = (int64_t)2147483647;
    static constexpr bool isKvContinuous = (bool)1;
    static constexpr uint64_t actualSeqLenSize = (uint64_t)48;
    static constexpr uint64_t actualSeqLenKVSize = (uint64_t)48;

    static constexpr uint64_t actSeqLensKv = 10240;//修改
    static constexpr uint64_t actSeqLensQ = 4;
    static constexpr uint32_t curS2Start = 0;
    static constexpr uint32_t curS2End = 20;//修改
    static constexpr uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
    static constexpr uint64_t gS1Size = actSeqLensQ * realGSize;
    static constexpr uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

    uint32_t aicIdx = (uint32_t)0;
    uint32_t aivIdx = (uint32_t)0;
    uint8_t subBlockIdx = (uint8_t)0;
    uint32_t bN2Start = (uint32_t)0;
    uint32_t bN2End = (uint32_t)0;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    __ubuf__ void *mm1Res[2], *mm2Res;
    __cbuf__ void *p1[3];
    int bmm1_id{0};
    int bmm2_id{4};
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;

    uint32_t createdTaskCount = 0U;
    uint32_t executedTaskCount = 0U;
    uint32_t mloop = 0;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionFullQuantMxKernel()
        : cubeBlock(), vecFaBlock(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace)
    {
        InitConstInfo();

        keyPtr = key;
        valuePtr = value;
        InitMMResBuf();

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock(query, key, value, blockTable, queryRope, keyRope,
                                    actualSeqLengths, actualSeqLengthsKv, dequantScaleQuery, dequantScaleKey,
                                    dequantScaleValue);
        }

    }

    __aicore__ inline void InitMMResBuf()
    {
        constexpr uint32_t mm2LeftSize = 128 * 512  + 128 * 512 / 32;
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        p1[0] = (__cbuf__ void *)get_imm(0);
        p1[1] = (__cbuf__ void *)get_imm(mm2LeftSize);
        p1[2] = (__cbuf__ void *)get_imm(mm2LeftSize * 2);

        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;
        mm1Res[0] = (__ubuf__ void *)get_imm(0); 
        mm1Res[1] = (__ubuf__ void *)get_imm(mm1ResultSize);
        mm2Res = (__ubuf__ void *)get_imm(mm1ResultSize * 2);
    }

    __aicore__ inline void InitConstInfo()
    {

        if ASCEND_IS_AIC {
            aicIdx = GetBlockIdx();
        } else {
            aivIdx = GetBlockIdx();
            aicIdx = GetBlockIdx() / GetSubBlockNum();
            subBlockIdx = GetSubBlockIdx();
        }
        bN2Start = 12 * aicIdx;//修改
        bN2End = 12 * (aicIdx + 1);//修改

    }
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
        set_flag(PIPE_M, PIPE_MTE1, (event_t)0);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)1);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)2);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)3);
    }

    __aicore__ inline void postalloc_l0ab(){
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)0);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)1);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)2);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)3);
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
        if (aicIdx >= coreNum) {
            return;
        }

        uint32_t bN2Cur = bN2Start;
        uint32_t gS1Cur = gS1OStart;
        uint32_t s2Cur = s2OStart;
        uint32_t s2Idxp[3], loopIdxp[3];
        uint32_t bIdxp[3], n2Idxp[3], gS1Idxp[3];
        uint32_t curS2Loop[3];
        bool runInfo[3] = {false, false, false};

        bool shouldDispatchTask = true;
        bool shouldExecuteTask_MX = false;

        while (shouldDispatchTask || shouldExecuteTask_MX) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                // if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    bIdxp[createdTaskCount % 3] = bN2Cur / n2Size;
                    n2Idxp[createdTaskCount % 3] = bN2Cur  % n2Size;
                    gS1Idxp[createdTaskCount % 3] = gS1Cur * mBaseSize;
                    s2Idxp[createdTaskCount % 3] = s2Cur * s2BaseSize;
                    loopIdxp[createdTaskCount % 3] = createdTaskCount;
                    curS2Loop[createdTaskCount % 3] = s2Cur;
                    runInfo[createdTaskCount % 3] = true;

                    createdTaskCount++;
                    UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                // }
            }
            // 执行任务
            shouldExecuteTask_MX = ShouldExecuteTask_MX(runInfo);
            if (shouldExecuteTask_MX) {
                ExecuteTask_MX(executedTaskCount, runInfo, bIdxp, n2Idxp, gS1Idxp, s2Idxp, loopIdxp, curS2Loop);
                executedTaskCount++;//loop
            }
        }
    }
    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != bN2End) || (gS1Cur != gS1OEnd) || (s2Cur != s2OEnd));
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

    
    __aicore__ inline void ExecuteTask_MX(uint64_t loop, bool runInfo[3], uint32_t bIdxp[3], uint32_t n2Idxp[3], uint32_t gS1Idxp[3],
                                          uint32_t s2Idxp[3],
                                                    uint32_t loopIdx[3], uint32_t curS2Loop[3])
    {
        if (runInfo[loop % 3]) {
            for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
                bmm1_id ++;
                if ASCEND_IS_AIC {
                    cubeBlock.IterateBmm1(mm1Res[subLoop], bmm1_id % 2, bIdxp[loop % 3], n2Idxp[loop % 3], gS1Idxp[loop % 3], 
                                    s2Idxp[loop % 3], curS2Loop[loop % 3], subLoop);
                } else {
                    vecFaBlock.ProcessVec1(p1[loop % 3], bmm1_id % 2, loop % 3, loopIdx[loop % 3], subLoop);
                }
            }
        }
        if (loop >= PRELOAD_N) {
            if (runInfo[(loop - 2) % 3]) {
                if ASCEND_IS_AIC {
                    cubeBlock.IterateBmm2((loop - 2) % 3, mm2Res, p1[(loop - 2) % 3], bIdxp[(loop - 2) % 3], n2Idxp[(loop - 2) % 3], s2Idxp[(loop - 2) % 3]);
                }
                runInfo[(loop - 2) % 3] = false;
            }
        }
    }



    __aicore__ inline void UpdateAxisInfo(uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        
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
        if (aicIdx < coreNum) {
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
    }
}; // FlashAttentionFullQuantMxKernel

} // namespace BaseApi

#endif // FIA_KERNEL_FULLQUANT_MX_H_