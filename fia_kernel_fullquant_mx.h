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
    static constexpr uint32_t bSize = (uint32_t)2;//修改
    static constexpr uint32_t n2Size = (uint32_t)8;//修改
    static constexpr uint32_t realN2Size = (uint32_t)8;//修改
    static constexpr uint32_t gSize = (uint32_t)8;
    static constexpr uint32_t s1Size = (uint32_t)4;
    static constexpr uint32_t s2Size = (uint32_t)512;
    // static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t realGSize = (uint32_t)8;
    static constexpr uint32_t maxBlockNumPerBatch = (uint32_t)1;//修改
    static constexpr uint32_t gS1OStart = (uint32_t)0;
    static constexpr uint32_t s2OStart = (uint32_t)0;
    static constexpr uint32_t coreNum = (uint32_t)32;
    static constexpr uint32_t gS1OEnd = (uint32_t)0;
    static constexpr uint32_t s2OEnd = (uint32_t)0;
    static constexpr int64_t preTokens = (int64_t)2147483647;
    static constexpr bool isKvContinuous = (bool)1;
    static constexpr uint64_t actualSeqLenSize = (uint64_t)48;
    static constexpr uint64_t actualSeqLenKVSize = (uint64_t)48;

    static constexpr uint64_t actSeqLensKv = 512;
    static constexpr uint64_t actSeqLensQ = 4;
    static constexpr uint32_t curS2Start = 0;
    static constexpr uint32_t curS2End = 1;
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

    // using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    TPipe tPipe;
    TBuf<TPosition::A1> L1_P0, L1_P1, L1_P2;
    TBuf<TPosition::A1> L1_PScale;
    TBuf<TPosition::VECIN> UB_MM1Res0, UB_MM1Res1;
    TBuf<TPosition::VECIN> UB_MM2Res;
    int bmm1_id{0};
    int bmm2_id{4};

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    __gm__ uint64_t *actualSeqLengthsGmQ = nullptr;
    __gm__ uint64_t *actualSeqLengthsGm = nullptr;

    // ConstInfoX constInfo;

    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    // VecFdBlockType vecFdBlock;

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
        // if ASCEND_IS_AIC {
        //     constInfo.aicIdx = GetBlockIdx();
        // } else {
        //     constInfo.aivIdx = GetBlockIdx();
        //     constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
        //     constInfo.subBlockIdx = GetSubBlockIdx();
        // }

        if ASCEND_IS_AIC {
            aicIdx = GetBlockIdx();
        } else {
            aivIdx = GetBlockIdx();
            aicIdx = GetBlockIdx() / GetSubBlockNum();
            subBlockIdx = GetSubBlockIdx();
        }
        bN2Start = 1 * aicIdx;//修改
        bN2End = 1 * (aicIdx + 1);//修改

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
        if (aicIdx >= coreNum) {
            return;
        }

        // RunInfoX taskRunInfo[3];
        uint32_t bN2Cur = bN2Start;
        uint32_t gS1Cur = gS1OStart;
        uint32_t s2Cur = s2OStart;
        // prevBN2Idx = bN2Cur;
        // prevGS1Idx = gS1Cur;

        // uint32_t bIdx = 0, n2Idx = 0, gS1Idx = 0, s2Idx = 0, actMSize = 0, loopIdx = 0, mloopIdx = 0;
        uint32_t s2Idxp[3], loopIdxp[3], mloopIdxp[3], vecMbaseIdxp[3];
        uint32_t bIdxp[3], n2Idxp[3];
        uint32_t curS2Loop[3];
        bool isFirstS2Loop[3] = {false, false, false};
        bool isLastS2Loop[3] = {false, false, false};
        bool runInfo[3] = {false, false, false};

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        bool shouldExecuteTask_MX = false;
        uint32_t inner = 0, outer = 0;


        while (shouldDispatchTask || shouldExecuteTask_MX) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            // //set_mark(0x988773);
            // //set_mark(shouldDispatchTask);
            if (shouldDispatchTask) {
                // TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (s2Cur == 0 ) {
                    mloop ++;
                }
                // if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    bIdxp[createdTaskCount % 3] = bN2Cur / n2Size;
                    n2Idxp[createdTaskCount % 3] = bN2Cur  % n2Size;
                    // gS1Idxp[createdTaskCount % 3] = gS1Cur * mBaseSize;//0
                    s2Idxp[createdTaskCount % 3] = s2Cur * s2BaseSize;
                    // actMSizexp[createdTaskCount % 3] = actSeqLensQ * gSize - gS1Cur * mBaseSize;//32
                    loopIdxp[createdTaskCount % 3] = createdTaskCount;
                    mloopIdxp[createdTaskCount % 3] = mloop;
                    runInfo[createdTaskCount % 3] = true;
                    
                    curS2Loop[createdTaskCount % 3] = s2Cur;
                    isFirstS2Loop[createdTaskCount % 3] = ((createdTaskCount == 0) || (s2Cur == curS2Start));
                    isLastS2Loop[createdTaskCount % 3] = (s2Cur + 1 == curS2End);//(s2Cur + 1 == curS2End);

                    // actVecMSizexp[createdTaskCount % 3] = (actMSizexp[createdTaskCount % 3] + 1) >> 1;
                    vecMbaseIdxp[createdTaskCount % 3] = 0;
                    if (subBlockIdx == 1) {
                        vecMbaseIdxp[createdTaskCount % 3] = 16;//actVecMSizexp[createdTaskCount % 3]
                        // actVecMSizexp[createdTaskCount % 3] = actMSizexp[createdTaskCount % 3] - actVecMSizexp[createdTaskCount % 3];
                    }
                    createdTaskCount++;
                    UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                // }
            }
            // 执行任务
            shouldExecuteTask_MX = ShouldExecuteTask_MX(runInfo);
            if (shouldExecuteTask_MX) {
                ExecuteTask_MX(executedTaskCount, runInfo, bIdxp, n2Idxp, 
                                            s2Idxp, 
                                                    loopIdxp, mloopIdxp, vecMbaseIdxp, curS2Loop, isFirstS2Loop, isLastS2Loop);
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

    
    __aicore__ inline void ExecuteTask_MX(uint64_t loop, bool runInfo[3], uint32_t bIdxp[3], uint32_t n2Idxp[3],
                                            uint32_t s2Idxp[3],
                                                    uint32_t loopIdx[3], uint32_t mloopIdx[3], uint32_t vecMbaseIdxp[3], uint32_t curS2Loop[3],
                                                    bool isFirstS2Loop[3], bool isLastS2Loop[3])
    {
        if (runInfo[loop % 3]) {
            for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
                bmm1_id ++;
                if ASCEND_IS_AIC {
                    LocalTensor<float> mm1ResTensor = (subLoop == 0) ? UB_MM1Res0.Get<float>() : UB_MM1Res1.Get<float>();
                    cubeBlock.IterateBmm1(mm1ResTensor, bmm1_id % 2, bIdxp[loop % 3], n2Idxp[loop % 3],
                                    s2Idxp[loop % 3], curS2Loop[loop % 3], subLoop);
                } else {
                    LocalTensor<float> mm1ResTensor = (subLoop == 0) ? UB_MM1Res0.Get<float>() : UB_MM1Res1.Get<float>();
                    LocalTensor<float8_e4m3_t> p1Tensor = (loop % 3 == 0) ? L1_P0.Get<float8_e4m3_t>() : ((loop % 3 == 1) ? L1_P1.Get<float8_e4m3_t>() : L1_P2.Get<float8_e4m3_t>());
                    vecFaBlock.ProcessVec1(mm1ResTensor, p1Tensor, bmm1_id % 2, loop % 3, bIdxp[loop % 3], s2Idxp[loop % 3],
                                            loopIdx[loop % 3], mloopIdx[loop % 3], vecMbaseIdxp[loop % 3], curS2Loop[loop % 3], subLoop);
                }
            }
        }
        if (loop >= PRELOAD_N) {
            if (runInfo[(loop - 2) % 3]) {
                if ASCEND_IS_AIC {
                    LocalTensor<float> mm2ResTensor = UB_MM2Res.Get<float>();
                    LocalTensor<float8_e4m3_t> p1Tensor = ((loop - 2) % 3 == 0) ? L1_P0.Get<float8_e4m3_t>() : (((loop - 2) % 3 == 1) ? L1_P1.Get<float8_e4m3_t>() : L1_P2.Get<float8_e4m3_t>());
                    LocalTensor<float8_e8m0_t> pscaleTensor = L1_PScale.Get<float8_e8m0_t>();
                    cubeBlock.IterateBmm2((loop - 2) % 3, mm2ResTensor, p1Tensor, bmm2_id, pscaleTensor,
                                            bIdxp[(loop - 2) % 3], n2Idxp[(loop - 2) % 3], s2Idxp[(loop - 2) % 3]);
                } else {
                    LocalTensor<float> mm2ResTensor = UB_MM2Res.Get<float>();
                    vecFaBlock.ProcessVec2(mm2ResTensor, bmm2_id, loopIdx[(loop - 2) % 3], mloopIdx[(loop - 2) % 3], bIdxp[(loop - 2) % 3], n2Idxp[(loop - 2) % 3],
                                            vecMbaseIdxp[(loop - 2) % 3],
                                            curS2Loop[(loop - 2) % 3]);
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

        // if constexpr (FLASH_DECODE) {
        //     if ASCEND_IS_AIV {
        //         FlashDecode();
        //     }
        // }
    }
}; // FlashAttentionFullQuantMxKernel

} // namespace BaseApi

#endif // FIA_KERNEL_FULLQUANT_MX_H_