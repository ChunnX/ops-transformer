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
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/moe_distribute_combine_v2.h"
#include "moe_distribute_combine_v2_entry.h"
#include "../../moe_distribute_dispatch_v2/ascend910_93/aot_framework.h"
#include "aot_tiling_config.h"

using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Kernel;
using namespace AscendC;
using namespace AOT;

/*
* A3 tilingkey说明
* 3位的十进制数
* 第1位（个位）：quantMode:
*     0: 不量化, 2: 使能量化
* 第2位（十位）：x输入类型:
*     0: float16, 1: bfloat16
* 第3位（百位）：无实际含义
*/

// <<<>>> 修改模板定义 - 增加 TilingHolder 模板参数以支持AOT
template<typename ExpandXType, bool IsInt8Quant, typename TH = TilingRuntimeHolder>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_combine_v2(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeCombineV2Info tilingData)
{
    TPipe pipe;
    MoeDistributeCombineV2<ExpandXType, IsInt8Quant, TH> op;
    // <<<>>> 修改 Init 调用 - 传递 pipe 地址
    op.Init(mc2Context, expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales,
            xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo,
            XOut, workspaceGM, tilingData, &pipe);
        
    op.Process();

    pipe.DestroyWithoutPipeAll();
    return;
}

// <<<>>> 修改 Switch 函数 - 增加 TilingHolder 模板参数
template <typename TH = TilingRuntimeHolder, int32_t tilingKey>
__aicore__ inline void moe_combine_switch(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
    GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context,
    MoeDistributeCombineV2Info tilingData)
{
    if constexpr (tilingKey == 100) {
            moe_distribute_combine_v2<float16_t, false, TH>(
                expandX, expertIds, expandIdx, epSendCount, residualX,
                gamma, expertScales, xActiveMask,
                sharedExpertX, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV,
                performanceInfo, XOut,
                workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 102) {
            moe_distribute_combine_v2<float16_t, true, TH>(
                expandX, expertIds, expandIdx, epSendCount, residualX,
                gamma, expertScales, xActiveMask,
                sharedExpertX, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV,
                performanceInfo, XOut,
                workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 110) {
            moe_distribute_combine_v2<bfloat16_t, false, TH>(
                expandX, expertIds, expandIdx, epSendCount, residualX,
                gamma, expertScales, xActiveMask,
                sharedExpertX, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV,
                performanceInfo, XOut,
                workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 112) {
            moe_distribute_combine_v2<bfloat16_t, true, TH>(
                expandX, expertIds, expandIdx, epSendCount, residualX,
                gamma, expertScales, xActiveMask,
                sharedExpertX, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV,
                performanceInfo, XOut,
                workspaceGM, mc2Context, tilingData);
    }
}

template <typename TH>
__aicore__ inline void test();

// <<<>>> 修改 Generic 函数 - 增加 TilingHolder 模板参数
template <typename TH = TilingRuntimeHolder, int32_t tilingKey>
__global__ __vector__ void moe_distribute_combine_v2_generic(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeCombineV2Info tilingData)
{
    test<TH>();
    // 根据不同的数据类型调用不同的模板
    moe_combine_switch<TH, tilingKey>(
        expandX, expertIds, expandIdx, epSendCount, residualX,
        gamma, expertScales, xActiveMask,
        sharedExpertX, oriX,
        constExpertAlpha1, constExpertAlpha2, constExpertV,
        performanceInfo, XOut,
        workspaceGM, mc2Context,
        tilingData);
}

template <int32_t split_num, typename TH = TilingRuntimeHolder, int32_t tilingKey>
__sk__ __vector__ void moe_distribute_combine_v2_generic_sk(MoeDistributeCombineV2SkArgs *args)
{
    GM_ADDR expandX = args->expandX;
    GM_ADDR expertIds = args->expertIds;
    GM_ADDR expandIdx = args->expandIdx;
    GM_ADDR epSendCount = args->epSendCount;
    GM_ADDR residualX = args->residualX;
    GM_ADDR gamma = args->gamma;
    GM_ADDR expertScales = args->expertScales;
    GM_ADDR xActiveMask = args->xActiveMask;
    GM_ADDR sharedExpertX = args->sharedExpertX;
    GM_ADDR oriX = args->oriX;
    GM_ADDR constExpertAlpha1 = args->constExpertAlpha1;
    GM_ADDR constExpertAlpha2 = args->constExpertAlpha2;
    GM_ADDR constExpertV = args->constExpertV;
    GM_ADDR performanceInfo = args->performanceInfo;
    GM_ADDR XOut = args->XOut;
    GM_ADDR workspaceGM = args->workspaceGM;
    GM_ADDR mc2Context = args->mc2Context;
    MoeDistributeCombineV2Info tilingData = args->tilingData;

    // 根据不同的数据类型调用不同的模板
    moe_combine_switch<TH, tilingKey>(
        expandX, expertIds, expandIdx, epSendCount, residualX,
        gamma, expertScales, xActiveMask,
        sharedExpertX, oriX,
        constExpertAlpha1, constExpertAlpha2, constExpertV,
        performanceInfo, XOut,
        workspaceGM, mc2Context,
        tilingData);
}

template <typename TH>
__aicore__ inline void test()
{
    if ASCEND_IS_AIC {
        return;
    }

    SK_BIND((moe_distribute_combine_v2_generic<TH, 100>), 4, moe_distribute_combine_v2_generic_sk<0, TH, 100>, moe_distribute_combine_v2_generic_sk<1, TH, 100>, moe_distribute_combine_v2_generic_sk<2, TH, 100>, moe_distribute_combine_v2_generic_sk<3, TH, 100>);
    SK_BIND((moe_distribute_combine_v2_generic<TH, 102>), 4, moe_distribute_combine_v2_generic_sk<0, TH, 102>, moe_distribute_combine_v2_generic_sk<1, TH, 102>, moe_distribute_combine_v2_generic_sk<2, TH, 102>, moe_distribute_combine_v2_generic_sk<3, TH, 102>);
    SK_BIND((moe_distribute_combine_v2_generic<TH, 110>), 4, moe_distribute_combine_v2_generic_sk<0, TH, 110>, moe_distribute_combine_v2_generic_sk<1, TH, 110>, moe_distribute_combine_v2_generic_sk<2, TH, 110>, moe_distribute_combine_v2_generic_sk<3, TH, 110>);
    SK_BIND((moe_distribute_combine_v2_generic<TH, 112>), 4, moe_distribute_combine_v2_generic_sk<0, TH, 112>, moe_distribute_combine_v2_generic_sk<1, TH, 112>, moe_distribute_combine_v2_generic_sk<2, TH, 112>, moe_distribute_combine_v2_generic_sk<3, TH, 112>);
}

// <<<>>> 修改 Entry 函数 - 使用 AOTDispatcher 进行分发
void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream,
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeCombineV2Info tilingData)
{
    auto tilingPtr = reinterpret_cast<const uint8_t*>(&tilingData);

    // 使用 AOTDispatcher 进行内核分发
    AOTDispatcher<MoeDistributeCombineV2Info, MoeDistributeCombineV2AOTRegistry>::dispatch(
        tilingPtr,
        [&](auto tiling_holder, MoeDistributeCombineV2Info value) {
            using Holder = decltype(tiling_holder);
            if constexpr (std::is_same_v<Holder, TilingRuntimeHolder>) {

                switch (tilingKey) {
                case 100: moe_distribute_combine_v2_generic<Holder, 100><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                case 102: moe_distribute_combine_v2_generic<Holder, 102><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                case 110: moe_distribute_combine_v2_generic<Holder, 110><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                case 112: moe_distribute_combine_v2_generic<Holder, 112><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                default: break;
                }
            } else {
                switch (tilingKey) {
                case 100: moe_distribute_combine_v2_generic<Holder, 100><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                case 102: moe_distribute_combine_v2_generic<Holder, 102><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                case 110: moe_distribute_combine_v2_generic<Holder, 110><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                case 112: moe_distribute_combine_v2_generic<Holder, 112><<<blockDim, nullptr, stream>>>(expandX, expertIds, expandIdx, epSendCount, residualX, gamma, expertScales, xActiveMask, sharedExpertX, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, mc2Context, tilingData); break;
                default: break;
                }
            }
        }
    );
}