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
 * \file moe_distribute_combine_v2_a5_entry.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/moe_distribute_combine_v2.h"
#include "op_kernel/moe_distribute_combine_v2_tiling.h"


using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Kernel;
using namespace AscendC;

template<typename ExpandXType, typename XType>
__attribute__((always_inline)) __aicore__ __inline__ void moe_combine_v2_a5(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfo, GM_ADDR epSendCounts,
    GM_ADDR expertScales, GM_ADDR tpSendCounts, GM_ADDR xActiveMask,
    GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
    GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM,
    MoeDistributeCombineV2Info tilingData)
{
    TPipe pipe;
    MoeDistributeCombineV2TilingData td{};
    td.moeDistributeCombineV2Info = tilingData;
    MoeDistributeCombineV2<ExpandXType, XType, int32_t, false, UNQUANT, false> op;
    op.Init(nullptr, expandX, expertIds, assistInfo, epSendCounts, tpSendCounts,
        nullptr, nullptr, expertScales, xActiveMask, sharedExpertX, elasticInfo, oriX,
        constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo,
        nullptr, nullptr, XOut, workspaceGM, &pipe, &td);

    op.Process();

    pipe.DestroyWithoutPipeAll();
}

extern "C" __global__ __aicore__ void combine_a5_generic(
    int32_t tilingKey,
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfo, GM_ADDR epSendCounts,
    GM_ADDR expertScales, GM_ADDR tpSendCounts, GM_ADDR xActiveMask,
    GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
    GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM,
    GM_ADDR mc2Context,
    MoeDistributeCombineV2Info tilingData)
{
    switch (tilingKey) {
    case 0:
        moe_combine_v2_a5<half, half>(
            expandX, expertIds, assistInfo, epSendCounts, expertScales,
            tpSendCounts, xActiveMask, sharedExpertX, elasticInfo, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo,
            XOut, workspaceGM, tilingData);
        break;

    case 1:
        moe_combine_v2_a5<bfloat16_t, bfloat16_t>(
            expandX, expertIds, assistInfo, epSendCounts, expertScales,
            tpSendCounts, xActiveMask, sharedExpertX, elasticInfo, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo,
            XOut, workspaceGM, tilingData);
        break;

    default:
        AscendC::PRINTF("moe_distribute_combine_v2_a5 Error: invalid tilingKey = %d\n", tilingKey);
        break;
    }
}

void moe_distribute_combine_v2_a5_entry(int32_t tilingKey, uint32_t blockDim, void* stream,
    void* expandX, void* expertIds, void* assistInfo, void* epSendCounts,
    void* expertScales, void* tpSendCounts, void* xActiveMask,
    void* sharedExpertX, void* elasticInfo, void* oriX,
    void* constExpertAlpha1, void* constExpertAlpha2, void* constExpertV,
    void* performanceInfo, void* XOut, void* workspaceGM,
    void* mc2Context,
    MoeDistributeCombineV2Info tilingData)
{
    combine_a5_generic<<<blockDim, nullptr, stream>>>(
        tilingKey,
        (GM_ADDR)expandX, (GM_ADDR)expertIds, (GM_ADDR)assistInfo, (GM_ADDR)epSendCounts,
        (GM_ADDR)expertScales, (GM_ADDR)tpSendCounts, (GM_ADDR)xActiveMask,
        (GM_ADDR)sharedExpertX, (GM_ADDR)elasticInfo, (GM_ADDR)oriX,
        (GM_ADDR)constExpertAlpha1, (GM_ADDR)constExpertAlpha2, (GM_ADDR)constExpertV,
        (GM_ADDR)performanceInfo, (GM_ADDR)XOut, (GM_ADDR)workspaceGM,
        (GM_ADDR)mc2Context, tilingData
    );
}
