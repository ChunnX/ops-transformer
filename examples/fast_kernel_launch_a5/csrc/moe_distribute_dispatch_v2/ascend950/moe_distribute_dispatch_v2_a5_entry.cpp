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
 * \file moe_distribute_dispatch_v2_a5_entry.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/moe_distribute_dispatch_v2.h"
#include "op_kernel/moe_distribute_dispatch_v2_full_mesh.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"


using namespace MoeDistributeDispatchV2Impl;
using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Kernel;
using namespace AscendC;

template<typename XType, typename ExpandXType>
__attribute__((always_inline)) __aicore__ __inline__ void dispatch_a5_no_fullmesh(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, MoeDistributeDispatchV2Info tilingData)
{
    TPipe pipe;
    MoeDistributeDispatchV2TilingData td{};
    td.moeDistributeDispatchV2Info = tilingData;
    MoeDistributeDispatchV2<XType, ExpandXType, UNQUANT, false, false> op;
    op.Init((GM_ADDR)nullptr, x, expertIds, scales, xActiveMask, expertScales,
            performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut,
            expertTokenNumsOut, epSendCountsOut, expandScalesOut,
            workspaceGM, &pipe, &td);

    op.Process();

    pipe.DestroyWithoutPipeAll();
}

template<typename XType, typename ExpandXType>
__attribute__((always_inline)) __aicore__ __inline__ void dispatch_a5_fullmesh(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, MoeDistributeDispatchV2Info tilingData)
{
    TPipe pipe;
    MoeDistributeDispatchV2TilingData td{};
    td.moeDistributeDispatchV2Info = tilingData;
    MoeDistributeDispatchV2FullMesh<XType, ExpandXType, UNQUANT, false, false> op;
    op.Init((GM_ADDR)nullptr, x, expertIds, scales, xActiveMask, expertScales,
            performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut,
            expertTokenNumsOut, epSendCountsOut, expandScalesOut,
            workspaceGM, &pipe, &td);

    op.Process();

    pipe.DestroyWithoutPipeAll();
}

__aicore__ inline void moe_dispatch_a5_switch(
    int32_t tilingKey,
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
    MoeDistributeDispatchV2Info tilingData)
{
    switch (tilingKey) {
    case 0:
        dispatch_a5_no_fullmesh<half, half>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, tilingData);
        break;

    case 1:
        dispatch_a5_no_fullmesh<bfloat16_t, bfloat16_t>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, tilingData);
        break;

    case 100:
        dispatch_a5_fullmesh<half, half>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, tilingData);
        break;

    case 101:
        dispatch_a5_fullmesh<bfloat16_t, bfloat16_t>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, tilingData);
        break;

    default:
        AscendC::PRINTF("moe_distribute_dispatch_v2_a5 Error: invalid tilingKey = %d\n", tilingKey);
        break;
    }
}

extern "C" __global__ __aicore__ void dispatch_a5_generic(
    int32_t tilingKey,
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context,
    MoeDistributeDispatchV2Info tilingData)
{
    moe_dispatch_a5_switch(
        tilingKey,
        x, expertIds, scales, xActiveMask, expertScales,
        performanceInfo, expandXOut, dynamicScalesOut,
        assistInfoOut, expertTokenNumsOut, epSendCountsOut,
        expandScalesOut, workspaceGM, tilingData);
}

void moe_distribute_dispatch_v2_a5_entry(int32_t tilingKey, uint32_t blockDim, void* stream,
    void* x, void* expertIds,
    void* scales, void* xActiveMask, void* expertScales, void* performanceInfo,
    void* expandXOut, void* dynamicScalesOut, void* assistInfoOut,
    void* expertTokenNumsOut, void* epSendCountsOut,
    void* expandScalesOut, void* workspaceGM, void* mc2Context,
    MoeDistributeDispatchV2Info tilingData)
{
    dispatch_a5_generic<<<blockDim, nullptr, stream>>>(
        tilingKey, (GM_ADDR)x, (GM_ADDR)expertIds, (GM_ADDR)scales,
        (GM_ADDR)xActiveMask, (GM_ADDR)expertScales,
        (GM_ADDR)performanceInfo, (GM_ADDR)expandXOut, (GM_ADDR)dynamicScalesOut,
        (GM_ADDR)assistInfoOut, (GM_ADDR)expertTokenNumsOut, (GM_ADDR)epSendCountsOut,
        (GM_ADDR)expandScalesOut, (GM_ADDR)workspaceGM, (GM_ADDR)mc2Context, tilingData
    );
}
