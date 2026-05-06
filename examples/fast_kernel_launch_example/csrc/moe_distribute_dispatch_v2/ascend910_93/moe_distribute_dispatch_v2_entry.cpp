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
 * \file moe_distribute_dispatch_v2.cpp
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
#include "moe_distribute_dispatch_v2_entry.h"

using namespace MoeDistributeDispatchV2Impl;
using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Kernel;
using namespace AscendC;

/*
* A3 tilingkey说明
* 5位的十进制数
* 第1位（个位）：quantMode:
*     0: 不量化, 1: 静态量化, 2: 动态量化
* 第2位（十位）：x输入类型:
*     0: float16, 1: bfloat16
* 第3位（百位）：是否有smoothScale:
*     0: 无, 1: 有
* 第4位（千位）：是否走fullmesh_v2模板:
*     0: 不做, 1: 做
* 第5位（万位）：无实际含义
*/

template<typename XType, typename ExpandxType, int32_t QuantMode, bool IsSmoothScaleExist>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeDispatchV2Info tilingData)
{

    TPipe pipe;
    MoeDistributeDispatchV2<XType, ExpandxType, QuantMode, IsSmoothScaleExist> op;
    op.Init(mc2Context, x, expertIds, scales, xActiveMask, performanceInfo, expandXOut, dynamicScalesOut,
        assistInfoOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingData, &pipe);
        
    op.Process();

    pipe.DestroyWithoutPipeAll();
    return;
}

template<typename XType, typename ExpandxType, int32_t QuantMode, bool IsSmoothScaleExist>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_v2_full_mesh(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeDispatchV2Info tilingData)
{

    TPipe pipe;
    MoeDistributeDispatchV2FullMesh<XType, ExpandxType, QuantMode, IsSmoothScaleExist> op;
    op.Init(x, expertIds, scales, xActiveMask, performanceInfo, expandXOut, dynamicScalesOut,
        assistInfoOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, mc2Context, tilingData, &pipe);
        
    op.Process();

    pipe.DestroyWithoutPipeAll();
    return;
}

template<int32_t tilingKey>
__aicore__ inline void moe_dispatch_switch(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context,
    MoeDistributeDispatchV2Info tilingData)
{
    if constexpr (tilingKey == 10000) {
        moe_distribute_dispatch_v2<float16_t, float16_t, MoeDistributeDispatchV2Impl::UNQUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 10002) {
        moe_distribute_dispatch_v2<float16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 10010) {
        moe_distribute_dispatch_v2<bfloat16_t, bfloat16_t, MoeDistributeDispatchV2Impl::UNQUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 10012) {
        moe_distribute_dispatch_v2<bfloat16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 10100) {
        moe_distribute_dispatch_v2<float16_t, float16_t, MoeDistributeDispatchV2Impl::UNQUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 10102) {
        moe_distribute_dispatch_v2<float16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 10110) {
        moe_distribute_dispatch_v2<bfloat16_t, bfloat16_t, MoeDistributeDispatchV2Impl::UNQUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 10112) {
        moe_distribute_dispatch_v2<bfloat16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);

    /* ---------- FullMesh ---------- */

    } else if constexpr (tilingKey == 11000) {
        moe_distribute_dispatch_v2_full_mesh<float16_t, float16_t, MoeDistributeDispatchV2Impl::UNQUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 11002) {
        moe_distribute_dispatch_v2_full_mesh<float16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 11010) {
        moe_distribute_dispatch_v2_full_mesh<bfloat16_t, bfloat16_t, MoeDistributeDispatchV2Impl::UNQUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 11012) {
        moe_distribute_dispatch_v2_full_mesh<bfloat16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, false>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 11100) {
        moe_distribute_dispatch_v2_full_mesh<float16_t, float16_t, MoeDistributeDispatchV2Impl::UNQUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 11102) {
        moe_distribute_dispatch_v2_full_mesh<float16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 11110) {
        moe_distribute_dispatch_v2_full_mesh<bfloat16_t, bfloat16_t, MoeDistributeDispatchV2Impl::UNQUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else if constexpr (tilingKey == 11112) {
        moe_distribute_dispatch_v2_full_mesh<bfloat16_t, int8_t, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, true>(
            x, expertIds, scales, xActiveMask, expertScales, performanceInfo,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
    } else {
        AscendC::PRINTF("moe_distribute_dispatch_v2 Error: invalid tilingKey = %d\n", tilingKey);
    }
}

template<int32_t tilingKey>
__global__ __aicore__ void moe_distribute_dispatch_v2_generic(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeDispatchV2Info tilingData)
{
    // 根据不同的数据类型调用不同的模板
    moe_dispatch_switch<tilingKey>(
        x, expertIds, scales, xActiveMask, expertScales,
        performanceInfo, expandXOut, dynamicScalesOut,
        assistInfoOut, expertTokenNumsOut, epSendCountsOut,
        expandScalesOut, workspaceGM, mc2Context, tilingData);
}

template <int32_t split_num, int32_t tilingKey>
__sk__ __vector__ void moe_distribute_dispatch_v2_generic_sk(MoeDistributeDispatchV2SkArgs *args)
{
    GM_ADDR x = args->x;
    GM_ADDR expertIds = args->expertIds;
    GM_ADDR scales = args->scales;
    GM_ADDR xActiveMask = args->xActiveMask;
    GM_ADDR expertScales = args->expertScales;
    GM_ADDR performanceInfo = args->performanceInfo;
    GM_ADDR expandXOut = args->expandXOut;
    GM_ADDR dynamicScalesOut = args->dynamicScalesOut;
    GM_ADDR assistInfoOut = args->assistInfoOut;
    GM_ADDR expertTokenNumsOut = args->expertTokenNumsOut;
    GM_ADDR epSendCountsOut = args->epSendCountsOut;
    GM_ADDR expandScalesOut = args->expandScalesOut;
    GM_ADDR workspaceGM = args->workspaceGM;
    GM_ADDR mc2Context = args->mc2Context;
    MoeDistributeDispatchV2Info tilingData = args->tilingData;

    // 根据不同的数据类型调用不同的模板
    moe_dispatch_switch<tilingKey>(
        x, expertIds, scales, xActiveMask, expertScales,
        performanceInfo, expandXOut, dynamicScalesOut,
        assistInfoOut, expertTokenNumsOut, epSendCountsOut,
        expandScalesOut, workspaceGM, mc2Context, tilingData);
}

SK_BIND((moe_distribute_dispatch_v2_generic<10000>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10000>, moe_distribute_dispatch_v2_generic_sk<1, 10000>, moe_distribute_dispatch_v2_generic_sk<2, 10000>, moe_distribute_dispatch_v2_generic_sk<3, 10000>);
SK_BIND((moe_distribute_dispatch_v2_generic<10002>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10002>, moe_distribute_dispatch_v2_generic_sk<1, 10002>, moe_distribute_dispatch_v2_generic_sk<2, 10002>, moe_distribute_dispatch_v2_generic_sk<3, 10002>);
SK_BIND((moe_distribute_dispatch_v2_generic<10010>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10010>, moe_distribute_dispatch_v2_generic_sk<1, 10010>, moe_distribute_dispatch_v2_generic_sk<2, 10010>, moe_distribute_dispatch_v2_generic_sk<3, 10010>);
SK_BIND((moe_distribute_dispatch_v2_generic<10012>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10012>, moe_distribute_dispatch_v2_generic_sk<1, 10012>, moe_distribute_dispatch_v2_generic_sk<2, 10012>, moe_distribute_dispatch_v2_generic_sk<3, 10012>);
SK_BIND((moe_distribute_dispatch_v2_generic<10100>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10100>, moe_distribute_dispatch_v2_generic_sk<1, 10100>, moe_distribute_dispatch_v2_generic_sk<2, 10100>, moe_distribute_dispatch_v2_generic_sk<3, 10100>);
SK_BIND((moe_distribute_dispatch_v2_generic<10102>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10102>, moe_distribute_dispatch_v2_generic_sk<1, 10102>, moe_distribute_dispatch_v2_generic_sk<2, 10102>, moe_distribute_dispatch_v2_generic_sk<3, 10102>);
SK_BIND((moe_distribute_dispatch_v2_generic<10110>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10110>, moe_distribute_dispatch_v2_generic_sk<1, 10110>, moe_distribute_dispatch_v2_generic_sk<2, 10110>, moe_distribute_dispatch_v2_generic_sk<3, 10110>);
SK_BIND((moe_distribute_dispatch_v2_generic<10112>), 4, moe_distribute_dispatch_v2_generic_sk<0, 10112>, moe_distribute_dispatch_v2_generic_sk<1, 10112>, moe_distribute_dispatch_v2_generic_sk<2, 10112>, moe_distribute_dispatch_v2_generic_sk<3, 10112>);
SK_BIND((moe_distribute_dispatch_v2_generic<11000>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11000>, moe_distribute_dispatch_v2_generic_sk<1, 11000>, moe_distribute_dispatch_v2_generic_sk<2, 11000>, moe_distribute_dispatch_v2_generic_sk<3, 11000>);
SK_BIND((moe_distribute_dispatch_v2_generic<11002>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11002>, moe_distribute_dispatch_v2_generic_sk<1, 11002>, moe_distribute_dispatch_v2_generic_sk<2, 11002>, moe_distribute_dispatch_v2_generic_sk<3, 11002>);
SK_BIND((moe_distribute_dispatch_v2_generic<11010>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11010>, moe_distribute_dispatch_v2_generic_sk<1, 11010>, moe_distribute_dispatch_v2_generic_sk<2, 11010>, moe_distribute_dispatch_v2_generic_sk<3, 11010>);
SK_BIND((moe_distribute_dispatch_v2_generic<11012>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11012>, moe_distribute_dispatch_v2_generic_sk<1, 11012>, moe_distribute_dispatch_v2_generic_sk<2, 11012>, moe_distribute_dispatch_v2_generic_sk<3, 11012>);
SK_BIND((moe_distribute_dispatch_v2_generic<11100>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11100>, moe_distribute_dispatch_v2_generic_sk<1, 11100>, moe_distribute_dispatch_v2_generic_sk<2, 11100>, moe_distribute_dispatch_v2_generic_sk<3, 11100>);
SK_BIND((moe_distribute_dispatch_v2_generic<11102>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11102>, moe_distribute_dispatch_v2_generic_sk<1, 11102>, moe_distribute_dispatch_v2_generic_sk<2, 11102>, moe_distribute_dispatch_v2_generic_sk<3, 11102>);
SK_BIND((moe_distribute_dispatch_v2_generic<11110>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11110>, moe_distribute_dispatch_v2_generic_sk<1, 11110>, moe_distribute_dispatch_v2_generic_sk<2, 11110>, moe_distribute_dispatch_v2_generic_sk<3, 11110>);
SK_BIND((moe_distribute_dispatch_v2_generic<11112>), 4, moe_distribute_dispatch_v2_generic_sk<0, 11112>, moe_distribute_dispatch_v2_generic_sk<1, 11112>, moe_distribute_dispatch_v2_generic_sk<2, 11112>, moe_distribute_dispatch_v2_generic_sk<3, 11112>);

// <<<>>>调用函数
void moe_distribute_dispatch_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR performanceInfo, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context,
    MoeDistributeDispatchV2Info tilingData)
{
    switch (tilingKey) {
    case 10000: moe_distribute_dispatch_v2_generic<10000><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 10002: moe_distribute_dispatch_v2_generic<10002><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 10010: moe_distribute_dispatch_v2_generic<10010><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 10012: moe_distribute_dispatch_v2_generic<10012><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 10100: moe_distribute_dispatch_v2_generic<10100><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 10102: moe_distribute_dispatch_v2_generic<10102><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 10110: moe_distribute_dispatch_v2_generic<10110><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 10112: moe_distribute_dispatch_v2_generic<10112><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11000: moe_distribute_dispatch_v2_generic<11000><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11002: moe_distribute_dispatch_v2_generic<11002><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11010: moe_distribute_dispatch_v2_generic<11010><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11012: moe_distribute_dispatch_v2_generic<11012><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11100: moe_distribute_dispatch_v2_generic<11100><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11102: moe_distribute_dispatch_v2_generic<11102><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11110: moe_distribute_dispatch_v2_generic<11110><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    case 11112: moe_distribute_dispatch_v2_generic<11112><<<blockDim, nullptr, stream>>>(x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData); break;
    default: AscendC::PRINTF("moe_distribute_dispatch_v2 Error: invalid tilingKey = %d\n", tilingKey); break;
    }
}
