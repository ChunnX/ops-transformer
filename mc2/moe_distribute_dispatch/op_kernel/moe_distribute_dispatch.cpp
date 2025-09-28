/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "moe_distribute_dispatch_tiling.h"
#include "moe_distribute_dispatch_a2.h"
#include "moe_distribute_dispatch_a2_layered.h"
#include "moe_distribute_dispatch_a2_layered_aicpu.h"
#include "moe_distribute_dispatch.h"
using namespace MoeDistributeDispatchImpl;
using namespace MoeDistributeDispatchA2Impl;

using namespace AscendC;

extern "C" __global__ __aicore__ void moe_distribute_dispatch(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchA2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000000000", MoeDistributeDispatchTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 2000000000", MoeDistributeDispatchA2TilingData);

    TPipe pipe;
    
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(1000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1100)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, true> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2000001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2100001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(1011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, true, false, false, false> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1111)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, true, false, false, true> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1102)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, false, true> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1112)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingData, tilingData, tilingGM);
        MoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, true, true> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2000001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2000001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2100001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    } else if (TILING_KEY_IS(2100001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2LayeredAicpu<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
            op.Init(x, expertIds, scales, expertScales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, 
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        }
    }
#endif
}