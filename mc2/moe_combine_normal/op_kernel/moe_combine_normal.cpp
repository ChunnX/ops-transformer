/*

 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: Normal combine function device implementation file
 * Create: 2025-11-28
 * Note:
 * History: 2025-11-28 create normal combine function file in device part
*/
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_combine_normal.h"
#include "moe_combine_normal_a5.h"
#include "moe_combine_normal_multi_round.h"
#include "moe_combine_normal_tiling.h"
using namespace AscendC;
using namespace MoeCombineNormalMultiRoundImpl;
using namespace MoeCombineNormalImpl;
using namespace MoeCombineNormalA5Impl;

/*
5: INIT
4: A5/A3/A2
*/
#define TILINGKEY_A3_MULTI_ROUND 13001
#define TILINGKEY_A3_SINGLE_ROUND 13000
#define TILINGKEY_A5_MULTI_ROUND 15001
#define TILINGKEY_A5_SINGLE_ROUND 15000

extern "C" __global__ __aicore__ void moe_combine_normal(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount,
                                                 GM_ADDR topkWeights, GM_ADDR tokenIdx, GM_ADDR tpRecvCount,
                                                 GM_ADDR XOut, GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM,
                                                 GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeCombineNormalTilingData);
    TPipe pipe;

#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    GET_TILING_DATA_WITH_STRUCT(MoeCombineNormalTilingData, tilingData, tilingGM);
    if (TILING_KEY_IS(TILINGKEY_A3_MULTI_ROUND)) {
        MoeCombineNormalMultiRound<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
        workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_A3_SINGLE_ROUND)) {
        MoeCombineNormal<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
        workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_A5_SINGLE_ROUND)) {
        MoeCombineNormalA5<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
        workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}