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
#include "combine_normal_tiling_key.h"
using namespace AscendC;
using namespace MoeCombineNormalMultiRoundImpl;
using namespace MoeCombineNormalImpl;
using namespace MoeCombineNormalA5Impl;
using namespace Mc2Tiling;

extern "C" __global__ __aicore__ void combine_normal(GM_ADDR context, GM_ADDR recvX, GM_ADDR tokenSrcInfo,
                                                     GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR tokenIdx,
                                                     GM_ADDR tpRecvCount, GM_ADDR XOut, GM_ADDR sendCostStatsOut,
                                                     GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)context;
    REGISTER_TILING_DEFAULT(MoeCombineNormalTilingData);
    TPipe pipe;

#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    GET_TILING_DATA_WITH_STRUCT(MoeCombineNormalTilingData, tilingData, tilingGM);
    if (TILING_KEY_IS(TILINGKEY_COMBINE_NORMAL_A3_MULTI)) {
        MoeCombineNormalMultiRound<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_COMBINE_NORMAL_A3_SINGLE)) {
        MoeCombineNormal<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_COMBINE_NORMAL_A5_MULTI)) {
        MoeCombineNormalMultiRound<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_COMBINE_NORMAL_A5_SINGLE)) {
        MoeCombineNormalA5<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
