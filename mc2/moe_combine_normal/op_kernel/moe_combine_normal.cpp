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
 * \file moe_combine_normal.cpp
 * \brief Normal combine function device implementation file
 */

#include "moe_combine_normal.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_combine_normal_tiling.h"

using namespace AscendC;
using namespace MoeCombineNormalImpl;

extern "C" __global__ __aicore__ void moe_combine_normal(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount,
                                                         GM_ADDR topkWeights, GM_ADDR tpRecvCount, GM_ADDR XOut,
                                                         GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM,
                                                         GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeCombineNormalTilingData);
    TPipe pipe;

#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    GET_TILING_DATA_WITH_STRUCT(MoeCombineNormalTilingData, tilingData, tilingGM);
    MoeCombineNormal<DTYPE_RECV_X, DTYPE_X, int32_t> op;
    op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tpRecvCount, XOut, sendCostStatsOut, workspaceGM, &pipe,
            &tilingData);
    op.Process();
#endif
}