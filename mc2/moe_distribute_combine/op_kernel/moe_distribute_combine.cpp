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
 * \file moe_distribute_combine.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine.h"
#include "moe_distribute_combine_tiling.h"
#include "moe_distribute_combine_a2.h"
#include "moe_distribute_combine_a2_layered.h"
#include "moe_distribute_combine_a2_layered_aicpu.h"
using namespace MoeDistributeCombineImpl;
using namespace MoeDistributeCombineA2Impl;
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                                             GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                             GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                             GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR XOut, 
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
  REGISTER_TILING_DEFAULT(MoeDistributeCombineA2TilingData);
  REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", MoeDistributeCombineTilingData);
  REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 2000) || (TILING_KEY_VAR == 3000)", MoeDistributeCombineA2TilingData);

  TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16) 
  if (TILING_KEY_IS(1100)) { // tp=2
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, true, false> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(1000)) { // tp=1
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, false, false> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(1120)) { // tp=2
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, true, true> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(1020)) { // tp=1
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, false, true> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
  }
  if (TILING_KEY_IS(2000)) {
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
    MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, scales, xActiveMask, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
  }
  if (TILING_KEY_IS(3000)) {
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
    if (dataplaneMode == DataplaneMode::AICPU) {
      MoeDistributeCombineA2LayeredAicpu<DTYPE_EXPAND_X, int32_t> op;
      op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
        contextGM0);
      op.Process();
    } else if (dataplaneMode == DataplaneMode::AIV) {
      MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, DTYPE_EXPAND_X> op;
      op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
        contextGM0);
      op.Process();
    }
  } else if (TILING_KEY_IS(3100)) {
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
    if (dataplaneMode == DataplaneMode::AIV) {
      MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, int8_t> op;
      op.Init(expandX, expertIds, expandIdx, epSendCount, expandScales, XOut, workspaceGM, &pipe, &tilingData,
        contextGM0);
      op.Process();
    }
  }
#endif
}