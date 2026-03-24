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
 * \file mega_moe.cpp
 * \brief
 */
#include "mega_moe.h"
#include "kernel_operator.h"

using namespace MegaMoeImpl;

extern "C" __global__ __aicore__ void mega_moe(
  GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
  GM_ADDR weight1, GM_ADDR weightScales1, GM_ADDR weight2, GM_ADDR weightScales2,
  GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
  REGISTER_TILING_DEFAULT(MegaMoeTilingData);

  GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
  MegaMoe op;
  op.Init(mc2Context, x, expertIdx, scales, xActiveMask, weight1, expertScales, weightScales1, weight2,
    weightScales2, y, workspaceGM, &tilingData);
  op.Process();
}