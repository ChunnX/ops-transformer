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
 * \file dispatch_layout.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "dispatch_layout.h"
#include "dispatch_layout_tiling.h"
#include "dispatch_layout_tiling_key.h"

using namespace AscendC;
using namespace Mc2Kernel;

__global__ __aicore__ void dispatch_layout(
    GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert, GM_ADDR isTokenInRank,
    GM_ADDR notifySendData, GM_ADDR sendTokenIdxSmall, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DispatchLayoutTilingData);
    GET_TILING_DATA_WITH_STRUCT(DispatchLayoutTilingData, tilingData, tiling);

    TPipe pipe;

    MoeDispatchLayout::DispatchLayout<int32_t> op;
    op.Init(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData, sendTokenIdxSmall,
            workspace, &pipe, &tilingData);
    op.Process();

}
