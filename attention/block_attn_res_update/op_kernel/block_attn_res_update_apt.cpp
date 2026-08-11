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
 * \file block_attn_res_update_apt.cpp
 * \brief BlockAttnResUpdate arch35 kernel entry point.
 */

#include "kernel_operator.h"
#include "arch35/block_attn_res_update.h"
#include "arch35/block_attn_res_update_tiling_key.h"

using namespace AscendC;

template <int8_t RESERVED_MODE>
__global__ __aicore__ void block_attn_res_update(GM_ADDR partialBlock, GM_ADDR partialDelta, GM_ADDR effectiveQuery,
                                                 GM_ADDR interMax, GM_ADDR interExpSum, GM_ADDR interNumerator,
                                                 GM_ADDR h, GM_ADDR partialBlocks, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BlockAttnResUpdateTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(BlockAttnResUpdateTilingData, tilingData, tiling);

    TPipe pipe;
    BlockAttnResUpdate::BlockAttnResUpdateKernel<RESERVED_MODE> op;
    op.Init(partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator, h, partialBlocks,
            workspace, &tilingData, &pipe);
    op.Process();
}

