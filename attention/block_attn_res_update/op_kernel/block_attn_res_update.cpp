/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/block_attn_res_update.h"
#include "arch35/block_attn_res_update_tiling_data.h"

// Six tensor inputs + two tensor outputs + workspace + tiling.
// ACLNNGraph aliases partial_block_ref with partial_block. Direct kernel launchers must pass the same GM address in
// both ABI slots. This kernel writes through partial_block and does not use workspace storage.
extern "C" __global__ __aicore__ void block_attn_res_update(
    GM_ADDR partial_block, GM_ADDR delta, GM_ADDR inter_numerator, GM_ADDR inter_max, GM_ADDR inter_exp_sum,
    GM_ADDR effective_query, GM_ADDR partial_block_ref, GM_ADDR h, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)partial_block_ref;
    (void)workspace;
    REGISTER_NONE_TILING;
    GET_TILING_DATA_WITH_STRUCT(BlockAttnResUpdateTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    BlockAttnResUpdateOps::BlockAttnResUpdateKernel op;
    op.Init(partial_block, delta, inter_numerator, inter_max, inter_exp_sum, effective_query, h, &tilingData);
    op.Process();
}
