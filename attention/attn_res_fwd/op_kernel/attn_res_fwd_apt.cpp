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
 * \file attn_res_fwd_apt.cpp
 * \brief AttnResFwd A5(ascend950) kernel entry — same logic as A2, includes arch35/
 */
#include "arch35/attn_res_fwd_reload.h"
#include "arch35/attn_res_fwd_resident.h"
#include "attn_res_fwd_tiling_data.h"
#include "tiling_key_attn_res_fwd.h"

using namespace AscendC;
using namespace AttnResFwd;

// 输入 dtype 由编译期宏注入（框架按 def 的 DataType 组合为每个 dtype 生成独立二进制，
// 并以 -DDTYPE_PREFIX_SUM=<type> 编译，参考 moe_token_permute）；未注入时默认 BF16。
#if !defined(DTYPE_PREFIX_SUM)
#define DTYPE_PREFIX_SUM bfloat16_t
#endif

extern "C" __global__ __aicore__ void attn_res_fwd(GM_ADDR prefixSum, GM_ADDR blockResidual, GM_ADDR projWeight,
                                                   GM_ADDR normWeight, GM_ADDR hiddenStates, GM_ADDR invRms,
                                                   GM_ADDR probs, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AttnResFwdTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspaceGM; // 无用户 FP32-v Workspace；仅系统预留

    TPipe pipe;
    AttnResFwdInitParams initParams{prefixSum, blockResidual, projWeight, normWeight, hiddenStates, invRms, probs};

    // 按 tiling key 分派算法分支（RELOAD/RESIDENT）；dtype 由编译期宏 DTYPE_PREFIX_SUM 决定
    if (TILING_KEY_IS(TILING_KEY_RELOAD)) {
        AttnResFwdReload<DTYPE_PREFIX_SUM> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RESIDENT)) {
        AttnResFwdResident<DTYPE_PREFIX_SUM> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    }
}
