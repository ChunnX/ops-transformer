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
 * \file moe_update_expert.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "moe_update_expert.h"

using namespace AscendC;
using namespace MoeUpdateExpertNamespace;

#define INVOKE_MOE_UPDATE_EXPERT_OP_IMPL()                                                                      \
    do {                                                                                                        \
        op.Init(                                                                                                \
            expertIdsGM, eplbTableGM, expertScalesGM, pruningThresholdGM, activeMaskGM, balancedExpertIdsOutGM, \
            balancedActiveMaskOutGM, workspaceGM, &tilingData, &pipe);                                          \
        op.Process();                                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void moe_update_expert(
    GM_ADDR expertIdsGM, GM_ADDR eplbTableGM, GM_ADDR expertScalesGM, GM_ADDR pruningThresholdGM, GM_ADDR activeMaskGM,
    GM_ADDR balancedExpertIdsOutGM, GM_ADDR balancedActiveMaskOutGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeUpdateExpertTilingData);
    auto tiling = (__gm__ MoeUpdateExpertTilingData*)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        // 不使能专家裁剪，按rank_id负载均衡
        MoeUpdateExpert<DTYPE_EXPERT_IDS, float, false> op;
        INVOKE_MOE_UPDATE_EXPERT_OP_IMPL();
    } else if (TILING_KEY_IS(1)) {
        // 使能专家裁剪，按token_id负载均衡，expert_scales 参数类型 float
        MoeUpdateExpert<DTYPE_EXPERT_IDS, float, true> op;
        INVOKE_MOE_UPDATE_EXPERT_OP_IMPL();
    } else if (TILING_KEY_IS(11)) {
        // 使能专家裁剪，按token_id负载均衡，expert_scales 参数类型 half
        MoeUpdateExpert<DTYPE_EXPERT_IDS, half, true> op;
        INVOKE_MOE_UPDATE_EXPERT_OP_IMPL();
    } else if (TILING_KEY_IS(21)) {
        // 使能专家裁剪，按token_id负载均衡，expert_scales 参数类型 bfloat16_t
        MoeUpdateExpert<DTYPE_EXPERT_IDS, bfloat16_t, true> op;
        INVOKE_MOE_UPDATE_EXPERT_OP_IMPL();
    }
}