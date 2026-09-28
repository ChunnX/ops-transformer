/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "arch35/grouped_matmul_add_kernel_arch35.h"

extern "C" __global__ __aicore__ void pointer_grouped_matmul_add(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR yRefs, GM_ADDR yRefsOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)yRefs;
    (void)workspace;
    REGISTER_TILING_DEFAULT(GroupedMatmulAdd::GmmAddTilingDataParams);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (TILING_KEY_IS(10000900009000090001UL)) {
        GroupedMatmulAdd::GroupedMatMulAddKernel<asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn, false>(
            x, weight, groupList, yRefsOut, tiling);
    }
}
