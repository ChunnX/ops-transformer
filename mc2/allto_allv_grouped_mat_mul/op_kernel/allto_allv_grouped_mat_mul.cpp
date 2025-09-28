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
 * \file allto_allv_grouped_mat_mul.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "allto_allv_grouped_mat_mul_coarse_grained.h"

using namespace AscendC;

#define INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL()                                                                 \
    do {                                                                                                          \
        op.Init(                                                                                                  \
            gmmxGM, gmmweightGM, sendCountsTensorOptionalGM, recvCountsTensorOptionalGM, mmxOptionalGM,           \
            mmweightOptionalGM, gmmyGM, mmyOptionalGM, permuteOutOptionalGM, workspaceGM, contextGM, &tilingData, \
            hcclInitTiling, alltoAllvCcTiling, &pipe);                                                            \
        op.Process();                                                                                             \
    } while (0)

extern "C" __global__ __aicore__ void allto_allv_grouped_mat_mul(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
    GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
    GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    auto tiling = (__gm__ AlltoAllvGmmTilingData*)tilingGM;
    __gm__ void* hcclInitTiling = (__gm__ void*)(&(tiling->hcclInitTiling));
    __gm__ void* alltoAllvCcTiling = (__gm__ void*)(&(tiling->alltoAllvCcTiling));
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

    if (TILING_KEY_IS(0)) {
        // BF16 + no mm + no gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<bfloat16_t, false, false, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(10)) {
        // BF16 + no mm + gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<bfloat16_t, false, true, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(100)) {
        // BF16 + mm + no gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<bfloat16_t, true, false, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(101)) {
        // BF16 + mm + no gmmweight trans + mmweight trans
        AlltoAllvGmmCoarseGrained<bfloat16_t, true, false, true> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(110)) {
        // BF16 + mm + gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<bfloat16_t, true, true, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(111)) {
        // BF16 + mm + gmmweight trans + mmweight trans
        AlltoAllvGmmCoarseGrained<bfloat16_t, true, true, true> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(1000)) {
        // FP16 + no mm + no gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<half, false, false, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(1010)) {
        // FP16 + no mm + gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<half, false, true, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(1100)) {
        // FP16 + mm + no gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<half, true, false, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(1101)) {
        // FP16 + mm + no gmmweight trans + mmweight trans
        AlltoAllvGmmCoarseGrained<half, true, false, true> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(1110)) {
        // FP16 + mm + gmmweight trans + no mmweight trans
        AlltoAllvGmmCoarseGrained<half, true, true, false> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    } else if (TILING_KEY_IS(1111)) {
        // FP16 + mm + gmmweight trans + mmweight trans
        AlltoAllvGmmCoarseGrained<half, true, true, true> op;
        INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL();
        return;
    }

    return;
}