/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

/*!
 * \file all_gather_quant_matmul.cpp
 * \brief Kernel entry: AllGather + Matmul fusion
 *
 * Flow:
 *   AIV: CopyLocalData → SetRemoteFlag → WaitRemoteFlag → ExecuteAllGather
 *   AIC: MatmulProcess (TensorAPI MX quantized matmul)
 *
 * Inputs: pre-quantized fp8 A per rank, fp8 B, e8m0 scales
 * Output: bf16 C per rank
 */

#include "basic_api/kernel_basic_intf.h"
#include "kernel/all_gather_matmul_impl.h"
#include "all_gather_matmul_tiling_data.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void
all_gather_quant_matmul(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale,
                        GM_ADDR output, GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
                        GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(allGatherMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(allGatherMatmulTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if (TILING_KEY_IS(0)) {
        // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
        // split-K hard sync
        allGatherMatmulImpl::allGatherMatmulImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e8m0_t, 0> impl;
        impl.Init(x1, x2, x1Scale, x2Scale, output, allGatherDataOut, allGatherScalesOut, tilingGM);
        impl.Process();
        return;
    }
    if (TILING_KEY_IS(1)) {
        // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
        // non-split-K (NO_TILE_K)
        allGatherMatmulImpl::allGatherMatmulImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e8m0_t, 1> impl;
        impl.Init(x1, x2, x1Scale, x2Scale, output, allGatherDataOut, allGatherScalesOut, tilingGM);
        impl.Process();
        return;
    }
}
