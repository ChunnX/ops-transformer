/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_scatter_apt.cpp
 * \brief Kernel entry for QuantMatmulReduceScatter (MX-quant FP8, apace UDMA path, compute-first)
 */

#include <kernel_operator.h>
#include "quant_matmul_reduce_scatter_tiling_key.h"
#include "apace/kernel/fusions/matmul_reduce_scatter/quant_matmul_reduce_scatter_urma_tiling_data.h"
#include "apace/kernel/fusions/matmul_reduce_scatter/quant_matmul_reduce_scatter_urma_impl.h"

using namespace AscendC;
using namespace Apace;

template <uint32_t QUANTMODE>
__global__ __aicore__ void quant_matmul_reduce_scatter(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_scale,
                                                       GM_ADDR x2_scale, GM_ADDR y, GM_ADDR workspaceGM,
                                                       GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(QuantMatmulReduceScatterTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulReduceScatterTilingData, tilingData, tilingGM);

    __gm__ CommContext *hcommCtx = reinterpret_cast<__gm__ CommContext *>(context);

    // workspaceGM 前段为 AIC 写入 C_i 的缓冲区（[m, n] * sizeof(CType)，
    // 大小由 host tiling 计入 workspaces[0]），libApi workspace 附于其后
#if (ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_Y == DT_BF16)
    using Impl = QuantMatmulReduceScatterImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true>;
#else
#error "QuantMatmulReduceScatter only supports x1/x2 = FP8_E4M3FN, y = BF16"
#endif

    Impl impl;
    impl.Init(hcommCtx, x1, x1_scale, x2, x2_scale, workspaceGM, y, &tilingData);
    impl.Run();
}
