/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <kernel_operator.h>
#include "matmul_allto_all_v2_tiling_key.h"
#include "apace/kernel/fusions/quant_matmul_all_to_all/matmul_all_to_all_tiling_data.h"
#include "apace/kernel/fusions/quant_matmul_all_to_all/mx_quant_matmul_all_to_all_urma_impl.h"

using namespace AscendC;
using namespace Apace;

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS, bool ISSMALLK, uint32_t COMMTYPE>
__global__ __aicore__ void matmul_allto_all_v2(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                               GM_ADDR x2_scale, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(MatmulAllToAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAllToAllTilingData, tilingData, tilingGM);
    __gm__ CommContext *hcommCtx = reinterpret_cast<__gm__ CommContext *>(context);

#if (ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN)
    using Impl = Apace::MatmulAllToAllMxImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, DTYPE_Y, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2 && ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)
    using Impl = Apace::MatmulAllToAllMxImpl<fp8_e5m2_t, fp8_e5m2_t, DTYPE_Y, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)
    using Impl = Apace::MatmulAllToAllMxImpl<fp8_e4m3fn_t, fp8_e5m2_t, DTYPE_Y, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2 && ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN)
    using Impl = Apace::MatmulAllToAllMxImpl<fp8_e5m2_t, fp8_e4m3fn_t, DTYPE_Y, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1 && ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)
    using Impl = Apace::MatmulAllToAllMxImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, true>;
#else
    using Impl = Apace::MatmulAllToAllMxImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, true>;
#endif

    Impl impl;
    impl.Init(hcommCtx, x1, x1_scale, x2, x2_scale, bias, y, &tilingData);
    impl.Run();
}
