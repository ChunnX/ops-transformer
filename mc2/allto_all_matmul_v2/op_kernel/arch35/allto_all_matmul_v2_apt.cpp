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
#include "allto_all_matmul_v2_tiling_key.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_urma_impl.h"
#if MC2_DFX_ENABLE
#include "mc2_runtime_info.h"
#endif

using namespace AscendC;
using namespace Apace;

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS, bool ISSMALLK, uint32_t COMMTYPE>
__global__ __aicore__ void allto_all_matmul_v2(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                               GM_ADDR x2_scale, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR workspaceGM,
                                               GM_ADDR tilingGM)
{
    (void)all2all_out;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(allToAllMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(allToAllMatmulTilingData, tilingData, tilingGM);
    AscendC::GlobalTensor<DTYPE_X1> x1DumpGM;
    x1DumpGM.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X1 *>(x1));
    AscendC::DumpTensor(x1DumpGM, 9001, 64);
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x1ScaleDumpGM;
    x1ScaleDumpGM.SetGlobalBuffer(reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(x1_scale));
    AscendC::DumpTensor(x1ScaleDumpGM, 9002, 64);
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            printf("workspaceGM=0x%lx tilingGM=0x%lx  context=0x%lx\n",
                   static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workspaceGM)),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tilingGM)),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context)));
        }
    }

    __gm__ CommContext *hcommCtx = reinterpret_cast<__gm__ CommContext *>(context);

#if (ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN)
    using Impl = AllToAllMxQuantMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, DTYPE_Y, false, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2 && ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)
    using Impl = AllToAllMxQuantMatmulUrmaImpl<fp8_e5m2_t, fp8_e5m2_t, DTYPE_Y, false, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)
    using Impl = AllToAllMxQuantMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e5m2_t, DTYPE_Y, false, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2 && ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN)
    using Impl = AllToAllMxQuantMatmulUrmaImpl<fp8_e5m2_t, fp8_e4m3fn_t, DTYPE_Y, false, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1 && ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)
    using Impl = AllToAllMxQuantMatmulUrmaImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, false, true>;
#else
    using Impl = AllToAllMxQuantMatmulUrmaImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, false, true>;
#endif

    Impl impl;
#if MC2_DFX_ENABLE
    Mc2Kernel::Mc2RuntimeInfo runtimeInfo;
    runtimeInfo.Init(workspaceGM, &tilingData.workspaceLayout, tilingData.tileQbmmTilingData.usedCoreNum);
    impl.SetRuntimeInfo(&runtimeInfo);
#endif
    impl.Init(hcommCtx, x1, x1_scale, x2, x2_scale, y, bias, &tilingData);
    impl.Run();
    trap();
}
