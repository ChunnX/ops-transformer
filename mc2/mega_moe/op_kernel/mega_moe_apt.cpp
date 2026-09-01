/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)
#define ENABLE_TENSOR_API
#endif
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#ifdef ENABLE_TENSOR_API
#include "arch92/mega_moe_a8w8_wave.h"
#endif

#include "arch92/mega_moe_tiling.h"
#include "arch92/mega_moe_tiling_key.h"

using namespace AscendC;

template <uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType,
          uint8_t CommModeType, bool TopkWeightsPrefetch>
__global__ __aicore__ void mega_moe(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
    GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR bias1,
    GM_ADDR bias2, GM_ADDR xActiveMask, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2,
    GM_ADDR sharedBias1, GM_ADDR sharedBias2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(MegaMoeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
#if defined(ENABLE_TENSOR_API) && defined(ORIG_DTYPE_X) && (ORIG_DTYPE_X == DT_BF16) && \
    defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y == DT_BF16) && defined(ORIG_DTYPE_WEIGHT1) && \
    ((ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E5M2) || (ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E4M3FN)) && \
    defined(ORIG_DTYPE_WEIGHT2) && (ORIG_DTYPE_WEIGHT2 == ORIG_DTYPE_WEIGHT1)
    if constexpr (CommModeType == TILINGKEY_TPL_MTE &&
                  DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
        constexpr bool isA8W8DtypePair =
            (DispatchQuantOutType == DISPATCH_QUANT_OUT_DTYPE_E5M2 &&
             Std::IsSame<DTYPE_WEIGHT1, fp8_e5m2_t>::value) ||
            (DispatchQuantOutType == DISPATCH_QUANT_OUT_DTYPE_E4M3FN &&
             Std::IsSame<DTYPE_WEIGHT1, fp8_e4m3fn_t>::value);
        if constexpr (isA8W8DtypePair) {
            if (tilingData.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W8_NZ) {
                MegaMoeImpl::MegaMoeArch92A8W8Wave<
                    DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1, DispatchQuantOutType,
                    CombineQuantOutType, TopkWeightsPrefetch, true>
                    op;
                op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask,
                        weightScales1, weightScales2, scales, sharedWeight1, sharedWeight2,
                        sharedWeightScales1, sharedWeightScales2, yOut, expertTokenNumsOut,
                        workspaceGM, &tilingData);
                op.Process();
            } else {
                MegaMoeImpl::MegaMoeArch92A8W8Wave<
                    DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1, DispatchQuantOutType,
                    CombineQuantOutType, TopkWeightsPrefetch, false>
                    op;
                op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask,
                        weightScales1, weightScales2, scales, sharedWeight1, sharedWeight2,
                        sharedWeightScales1, sharedWeightScales2, yOut, expertTokenNumsOut,
                        workspaceGM, &tilingData);
                op.Process();
            }
        }
    }
#endif
}
