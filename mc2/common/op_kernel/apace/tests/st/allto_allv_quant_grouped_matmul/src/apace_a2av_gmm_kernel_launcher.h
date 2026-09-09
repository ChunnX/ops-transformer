/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apace_a2av_gmm_kernel_launcher.h
 * \brief AllToAllvQuantGroupedMatmul kernel launcher 入口
 */

#pragma once

#include "apace/kernel/fusions/allto_allv_quant_grouped_matmul/allto_allv_quant_grouped_matmul_urma_impl.h"

// 入口命名规则：AllToAllvQuantGmmKernel_<ADtype><BDtype>_<TransB>
#define DEFINE_A2AV_GMM_KERNEL(FuncName, AType, BType, TransB) \
    __global__ __aicore__ void FuncName(__gm__ CommContext *hcommCtx, GM_ADDR xGM, GM_ADDR xScaleGM, \
                                         GM_ADDR weightGM, GM_ADDR weightScaleGM, GM_ADDR yGM, GM_ADDR tilingGM) \
    { \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1); \
        Apace::AllToAllvQuantGroupedMatmulUrmaImpl<AType, BType, bfloat16_t, TransB> impl; \
        impl.Init(hcommCtx, xGM, xScaleGM, weightGM, weightScaleGM, yGM, tilingGM); \
        impl.Run(); \
    }

DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E4M3E4M3_NonTrans, fp8_e4m3fn_t, fp8_e4m3fn_t, false)
DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E4M3E4M3_Trans, fp8_e4m3fn_t, fp8_e4m3fn_t, true)
DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E5M2E5M2_NonTrans, fp8_e5m2_t, fp8_e5m2_t, false)
DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E5M2E5M2_Trans, fp8_e5m2_t, fp8_e5m2_t, true)
DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E4M3E5M2_NonTrans, fp8_e4m3fn_t, fp8_e5m2_t, false)
DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E4M3E5M2_Trans, fp8_e4m3fn_t, fp8_e5m2_t, true)
DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E5M2E4M3_NonTrans, fp8_e5m2_t, fp8_e4m3fn_t, false)
DEFINE_A2AV_GMM_KERNEL(AllToAllvQuantGmmKernel_E5M2E4M3_Trans, fp8_e5m2_t, fp8_e4m3fn_t, true)
