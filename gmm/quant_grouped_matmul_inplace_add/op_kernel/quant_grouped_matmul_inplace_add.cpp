/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_inplace_add.cpp
 * \brief
 */
#include "qgmm_inplace_add_utils.h"
#include "quant_grouped_matmul_inplace_add_tiling_data.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#if defined(V310_QGMM_QUANT_MX)
#include "arch35/qgmm_inplace_add_cube_on_the_fly.h"
#endif
#if defined(V310_QGMM_QUANT_MIX)
#include "arch35/qgmm_inplace_add_mix_online_dynamic.h"
#endif
#endif

using namespace AscendC;
using namespace matmul;

extern "C" __global__ __aicore__ void quant_grouped_matmul_inplace_add(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2,
                                                                       GM_ADDR groupList, GM_ADDR yIn, GM_ADDR scale1,
                                                                       GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams);
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef __CCE_KT_TEST__
#if defined(V310_QGMM_QUANT_MX)       // mxfpx
    if (TILING_KEY_IS(20000000010)) { // transX = true, transW = false
        KERNEL_TASK_TYPE(20000000010, KERNEL_TYPE_AIC_ONLY);
        QGmmInplaceAddAswt<Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::RowMajor>(x1, x2, scale2, groupList,
                                                                                        scale1, y, tiling);
    }
#endif
#if defined(V310_QGMM_QUANT_MIX)
    if (TILING_KEY_IS(20000000110)) { // transX = true, transW = false
        KERNEL_TASK_TYPE(20000000110, KERNEL_TYPE_MIX_AIC_1_2);
        QGmmInplaceAddMixAswt<Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::RowMajor>(x1, x2, scale2, groupList,
                                                                                           scale1, y, tiling);
    }
#endif
#endif
#endif
}