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
 * \file grouped_matmul_finalize_routing.cpp
 * \brief
 */
#include "grouped_matmul_finalize_routing.h"
#include "grouped_matmul_finalize_routing_antiquant_a8w4_msd_pre.h"
#include "grouped_matmul_finalize_routing_antiquant_a8w4_msd.h"

namespace GroupedMatmulFinalizeRouting {
    constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();

#if defined(FORMAT_W) && defined(FORMAT_FRACTAL_NZ) && (FORMAT_W == FORMAT_FRACTAL_NZ)
    constexpr CubeFormat wFormat = CubeFormat::NZ;
#else
    constexpr CubeFormat wFormat = CubeFormat::ND;
#endif
}

using namespace AscendC;
using namespace matmul;
using namespace GroupedMatmulFinalizeRouting;

extern "C" __global__ __aicore__ void grouped_matmul_finalize_routing(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
                                                                      GM_ADDR pertoken_scale, GM_ADDR group_list,
                                                                      GM_ADDR share_input, GM_ADDR logit,
                                                                      GM_ADDR row_index, GM_ADDR offset,
                                                                      GM_ADDR y, GM_ADDR workspaceGM,
                                                                      GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);
    MMInitParams initParams{x, w, bias, group_list, scale, pertoken_scale,
                            offset, logit, row_index, share_input, y, workspaceGM};
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(10000000000000000001UL)) {
        TPipe pipe;
        MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.matmulTiling, &pipe);
        }
        using param = Param<true, int64_t, GroupMatmulFRTilingData, float>;
        QuantGroupMatmul<param> op(mm);
        op.Init(initParams, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(10000000000000000011UL)) {
        TPipe pipe;
        MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.matmulTiling, &pipe);
        }
        using param = Param<true, int32_t, GroupMatmulFRTilingData, float>;
        QuantGroupMatmul<param> op(mm);
        op.Init(initParams, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(10000000000000000101UL)) {
        TPipe pipe;
        MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.matmulTiling, &pipe);
        }
        using param = Param<true, int64_t, GroupMatmulFRTilingData, bfloat16_t>;
        QuantGroupMatmul<param> op(mm);
        op.Init(initParams, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(10000000000000000111UL)) {
        TPipe pipe;
        MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.matmulTiling, &pipe);
        }
        using param = Param<true, int32_t, GroupMatmulFRTilingData, bfloat16_t>;
        QuantGroupMatmul<param> op(mm);
        op.Init(initParams, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(11000000000000000011UL)) {
        TPipe pipe;
        if ASCEND_IS_AIV {
            GMMA8W4PreProcess op1;
            MMPreInitParams preInitParams {x, x, group_list, workspaceGM};
            op1.Init(preInitParams, tilingData, &pipe);
            op1.Process();
            pipe.Reset();
            pipe.Destroy();
            pipe.Init();
        }
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
        using bT = MatmulType<TPosition::GM, wFormat, int4b_t, false>;
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using matmulType = MMImplType<aT, bT, cT, biasT, GroupedMatmulFinalizeRouting::A8W4_GMM_CFG_MDL>;
        matmulType::MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.matmulTiling, &pipe);
        }

        GMMA8W4MSDCompute<matmulType> op(mm);
        op.Init(initParams, &tilingData, &pipe);
        op.Process();
    }
}
