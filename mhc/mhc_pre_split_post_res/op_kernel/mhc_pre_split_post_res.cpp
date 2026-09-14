/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res.cpp
 * \brief mhc_pre_split_post_res kernel 入口
 *
 * TILING_KEY_SINGLE_TILE (0)：dimLoop==1（D<=2560，token 不足 T < coreNum/2 时双核/token，否则单核）
 *   → KernelMhcSingleTile<T>：无 dimLoop 循环
 *
 * TILING_KEY_MULTI_TILE  (1)：dimLoop>1（D>2560，token 不足 T < coreNum/2 时双核/token，否则单核）
 *   → KernelMhcMultiTile<T>：分 tile 循环
 */

#include "mhc_pre_split_post_res_multi_tile.h"

using namespace AscendC;
using namespace MhcPreSplitPostRes;

#define TILING_KEY_SINGLE_TILE 0
#define TILING_KEY_MULTI_TILE 1

extern "C" __global__ __aicore__ void mhc_pre_split_post_res(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha,
    GM_ADDR bias, GM_ADDR h_post, GM_ADDR h_res, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(MhcPreSplitPostResTilingData, tilingDataIn, tiling);
    const MhcPreSplitPostResTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_SINGLE_TILE)) {
        KERNEL_TASK_TYPE(TILING_KEY_SINGLE_TILE, KERNEL_TYPE_AIV_ONLY);
        KernelMhcSingleTile<DTYPE_X> op;
        op.Init(x, phi, alpha, bias, h_post, h_res, usrWorkspace, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_MULTI_TILE)) {
        KERNEL_TASK_TYPE(TILING_KEY_MULTI_TILE, KERNEL_TYPE_AIV_ONLY);
        KernelMhcMultiTile<DTYPE_X> op;
        op.Init(x, phi, alpha, bias, h_post, h_res, usrWorkspace, &pipe, tilingData);
        op.Process();
    }
}
