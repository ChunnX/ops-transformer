/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res_apt.cpp
 * \brief mhc_pre_split_post_res ascend950（arch35）kernel 入口
 *
 * schMode 由 Host 侧 GET_TPL_TILING_KEY 编码进 TilingKey，框架按 SEL 表实例化模板入口，
 * 编译期 if constexpr 派发（不使用 TILING_KEY_IS 运行时分支）：
 *
 * MHC_PRS_SCH_MODE_SINGLE_TILE (0)：dimLoop==1（D<=2560，token 不足 T < coreNum/2 时双核/token，否则单核）
 *   → KernelMhcSingleTile<T>：无 dimLoop 循环
 *
 * MHC_PRS_SCH_MODE_MULTI_TILE  (1)：dimLoop>1（D>2560，token 不足 T < coreNum/2 时双核/token，否则单核）
 *   → KernelMhcMultiTile<T>：分 tile 循环
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"

// 算法头（TilingData 定义由编译链从 op_host 宏版自动注入，同 mhc_pre 单份模式）
#include "mhc_pre_split_post_res_multi_tile.h"
// schMode 模板参数与 TilingKey 的映射声明（APT 链路，Host 侧同用）
#include "arch35/mhc_pre_split_post_res_tiling_key.h"

using namespace AscendC;
using namespace MhcPreSplitPostRes;

template <uint32_t schMode>
__global__ __aicore__ void mhc_pre_split_post_res(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha,
    GM_ADDR bias, GM_ADDR h_post, GM_ADDR h_res, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }
    if constexpr (schMode == MHC_PRS_SCH_MODE_SINGLE_TILE) {
        KernelMhcSingleTile<DTYPE_X> op;
        op.Init(x, phi, alpha, bias, h_post, h_res, usrWorkspace, &pipe, &tilingData);
        op.Process();
    } else if constexpr (schMode == MHC_PRS_SCH_MODE_MULTI_TILE) {
        KernelMhcMultiTile<DTYPE_X> op;
        op.Init(x, phi, alpha, bias, h_post, h_res, usrWorkspace, &pipe, &tilingData);
        op.Process();
    }
}
