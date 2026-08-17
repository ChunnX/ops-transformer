/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_combine_gates.cpp
 * \brief MoeTutelCombineGates kernel 入口（仅 Ascend950 / arch35）
 */

#include "kernel_operator.h"
#include "arch35/moe_tutel_combine_gates_regbase.h"

using namespace AscendC;

// 必须是宏而非 constexpr 变量：opc 在预处理后的源码上扫 TILING_KEY_IS 的实参来切分
// binary 子核（kernelList 里的 <kernel>_<tilingKey>）。写成 constexpr 时实参仍是标识符，
// opc 识别不出 key 集合，只会产出单个 <kernel>_0，运行期按 tilingKey 找符号必然失败
// （RTS_BINARY_FAILURE: Expected symbol ..._10001 does not exist）。仓内同族算子同此写法。
#define TILING_KEY_NOT_CUT_H 10001
#define TILING_KEY_CUT_H 10002

extern "C" __global__ __aicore__ void moe_tutel_combine_gates(GM_ADDR x, GM_ADDR y_grad, GM_ADDR indices,
                                                              GM_ADDR locations, GM_ADDR gates_grad, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::MoeTutelCombineGatesTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_NOT_CUT_H)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::MoeTutelCombineGatesTilingData, tilingData, tiling);
        MoeTutelCombineGates::MoeTutelCombineGatesRegbase<DTYPE_X, false> op;
        op.Init(x, y_grad, indices, locations, gates_grad, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_CUT_H)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::MoeTutelCombineGatesTilingData, tilingData, tiling);
        MoeTutelCombineGates::MoeTutelCombineGatesRegbase<DTYPE_X, true> op;
        op.Init(x, y_grad, indices, locations, gates_grad, &tilingData, &pipe);
        op.Process();
    } else {
        return;
    }
}
