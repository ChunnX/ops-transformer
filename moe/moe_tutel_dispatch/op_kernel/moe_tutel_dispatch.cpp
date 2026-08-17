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
 * \file moe_tutel_dispatch.cpp
 * \brief MoeTutelDispatch kernel entry, Ascend950 (arch35) only
 */

#include "arch35/moe_tutel_dispatch_full_row.h"
#include "arch35/moe_tutel_dispatch_split_h.h"

using namespace MoeTutelDispatchNs;

// 必须是宏或字面量：TILING_KEY_IS 的实参要在预编译期被解析成数值常量，
// 写成 constexpr 变量会报 "can not be processed as numeric variables in the
// precompilation phase"，退化成单个不按 key 特化的 kernel。
// 取值须与 op_host/moe_tutel_dispatch_tiling.h 的 MOE_TUTEL_DISPATCH_TILING_KEY_* 一致。
#define TILING_KEY_FULL_ROW 0
#define TILING_KEY_SPLIT_H 1
#define TILING_KEY_EMPTY 2

/*
 * x / gates / y 按同宽无符号整型搬运：本算子是纯 bit 拷贝，不做任何算术，
 * 因此 float16 / bfloat16 走 uint16_t、float32 走 uint32_t，逐位等价且规避 bf16 cast。
 */
extern "C" __global__ __aicore__ void moe_tutel_dispatch(GM_ADDR x, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations,
                                                         GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILING_KEY_EMPTY)) {
        return;
    }

    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(TILING_KEY_FULL_ROW)) {
        if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
            MoeTutelDispatchFullRow<uint32_t> op(&pipe, &tilingData);
            op.Init(x, gates, indices, locations, y);
            op.Process();
        } else {
            MoeTutelDispatchFullRow<uint16_t> op(&pipe, &tilingData);
            op.Init(x, gates, indices, locations, y);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_SPLIT_H)) {
        if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
            MoeTutelDispatchSplitH<uint32_t> op(&pipe, &tilingData);
            op.Init(x, gates, indices, locations, y);
            op.Process();
        } else {
            MoeTutelDispatchSplitH<uint16_t> op(&pipe, &tilingData);
            op.Init(x, gates, indices, locations, y);
            op.Process();
        }
    }
}
