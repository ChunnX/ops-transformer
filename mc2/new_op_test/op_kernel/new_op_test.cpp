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
 * \file new_op_test.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "new_op_test_tiling_data.h"
#include "new_op_test_tiling_key.h"
#include "new_op_test_mte.h"

using namespace AscendC;
using namespace NewOpTestImpl;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#endif

template<uint32_t newOpTestCommMode>
__global__ __aicore__ void new_op_test(GM_ADDR x, GM_ADDR scales, GM_ADDR output, GM_ADDR workspaceGM,
                                                           GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(NewOpTestTilingData);
    GET_TILING_DATA_WITH_STRUCT(NewOpTestTilingData, tilingData, tilingGM);
    TPipe pipe;
    if constexpr (newOpTestCommMode == MTE_COMM) {
        NewOpTestMte<DTYPE_X, DTYPE_SCALES, DTYPE_OUT_PUT> op;
        op.Init(x, scales, output, &pipe, &tilingData);
        op.Process();
    }
}
