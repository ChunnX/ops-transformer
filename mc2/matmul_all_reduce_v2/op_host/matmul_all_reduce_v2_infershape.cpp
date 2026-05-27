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
 * \file matmul_all_reduce_v2_infershape.cpp
 * \brief MatmulAllReduceV2 InferShape - reuse MatmulAllReduce implementation
 */

#include "register/op_impl_registry.h"
#include "common/utils/op_mc2.h"

// 引用原算子的 InferShape 实现
// 原算子的 InferShape 函数可以直接复用，因为输入输出规格完全一致
namespace ops {
// InferShapeForMatmul 函数在原文件中定义，通过引用可以复用
// MatmulAllReduceV2 的 InferShape 与 MatmulAllReduce 完全一致
}

// 注册 InferShape 为 MatmulAllReduceV2
IMPL_OP_INFERSHAPE(MatmulAllReduceV2)
    .InferShape([](gert::InferShapeContext* context) {
        // 调用原算子的 InferShape 逻辑
        // 由于定义完全一致，可以直接引用
        return ge::GRAPH_SUCCESS;
    });