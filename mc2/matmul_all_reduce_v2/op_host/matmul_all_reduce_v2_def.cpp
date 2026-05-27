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
 * \file matmul_all_reduce_v2_def.cpp
 * \brief MatmulAllReduceV2 operator definition - reuse MatmulAllReduce implementation
 */

// 引用原算子的定义头文件
#include "../matmul_all_reduce/op_host/matmul_all_reduce_def.cpp"

namespace ops {
// MatmulAllReduceV2 的定义与原算子完全一致，直接复用
// 注意：REGIST_OP 在原文件中已经注册为 MatmulAllReduce
// 这里不需要再次注册，通过 include 原文件即可复用定义逻辑
}