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
 * \file matmul_all_reduce_v2_gen_task.cpp
 * \brief MatmulAllReduceV2 task generation - reuse MatmulAllReduce implementation
 */

// 引用原算子的 task generation 实现
// 由于任务生成逻辑完全一致，可以直接复用
#include "../matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp"

// MatmulAllReduceV2 的任务生成逻辑与 MatmulAllReduce 完全一致
// 直接复用原算子的实现