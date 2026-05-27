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
 * \file matmul_all_reduce_v2_tiling_key.h
 * \brief MatmulAllReduceV2 TilingKey definitions
 */

#ifndef MATMUL_ALL_REDUCE_V2_TILING_KEY_H_
#define MATMUL_ALL_REDUCE_V2_TILING_KEY_H_

// 引用原算子的 TilingKey 定义
#include "../matmul_all_reduce/op_kernel/matmul_all_reduce_apt_tiling_key.h"

// MatmulAllReduceV2 的 TilingKey 与 MatmulAllReduce 完全一致
// 直接复用原算子的定义

namespace matmul_all_reduce_v2 {
// 使用原算子的 TilingKey 定义
using namespace matmul_all_reduce_apt;
} // namespace matmul_all_reduce_v2

#endif // MATMUL_ALL_REDUCE_V2_TILING_KEY_H_