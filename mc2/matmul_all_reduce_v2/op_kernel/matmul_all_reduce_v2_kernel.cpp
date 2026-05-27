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
 * \file matmul_all_reduce_v2_kernel.cpp
 * \brief MatmulAllReduceV2 kernel entry - reuse MatmulAllReduce kernel implementation
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// 引用原算子的 kernel 实现文件
#include "../matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_apt.cpp"

using namespace AscendC;

// MatmulAllReduceV2 的 Kernel 实现与 MatmulAllReduce 完全一致
// 直接复用原算子的 kernel 函数
// 注意：原算子中已经定义了 fp_matmul_all_reduce 和 quant_matmul_all_reduce 模板函数
// MatmulAllReduceV2 使用相同的 kernel 实现