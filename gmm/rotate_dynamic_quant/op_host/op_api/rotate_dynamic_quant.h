/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_ROTATE_DYNAMIC_QUANT_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_ROTATE_DYNAMIC_QUANT_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {

/**
 * @brief RotateDynamicQuant L0 算子函数
 * @param x 输入 tensor [M, N]
 * @param rot 旋转矩阵 tensor [N, N]
 * @param dst_type 输出数据类型属性，0=INT8, 1=INT32
 * @param executor 执行器
 * @return 返回 (yOut, scaleOut) 元组
 */
const std::tuple<aclTensor *, aclTensor *>
RotateDynamicQuant(
    const aclTensor *x,
    const aclTensor *rot,
    int32_t dst_type,
    aclOpExecutor *executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_ROTATE_DYNAMIC_QUANT_H
