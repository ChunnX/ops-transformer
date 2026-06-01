/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_NORM_FLASH_ATTENTION_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_NORM_FLASH_ATTENTION_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
// TODO: 定义算子函数，返回输出tensor的tuple
const std::tuple<aclTensor *>
NormFlashAttention(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    aclOpExecutor *executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_NORM_FLASH_ATTENTION_H
