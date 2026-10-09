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
 * \file causal_conv1d_backward.h
 * \brief ACLNN L0 API 接口声明 - CausalConv1dBackward
 *
 * L0 API 职责：Kernel 调度（输出张量由调用方提供并写入）
 */

#ifndef OP_API_INC_LEVEL0_CAUSAL_CONV1D_BACKWARD_H_
#define OP_API_INC_LEVEL0_CAUSAL_CONV1D_BACKWARD_H_

#include "opdev/op_executor.h"

namespace l0op {

bool CausalConv1dBackward(const aclTensor *gradY, const aclTensor *x, const aclTensor *weight,
                          const aclTensor *queryStartLoc, const char *activation,
                          aclTensor *gradX, aclTensor *gradWeight, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_CAUSAL_CONV1D_BACKWARD_H_
