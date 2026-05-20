/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OP_API_INC_LEVEL0_CAUSAL_CONV1D_L0OP_H_
#define OP_API_INC_LEVEL0_CAUSAL_CONV1D_L0OP_H_

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {
bool CausalConv1d(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, aclTensor *convStates,
                  const aclTensor *queryStartLoc, const aclTensor *cacheIndices, const aclTensor *initialStateMode,
                  const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId, int64_t runMode,
                  aclTensor *y, aclOpExecutor *executor);
} // namespace l0op
#endif // OP_API_INC_LEVEL0_CAUSAL_CONV1D_L0OP_H_
