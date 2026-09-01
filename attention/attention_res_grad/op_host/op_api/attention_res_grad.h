/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_ATTENTION_RES_GRAD_H_
#define OP_API_INC_LEVEL0_ATTENTION_RES_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *AttentionResGrad(
    const aclTensor *prefix_sum,
    const aclTensor *block_residual,
    const aclTensor *proj_weight,
    const aclTensor *norm_weight,
    const aclTensor *grad_output,
    const aclTensor *inv_rms,
    const aclTensor *probs,
    aclTensor *grad_prefix_sum,
    aclTensor *grad_block_residual,
    aclTensor *grad_proj_weight,
    aclTensor *grad_norm_weight,
    aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_ATTENTION_RES_GRAD_H_
