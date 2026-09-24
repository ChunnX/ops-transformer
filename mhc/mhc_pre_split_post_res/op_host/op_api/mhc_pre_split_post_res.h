/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MHC_PRE_SPLIT_POST_RES_L0OP_H_
#define MHC_PRE_SPLIT_POST_RES_L0OP_H_

#include "opdev/op_executor.h"

namespace l0op {

bool MhcPreSplitPostRes(
    const aclTensor *x,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    float normEps,
    const aclTensor *hPostOut,
    const aclTensor *hResOut,
    aclOpExecutor *executor);

}  // namespace l0op

#endif  // MHC_PRE_SPLIT_POST_RES_L0OP_H_
