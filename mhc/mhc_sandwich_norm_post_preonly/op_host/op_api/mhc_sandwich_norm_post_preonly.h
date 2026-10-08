/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_L0OP_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_L0OP_H_

#include "opdev/op_executor.h"

namespace MhcSandwichNormPostPreonlyl0op {

bool MhcSandwichNormPostPreonly(const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
                                const aclTensor *hResOptional, const aclTensor *phi, const aclTensor *alpha0,
                                const aclTensor *bias, const aclTensor *gamma0Optional, const aclTensor *gamma1,
                                const aclTensor *gamma2Optional, float normEps, float mhcPreNormEps, float hcEps,
                                bool returnHInF32, const aclTensor *hInPrimeOut, const aclTensor *x2Out,
                                const aclTensor *hInF32OutOptional, aclOpExecutor *executor);

} // namespace MhcSandwichNormPostPreonlyl0op

#endif // MHC_SANDWICH_NORM_POST_PREONLY_L0OP_H_
