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
 * \file mhc_pre_sinkhorn.h
 * \brief mhc_pre_sinkhorn
 */

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_

#include "opdev/op_executor.h"
namespace l0op {

const aclTensor *MhcPreSinkhorn(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                const aclTensor *bias, int64_t hcMult, int64_t numIters, float hcEps,
                                float normEps, bool needBackward, aclTensor *y, aclTensor *post,
                                aclTensor *combFrag, aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_
