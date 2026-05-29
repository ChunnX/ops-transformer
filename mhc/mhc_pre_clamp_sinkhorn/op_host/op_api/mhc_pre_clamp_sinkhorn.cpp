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
 * \file mhc_pre_clamp_sinkhorn.cpp
 * \brief mhc_pre_clamp_sinkhorn
 */

#include "mhc_pre_clamp_sinkhorn.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(MhcPreClampSinkhorn);

static const aclTensor *MhcPreClampSinkhornAiCore(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                             const aclTensor *bias, float min, float max,
                                             int64_t hcMult, int64_t numIters, double hcEps,
                                             double normEps, bool needBackward,
                                             aclTensor *hin, aclTensor *hPost, aclTensor *hRes,
                                             aclTensor *hPre, aclTensor *hcBeforeNorm, aclTensor *invRms,
                                             aclTensor *sumOut, aclTensor *normOut, aclTensor *reduceOut,
                                             aclOpExecutor *executor)
{
    L0_DFX(MhcPreClampSinkhornAiCore, x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, needBackward,
           hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, reduceOut);
    
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcPreClampSinkhorn,
        OP_INPUT(x, phi, alpha, bias),
        OP_OUTPUT(hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, reduceOut),
        OP_ATTR(hcMult, numIters,  static_cast<float>(hcEps),  static_cast<float>(normEps), needBackward, min, max));
    
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MhcPreClampSinkhorn ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return hin;
}

const aclTensor *MhcPreClampSinkhorn(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                const aclTensor *bias, float min, float max,
                                int64_t hcMult, int64_t numIters, double hcEps,
                                double normEps, bool needBackward,
                                aclTensor *hin, aclTensor *hPost, aclTensor *hRes,
                                aclTensor *hPre, aclTensor *hcBeforeNorm, aclTensor *invRms,
                                aclTensor *sumOut, aclTensor *normOut, aclTensor *reduceOut,
                                aclOpExecutor *executor)
{
    return MhcPreClampSinkhornAiCore(x, phi, alpha, bias, min, max, hcMult, numIters, hcEps, normEps, needBackward,
                                hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, reduceOut, executor);
}

} // namespace l0op
