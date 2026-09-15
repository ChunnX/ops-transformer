/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res.cpp
 * \brief MhcPreSplitPostRes L0 op_api
 */

#include "mhc_pre_split_post_res.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPreSplitPostRes);

// h_post_out / h_res_out 为 def 显式声明的输出（用户分配），无需 AllocTensor/INFER_SHAPE，
// 直接加入 launch 列表，由 op_host 注册的 tiling/kernel 流程执行。
bool MhcPreSplitPostRes(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                        const aclTensor *bias, double normEps,
                        const aclTensor *hPostOut, const aclTensor *hResOut, aclOpExecutor *executor)
{
    L0_DFX(MhcPreSplitPostRes, x, phi, alpha, bias, normEps, hPostOut, hResOut);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcPreSplitPostRes,
        OP_INPUT(x, phi, alpha, bias),
        OP_OUTPUT(hPostOut, hResOut),
        OP_ATTR(static_cast<float>(normEps)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MhcPreSplitPostRes ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }

    return true;
}
} // namespace l0op
