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
#include "../mhc_pre_split_post_res_const.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPreSplitPostRes);

// h_post 为 def 显式声明的输出（用户分配）直接加入 launch 列表；h_res 为可选输出：
// 完整形态（phi 行数 N^2+N）时同样用户分配直接加入；仅 post 形态（phi 行数 N，
// hResOut 为空）时分配 1 元素 dummy 占位保持 launch 输出形态，kernel 不写该段。
bool MhcPreSplitPostRes(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                        const aclTensor *bias, float normEps,
                        const aclTensor *hPostOut, const aclTensor *hResOut, aclOpExecutor *executor)
{
    L0_DFX(MhcPreSplitPostRes, x, phi, alpha, bias, normEps, hPostOut, hResOut);

    if (executor == nullptr || hPostOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "MhcPreSplitPostRes executor and h_post output must not be nullptr.");
        return false;
    }

    // 完整形态（phi 行数 N^2+N）必须提供 h_res：dummy 无 shape，kernel 会写 res 段导致越界。
    // （aclnn 层已拦此组合，此处为绕过 aclnn 直调 L0 的防御双保险。）
    const size_t xRank = x->GetViewShape().GetDimNum();
    const int64_t nValue = (xRank == optiling::MHC_DIM_NUM_4) ? x->GetViewShape().GetDim(optiling::MHC_DIM_TWO)
                                                                : x->GetViewShape().GetDim(optiling::MHC_DIM_ONE);
    // n 白名单校验（aclnn 层同款）：同时消除 nValue*nValue+nValue 的 int64 溢出面
    if (nValue != optiling::MHC_N_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MhcPreSplitPostRes n must be 4 (%ld).", nValue);
        return false;
    }
    if (hResOut == nullptr && phi->GetViewShape().GetDim(optiling::MHC_DIM_ZERO) == nValue * nValue + nValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MhcPreSplitPostRes h_res output is required when the number of phi rows is N^2+N (%ld).",
                nValue * nValue + nValue);
        return false;
    }

    const aclTensor *hRes = hResOut;
    if (hResOut == nullptr) {
        hRes = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
        if (hRes == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MhcPreSplitPostRes alloc dummy h_res failed.");
            return false;
        }
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcPreSplitPostRes,
        OP_INPUT(x, phi, alpha, bias),
        OP_OUTPUT(hPostOut, hRes),
        OP_ATTR(normEps));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MhcPreSplitPostRes ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }

    return true;
}
} // namespace l0op
