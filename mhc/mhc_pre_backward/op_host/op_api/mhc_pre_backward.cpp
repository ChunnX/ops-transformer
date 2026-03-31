/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mhc_pre_backward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPreBackward);

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *>  MhcPreBackward(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
    const aclTensor *h_in_grad, const aclTensor *h_post_grad, const aclTensor *h_res_grad,
    const aclTensor *inv_rms, const aclTensor *mm_res, const aclTensor *h_pre, const aclTensor *h_post,
    const aclTensor *gamma, float hc_eps,
    aclOpExecutor *executor)
{
    L0_DFX(MhcPreBackward, x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms,
           mm_res, h_pre, h_post, gamma, hc_eps);

    DataType outType = DataType::DT_FLOAT; // 输出类型
    Format format = Format::FORMAT_ND; // 输出分形
    auto outXGrad = executor->AllocTensor(h_in_grad->GetDataType(), format, format);
    auto outHcWeightGrad = executor->AllocTensor(outType, format, format);
    auto outAlphaGrad = executor->AllocTensor(outType, format, format);
    auto outBiasPostGrad = executor->AllocTensor(outType, format, format);
    auto outGammaGrad = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(MhcPreBackward, OP_INPUT(x, phi, alpha, h_in_grad,
        h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma),
        OP_OUTPUT(outXGrad, outHcWeightGrad, outAlphaGrad, outBiasPostGrad, outGammaGrad), OP_ATTR(hc_eps));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr),
        "MhcPreBackward InferShape failed.");
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(MhcPreBackward,
        OP_INPUT(x, phi, alpha, h_in_grad,
        h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma),
        OP_OUTPUT(outXGrad, outHcWeightGrad, outAlphaGrad, outBiasPostGrad, outGammaGrad), OP_ATTR(hc_eps));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr),
        "MhcPreBackward ADD_TO_LAUNCHER_LIST_AICORE failed.");
        
    return std::tuple(outXGrad, outHcWeightGrad, outAlphaGrad, outBiasPostGrad, outGammaGrad);
}
}
