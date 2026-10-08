/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_grad_ext.cpp
 * \brief Torch bridge — causal_conv1d 反向算子，delegates to aclnnCausalConv1dGrad（自定义算子包）。
 *
 * 依赖运行期环境变量 ASCEND_CUSTOM_OPP_PATH 指向 custom_opp/vendors/causal_conv1d_grad_custom：
 * aclnn_common.h 的 GetOpApiFuncAddr 按该路径优先 dlopen libcust_opapi.so 并 dlsym
 * aclnnCausalConv1dGrad / aclnnCausalConv1dGradGetWorkspaceSize。
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor> CausalConv1dGrad(
    const at::Tensor &gradY, const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias)
{
    TORCH_CHECK(gradY.dim() == 3, "grad_y must be 3D [B, T, D], got dim=", gradY.dim());
    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, T, D], got dim=", x.dim());
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [kW, D], got dim=", weight.dim());
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [D], got dim=", bias.dim());
    TORCH_CHECK(gradY.sizes() == x.sizes(), "grad_y and x must have the same shape");
    TORCH_CHECK(weight.size(1) == x.size(2), "weight dim mismatch: weight[1]=", weight.size(1),
                ", x[2]=", x.size(2));
    TORCH_CHECK(bias.size(0) == x.size(2), "bias dim mismatch: bias[0]=", bias.size(0),
                ", x[2]=", x.size(2));
    TORCH_CHECK(gradY.scalar_type() == x.scalar_type() &&
                    gradY.scalar_type() == weight.scalar_type() &&
                    gradY.scalar_type() == bias.scalar_type(),
                "grad_y/x/weight/bias must have the same dtype, got ",
                toString(gradY.scalar_type()), "/", toString(x.scalar_type()), "/",
                toString(weight.scalar_type()), "/", toString(bias.scalar_type()));

    at::Tensor dx{nullptr}, dweight{nullptr}, dbias{nullptr};
    {
        auto localDevice = c10::Device(gradY.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        dx = at::empty_like(x);
        dweight = at::empty_like(weight);
        dbias = at::empty_like(bias);
    }

    ACLNN_CMD(aclnnCausalConv1dGrad, gradY, x, weight, bias, dx, dweight, dbias);

    return std::make_tuple(dx, dweight, dbias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("causal_conv1d_grad", &CausalConv1dGrad, "causal_conv1d backward (dx, dweight, dbias)");
}

} // namespace op_api
