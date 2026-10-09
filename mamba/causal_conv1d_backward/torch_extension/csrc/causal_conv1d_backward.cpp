/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_backward.cpp
 * \brief Torch bridge — CausalConv1d（prefill, bias=None, cold start）的 VJP，
 *        delegates to aclnnCausalConv1dBackward.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor> CausalConv1dBackward(
    const at::Tensor &gradY, const at::Tensor &x, const at::Tensor &weight,
    const c10::optional<at::Tensor> &queryStartLoc, const std::string &activation)
{
    TORCH_CHECK(activation == "silu" || activation == "none", "activation must be 'silu' or 'none', got: ", activation);

    at::Tensor gradX{nullptr};
    at::Tensor gradWeight{nullptr};
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        // 用 empty(sizes, options) 而不是 empty_like：后者会 preserve_format，
        // 把 weight 的非连续 strides 一起继承到输出上。输出必须连续。
        gradX = at::empty(x.sizes(), x.options());
        gradWeight = at::empty(weight.sizes(), weight.options());
    }

    const char *activationMode = activation.c_str();
    ACLNN_CMD(aclnnCausalConv1dBackward, gradY, x, weight, queryStartLoc, activationMode, gradX, gradWeight);

    return {gradX, gradWeight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("causal_conv1d_backward", &CausalConv1dBackward, "causal_conv1d backward"); }

} // namespace op_api
