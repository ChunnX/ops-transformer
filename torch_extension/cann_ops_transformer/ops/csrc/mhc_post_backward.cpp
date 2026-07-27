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
 * \file mhc_post_backward.cpp
 * \brief ACLNN Wrapper for aclnnMhcPostBackward
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

namespace {

constexpr int64_t kTndRank = 3;
constexpr int64_t kBsndRank = 4;

void CheckTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on an NPU device, but got ",
                tensor.device());
    TORCH_CHECK(tensor.numel() > 0, name, " must not be empty");
}

void CheckDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same NPU device as x, but got ",
                tensor.device(), " and ", reference.device());
}

void CheckDtype(const at::Tensor &gradOutput, const at::Tensor &x, const at::Tensor &hRes, const at::Tensor &hOut,
                const at::Tensor &hPost)
{
    TORCH_CHECK(gradOutput.scalar_type() == at::kHalf || gradOutput.scalar_type() == at::kBFloat16,
                "grad_output must be float16 or bfloat16, but got ", c10::toString(gradOutput.scalar_type()));
    TORCH_CHECK(x.scalar_type() == gradOutput.scalar_type(), "x dtype must match grad_output dtype, but got ",
                c10::toString(x.scalar_type()), " and ", c10::toString(gradOutput.scalar_type()));
    TORCH_CHECK(hOut.scalar_type() == gradOutput.scalar_type(), "h_out dtype must match grad_output dtype, but got ",
                c10::toString(hOut.scalar_type()), " and ", c10::toString(gradOutput.scalar_type()));
    TORCH_CHECK(hRes.scalar_type() == at::kFloat, "h_res must be float32, but got ", c10::toString(hRes.scalar_type()));
    TORCH_CHECK(hPost.scalar_type() == at::kFloat, "h_post must be float32, but got ",
                c10::toString(hPost.scalar_type()));
}

void CheckShape(const at::Tensor &gradOutput, const at::Tensor &x, const at::Tensor &hRes, const at::Tensor &hOut,
                const at::Tensor &hPost)
{
    TORCH_CHECK(x.dim() == kTndRank || x.dim() == kBsndRank, "x must be 3D or 4D, but got ", x.dim(), "D");
    TORCH_CHECK(gradOutput.sizes() == x.sizes(), "grad_output shape must match x shape, but got ", gradOutput.sizes(),
                " and ", x.sizes());

    if (x.dim() == kTndRank) {
        TORCH_CHECK(hRes.dim() == kTndRank && hRes.size(0) == x.size(0) && hRes.size(1) == x.size(1) &&
                        hRes.size(2) == x.size(1),
                    "h_res must have shape [T, N, N] for 3D x, but got h_res ", hRes.sizes(), " and x ", x.sizes());
        TORCH_CHECK(hOut.dim() == 2 && hOut.size(0) == x.size(0) && hOut.size(1) == x.size(2),
                    "h_out must have shape [T, D] for 3D x, but got h_out ", hOut.sizes(), " and x ", x.sizes());
        TORCH_CHECK(hPost.dim() == 2 && hPost.size(0) == x.size(0) && hPost.size(1) == x.size(1),
                    "h_post must have shape [T, N] for 3D x, but got h_post ", hPost.sizes(), " and x ", x.sizes());
        return;
    }

    TORCH_CHECK(hRes.dim() == 4 && hRes.size(0) == x.size(0) && hRes.size(1) == x.size(1) &&
                    hRes.size(2) == x.size(2) && hRes.size(3) == x.size(2),
                "h_res must have shape [B, S, N, N] for 4D x, but got h_res ", hRes.sizes(), " and x ", x.sizes());
    TORCH_CHECK(hOut.dim() == 3 && hOut.size(0) == x.size(0) && hOut.size(1) == x.size(1) && hOut.size(2) == x.size(3),
                "h_out must have shape [B, S, D] for 4D x, but got h_out ", hOut.sizes(), " and x ", x.sizes());
    TORCH_CHECK(hPost.dim() == 3 && hPost.size(0) == x.size(0) && hPost.size(1) == x.size(1) &&
                    hPost.size(2) == x.size(2),
                "h_post must have shape [B, S, N] for 4D x, but got h_post ", hPost.sizes(), " and x ", x.sizes());
}

void CheckInputs(const at::Tensor &gradOutput, const at::Tensor &x, const at::Tensor &hRes, const at::Tensor &hOut,
                 const at::Tensor &hPost)
{
    CheckTensor(gradOutput, "grad_output");
    CheckTensor(x, "x");
    CheckTensor(hRes, "h_res");
    CheckTensor(hOut, "h_out");
    CheckTensor(hPost, "h_post");
    CheckDevice(gradOutput, x, "grad_output");
    CheckDevice(hRes, x, "h_res");
    CheckDevice(hOut, x, "h_out");
    CheckDevice(hPost, x, "h_post");
    CheckDtype(gradOutput, x, hRes, hOut, hPost);
    CheckShape(gradOutput, x, hRes, hOut, hPost);
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcPostBackward(const at::Tensor &gradOutput,
                                                                           const at::Tensor &x, const at::Tensor &hRes,
                                                                           const at::Tensor &hOut,
                                                                           const at::Tensor &hPost)
{
    CheckInputs(gradOutput, x, hRes, hOut, hPost);

    at::Tensor gradX{nullptr};
    at::Tensor gradHres{nullptr};
    at::Tensor gradHout{nullptr};
    at::Tensor gradHpost{nullptr};
    {
        // Output allocation must happen under the input device guard for non-default NPU cards.
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        gradX = at::empty_like(x);
        gradHres = at::empty_like(hRes);
        gradHout = at::empty_like(hOut);
        gradHpost = at::empty_like(hPost);
    }

    ACLNN_CMD(aclnnMhcPostBackward, gradOutput, x, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(gradX, gradHres, gradHout, gradHpost);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_post_backward", &MhcPostBackward, "mhc_post_backward");
}

} // namespace op_api
