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
 * \file attention_res_grad.cpp
 * \brief ACLNN Wrapper for aclnnAttentionResGrad
 */

#include <tuple>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> AttentionResGrad(
    const at::Tensor &prefixSum, const at::Tensor &blockResidual,
    const at::Tensor &projWeight, const at::Tensor &normWeight,
    const at::Tensor &gradOutput, const at::Tensor &invRms, const at::Tensor &probs)
{
    int64_t numTokens = prefixSum.size(0);
    int64_t hiddenSize = prefixSum.size(1);
    int64_t blockCount = blockResidual.size(1) + 1;
    at::Tensor gradPrefixSum;
    at::Tensor gradBlockResidual;
    at::Tensor gradProjWeight;
    at::Tensor gradNormWeight;
    {
        auto local_device = c10::Device(prefixSum.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        gradPrefixSum = at::empty({numTokens, hiddenSize}, prefixSum.options());
        gradBlockResidual = at::empty({numTokens, blockCount - 1, hiddenSize}, prefixSum.options());
        gradProjWeight = at::empty({1, hiddenSize}, prefixSum.options());
        gradNormWeight = at::empty({hiddenSize}, prefixSum.options());
    }

    // 反向 kernel 将 inv_rms/probs 按主 dtype（FP16/BF16）解释，正向算子输出为 FP32，需先 cast 对齐。
    at::Tensor invRmsCast =
        invRms.scalar_type() == prefixSum.scalar_type() ? invRms : invRms.to(prefixSum.scalar_type());
    at::Tensor probsCast =
        probs.scalar_type() == prefixSum.scalar_type() ? probs : probs.to(prefixSum.scalar_type());

    ACLNN_CMD(aclnnAttentionResGrad, prefixSum, blockResidual, projWeight, normWeight, gradOutput, invRmsCast,
              probsCast, gradPrefixSum, gradBlockResidual, gradProjWeight, gradNormWeight);

    return std::make_tuple(gradPrefixSum, gradBlockResidual, gradProjWeight, gradNormWeight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention_res_grad", &AttentionResGrad, "attention_res_grad",
          py::arg("prefix_sum"), py::arg("block_residual"), py::arg("proj_weight"), py::arg("norm_weight"),
          py::arg("grad_output"), py::arg("inv_rms"), py::arg("probs"));
}

} // namespace op_api
