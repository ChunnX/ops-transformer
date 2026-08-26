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
 * \file block_attention_residuals.cpp
 * \brief ACLNN Wrapper for aclnnBlockAttentionResiduals
 */

#include <tuple>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor> BlockAttentionResiduals(const at::Tensor &prefixSum,
                                                         const at::Tensor &blockRes,
                                                         const at::Tensor &projWeight,
                                                         const at::Tensor &normWeight, int64_t validBlockNum, double normEps,
                                                         bool needBackward)
{
    int64_t numTokens = prefixSum.size(0);
    int64_t hiddenSize = prefixSum.size(1);
    int64_t blockCount = blockRes.size(1) + 1;
    at::Tensor hiddenStates;
    at::Tensor invNorm;
    at::Tensor probs;
    {
        auto local_device = c10::Device(prefixSum.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        hiddenStates = at::empty({numTokens, hiddenSize}, prefixSum.options());
        if (needBackward) {
            invNorm = at::empty({numTokens, blockCount}, prefixSum.options().dtype(at::kFloat));
            probs = at::empty({numTokens, blockCount}, prefixSum.options().dtype(at::kFloat));
        } else {
            invNorm = at::empty({0}, prefixSum.options().dtype(at::kFloat));
            probs = at::empty({0}, prefixSum.options().dtype(at::kFloat));
        }
    }

    c10::optional<at::Tensor> invNormOpt = c10::nullopt;
    c10::optional<at::Tensor> probsOpt = c10::nullopt;
    if (needBackward) {
        invNormOpt = invNorm;
        probsOpt = probs;
    }

    ACLNN_CMD(aclnnBlockAttentionResiduals, prefixSum, blockRes, projWeight, normWeight, validBlockNum, normEps, needBackward,
              hiddenStates, invNormOpt, probsOpt);

    return std::make_tuple(hiddenStates, invNorm, probs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("block_attention_residuals", &BlockAttentionResiduals, "block_attention_residuals",
          py::arg("prefix_sum"), py::arg("block_res"), py::arg("proj_weight"), py::arg("norm_weight"),
          py::arg("norm_eps") = 1e-5, py::arg("need_backward") = false);
}

} // namespace op_api
