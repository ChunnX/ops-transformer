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
 * \file attn_res_fwd.cpp
 * \brief ACLNN Wrapper for aclnnAttnResFwd
 */

#include <tuple>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor> AttnResFwd(const at::Tensor &prefixSum,
                                                         const at::Tensor &blockResidual,
                                                         const at::Tensor &projWeight,
                                                         const at::Tensor &normWeight, double normEps,
                                                         bool needBackward)
{
    int64_t numTokens = prefixSum.size(0);
    int64_t hiddenSize = prefixSum.size(1);
    int64_t blockCount = blockResidual.size(1) + 1;
    at::Tensor hiddenStates;
    at::Tensor invRms;
    at::Tensor probs;
    {
        auto local_device = c10::Device(prefixSum.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        hiddenStates = at::empty({numTokens, hiddenSize}, prefixSum.options());
        if (needBackward) {
            invRms = at::empty({numTokens, blockCount}, prefixSum.options().dtype(at::kFloat));
            probs = at::empty({numTokens, blockCount}, prefixSum.options().dtype(at::kFloat));
        } else {
            invRms = at::empty({0}, prefixSum.options().dtype(at::kFloat));
            probs = at::empty({0}, prefixSum.options().dtype(at::kFloat));
        }
    }

    c10::optional<at::Tensor> invRmsOpt = c10::nullopt;
    c10::optional<at::Tensor> probsOpt = c10::nullopt;
    if (needBackward) {
        invRmsOpt = invRms;
        probsOpt = probs;
    }

    ACLNN_CMD(aclnnAttnResFwd, prefixSum, blockResidual, projWeight, normWeight, normEps, needBackward,
              hiddenStates, invRmsOpt, probsOpt);

    return std::make_tuple(hiddenStates, invRms, probs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attn_res_fwd", &AttnResFwd, "attn_res_fwd",
          py::arg("prefix_sum"), py::arg("block_residual"), py::arg("proj_weight"), py::arg("norm_weight"),
          py::arg("norm_eps") = 1e-5, py::arg("need_backward") = false);
}

} // namespace op_api
