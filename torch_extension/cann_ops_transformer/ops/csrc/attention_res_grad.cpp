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
 * \brief PyTorch bridge for aclnnAttentionResGrad
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AttentionResGrad(const at::Tensor &prefixSum, const at::Tensor &blockResidual,
                 const at::Tensor &projWeight, const at::Tensor &normWeight,
                 const at::Tensor &gradOutput, const at::Tensor &invRms,
                 const at::Tensor &probs)
{
    TORCH_CHECK(prefixSum.numel() > 0, "prefix_sum is empty.");
    TORCH_CHECK(blockResidual.numel() > 0, "block_residual is empty.");
    TORCH_CHECK(projWeight.numel() > 0, "proj_weight is empty.");
    TORCH_CHECK(normWeight.numel() > 0, "norm_weight is empty.");
    TORCH_CHECK(gradOutput.numel() > 0, "grad_output is empty.");
    TORCH_CHECK(invRms.numel() > 0, "inv_rms is empty.");
    TORCH_CHECK(probs.numel() > 0, "probs is empty.");

    at::Tensor gradPrefixSum = at::empty_like(prefixSum);
    at::Tensor gradBlockResidual = at::empty_like(blockResidual);
    at::Tensor gradProjWeight = at::empty_like(projWeight);
    at::Tensor gradNormWeight = at::empty_like(normWeight);

    ACLNN_CMD(aclnnAttentionResGrad,
              prefixSum, blockResidual, projWeight, normWeight,
              gradOutput, invRms, probs,
              gradPrefixSum, gradBlockResidual, gradProjWeight, gradNormWeight);

    return std::make_tuple(gradPrefixSum, gradBlockResidual, gradProjWeight, gradNormWeight);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention_res_grad", &op_api::AttentionResGrad, "attention_res_grad");
}
