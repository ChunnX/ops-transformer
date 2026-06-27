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
 * \file moe_ep_combine.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, c10::optional<at::Tensor>>;
const int DIM_TWO = 2;

tensor_list npu_moe_ep_combine(const at::Tensor &context, const at::Tensor &x,
    const at::Tensor &topk_idx, const at::Tensor &recv_src_metadata,
    const at::Tensor &num_recv_tokens_per_expert, const c10::optional<at::Tensor> &topk_weights,
    const c10::optional<at::Tensor> &bias_0, const c10::optional<at::Tensor> &bias_1,
    int64_t ep_world_size, int64_t ep_rank_id, int64_t num_experts,
    int64_t num_max_tokens_per_rank, int64_t ccl_buffer_size)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D");
    TORCH_CHECK(!bias_0.has_value(), "bias not supported in first release");
    TORCH_CHECK(!bias_1.has_value(), "bias not supported in first release");

    int64_t num_tokens = topk_idx.size(0);
    int64_t hidden = x.size(1);
    int64_t top_k = topk_idx.size(1);

    at::Tensor combined_x;
    at::Tensor combined_topk_weights;

    combined_x = at::empty({num_tokens, hidden}, x.options());
    if (topk_weights.has_value()) {
        combined_topk_weights = at::empty({num_tokens, top_k}, x.options().dtype(at::kFloat));
    } else {
        combined_topk_weights = at::Tensor();
    }

    c10::optional<at::Tensor> topk_weights_opt = topk_weights;
    c10::optional<at::Tensor> bias_0_opt = c10::optional<at::Tensor>();
    c10::optional<at::Tensor> bias_1_opt = c10::optional<at::Tensor>();

    ACLNN_CMD(aclnnMoeEpCombine, context, x, topk_idx, recv_src_metadata,
        num_recv_tokens_per_expert, topk_weights_opt, bias_0_opt, bias_1_opt,
        ep_world_size, ep_rank_id, num_experts, num_max_tokens_per_rank, ccl_buffer_size,
        combined_x, combined_topk_weights);

    c10::optional<at::Tensor> combined_topk_weights_opt;
    if (topk_weights.has_value()) {
        combined_topk_weights_opt = combined_topk_weights;
    }
    return std::make_tuple(combined_x, combined_topk_weights_opt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_ep_combine", &npu_moe_ep_combine, "moe_ep_combine");
}

}  // namespace op_api