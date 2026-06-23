/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_re_routing.cpp
 * \brief MoE re routing operator, bridging PyTorch to aclnnMoeReRoutingV2.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int DIM_TWO = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_re_routing(
    const at::Tensor &tokens, const at::Tensor &expert_token_num_per_rank,
    const c10::optional<at::Tensor> &per_token_scales,
    const c10::optional<at::Tensor> &expert_topk_weight,
    int64_t expert_token_num_type, int64_t idx_type)
{
    TORCH_CHECK(tokens.dim() == DIM_TWO, "tokens must be 2D, but got ", tokens.dim(), "D.");
    TORCH_CHECK(expert_token_num_per_rank.dim() == DIM_TWO,
        "expert_token_num_per_rank must be 2D, but got ", expert_token_num_per_rank.dim(), "D.");
    TORCH_CHECK((expert_token_num_type == 0 || expert_token_num_type == 1),
        "expert_token_num_type must be 0 or 1, current is: ", expert_token_num_type);
    TORCH_CHECK((idx_type == 0 || idx_type == 1),
        "idx_type must be 0 or 1, current is: ", idx_type);
    if (expert_topk_weight.has_value() && expert_topk_weight.value().defined()) {
        TORCH_CHECK(expert_topk_weight.value().scalar_type() == at::kFloat,
            "expert_topk_weight dtype must be float32, got ", expert_topk_weight.value().scalar_type());
    }

    at::Tensor permute_tokens;
    at::Tensor permute_per_token_scales;
    at::Tensor permute_token_idx;
    at::Tensor expert_token_num;
    at::Tensor permute_topk_weight;

    int64_t A = tokens.size(0);
    int64_t H = tokens.size(1);
    int64_t E = expert_token_num_per_rank.size(1);

    bool has_topk_weight = expert_topk_weight.has_value() && expert_topk_weight.value().defined();

    {
        auto localDevice = c10::Device(tokens.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);

        permute_tokens = at::empty({A, H}, tokens.options());

        if (per_token_scales.has_value() && per_token_scales.value().defined()) {
            permute_per_token_scales = at::empty(per_token_scales.value().sizes(), per_token_scales.value().options());
        } else {
            permute_per_token_scales = at::empty({A}, tokens.options().dtype(at::kFloat));
        }

        permute_token_idx = at::empty({A}, tokens.options().dtype(at::kInt));
        expert_token_num = at::empty({E}, expert_token_num_per_rank.options());

        if (has_topk_weight) {
            permute_topk_weight = at::empty(expert_topk_weight.value().sizes(), expert_topk_weight.value().options());
        } else {
            permute_topk_weight = at::empty({0}, tokens.options().dtype(at::kFloat));
        }
    }

    c10::optional<at::Tensor> permute_topk_weight_opt =
        has_topk_weight ? c10::optional<at::Tensor>(permute_topk_weight) : c10::optional<at::Tensor>();

    ACLNN_CMD(aclnnMoeReRoutingV2, tokens, expert_token_num_per_rank, per_token_scales, expert_topk_weight,
              expert_token_num_type, idx_type,
              permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight_opt);

    return std::tie(permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_re_routing", &moe_re_routing, "moe_re_routing");
}

} // namespace op_api
