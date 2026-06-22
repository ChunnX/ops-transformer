/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing.cpp
 * \brief MoE init routing operator, bridging PyTorch to aclnnMoeInitRoutingV4.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int DIM_TWO = 2;
constexpr int64_t COLS_ALIGN_FACTOR = 256;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_init_routing(
    const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset,
    const c10::optional<at::Tensor> &topk_weight,
    int64_t active_num, int64_t expert_capacity, int64_t expert_num,
    int64_t drop_pad_mode, int64_t expert_tokens_num_type, bool expert_tokens_num_flag,
    int64_t quant_mode,
    const c10::optional<std::vector<int64_t>> &active_expert_range,
    int64_t row_idx_type)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D, but got ", x.dim(), "D.");
    TORCH_CHECK(expert_idx.dim() == DIM_TWO, "expert_idx must be 2D, but got ", expert_idx.dim(), "D.");
    TORCH_CHECK(expert_num > 0, "expert_num must be greater than 0, current is: ", expert_num);
    TORCH_CHECK((drop_pad_mode == 0 || drop_pad_mode == 1),
        "drop_pad_mode must be 0 or 1, current is: ", drop_pad_mode);
    TORCH_CHECK((expert_tokens_num_type == 0 || expert_tokens_num_type == 1),
        "expert_tokens_num_type must be 0 or 1, current is: ", expert_tokens_num_type);
    TORCH_CHECK((row_idx_type == 0 || row_idx_type == 1),
        "row_idx_type must be 0 or 1, current is: ", row_idx_type);
    if (drop_pad_mode == 1) {
        TORCH_CHECK(expert_capacity > 0,
            "expert_capacity must be greater than 0 when drop_pad_mode=1, current is: ", expert_capacity);
    }
    if (topk_weight.has_value() && topk_weight.value().defined()) {
        TORCH_CHECK(topk_weight.value().dim() == DIM_TWO,
            "topk_weight must be 2D, but got ", topk_weight.value().dim(), "D.");
        TORCH_CHECK(topk_weight.value().size(0) == x.size(0),
            "topk_weight row count must equal x row count, got ", topk_weight.value().size(0),
            " vs ", x.size(0));
        TORCH_CHECK(topk_weight.value().size(1) == expert_idx.size(1),
            "topk_weight col count must equal expert_idx col count, got ", topk_weight.value().size(1),
            " vs ", expert_idx.size(1));
        TORCH_CHECK(topk_weight.value().scalar_type() == at::kFloat,
            "topk_weight dtype must be float32, got ", topk_weight.value().scalar_type());
    }

    at::Tensor expanded_x;
    at::Tensor expanded_row_idx;
    at::Tensor expert_tokens_count_or_cumsum;
    at::Tensor expanded_scale;
    at::Tensor expanded_topk_weight;

    int64_t n = x.size(0);
    int64_t cols = x.size(1);
    int64_t k = expert_idx.size(1);

    int64_t active_num_real = (active_num < 0) ? n * k : active_num;
    int64_t out_num = active_num_real;

    int64_t expanded_x_rows = out_num;
    if (drop_pad_mode == 1) {
        expanded_x_rows = expert_num * expert_capacity;
    }

    auto output_dtype = x.scalar_type();
    if (quant_mode == QuantMode::QUANT_MODE_STATIC_QUANT ||
        quant_mode == QuantMode::QUANT_MODE_PERTOKEN ||
        quant_mode == QuantMode::QUANT_MODE_PERGROUP ||
        quant_mode == QuantMode::INT4_DYNAMIC_QUANT) {
        output_dtype = at::kInt8;
    } else if (quant_mode == QuantMode::QUANT_MODE_MX) {
        output_dtype = at::kFloat8_e5m2;
    }

    bool has_topk_weight = topk_weight.has_value() && topk_weight.value().defined();

    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);

        expanded_x = at::empty({expanded_x_rows, cols}, x.options().dtype(output_dtype));
        expanded_row_idx = at::empty({expanded_x_rows}, x.options().dtype(at::kInt));

        int64_t expert_tokens_count_size = expert_num;
        if (expert_tokens_num_type == 1 && expert_tokens_num_flag) {
            expert_tokens_count_size = expert_num + 1;
        }
        expert_tokens_count_or_cumsum = at::empty({expert_tokens_count_size}, x.options().dtype(at::kLong));

        if (quant_mode == QuantMode::QUANT_MODE_NO_QUANT ||
            quant_mode == QuantMode::QUANT_MODE_STATIC_QUANT) {
            expanded_scale = at::empty({out_num}, x.options().dtype(at::kFloat));
        } else if (quant_mode == QuantMode::QUANT_MODE_MX) {
            int64_t cols_aligned = (cols + COLS_ALIGN_FACTOR - 1) / COLS_ALIGN_FACTOR;
            expanded_scale = at::empty({out_num, cols_aligned, 2}, x.options().dtype(at::kFloat8_e5m2));
        } else {
            expanded_scale = at::empty({0}, x.options().dtype(at::kFloat));
        }

        if (has_topk_weight) {
            int64_t topk_weight_rows = out_num;
            if (drop_pad_mode == 1) {
                topk_weight_rows = expert_num * expert_capacity;
            }
            expanded_topk_weight = at::empty({topk_weight_rows, 1}, x.options().dtype(at::kFloat));
        } else {
            expanded_topk_weight = at::empty({0}, x.options().dtype(at::kFloat));
        }
    }

    // topk_weight和expanded_topk_weight必须同时传入或同时不传入aclnn（联动约束）
    // 当topk_weight未提供时，expanded_topk_weight传nullptr而非空shape tensor
    c10::optional<at::Tensor> expanded_topk_weight_opt =
        has_topk_weight ? c10::optional<at::Tensor>(expanded_topk_weight) : c10::optional<at::Tensor>();

    // active_expert_range: std::vector → at::IntArrayRef for ACLNN_CMD
    c10::optional<at::IntArrayRef> active_expert_range_opt;
    if (active_expert_range.has_value()) {
        active_expert_range_opt = c10::optional<at::IntArrayRef>(at::IntArrayRef(active_expert_range.value()));
    } else {
        active_expert_range_opt = c10::optional<at::IntArrayRef>();
    }

    ACLNN_CMD(aclnnMoeInitRoutingV4, x, expert_idx, scale, offset, topk_weight,
              active_num, expert_capacity, expert_num, drop_pad_mode,
              expert_tokens_num_type, expert_tokens_num_flag, quant_mode,
              active_expert_range_opt, row_idx_type,
              expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum,
              expanded_scale, expanded_topk_weight_opt);

    return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum,
                    expanded_scale, expanded_topk_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_init_routing", &npu_moe_init_routing, "npu_moe_init_routing");
}

} // namespace op_api