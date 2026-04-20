/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FIT FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_v5.cpp
 * \brief Wrapper for aclnnGroupedMatmulV5
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;

std::vector<at::Tensor> npu_grouped_matmul_v5(
    const at::TensorList x,
    const at::TensorList weight,
    const c10::optional<at::TensorList> bias,
    const c10::optional<at::TensorList> scale,
    const c10::optional<at::TensorList> offset,
    const c10::optional<at::TensorList> antiquant_scale,
    const c10::optional<at::TensorList> antiquant_offset,
    const c10::optional<at::TensorList> per_token_scale,
    const c10::optional<at::Tensor>& group_list,
    const c10::optional<at::TensorList> activation_input,
    const c10::optional<at::TensorList> activation_quant_scale,
    const c10::optional<at::TensorList> activation_quant_offset,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type,
    int64_t act_type,
    const c10::optional<at::IntArrayRef> tuning_config,
    const c10::optional<at::ScalarType> output_dtype)
{
    auto num_x = x.size();
    auto num_weight = weight.size();
    TORCH_CHECK(num_x > 0, "x list must not be empty");
    TORCH_CHECK(num_weight > 0, "weight list must not be empty");

    // split_item: 0=IN_NOT_SPLIT_OUT_NOT_SPLIT, 1=IN_SPLIT_OUT_NOT_SPLIT,
    //             2=IN_NOT_SPLIT_OUT_SPLIT, 3=IN_SPLIT_OUT_SPLIT
    TORCH_CHECK(split_item >= 0 && split_item <= 3,
                "split_item must be in [0, 3], got ", split_item);

    auto group_list_real = group_list.value_or(at::Tensor());
    auto num_group_list = group_list_real.defined() ? group_list_real.size(0) : 0;

    // Validate dimensions
    if (split_item == 0 || split_item == 1) {
        if (num_group_list > 0) {
            TORCH_CHECK(num_x == 1 && num_weight == num_group_list,
                        "When split_item is 0 or 1 and group_list is not None, "
                        "num of x must be 1 and num of weight must equal group_list length. "
                        "Got x=", num_x, ", weight=", num_weight, ", group_list=", num_group_list);
        } else {
            TORCH_CHECK(num_x == num_weight,
                        "When split_item is 0 or 1 and group_list is None, "
                        "num of x must equal num of weight. Got x=", num_x, ", weight=", num_weight);
        }
    } else {
        if (num_group_list > 0) {
            TORCH_CHECK(num_x == num_group_list && num_weight == num_group_list,
                        "When split_item is 2 or 3 and group_list is not None, "
                        "num of x and weight must both equal group_list length. "
                        "Got x=", num_x, ", weight=", num_weight, ", group_list=", num_group_list);
        }
    }

    c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));
    size_t dim_num_w = weight[0].sizes().size();
    size_t n0 = static_cast<size_t>(weight[0].size(dim_num_w - 1));

    std::vector<at::Tensor> y;

    if (split_item == 0 || split_item == 1) {
        if (num_group_list > 0) {
            y.reserve(num_group_list);
            TORCH_CHECK(group_list_real[0].item<int64_t>() >= 0,
                        "group_list[0] should be >= 0");
            int64_t glr_value_0 = group_list_real[0].item<int64_t>();
            y.emplace_back(at::empty({glr_value_0, static_cast<int64_t>(n0)}, options));
            int64_t glr_value_pre = glr_value_0;
            for (int64_t i = 1; i < num_group_list; i++) {
                int64_t glr_value_cur = group_list_real[i].item<int64_t>();
                TORCH_CHECK(glr_value_cur >= glr_value_pre,
                            "group_list[", i, "] should be >= group_list[", i-1, "]");
                size_t ni = (dim_num_w == 3) ? static_cast<size_t>(weight[i].size(0)) : static_cast<size_t>(weight[i].size(dim_num_w - 1));
                y.emplace_back(at::empty({glr_value_cur - glr_value_pre, static_cast<int64_t>(ni)}, options));
                glr_value_pre = glr_value_cur;
            }
        } else {
            y.reserve(num_x);
            for (size_t i = 0; i < num_x; i++) {
                size_t ni = (dim_num_w == 3) ? static_cast<size_t>(weight[i].size(0)) : static_cast<size_t>(weight[i].size(dim_num_w - 1));
                auto x_sizes = x[i].sizes();
                std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
                y_sizes.at(x_sizes.size() - 1) = static_cast<int64_t>(ni);
                y.emplace_back(at::empty(y_sizes, options));
            }
        }
    } else {
        if (num_x > 1) {
            int64_t dim_m = 0;
            for (size_t i = 0; i < num_x; i++) {
                dim_m += x[i].size(0);
            }
            weight[0].dtype() == at::ScalarType::Int ?
                y.emplace_back(at::empty({dim_m, static_cast<int64_t>(n0) * 8}, options)) :
                y.emplace_back(at::empty({dim_m, static_cast<int64_t>(n0)}, options));
        } else if (num_x == 1) {
            if (group_type == 2) {
                TORCH_CHECK(num_weight == 1,
                            "When group_list is 2 (K_SPLIT) and split_item is 2/3, "
                            "the length of weight must equal x.");
                y.emplace_back(at::empty({num_group_list, x[0].size(0), static_cast<int64_t>(n0)}, options));
            } else {
                weight[0].dtype() == at::ScalarType::Int ?
                    y.emplace_back(at::empty({x[0].size(0), static_cast<int64_t>(n0) * 8}, options)) :
                    y.emplace_back(at::empty({x[0].size(0), static_cast<int64_t>(n0)}, options));
            }
        }
    }

    at::TensorList result = at::TensorList(y);

    auto bias_real = bias.value_or(at::TensorList());
    auto scale_real = scale.value_or(at::TensorList());
    auto offset_real = offset.value_or(at::TensorList());
    auto antiquant_scale_real = antiquant_scale.value_or(at::TensorList());
    auto antiquant_offset_real = antiquant_offset.value_or(at::TensorList());
    auto per_token_scale_real = per_token_scale.value_or(at::TensorList());
    auto activation_input_real = activation_input.value_or(at::TensorList());
    auto activation_quant_scale_real = activation_quant_scale.value_or(at::TensorList());
    auto activation_quant_offset_real = activation_quant_offset.value_or(at::TensorList());
    auto act_out = at::TensorList();
    auto dynamic_quant_scale_out = at::TensorList();
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});

    // Check if weight is FRACTAL_NZ format
    bool is_weight_nz = false;
    try {
        is_weight_nz = at_npu::native::custom_ops::get_npu_format(weight[0]) == ACL_FORMAT_FRACTAL_NZ;
    } catch (...) {
        is_weight_nz = false;
    }

    if (is_weight_nz) {
        static const bool is_weight_nz_available = check_aclnn_kernel_available("aclnnGroupedMatmulWeightNz");
        TORCH_CHECK(is_weight_nz_available,
                    "Format of weight in npu_grouped_matmul_v5 is FRACTAL_NZ, "
                    "but aclnnGroupedMatmulWeightNz is not available in current CANN version.");
        int64_t quant_per_group_size = 0;
        EXEC_NPU_CMD(aclnnGroupedMatmulWeightNz, x, weight, bias_real, scale_real, offset_real,
            antiquant_scale_real, antiquant_offset_real, per_token_scale_real, group_list_real,
            activation_input_real, activation_quant_scale_real, activation_quant_offset_real,
            split_item, group_type, group_list_type, act_type, tuning_config_real,
            quant_per_group_size, result, act_out, dynamic_quant_scale_out);
        return y;
    }

    // Use aclnnGroupedMatmulV5 (default, no is_weight_nz check needed)
    EXEC_NPU_CMD(aclnnGroupedMatmulV5, x, weight, bias_real, scale_real, offset_real,
        antiquant_scale_real, antiquant_offset_real, per_token_scale_real, group_list_real,
        activation_input_real, activation_quant_scale_real, activation_quant_offset_real,
        split_item, group_type, group_list_type, act_type, tuning_config_real,
        result, act_out, dynamic_quant_scale_out);

    return y;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_grouped_matmul_v5", &npu_grouped_matmul_v5, "npu_grouped_matmul_v5");
}
} // namespace op_api
