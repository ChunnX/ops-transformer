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
 * \file grouped_matmul_v5.cpp
 * \brief ACLNN wrapper for GroupedMatmulV5
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::TensorList, c10::optional<at::TensorList>, c10::optional<at::TensorList>>;

/**
 * @brief ACLNN Wrapper for aclnnGroupedMatmulV5
 * @param x: Input tensor list
 * @param weight: Weight tensor list
 * @param bias: Optional bias tensor list
 * @param scale: Optional quantization scale tensor list
 * @param offset: Optional quantization offset tensor list
 * @param antiquant_scale: Optional antiquant scale tensor list
 * @param antiquant_offset: Optional antiquant offset tensor list
 * @param per_token_scale: Optional per-token scale tensor list
 * @param group_list: Optional group list tensor
 * @param activation_input: Optional activation input tensor list
 * @param activation_quant_scale: Optional activation quant scale tensor list
 * @param activation_quant_offset: Optional activation quant offset tensor list
 * @param split_item: Split item mode (0/1: multi-tensor, 2/3: single-tensor)
 * @param group_type: Group type (-1: no split, 0: M-axis, 1: N-axis, 2: K-axis)
 * @param group_list_type: Group list type (0: cumsum, 1: size per group)
 * @param act_type: Activation type (0: none, 1: relu, 2: gelu_tanh, 3: gelu_erf, 4: fast_gelu, 5: silu)
 * @param tuning_config: Optional tuning config
 * @return Tuple of (out, activation_feature_out, dyn_quant_scale_out)
 */
tensor_list npu_grouped_matmul_v5(
    const at::TensorList &x,
    const at::TensorList &weight,
    const c10::optional<at::TensorList> &bias,
    const c10::optional<at::TensorList> &scale,
    const c10::optional<at::TensorList> &offset,
    const c10::optional<at::TensorList> &antiquant_scale,
    const c10::optional<at::TensorList> &antiquant_offset,
    const c10::optional<at::TensorList> &per_token_scale,
    const c10::optional<at::Tensor> &group_list,
    const c10::optional<at::TensorList> &activation_input,
    const c10::optional<at::TensorList> &activation_quant_scale,
    const c10::optional<at::TensorList> &activation_quant_offset,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type,
    int64_t act_type,
    const c10::optional<at::IntArrayRef> &tuning_config)
{
    // Parameter validation
    TORCH_CHECK(!x.empty(), "x tensor list cannot be empty");
    TORCH_CHECK(!weight.empty(), "weight tensor list cannot be empty");
    TORCH_CHECK(x.size() == weight.size(),
                "x and weight must have the same number of tensors, but got ",
                "x.size()=", x.size(), ", weight.size()=", weight.size());

    TORCH_CHECK(split_item >= 0 && split_item <= 3,
                "split_item must be in [0, 3], but got ", split_item);
    TORCH_CHECK(group_type >= -1 && group_type <= 2,
                "group_type must be in [-1, 2], but got ", group_type);
    TORCH_CHECK(group_list_type >= 0 && group_list_type <= 1,
                "group_list_type must be in [0, 1], but got ", group_list_type);
    TORCH_CHECK(act_type >= 0 && act_type <= 5,
                "act_type must be in [0, 5], but got ", act_type);

    // Allocate output tensors based on split_item
    std::vector<at::Tensor> out_vec;
    if (split_item == 0 || split_item == 1) {
        // Multi-tensor output
        for (size_t i = 0; i < x.size(); i++) {
            auto x_shape = x[i].sizes();
            auto weight_shape = weight[i].sizes();
            TORCH_CHECK(x_shape.size() >= 2 && weight_shape.size() >= 2,
                        "x and weight must be at least 2D tensors");

            int64_t m = x_shape[x_shape.size() - 2];
            int64_t n = weight_shape[weight_shape.size() - 1];

            std::vector<int64_t> out_shape(x_shape.begin(), x_shape.end() - 1);
            out_shape.push_back(n);

            out_vec.push_back(at::empty(out_shape, x[i].options()));
        }
    } else {
        // Single-tensor output
        int64_t total_m = 0;
        int64_t n = weight[0].size(-1);

        for (const auto &x_tensor : x) {
            total_m += x_tensor.size(-2);
        }

        out_vec.push_back(at::empty({total_m, n}, x[0].options()));
    }

    // Optional outputs
    c10::optional<at::TensorList> activation_feature_out = c10::nullopt;
    c10::optional<at::TensorList> dyn_quant_scale_out = c10::nullopt;

    // Call ACLNN API
    ACLNN_CMD(aclnnGroupedMatmulV5,
              x, weight, bias, scale, offset,
              antiquant_scale, antiquant_offset, per_token_scale,
              group_list, activation_input, activation_quant_scale,
              activation_quant_offset, split_item, group_type,
              group_list_type, act_type, tuning_config,
              out_vec, activation_feature_out, dyn_quant_scale_out);

    return std::make_tuple(out_vec, activation_feature_out, dyn_quant_scale_out);
}

} // namespace op_api

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_grouped_matmul_v5", &op_api::npu_grouped_matmul_v5, "GroupedMatmulV5");
}
