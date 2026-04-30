/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdint>
#include <climits>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
using namespace at_npu::native;

at::Tensor npu_ai_infra_causal_conv1d_fn_add(const at::Tensor &x, const at::Tensor &weight, at::Tensor &conv_states,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &cache_indices, const c10::optional<at::Tensor> &initial_state_model,
    const c10::optional<c10::string_view> activation, const c10::optional<int64_t> residual_connection, const c10::optional<int64_t> pad_slot_id)
{

    at::Tensor y = at::empty_like(x);

    std::string activation_str = "None";

    if(activation.has_value())
       activation_str = *activation;

    int64_t activation_mode = 0;

    if (activation_str == "silu") {
        activation_mode = 1;
    } else if (activation_str == "swish") {
        activation_mode = 2;
    }

    int64_t run_mode = 0;

    int64_t pad_slot_id_const = pad_slot_id.value_or(-1);
    int64_t residual_connection_const = residual_connection.value_or(1);

    EXEC_NPU_CMD_V1(aclnnAiInfraCausalConv1dAdd,
        x,
        weight,
        conv_states,
        query_start_loc,
        cache_indices,
        initial_state_model,
        bias,
        c10::nullopt,  // num_accepted_tokens
        activation_mode,
        pad_slot_id_const,
        run_mode,
        residual_connection_const,
        y);

    return y;
}

// 为META设备实现前向接口
at::Tensor npu_ai_infra_causal_conv1d_fn_add_meta(const at::Tensor &x, const at::Tensor &weight, at::Tensor &conv_states,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &cache_indices, const c10::optional<at::Tensor> &initial_state_model,
    const c10::optional<c10::string_view> activation, const c10::optional<int64_t> residual_connection, const c10::optional<int64_t> pad_slot_id)
{
    at::Tensor y = at::empty_like(x);
    return y;
}
}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_ai_infra_causal_conv1d_fn_add", &custom::npu_ai_infra_causal_conv1d_fn_add);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_ai_infra_causal_conv1d_fn_add", &custom::npu_ai_infra_causal_conv1d_fn_add_meta);
}