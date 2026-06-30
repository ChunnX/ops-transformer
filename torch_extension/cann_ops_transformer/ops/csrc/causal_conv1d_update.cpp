/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update.cpp
 * \brief Torch bridge — decode / update, delegates to aclnnCausalConv1dUpdate.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor causal_conv1d_update(const at::Tensor &x, at::Tensor &conv_states,
                                const at::Tensor &weight,
                                const c10::optional<at::Tensor> &bias,
                                const std::string &activation,
                                const c10::optional<at::Tensor> &cache_indices,
                                const c10::optional<at::Tensor> &num_accepted_tokens,
                                const c10::optional<at::Tensor> &query_start_loc,
                                int64_t max_query_len, int64_t null_block_id,
                                const c10::optional<at::Tensor> &block_idx_last_scheduled_token,
                                const c10::optional<at::Tensor> &initial_state_idx)
{
    TORCH_CHECK(activation == "silu" || activation == "none",
                "activation must be 'silu' or 'none', got: ", activation);

    at::Tensor y{nullptr};
    {
        auto local_device = c10::Device(x.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        y = at::empty_like(x);
    }

    const char* activation_mode = activation.c_str();
    ACLNN_CMD(aclnnCausalConv1dUpdate, x, weight, conv_states, bias, query_start_loc, cache_indices,
              num_accepted_tokens, block_idx_last_scheduled_token, initial_state_idx,
              activation_mode, null_block_id, max_query_len, y);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("causal_conv1d_update", &causal_conv1d_update, "causal_conv1d decode/update");
}

} // namespace op_api
