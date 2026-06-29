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
 * \file causal_conv1d_fn.cpp
 * \brief Torch bridge — prefill / chunk-prefill, delegates to aclnnCausalConv1dFn.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor causal_conv1d_fn(const at::Tensor &x, const at::Tensor &weight,
                            const at::Tensor &bias,
                            at::Tensor &conv_states,
                            const c10::optional<at::Tensor> &query_start_loc,
                            const c10::optional<at::Tensor> &cache_indices,
                            const c10::optional<at::Tensor> &has_initial_state,
                            const std::string &activation, int64_t pad_slot_id, int64_t null_block_id,
                            const c10::optional<at::Tensor> &block_idx_first_scheduled_token,
                            const c10::optional<at::Tensor> &block_idx_last_scheduled_token,
                            const c10::optional<at::Tensor> &initial_state_idx,
                            const c10::optional<at::Tensor> &num_computed_tokens,
                            int64_t block_size_to_align)
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
    ACLNN_CMD(aclnnCausalConv1dFn, x, weight, conv_states, bias, query_start_loc, cache_indices, has_initial_state,
              block_idx_first_scheduled_token, block_idx_last_scheduled_token,
              initial_state_idx, num_computed_tokens,
              activation_mode, null_block_id, block_size_to_align, y);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("causal_conv1d_fn", &causal_conv1d_fn, "causal_conv1d prefill");
}

} // namespace op_api
