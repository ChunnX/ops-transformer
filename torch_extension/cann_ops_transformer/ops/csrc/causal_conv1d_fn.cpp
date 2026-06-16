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

at::Tensor npu_causal_conv1d_fn(const at::Tensor &x, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                                at::Tensor &conv_states, const c10::optional<at::Tensor> &query_start_loc,
                                const c10::optional<at::Tensor> &cache_indices,
                                const c10::optional<at::Tensor> &has_initial_state,
                                const c10::optional<at::Tensor> &initial_state_idx,
                                const c10::optional<at::Tensor> &num_computed_tokens,
                                const c10::optional<at::Tensor> &block_idx_first_scheduled_token,
                                const c10::optional<at::Tensor> &block_idx_last_scheduled_token,
                                int64_t activation_mode, int64_t pad_slot_id, int64_t null_block_id,
                                int64_t block_size_to_align)
{
    at::Tensor y = at::empty_like(x);

    ACLNN_CMD(aclnnCausalConv1dFn, x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state,
              activation_mode, pad_slot_id, null_block_id, block_idx_first_scheduled_token,
              block_idx_last_scheduled_token, initial_state_idx, num_computed_tokens, block_size_to_align, y);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_causal_conv1d_fn", &npu_causal_conv1d_fn, "causal_conv1d prefill");
}

} // namespace op_api
