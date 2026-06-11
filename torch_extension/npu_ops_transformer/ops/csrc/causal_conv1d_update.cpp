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
 * \brief Torch bridge — decode / update, delegates to aclnnCausalConv1d.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor npu_causal_conv1d_update(const at::Tensor &x, const at::Tensor &weight,
                                    const c10::optional<at::Tensor> &bias, at::Tensor &conv_states,
                                    const c10::optional<at::Tensor> &conv_state_indices,
                                    const c10::optional<at::Tensor> &query_start_loc,
                                    const c10::optional<at::Tensor> &initial_state_idx,
                                    const c10::optional<at::Tensor> &num_accepted_tokens,
                                    const c10::optional<at::Tensor> &block_idx_last_scheduled_token,
                                    int64_t activation_mode, int64_t null_block_id, int64_t max_query_len)
{
    // TODO: conv_state_indices, initial_state_idx, block_idx_last_scheduled_token,
    //       max_query_len not yet supported by current aclnnCausalConv1d.
    (void)conv_state_indices;
    (void)initial_state_idx;
    (void)block_idx_last_scheduled_token;
    (void)max_query_len;

    at::Tensor y = at::empty_like(x);

    c10::optional<at::Tensor> no_cache_indices;
    c10::optional<at::Tensor> no_initial_state_mode;

    ACLNN_CMD(aclnnCausalConv1d, x, weight, bias, conv_states, query_start_loc,
              no_cache_indices,      // cacheIndices — not used in decode
              no_initial_state_mode, // initialStateMode — not used in decode
              num_accepted_tokens, activation_mode, null_block_id, y);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_causal_conv1d_update", &npu_causal_conv1d_update, "causal_conv1d decode/update");
}

} // namespace op_api
