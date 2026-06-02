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
 * \brief Torch bridge — decode / update, delegates to aclnnCausalConv1d(runMode=1).
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor>
npu_causal_conv1d_update(
    const at::Tensor &x, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
    at::Tensor &conv_states,
    const c10::optional<at::Tensor> &conv_state_indices, const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &initial_state_idx,
    const c10::optional<at::Tensor> &num_accepted_tokens,
    const c10::optional<at::Tensor> &block_idx_last_scheduled_token,
    int64_t activation_mode, int64_t null_block_id, int64_t max_query_len)
{
    // TODO: conv_state_indices, initial_state_idx, block_idx_last_scheduled_token,
    //       null_block_id, max_query_len not yet supported by current aclnnCausalConv1d.
    (void)conv_state_indices;
    (void)initial_state_idx;
    (void)block_idx_last_scheduled_token;
    (void)null_block_id;
    (void)max_query_len;

    at::Tensor y = at::empty_like(x);

    ACLNN_CMD(aclnnCausalConv1d,
              x, weight, bias, conv_states,
              query_start_loc,
              c10::nullopt,           // cacheIndices — not used in decode
              c10::nullopt,           // initialStateMode — not used in decode
              num_accepted_tokens,
              activation_mode,
              0,                     // padSlotId — not used in decode
              1,                     // runMode = decode / update
              y);

    return std::make_tuple(y, conv_states);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_causal_conv1d_update", &npu_causal_conv1d_update, "causal_conv1d decode/update");
}

} // namespace op_api
