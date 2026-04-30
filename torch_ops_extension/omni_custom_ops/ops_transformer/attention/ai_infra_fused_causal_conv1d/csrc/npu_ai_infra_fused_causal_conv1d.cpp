/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"


namespace custom {
using namespace at_npu::native;

// 如果x原地更新的话，kernel侧可以不copyout y，GM上y是垃圾值，卷积output写到x的GM上
// EXEC_NPU_CMD_V1后，x是原地更新后的，y可能是垃圾值。如果需要算子的输出依然是正确的卷积结果，可以return x
at::Tensor npu_ai_infra_fused_causal_conv1d_npu(
    const at::Tensor &x, const at::Tensor &weight, at::Tensor &conv_states,
    const c10::optional<at::Tensor> &query_start_loc, const c10::optional<at::Tensor> &cache_indices,
    const c10::optional<at::Tensor> &initial_state_mode, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &num_accepted_tokens, const c10::optional<at::Tensor> &num_computed_tokens,
    const c10::optional<at::Tensor> &block_idx_first_scheduled_token,
    const c10::optional<at::Tensor> &block_idx_last_scheduled_token, const c10::optional<at::Tensor> &initial_state_idx,
    c10::optional<std::string_view> activation, int64_t pad_slot_id, int64_t run_mode,
    c10::optional<int64_t> max_query_len, int64_t residual_connection, int64_t block_size, int64_t conv_mode,
    bool inplace)
{
    at::Tensor y = at::empty_like(x);

    // activation在PTA层是str，在aclnn层是int
    int64_t activation_mode = 0;
    if (activation.has_value()) {
        if (activation.value() == "silu") {
            activation_mode = 1;
        } else if (activation.value() == "swish") {
            activation_mode = 2;
        }
    }

    // c10::optional定义的参数 需要先取值再传入L2接口，否则地址错误导致yOut==nullptr
    int64_t max_query_len_const = max_query_len.value_or(-1); // 为空时返回默认值

    // 传入顺序和aclnn接口一致
    EXEC_NPU_CMD_V1(aclnnAiInfraFusedCausalConv1d, x, weight, conv_states, query_start_loc, cache_indices,
                    c10::nullopt, // initial_state_model空值, 空的Optional对应py的None
                    bias, num_accepted_tokens, num_computed_tokens, block_idx_first_scheduled_token,
                    block_idx_last_scheduled_token, initial_state_idx, activation_mode, pad_slot_id, run_mode,
                    max_query_len_const, residual_connection, block_size, conv_mode, inplace, y);

    return y;

    // 考虑需求4： x原地更新
    // 若kernel完全不动, host侧再多次copy
    // if (inplace) {
    //     x = y;
    // }
    // return y;

    // 若kernel修改 copyout卷积结果到xGm, 可以零开销让返回值也是正确结果，即使yGm是垃圾值
    // if (!inplace) {
    //     return y;
    // } else {
    //     return x;
    // }
}

// Meta implementation: shape/dtype inference only, no NPU execution.
// Return empty tensors of the correct shape/dtype — no device required.
at::Tensor npu_ai_infra_fused_causal_conv1d_meta(
    const at::Tensor &x, const at::Tensor &weight, at::Tensor &conv_states,
    const c10::optional<at::Tensor> &query_start_loc, const c10::optional<at::Tensor> &cache_indices,
    const c10::optional<at::Tensor> &initial_state_mode, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &num_accepted_tokens, const c10::optional<at::Tensor> &num_computed_tokens,
    const c10::optional<at::Tensor> &block_idx_first_scheduled_token,
    const c10::optional<at::Tensor> &block_idx_last_scheduled_token, const c10::optional<at::Tensor> &initial_state_idx,
    c10::optional<std::string_view> activation, int64_t pad_slot_id, int64_t run_mode,
    c10::optional<int64_t> max_query_len, int64_t residual_connection, int64_t block_size, int64_t conv_mode,
    bool inplace)
{
    return at::empty_like(x);
}

// Register NPU dispatch — TORCH_LIBRARY_IMPL must be inside namespace custom {}
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_ai_infra_fused_causal_conv1d", &custom::npu_ai_infra_fused_causal_conv1d_npu);
}

// Register Meta dispatch
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_ai_infra_fused_causal_conv1d", &custom::npu_ai_infra_fused_causal_conv1d_meta);
}

} // namespace custom