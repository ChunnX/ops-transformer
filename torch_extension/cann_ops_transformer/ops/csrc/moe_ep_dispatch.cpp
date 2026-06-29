/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_dispatch.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_moe_ep_dispatch(const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_idx,
    const c10::optional<at::Tensor> &topk_weights, const c10::optional<at::Tensor> &scales,
    const c10::optional<at::Tensor> &cached_handle_dst_buffer_slot_idx, int64_t ep_world_size,
    int64_t ep_rank_id, int64_t num_experts, int64_t num_max_tokens_per_rank,
    int64_t ccl_buffer_size, int64_t expert_alignment, bool do_cpu_sync, int64_t host_pinned_counter_addr)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D");
    TORCH_CHECK((ep_world_size > 1), "The ep_world_sizes should be greater than 1, current is: ", ep_world_size);
    TORCH_CHECK(ep_rank_id >= 0 && ep_rank_id < ep_world_size, "ep_rank_id out of range");
    TORCH_CHECK(num_experts % ep_world_size == 0, "num_experts must be divisible by ep_world_size");

    bool any_cached = cached_handle_dst_buffer_slot_idx.has_value();
    TORCH_CHECK(!(any_cached && do_cpu_sync), "cached mode is incompatible with do_cpu_sync=True");

    auto x_size = x.sizes();
    int64_t num_tokens = x_size[0];
    int64_t hidden = x_size[1];
    int64_t top_k = topk_idx.size(1);
    int64_t num_local_experts = num_experts / ep_world_size;
    int64_t upper_a = ep_world_size * num_max_tokens_per_rank;

    at::Tensor num_recv_per_rank;
    at::Tensor num_recv_per_expert;
    at::Tensor dst_slot;
    num_recv_per_rank = at::empty({ep_world_size}, x.options().dtype(at::kInt));
    num_recv_per_expert = at::empty({num_local_experts}, x.options().dtype(at::kLong));
    dst_slot = at::empty({num_tokens, top_k}, x.options().dtype(at::kInt));

    at::Tensor topk_weights_tensor = topk_weights.has_value() ? *topk_weights : at::Tensor();
    at::Tensor cached_slot_tensor =
        cached_handle_dst_buffer_slot_idx.has_value() ? *cached_handle_dst_buffer_slot_idx : at::Tensor();

    aclDataType scales_dtype = aclDataType::ACL_FLOAT;
    at::Tensor scales_tensor = scales.has_value() ? *scales : at::Tensor();
    if (scales.has_value() && scales_tensor.scalar_type() == at::kByte) {
        scales_dtype = aclDataType::ACL_FLOAT8_E8M0;
    }
    TensorWrapper scales_wrapper = TensorWrapper{scales_tensor, scales_dtype};

    ACLNN_CMD(aclnnMoeEpDispatch, context, x, topk_idx, topk_weights_tensor, scales_wrapper, cached_slot_tensor,
        ep_world_size, ep_rank_id, num_experts, num_max_tokens_per_rank, ccl_buffer_size, expert_alignment,
        do_cpu_sync, host_pinned_counter_addr, num_recv_per_rank, num_recv_per_expert, dst_slot);

    return std::tie(num_recv_per_rank, num_recv_per_expert, dst_slot);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_ep_dispatch", &npu_moe_ep_dispatch, "moe_ep_dispatch");
}

}  // namespace op_api