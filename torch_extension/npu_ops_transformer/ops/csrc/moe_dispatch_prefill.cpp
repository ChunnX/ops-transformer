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
 * \file moe_dispatch_prefill.cpp
 * \brief Fused MoE EP prefill: notify_dispatch + dispatch_normal (Mc2 context).
 */

#include <algorithm>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t QUANT_MODE_NO_QUANT = 0;
constexpr int64_t QUANT_MODE_DYNAMIC_SCALES = 2;
constexpr int64_t EXPAND_IDX_INFO = 3;
constexpr int64_t SEND_PER_GROUP = 3;

tensor_list npu_moe_dispatch_prefill(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_idx,
    const std::vector<at::Tensor> &ext_handle_data, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t local_ranksize, int64_t moe_expert_num, int64_t ccl_buffer_size,
    const c10::optional<at::Tensor> &x_scales, const c10::optional<at::Tensor> &topk_weights,
    const c10::optional<at::Tensor> &num_tokens_per_rank, const c10::optional<at::Tensor> &is_token_in_rank,
    const c10::optional<at::Tensor> &num_tokens_per_expert,
    const c10::optional<at::Tensor> &dispatch_wait_recv_cost_stats, int64_t num_worst_tokens,
    int64_t quant_mode, int64_t round, int64_t per_round_tokens)
{
    (void)x_scales;
    (void)topk_weights;
    (void)is_token_in_rank;

    TORCH_CHECK(x.dim() == DIM_TWO && topk_idx.dim() == DIM_TWO, "x and topk_idx must be 2D tensors.");
    TORCH_CHECK(ext_handle_data.size() == 2,
                "ext_handle_data must contain exactly 2 tensors [notify_send_data, send_token_idx_small].");
    TORCH_CHECK(num_tokens_per_expert.has_value(), "num_tokens_per_expert is required for notify_dispatch.");
    TORCH_CHECK((moe_expert_num % ep_world_size) == 0,
                "moe_expert_num must be divisible by ep_world_size, but got moe_expert_num=", moe_expert_num,
                ", ep_world_size=", ep_world_size, ".");
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got ep_world_size=", ep_world_size,
                ", ep_rank_id=", ep_rank_id, ".");
    TORCH_CHECK(ccl_buffer_size > 0, "ccl_buffer_size must be positive, but got ", ccl_buffer_size, ".");
    TORCH_CHECK(round > 0, "round must be greater than 0, but got ", round, ".");
    TORCH_CHECK(per_round_tokens > 0, "per_round_tokens must be greater than 0, but got ", per_round_tokens, ".");
    TORCH_CHECK((quant_mode == QUANT_MODE_NO_QUANT) || (quant_mode == QUANT_MODE_DYNAMIC_SCALES),
                "quant_mode should be 0 (no quant) or 2 (dynamic int8), but got ", quant_mode, ".");

    if (num_tokens_per_rank.has_value()) {
        const at::Tensor &num_tokens_per_rank_tensor = num_tokens_per_rank.value();
        TORCH_CHECK(num_tokens_per_rank_tensor.dim() == DIM_ONE,
                    "num_tokens_per_rank must be 1D tensor.");
        TORCH_CHECK(num_tokens_per_rank_tensor.size(0) == ep_world_size,
                    "num_tokens_per_rank size(0) must equal ep_world_size.");
    }

    const at::Tensor &token_per_expert_data = num_tokens_per_expert.value();
    TORCH_CHECK(token_per_expert_data.dim() == DIM_ONE, "num_tokens_per_expert must be 1D tensor.");
    TORCH_CHECK(token_per_expert_data.size(0) == moe_expert_num * round,
                "num_tokens_per_expert size(0) must equal moe_expert_num * round, but got ",
                token_per_expert_data.size(0), " and ", moe_expert_num * round, ".");

    const at::Tensor &send_data = ext_handle_data[0];
    const at::Tensor &send_token_idx_small = ext_handle_data[1];
    TORCH_CHECK(send_data.dim() == DIM_ONE && send_token_idx_small.dim() == DIM_TWO,
                "send_data must be 1D and send_token_idx_small must be 2D.");

    int64_t num_tokens = x.size(0);
    int64_t h = x.size(1);
    TORCH_CHECK(topk_idx.size(0) == num_tokens, "topk_idx dim0 must equal x dim0.");
    int64_t local_rank_id = ep_rank_id % local_ranksize;
    int64_t send_count = SEND_PER_GROUP * moe_expert_num * round;
    int64_t num_local_experts = moe_expert_num / ep_world_size;

    at::Tensor send_offset = at::empty({round * moe_expert_num},
                                       at::TensorOptions()
                                           .dtype(at::kInt)
                                           .device(c10::DeviceType::PrivateUse1)
                                           .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_data = at::empty({send_count},
                                     at::TensorOptions()
                                         .dtype(at::kInt)
                                         .device(c10::DeviceType::PrivateUse1)
                                         .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_count = at::empty({round * moe_expert_num},
                                      at::TensorOptions()
                                          .dtype(at::kInt)
                                          .device(c10::DeviceType::PrivateUse1)
                                          .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_offset = at::empty({round * moe_expert_num},
                                       at::TensorOptions()
                                           .dtype(at::kInt)
                                           .device(c10::DeviceType::PrivateUse1)
                                           .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor expert_global_offset = at::empty({num_local_experts},
                                                at::TensorOptions()
                                                    .dtype(at::kInt)
                                                    .device(c10::DeviceType::PrivateUse1)
                                                    .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor srcrank_in_expert_offset = at::empty({num_local_experts * ep_world_size},
                                                    at::TensorOptions()
                                                        .dtype(at::kInt)
                                                        .device(c10::DeviceType::PrivateUse1)
                                                        .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor r_in_srcrank_offset = at::empty({num_local_experts * ep_world_size * round},
                                               at::TensorOptions()
                                                   .dtype(at::kInt)
                                                   .device(c10::DeviceType::PrivateUse1)
                                                   .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor total_recv_tokens = at::empty({1},
                                             at::TensorOptions()
                                                 .dtype(at::kInt)
                                                 .device(c10::DeviceType::PrivateUse1)
                                                 .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor max_bs = at::empty({1},
                                  at::TensorOptions()
                                      .dtype(at::kInt)
                                      .device(c10::DeviceType::PrivateUse1)
                                      .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_tokens_per_expert = at::empty({round * num_local_experts},
                                                  at::TensorOptions()
                                                      .dtype(at::kInt)
                                                      .device(c10::DeviceType::PrivateUse1)
                                                      .memory_format(c10::MemoryFormat::Contiguous));

    ACLNN_CMD(aclnnNotifyDispatch, context, send_data, token_per_expert_data, send_count, num_tokens, ep_world_size,
              ep_rank_id, local_ranksize, local_rank_id, ccl_buffer_size, round, per_round_tokens, send_offset,
              recv_data, recv_count, recv_offset, expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset,
              total_recv_tokens, max_bs, recv_tokens_per_expert);

    std::vector<at::Tensor> recv_offset_list = {recv_offset, recv_count};
    std::vector<at::Tensor> expert_output_offset_list = {expert_global_offset, srcrank_in_expert_offset,
                                                         r_in_srcrank_offset};

    int64_t num_recv_tokens = total_recv_tokens.item<int64_t>();
    if (num_recv_tokens <= 0) {
        num_recv_tokens = 1;
    }
    int64_t real_max_bs = std::max(max_bs.item<int64_t>(), num_worst_tokens);
    int64_t global_bs = std::min(per_round_tokens, real_max_bs) * ep_world_size;

    auto output_dtype = at::kChar;
    if (quant_mode == QUANT_MODE_NO_QUANT) {
        output_dtype = x.scalar_type();
    }

    at::Tensor recv_x = at::empty({num_recv_tokens, h},
                                  at::TensorOptions()
                                      .dtype(output_dtype)
                                      .device(c10::DeviceType::PrivateUse1)
                                      .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor dynamic_scales = at::empty({num_recv_tokens},
                                           at::TensorOptions()
                                               .dtype(at::kFloat)
                                               .device(c10::DeviceType::PrivateUse1)
                                               .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor assist_info_for_combine = at::empty({num_recv_tokens * EXPAND_IDX_INFO},
                                                   at::TensorOptions()
                                                       .dtype(at::kInt)
                                                       .device(c10::DeviceType::PrivateUse1)
                                                       .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor ext_handle_data_out = at::empty({0},
                                               at::TensorOptions()
                                                   .dtype(at::kInt)
                                                   .device(c10::DeviceType::PrivateUse1)
                                                   .memory_format(c10::MemoryFormat::Contiguous));

    at::Tensor dispatch_wait_recv_cost_stats_out;
    if (dispatch_wait_recv_cost_stats.has_value()) {
        const at::Tensor &stats = dispatch_wait_recv_cost_stats.value();
        TORCH_CHECK(stats.scalar_type() == at::kInt, "dispatch_wait_recv_cost_stats dtype must be int32.");
        TORCH_CHECK(stats.dim() == DIM_ONE && stats.is_contiguous(),
                    "dispatch_wait_recv_cost_stats must be contiguous 1D tensor.");
        TORCH_CHECK(stats.size(0) == ep_world_size,
                    "dispatch_wait_recv_cost_stats shape[0] must equal ep_world_size.");
        dispatch_wait_recv_cost_stats_out = stats;
    }

    at::TensorList recv_offset_ref = recv_offset_list;
    at::TensorList expert_output_offset_ref = expert_output_offset_list;

    ACLNN_CMD(aclnnDispatchNormal, context, x, topk_idx, send_offset, send_token_idx_small, recv_offset_ref,
              expert_output_offset_ref, ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size, quant_mode,
              real_max_bs, global_bs, round, per_round_tokens, recv_x, dynamic_scales, assist_info_for_combine,
              dispatch_wait_recv_cost_stats_out, ext_handle_data_out);

    return std::make_tuple(recv_x, dynamic_scales, assist_info_for_combine, recv_count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_dispatch_prefill", &npu_moe_dispatch_prefill, "moe_dispatch_prefill");
}
} // namespace op_api
