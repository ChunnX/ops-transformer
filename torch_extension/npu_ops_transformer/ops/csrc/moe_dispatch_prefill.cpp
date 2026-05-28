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
 * \brief ACLNN wrapper for npu_moe_dispatch_prefill (MoE EP dispatch with Mc2 context).
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t QUANT_MODE_NO_QUANT = 0;
constexpr int64_t QUANT_MODE_DYNAMIC_SCALES = 2;
constexpr int64_t EXPAND_IDX_INFO = 3;
constexpr int64_t SEND_PER_GROUP = 3;   // (send_to_expert_num, send_to_expert_offset, send_rank_tokens)

// tensor_list npu_notify_dispatch(const at::Tensor &send_data, const at::Tensor &token_per_expert_data,
//                                 int64_t send_count, int64_t num_tokens,
//                                 std::string group_ep, int64_t rank_size, int64_t rank_id,
//                                 int64_t local_rank_size, int64_t local_rank_id,
//                                 int64_t round, int64_t per_round_tokens)

tensor_list npu_moe_dispatch_prefill(const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_idx,
                                     const std::vector<at::Tensor> &ext_handle_data, int64_t ep_world_size, int64_t ep_rank_id,
                                     int64_t local_ranksize, int64_t moe_expert_num, int64_t ccl_buffer_size,
                                     const c10::optional<at::Tensor> &x_scales,
                                     const c10::optional<at::Tensor> &topk_weights,
                                     const c10::optional<at::Tensor> &num_tokens_per_rank,
                                     const c10::optional<at::Tensor> &is_token_in_rank,
                                     const c10::optional<at::Tensor> &num_tokens_per_expert,
                                     const c10::optional<at::Tensor> &dispatch_wait_recv_cost_stats,
                                     int64_t num_worst_tokens, int64_t quant_mode,
                                     int64_t round, int64_t per_round_tokens)
// tensor_list npu_dispatch_normal(const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_idx,
//                                  const at::Tensor &send_offset, const at::Tensor &send_token_idx,
//                                  const std::vector<at::Tensor> &recv_offset,
//                                  const std::vector<at::Tensor> &expert_output_offset, int64_t ep_world_size,
//                                  int64_t ep_rank_id, int64_t moe_expert_num, int64_t ccl_buffer_size,
//                                  int64_t num_recv_tokens, int64_t quant_mode, int64_t real_max_bs, int64_t global_bs,
//                                  int64_t round, int64_t per_round_tokens,
//                                  const c10::optional<at::Tensor> &dispatch_wait_recv_cost_stats,
//                                  const c10::optional<at::Tensor> &ext_handle_data)
{
    TORCH_CHECK(x.dim() == DIM_TWO && topk_idx.dim() == DIM_TWO, "x and topk_idx must be 2D.");
    TORCH_CHECK(num_tokens_per_rank.dim() == DIM_ONE, "num_tokens_per_rank should be 1D.");
    TORCH_CHECK(num_tokens_per_expert.dim() == DIM_ONE, "num_tokens_per_expert should be 1D.");

    int64_t num_tokens = x.size(0)
    int64_t h = x.size(1)
    int64_t k = topk_idx.size(1)
    TORCH_CHECK(topk_idx.size(0) == num_tokens, "topk_idx's dim0 should be equal to x's dim0.");
    TORCH_CHECK((moe_expert_num % ep_world_size) == 0,
                "moe_expert_num should be divisible by ep_world_size, current moe_expert_num: ",
                moe_expert_num, ", ep_world_size: ", ep_world_size);
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), current ep_world_size: ",
                ep_world_size, ", ep_rank_id: ", ep_rank_id);
    TORCH_CHECK((local_rank_id >= 0) && (local_rank_id < local_ranksize),
                "local_rank_id should be in [0, local_ranksize), current local_ranksize: ",
                local_ranksize, ", local_rank_id: ", local_rank_id);
    TORCH_CHECK(ccl_buffer_size > 0, "ccl_buffer_size must be positive, but got ", ccl_buffer_size, ".");
    TORCH_CHECK(round > 0, "round should be greater than 0, current is: ", round);
    TORCH_CHECK((quant_mode == QUANT_MODE_NO_QUANT) || (quant_mode == QUANT_MODE_DYNAMIC_SCALES),
                "quant_mode should be 0 (no quant) or 2 (dynamic int8), but got ", quant_mode, ".");
    TORCH_CHECK(num_tokens_per_rank.size(0) == ep_world_size,
                "the value of num_tokens_per_rank's dim0 should be equal to ep_world_size.");
    // TORCH_CHECK(num_tokens_per_expert.size(0) % moe_expert_num == 0,
    //             "moe_expert_num should be divisible by the value of num_tokens_per_expert's dim0.");
    TORCH_CHECK(num_tokens_per_expert.size(0) == moe_expert_num * round,
                "the value of num_tokens_per_expert's dim0 should be equal to (moe_expert_num * round).");

    at::Tensor &send_data = ext_handle_data[0];
    at::Tensor &send_token_idx_small = ext_handle_data[1];
    at::Tensor dispatch_wait_recv_cost_stats_out;
    if (dispatch_wait_recv_cost_stats.has_value()) {
        const at::Tensor &dispatch_wait_recv_cost_stats_tensor = dispatch_wait_recv_cost_stats.value();
        TORCH_CHECK(dispatch_wait_recv_cost_stats_tensor.scalar_type() == at::kInt,
                    "dispatch_wait_recv_cost_stats dtype must be int32.");
        TORCH_CHECK(dispatch_wait_recv_cost_stats_tensor.dim() == DIM_ONE,
                    "dispatch_wait_recv_cost_stats must be contiguous 1D tensor.");
        TORCH_CHECK(dispatch_wait_recv_cost_stats_tensor.size(0) == ep_world_size,
                    "dispatch_wait_recv_cost_stats shape[0] must equal ep_world_size, but got ",
                    dispatch_wait_recv_cost_stats_tensor.size(0), " and ", ep_world_size, ".");
        dispatch_wait_recv_cost_stats_out = dispatch_wait_recv_cost_stats_tensor;
    }

    // int64_t num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    // int64_t num_experts = num_tokens_per_expert.size(0) / round;
    int64_t num_local_experts = moe_expert_num / ep_world_size;
    int64_t local_rank_id = ep_rank_id % local_ranksize;
    int64_t send_count = SEND_PER_GROUP * num_experts * round;
    auto token_per_expert_data = num_tokens_per_expert.value();
    // char *group_ep_ptr = const_cast<char *>(group_ep.data());

    at::Tensor send_data_offset = at::empty({round, num_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_data = at::empty({round, num_experts * SEND_PER_GROUP},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_count = at::empty({round, num_experts},     //维度？
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_offset = at::empty({round, num_experts},    //维度？
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor expert_global_offset = at::empty({num_local_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor srcrank_in_expert_offset = at::empty({moe_expert_num},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor r_in_srcrank_offset = at::empty({moe_expert_num * round},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor total_recv_tokens = at::empty({1},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor max_bs = at::empty({1},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_tokens_per_expert = at::empty({round * num_local_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));

    // ACLNN_CMD(aclnnNotifyDispatch, send_data, token_per_expert_data, send_count, num_tokens,
    //           group_ep_ptr, rank_size, rank_id, local_rank_size, local_rank_id,
    //           round, per_round_tokens, send_data_offset, recv_data, recv_count, recv_offset,
    //           expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset,
    //           total_recv_tokens, max_bs, recv_tokens_per_expert);
    ACLNN_CMD(aclnnNotifyDispatch, context, send_data, token_per_expert_data, send_count, num_tokens,
              ep_world_size, ep_rank_id, local_ranksize, local_rank_id, ccl_buffer_size
              round, per_round_tokens, send_data_offset, recv_data, recv_count, recv_offset,
              expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset,
              total_recv_tokens, max_bs, recv_tokens_per_expert);
    
    std::vector<at::Tensor> recv_offset, expert_output_offset;
    // c10::List<at::Tensor> recv_offset, expert_output_offset;
    recv_offset.push_back(recv_offset);
    recv_offset.push_back(recv_count);
    expert_output_offset.push_back(expert_global_offset);
    expert_output_offset.push_back(srcrank_in_expert_offset);
    expert_output_offset.push_back(r_in_srcrank_offset);
    int64_t real_max_bs = static_cast<int64_t>(std::max(max_bs.item<int>(), static_cast<int>(num_worst_tokens)));
    // dispatch算子内部按照 min(per_round_tokens, real_max_bs)来预留显存
    int64_t global_bs = static_cast<int64_t>(std::min(static_cast<int64_t>(per_round_tokens), real_max_bs) * ep_world_size);
    int64_t num_recv_tokens = (total_recv_token.item<int>(); == 0) ? 1 : total_recv_token.item<int>();
    at::TensorList recv_offset_ref = recv_offset;
    at::TensorList expert_output_offset_ref = expert_output_offset;

    auto output_dtype = at::kChar;
    if (quant_mode == QUANT_MODE_NO_QUANT) {
        output_dtype = x.scalar_type();
    }

    at::Tensor recv_x = at::empty({num_recv_tokens, h}, 
        at::TensorOptions().dtype(at::output_dtype)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor x_scales = at::empty({num_recv_tokens},
        at::TensorOptions()..dtype(at::kFloat)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor assist_info_for_combine = at::empty({num_recv_tokens * EXPAND_IDX_INFO},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor ext_handle_data_out = at::empty({0},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));

    ACLNN_CMD(aclnnDispatchNormal, context, x, topk_idx, send_data_offset, send_token_idx_small, recv_offset_ref,
              expert_output_offset_ref, ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size, quant_mode,
              real_max_bs, global_bs, round, per_round_tokens, recv_x, x_scales, assist_info_for_combine,
              dispatch_wait_recv_cost_stats_out, ext_handle_data_out);

    return std::tie(recv_x, x_scales, assist_info_for_combine, recv_count, ext_handle_data_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_dispatch_normal", &npu_dispatch_normal, "dispatch_normal");
}
} // namespace op_api
