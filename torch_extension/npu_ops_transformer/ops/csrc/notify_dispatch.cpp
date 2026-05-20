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
 * \file notify_dispatch.cpp
 * \brief Torch extension for notify_dispatch operator
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using tensor_list = std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>, at::Tensor, at::Tensor,
                               at::Tensor, at::Tensor>;
const int DIM_ONE = 1;
const int SEND_PER_GROUP = 3;

tensor_list npu_notify_dispatch(const at::Tensor &context, const at::Tensor &send_data,
                                const at::Tensor &token_per_expert_data, int64_t send_count, int64_t num_tokens,
                                int64_t ep_world_size, int64_t ep_rank_id, int64_t local_rank_size,
                                int64_t local_rank_id, int64_t ccl_buffer_size, int64_t round,
                                int64_t per_round_tokens)
{
    TORCH_CHECK(send_data.dim() == DIM_ONE, "The send_data should be 1D");
    TORCH_CHECK(token_per_expert_data.dim() == DIM_ONE, "The token_per_expert_data should be 1D");
    TORCH_CHECK(send_data.scalar_type() == at::kInt, "dtype of send_data should be int32");
    TORCH_CHECK(token_per_expert_data.scalar_type() == at::kInt, "dtype of token_per_expert_data should be int32");
    TORCH_CHECK(context.scalar_type() == at::kInt, "dtype of context should be int32");
    TORCH_CHECK(context.dim() == DIM_ONE, "The context should be 1D");

    int64_t send_data_len = send_data.size(0);
    TORCH_CHECK(send_data_len == send_count,
                "send_data size should equal send_count, but got len: ", send_data_len,
                ", send_count: ", send_count);
    TORCH_CHECK(send_count > 0, "send_count should be greater than 0, current is: ", send_count);
    TORCH_CHECK(ep_world_size > 0, "ep_world_size should be greater than 0, current is: ", ep_world_size);
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), current ep_world_size: ",
                ep_world_size, ", ep_rank_id: ", ep_rank_id);
    TORCH_CHECK((local_rank_id >= 0) && (local_rank_id < local_rank_size),
                "local_rank_id should be in [0, local_rank_size), current local_rank_size: ",
                local_rank_size, ", local_rank_id: ", local_rank_id);
    TORCH_CHECK(ccl_buffer_size > 0, "ccl_buffer_size must be positive, but got ", ccl_buffer_size, ".");
    TORCH_CHECK(round > 0, "round should be greater than 0, current is: ", round);

    int64_t num_experts = send_count / SEND_PER_GROUP / round;
    int64_t num_local_experts = num_experts / ep_world_size;
    TORCH_CHECK(num_experts > 0, "num_experts should be greater than 0, current is: ", num_experts);
    TORCH_CHECK((num_experts % ep_world_size) == 0,
                "num_experts should be divisible by ep_world_size, current num_experts: ",
                num_experts, ", ep_world_size: ", ep_world_size);
    TORCH_CHECK(token_per_expert_data.size(0) == round * num_experts,
                "token_per_expert_data size(0) must equal round * num_experts, but got ",
                token_per_expert_data.size(0), " and ", round * num_experts, ".");
    TORCH_CHECK(num_tokens > 0, "num_tokens should be greater than 0, current is: ", num_tokens);

    at::Tensor send_offset = at::empty({round * num_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_data = at::empty({send_count},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_count = at::empty({round * num_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor recv_offset = at::empty({round * num_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor expert_global_offset = at::empty({num_local_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor srcrank_in_expert_offset = at::empty({num_local_experts * ep_world_size},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor r_in_srcrank_offset = at::empty({num_local_experts * ep_world_size * round},
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

    ACLNN_CMD(aclnnNotifyDispatch, context, send_data, token_per_expert_data, send_count, num_tokens,
              ep_world_size, ep_rank_id, local_rank_size, local_rank_id, ccl_buffer_size,
              round, per_round_tokens, send_offset, recv_data, recv_count, recv_offset,
              expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset,
              total_recv_tokens, max_bs, recv_tokens_per_expert);

    std::vector<at::Tensor> recv_offset_list;
    recv_offset_list.push_back(recv_offset);
    recv_offset_list.push_back(recv_count);
    std::vector<at::Tensor> expert_output_offset_list;
    expert_output_offset_list.push_back(expert_global_offset);
    expert_output_offset_list.push_back(srcrank_in_expert_offset);
    expert_output_offset_list.push_back(r_in_srcrank_offset);

    return std::make_tuple(send_offset, recv_offset_list, expert_output_offset_list, total_recv_tokens, max_bs,
                           recv_data, recv_tokens_per_expert);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_notify_dispatch", &npu_notify_dispatch, "notify_dispatch");
}
}  // namespace op_api
