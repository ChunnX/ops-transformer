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
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                               at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_ONE = 1;
const int DIM_TWO = 2;
const int SEND_PER_GROUP = 3;

tensor_list npu_notify_dispatch(const at::Tensor &send_data, const at::Tensor &token_per_expert_data,
                                int64_t send_count, int64_t num_tokens,
                                std::string group_ep, int64_t rank_size, int64_t rank_id,
                                int64_t local_rank_size, int64_t local_rank_id,
                                int64_t round, int64_t per_round_tokens)
{
    TORCH_CHECK(send_data.dim() == DIM_ONE, "The send_data should be 1D");
    TORCH_CHECK(token_per_expert_data.dim() == DIM_ONE, "The token_per_expert_data should be 1D");
    TORCH_CHECK(send_data.scalar_type() == at::kInt, "dtype of send_data should be int32");
    TORCH_CHECK(token_per_expert_data.scalar_type() == at::kInt, "dtype of token_per_expert_data should be int32");

    auto send_data_size = send_data.sizes();
    int64_t send_data_len = send_data_size[0];
    TORCH_CHECK(send_data_len == send_count,
                "send_data size should equal send_count, but got len: ", send_data_len,
                ", send_count: ", send_count);
    TORCH_CHECK(send_count > 0, "send_count should be greater than 0, current is: ", send_count);
    TORCH_CHECK(rank_size > 0, "rank_size should be greater than 0, current is: ", rank_size);
    TORCH_CHECK((rank_id >= 0) && (rank_id < rank_size),
                "rank_id should be in [0, rank_size), current rank_size: ",
                rank_size, ", rank_id: ", rank_id);
    TORCH_CHECK((local_rank_id >= 0) && (local_rank_id < local_rank_size),
                "local_rank_id should be in [0, local_rank_size), current local_rank_size: ",
                local_rank_size, ", local_rank_id: ", local_rank_id);
    TORCH_CHECK(round > 0, "round should be greater than 0, current is: ", round);

    int64_t num_experts = send_count / SEND_PER_GROUP / round;
    int64_t num_local_experts = num_experts / rank_size;
    TORCH_CHECK(num_experts > 0, "num_experts should be greater than 0, current is: ", num_experts);
    TORCH_CHECK((num_experts % rank_size) == 0,
                "num_experts should be divisible by rank_size, current num_experts: ",
                num_experts, ", rank_size: ", rank_size);

    char *group_ep_ptr = const_cast<char *>(group_ep.data());

    at::Tensor send_data_offset = at::empty({round * num_experts},
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
    at::Tensor srcrank_in_expert_offset = at::empty({num_local_experts * rank_size},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor r_in_srcrank_offset = at::empty({num_local_experts * rank_size * round},
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

    ACLNN_CMD(aclnnNotifyDispatch, send_data, token_per_expert_data, send_count, num_tokens,
              group_ep_ptr, rank_size, rank_id, local_rank_size, local_rank_id,
              round, per_round_tokens, send_data_offset, recv_data, recv_count, recv_offset,
              expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset,
              total_recv_tokens, max_bs, recv_tokens_per_expert);

    return std::tie(send_data_offset, recv_data, recv_count, recv_offset, expert_global_offset,
                    srcrank_in_expert_offset, r_in_srcrank_offset, total_recv_tokens, max_bs,
                    recv_tokens_per_expert);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_notify_dispatch", &npu_notify_dispatch, "notify_dispatch");
}
}  // namespace op_api