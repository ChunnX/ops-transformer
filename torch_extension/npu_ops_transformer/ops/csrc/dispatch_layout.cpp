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
 * \file dispatch_layout.cpp
 * \brief Torch extension for dispatch_layout operator
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

tensor_list npu_dispatch_layout(const at::Tensor &topk_idx,
                                int64_t num_tokens, int64_t num_ranks, int64_t num_experts,
                                int64_t num_topk, int64_t local_ranksize,
                                int64_t per_round_tokens, int64_t rank_id)
{
    TORCH_CHECK(topk_idx.dim() == DIM_TWO, "The topk_idx should be 2D");
    TORCH_CHECK(topk_idx.scalar_type() == at::kLong, "dtype of topk_idx should be int64");

    auto topk_idx_size = topk_idx.sizes();
    int64_t bs = topk_idx_size[0];
    int64_t k = topk_idx_size[1];
    TORCH_CHECK(bs == num_tokens,
                "topk_idx size(0) should equal num_tokens, but got bs: ", bs,
                ", num_tokens: ", num_tokens);
    TORCH_CHECK(k == num_topk,
                "topk_idx size(1) should equal num_topk, but got k: ", k,
                ", num_topk: ", num_topk);

    TORCH_CHECK(num_tokens > 0, "num_tokens should be greater than 0, current is: ", num_tokens);
    TORCH_CHECK(num_ranks > 0, "num_ranks should be greater than 0, current is: ", num_ranks);
    TORCH_CHECK(num_experts > 0, "num_experts should be greater than 0, current is: ", num_experts);
    TORCH_CHECK(num_topk > 0, "num_topk should be greater than 0, current is: ", num_topk);
    TORCH_CHECK(per_round_tokens > 0, "per_round_tokens should be greater than 0, current is: ", per_round_tokens);
    TORCH_CHECK((rank_id >= 0) && (rank_id < num_ranks),
                "rank_id should be in [0, num_ranks), current num_ranks: ",
                num_ranks, ", rank_id: ", rank_id);
    TORCH_CHECK((num_experts % num_ranks) == 0,
                "num_experts should be divisible by num_ranks, current num_experts: ",
                num_experts, ", num_ranks: ", num_ranks);

    int64_t round = (num_tokens + per_round_tokens - 1) / per_round_tokens;
    int64_t num_local_experts = num_experts / num_ranks;

    at::Tensor num_tokens_per_rank = at::empty({num_ranks},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor num_tokens_per_expert = at::empty({round * num_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor is_token_in_rank = at::empty({round * per_round_tokens, num_ranks},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor notify_send_data = at::empty({round * num_local_experts},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor send_token_idx_small = at::empty({num_tokens * num_topk},
        at::TensorOptions().dtype(at::kInt)
            .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));

    ACLNN_CMD(aclnnDispatchLayout, topk_idx, num_tokens, num_ranks, num_experts, num_topk,
              local_ranksize, per_round_tokens, rank_id, num_tokens_per_rank, num_tokens_per_expert,
              is_token_in_rank, notify_send_data, send_token_idx_small);

    return std::tie(num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, notify_send_data,
                    send_token_idx_small);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_dispatch_layout", &npu_dispatch_layout, "dispatch_layout");
}
}  // namespace op_api