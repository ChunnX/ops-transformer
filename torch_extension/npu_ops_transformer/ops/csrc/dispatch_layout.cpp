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
#include <optional>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using layout_result =
    std::tuple<at::Tensor, std::optional<at::Tensor>, at::Tensor, at::Tensor, std::vector<at::Tensor>>;

const int DIM_TWO = 2;
const int64_t MAX_TOTAL_TOKENS = 131072;
const int MAX_BATCH_SIZE = 4096;
const int EXPERT_DATA_SIZE = 1 + MAX_BATCH_SIZE;  // 4097

layout_result npu_dispatch_layout(const at::Tensor &topk_idx, int64_t num_tokens, int64_t num_topk,
                                  int64_t num_experts, int64_t num_ranks, int64_t local_ranksize,
                                  int64_t per_round_tokens, int64_t rank_id)
{
    TORCH_CHECK(topk_idx.dim() == DIM_TWO, "The topk_idx should be 2D");
    TORCH_CHECK(topk_idx.scalar_type() == at::kLong, "dtype of topk_idx should be int64");

    auto topk_idx_size = topk_idx.sizes();
    TORCH_CHECK(topk_idx_size[0] == num_tokens,
                "topk_idx.size(0) must equal num_tokens, got topk_idx dim0: ",
                topk_idx_size[0], ", num_tokens: ", num_tokens);
    TORCH_CHECK(topk_idx_size[1] == num_topk,
                "topk_idx.size(1) must equal num_topk, got topk_idx dim1: ",
                topk_idx_size[1], ", num_topk: ", num_topk);
    TORCH_CHECK(num_tokens > 0, "num_tokens should be greater than 0, current is: ", num_tokens);
    TORCH_CHECK(num_topk > 0, "num_topk should be greater than 0, current is: ", num_topk);
    TORCH_CHECK(num_ranks > 0, "num_ranks should be greater than 0, current is: ", num_ranks);
    TORCH_CHECK(num_experts > 0, "num_experts should be greater than 0, current is: ", num_experts);
    TORCH_CHECK(per_round_tokens > 0, "per_round_tokens should be greater than 0, current is: ", per_round_tokens);
    TORCH_CHECK((rank_id >= 0) && (rank_id < num_ranks),
                "rank_id should be in [0, num_ranks), current num_ranks: ",
                num_ranks, ", rank_id: ", rank_id);
    TORCH_CHECK((num_experts % num_ranks) == 0,
                "num_experts should be divisible by num_ranks, current num_experts: ",
                num_experts, ", num_ranks: ", num_ranks);

    TORCH_CHECK(local_ranksize > 0 && local_ranksize <= num_ranks,
                "local_ranksize must be in (0, num_ranks], got local_ranksize=", local_ranksize,
                ", num_ranks=", num_ranks);
    TORCH_CHECK((num_ranks % local_ranksize) == 0,
                "num_ranks must be divisible by local_ranksize for dispatch_layout, got num_ranks=",
                num_ranks, ", local_ranksize=", local_ranksize);

    int64_t round = (num_tokens + per_round_tokens - 1) / per_round_tokens;
    TORCH_CHECK(num_tokens <= round * per_round_tokens,
                "num_tokens must be <= round * per_round_tokens, got num_tokens=", num_tokens,
                ", round=", round, ", per_round_tokens=", per_round_tokens);
    TORCH_CHECK(round * per_round_tokens <= MAX_TOTAL_TOKENS,
                "round * per_round_tokens must be <= ", MAX_TOTAL_TOKENS, ", got ", round * per_round_tokens);
    auto server_num = num_ranks / local_ranksize;
    const int notify_send_data_size =
        num_experts * EXPERT_DATA_SIZE + server_num + MAX_BATCH_SIZE * (1 + 2 * server_num + 2 * num_topk);

    auto options = at::TensorOptions()
                       .dtype(at::kInt)
                       .device(c10::DeviceType::PrivateUse1)
                       .memory_format(c10::MemoryFormat::Contiguous);

    at::Tensor num_tokens_per_rank = at::zeros({num_ranks}, options);
    at::Tensor num_tokens_per_expert = at::zeros({round * num_experts}, options);
    at::Tensor is_token_in_rank = at::zeros({round * per_round_tokens, num_ranks}, options);
    at::Tensor notify_send_data = at::zeros({notify_send_data_size}, options);
    at::Tensor send_token_idx_small = at::zeros({num_tokens, num_topk}, options);

    int32_t per_round_tokens_i32 = static_cast<int32_t>(per_round_tokens);
    int32_t rank_id_i32 = static_cast<int32_t>(rank_id);
    at::TensorList ext_handle_data_ref = {notify_send_data, send_token_idx_small};
    ACLNN_CMD(aclnnDispatchLayout, topk_idx, num_tokens, num_ranks, num_experts, num_topk,
              local_ranksize, per_round_tokens_i32, rank_id_i32,
              num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, ext_handle_data_ref);

    std::optional<at::Tensor> num_tokens_per_rdma_rank = std::nullopt;
    std::vector<at::Tensor> ext_handle_data;
    ext_handle_data.push_back(notify_send_data);
    ext_handle_data.push_back(send_token_idx_small);

    return std::make_tuple(num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank,
                           ext_handle_data);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_dispatch_layout", &npu_dispatch_layout, "dispatch_layout");
}
}  // namespace op_api
