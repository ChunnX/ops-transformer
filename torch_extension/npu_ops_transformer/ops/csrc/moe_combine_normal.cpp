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
 * \file moe_combine_normal.cpp
 * \brief ACLNN wrapper for moe_combine_normal (MoE EP combine).
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DIM_TWO = 2;

at::Tensor npu_moe_combine_normal(const at::Tensor &recv_x, const at::Tensor &token_src_info,
                                  const at::Tensor &ep_recv_counts, const at::Tensor &recv_topk_weights,
                                  const at::Tensor &token_idx, const c10::optional<at::Tensor> &tp_recv_counts,
                                  const std::string &ep_group_name, int64_t ep_world_size, int64_t ep_rank_id,
                                  int64_t moe_expert_num, const std::string &tp_group_name, int64_t tp_world_size,
                                  int64_t tp_rank_id, int64_t real_max_bs, int32_t round, int32_t per_round_tokens,
                                  const c10::optional<at::Tensor> &combine_send_cost_stats)
{
    TORCH_CHECK(recv_x.dim() == DIM_TWO, "recv_x must be 2D tensor (num_recv_tokens, H).");
    TORCH_CHECK(token_src_info.dim() == 1, "token_src_info must be 1D tensor.");
    TORCH_CHECK(ep_recv_counts.dim() == 1, "ep_recv_counts must be 1D tensor.");
    TORCH_CHECK(recv_topk_weights.dim() == DIM_TWO, "recv_topk_weights must be 2D tensor (BS, K).");
    TORCH_CHECK(token_idx.dim() == DIM_TWO, "token_idx must be 2D tensor.");
    TORCH_CHECK(token_idx.scalar_type() == at::kInt, "token_idx dtype must be int32.");
    TORCH_CHECK(round > 0, "round must be greater than 0, but got round=", round, ".");
    TORCH_CHECK(per_round_tokens > 0,
                "per_round_tokens must be greater than 0, but got per_round_tokens=", per_round_tokens, ".");
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got ep_world_size=", ep_world_size,
                ", ep_rank_id=", ep_rank_id, ".");
    TORCH_CHECK((moe_expert_num % ep_world_size) == 0,
                "moe_expert_num must be divisible by ep_world_size, but got moe_expert_num=", moe_expert_num,
                ", ep_world_size=", ep_world_size, ".");
    TORCH_CHECK(moe_expert_num > 0 && moe_expert_num <= 512,
                "moe_expert_num must be in range (0, 512], but got moe_expert_num=", moe_expert_num, ".");
    TORCH_CHECK(ep_world_size >= 2 && ep_world_size <= 384,
                "ep_world_size must be in range [2, 384], but got ep_world_size=", ep_world_size, ".");

    const int64_t num_recv_tokens = recv_x.size(0);
    const int64_t h = recv_x.size(1);
    const int64_t bs = recv_topk_weights.size(0);
    const int64_t k = recv_topk_weights.size(1);

    TORCH_CHECK(token_src_info.numel() == num_recv_tokens * 3,
                "token_src_info numel must be num_recv_tokens * 3, expected ", num_recv_tokens * 3, ", but got ",
                token_src_info.numel(), ".");
    TORCH_CHECK(ep_recv_counts.numel() == moe_expert_num, "ep_recv_counts numel must be moe_expert_num, expected ",
                moe_expert_num, ", but got ", ep_recv_counts.numel(), ".");
    TORCH_CHECK(k > 0 && k <= std::min(16L, moe_expert_num),
                "recv_topk_weights K must be in range (0, min(16, moe_expert_num)], but got k=", k,
                ", moe_expert_num=", moe_expert_num, ".");
    TORCH_CHECK(bs > 0 && bs <= 8000, "recv_topk_weights BS must be in range [1, 8000], but got bs=", bs, ".");
    TORCH_CHECK(h >= 1024 && h <= 7168, "recv_x H must be in range [1024, 7168], but got h=", h, ".");

    at::Tensor tp_recv_counts_tensor;
    if (tp_recv_counts.has_value()) {
        const at::Tensor &tp_recv_counts_val = tp_recv_counts.value();
        TORCH_CHECK(tp_recv_counts_val.scalar_type() == at::kInt, "tp_recv_counts dtype must be int32.");
        TORCH_CHECK(tp_recv_counts_val.dim() == 1, "tp_recv_counts must be 1D tensor.");
        tp_recv_counts_tensor = tp_recv_counts_val;
    }

    at::Tensor combine_send_cost_stats_tensor;
    if (combine_send_cost_stats.has_value()) {
        const at::Tensor &combine_send_cost_stats_val = combine_send_cost_stats.value();
        TORCH_CHECK(combine_send_cost_stats_val.scalar_type() == at::kInt,
                    "combine_send_cost_stats dtype must be int32.");
        TORCH_CHECK(combine_send_cost_stats_val.dim() == 1 && combine_send_cost_stats_val.is_contiguous(),
                    "combine_send_cost_stats must be contiguous 1D tensor.");
        TORCH_CHECK(combine_send_cost_stats_val.size(0) == ep_world_size,
                    "combine_send_cost_stats shape[0] must equal ep_world_size, but got ",
                    combine_send_cost_stats_val.size(0), " and ", ep_world_size, ".");
        combine_send_cost_stats_tensor = combine_send_cost_stats_val;
    }

    at::Tensor x = at::empty({bs, h}, at::TensorOptions()
                                          .dtype(recv_x.scalar_type())
                                          .device(c10::DeviceType::PrivateUse1)
                                          .memory_format(c10::MemoryFormat::Contiguous));

    const char *ep_group_name_ptr = ep_group_name.c_str();
    const char *tp_group_name_ptr = tp_group_name.c_str();

    ACLNN_CMD(aclnnMoeCombineNormal, recv_x, token_src_info, ep_recv_counts, recv_topk_weights, token_idx,
            tp_recv_counts_tensor, ep_group_name_ptr, ep_world_size, ep_rank_id, tp_group_name_ptr, tp_world_size,
            tp_rank_id, moe_expert_num, real_max_bs, round, per_round_tokens, x, combine_send_cost_stats_tensor);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_combine_normal", &npu_moe_combine_normal, "moe_combine_normal");
}

} // namespace op_api