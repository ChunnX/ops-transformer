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
 * \file dispatch_normal.cpp
 * \brief ACLNN wrapper for dispatch_normal (MoE EP dispatch with Mc2 context).
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using tensor_list4 = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

constexpr int64_t DIM_TWO = 2;
constexpr int64_t QUANT_MODE_NO_QUANT = 0;
constexpr int64_t QUANT_MODE_DYNAMIC_SCALES = 2;

tensor_list4 npu_dispatch_normal(const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_idx,
                                 const at::Tensor &send_offset, const at::Tensor &send_token_idx,
                                 const std::vector<at::Tensor> &recv_offset,
                                 const std::vector<at::Tensor> &expert_output_offset, int64_t ep_world_size,
                                 int64_t ep_rank_id, int64_t moe_expert_num, int64_t ccl_buffer_size,
                                 int64_t num_recv_tokens, int64_t quant_mode, int64_t real_max_bs, int64_t global_bs,
                                 int64_t round, int64_t per_round_tokens,
                                 const c10::optional<at::Tensor> &dispatch_wait_recv_cost_stats,
                                 const c10::optional<at::Tensor> &ext_handle_data)
{
    TORCH_CHECK(x.dim() == DIM_TWO && topk_idx.dim() == DIM_TWO, "x and topk_idx must be 2D tensors.");
    TORCH_CHECK(recv_offset.size() == 2, "recv_offset must contain 2 tensors [recv_offset, recv_count].");
    TORCH_CHECK(expert_output_offset.size() == 3,
                "expert_output_offset must contain 3 tensors "
                "[expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset].");
    const at::Tensor &recv_offset_tensor = recv_offset[0];
    const at::Tensor &recv_count_tensor = recv_offset[1];
    const at::Tensor &expert_global_offset_tensor = expert_output_offset[0];
    const at::Tensor &srcrank_in_expert_offset_tensor = expert_output_offset[1];
    const at::Tensor &r_in_srcrank_offset_tensor = expert_output_offset[2];
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got ep_world_size=", ep_world_size,
                ", ep_rank_id=", ep_rank_id, ".");
    TORCH_CHECK((quant_mode == QUANT_MODE_NO_QUANT) || (quant_mode == QUANT_MODE_DYNAMIC_SCALES),
                "quant_mode should be 0 (no quant) or 2 (dynamic int8), but got ", quant_mode, ".");
    TORCH_CHECK((recv_offset_tensor.dim() == 1) && (recv_count_tensor.dim() == 1) &&
                    (expert_global_offset_tensor.dim() == 1) && (srcrank_in_expert_offset_tensor.dim() == 1) &&
                    (r_in_srcrank_offset_tensor.dim() == 1),
                "all tensors in recv_offset and expert_output_offset must be 1D tensors.");
    TORCH_CHECK((moe_expert_num % ep_world_size) == 0,
                "moe_expert_num must be divisible by ep_world_size, but got moe_expert_num=", moe_expert_num,
                ", ep_world_size=", ep_world_size, ".");
    const int64_t recv_offset_expected_numel = 2 * round * moe_expert_num;
    const int64_t expert_output_expected_numel =
        moe_expert_num / ep_world_size + moe_expert_num + round * moe_expert_num;
    const int64_t recv_offset_numel = recv_offset_tensor.numel() + recv_count_tensor.numel();
    TORCH_CHECK(recv_offset_numel == recv_offset_expected_numel, "recv_offset numel mismatch, expected ",
                recv_offset_expected_numel, " (=2*round*moe_expert_num), but got ", recv_offset_numel, ".");
    const int64_t expert_output_numel = expert_global_offset_tensor.numel() + srcrank_in_expert_offset_tensor.numel() +
                                        r_in_srcrank_offset_tensor.numel();
    TORCH_CHECK(expert_output_numel == expert_output_expected_numel, "expert_output_offset numel mismatch, expected ",
                expert_output_expected_numel,
                " (=moe_expert_num/ep_world_size + moe_expert_num + round*moe_expert_num), but got ",
                expert_output_numel, ".");
    TORCH_CHECK(num_recv_tokens > 0, "num_recv_tokens must be positive, but got ", num_recv_tokens, ".");
    TORCH_CHECK(global_bs > 0, "global_bs must be positive, but got ", global_bs, ".");
    TORCH_CHECK(ccl_buffer_size > 0, "ccl_buffer_size must be positive, but got ", ccl_buffer_size, ".");

    const int64_t h = x.size(1);
    auto output_dtype = at::kChar;
    if (quant_mode == QUANT_MODE_NO_QUANT) {
        output_dtype = x.scalar_type();
    }

    at::Tensor recv_x = at::empty({num_recv_tokens, h}, at::TensorOptions()
                                                            .dtype(output_dtype)
                                                            .device(c10::DeviceType::PrivateUse1)
                                                            .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor x_scales = at::empty({num_recv_tokens}, at::TensorOptions()
                                                           .dtype(at::kFloat)
                                                           .device(c10::DeviceType::PrivateUse1)
                                                           .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor assist_info_for_combine =
        at::empty({num_recv_tokens * 3}, at::TensorOptions()
                                             .dtype(at::kInt)
                                             .device(c10::DeviceType::PrivateUse1)
                                             .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor dispatch_wait_recv_cost_stats_out;
    if (dispatch_wait_recv_cost_stats.has_value()) {
        const at::Tensor &dispatch_wait_recv_cost_stats_tensor = dispatch_wait_recv_cost_stats.value();
        TORCH_CHECK(dispatch_wait_recv_cost_stats_tensor.scalar_type() == at::kInt,
                    "dispatch_wait_recv_cost_stats dtype must be int32.");
        TORCH_CHECK(dispatch_wait_recv_cost_stats_tensor.dim() == 1 &&
                        dispatch_wait_recv_cost_stats_tensor.is_contiguous(),
                    "dispatch_wait_recv_cost_stats must be contiguous 1D tensor.");
        TORCH_CHECK(dispatch_wait_recv_cost_stats_tensor.size(0) == ep_world_size,
                    "dispatch_wait_recv_cost_stats shape[0] must equal ep_world_size, but got ",
                    dispatch_wait_recv_cost_stats_tensor.size(0), " and ", ep_world_size, ".");
        dispatch_wait_recv_cost_stats_out = dispatch_wait_recv_cost_stats_tensor;
    }
    at::Tensor ext_handle_data_out = at::empty({0}, at::TensorOptions()
                                                        .dtype(at::kInt)
                                                        .device(c10::DeviceType::PrivateUse1)
                                                        .memory_format(c10::MemoryFormat::Contiguous));
    if (ext_handle_data.has_value()) {
        const at::Tensor &ext_handle_data_tensor = ext_handle_data.value();
        TORCH_CHECK(ext_handle_data_tensor.scalar_type() == at::kInt, "ext_handle_data dtype must be int32.");
        TORCH_CHECK(ext_handle_data_tensor.dim() == 1 && ext_handle_data_tensor.is_contiguous(),
                    "ext_handle_data must be contiguous 1D tensor.");
        TORCH_CHECK(ext_handle_data_tensor.size(0) == 0, "ext_handle_data shape[0] must be 0.");
        ext_handle_data_out = ext_handle_data_tensor;
    }
    at::TensorList recv_offset_ref = recv_offset;
    at::TensorList expert_output_offset_ref = expert_output_offset;

    ACLNN_CMD(aclnnDispatchNormal, context, x, topk_idx, send_offset, send_token_idx, recv_offset_ref,
              expert_output_offset_ref, ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size, quant_mode,
              real_max_bs, global_bs, round, per_round_tokens, recv_x, x_scales, assist_info_for_combine,
              dispatch_wait_recv_cost_stats_out, ext_handle_data_out);

    return std::make_tuple(recv_x, x_scales, assist_info_for_combine, ext_handle_data_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_dispatch_normal", &npu_dispatch_normal, "dispatch_normal");
}
} // namespace op_api
