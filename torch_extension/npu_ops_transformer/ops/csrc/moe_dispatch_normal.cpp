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
 * \file moe_dispatch_normal.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

tensor_list npu_moe_dispatch_normal(const at::Tensor &x, const at::Tensor &topk_idx, const at::Tensor &send_offset,
                                    const at::Tensor &send_token_idx, const at::Tensor &recv_offset,
                                    const at::Tensor &recv_count, std::string group_ep, int64_t ep_world_size,
                                    int64_t ep_rank_id, std::string group_tp, int64_t tp_world_size, int64_t tp_rank_id,
                                    int64_t moe_expert_num, int64_t quant_mode, int64_t global_bs)
{
    TORCH_CHECK((x.dim() == DIM_TWO) && (topk_idx.dim() == DIM_TWO), "The x and topk_idx should be 2D");
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", ep_rank_id: ", ep_rank_id,
                ". ");

    auto x_size = x.sizes();
    auto topk_idx_size = topk_idx.sizes();

    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = topk_idx_size[1];

    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t local_moe_expert_num = moe_expert_num / ep_world_size;
    int64_t a = global_bs_real * std::min(local_moe_expert_num, k);

    int64_t real_a = (tp_world_size == 0) ? a : (a * tp_world_size);

    auto output_dtype = at::kChar;
    if (quant_mode == QuantMode::QUANT_MODE_NO_QUANT) {
        output_dtype = x.scalar_type();
    }

    char *group_ep_ptr = const_cast<char *>(group_ep.data());
    char *group_tp_ptr = const_cast<char *>(group_tp.c_str());

    at::Tensor recv_x{nullptr};
    at::Tensor recv_x_scales{nullptr};
    at::Tensor assist_info_for_combine{nullptr};
    at::Tensor wait_recv_cost_stats{nullptr};

    recv_x = at::empty({real_a, h}, at::TensorOptions().dtype(output_dtype)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous)); 
    recv_x_scales = at::empty({real_a}, at::TensorOptions().dtype(at::kFloat)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    assist_info_for_combine = at::empty({std::max(bs * k, a * 128)}, at::TensorOptions().dtype(at::kInt)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    wait_recv_cost_stats = at::empty({ep_world_size}, at::TensorOptions().dtype(at::kInt)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));

    ACLNN_CMD(aclnnMoeDispatchNormal, x, topk_idx, send_offset, send_token_idx, recv_offset, recv_count, group_ep_ptr,
              ep_world_size, ep_rank_id, group_tp_ptr, tp_world_size, tp_rank_id, moe_expert_num, quant_mode,
              global_bs_real, recv_x, recv_x_scales, assist_info_for_combine, wait_recv_cost_stats);

    return std::tie(recv_x, recv_x_scales, assist_info_for_combine, wait_recv_cost_stats);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_dispatch_normal", &npu_moe_dispatch_normal, "moe_dispatch_normal");
}
} // namespace op_api