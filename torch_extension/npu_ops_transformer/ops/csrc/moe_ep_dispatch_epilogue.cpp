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
 * \file moe_ep_dispatch_epilogue.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_moe_ep_dispatch_epilogue(const at::Tensor &context, const at::Tensor &dst_buffer_slot_idx,
    const at::Tensor &recv_src_metadata_internal, const at::Tensor &num_recv_per_rank_hp,
    const at::Tensor &num_recv_per_expert_hp, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t num_experts, int64_t num_max_tokens_per_rank, int64_t ccl_buffer_size, int64_t expert_alignment,
    at::Tensor &recv_x, at::Tensor &recv_topk_idx, at::Tensor &recv_topk_weights, at::Tensor &recv_src_metadata,
    at::Tensor &num_recv_tokens_per_rank, at::Tensor &num_recv_tokens_per_expert)
{
    TORCH_CHECK(dst_buffer_slot_idx.dim() == DIM_TWO, "dst_buffer_slot_idx must be 2D");

    ACLNN_CMD(aclnnMoeEpDispatchEpilogue, context, dst_buffer_slot_idx, recv_src_metadata_internal,
        num_recv_per_rank_hp, num_recv_per_expert_hp, ep_world_size, ep_rank_id, num_experts,
        num_max_tokens_per_rank, ccl_buffer_size, expert_alignment, recv_x, recv_topk_idx,
        recv_topk_weights, recv_src_metadata, num_recv_tokens_per_rank, num_recv_tokens_per_expert);

    return std::tie(recv_x, recv_topk_idx, recv_topk_weights, recv_src_metadata,
                    num_recv_tokens_per_rank, num_recv_tokens_per_expert);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_ep_dispatch_epilogue", &npu_moe_ep_dispatch_epilogue, "moe_ep_dispatch_epilogue");
}

}  // namespace op_api