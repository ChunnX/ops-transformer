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
using tensor_list = std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>>;
const int DIM_TWO = 2;

tensor_list npu_moe_ep_dispatch_epilogue(const at::Tensor &context, const at::Tensor &dst_buffer_slot_idx,
    const at::Tensor &num_recv_per_rank, const at::Tensor &num_recv_per_expert,
    const c10::optional<at::Tensor> &cached_recv_src_metadata,
    int64_t ep_world_size, int64_t ep_rank_id,
    int64_t num_experts, int64_t num_max_tokens_per_rank, int64_t ccl_buffer_size, int64_t expert_alignment,
    at::Tensor &recv_x, at::Tensor &recv_src_metadata,
    const c10::optional<at::Tensor> &recv_topk_weights_opt, const c10::optional<at::Tensor> &recv_scales_opt)
{
    TORCH_CHECK(dst_buffer_slot_idx.dim() == DIM_TWO, "dst_buffer_slot_idx must be 2D");

    at::Tensor cached_recv_src_metadata_tensor = cached_recv_src_metadata.has_value()
        ? *cached_recv_src_metadata : at::Tensor();

    aclDataType recv_scales_dtype = aclDataType::ACL_FLOAT;
    at::Tensor recv_scales_tensor = recv_scales_opt.has_value() ? *recv_scales_opt : at::Tensor();
    if (recv_scales_opt.has_value() && recv_scales_tensor.scalar_type() == at::kByte) {
        recv_scales_dtype = aclDataType::ACL_FLOAT8_E8M0;
    }
    TensorWrapper recv_scales_wrapper = TensorWrapper{recv_scales_tensor, recv_scales_dtype};

    at::Tensor recv_topk_weights_tensor = recv_topk_weights_opt.has_value()
        ? *recv_topk_weights_opt : at::Tensor();

    ACLNN_CMD(aclnnMoeEpDispatchEpilogue, context, dst_buffer_slot_idx,
        num_recv_per_rank, num_recv_per_expert, cached_recv_src_metadata_tensor,
        ep_world_size, ep_rank_id, num_experts,
        num_max_tokens_per_rank, ccl_buffer_size, expert_alignment, recv_x,
        recv_src_metadata, recv_topk_weights_tensor, recv_scales_wrapper);

    c10::optional<at::Tensor> recv_topk_weights_output;
    if (recv_topk_weights_opt.has_value()) {
        recv_topk_weights_output = *recv_topk_weights_opt;
    }
    c10::optional<at::Tensor> recv_scales_output;
    if (recv_scales_opt.has_value()) {
        recv_scales_output = *recv_scales_opt;
    }
    return std::make_tuple(recv_x, recv_src_metadata, recv_topk_weights_output, recv_scales_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_ep_dispatch_epilogue", &npu_moe_ep_dispatch_epilogue, "moe_ep_dispatch_epilogue");
}

}  // namespace op_api
