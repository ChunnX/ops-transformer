// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;

// const std::vector<at::Tensor> &weight1
std::tuple<at::Tensor, at::Tensor> npu_mega_moe(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_ids,
    const at::Tensor &topk_weights, const std::vector<at::Tensor> &weight1,
    const std::vector<at::Tensor> &weight2, int64_t moe_expert_num, int64_t ep_world_size,
    int64_t ccl_buffer_size,
    const c10::optional<std::vector<at::Tensor>> &weight_scales1,
    const c10::optional<std::vector<at::Tensor>> &weight_scales2,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &scales,
    int64_t max_recv_token_num,
    int64_t dispatch_quant_mode,
    int64_t combine_quant_mode, std::string comm_alg, int64_t global_bs,
    c10::optional<int64_t> dispatch_quant_out_type,
    c10::optional<int64_t> weight1_type,
    c10::optional<int64_t> weight2_type)
{
    TORCH_CHECK((ep_world_size > 0), "The ep_world_sizes should be greater than 0, current is: ", ep_world_size);
    TORCH_CHECK((x.dim() == DIM_TWO) && (topk_ids.dim() == DIM_TWO), "The x and topk_ids should be 2D");
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)) &&
                (topk_ids.scalar_type() == at::kInt),
                "dtype of x should be bfloat16, float16, dtype of topk_ids should be int.");

    at::TensorList weight1_ref = weight1;
    at::TensorList weight2_ref = weight2;

    at::TensorList weight_scales1_ref;
    if (weight_scales1.has_value()) {
        weight_scales1_ref = at::TensorList(weight_scales1.value());
    } else {
        weight_scales1_ref = at::TensorList();
    }
    at::TensorList weight_scales2_ref;
    if (weight_scales2.has_value()) {
        weight_scales2_ref = at::TensorList(weight_scales2.value());
    } else {
        weight_scales2_ref = at::TensorList();
    }

    aclDataType weight1_ref_dtype = weight1_type.has_value() ? GetAclDataType(weight1_type.value())
        : ConvertToAclDataType(weight1_ref[0].scalar_type());
    aclDataType weight_scales1_dtype;
    if (weight1_ref_dtype == aclDataType::ACL_FLOAT8_E5M2 ||
        weight1_ref_dtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight1_ref_dtype == aclDataType::ACL_FLOAT4_E2M1) {
        weight_scales1_dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weight_scales1_dtype = aclDataType::ACL_FLOAT;
    }

    aclDataType weight2_ref_dtype = weight2_type.has_value() ? GetAclDataType(weight2_type.value())
        : ConvertToAclDataType(weight2_ref[0].scalar_type());
    aclDataType weight_scales2_dtype;
    if (weight2_ref_dtype == aclDataType::ACL_FLOAT8_E5M2 ||
        weight2_ref_dtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight2_ref_dtype == aclDataType::ACL_FLOAT4_E2M1) {
        weight_scales2_dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weight_scales2_dtype = aclDataType::ACL_FLOAT;
    }
    
    auto x_size = x.sizes();
    auto topk_ids_size = topk_ids.sizes();
    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = topk_ids_size[1];

    if ((dispatch_quant_out_type.has_value()) &&
        (dispatch_quant_out_type.value() == static_cast<int64_t>(DType::FLOAT4_E2M1))) {
        TORCH_CHECK(h % 2 == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "dispatch quant output type is torch_npu.float4_e2m1");
    }

    int64_t local_moe_expert_num = 1;
    local_moe_expert_num = moe_expert_num / ep_world_size;
    at::Tensor expert_token_nums;
    expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kInt));

    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg.c_str());

    int64_t dispatch_quant_result_type = dispatch_quant_out_type.has_value()
        ? static_cast<int64_t>(GetAclDataType(dispatch_quant_out_type.value()))
        : 28;

    at::Tensor y;
    y = at::empty({bs, h}, topk_ids.options().dtype(x.scalar_type()));

    TensorListWrapper weight1_wrapper = {weight1_ref, weight1_ref_dtype};
    TensorListWrapper weight2_wrapper = {weight2_ref, weight2_ref_dtype};
    TensorListWrapper weight_scales1_wrapper = {weight_scales1_ref, weight_scales1_dtype};
    TensorListWrapper weight_scales2_wrapper = {weight_scales2_ref, weight_scales2_dtype};
    ACLNN_CMD(aclnnMegaMoe, context, x, topk_ids, topk_weights, weight1_wrapper,
        weight2_wrapper, weight_scales1_wrapper, weight_scales2_wrapper, x_active_mask, scales,
        moe_expert_num, ep_world_size, ccl_buffer_size, max_recv_token_num,
        dispatch_quant_mode, dispatch_quant_result_type, combine_quant_mode,
        comm_alg_ptr, global_bs, y, expert_token_nums);

    return std::tie(y, expert_token_nums);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_mega_moe", &npu_mega_moe, "npu_mega_moe");
}

} // namespace op_api
