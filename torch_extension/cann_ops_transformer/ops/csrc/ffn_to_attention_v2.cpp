// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <vector>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

/**
 * @brief Wrapper for ffn_to_attention_v2. The op has no host-visible output tensors;
 *        data is sent to peer ranks via the HCCL window. We therefore return an empty
 *        tensor so PyTorch's dispatcher contract (at least one output) is satisfied.
 */
void NpuFFNToAttentionV2(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &sessionIds, const at::Tensor &microBatchIds,
    const at::Tensor &tokenIds, const at::Tensor &expertOffsets, const at::Tensor &actualTokenNum,
    const c10::optional<at::Tensor> &attnRankTable, std::string group, int64_t worldSize,
    const std::vector<int64_t> &tokenInfoTableShape, const std::vector<int64_t> &tokenDataShape)
{
    TORCH_CHECK((x.dim() == DIM_TWO), "The x should be 2D, current dim is: ", x.dim());
    TORCH_CHECK((sessionIds.dim() == DIM_ONE) && (microBatchIds.dim() == DIM_ONE) && (tokenIds.dim() == DIM_ONE) &&
                    (expertOffsets.dim() == DIM_ONE) && (actualTokenNum.dim() == DIM_ONE),
                "sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum should be 1D.");
    TORCH_CHECK((worldSize > 0), "The world_size should be greater than 0, current is: ", worldSize);
    TORCH_CHECK(!group.empty(), "group should not be empty.");
    TORCH_CHECK((tokenInfoTableShape.size() == 3), "token_info_table_shape should be 3D, current size is: ",
                tokenInfoTableShape.size());
    TORCH_CHECK((tokenDataShape.size() == 4), "token_data_shape should be 4D, current size is: ",
                tokenDataShape.size());
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)),
                "dtype of x should be bfloat16 or float16, but got ", x.scalar_type());
    TORCH_CHECK((sessionIds.scalar_type() == at::kInt), "dtype of sessionIds should be int32.");
    TORCH_CHECK((microBatchIds.scalar_type() == at::kInt), "dtype of microBatchIds should be int32.");
    TORCH_CHECK((tokenIds.scalar_type() == at::kInt), "dtype of tokenIds should be int32.");
    TORCH_CHECK((expertOffsets.scalar_type() == at::kInt), "dtype of expertOffsets should be int32.");
    TORCH_CHECK((actualTokenNum.scalar_type() == at::kLong), "dtype of actualTokenNum should be int64.");

    // Set DeviceGuard on x (the first input tensor that lives on NPU)
    auto localDevice = c10::Device(x.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    std::string groupStr = std::string(group);
    char *groupPtr = const_cast<char *>(groupStr.c_str());

    at::IntArrayRef tokenInfoTableShapeRef(tokenInfoTableShape);
    at::IntArrayRef tokenDataShapeRef(tokenDataShape);

    ACLNN_CMD(aclnnFFNToAttentionV2, context, x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum,
              attnRankTable, groupPtr, worldSize, tokenInfoTableShapeRef, tokenDataShapeRef);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_ffn_to_attention_v2", &NpuFFNToAttentionV2, "npu_ffn_to_attention_v2");
}
} // namespace op_api
