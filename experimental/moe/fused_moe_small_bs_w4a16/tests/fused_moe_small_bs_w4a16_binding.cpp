// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

/*!
 * \file fused_moe_small_bs_w4a16_binding.cpp
 * \brief 实验性质算子 fused_moe_small_bs_w4a16 的自包含 torch 绑定
 *
 * 直接把 at::Tensor 桥接到 custom_transformer 算子包的 aclnn 两段式接口，
 * 不依赖 cann_ops_transformer 共享扩展包。通过 torch.utils.cpp_extension.load
 * JIT 编译（见同目录 _binding.py）。
 */

#include <vector>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclnnop/aclnn_fused_moe_small_bs_w4a16.h"

namespace {
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t PACK_RATIO = 8;
constexpr int64_t GROUP_SIZE = 32;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t W13_OUT_FACTOR = 2;

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kInt:
            return ACL_INT32;
        case at::kFloat:
            return ACL_FLOAT;
        default:
            TORCH_CHECK(false, "unsupported dtype: ", type);
    }
}

// 由连续 at::Tensor 构造 aclTensor（调用方负责 aclDestroyTensor）
aclTensor *ToAclTensor(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
    auto sizes = tensor.sizes();
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = sizes[i + 1] * strides[i + 1];
    }
    return aclCreateTensor(sizes.data(), sizes.size(), ToAclDataType(tensor.scalar_type()), strides.data(), 0,
                           ACL_FORMAT_ND, sizes.data(), sizes.size(), tensor.data_ptr());
}

void CheckInputs(const at::Tensor &x, const at::Tensor &w13Weight, const at::Tensor &w13Scales,
                 const at::Tensor &w2Weight, const at::Tensor &w2Scales, const at::Tensor &expertIds,
                 const at::Tensor &topkWeights)
{
    auto xType = x.scalar_type();
    TORCH_CHECK(xType == at::kHalf || xType == at::kBFloat16,
                "x only supports float16/bfloat16, current is: ", xType);
    TORCH_CHECK(w13Weight.scalar_type() == at::kInt && w2Weight.scalar_type() == at::kInt,
                "w13_weight/w2_weight must be int32 (packed int4).");
    TORCH_CHECK(w13Scales.scalar_type() == xType && w2Scales.scalar_type() == xType,
                "scales must have the same dtype as x.");
    TORCH_CHECK(expertIds.scalar_type() == at::kInt, "expert_ids must be int32.");
    TORCH_CHECK(topkWeights.scalar_type() == at::kFloat, "topk_weights must be float32.");

    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D [BatchSize, HiddenSize], current dim is: ", x.dim());
    int64_t batchSize = x.size(0);
    int64_t hiddenSize = x.size(1);
    TORCH_CHECK(hiddenSize % BLOCK_SIZE == 0,
                "hiddenSize must be a multiple of ", BLOCK_SIZE, ", current is: ", hiddenSize);
    TORCH_CHECK(expertIds.dim() == DIM_TWO && expertIds.size(0) == batchSize,
                "expert_ids must be 2D [BatchSize, TopK] and match x.");
    TORCH_CHECK(topkWeights.sizes() == expertIds.sizes(), "topk_weights must match expert_ids shape.");

    TORCH_CHECK(w13Weight.dim() == DIM_THREE, "w13_weight must be 3D [NumExperts, HiddenSize, 2 * InterSize / 8].");
    int64_t numExperts = w13Weight.size(0);
    TORCH_CHECK(w13Weight.size(1) == hiddenSize, "w13_weight dim1 must equal hiddenSize.");
    int64_t w13OutSize = w13Weight.size(2) * PACK_RATIO;
    TORCH_CHECK(w13OutSize % W13_OUT_FACTOR == 0, "w13_weight packed out dim is invalid.");
    int64_t interSize = w13OutSize / W13_OUT_FACTOR;
    TORCH_CHECK(interSize % BLOCK_SIZE == 0,
                "interSize must be a multiple of ", BLOCK_SIZE, ", current is: ", interSize);
    TORCH_CHECK(w13Scales.dim() == DIM_THREE && w13Scales.size(0) == numExperts &&
                    w13Scales.size(1) == hiddenSize / GROUP_SIZE && w13Scales.size(2) == w13OutSize,
                "w13_scales must be [NumExperts, HiddenSize / 32, 2 * InterSize].");
    TORCH_CHECK(w2Weight.dim() == DIM_THREE && w2Weight.size(0) == numExperts && w2Weight.size(1) == interSize &&
                    w2Weight.size(2) * PACK_RATIO == hiddenSize,
                "w2_weight must be [NumExperts, InterSize, HiddenSize / 8].");
    TORCH_CHECK(w2Scales.dim() == DIM_THREE && w2Scales.size(0) == numExperts &&
                    w2Scales.size(1) == interSize / GROUP_SIZE && w2Scales.size(2) == hiddenSize,
                "w2_scales must be [NumExperts, InterSize / 32, HiddenSize].");
}
} // namespace

at::Tensor fused_moe_small_bs_w4a16(const at::Tensor &x, const at::Tensor &w13Weight, const at::Tensor &w13Scales,
                                    const at::Tensor &w2Weight, const at::Tensor &w2Scales,
                                    const at::Tensor &expertIds, const at::Tensor &topkWeights)
{
    CheckInputs(x, w13Weight, w13Scales, w2Weight, w2Scales, expertIds, topkWeights);

    // DeviceGuard + 输出申请
    const c10::OptionalDeviceGuard deviceGuard(x.device());
    at::Tensor y = at::empty(x.sizes(), x.options());

    // at::Tensor -> aclTensor
    aclTensor *aclX = ToAclTensor(x);
    aclTensor *aclW13Weight = ToAclTensor(w13Weight);
    aclTensor *aclW13Scales = ToAclTensor(w13Scales);
    aclTensor *aclW2Weight = ToAclTensor(w2Weight);
    aclTensor *aclW2Scales = ToAclTensor(w2Scales);
    aclTensor *aclExpertIds = ToAclTensor(expertIds);
    aclTensor *aclTopkWeights = ToAclTensor(topkWeights);
    aclTensor *aclY = ToAclTensor(y);

    // 第一段：计算 workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnFusedMoeSmallBsW4a16GetWorkspaceSize(aclX, aclW13Weight, aclW13Scales, aclW2Weight, aclW2Scales,
                                                         aclExpertIds, aclTopkWeights, aclY, &workspaceSize,
                                                         &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedMoeSmallBsW4a16GetWorkspaceSize failed, ret: ", ret);

    at::Tensor workspaceTensor;
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));
        workspaceAddr = workspaceTensor.data_ptr();
    }

    // 第二段：在当前 NPU stream 上执行
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ret = aclnnFusedMoeSmallBsW4a16(workspaceAddr, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedMoeSmallBsW4a16 failed, ret: ", ret);

    aclDestroyTensor(aclX);
    aclDestroyTensor(aclW13Weight);
    aclDestroyTensor(aclW13Scales);
    aclDestroyTensor(aclW2Weight);
    aclDestroyTensor(aclW2Scales);
    aclDestroyTensor(aclExpertIds);
    aclDestroyTensor(aclTopkWeights);
    aclDestroyTensor(aclY);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_moe_small_bs_w4a16", &fused_moe_small_bs_w4a16, "fused moe w4a16 small batch size");
}
