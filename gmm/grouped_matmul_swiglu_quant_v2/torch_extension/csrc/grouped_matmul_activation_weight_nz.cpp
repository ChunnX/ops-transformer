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

#include <c10/core/DeviceGuard.h>

#include <tuple>
#include <vector>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t kDim0 = 0L;
constexpr int64_t kDim2 = 2L;
constexpr int64_t kWeightScaleRank = 4L;
constexpr int64_t kSwigluSplitFactor = 2L;
constexpr int64_t kMxAlignment = 128L;
constexpr int64_t kMxDequantMode = 2L;
constexpr int64_t kFloatDequantDtype = 0L;
constexpr int64_t kSwigluActType = 0L;

void CheckPackedFp4(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined.");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Char || tensor.scalar_type() == at::ScalarType::Byte, name,
                " must use torch.int8 or torch.uint8 storage for FLOAT4_E2M1.");
}

void CheckE8M0(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined.");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Char || tensor.scalar_type() == at::ScalarType::Byte, name,
                " must use torch.int8 or torch.uint8 storage for FLOAT8_E8M0.");
}

void CheckSingleTensorList(const std::vector<at::Tensor>& tensors, const char* name)
{
    TORCH_CHECK(tensors.size() == 1UL && tensors[0].defined(), name, " must contain exactly one defined tensor.");
}
} // namespace

at::Tensor grouped_matmul_activation_weight_nz(
    const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& groupList, const std::vector<at::Tensor>& weight,
    const std::vector<at::Tensor>& weightScale, const c10::optional<at::Tensor>& bias, int64_t dequantMode,
    int64_t dequantDtype, int64_t actType, bool transposeWeight, int64_t groupListType,
    const c10::optional<std::vector<int64_t>>& tuningConfig, int64_t swigluMode, double clampLimit, double gluAlpha,
    double gluBias)
{
    CheckPackedFp4(x, "x");
    CheckE8M0(xScale, "x_scale");
    CheckSingleTensorList(weight, "weight");
    CheckSingleTensorList(weightScale, "weight_scale");
    CheckPackedFp4(weight[0], "weight[0]");
    CheckE8M0(weightScale[0], "weight_scale[0]");
    TORCH_CHECK(groupList.scalar_type() == at::ScalarType::Long, "group_list must have torch.int64 dtype.");
    TORCH_CHECK(!bias.has_value() || !bias.value().defined(), "bias must be None for the current A4W4 path.");
    TORCH_CHECK(x.dim() > kDim0 && weightScale[0].dim() == kWeightScaleRank,
                "x must be at least 1D and weight_scale[0] must be 4D.");
    TORCH_CHECK(dequantMode == kMxDequantMode && dequantDtype == kFloatDequantDtype && actType == kSwigluActType,
                "only dequant_mode=2, dequant_dtype=0 and act_type=0 are supported.");

    const int64_t m = x.size(kDim0);
    const int64_t logicalN = weightScale[0].size(kDim2);
    TORCH_CHECK(logicalN > 0L && logicalN % kSwigluSplitFactor == 0L && logicalN % kMxAlignment == 0L,
                "A4W4 requires logical N to be divisible by 128.");
    at::Tensor y;
    {
        const c10::OptionalDeviceGuard deviceGuard(c10::Device(x.device()));
        y = at::empty({m, logicalN / kSwigluSplitFactor}, x.options().dtype(at::ScalarType::BFloat16));
    }

    at::TensorList weightRef = weight;
    at::TensorList weightScaleRef = weightScale;
    const at::Tensor biasRef;
    c10::optional<at::IntArrayRef> tuningConfigRef = c10::nullopt;
    if (tuningConfig.has_value()) {
        tuningConfigRef = at::IntArrayRef(tuningConfig.value());
    }
    TensorWrapper xWrapper = {x, ACL_FLOAT4_E2M1};
    TensorWrapper xScaleWrapper = {xScale, ACL_FLOAT8_E8M0};
    TensorListWrapper weightWrapper = {weightRef, ACL_FLOAT4_E2M1};
    TensorListWrapper weightScaleWrapper = {weightScaleRef, ACL_FLOAT8_E8M0};
    TensorWrapper yWrapper = {y, ACL_BF16};

    ACLNN_CMD(aclnnGroupedMatmulActivationWeightNz, xWrapper, xScaleWrapper, groupList, weightWrapper,
              weightScaleWrapper, biasRef, dequantMode, dequantDtype, actType, transposeWeight, groupListType,
              tuningConfigRef, swigluMode, clampLimit, gluAlpha, gluBias, yWrapper);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_matmul_activation_weight_nz", &grouped_matmul_activation_weight_nz,
          "GroupedMatmulActivationWeightNZ A4W4 torch wrapper");
}
} // namespace op_api
