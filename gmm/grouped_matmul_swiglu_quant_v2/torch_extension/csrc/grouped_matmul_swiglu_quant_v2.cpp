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

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t kDim0 = 0;
constexpr int64_t kDim1 = 1;
constexpr int64_t kDim2 = 2;
constexpr int64_t kDim3 = 3;
constexpr int64_t kMxGroupSize = 64;
constexpr int64_t kSwigluMode = 2;
constexpr int64_t kGeDtypeFloat8E4M3Fn = 36;
constexpr int64_t kGeDtypeFloat8E8M0 = 37;
constexpr int64_t kAclDtypeOffset = 256;
constexpr double kDefaultClampLimit = 7.0;
constexpr double kDefaultGluAlpha = 1.702;
constexpr double kDefaultGluBias = 1.0;

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

bool IsTransposeWeight(const at::Tensor &tensor, int64_t k)
{
    if (tensor.dim() < 2) {
        return false;
    }
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    const int64_t lastDim = tensor.dim() - 1;
    const int64_t penultimateDim = tensor.dim() - 2;
    // An explicit (E, N, K) logical view identifies ZN even when it is a
    // contiguous FRACTAL_NZ tensor.  For K == N, retain the stride check
    // used by the existing WeightNZ Torch interfaces.
    if (sizes[lastDim] == k && sizes[penultimateDim] != k) {
        return true;
    }
    if (sizes[penultimateDim] == k && sizes[lastDim] != k) {
        return false;
    }
    if (strides[penultimateDim] != 1 || strides[lastDim] != sizes[penultimateDim]) {
        return false;
    }
    int64_t expectedStride = sizes[lastDim] * sizes[penultimateDim];
    for (int64_t dim = tensor.dim() - 3; dim >= 0; --dim) {
        if (strides[dim] != expectedStride) {
            return false;
        }
        expectedStride *= sizes[dim];
    }
    return true;
}

bool IsNzWeight(const at::Tensor &weight)
{
    const aclFormat format = static_cast<aclFormat>(at_npu::native::get_npu_format(weight));
    return format == ACL_FORMAT_FRACTAL_NZ || format == ACL_FORMAT_FRACTAL_NZ_C0_16;
}

int64_t NormalizeGeDtype(int64_t dtype)
{
    if (dtype == kGeDtypeFloat8E4M3Fn || dtype == kGeDtypeFloat8E8M0) {
        return dtype;
    }
    if (dtype >= kAclDtypeOffset) {
        return dtype - kAclDtypeOffset;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn)) {
        return kGeDtypeFloat8E4M3Fn;
    }
    return dtype;
}

aclDataType GetAclDtypeFromValue(int64_t dtype)
{
    const int64_t geDtype = NormalizeGeDtype(dtype);
    if (geDtype == kGeDtypeFloat8E4M3Fn || geDtype == kGeDtypeFloat8E8M0) {
        return static_cast<aclDataType>(geDtype);
    }
    return GetAclDataType(dtype);
}

aclDataType GetTensorAclDtype(const c10::optional<int64_t> &dtype, const at::Tensor &tensor)
{
    return dtype.has_value() ? GetAclDtypeFromValue(dtype.value()) : ConvertToAclDataType(tensor.scalar_type());
}

aclDataType GetMxScaleAclDtype(const c10::optional<int64_t> &dtype)
{
    return dtype.has_value() ? GetAclDtypeFromValue(dtype.value()) : ACL_FLOAT8_E8M0;
}

void CheckList(const std::vector<at::Tensor> &tensors, const char *name)
{
    TORCH_CHECK(tensors.size() == 1, name, " must contain exactly one Tensor in the V3 MXFP8 scenario, but got ",
                tensors.size(), ".");
    TORCH_CHECK(tensors[0].defined(), name, "[0] must be defined.");
}

int64_t InferLogicalN(const at::Tensor &weightScale, bool transposeWeight)
{
    TORCH_CHECK(weightScale.dim() == 4, "weight_scale[0] must be 4D in the V3 MXFP8 scenario, but got ",
                weightScale.dim(), ".");
    return weightScale.size(transposeWeight ? kDim1 : kDim2);
}

void CheckV3Args(const at::Tensor &x, const std::vector<at::Tensor> &weight,
                 const std::vector<at::Tensor> &weightScale, const at::Tensor &xScale,
                 const c10::optional<std::vector<at::Tensor>> &weightAssistMatrix,
                 const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &smoothScale,
                 const c10::optional<std::vector<int64_t>> &tuningConfig, const c10::optional<int64_t> &dequantMode,
                 const c10::optional<int64_t> &dequantDtype, const c10::optional<int64_t> &quantMode,
                 const c10::optional<int64_t> &groupListType, const c10::optional<int64_t> &xDtype,
                 const c10::optional<int64_t> &weightDtype, const c10::optional<int64_t> &weightScaleDtype,
                 const c10::optional<int64_t> &xScaleDtype, const c10::optional<int64_t> &swigluMode,
                 double clampLimit, const std::string &roundMode, int64_t scaleAlg, double dstTypeMax)
{
    CheckList(weight, "weight");
    CheckList(weightScale, "weight_scale");
    TORCH_CHECK(swigluMode.has_value() && swigluMode.value() == kSwigluMode,
                "cann_ops_transformer.grouped_matmul_swiglu_quant_v2 currently only supports swiglu_mode=2.");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "swiglu_mode=2 requires x dtype torch.float8_e4m3fn.");
    TORCH_CHECK(weight[0].scalar_type() == at::ScalarType::Float8_e4m3fn,
                "swiglu_mode=2 requires weight dtype torch.float8_e4m3fn.");
    TORCH_CHECK(IsNzWeight(weight[0]), "swiglu_mode=2 requires weight in FRACTAL_NZ format.");
    // E8M0 is interpreted from its 8-bit bit pattern by the explicit wrapper
    // dtype. Both signed and unsigned 8-bit Torch storage preserve that pattern.
    TORCH_CHECK(xScale.scalar_type() == at::ScalarType::Char || xScale.scalar_type() == at::ScalarType::Byte,
                "swiglu_mode=2 requires x_scale to use torch.int8 or torch.uint8 storage for "
                "torch_npu.float8_e8m0fnu.");
    TORCH_CHECK(weightScale[0].scalar_type() == at::ScalarType::Char ||
                    weightScale[0].scalar_type() == at::ScalarType::Byte,
                "swiglu_mode=2 requires weight_scale to use torch.int8 or torch.uint8 storage for "
                "torch_npu.float8_e8m0fnu.");
    TORCH_CHECK(!weightAssistMatrix.has_value() || weightAssistMatrix->empty(),
                "swiglu_mode=2 does not support weight_assist_matrix.");
    TORCH_CHECK(!bias.has_value() || !bias->defined(), "swiglu_mode=2 does not support bias.");
    TORCH_CHECK(!smoothScale.has_value() || !smoothScale->defined(), "swiglu_mode=2 does not support smooth_scale.");
    TORCH_CHECK(!tuningConfig.has_value() || tuningConfig->empty(), "swiglu_mode=2 does not support tuning_config.");
    TORCH_CHECK(!dequantMode.has_value() || dequantMode.value() == 2,
                "swiglu_mode=2 requires dequant_mode=2.");
    TORCH_CHECK(!dequantDtype.has_value() || dequantDtype.value() == 0,
                "swiglu_mode=2 requires dequant_dtype=torch.float32.");
    TORCH_CHECK(!quantMode.has_value() || quantMode.value() == 2,
                "swiglu_mode=2 requires quant_mode=2.");
    TORCH_CHECK(!groupListType.has_value() || groupListType.value() == 0 || groupListType.value() == 1,
                "group_list_type only supports 0 or 1.");
    TORCH_CHECK(!xDtype.has_value() || GetAclDtypeFromValue(xDtype.value()) == ACL_FLOAT8_E4M3FN,
                "x_dtype only supports torch_npu.float8_e4m3fn.");
    TORCH_CHECK(!weightDtype.has_value() || GetAclDtypeFromValue(weightDtype.value()) == ACL_FLOAT8_E4M3FN,
                "weight_dtype only supports torch_npu.float8_e4m3fn.");
    TORCH_CHECK(!weightScaleDtype.has_value() || GetAclDtypeFromValue(weightScaleDtype.value()) == ACL_FLOAT8_E8M0,
                "weight_scale_dtype only supports torch_npu.float8_e8m0fnu.");
    TORCH_CHECK(!xScaleDtype.has_value() || GetAclDtypeFromValue(xScaleDtype.value()) == ACL_FLOAT8_E8M0,
                "x_scale_dtype only supports torch_npu.float8_e8m0fnu.");
    TORCH_CHECK(std::isfinite(clampLimit) && clampLimit > 0.0, "clamp_limit must be a finite positive number.");
    TORCH_CHECK(roundMode == "rint", "round_mode currently only supports rint.");
    TORCH_CHECK(scaleAlg == 0 || scaleAlg == 1, "MXFP8 only supports scale_alg 0 or 1.");
    TORCH_CHECK(std::isfinite(dstTypeMax) && dstTypeMax == 0.0, "MXFP8 only supports dst_type_max=0.0.");
}

} // namespace

std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_v2(
    const at::Tensor &x, const std::vector<at::Tensor> &weight, const std::vector<at::Tensor> &weightScale,
    const at::Tensor &xScale, const at::Tensor &groupList, const c10::optional<at::Tensor> &smoothScale,
    const c10::optional<std::vector<at::Tensor>> &weightAssistMatrix, const c10::optional<at::Tensor> &bias,
    c10::optional<int64_t> dequantMode, c10::optional<int64_t> dequantDtype, c10::optional<int64_t> quantMode,
    c10::optional<int64_t> quantDtype, c10::optional<int64_t> groupListType,
    const c10::optional<std::vector<int64_t>> &tuningConfig, c10::optional<int64_t> xDtype,
    c10::optional<int64_t> weightDtype, c10::optional<int64_t> weightScaleDtype, c10::optional<int64_t> xScaleDtype,
    c10::optional<int64_t> swigluMode, c10::optional<double> clampLimit, c10::optional<double> gluAlpha,
    c10::optional<double> gluBias, const std::string &roundMode, int64_t scaleAlg, double dstTypeMax)
{
    const double clampLimitValue = clampLimit.value_or(kDefaultClampLimit);
    const double gluAlphaValue = gluAlpha.value_or(kDefaultGluAlpha);
    const double gluBiasValue = gluBias.value_or(kDefaultGluBias);
    CheckV3Args(x, weight, weightScale, xScale, weightAssistMatrix, bias, smoothScale, tuningConfig, dequantMode,
                dequantDtype, quantMode, groupListType, xDtype, weightDtype, weightScaleDtype, xScaleDtype,
                swigluMode, clampLimitValue, roundMode, scaleAlg, dstTypeMax);
    TORCH_CHECK(!quantDtype.has_value() || quantDtype.value() == 1 ||
                    GetAclDtypeFromValue(quantDtype.value()) == ACL_FLOAT8_E4M3FN,
                "swiglu_mode=2 only supports torch.float8_e4m3fn output.");

    const bool transposeWeight = IsTransposeWeight(weight[0], x.size(kDim1));
    const int64_t m = x.size(kDim0);
    const int64_t n = InferLogicalN(weightScale[0], transposeWeight);
    TORCH_CHECK(n > 0 && n % 2 == 0, "The logical N dimension must be positive and even, but got ", n, ".");

    at::Tensor output;
    at::Tensor outputScale;
    {
        const c10::OptionalDeviceGuard deviceGuard(c10::Device(x.device()));
        output = at::empty({m, n / 2}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
        outputScale = at::empty({m, CeilDiv(n / 2, kMxGroupSize), 2},
                                x.options().dtype(at::ScalarType::Float8_e8m0fnu));
    }

    at::TensorList weightRef = weight;
    at::TensorList weightScaleRef = weightScale;
    const std::vector<at::Tensor> emptyAssist;
    const std::vector<at::Tensor> &assist = weightAssistMatrix.has_value() ? weightAssistMatrix.value() : emptyAssist;
    at::TensorList assistRef = assist;
    const at::Tensor biasRef = bias.value_or(at::Tensor());
    const at::Tensor smoothScaleRef = smoothScale.value_or(at::Tensor());
    c10::optional<at::IntArrayRef> tuningConfigRef = c10::nullopt;
    if (tuningConfig.has_value()) {
        tuningConfigRef = at::IntArrayRef(tuningConfig.value());
    }

    TensorWrapper xWrapper = {x, GetTensorAclDtype(xDtype, x)};
    TensorListWrapper weightWrapper = {weightRef, GetTensorAclDtype(weightDtype, weight[0])};
    TensorListWrapper weightScaleWrapper = {weightScaleRef, GetMxScaleAclDtype(weightScaleDtype)};
    TensorWrapper xScaleWrapper = {xScale, GetMxScaleAclDtype(xScaleDtype)};
    TensorWrapper outputWrapper = {output, ACL_FLOAT8_E4M3FN};
    TensorWrapper outputScaleWrapper = {outputScale, ACL_FLOAT8_E8M0};
    const char *roundModePtr = "rint";
    // ACLNN_CMD forwards every argument through the ACL type converter, whose
    // interface takes non-const lvalue references.  Keep the scalar attributes
    // in named variables instead of passing value_or() / temporary expressions.
    int64_t dequantModeValue = dequantMode.value_or(2);
    int64_t dequantDtypeValue = dequantDtype.value_or(0);
    int64_t quantModeValue = quantMode.value_or(2);
    int64_t groupListTypeValue = groupListType.value_or(0);
    int64_t swigluModeValue = swigluMode.value();
    double clampLimitArg = clampLimitValue;
    double gluAlphaArg = gluAlphaValue;
    double gluBiasArg = gluBiasValue;
    int64_t scaleAlgValue = scaleAlg;
    double dstTypeMaxArg = dstTypeMax;

    ACLNN_CMD(aclnnGroupedMatmulSwigluQuantWeightNzV3, xWrapper, weightWrapper, weightScaleWrapper, assistRef,
              biasRef, xScaleWrapper, smoothScaleRef, groupList, dequantModeValue, dequantDtypeValue,
              quantModeValue, groupListTypeValue, tuningConfigRef, swigluModeValue, clampLimitArg,
              gluAlphaArg, gluBiasArg, roundModePtr, scaleAlgValue, dstTypeMaxArg, outputWrapper, outputScaleWrapper);
    return std::tie(output, outputScale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_matmul_swiglu_quant_v2", &grouped_matmul_swiglu_quant_v2,
          "GroupedMatmulSwigluQuant V3 MXFP8 WeightNz torch wrapper");
}
} // namespace op_api
