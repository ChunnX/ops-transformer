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

#include <cmath>
#include <string>
#include <vector>

#include "aclnn_common.h"

struct TensorWithFormatWrapper {
    const at::Tensor &tensor_;
    aclDataType dtype;
    aclFormat format;
};

struct TensorListWithFormatWrapper {
    const at::TensorList &tensor_list_;
    aclDataType dtype;
    aclFormat format;
};

aclTensor *ConvertType(const TensorWithFormatWrapper &tensorWrapper)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    const at::Tensor &atTensor = tensorWrapper.tensor_;
    if (!atTensor.defined()) {
        return nullptr;
    }

    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(atTensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(atTensor.sizes());
    if (tensorWrapper.format == ACL_FORMAT_FRACTAL_NZ) {
        for (auto dim : wrapperShape) {
            storageDims.push_back(dim);
        }
    } else if (tensorWrapper.dtype != ACL_STRING) {
        TORCH_CHECK(atTensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
        if (Is4BitDtype(tensorWrapper.dtype)) {
            storageDims.push_back(atTensor.storage().nbytes() / atTensor.itemsize() * FP4_IN_INT8);
            CollectB4ShapeInfo(atTensor, wrapperStride, wrapperShape);
        } else {
            storageDims.push_back(atTensor.storage().nbytes() / atTensor.itemsize());
        }
    }

    return aclCreateTensor(wrapperShape.data(), atTensor.sizes().size(), tensorWrapper.dtype, wrapperStride.data(),
        atTensor.storage_offset(), tensorWrapper.format, storageDims.data(), storageDims.size(),
        const_cast<void *>(atTensor.storage().data()));
}

aclTensorList *ConvertType(const TensorListWithFormatWrapper &tensorListWrapper)
{
    if (tensorListWrapper.tensor_list_.size() == 0) {
        return nullptr;
    }
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensorList(tensorListWrapper.tensor_list_.size());
    for (size_t i = 0; i < tensorList.size(); ++i) {
        tensorList[i] = ConvertType(TensorWithFormatWrapper{
            tensorListWrapper.tensor_list_[i], tensorListWrapper.dtype, tensorListWrapper.format});
    }
    return aclCreateTensorList(tensorList.data(), tensorList.size());
}

namespace op_api {
namespace {
constexpr int64_t GE_DTYPE_FLOAT = 0;
constexpr int64_t GE_DTYPE_FLOAT8_E5M2 = 35;
constexpr int64_t GE_DTYPE_FLOAT8_E4M3FN = 36;
constexpr int64_t GE_DTYPE_FLOAT8_E8M0 = 37;
constexpr int64_t ACL_DTYPE_OFFSET = 256;
constexpr int64_t DEFAULT_Y_DTYPE = GE_DTYPE_FLOAT8_E4M3FN;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;
constexpr int64_t MX_GROUP_SIZE = 64;
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr float DEFAULT_DST_TYPE_MAX = 0.0F;
constexpr float FLOAT_EPS = 1e-6F;

bool IsKnownGeDtype(int64_t dtype)
{
    return dtype == GE_DTYPE_FLOAT || dtype == GE_DTYPE_FLOAT8_E5M2 ||
        dtype == GE_DTYPE_FLOAT8_E4M3FN || dtype == GE_DTYPE_FLOAT8_E8M0;
}

int64_t NormalizeGeDtype(int64_t dtype)
{
    if (IsKnownGeDtype(dtype)) {
        return dtype;
    }
    if (dtype >= ACL_DTYPE_OFFSET) {
        return dtype - ACL_DTYPE_OFFSET;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float)) {
        return GE_DTYPE_FLOAT;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float8_e5m2)) {
        return GE_DTYPE_FLOAT8_E5M2;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn)) {
        return GE_DTYPE_FLOAT8_E4M3FN;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float8_e8m0fnu)) {
        return GE_DTYPE_FLOAT8_E8M0;
    }
    return dtype;
}

aclDataType GetAclDataTypeFromValue(int64_t dtype)
{
    int64_t geDtype = NormalizeGeDtype(dtype);
    if (IsKnownGeDtype(geDtype)) {
        return static_cast<aclDataType>(geDtype);
    }
    return GetAclDataType(dtype);
}

aclDataType GetTensorAclDataType(const c10::optional<int64_t> &dtypeOptional, const at::Tensor &tensor)
{
    if (dtypeOptional.has_value()) {
        return GetAclDataTypeFromValue(dtypeOptional.value());
    }
    return ConvertToAclDataType(tensor.scalar_type());
}

at::ScalarType GetYScalarType(int64_t yDtype)
{
    int64_t geDtype = NormalizeGeDtype(yDtype);
    if (geDtype == GE_DTYPE_FLOAT8_E4M3FN) {
        return at::ScalarType::Float8_e4m3fn;
    }
    if (geDtype == GE_DTYPE_FLOAT8_E5M2) {
        return at::ScalarType::Float8_e5m2;
    }
    TORCH_CHECK(false, "y_dtype only supports FLOAT8_E4M3FN or FLOAT8_E5M2, but got ", yDtype, ".");
}

int64_t CeilDiv(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

struct WeightLayout {
    bool transpose;
    int64_t n;
};

bool IsNonTransWeightLayout(const at::Tensor &weightTensor, const at::Tensor &weightScaleTensor, int64_t e, int64_t k,
    int64_t n)
{
    return weightTensor.size(DIM_0) == e &&
        weightTensor.size(DIM_1) == CeilDiv(n, 32) &&
        weightTensor.size(DIM_2) == CeilDiv(k, 16) &&
        weightTensor.size(DIM_3) == 16 &&
        weightTensor.size(DIM_4) == 32 &&
        weightScaleTensor.size(DIM_0) == e &&
        weightScaleTensor.size(DIM_1) == CeilDiv(k, MX_GROUP_SIZE) &&
        weightScaleTensor.size(DIM_2) == n &&
        weightScaleTensor.size(DIM_3) == 2;
}

bool IsTransWeightLayout(const at::Tensor &weightTensor, const at::Tensor &weightScaleTensor, int64_t e, int64_t k,
    int64_t n)
{
    return weightTensor.size(DIM_0) == e &&
        weightTensor.size(DIM_1) == CeilDiv(k, 32) &&
        weightTensor.size(DIM_2) == CeilDiv(n, 16) &&
        weightTensor.size(DIM_3) == 16 &&
        weightTensor.size(DIM_4) == 32 &&
        weightScaleTensor.size(DIM_0) == e &&
        weightScaleTensor.size(DIM_1) == n &&
        weightScaleTensor.size(DIM_2) == CeilDiv(k, MX_GROUP_SIZE) &&
        weightScaleTensor.size(DIM_3) == 2;
}

WeightLayout InferWeightLayout(const at::Tensor &weightTensor, const at::Tensor &weightScaleTensor, int64_t e,
    int64_t k)
{
    int64_t nonTransN = weightScaleTensor.size(DIM_2);
    if (IsNonTransWeightLayout(weightTensor, weightScaleTensor, e, k, nonTransN)) {
        return {false, nonTransN};
    }

    int64_t transN = weightScaleTensor.size(DIM_1);
    if (IsTransWeightLayout(weightTensor, weightScaleTensor, e, k, transN)) {
        return {true, transN};
    }

    TORCH_CHECK(false,
        "weight and weight_scale shape must match one of the supported WeightNz layouts: "
        "non-transposed weight (E, ceil(N / 32), ceil(K / 16), 16, 32), "
        "weight_scale (E, ceil(K / 64), N, 2); "
        "transposed weight (E, ceil(K / 32), ceil(N / 16), 16, 32), "
        "weight_scale (E, N, ceil(K / 64), 2).");
    return {false, 0};
}

void CheckTensorList(const std::vector<at::Tensor> &tensorList, const char *name)
{
    TORCH_CHECK(tensorList.size() == 1, name, " only supports one tensor in current MXFP8 mode, but got ",
        tensorList.size(), ".");
    TORCH_CHECK(tensorList[0].defined(), name, "[0] must be a defined tensor.");
}

void CheckBias(const c10::optional<std::vector<at::Tensor>> &biasOptional)
{
    TORCH_CHECK(!biasOptional.has_value() || biasOptional.value().empty(),
        "bias must be None or an empty tensor list in current MXFP8 mode.");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_grouped_matmul_activation_quant(
    const at::Tensor &x,
    const std::vector<at::Tensor> &weight,
    const std::vector<at::Tensor> &weightScale,
    const at::Tensor &groupList,
    const c10::optional<at::Tensor> &xScaleOptional,
    const c10::optional<std::vector<at::Tensor>> &biasOptional,
    std::string activationType,
    int64_t groupListType,
    const c10::optional<std::vector<int64_t>> &tuningConfig,
    std::string quantMode,
    int64_t yDtype,
    std::string roundMode,
    int64_t scaleAlg,
    float dstTypeMax,
    c10::optional<int64_t> xDtype,
    c10::optional<int64_t> weightDtype,
    c10::optional<int64_t> weightScaleDtype,
    c10::optional<int64_t> xScaleDtype)
{
    CheckTensorList(weight, "weight");
    CheckTensorList(weightScale, "weight_scale");
    CheckBias(biasOptional);
    TORCH_CHECK(xScaleOptional.has_value() && xScaleOptional.value().defined(),
        "x_scale must be provided in current MXFP8 mode.");
    TORCH_CHECK(activationType == "gelu_tanh", "activation_type only supports gelu_tanh, but got ", activationType, ".");
    TORCH_CHECK(roundMode == "rint", "round_mode only supports rint, but got ", roundMode, ".");
    TORCH_CHECK(groupListType == 0 || groupListType == 1, "group_list_type only supports 0 or 1, but got ",
        groupListType, ".");
    TORCH_CHECK(scaleAlg == 0 || scaleAlg == 1, "scale_alg only supports 0 or 1, but got ", scaleAlg, ".");
    TORCH_CHECK(std::fabs(dstTypeMax - DEFAULT_DST_TYPE_MAX) <= FLOAT_EPS,
        "dst_type_max is reserved and only supports 0.0 currently, but got ", dstTypeMax, ".");

    int64_t yDtypeValue = NormalizeGeDtype(yDtype);
    TORCH_CHECK(yDtypeValue == GE_DTYPE_FLOAT8_E4M3FN || yDtypeValue == GE_DTYPE_FLOAT8_E5M2,
        "y_dtype only supports FLOAT8_E4M3FN or FLOAT8_E5M2, but got ", yDtype, ".");

    const at::Tensor &weightTensor = weight[0];
    const at::Tensor &weightScaleTensor = weightScale[0];
    const at::Tensor &xScale = xScaleOptional.value();
    TORCH_CHECK(x.dim() == 2, "x must be 2D, but got ", x.dim(), "D.");
    TORCH_CHECK(xScale.dim() == 3, "x_scale must be 3D, but got ", xScale.dim(), "D.");
    TORCH_CHECK(groupList.dim() == 1, "group_list must be 1D, but got ", groupList.dim(), "D.");
    TORCH_CHECK(weightTensor.dim() == 5,
        "weight must be a 5D WeightNz tensor, but got ", weightTensor.dim(), "D.");
    TORCH_CHECK(weightScaleTensor.dim() == 4, "weight_scale must be 4D, but got ", weightScaleTensor.dim(), "D.");
    TORCH_CHECK(groupList.scalar_type() == at::kLong, "group_list dtype must be torch.int64.");

    int64_t m = x.size(DIM_0);
    int64_t k = x.size(DIM_1);
    int64_t e = groupList.size(DIM_0);
    TORCH_CHECK(e >= 1 && e <= MAX_GROUP_NUM, "group_list first dim must be in [1, 1024], but got ", e, ".");
    WeightLayout weightLayout = InferWeightLayout(weightTensor, weightScaleTensor, e, k);
    int64_t n = weightLayout.n;
    TORCH_CHECK(m == 0 || n == 0 || k > 0, "K must be greater than 0 unless M or N is 0.");
    TORCH_CHECK(n % MX_GROUP_SIZE == 0, "N must be divisible by 64, but got ", n, ".");
    TORCH_CHECK(xScale.size(DIM_0) == m && xScale.size(DIM_1) == CeilDiv(k, MX_GROUP_SIZE) &&
                    xScale.size(DIM_2) == 2,
        "x_scale shape must be (M, ceil(K / 64), 2).");

    aclDataType xAclDtype = GetTensorAclDataType(xDtype, x);
    aclDataType weightAclDtype = GetTensorAclDataType(weightDtype, weightTensor);
    aclDataType weightScaleAclDtype = GetTensorAclDataType(weightScaleDtype, weightScaleTensor);
    aclDataType xScaleAclDtype = GetTensorAclDataType(xScaleDtype, xScale);
    TORCH_CHECK(xAclDtype == ACL_FLOAT8_E4M3FN || xAclDtype == ACL_FLOAT8_E5M2,
        "x only supports FLOAT8_E4M3FN or FLOAT8_E5M2.");
    TORCH_CHECK(weightAclDtype == ACL_FLOAT8_E4M3FN, "weight only supports FLOAT8_E4M3FN.");
    TORCH_CHECK(weightScaleAclDtype == ACL_FLOAT8_E8M0, "weight_scale only supports FLOAT8_E8M0.");
    TORCH_CHECK(xScaleAclDtype == ACL_FLOAT8_E8M0, "x_scale only supports FLOAT8_E8M0.");
    if (quantMode.empty() && (xAclDtype == ACL_FLOAT8_E4M3FN || xAclDtype == ACL_FLOAT8_E5M2)) {
        quantMode = "MX";
    }
    TORCH_CHECK(!(xAclDtype == ACL_FLOAT8_E4M3FN || xAclDtype == ACL_FLOAT8_E5M2) || quantMode == "MX",
        "当xdtype为float8_e4m3 或 float8_e5m2时，quant_mode必须为MX");

    at::Tensor y = at::empty({m, n}, x.options().dtype(GetYScalarType(yDtypeValue)));
    at::Tensor yScale = at::empty({m, CeilDiv(n, MX_GROUP_SIZE), 2},
        x.options().dtype(at::ScalarType::Float8_e8m0fnu));

    at::TensorList weightRef = weight;
    at::TensorList weightScaleRef = weightScale;
    at::TensorList biasRef;
    if (biasOptional.has_value()) {
        biasRef = at::TensorList(biasOptional.value());
    } else {
        biasRef = at::TensorList();
    }
    c10::optional<at::IntArrayRef> tuningConfigRef = c10::nullopt;
    if (tuningConfig.has_value()) {
        tuningConfigRef = at::IntArrayRef(tuningConfig.value());
    }

    TensorWithFormatWrapper xWrapper = {x, xAclDtype, ACL_FORMAT_ND};
    TensorWithFormatWrapper groupListWrapper = {groupList, ACL_INT64, ACL_FORMAT_ND};
    TensorWithFormatWrapper xScaleWrapper = {xScale, xScaleAclDtype, ACL_FORMAT_ND};
    TensorListWithFormatWrapper weightWrapper = {weightRef, weightAclDtype, ACL_FORMAT_FRACTAL_NZ};
    TensorListWithFormatWrapper weightScaleWrapper = {weightScaleRef, weightScaleAclDtype, ACL_FORMAT_ND};
    TensorListWrapper biasWrapper = {biasRef, ACL_FLOAT};
    TensorWithFormatWrapper yWrapper = {y, static_cast<aclDataType>(yDtypeValue), ACL_FORMAT_ND};
    TensorWithFormatWrapper yScaleWrapper = {yScale, ACL_FLOAT8_E8M0, ACL_FORMAT_ND};
    char *activationTypePtr = const_cast<char *>(activationType.c_str());
    char *quantModePtr = const_cast<char *>(quantMode.c_str());
    char *roundModePtr = const_cast<char *>(roundMode.c_str());

    ACLNN_CMD(aclnnGroupedMatmulActivationQuantWeightNz,
        xWrapper, groupListWrapper, weightWrapper, weightScaleWrapper, biasWrapper, xScaleWrapper,
        activationTypePtr, weightLayout.transpose, groupListType, tuningConfigRef, quantModePtr, yDtypeValue,
        roundModePtr, scaleAlg, dstTypeMax, yWrapper, yScaleWrapper);
    return std::tie(y, yScale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_grouped_matmul_activation_quant", &npu_grouped_matmul_activation_quant,
        "GroupedMatmulActivationQuant WeightNz torch wrapper");
}
} // namespace op_api
