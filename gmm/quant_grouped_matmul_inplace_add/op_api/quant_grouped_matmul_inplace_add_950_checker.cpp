/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_grouped_matmul_inplace_add_950_checker.h"

using namespace QGmmInPlaceAdd;

namespace QGmmInPlaceAdd {
template class AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<aclTensor>;
}

namespace {
const aclTensor *GetInputTensor(const aclTensorList *input, size_t index = 0)
{
    if (input == nullptr || index >= input->Size()) {
        return nullptr;
    }
    return (*input)[index];
}

const aclTensor *GetInputTensor(const aclTensor *input, size_t index = 0)
{
    (void)index;
    return input;
}

size_t GetInputTensorSize(const aclTensorList *input)
{
    if (input == nullptr) {
        return 0;
    }
    return input->Size();
}

size_t GetInputTensorSize(const aclTensor *input)
{
    (void)input;
    return 1;
}
} // namespace

template <typename T>
void AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::SetInputName(const std::string &xName,
                                                                      const std::string &weightName,
                                                                      const std::string &perTokenScaleName,
                                                                      const std::string &scaleName,
                                                                      const std::string &groupTensorName)
{
    this->xName_ = xName;
    this->weightName_ = weightName;
    this->perTokenScaleName_ = perTokenScaleName;
    this->scaleName_ = scaleName;
    this->groupTensorName_ = groupTensorName;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckTensorListSizeForEachInput() const
{
    CHECK_COND(GetInputTensorSize(gmmParams_.scaleOptional) == GetInputTensorSize(gmmParams_.x),
               ACLNN_ERR_PARAM_INVALID,
               "In quant case, the scale size must equal x size but scale size is [%zu] x size is [%zu].",
               GetInputTensorSize(gmmParams_.scaleOptional), GetInputTensorSize(gmmParams_.x));
    if (gmmParams_.perTokenScaleOptional != nullptr) {
        CHECK_COND(GetInputTensorSize(gmmParams_.perTokenScaleOptional) == GetInputTensorSize(gmmParams_.x),
                   ACLNN_ERR_PARAM_INVALID,
                   "In quant case, the per-token scale size must equal x size but scale per-token size is [%zu] x size "
                   "is [%zu].",
                   GetInputTensorSize(gmmParams_.perTokenScaleOptional), GetInputTensorSize(gmmParams_.x));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckGeneralQuantShape() const
{
    CHECK_RET(CheckTensorListSizeForEachInput() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto yNIndex = GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum() - 1;
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(0) == groupNum, ACLNN_ERR_PARAM_INVALID,
                   "When groupType is 2 (split K), the first dim of %s[%ld] should be equal to that of \
%s[%ld].",
                   yName_.c_str(), GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(0), groupTensorName_.c_str(),
                   groupNum);
        // y shape dim num must 3
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID,
                   "The %s dim num should be equal 3, but actual dim num is [%zu]", yName_.c_str(),
                   GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum());
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(1) ==
                       GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0),
                   ACLNN_ERR_PARAM_INVALID,
                   "The m dim of %s should be equal %s m dim, but actual %s m dim is [%ld], %s m dim is [%ld]",
                   yName_.c_str(), xName_.c_str(), yName_.c_str(),
                   GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(1), xName_.c_str(),
                   GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0));
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(yNIndex) ==
                       GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex),
                   ACLNN_ERR_PARAM_INVALID,
                   "The n dim of %s should be equal %s n dim, but actual %s n dim is [%ld], %s n dim is [%ld]",
                   yName_.c_str(), weightName_.c_str(), yName_.c_str(),
                   GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(yNIndex), weightName_.c_str(),
                   GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckQuantCasesFormat() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of %s[%zu] %s is invalid. It should only be ND.", xName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()).GetString());
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()),
                   ACLNN_ERR_PARAM_INVALID, "The format of %s[%zu] %s is invalid. It should only be ND.",
                   weightName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()).GetString());
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of %s[%zu] %s is invalid. It should only be ND.", yName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()).GetString());
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckHif8QuantParamsShape() const
{
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.weight); ++i) {
        auto xDimNumber = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDimNum();
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();

        // 校验 scale1 (perTokenScale)
        CHECK_COND(perTokenDimNumber == 1 || perTokenDimNumber == 2, ACLNN_ERR_PARAM_INVALID,
                   "In T-T/T-C mode, the %s dim should be 1 or 2, but actual dim is [%zu]", perTokenScaleName_.c_str(),
                   perTokenDimNumber);
        auto perTokenFirstDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(0);
        CHECK_COND(perTokenFirstDim == groupNum, ACLNN_ERR_PARAM_INVALID,
                   "In T-T/T-C mode, the %s first dim must equal groupnum, but actual is [%zu], "
                   "groupnum is [%zu]",
                   perTokenScaleName_.c_str(), perTokenFirstDim, groupNum);
        if (perTokenDimNumber == 2) {
            auto perTokenLastDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(1);
            CHECK_COND(perTokenLastDim == 1, ACLNN_ERR_PARAM_INVALID,
                       "In T-T/T-C mode, the %s last dim should be 1, but actual is [%zu]", perTokenScaleName_.c_str(),
                       perTokenLastDim);
        }

        CHECK_COND(scaleDimNumber == 1 || scaleDimNumber == 2, ACLNN_ERR_PARAM_INVALID,
                   "The %s dim should be 1 or 2, but actual dim is [%zu]", scaleName_.c_str(), scaleDimNumber);
        auto scaleFirstDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(0);
        CHECK_COND(scaleFirstDim == groupNum, ACLNN_ERR_PARAM_INVALID,
                   "In T-T mode, the %s first dim must equal groupnum, but actual is [%zu], "
                   "groupnum is [%zu]",
                   scaleName_.c_str(), scaleFirstDim, groupNum);
        if (scaleDimNumber == 2) {
            auto n = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(1);
            auto scaleLastDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(1);
            CHECK_COND(scaleLastDim == 1 || scaleLastDim == n, ACLNN_ERR_PARAM_INVALID,
                       "In T-T/T-C mode, the %s last dim should be 1 or n[%zu], but actual is [%zu]",
                       scaleName_.c_str(), n, scaleLastDim);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckHif8QuantParams() const
{
    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    CHECK_COND(scaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "With hifloat8 inputs, scale dtype should be float32, but actual dtype is %s",
               op::ToString(scaleDtype).GetString());
    CHECK_COND(gmmParams_.perTokenScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Hifloat8 case perTokenScaleOptional not be null.");
    DataType perTokenScaleDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
    CHECK_COND(perTokenScaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "The %s dtype should be float32 in hifloat8 case, but actual dtype is %s", perTokenScaleName_.c_str(),
               op::ToString(perTokenScaleDtype).GetString());
    CHECK_COND(gmmParams_.biasOptional == nullptr, ACLNN_ERR_PARAM_INVALID, "Hifloat8 case does not support bias.");
    CHECK_COND(gmmParams_.y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Hifloat8 case y not be null.");
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    CHECK_COND(yDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "Expect yDtype to be float32 in hifloat8 quant case, but actual dtype is %s",
               op::ToString(yDtype).GetString());
    CHECK_RET(CheckHif8QuantParamsShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckQuantGroupedMatmulInplaceAddDAV3510() const
{
    DataType xDtype = gmmParams_.xDtype;
    CHECK_COND(gmmParams_.weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In quant case, weight should not be nullptr.");
    DataType weightDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    CHECK_COND(gmmParams_.scaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In quant case, scaleOptional should not be nullptr.");
    CHECK_COND(gmmParams_.groupTensorOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In quant case, groupListOptional should not be nullptr.");
    CHECK_COND(gmmParams_.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID, "Quant case does not support offset.");
    CHECK_COND(GetInputTensorSize(gmmParams_.x) == 1 && GetInputTensorSize(gmmParams_.weight) == 1 &&
                   GetInputTensorSize(gmmParams_.y) == 1,
               ACLNN_ERR_PARAM_INVALID,
               "In quant case, the size of x, weight and y should all be 1, but actual sizes are %zu, %zu and %zu.",
               GetInputTensorSize(gmmParams_.x), GetInputTensorSize(gmmParams_.weight),
               GetInputTensorSize(gmmParams_.y));
    int64_t groupListLen = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    CHECK_COND(groupListLen <= 1024, ACLNN_ERR_PARAM_INVALID, // The group number should not be greater than 1024
                "The length of groupList should not be greater than 1024, but actual is %ld.", groupListLen);
    CHECK_RET(CheckQuantCasesFormat() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGeneralQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (xDtype == DataType::DT_HIFLOAT8 && weightDtype == DataType::DT_HIFLOAT8) {
        return CheckHif8QuantParams();
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GmmInplaceAdd T-C/T-T Quant case with x dtype %s and weight dtype %s is not supported.",
                op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
}