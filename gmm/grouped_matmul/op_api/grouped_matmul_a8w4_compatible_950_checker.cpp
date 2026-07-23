/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_a8w4_compatible_950_checker.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "log/log.h"
#include "opdev/format_utils.h"

using namespace gmm;

namespace {
constexpr int64_t MAX_COMPATIBLE_K = 18432L;
constexpr int64_t K_ALIGNMENT = 64L;
constexpr int64_t N_ALIGNMENT = 8L;
constexpr int64_t PER_GROUP_SIZE = 256L;
constexpr size_t MAX_BATCHED_EXPERTS = 1024UL;
constexpr size_t MAX_EXPERT_LIST_EXPERTS = 128UL;

struct CompatibleShapeInfo {
    bool isBatched = false;
    bool isNz = false;
    size_t expertNum = 0;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t q = 0;
};

bool IsNzFormat(const aclTensor *tensor)
{
    const auto format = tensor->GetStorageFormat();
    return format == op::Format::FORMAT_FRACTAL_NZ || format == op::Format::FORMAT_FRACTAL_NZ_C0_16 ||
           format == op::Format::FORMAT_FRACTAL_NZ_C0_32;
}

bool IsNdFormat(const aclTensor *tensor)
{
    return tensor->GetStorageFormat() == op::Format::FORMAT_ND;
}

aclnnStatus CheckRequiredList(const aclTensorList *tensorList, const char *name)
{
    CHECK_COND(tensorList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s must not be nullptr in A5 A8W4 Compatible.", name);
    CHECK_COND(tensorList->Size() > 0, ACLNN_ERR_PARAM_INVALID,
               "%s must contain at least one tensor in A5 A8W4 Compatible.", name);
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        CHECK_COND((*tensorList)[i] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "%s[%zu] must not be nullptr in A5 A8W4 Compatible.", name, i);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSingleTensorList(const aclTensorList *tensorList, const char *name)
{
    CHECK_RET(CheckRequiredList(tensorList, name) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(tensorList->Size() == 1, ACLNN_ERR_PARAM_INVALID,
               "%s must contain exactly one tensor in A5 A8W4 Compatible, but got %zu.", name, tensorList->Size());
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNdTensor(const aclTensor *tensor, const char *name)
{
    CHECK_COND(IsNdFormat(tensor), ACLNN_ERR_PARAM_INVALID,
               "%s must use ND format in A5 A8W4 Compatible, but got %s.", name,
               op::ToString(tensor->GetStorageFormat()).GetString());
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShape(const aclTensor *tensor, const std::vector<int64_t> &expected, const char *name)
{
    const auto shape = tensor->GetViewShape();
    CHECK_COND(shape.GetDimNum() == expected.size(), ACLNN_ERR_PARAM_INVALID,
               "%s must have %zu dimensions in A5 A8W4 Compatible, but got %zu.", name, expected.size(),
               shape.GetDimNum());
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK_COND(shape.GetDim(i) == expected[i], ACLNN_ERR_PARAM_INVALID,
                   "%s dim %zu must be %ld in A5 A8W4 Compatible, but got %ld.", name, i, expected[i],
                   shape.GetDim(i));
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveWeight(const GroupedMatmulParams &params, CompatibleShapeInfo &info)
{
    CHECK_RET(CheckRequiredList(params.weight, "weight") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const aclTensor *firstWeight = (*params.weight)[0];
    const auto firstShape = firstWeight->GetViewShape();
    CHECK_COND(firstWeight->GetDataType() == DataType::DT_INT4, ACLNN_ERR_PARAM_INVALID,
               "weight dtype must be INT4 in A5 A8W4 Compatible.");
    CHECK_COND(IsNdFormat(firstWeight) || IsNzFormat(firstWeight), ACLNN_ERR_PARAM_INVALID,
               "weight format must be ND or NZ in A5 A8W4 Compatible.");

    if (firstShape.GetDimNum() == 3 && params.weight->Size() == 1) {
        info.isBatched = true;
        info.expertNum = static_cast<size_t>(firstShape.GetDim(0));
        info.k = firstShape.GetDim(1);
        info.n = firstShape.GetDim(2);
        CHECK_COND(info.expertNum > 0 && info.expertNum <= MAX_BATCHED_EXPERTS, ACLNN_ERR_PARAM_INVALID,
                   "Batched-expert E must be in [1, %zu] in A5 A8W4 Compatible, but got %zu.",
                   MAX_BATCHED_EXPERTS, info.expertNum);
    } else {
        info.isBatched = false;
        info.expertNum = params.weight->Size();
        CHECK_COND(info.expertNum <= MAX_EXPERT_LIST_EXPERTS, ACLNN_ERR_PARAM_INVALID,
                   "Expert-list E must not exceed %zu in A5 A8W4 Compatible, but got %zu.",
                   MAX_EXPERT_LIST_EXPERTS, info.expertNum);
        CHECK_COND(firstShape.GetDimNum() == 2, ACLNN_ERR_PARAM_INVALID,
                   "Each expert-list weight must have shape [K,N] in A5 A8W4 Compatible.");
        info.k = firstShape.GetDim(0);
        info.n = firstShape.GetDim(1);
    }

    info.isNz = IsNzFormat(firstWeight);
    for (size_t i = 0; i < params.weight->Size(); ++i) {
        const aclTensor *weight = (*params.weight)[i];
        CHECK_COND(weight->GetDataType() == DataType::DT_INT4, ACLNN_ERR_PARAM_INVALID,
                   "weight[%zu] dtype must be INT4 in A5 A8W4 Compatible.", i);
        CHECK_COND(info.isNz ? IsNzFormat(weight) : IsNdFormat(weight), ACLNN_ERR_PARAM_INVALID,
                   "All weight tensors must use the same ND/NZ representation in A5 A8W4 Compatible.");
        if (info.isBatched) {
            CHECK_RET(CheckShape(weight, {static_cast<int64_t>(info.expertNum), info.k, info.n}, "weight") ==
                          ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        } else {
            CHECK_RET(CheckShape(weight, {info.k, info.n}, "weight") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckScale(const GroupedMatmulParams &params, CompatibleShapeInfo &info)
{
    CHECK_RET(CheckRequiredList(params.scaleOptional, "scale") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const size_t expectedSize = info.isBatched ? 1UL : info.expertNum;
    CHECK_COND(params.scaleOptional->Size() == expectedSize, ACLNN_ERR_PARAM_INVALID,
               "scale and weight must use the same expert representation in A5 A8W4 Compatible.");

    for (size_t i = 0; i < params.scaleOptional->Size(); ++i) {
        const aclTensor *scale = (*params.scaleOptional)[i];
        CHECK_COND(scale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale[%zu] must not be nullptr.", i);
        CHECK_COND(scale->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "scale[%zu] dtype must be FLOAT in A5 A8W4 Compatible.", i);
        CHECK_RET(CheckNdTensor(scale, "scale") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        const auto shape = scale->GetViewShape();
        const size_t expectedRank = info.isBatched ? 3UL : 2UL;
        CHECK_COND(shape.GetDimNum() == expectedRank, ACLNN_ERR_PARAM_INVALID,
                   "scale and weight must use the same expert representation in A5 A8W4 Compatible.");
        const int64_t q = shape.GetDim(info.isBatched ? 1UL : 0UL);
        const int64_t n = shape.GetDim(info.isBatched ? 2UL : 1UL);
        if (info.isBatched) {
            CHECK_COND(shape.GetDim(0) == static_cast<int64_t>(info.expertNum), ACLNN_ERR_PARAM_INVALID,
                       "scale E must match weight E in A5 A8W4 Compatible.");
        }
        CHECK_COND(n == info.n, ACLNN_ERR_PARAM_INVALID,
                   "scale N must match weight N in A5 A8W4 Compatible.");
        if (i == 0) {
            info.q = q;
        } else {
            CHECK_COND(q == info.q, ACLNN_ERR_PARAM_INVALID,
                       "All scale tensors must use the same Q in A5 A8W4 Compatible.");
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExpertIndexedOptional(const aclTensorList *tensorList, const CompatibleShapeInfo &info,
                                       const std::vector<int64_t> &batchedShape,
                                       const std::vector<int64_t> &expertShape, const char *name)
{
    if (tensorList == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckRequiredList(tensorList, name) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const size_t expectedSize = info.isBatched ? 1UL : info.expertNum;
    CHECK_COND(tensorList->Size() == expectedSize, ACLNN_ERR_PARAM_INVALID,
               "%s and weight must use the same expert representation in A5 A8W4 Compatible.", name);
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        const aclTensor *tensor = (*tensorList)[i];
        CHECK_COND(tensor->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "%s[%zu] dtype must be FLOAT in A5 A8W4 Compatible.", name, i);
        CHECK_RET(CheckNdTensor(tensor, name) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckShape(tensor, info.isBatched ? batchedShape : expertShape, name) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}
}  // namespace

aclnnStatus AclnnGroupedMatmulA8W4CompatibleDAV3510Checker::CheckGroupedMatmulA8W4CompatibleDAV3510() const
{
    CompatibleShapeInfo info;
    CHECK_RET(CheckSingleTensorList(gmmParams_.x, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSingleTensorList(gmmParams_.y, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSingleTensorList(gmmParams_.perTokenScaleOptional, "perTokenScale") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ResolveWeight(gmmParams_, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const aclTensor *x = (*gmmParams_.x)[0];
    CHECK_COND(x->GetDataType() == DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID,
               "x dtype must be INT8 in A5 A8W4 Compatible.");
    CHECK_RET(CheckNdTensor(x, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const auto xShape = x->GetViewShape();
    CHECK_COND(xShape.GetDimNum() == 2, ACLNN_ERR_PARAM_INVALID,
               "x must have shape [M,K] in A5 A8W4 Compatible.");
    info.m = xShape.GetDim(0);
    CHECK_COND(info.m >= 0 && xShape.GetDim(1) == info.k, ACLNN_ERR_PARAM_INVALID,
               "x must have shape [M,K] matching weight K in A5 A8W4 Compatible.");

    CHECK_COND(info.k > 0 && info.k <= MAX_COMPATIBLE_K && info.k % K_ALIGNMENT == 0,
               ACLNN_ERR_PARAM_INVALID,
               "K must be positive, no greater than %ld, and divisible by %ld in A5 A8W4 Compatible, but got %ld.",
               MAX_COMPATIBLE_K, K_ALIGNMENT, info.k);
    CHECK_COND(info.n > 0 && info.n % N_ALIGNMENT == 0, ACLNN_ERR_PARAM_INVALID,
               "N must be positive and divisible by %ld in A5 A8W4 Compatible, but got %ld.", N_ALIGNMENT, info.n);

    CHECK_RET(CheckScale(gmmParams_, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const bool hasOffset = gmmParams_.offsetOptional != nullptr;
    CHECK_COND(info.q > 0 && info.k % info.q == 0, ACLNN_ERR_PARAM_INVALID,
               "Q must be positive and divide K in A5 A8W4 Compatible, but got Q=%ld and K=%ld.", info.q, info.k);
    if (hasOffset) {
        CHECK_COND(info.q == 1, ACLNN_ERR_PARAM_INVALID,
                   "Asymmetric A5 A8W4 Compatible requires Q=1, but got %ld.", info.q);
    } else {
        CHECK_COND(info.q == 1 || (info.k % PER_GROUP_SIZE == 0 && info.q == info.k / PER_GROUP_SIZE),
                   ACLNN_ERR_PARAM_INVALID,
                   "Symmetric A5 A8W4 Compatible requires Q=1 or Q=K/%ld, but got Q=%ld and K=%ld.",
                   PER_GROUP_SIZE, info.q, info.k);
    }

    CHECK_RET(CheckExpertIndexedOptional(gmmParams_.offsetOptional, info,
                                         {static_cast<int64_t>(info.expertNum), 1, info.n}, {1, info.n}, "offset") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckExpertIndexedOptional(gmmParams_.biasOptional, info,
                                         {static_cast<int64_t>(info.expertNum), info.n}, {info.n}, "bias") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    const aclTensor *perTokenScale = (*gmmParams_.perTokenScaleOptional)[0];
    CHECK_COND(perTokenScale->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "perTokenScale dtype must be FLOAT in A5 A8W4 Compatible.");
    CHECK_RET(CheckNdTensor(perTokenScale, "perTokenScale") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(perTokenScale, hasOffset ? std::vector<int64_t>{info.m, 1} : std::vector<int64_t>{info.m},
                         "perTokenScale") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    const aclTensor *y = (*gmmParams_.y)[0];
    CHECK_RET(CheckNdTensor(y, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(y, {info.m, info.n}, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (hasOffset) {
        CHECK_COND(y->GetDataType() == DataType::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
                   "Asymmetric A5 A8W4 Compatible supports only FLOAT16 output.");
    } else {
        CHECK_COND(y->GetDataType() == DataType::DT_FLOAT16 || y->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID,
                   "Symmetric A5 A8W4 Compatible supports FLOAT16 or BF16 output.");
    }

    CHECK_COND(gmmParams_.groupTensorOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "groupList must not be nullptr in A5 A8W4 Compatible.");
    CHECK_COND(gmmParams_.groupTensorOptional->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "groupList dtype must be INT64 in A5 A8W4 Compatible.");
    CHECK_RET(CheckNdTensor(gmmParams_.groupTensorOptional, "groupList") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(gmmParams_.groupTensorOptional, {static_cast<int64_t>(info.expertNum)}, "groupList") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    CHECK_COND(gmmParams_.groupType == SPLIT_M, ACLNN_ERR_PARAM_INVALID,
               "groupType must be 0 in A5 A8W4 Compatible.");
    CHECK_COND(gmmParams_.splitItem == 2 || gmmParams_.splitItem == 3, ACLNN_ERR_PARAM_INVALID,
               "splitItem must be 2 or 3 in A5 A8W4 Compatible.");
    CHECK_COND(!gmmParams_.transposeX && !gmmParams_.transposeWeight, ACLNN_ERR_PARAM_INVALID,
               "transposeX and transposeWeight must both be false in A5 A8W4 Compatible.");
    CHECK_COND(gmmParams_.groupListType == 0 || gmmParams_.groupListType == 1 || gmmParams_.groupListType == 2,
               ACLNN_ERR_PARAM_INVALID, "groupListType must be 0, 1, or 2 in A5 A8W4 Compatible.");

    CHECK_COND(gmmParams_.antiquantScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "antiquantScale must be nullptr in A5 A8W4 Compatible.");
    CHECK_COND(gmmParams_.antiquantOffsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "antiquantOffset must be nullptr in A5 A8W4 Compatible.");

    CHECK_COND((info.isNz && gmmParams_.apiVersion == GMMApiVersion::WeightNz) ||
                   (!info.isNz && gmmParams_.apiVersion == GMMApiVersion::V5),
               ACLNN_ERR_PARAM_INVALID,
               "ND weight requires V5 and NZ weight requires WeightNz in A5 A8W4 Compatible.");
    return ACLNN_SUCCESS;
}
