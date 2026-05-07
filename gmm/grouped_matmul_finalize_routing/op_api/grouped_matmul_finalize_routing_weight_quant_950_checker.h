/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_950_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_950_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "quant_grouped_matmul_finalize_routing_util.h"
#include "../../grouped_matmul/op_api/grouped_matmul_util.h"
#include "util/math_util.h"

namespace GmmFinalizeRouting {

constexpr size_t WQ_DIM_ZERO = 0UL;
constexpr size_t WQ_DIM_ONE = 1UL;
constexpr size_t WQ_DIM_TWO = 2UL;
constexpr size_t WQ_DIM_THREE = 3UL;
constexpr size_t WQ_DIM_FOUR = 4UL;
constexpr size_t WQ_DIM_FIVE = 5UL;
constexpr int64_t WQ_GMMFR_SPLIT_SIZE = 64L;
constexpr int64_t WQ_GMMFR_SPLIT_FACTOR = 2L;
constexpr int64_t WQ_MAX_NUM_EXPERTS = 1024L;
constexpr int64_t K_ALIGN_SIZE = 32L;
constexpr int64_t N_ALIGN_SIZE = 32L;

static const std::initializer_list<op::DataType> A8W4_X_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> A8W4_W_TYPE_LIST = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> A8W4_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> A8W4_PERTOKEN_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> A8W4_BIAS_TYPE_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> A8W4_GROUP_LIST_TYPE_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> A8W4_SHARED_INPUT_TYPE_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> A8W4_LOGIT_TYPE_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> A8W4_ROW_INDEX_TYPE_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> A8W4_OUT_TYPE_LIST = {op::DataType::DT_FLOAT};

class GroupedMatmulFinalizeRoutingWeightQuant950Checker {
public:
    GroupedMatmulFinalizeRoutingWeightQuant950Checker() = default;

    aclnnStatus CheckParams(GroupedMatmulParams &gmmParams)
    {
        gmmParams_ = gmmParams;
        // 1. 检查参数是否为空指针
        CHECK_RET(CheckNotNull() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
        // 2. 检查输入的数据类型是否在支持的数据类型范围之内
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);
        // 3. 校验转置属性
        CHECK_RET(CheckTranspose() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        // 4. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        // 5. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape(), ACLNN_ERR_PARAM_INVALID);
        // 6. 检查数据形状是否支持
        CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckNotNull()
    {
        CHECK_COND(gmmParams_.x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x should not be nullptr.");
        CHECK_COND(gmmParams_.x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight should not be nullptr.");
        CHECK_COND(gmmParams_.scale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale should not be nullptr.");
        CHECK_COND(gmmParams_.groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "groupList should not be nullptr.");
        CHECK_COND(gmmParams_.logit != nullptr, ACLNN_ERR_PARAM_NULLPTR, "logit should not be nullptr.");
        CHECK_COND(gmmParams_.rowIndex != nullptr, ACLNN_ERR_PARAM_NULLPTR, "rowIndex should not be nullptr.");
        CHECK_COND(gmmParams_.out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "out should not be nullptr.");
        CHECK_COND(gmmParams_.pertokenScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "In A8W4 quant, pertokenScaleOptional should not be nullptr.");
        CHECK_COND(gmmParams_.offset == nullptr, ACLNN_ERR_PARAM_INVALID, "A8W4 quant mode does not support offset.");
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckTranspose()
    {
        CHECK_COND(gmmParams_.transposeX1 == false, ACLNN_ERR_PARAM_INVALID,
                   "In A8W4 quant, x1 must not be transposed (transposeX1 should be false).");
        CHECK_COND(gmmParams_.transposeX2 == true, ACLNN_ERR_PARAM_INVALID,
                   "In A8W4 quant, weight must be transposed (transposeX2 should be true).");
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputOutDims()
    {
        auto xDimNumber = gmmParams_.x1->GetViewShape().GetDimNum();
        auto wDimNumber = gmmParams_.x2->GetViewShape().GetDimNum();
        auto wStorageDimNumber = gmmParams_.x2->GetStorageShape().GetDimNum();
        auto wScaleDimNumber = gmmParams_.scale->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmParams_.pertokenScaleOptional->GetViewShape().GetDimNum();
        auto grouplistDimNumber = gmmParams_.groupList->GetViewShape().GetDimNum();
        auto logitDimNumber = gmmParams_.logit->GetViewShape().GetDimNum();
        auto rowindexDimNumber = gmmParams_.rowIndex->GetViewShape().GetDimNum();
        auto outDimNumber = gmmParams_.out->GetViewShape().GetDimNum();
        CHECK_COND(xDimNumber == WQ_DIM_TWO, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of x should be equal to 2, current dim is %lu.", xDimNumber);
        CHECK_COND(wDimNumber == WQ_DIM_THREE, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of w should be equal to 3, current dim is %lu.", wDimNumber);
        CHECK_COND(wStorageDimNumber == WQ_DIM_FIVE, ACLNN_ERR_PARAM_INVALID,
                   "The storage dim num of w should be equal to 5, current dim is %lu.", wStorageDimNumber);
        CHECK_COND(wScaleDimNumber == WQ_DIM_FOUR, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of scale should be equal to 4, current dim is %lu.", wScaleDimNumber);
        CHECK_COND(xScaleDimNumber == WQ_DIM_THREE, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of pertokenscale should be equal to 3, current dim is %lu.", xScaleDimNumber);
        CHECK_COND(grouplistDimNumber == WQ_DIM_ONE, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of grouplist should be equal to 1, current dim is %lu.", grouplistDimNumber);
        CHECK_COND(logitDimNumber == WQ_DIM_ONE, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of logit should be equal to 1, current dim is %lu.", logitDimNumber);
        CHECK_COND(rowindexDimNumber == WQ_DIM_ONE, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of rowindex should be equal to 1, current dim is %lu.", rowindexDimNumber);
        CHECK_COND(outDimNumber == WQ_DIM_TWO, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of out should be equal to 2, current dim is %lu.", outDimNumber);
        if (gmmParams_.bias != nullptr) {
            auto biasDimNumber = gmmParams_.bias->GetViewShape().GetDimNum();
            CHECK_COND(biasDimNumber == WQ_DIM_TWO, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of bias should be equal to 2, current dim is %lu.", biasDimNumber);
        }
        if (gmmParams_.shareInput != nullptr) {
            auto shareInputDimNumber = gmmParams_.shareInput->GetViewShape().GetDimNum();
            CHECK_COND(shareInputDimNumber == WQ_DIM_TWO, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of shareinput should be equal to 2, current dim is %lu.", shareInputDimNumber);
        }
        return ACLNN_SUCCESS;
    }

    bool CheckInputOutShape()
    {
        if (!CheckInputOutShapeConsistency()) {
            return false;
        }
        int64_t m = gmmParams_.x1->GetViewShape().GetDim(WQ_DIM_ZERO);
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(WQ_DIM_ONE);
        // MxA8W4 only supports transposed weight: viewShape=(e, n, k)
        int64_t n = (gmmParams_.x2)->GetViewShape().GetDim(WQ_DIM_ONE);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0);
        int64_t outputBS = gmmParams_.out->GetViewShape().GetDim(0);
        if (k <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GroupedMatmulFinalizeRoutingWeightQuant", "k", std::to_string(k).c_str(), "K value should be positive");
            return false;
        }
        // A8W4约束: k % 32 == 0
        if (k % K_ALIGN_SIZE != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GroupedMatmulFinalizeRoutingWeightQuant", "k", std::to_string(k).c_str(),
                                                  "k must be a multiple of 32 in A8W4 quant");
            return false;
        }
        // A8W4约束: n % 32 == 0
        if (n % N_ALIGN_SIZE != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GroupedMatmulFinalizeRoutingWeightQuant", "n", std::to_string(n).c_str(),
                                                  "n must be a multiple of 32 in A8W4 quant");
            return false;
        }
        if (!CheckRequiredShapes(m, k, n, e, outputBS)) {
            return false;
        }
        if (!CheckOptionalShapes(m, k, e, n)) {
            return false;
        }
        return true;
    }

    bool CheckRequiredShapes(int64_t m, int64_t k, int64_t n, int64_t e, int64_t outputBS)
    {
        op::Shape xExpectShape = {m, k};
        // MxA8W4 only supports transposed weight: viewShape=(e, n, k)
        op::Shape weightExpectShape = {e, n, k};
        // MxA8W4 scale shape: (e, n, ceil(k/64), 2)
        op::Shape weightScaleExpectShape =
            op::Shape{e, n, Ops::Base::CeilDiv(k, WQ_GMMFR_SPLIT_SIZE), WQ_GMMFR_SPLIT_FACTOR};
        op::Shape xScaleExpectShape = op::Shape{m, Ops::Base::CeilDiv(k, WQ_GMMFR_SPLIT_SIZE), WQ_GMMFR_SPLIT_FACTOR};
        op::Shape grouplistExpectShape = {e};
        op::Shape logitExpectShape = {m};
        op::Shape rowindexExpectShape = {m};
        op::Shape outputExpectShape = {outputBS, n};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x1, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x2, weightExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.scale, weightScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.pertokenScaleOptional, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.groupList, grouplistExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.logit, logitExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.rowIndex, rowindexExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.out, outputExpectShape, return false);
        return true;
    }

    bool CheckOptionalShapes(int64_t m, int64_t k, int64_t e, int64_t n)
    {
        if (gmmParams_.bias != nullptr) {
            op::Shape biasExpectShape = {e, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.bias, biasExpectShape, return false);
        }
        if (gmmParams_.shareInput != nullptr) {
            int64_t bsdp = gmmParams_.shareInput->GetViewShape().GetDim(0);
            if (bsdp > gmmParams_.out->GetViewShape().GetDim(0)) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    "GroupedMatmulFinalizeRoutingWeightQuant", "shareInput_batch, outputBS",
                    (std::to_string(bsdp) + ", " + std::to_string(gmmParams_.out->GetViewShape().GetDim(0))).c_str(),
                    "shareInput batch should <= outputBS");
                return false;
            }
            if (gmmParams_.shareInputOffset > gmmParams_.out->GetViewShape().GetDim(0) - bsdp) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("GroupedMatmulFinalizeRoutingWeightQuant", "sharedInputOffset, shareInput_batch, outputBS",
                                                       (std::to_string(gmmParams_.shareInputOffset) + ", " +
                                                        std::to_string(bsdp) + ", " +
                                                        std::to_string(gmmParams_.out->GetViewShape().GetDim(0)))
                                                           .c_str(),
                                                       "sharedInputOffset + shareInput_batch should <= outputBS");
                return false;
            }
            op::Shape shareInputExpectShape = {bsdp, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.shareInput, shareInputExpectShape, return false);
        }
        return true;
    }

    bool CheckInputOutShapeConsistency()
    {
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(WQ_DIM_ONE);
        // MxA8W4 only supports transposed weight: viewShape=(e, n, k), so k is at dim 2
        int64_t kInWeight = gmmParams_.x2->GetViewShape().GetDim(WQ_DIM_TWO);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0);
        if (kInWeight != k) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("GroupedMatmulFinalizeRoutingWeightQuant", "x, weight",
                                                   (std::to_string(k) + ", " + std::to_string(kInWeight)).c_str(),
                                                   "The k-axis of x and weight must be equal");
            return false;
        }
        int64_t groupListLen = gmmParams_.groupList->GetViewShape().GetDim(WQ_DIM_ZERO);
        if (groupListLen != e) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "GroupedMatmulFinalizeRoutingWeightQuant", "groupList, weight", (std::to_string(groupListLen) + ", " + std::to_string(e)).c_str(),
                "Length of groupList should be equal to the number of experts in weight");
            return false;
        }
        if (e > WQ_MAX_NUM_EXPERTS) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GroupedMatmulFinalizeRoutingWeightQuant", "e", std::to_string(e).c_str(),
                                                  "In A8W4 quant, e must be <= 1024");
            return false;
        }
        return true;
    }

    bool CheckDtypeValid()
    {
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.x1, A8W4_X_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.x2, A8W4_W_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.scale, A8W4_SCALE_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.rowIndex, A8W4_ROW_INDEX_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.pertokenScaleOptional, A8W4_PERTOKEN_SCALE_TYPE_LIST, return false);
        if (gmmParams_.bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.bias, A8W4_BIAS_TYPE_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.groupList, A8W4_GROUP_LIST_TYPE_LIST, return false);
        if (gmmParams_.shareInput != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.shareInput, A8W4_SHARED_INPUT_TYPE_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.logit, A8W4_LOGIT_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.out, A8W4_OUT_TYPE_LIST, return false);
        return true;
    }

    bool CheckFormat()
    {
        if (!CheckXAndWeightFormat()) {
            return false;
        }
        if (!CheckOtherTensorFormats()) {
            return false;
        }
        return true;
    }

    bool CheckXAndWeightFormat()
    {
        if (op::IsPrivateFormat(gmmParams_.x1->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "x", op::ToString(gmmParams_.x1->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        // A8W4场景：权重为NZ格式
        if (gmmParams_.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ &&
            gmmParams_.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ_C0_32) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "weight", op::ToString(gmmParams_.x2->GetStorageFormat()).GetString(),
                                       "NZ(C0_32)");
            return false;
        }
        return true;
    }

    bool CheckOtherTensorFormats()
    {
        if (op::IsPrivateFormat(gmmParams_.pertokenScaleOptional->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "pertokenScaleOptional",
                                       op::ToString(gmmParams_.pertokenScaleOptional->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.scale->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "scale", op::ToString(gmmParams_.scale->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.groupList->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "groupList",
                                       op::ToString(gmmParams_.groupList->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.logit->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "logit", op::ToString(gmmParams_.logit->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.rowIndex->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "rowIndex",
                                       op::ToString(gmmParams_.rowIndex->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.out->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "out", op::ToString(gmmParams_.out->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        if (gmmParams_.bias != nullptr && op::IsPrivateFormat(gmmParams_.bias->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "bias", op::ToString(gmmParams_.bias->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        if (gmmParams_.shareInput != nullptr && op::IsPrivateFormat(gmmParams_.shareInput->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("GroupedMatmulFinalizeRoutingWeightQuant", "shareInput",
                                       op::ToString(gmmParams_.shareInput->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        return true;
    }

private:
    GroupedMatmulParams gmmParams_;
};
} // namespace GmmFinalizeRouting
#endif
