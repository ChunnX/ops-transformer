/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H

#include "aclnn_grouped_matmul_swiglu_quant_utils.h"
#include "util/math_util.h"

namespace gmmSwigluQuantV2 {

using namespace gmm_dsq;

constexpr int64_t OUTPUT_IDX_0 = 0L;
constexpr int64_t OUTPUT_IDX_1 = 1L;
constexpr size_t MX_SPLIT_K_PER_TOKEN_SCALE_DIM = 3UL;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr size_t MX_SPLIT_M_SCALE_DIM = 4UL;
constexpr size_t MX_X_DIM = 2UL;
constexpr size_t MX_X_SCALE_DIM = 3UL;
constexpr size_t MX_WEIGHT_DIM = 3UL;
constexpr size_t MX_WEIGHT_SCALE_DIM = 4UL;
constexpr size_t MX_OUTPUT_DIM = 2UL;
constexpr size_t MX_OUTPUT_SCALE_DIM = 3UL;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2L;
constexpr int64_t SWIGLU_SPLIT_SIZE = 64L;
constexpr int64_t MXFP4_K_CONSTRAINT = 2L;
constexpr int64_t SWIGLU_N_CONSTRAINT = 2L;
constexpr int64_t MXFP4_N_CONSTRAINT = 4L;

const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                              DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_MXFP4 = {DataType::DT_FLOAT4_E1M2,
                                                                    DataType::DT_FLOAT4_E2M1};                                                                
const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                                   DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST_MXFP4 = {DataType::DT_FLOAT4_E1M2,
                                                                         DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                                     DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4 = {DataType::DT_FLOAT8_E4M3FN,
                                                                           DataType::DT_FLOAT8_E5M2,
                                                                           DataType::DT_FLOAT4_E1M2,
                                                                           DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
class GroupedMatmulSwigluQuantBaseHandler : public GroupedMatmulSwigluQuantHandler {
protected:

    bool IsTransposeForMxShape(const aclTensor *tensor) const
    {
        auto shape = tensor->GetViewShape();
        if (shape.GetDimNum() < MX_SPLIT_K_PER_TOKEN_SCALE_DIM) {
            return false;
        }
        int64_t firstLastDim = shape.GetDimNum() - 1;
        int64_t secondLastDim = shape.GetDimNum() - LAST_SECOND_DIM_INDEX;
        int64_t thirdLastDim = shape.GetDimNum() - LAST_THIRD_DIM_INDEX;
        auto strides = tensor->GetViewStrides();
        if (strides[firstLastDim] == 1 && strides[thirdLastDim] == MXFP_MULTI_BASE_SIZE &&
            strides[secondLastDim] == shape.GetDim(thirdLastDim) * MXFP_MULTI_BASE_SIZE) {
            return true;
        }
        return false;
    }

    bool IsTransposeLastTwoDims(const aclTensor *tensor) const
    {
        auto shape = tensor->GetViewShape();
        int64_t dim1 = shape.GetDimNum() - 1;
        int64_t dim2 = shape.GetDimNum() - 2;
        auto strides = tensor->GetViewStrides();
        if (strides[dim2] == 1 && strides[dim1] == shape.GetDim(dim2)) {
            int64_t tmpNxD = shape.GetDim(dim1) * shape.GetDim(dim2);
            for (int64_t batchDim = shape.GetDimNum() - 3; batchDim >= 0; batchDim--) {
                if (strides[batchDim] != tmpNxD) {
                    return false;
                }
                tmpNxD *= shape.GetDim(batchDim);
            }
            return true;
        }
        return false;
    }

    void CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                                aclOpExecutor *executor) const
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            shape.SetScalar();
            if (viewShape.GetDimNum() < MX_SPLIT_M_SCALE_DIM) {
                continue;
            }
            shape.AppendDim(viewShape.GetDim(0));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_SECOND_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_THIRD_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - 1));
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            newTensorList.emplace_back(tensor);
        }
    }

    void CreateContiguousTensorList(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                    aclOpExecutor *executor) const
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            uint32_t viewShapeDimsNum = viewShape.GetDimNum();
            shape.SetScalar();
            // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
            for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
                shape.AppendDim(viewShape.GetDim(i));
            }
            // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second last
            // dim.
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2)); // 2:the second last dim.
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            newTensorList.emplace_back(tensor);
        }
    }

    bool CheckMXTranspose()
    {
        // 判断weight和weightScale是否转置，是则对两者进行转置动作
        bool transposeWeightScale = IsTransposeForMxShape((*gmmDsqParams_.weightScale)[0]);
        bool transposeWeight = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[0]);
        bool transposeX = IsTransposeLastTwoDims(gmmDsqParams_.x);
        bool transposeXScale = IsTransposeLastTwoDims(gmmDsqParams_.xScale);

        if (transposeWeightScale != transposeWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The transposition of weightScale/weight should be equal, but actual transpositions are %s/%s.",
                    transposeWeightScale ? "true" : "false", transposeWeight ? "true" : "false");
            return false;
        }
        
        if (transposeX || transposeXScale) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The transposition of x/xScale should be false, but actual transposition are %s/%s.",
                    transposeX ? "true" : "false", transposeXScale ? "true" : "false");
            return false;
        }

        if (transposeWeightScale && transposeWeight) {
            gmmDsqParams_.transposeWeight = true;
            auto uniqueExecutor = CREATE_EXECUTOR();
            CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            aclOpExecutor *executorPtr = uniqueExecutor.get();
            CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            std::vector<aclTensor *> scaleTensorList;
            std::vector<aclTensor *> weightTensorList;
            CreateContiguousTensorListForMXTypeMScale(gmmDsqParams_.weightScale, scaleTensorList, executorPtr);
            gmmDsqParams_.weightScale = executorPtr->AllocTensorList(scaleTensorList.data(), scaleTensorList.size());
            CreateContiguousTensorList(gmmDsqParams_.weight, weightTensorList, executorPtr);
            gmmDsqParams_.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
            uniqueExecutor.ReleaseTo(executor_);
        }
        return true;
    }

    bool CheckFp8DtypeValid()
    {
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor* weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor* weight = (*gmmDsqParams_.weight)[i];      
            OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.x, X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.output, QUANTOUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
        return true;
    }

    bool CheckFp4DtypeValid()
    {
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor* weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor* weight = (*gmmDsqParams_.weight)[i];      
            OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST_MXFP4, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.x, X_DTYPE_SUPPORT_LIST_MXFP4, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.output, QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
        return true;
    }

    bool checkMxfp4InputShape()
    {   
        int64_t kValue = gmmDsqParams_.x->GetViewShape().GetDim(1);
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t nValue = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                         ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2);
        //mxfp4场景不支持k=2
        CHECK_COND(
            kValue != MXFP4_K_CONSTRAINT, ACLNN_ERR_PARAM_INVALID,
            "When the dtypes of x and weight inputs are fp4 , the K value should  be greater than 2, but actual \
value is %lu",
                     kValue);
        //1：检查K是否为偶数
        int64_t kModValue = kValue % MXFP4_K_CONSTRAINT;
        //2：检查N是否为偶数
        int64_t nModValue = nValue % MXFP4_N_CONSTRAINT;
        CHECK_COND(kModValue == 0, ACLNN_ERR_PARAM_INVALID,
                    "When the dtypes of x and weight inputs are fp4 , the K value should be even, but actual \
value is %lu",
                     kValue);
        
        DataType outputDtype = gmmDsqParams_.output->GetDataType();
        if ((outputDtype == DataType::DT_FLOAT4_E1M2 || outputDtype == DataType::DT_FLOAT4_E2M1)) {
            if ( !(nValue >= MXFP4_N_CONSTRAINT && nModValue == 0)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, 
                    "When the output dtype is DT_FLOAT4_E1M2 or DT_FLOAT4_E2M1, the N value should be even and greater or equal to 4, but actual  \
value is %lu",
                    nValue);
                return false;
                }
        }

        return true;
    }

    bool CheckInputOutDims() override
    {
        auto xDimNumber = gmmDsqParams_.x->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmDsqParams_.xScale->GetViewShape().GetDimNum();
        auto outputDimNumber = gmmDsqParams_.output->GetViewShape().GetDimNum();
        auto outputScaleDimNumber = gmmDsqParams_.outputScale->GetViewShape().GetDimNum();
        CHECK_COND(xDimNumber == MX_X_DIM, ACLNN_ERR_PARAM_INVALID,
                        "The dim num of x should be equal 2, current dim is %lu", xDimNumber);
        CHECK_COND(xScaleDimNumber == MX_X_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
                        "The dim num of xScale should be equal 3, current dim is %lu", xScaleDimNumber);
        CHECK_COND(outputDimNumber == MX_OUTPUT_DIM, ACLNN_ERR_PARAM_INVALID,
                        "The dim num of output should be equal 2, current dim is %lu", outputDimNumber);
        CHECK_COND(outputScaleDimNumber == MX_OUTPUT_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
                        "The dim num of outputScale should be equal 3, current dim is %lu", outputScaleDimNumber);
        CHECK_COND(gmmDsqParams_.weight->Size() == gmmDsqParams_.weightScale->Size(), ACLNN_ERR_PARAM_INVALID,
                        "The size of weightScale should be equal to weight");
        for (size_t i = 0; i < gmmDsqParams_.weight->Size(); i++) {
            auto weightDimNumber = ((*gmmDsqParams_.weight)[i])->GetViewShape().GetDimNum();
            auto weightScaleDimNumber = ((*gmmDsqParams_.weightScale)[i])->GetViewShape().GetDimNum();
            CHECK_COND(weightScaleDimNumber == MX_WEIGHT_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
                        "The dim num of weightScale should be equal 4, current dim is %lu", weightScaleDimNumber);
            CHECK_COND(weightDimNumber == MX_WEIGHT_DIM, ACLNN_ERR_PARAM_INVALID,
                        "The dim num of weight should be equal 3, current dim is %lu", weightDimNumber);
        }
        return true;
    }

    bool CheckInputOutShape() override
    {
        CHECK_COND(CheckMXTranspose(), ACLNN_ERR_PARAM_INVALID, "CheckMXTranspose failed");
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0); // 从x的第0维获取m
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1); // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) : 
                                                    ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2);
        int64_t e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0); // 从weight的第0维获取e

        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[M, CeilDiv(K, 64), 2]
        op::Shape xScaleExpectShape = {m, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        // weight的shape期望为[E, K, N]
        op::Shape weightExpectShape = {e, k, n};
        // weightScale的shape期望为[E, CeilDiv(K, 64), N, 2]
        op::Shape weightScaleExpectShape = {e, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), n, SWIGLU_SPLIT_FACTOR};
        // weight转置的shape期望为[E, N, K]
        op::Shape weightTransExpectShape = {e, n, k};
        // weightScale转置的shape期望为[E, N, CeilDiv(K, 64), 2]
        op::Shape weightScaleTransExpectShape = {e, n, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        int64_t nAfterHalve = static_cast<int64_t>(n / SWIGLU_SPLIT_FACTOR);
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M, CeilDiv(N / 2, 64), 2]
        op::Shape outputScaleExpectShape = {m, Ops::Base::CeilDiv(nAfterHalve, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};

        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.x, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.xScale, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.output, outputExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.outputScale, outputScaleExpectShape, return false);

        const aclTensor* weightScale = (*gmmDsqParams_.weightScale)[0];
        const aclTensor* weight = (*gmmDsqParams_.weight)[0];
        if (gmmDsqParams_.transposeWeight) {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleTransExpectShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightTransExpectShape, return false);
        } else {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleExpectShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightExpectShape, return false);
        }

        //进行swiglu操作需满足n为偶数
        if (n % SWIGLU_N_CONSTRAINT != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Swiglu operation requires n to be even , but n actual value is %lu", n);
            return false;
        }
        
        // groupList的长度应等于weight的专家数
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen != e) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Length of 'groupList' should be equal to the number of experts in weight");
            return false;
        }

        DataType xDtype = gmmDsqParams_.x->GetDataType();
        DataType weightDtype = ((*gmmDsqParams_.weight)[0])->GetDataType();  
        if ((xDtype == DataType::DT_FLOAT4_E2M1 || xDtype == DataType::DT_FLOAT4_E1M2) &&
                   (weightDtype == DataType::DT_FLOAT4_E2M1 || weightDtype == DataType::DT_FLOAT4_E1M2)) {
            return checkMxfp4InputShape();
        }
        return true;
    }

    bool CheckDtypeValid() override
    {
        DataType xDtype = gmmDsqParams_.x->GetDataType();
        DataType weightDtype = ((*gmmDsqParams_.weight)[0])->GetDataType();   
        if ((xDtype == DataType::DT_FLOAT8_E4M3FN || xDtype == DataType::DT_FLOAT8_E5M2) &&
                   (weightDtype == DataType::DT_FLOAT8_E4M3FN || weightDtype == DataType::DT_FLOAT8_E5M2)) {
            return CheckFp8DtypeValid();
        } else if ((xDtype == DataType::DT_FLOAT4_E2M1 || xDtype == DataType::DT_FLOAT4_E1M2) &&
                   (weightDtype == DataType::DT_FLOAT4_E2M1 || weightDtype == DataType::DT_FLOAT4_E1M2)) {
            return CheckFp4DtypeValid();
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Quant case with x dtype %s and weight dtype %s is not supported.",
                    op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
            return false;
        }
        return true;
    }

    bool CheckFormat() override
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor* weight = (*gmmDsqParams_.weight)[i];
            if (op::IsPrivateFormat(weight->GetStorageFormat())) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of weight should be ND, current format is format is %s",
                        op::ToString(weight->GetStorageFormat()).GetString());
                return false;
            }
            if (op::IsPrivateFormat(weightScale->GetStorageFormat())) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of weightScale should be ND, current format is format is %s",
                        op::ToString(weightScale->GetStorageFormat()).GetString());
                return false;
            }
        }
        
        if (op::IsPrivateFormat(gmmDsqParams_.x->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of x should be ND, current format is format is %s",
               op::ToString(gmmDsqParams_.x->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.xScale->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of xScale should be ND, current format is format is %s",
               op::ToString(gmmDsqParams_.xScale->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.groupList->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of groupList should be ND, current format is format is %s",
               op::ToString(gmmDsqParams_.groupList->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.output->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of output should be ND, current format is format is %s",
               op::ToString(gmmDsqParams_.output->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.outputScale->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of outputScale should be ND, current format is format is %s",
               op::ToString(gmmDsqParams_.outputScale->GetStorageFormat()).GetString());
            return false;
        }
        return true;
    }
};
}
#endif