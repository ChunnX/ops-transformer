/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_ROTATE_DYNAMIC_QUANT_BASE_HANDLER_H
#define OP_HOST_OP_API_ACLNN_ROTATE_DYNAMIC_QUANT_BASE_HANDLER_H

#include "aclnn_rotate_dynamic_quant_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

namespace rotate_dynamic_quant_base {

using namespace op;

/**
 * @brief SOC 特定常量定义
 */
constexpr int64_t DIM_IDX_0 = 0L;
constexpr int64_t DIM_IDX_1 = 1L;
constexpr size_t X_DIM_LIMIT = 2UL;
constexpr size_t ROT_DIM_LIMIT = 2UL;
constexpr size_t YOUT_DIM_LIMIT = 2UL;
constexpr size_t SCALEOUT_DIM_LIMIT = 1UL;

/**
 * @brief 支持的数据类型列表
 */
const std::initializer_list<DataType> X_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16
};

const std::initializer_list<DataType> ROT_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16
};

const std::initializer_list<DataType> YOUT_OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8, DataType::DT_INT32
};

const std::initializer_list<DataType> SCALEOUT_OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

/**
 * @brief 基础 Handler 实现类（DAV_2201 版本）
 */
class RotateDynamicQuantBaseHandler : public rotate_dynamic_quant::RotateDynamicQuantHandler {
protected:
    bool CheckInputOutDims() override
    {
        // 检查x维度为2
        auto xDimNum = params_.x->GetViewShape().GetDimNum();
        if (xDimNum != X_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x tensor dim num must be %zu, but got %zu",
                    X_DIM_LIMIT, xDimNum);
            return false;
        }

        // 检查rot维度为2
        auto rotDimNum = params_.rot->GetViewShape().GetDimNum();
        if (rotDimNum != ROT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "rot tensor dim num must be %zu, but got %zu",
                    ROT_DIM_LIMIT, rotDimNum);
            return false;
        }

        // 检查yOut维度为2
        auto yOutDimNum = params_.yOut->GetViewShape().GetDimNum();
        if (yOutDimNum != YOUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "yOut tensor dim num must be %zu, but got %zu",
                    YOUT_DIM_LIMIT, yOutDimNum);
            return false;
        }

        // 检查scaleOut维度为1
        auto scaleOutDimNum = params_.scaleOut->GetViewShape().GetDimNum();
        if (scaleOutDimNum != SCALEOUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "scaleOut tensor dim num must be %zu, but got %zu",
                    SCALEOUT_DIM_LIMIT, scaleOutDimNum);
            return false;
        }

        return true;
    }

    bool CheckInputOutShape() override
    {
        auto xShape = params_.x->GetViewShape();
        auto rotShape = params_.rot->GetViewShape();
        auto yOutShape = params_.yOut->GetViewShape();
        auto scaleOutShape = params_.scaleOut->GetViewShape();

        int64_t xDim0 = xShape.GetDim(DIM_IDX_0);  // M
        int64_t xDim1 = xShape.GetDim(DIM_IDX_1);  // N
        int64_t rotDim0 = rotShape.GetDim(DIM_IDX_0);  // N
        int64_t rotDim1 = rotShape.GetDim(DIM_IDX_1);  // N

        // rot的两个维度必须相等且等于x的第二维N
        if (rotDim0 != rotDim1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "rot tensor dimensions must be equal, but got [%ld, %ld]",
                    rotDim0, rotDim1);
            return false;
        }

        if (rotDim0 != xDim1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "rot dimension (%ld) must equal x second dimension (%ld)",
                    rotDim0, xDim1);
            return false;
        }

        // yOut shape必须与x一致 [M, N]
        int64_t yOutDim0 = yOutShape.GetDim(DIM_IDX_0);
        int64_t yOutDim1 = yOutShape.GetDim(DIM_IDX_1);
        if (yOutDim0 != xDim0 || yOutDim1 != xDim1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "yOut shape [%ld, %ld] must equal x shape [%ld, %ld]",
                    yOutDim0, yOutDim1, xDim0, xDim1);
            return false;
        }

        // scaleOut shape必须为[M]
        int64_t scaleOutDim0 = scaleOutShape.GetDim(DIM_IDX_0);
        if (scaleOutDim0 != xDim0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "scaleOut dimension (%ld) must equal x first dimension (%ld)",
                    scaleOutDim0, xDim0);
            return false;
        }

        return true;
    }

    bool CheckDtypeValid() override
    {
        // 检查x数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.x, X_INPUT_DTYPE_SUPPORT_LIST, return false);

        // 检查rot数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.rot, ROT_INPUT_DTYPE_SUPPORT_LIST, return false);

        // x和rot数据类型必须一致
        auto xDtype = params_.x->GetDataType();
        auto rotDtype = params_.rot->GetDataType();
        if (xDtype != rotDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x dtype (%s) must equal rot dtype (%s)",
                    ToString(xDtype).GetString(),
                    ToString(rotDtype).GetString());
            return false;
        }

        // 检查yOut数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.yOut, YOUT_OUTPUT_DTYPE_SUPPORT_LIST, return false);

        // 检查yOut数据类型与dstType属性一致
        auto yOutDtype = params_.yOut->GetDataType();
        DataType expectedYOutDtype = (params_.dstType == 1) ? DataType::DT_INT32 : DataType::DT_INT8;
        if (yOutDtype != expectedYOutDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "yOut dtype (%s) must match dstType (%ld), expected %s",
                    ToString(yOutDtype).GetString(),
                    params_.dstType,
                    ToString(expectedYOutDtype).GetString());
            return false;
        }

        // 检查scaleOut数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.scaleOut, SCALEOUT_OUTPUT_DTYPE_SUPPORT_LIST, return false);

        return true;
    }

    bool CheckFormat() override
    {
        // 检查x格式
        if (IsPrivateFormat(params_.x->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x tensor format must be ND, current format is %s",
                    ToString(params_.x->GetStorageFormat()).GetString());
            return false;
        }

        // 检查rot格式
        if (IsPrivateFormat(params_.rot->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "rot tensor format must be ND, current format is %s",
                    ToString(params_.rot->GetStorageFormat()).GetString());
            return false;
        }

        // 检查yOut格式
        if (IsPrivateFormat(params_.yOut->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "yOut tensor format must be ND, current format is %s",
                    ToString(params_.yOut->GetStorageFormat()).GetString());
            return false;
        }

        // 检查scaleOut格式
        if (IsPrivateFormat(params_.scaleOut->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "scaleOut tensor format must be ND, current format is %s",
                    ToString(params_.scaleOut->GetStorageFormat()).GetString());
            return false;
        }

        return true;
    }
};

} // namespace rotate_dynamic_quant_base

#endif // OP_HOST_OP_API_ACLNN_ROTATE_DYNAMIC_QUANT_BASE_HANDLER_H
