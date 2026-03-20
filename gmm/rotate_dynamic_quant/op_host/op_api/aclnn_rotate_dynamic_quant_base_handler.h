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
const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16
};

const std::initializer_list<DataType> ROT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16
};

const std::initializer_list<DataType> YOUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8, DataType::DT_INT32
};

const std::initializer_list<DataType> SCALEOUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

/**
 * @brief 基础 Handler 实现类（DAV_2201 版本）
 */
class RotateDynamicQuantBaseHandler : public rotate_dynamic_quant::RotateDynamicQuantHandler {
protected:
    bool CheckInputOutDims() override
    {
        // 检查 x 维度为 2
        auto xDimNum = params_.x->GetViewShape().GetDimNum();
        if (xDimNum != X_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x tensor dim num must be %zu, but got %zu",
                    X_DIM_LIMIT, xDimNum);
            return false;
        }

        // 检查 rot 维度为 2
        auto rotDimNum = params_.rot->GetViewShape().GetDimNum();
        if (rotDimNum != ROT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "rot tensor dim num must be %zu, but got %zu",
                    ROT_DIM_LIMIT, rotDimNum);
            return false;
        }

        // 检查 yOut 维度为 2
        auto yOutDimNum = params_.yOut->GetViewShape().GetDimNum();
        if (yOutDimNum != YOUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "yOut tensor dim num must be %zu, but got %zu",
                    YOUT_DIM_LIMIT, yOutDimNum);
            return false;
        }

        // 检查 scaleOut 维度为 1
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

        int64_t xM = xShape.GetDim(DIM_IDX_0);
        int64_t xN = xShape.GetDim(DIM_IDX_1);

        int64_t rotN0 = rotShape.GetDim(DIM_IDX_0);
        int64_t rotN1 = rotShape.GetDim(DIM_IDX_1);

        int64_t yOutM = yOutShape.GetDim(DIM_IDX_0);
        int64_t yOutN = yOutShape.GetDim(DIM_IDX_1);

        int64_t scaleOutM = scaleOutShape.GetDim(DIM_IDX_0);

        // rot 的两个维度必须相同 [N, N]
        if (rotN0 != rotN1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "rot tensor must be square matrix, but got [%ld, %ld]",
                    rotN0, rotN1);
            return false;
        }

        // x 的第二维 N 必须等于 rot 的维度
        if (xN != rotN0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x second dim (%ld) must equal rot dim (%ld)",
                    xN, rotN0);
            return false;
        }

        // yOut 的 shape 必须与 x 一致
        if (yOutM != xM || yOutN != xN) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "yOut shape [%ld, %ld] must equal x shape [%ld, %ld]",
                    yOutM, yOutN, xM, xN);
            return false;
        }

        // scaleOut 的第一维必须等于 x 的第一维 M
        if (scaleOutM != xM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "scaleOut first dim (%ld) must equal x first dim (%ld)",
                    scaleOutM, xM);
            return false;
        }

        return true;
    }

    bool CheckDtypeValid() override
    {
        // 检查输入数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.x, X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.rot, ROT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.yOut, YOUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.scaleOut, SCALEOUT_DTYPE_SUPPORT_LIST, return false);

        // x 和 rot 数据类型必须一致
        auto xDtype = params_.x->GetDataType();
        auto rotDtype = params_.rot->GetDataType();

        if (xDtype != rotDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x dtype (%s) must equal rot dtype (%s)",
                    ToString(xDtype).GetString(),
                    ToString(rotDtype).GetString());
            return false;
        }

        // 根据 dst_type 检查 yOut 数据类型
        auto yOutDtype = params_.yOut->GetDataType();
        if (params_.dst_type == 0 && yOutDtype != DataType::DT_INT8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when dst_type=0, yOut dtype must be DT_INT8, but got %s",
                    ToString(yOutDtype).GetString());
            return false;
        }
        if (params_.dst_type == 1 && yOutDtype != DataType::DT_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when dst_type=1, yOut dtype must be DT_INT32, but got %s",
                    ToString(yOutDtype).GetString());
            return false;
        }

        // scaleOut 必须是 DT_FLOAT
        auto scaleOutDtype = params_.scaleOut->GetDataType();
        if (scaleOutDtype != DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "scaleOut dtype must be DT_FLOAT, but got %s",
                    ToString(scaleOutDtype).GetString());
            return false;
        }

        return true;
    }

    bool CheckFormat() override
    {
        // 检查 format，支持非连续tensor，但需要检查是否为私有 format
        if (IsPrivateFormat(params_.x->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x tensor format must be ND, current format is %s",
                    ToString(params_.x->GetStorageFormat()).GetString());
            return false;
        }

        if (IsPrivateFormat(params_.rot->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "rot tensor format must be ND, current format is %s",
                    ToString(params_.rot->GetStorageFormat()).GetString());
            return false;
        }

        if (IsPrivateFormat(params_.yOut->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "yOut tensor format must be ND, current format is %s",
                    ToString(params_.yOut->GetStorageFormat()).GetString());
            return false;
        }

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
