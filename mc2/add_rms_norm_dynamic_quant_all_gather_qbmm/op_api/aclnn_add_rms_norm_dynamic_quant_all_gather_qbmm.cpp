/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_add_rms_norm_dynamic_quant_all_gather_qbmm.cpp
 * \brief
 */
#include "aclnn_add_rms_norm_dynamic_quant_all_gather_qbmm.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"

using namespace op;

namespace {
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

static constexpr size_t HCCL_GROUP_NAME_LENGTH_MAX = 128U; // group长度小于128字符

// 根据API定义，列出支持的所有dtype
const std::initializer_list<op::DataType> ARN_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16
};
const std::initializer_list<op::DataType> X2_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2
};
const std::initializer_list<op::DataType> GAMMA_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT
};
const std::initializer_list<op::DataType> SCALES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E8M0
};

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* y,
    const aclTensor* gamma, const aclTensor* scale, const aclTensor* output, const aclTensor* z)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(residual, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(z, return false);
    return true;
}

static bool CheckAllDtypesValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* y,
    const aclTensor* gamma, const aclTensor* scale, const aclTensor* output, const aclTensor* z)
{

    if (!CheckType(x1->GetDataType(), ARN_DTYPE_SUPPORT_LIST)
        || !CheckType(residual->GetDataType(), ARN_DTYPE_SUPPORT_LIST)
        || !CheckType(y->GetDataType(), ARN_DTYPE_SUPPORT_LIST)
        || !CheckType(output->GetDataType(), ARN_DTYPE_SUPPORT_LIST)
        || !CheckType(z->GetDataType(), ARN_DTYPE_SUPPORT_LIST)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1, residual, y, output, and z support only [DT_BF16, DT_FLOAT16]."
                "While tensors x1: %s, residual: %s, y: %s, output: %d, and z:%d are not simultaneously supported.",
                op::ToString(x1->GetDataType()).GetString(),
                op::ToString(residual->GetDataType()).GetString(),
                op::ToString(y->GetDataType()).GetString(),
                op::ToString(output->GetDataType()).GetString(),
                op::ToString(z->GetDataType()).GetString());
            return false;
    }

    if (!CheckType(x2->GetDataType(), X2_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 support only [DT_INT8]."
            "While tensor x2: %s is not supported.", op::ToString(x2->GetDataType()).GetString());
        return false;
    }

    if (!CheckType(gamma->GetDataType(), GAMMA_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma support only [DT_FLOAT]."
            "While tensor gamma: %s is not supported.", op::ToString(gamma->GetDataType()).GetString());
        return false;
    }

    if (!CheckType(scale->GetDataType(), SCALES_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale support only [DT_BF16, DT_FLOAT]."
            "While tensor scale: %s is not supported.", op::ToString(scale->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckGroupLength(const char* group)
{
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AddRmsNormDynamicQuantAllGatherQbmm, group is nullptr !");
        return false;
    }

    size_t groupLen = strnlen(group, HCCL_GROUP_NAME_LENGTH_MAX); // group长度≥128字符, 返回HCCL_GROUP_NAME_LENGTH_MAX
    if (groupLen >= HCCL_GROUP_NAME_LENGTH_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AddRmsNormDynamicQuantAllGatherQbmm, Limit the length of the group to less than %lu characters.",
                HCCL_GROUP_NAME_LENGTH_MAX);
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* y,
    const aclTensor* gamma, const aclTensor* scale, const aclTensor* output, const aclTensor* z, const char* group)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, residual, y, gamma, scale, output, z), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckAllDtypesValid(x1, x2, residual, y, gamma, scale, output, z), ACLNN_ERR_PARAM_INVALID);
    // // 3. 检查group参数是否在要求范围之内
    CHECK_RET(CheckGroupLength(group), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* CopyTensor(const aclTensor* x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (size_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm, CopyTensor storageDimsNum is %lu.", storageDimsNum);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    auto stride = x2->GetViewStrides();
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    auto stgFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    if (stgFormat == Format::FORMAT_ND) {
        OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm, CopyTensor format is ACL_FORMAT_ND");
        format = aclFormat::ACL_FORMAT_ND;
    } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
        format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    return aclCreateTensor(
        storageDims.data(), storageDimsNum, dataType, stride.data(), offset, format, storageDims.data(), storageDimsNum,
        x2->GetTensor()->GetAddr());
}
}

extern "C" aclnnStatus aclnnInnerAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* y, const aclTensor* gamma,
    const aclTensor* scaleInv1, const aclTensor* scaleInv2, const aclTensor* smoothScale, const aclTensor* bias, const char* group, int64_t rankSize,
    bool transposeX2, int64_t dtype, int64_t residualNormMode, bool outputOptional, aclTensor* output, aclTensor* z,
    aclTensor* dynamicQuantOut, aclTensor* allGatherDataOut, aclTensor* allGatherScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern "C" aclnnStatus aclnnInnerAddRmsNormDynamicQuantAllGatherQbmm(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* y, const aclTensor* gamma,
    const aclTensor* scaleInv1, const aclTensor* scaleInv2, const aclTensor* smoothScale, const aclTensor* bias, const char* group, int64_t rankSize,
    bool transposeX2, int64_t dtype, int64_t residualNormMode, bool outputOptional, aclTensor* output, aclTensor* z,
    aclTensor* dynamicQuantOut, aclTensor* allGatherDataOut, aclTensor* allGatherScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // aclnnStatus retParam = CheckParams(x1, x2, residual, y, gamma, scale, output, z, group);
    // CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // x2 格式变化
    auto tempX2 = x2;
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ) {
        tempX2 = CopyTensor(x2);
    }
    OP_LOGD("Invoking aclnnInnerAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize...");
    aclnnStatus ret = aclnnInnerAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize(
        x1, tempX2, residual, y, gamma, scaleInv1, scaleInv2, smoothScale, bias, group, rankSize, transposeX2, dtype, residualNormMode,
        outputOptional, output, z, dynamicQuantOut, allGatherDataOut, allGatherScalesOut, workspaceSize, executor);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm, aclnnGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAddRmsNormDynamicQuantAllGatherQbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerAddRmsNormDynamicQuantAllGatherQbmm(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}