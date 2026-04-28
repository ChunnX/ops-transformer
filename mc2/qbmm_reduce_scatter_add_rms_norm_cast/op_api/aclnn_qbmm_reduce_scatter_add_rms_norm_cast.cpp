/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_qbmm_reduce_scatter_add_rms_norm_cast.h"
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
enum class NnopbaseHcclServerType : uint32_t{
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// check nullptr
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* y,
                         const aclTensor* gamma, const aclTensor* scale, const aclTensor* bias, 
                         const aclTensor* pertokenScale, aclTensor* y1, aclTensor* y2, aclTensor* x)
{
    OP_LOGD("aclnn_qbmm_reduce_scatter_add_rms_norm_cast CheckNotNull start");
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(bias, return false);  
    OP_CHECK_NULL(pertokenScale, return false);
    OP_CHECK_NULL(y1, return false);
    OP_CHECK_NULL(y2, return false);
    OP_CHECK_NULL(x, return false);
    OP_LOGD("aclnn_qbmm_reduce_scatter_add_rms_norm_cast CheckNotNull success");
    return true;
}

// 入参教验
static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* y,
                               const aclTensor* gamma, const aclTensor* scale, const aclTensor* bias, 
                               const aclTensor* pertokenScale, aclTensor* y1, aclTensor* y2, aclTensor* x)
{
    OP_LOGD("aclnn_qbmm_reduce_scatter_add_rms_norm_cast CheckParams start");
    CHECK_RET(CheckNotNull(x1, x2, y, gamma, scale, bias, pertokenScale, y1, y2, x),
        ACLNN_ERR_PARAM_NULLPTR);
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

extern "C" aclnnStatus aclnnInnerQbmmReduceScatterAddRmsNormCastGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* y, const aclTensor* gamma,
    const aclTensor* scale, const aclTensor* bias, const aclTensor* pertokenScale, 
    const char* group, int64_t rankSize, bool transposeX2, int64_t dtype, float epsilon,
    aclTensor* y1, aclTensor* y2, aclTensor* x, 
    uint64_t* workspaceSize, aclOpExecutor** executor);

extern "C" aclnnStatus aclnnInnerQbmmReduceScatterAddRmsNormCast(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* y, 
    const aclTensor* gamma, const aclTensor* scale, const aclTensor* bias, const aclTensor* pertokenScale, 
    const char* group, int64_t rankSize, bool transposeX2, int64_t dtype, float epsilon, 
    aclTensor* y1, aclTensor* y2, aclTensor* x, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize start");
    aclnnStatus retParam = CheckParams(x1, x2, y, gamma, scale, bias, pertokenScale, y1, y2, x);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // x2 格式变化
    auto tempX2 = x2;
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ) {
        tempX2 = CopyTensor(x2);
    }
    OP_LOGD("aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize Inner start");
    aclnnStatus ret = aclnnInnerQbmmReduceScatterAddRmsNormCastGetWorkspaceSize(
        x1, tempX2, y, gamma, scale, bias, pertokenScale, group, rankSize, 
        transposeX2, dtype, epsilon, y1, y2, x, workspaceSize, executor);
    OP_LOGD("aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize Inner end");
    return ret;
}

extern "C" aclnnStatus aclnnQbmmReduceScatterAddRmsNormCast(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    OP_LOGD("aclnnQbmmReduceScatterAddRmsNormCast start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    OP_LOGD("aclnnQbmmReduceScatterAddRmsNormCast Inner start");
    aclnnStatus ret = aclnnInnerQbmmReduceScatterAddRmsNormCast(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("aclnnQbmmReduceScatterAddRmsNormCast Inner End");
    return ACLNN_SUCCESS;
}
