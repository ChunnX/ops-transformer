/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#include <algorithm>

#include "securec.h"
#include "common/utils/op_mc2.h"
#include "acl/acl.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "common/utils/hccl_util.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "aclnn_allto_allv_quant_grouped_mat_mul_v2_helper.h"

namespace {
using namespace op;
using namespace A2AVQuantGmmHelper;  // 复用V2 helper中的公共函数

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
};

// 需要使用的常量定义
static constexpr size_t TWO_DIMS = 2U;
static constexpr size_t THREE_DIMS = 3U;

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype, int64_t mmDtype,
    const char* commMode,
    aclTensor *gmmY, aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMulV2(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

extern "C" aclnnStatus InnerAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, const char* commMode,
    aclTensor *gmmY, aclTensor *mmYOptional,
    aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    int64_t yDtype = gmmY->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();

    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale, sendCountsTensorOptional, recvCountsTensorOptional,
        mmXOptional, mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, group, epWorldSize, sendCounts,
        recvCounts, gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag,
        mmXQuantMode, mmWeightQuantMode, groupSize, yDtype, mmDtype, commMode,
        gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, int64_t groupSize, bool permuteOutFlag, const char* commMode,
    aclTensor *gmmY, aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    using namespace A2AVQuantGmmHelper;
    CHECK_RET(CheckGmmWeightValid(gmmWeight), ACLNN_ERR_PARAM_NULLPTR);
    bool notContiguous = IsTransposeLastTwoDims(gmmWeight);
    auto transposeGmmWeight = gmmWeight;
    if (notContiguous && transGmmWeight) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight not contiguous, and set gmmWeight transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        transGmmWeight = !transGmmWeight;
        transposeGmmWeight = TransGmmWeightTensor(gmmWeight);
        CHECK_RET(transposeGmmWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("gmmWeight is a non-contiguous tensor. The original dim1 is %ld, and dim2 is %ld. After processing, transposeGmmWeight dim1 is %ld, and dim2 is %ld.",
            gmmWeight->GetViewShape().GetDim(1), gmmWeight->GetViewShape().GetDim(2), transposeGmmWeight->GetViewShape().GetDim(1), transposeGmmWeight->GetViewShape().GetDim(2));
    }

    auto transposeGmmWeightScale = gmmWeightScale;
    if (notContiguous && gmmWeightScale != nullptr && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        transposeGmmWeightScale = TransGmmWeightScaleTensor(gmmWeightScale);
        CHECK_RET(transposeGmmWeightScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *originalMmWeightOptional = mmWeightOptional;
    const aclTensor *originalMmWeightScaleOptional = mmWeightScaleOptional;
    if (CheckMmWeightValid(mmWeightOptional)) {
        bool notContiguous = IsTransposeLastTwoDims(mmWeightOptional);
        auto transMmWeightOptional = mmWeightOptional;
        if (notContiguous && transMmWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeightOptional not contiguous, and set mmWeightOptional transpose, it is error!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            transMmWeight = !transMmWeight;
            transMmWeightOptional = TransMmWeightOptionalTensor(mmWeightOptional);
            CHECK_RET(transMmWeightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            OP_LOGD("mmWeightOptional is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. After processing, transMmWeightOptional dim0 is %ld, and dim1 is %ld.",
                mmWeightOptional->GetViewShape().GetDim(0), mmWeightOptional->GetViewShape().GetDim(1),
                transMmWeightOptional->GetViewShape().GetDim(0), transMmWeightOptional->GetViewShape().GetDim(1));
            mmWeightOptional = transMmWeightOptional;
        }
    }

    auto transMmWeightScaleOptional = mmWeightScaleOptional;
    if (mmWeightScaleOptional != nullptr && mmWeightScaleOptional->GetViewShape().GetDimNum() >= 3 &&
        GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        bool mmWeightScaleNotContiguous = IsTransposeTwoDims(mmWeightScaleOptional, -2, -3);
        if (mmWeightScaleNotContiguous) {
            transMmWeightScaleOptional = TransMmWeightScaleTensor(mmWeightScaleOptional);
            CHECK_RET(transMmWeightScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    aclnnStatus ret_consistency = CheckWeightScaleTransposeConsistency(
        gmmWeight, gmmWeightScale, originalMmWeightOptional, originalMmWeightScaleOptional);
    CHECK_RET(ret_consistency == ACLNN_SUCCESS, ret_consistency);

    aclnnStatus ret_param = CheckParams(
        gmmX, transposeGmmWeight, gmmXScale, transposeGmmWeightScale,
        sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional,
        transMmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = InnerAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
        gmmX, transposeGmmWeight, gmmXScale, transposeGmmWeightScale, sendCountsTensorOptional, recvCountsTensorOptional,
        mmXOptional, mmWeightOptional, mmXScaleOptional, transMmWeightScaleOptional, group, epWorldSize, sendCounts,
        recvCounts, gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag, mmXQuantMode,
        mmWeightQuantMode, groupSize, commMode, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);

    if (ret == ACLNN_SUCCESS && *executor != nullptr) {
        char* commModePtr = new char[strlen(commMode) + 1];
        if (commModePtr != nullptr) {
            strcpy(commModePtr, commMode);
            NnopbaseSetUserHandle(*executor, commModePtr);
        }
    }
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    char* commModePtr = reinterpret_cast<char*>(NnopbaseGetUserHandle(executor));
    if (NnopbaseSetHcclServerType) {
        if (commModePtr != nullptr) {
            if (strcmp(commModePtr, "aicpu") == 0) {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
            delete[] commModePtr;
        } else {
            // fallback: V2只支持950，默认设置CCU
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMulV2(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace
