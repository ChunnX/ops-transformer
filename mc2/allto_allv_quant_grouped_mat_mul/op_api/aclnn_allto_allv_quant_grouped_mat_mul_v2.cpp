/* *
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "aclnn_allto_allv_quant_grouped_mat_mul_v2.h"
#include "aclnn_allto_allv_quant_grouped_mat_mul_utils.h"

namespace {
using namespace op;
using namespace AlltoAllvQuantGroupedMatMulUtils;

extern "C" aclnnStatus InnerAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, const char *commMode, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, aclTensor *gmmY, aclTensor *mmYOptional,
    aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    char *str_commMode = const_cast<char *>(commMode);
    int64_t yDtype = gmmY->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();

    uint8_t commModeEnum = 0;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode(group, commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);
    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale, sendCountsTensorOptional, recvCountsTensorOptional,
        mmXOptional, mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, group, epWorldSize, sendCounts,
        recvCounts, gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag,
        mmXQuantMode, mmWeightQuantMode, groupSize, yDtype, mmDtype, str_commMode, gmmY, mmYOptional, permuteOutOptional,
        workspaceSize, executor);
    if (*executor != nullptr) {
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(commModeEnum));
        NnopbaseSetUserHandle(executor, args);
    }
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    const char *group, const char *commMode, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, int64_t groupSize, bool permuteOutFlag, aclTensor *gmmY,
    aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckGmmWeightValid(gmmWeight), ACLNN_ERR_PARAM_NULLPTR);
    bool notContiguous = IsTransposeTwoDims(gmmWeight, -1, -2);
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
        bool notContiguous = IsTransposeTwoDims(mmWeightOptional, -1, -2);
        auto transMmWeightOptional = mmWeightOptional;
        int64_t mmDim0 = mmWeightOptional->GetViewShape().GetDim(0);
        int64_t mmDim1 = mmWeightOptional->GetViewShape().GetDim(1);
        int64_t mmStride0 = mmWeightOptional->GetViewStrides()[0];
        int64_t mmStride1 = mmWeightOptional->GetViewStrides()[1];
        if (notContiguous && transMmWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mmWeightOptional not contiguous, and set mmWeightOptional transpose, it is error!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            transMmWeight = !transMmWeight;
            transMmWeightOptional = TransMmWeightOptionalTensor(mmWeightOptional);
            CHECK_RET(transMmWeightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            mmWeightOptional = transMmWeightOptional;
        }
        
    }
    
    auto transMmWeightScaleOptional = mmWeightScaleOptional;
    bool mmWeightScaleNotContiguous = false;
    if (mmWeightScaleOptional != nullptr && mmWeightScaleOptional->GetViewShape().GetDimNum() >= 3 &&
        GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        mmWeightScaleNotContiguous = IsTransposeTwoDims(mmWeightScaleOptional, -2, -3);
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
        mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = InnerAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
        gmmX, transposeGmmWeight, gmmXScale, transposeGmmWeightScale,
        sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        mmXScaleOptional, transMmWeightScaleOptional, group, commMode, epWorldSize, sendCounts,
        recvCounts, gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight,
        permuteOutFlag, mmXQuantMode, mmWeightQuantMode, groupSize, gmmY, mmYOptional,
        permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, InnerAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize ret %d.", ret);
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                OP_LOGD("arch35 with ENV_MC2_COMM_MODE_AICPU, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                OP_LOGD("arch35, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        }
    }
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}
}