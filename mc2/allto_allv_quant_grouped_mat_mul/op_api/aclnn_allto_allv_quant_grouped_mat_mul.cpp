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
#include "aclnn_allto_allv_quant_grouped_mat_mul.h"
#include "../allto_allv_quant_grouped_mat_mul_v2/op_api/aclnn_allto_allv_quant_grouped_mat_mul_v2_helper.h"  // 复用V2公共函数

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

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
                               const aclTensor *gmmWeightScale, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               const char *group, int64_t epWorldSize, bool permuteOutFlag, const aclTensor *gmmY,
                               const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // 检查空状态
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                              mmXScaleOptional, mmWeightScaleOptional, permuteOutFlag, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查参数是否为空
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY, gmmXScale, gmmWeightScale, gmmXQuantMode, gmmWeightQuantMode),
              ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("AlltoAllvQuantGroupedMatmul checkParams success");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype, int64_t mmDtype,
    aclTensor *gmmY, aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, aclTensor *gmmY, aclTensor *mmYOptional,
    aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    int64_t yDtype = gmmY->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();

    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale, sendCountsTensorOptional, recvCountsTensorOptional,
        mmXOptional, mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, group, epWorldSize, sendCounts,
        recvCounts, gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag,
        mmXQuantMode, mmWeightQuantMode, groupSize, yDtype, mmDtype, gmmY, mmYOptional, permuteOutOptional,
        workspaceSize, executor);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, int64_t groupSize, bool permuteOutFlag, aclTensor *gmmY,
    aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前支持转置的gmmWeight涉及该处理
    CHECK_RET(CheckGmmWeightValid(gmmWeight), ACLNN_ERR_PARAM_NULLPTR);	// 先检查gmmWeight是否合法，避免非法操作
    bool notContiguous = IsTransposeTwoDims(gmmWeight, -1, -2);
    auto transposeGmmWeight = gmmWeight;    // 复制一个gmmWeight
    if (notContiguous && transGmmWeight) {    // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight not contiguous, and set gmmWeight transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {    // 只有当非连续时，才会涉及到转连续等情况
        transGmmWeight = !transGmmWeight;
        // 把非连续gmmWeight转成连续
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

    // 保存原始指针用于一致性校验
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

    aclnnStatus ret = InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, transposeGmmWeight, gmmXScale, transposeGmmWeightScale,
        sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        mmXScaleOptional, transMmWeightScaleOptional, group, epWorldSize, sendCounts,
        recvCounts, gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight,
        permuteOutFlag, mmXQuantMode, mmWeightQuantMode, groupSize, gmmY, mmYOptional,
        permuteOutOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace
