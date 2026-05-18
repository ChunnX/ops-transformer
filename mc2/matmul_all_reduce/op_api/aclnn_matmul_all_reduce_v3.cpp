/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul_all_reduce_v3.cpp
 * \brief
 */
#include "aclnn_matmul_all_reduce_v3.h"

#include "aclnnInner_matmul_all_reduce.h"
#include "matmul_all_reduce_util.h"
#include "common/utils/hccl_util.h"
#include "mc2_comm_utils.h"

#ifdef BUILD_OPEN_PROJECT
#include "version/hcomm_version.h"
#define HCCL_CHANNEL_SUPPORT_VERSION 90000000
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif
#endif

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void* executor, const size_t irIndex);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static aclnnStatus InnerMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* group,
    const char* reduceOp, const char* commMode, int64_t commTurn, const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    uint32_t rankSize = 0;
#if defined(BUILD_OPEN_PROJECT) && (HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION)
    aclnnStatus getRankSizeRet = Mc2Aclnn::Mc2Context::GetMatmulAllReduceRankSize(group, rankSize);
    CHECK_RET(getRankSizeRet == ACLNN_SUCCESS, getRankSizeRet);
#endif
    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2);
    aclTensor* pertokenScale = nullptr;
    aclTensor* dequantScale = nullptr;
    aclTensor* commQuantScale1 = nullptr;
    aclTensor* commQuantScale2 = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    int64_t antiquantGroupSize = 0;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, x2, bias, x3, scale, offset, dequantScale, pertokenScale, commQuantScale1, commQuantScale2,
        const_cast<char*>(group), const_cast<char*>(reduceOp),
        transposeX1, transposeX2, commTurn, antiquantGroupSize, 0, yDtype, 0, const_cast<char*>(commMode), output, workspaceSize, executor);
    OP_LOGD("MatmulAllReduceV3, aclnnMatmulAllReduceGetWorkspaceSize ret %d", ret);

    OP_LOGI("Group %s, commMode %s, reduce op %s, trans flag %u %u, ret %d.", group, commMode, reduceOp, transposeX1, transposeX2, ret);
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 4U); // 4 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 6U); // 6 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 8U); // 8 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 9U); // 9 is input irIndex
        }
    }
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            if (strcmp(commMode, COMM_MODE_AICPU) == 0) {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else if (strcmp(commMode, COMM_MODE_CCU) == 0) {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            } else if (strcmp(commMode, COMM_MODE_DEFAULT) == 0) {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        } else {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    return ret;
}

aclnnStatus aclnnMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* group,
    const char* reduceOp, const char* commMode, int64_t commTurn, int64_t streamMode, const aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = MatmulAllReduceCheckParams(x1, x2, x3, bias, reduceOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 检查commMode是否合法
    CHECK_RET(IsCommModeValid(commMode), ACLNN_ERR_PARAM_INVALID);

    aclnnStatus ret = InnerMatmulAllReduceV3GetWorkspaceSize(
        x1, x2, bias, x3, group, reduceOp, commMode, commTurn, output, workspaceSize, executor);
    OP_LOGD("MatmulAllReduceV3, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnMatmulAllReduceV3(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
    OP_API_CHECK(ret != ACLNN_SUCCESS, {
        OP_LOGE(ACLNN_ERR_INNER, "MatmulAllReduceV3, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    });
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif