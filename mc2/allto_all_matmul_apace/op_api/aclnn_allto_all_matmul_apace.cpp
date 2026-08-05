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
 * \file aclnn_allto_all_matmul_apace.cpp
 * \brief aclnn API implementation for AlltoAllMatmulApace (MX-quant FP8, apace UDMA path)
 */

#include "aclnn_allto_all_matmul_apace.h"

#include "common/utils/matmul_util.h"
#include "mc2/matmul_allto_all/op_api/matmul_allto_all_util.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnInnerAlltoAllMatmulApaceGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *commScaleOptional,
    const aclTensor *x1OffsetOptional, const aclTensor *x2OffsetOptional, char *group, int64_t worldSize,
    int64_t hcclBufferSize, const aclIntArray *all2allAxesOptional, int64_t yDtype, int64_t x1QuantMode,
    int64_t x2QuantMode, int64_t commQuantMode, int64_t x1QuantDtype, int64_t commQuantDtype, bool transposeX1,
    bool transposeX2, int64_t groupSize, bool alltoallOutFlag, char *commModeOptional, int64_t precisionMode,
    const aclTensor *yOut, const aclTensor *all2allOutOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllMatmulApace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

extern "C" aclnnStatus aclnnAlltoAllMatmulApaceGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const aclTensor *commScaleOptional,
    const aclTensor *x1OffsetOptional, const aclTensor *x2OffsetOptional, const char *group, int64_t worldSize,
    int64_t hcclBufferSize, const aclIntArray *alltoAllAxesOptional, int64_t x1QuantMode, int64_t x2QuantMode,
    int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype, bool transposeX1, bool transposeX2,
    int64_t groupSize, const char *commMode, int64_t precisionMode, const aclTensor *output,
    const aclTensor *alltoAllOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnAlltoAllMatmulApaceGetWorkspaceSize start");
    (void)biasOptional;
    (void)commScaleOptional;
    (void)x1OffsetOptional;
    (void)x2OffsetOptional;
    (void)alltoAllAxesOptional;
    (void)alltoAllOutOptional;
    (void)hcclBufferSize;

    if (x1 == nullptr || x2 == nullptr || x1ScaleOptional == nullptr || x2ScaleOptional == nullptr ||
        output == nullptr || workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "aclnnAlltoAllMatmulApace: null pointer in input parameters.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto transX2 = x2;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        bool notContiguous = IsTransposeLastTwoDims(x2);
        if (notContiguous && transposeX2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x2 is non-contiguous but transpose_x2 is true, this is invalid for apace.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous) {
            transposeX2 = true;
            transX2 = matmul_allto_all_check::TransX2Tensor(x2);
            if (transX2 == nullptr) {
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
    }

    constexpr int64_t kDefaultYdtype = 27; // ge::DT_BF16
    constexpr bool kDefaultAlltoallOutFlag = false;
    aclnnStatus ret = aclnnInnerAlltoAllMatmulApaceGetWorkspaceSize(
        context, x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, commScaleOptional, x1OffsetOptional,
        x2OffsetOptional, const_cast<char *>(group), worldSize, hcclBufferSize, alltoAllAxesOptional, kDefaultYdtype,
        x1QuantMode, x2QuantMode, commQuantMode, x1QuantDtype, commQuantDtype, transposeX1, transposeX2, groupSize,
        kDefaultAlltoallOutFlag, const_cast<char *>(commMode ? commMode : "ai_cpu"), precisionMode, output,
        alltoAllOutOptional, workspaceSize, executor);

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAlltoAllMatmulApaceGetWorkspaceSize failed, ret=%d", ret);
    }

    OP_LOGD("aclnnAlltoAllMatmulApaceGetWorkspaceSize end: workspaceSize=%lu", *workspaceSize);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllMatmulApace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream)
{
    OP_LOGD("aclnnAlltoAllMatmulApace start");

    NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    aclnnStatus ret = aclnnInnerAlltoAllMatmulApace(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAlltoAllMatmulApace failed, ret=%d", ret);
        return ACLNN_ERR_INNER;
    }

    OP_LOGD("aclnnAlltoAllMatmulApace end");
    return ACLNN_SUCCESS;
}
