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
 * \file aclnn_matmul_reduce_scatter_v2.cpp
 * \brief
 */
#include "securec.h"
#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "common/utils/hccl_util.h"
#include "mc2_comm_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mc2_log_compat.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "aclnn_matmul_reduce_scatter_v2.h"
#include "aclnnInner_matmul_reduce_scatter_v2.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr int64_t TWO_DIMS = 2;
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_MTE = 1
};

enum class CommType : uint64_t {
    AIV = 2,
    INVALID = 3
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static void SetNnopbaseHcclServerTypeByArch(aclOpExecutor *executor)
{
    if ((executor == nullptr) || (NnopbaseSetHcclServerType == nullptr)) {
        return;
    }
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510)) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
}

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x1, return false);
    return true;
}

static const std::initializer_list<op::DataType> AIV_MODE_INPUT_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> AIV_MODE_BIAS_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> AIV_MODE_OUTPUT_SUPPORT_TYPE = {op::DataType::DT_FLOAT16,
                                                                                 op::DataType::DT_BF16};
static bool CheckAivModeDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *output)
{
    // 检查x1、x2、output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, AIV_MODE_INPUT_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, AIV_MODE_INPUT_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, AIV_MODE_BIAS_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(output, AIV_MODE_OUTPUT_SUPPORT_TYPE, return false);
    return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(int64_t streamMode)
{
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "streamMode",
                                  std::to_string(streamMode).c_str(), std::to_string(NUM_ACL_STOP_ON_FAILURE).c_str());
        return false;
    }
    return true;
}
static aclnnStatus CheckAivModeParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                      int64_t streamMode, const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckAivModeDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAivModeCommMode(const char *commMode)
{
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "[aclnnMatmulReduceScatterV2GetWorkspaceSize] commMode must not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strcmp(commMode, "aiv") != 0) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "commMode", commMode, "aiv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, bool isTransA)
{
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1 must be 2D.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return false;
    }
    int64_t kVal1 = 0, kVal2 = 0;
    if (isTransA) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1", "transposed",
                                              "Does not support transpose x1 matrix.");
        return false;
    }
    kVal1 = x1->GetViewShape().GetDim(1);
    kVal2 = x2->GetViewShape().GetDim(0);
    OP_API_CHECK((kVal1 != kVal2), {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1/x2",
                                  (std::string("k=") + std::to_string(kVal1) + "/" + std::to_string(kVal2)).c_str(),
                                  "equal k values");
        return false;
    });
    OP_API_CHECK((kVal1 < KVALUE_MIN || kVal1 >= KVALUE_MAX), {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "k-axis", std::to_string(kVal1).c_str(),
                                  "[256, 65535)");
        return false;
    });
    return true;
}

static bool MatmulReduceScatterV2IsWeightNZFormat(const aclTensor *x2)
{
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    aclGetFormat(x2, &format);
    if (format == aclFormat::ACL_FORMAT_ND) {
        OP_LOGD("MatmulReduceScatterV2, Received weight format is ACL_FORMAT_ND");
    }
    if (format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        OP_LOGD("MatmulReduceScatterV2, Received weight format is ACL_FORMAT_FRACTAL_NZ");
        uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
        OP_LOGD("MatmulReduceScatterV2, Shape is %lu", storageShapeDimNum);
        const uint64_t transdataNzDim = 4U;
        if (storageShapeDimNum == transdataNzDim) {
            return true;
        }
    }
    return false;
}

aclnnStatus matmulReduceScatterV2GetWorkSpaceSizeAivMode(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *x1Scale, const aclTensor *x2Scale,
    const aclTensor *quantScale, int64_t blockSize, const char *group, const char *reduceOp, int64_t commTurn,
    int64_t streamMode, int64_t groupSize, const char *commMode, aclTensor *output, aclTensor *amaxOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMatmulReduceScatterV2GetWorkspaceSizeAivMode start");
    auto ret_param = CheckAivModeParams(x1, x2, bias, streamMode, output);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto retCommMode = CheckAivModeCommMode(commMode);
    CHECK_RET(retCommMode == ACLNN_SUCCESS, retCommMode);

    bool transposeX1 = Ops::Transformer::IsTransposeLastTwoDims(x1);
    bool transposeX2 = Ops::Transformer::IsTransposeLastTwoDims(x2);
    uint32_t rankSize = 0;
    bool isAmaxOut = false;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    (void)MatmulReduceScatterV2IsWeightNZFormat(x2);
    CHECK_RET(CheckShape(x1, x2, transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 【A2、A3】检查x2矩阵非连续合法性
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510)) {
        if (!Ops::Transformer::IsTransposeLastTwoDims(x2) && !MC2Aclnn::IsTensorContiguous(x2)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSizeAivMode", "x2",
                                                  "non-contiguous", "x2 without transpose must be contiguous.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    aclnnStatus ret = aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(
        x1, x2, bias, x1Scale, x2Scale, quantScale, const_cast<char *>(group), const_cast<char *>(reduceOp),
        transposeX1, transposeX2, commTurn, rankSize, blockSize, groupSize, isAmaxOut, yDtype,
        const_cast<char *>(commMode), output, amaxOutOptional, workspaceSize, executor);
    if ((ret == ACLNN_SUCCESS) && (executor != nullptr) && (*executor != nullptr)) {
        SetNnopbaseHcclServerTypeByArch(*executor);
    }
    OP_LOGD("MatmulReduceScatterV2AivMode, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    return ret;
}

aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                       const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                       const aclTensor *quantScale, int64_t blockSize,
                                                       const char *group, const char *reduceOp, int64_t commTurn,
                                                       int64_t streamMode, int64_t groupSize, const char *commMode,
                                                       aclTensor *output, aclTensor *amaxOutOptional,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = ACLNN_ERR_INNER;
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510)) {
        OP_LOGD("[MatmulReduceScatterV2] NpuArch is %u, support aiv commmode only.",
                static_cast<uint32_t>(npuArch));
        ret = matmulReduceScatterV2GetWorkSpaceSizeAivMode(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group,
                                                           reduceOp, commTurn, streamMode, groupSize, commMode, output,
                                                           amaxOutOptional, workspaceSize, executor);
    } else {
        OP_LOGE(ACLNN_ERR_INNER, "[MatmulReduceScatterV2] Unsupported NPU arch, "
                                 "only DAV_3510 and DAV_2201 are supported.");
        return ACLNN_ERR_INNER;
    }

    return ret;
}

aclnnStatus aclnnMatmulReduceScatterV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    if ((workspace == nullptr) || (workspaceSize == 0UL)) {
        OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
        return ACLNN_SUCCESS;
    }

    CommType commModeEnum = CommType::INVALID;
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510)) {
        commModeEnum = CommType::AIV;
    }

    if (commModeEnum >= CommType::INVALID) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2", "commModeEnum", "invalid",
                                              "Does not match any available case, please check the input commMode.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    SetNnopbaseHcclServerTypeByArch(executor);
    aclnnStatus ret = aclnnInnerMatmulReduceScatterV2(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnMatmulReduceScatterV2", "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
