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
 * \file aclnn_moe_distribute_combine_teardown.cpp
 * \brief
 */
#include "aclnn_moe_distribute_combine_teardown.h"
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/matmul_util.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"

namespace {

using namespace op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// 根据API定义，列出teardown支持的dtype
const std::initializer_list<op::DataType> EXPANDX_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

const std::initializer_list<op::DataType> QUANT_EXPANDX_LIST = {
    op::DataType::DT_INT8
};

const std::initializer_list<op::DataType> EXPERT_IDS_LIST = {
    op::DataType::DT_INT32
};

const std::initializer_list<op::DataType> EXPAND_IDX_LIST = {
    op::DataType::DT_INT32
};

const std::initializer_list<op::DataType> EXPERT_SCALES_LIST = {
    op::DataType::DT_FLOAT
};

const std::initializer_list<op::DataType> COMM_CMD_INFO_LIST = {
    op::DataType::DT_INT32
};

const std::initializer_list<op::DataType> X_ACTIVE_MASK_OPTIONAL_LIST = {
    op::DataType::DT_BOOL
};

const std::initializer_list<op::DataType> SHARED_EXPERT_X_OPTIONAL_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

const std::initializer_list<op::DataType> X_OUT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static bool CheckNotNull(const aclTensor *expandX, const aclTensor *quantExpandX, const aclTensor *expertIds,
                         const aclTensor *expandIdx, const aclTensor *expertScales, const aclTensor *commCmdInfo,
                         const aclTensor *xActiveMaskOptional, const aclTensor *sharedExpertXOptional,
                         const char *groupEp, aclTensor *xOut)
{
    OP_LOGD("aclnn_moe_distribute_combine_teardown CheckNotNull start");
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(quantExpandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandIdx, return false);
    OP_CHECK_NULL(expertScales, return false);
    OP_CHECK_NULL(commCmdInfo, return false);
    OP_CHECK_NULL(xOut, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name is Empty.");
        return false;
    }
    OP_LOGD("aclnn_moe_distribute_combine_teardown CheckNotNull success");
    return true;
}

static bool CheckInputDataType(const aclTensor* expandX, const aclTensor* quantExpandX, const aclTensor* expertIds,
                               const aclTensor* expandIdx, const aclTensor* expertScales, const aclTensor* commCmdInfo,
                               const aclTensor* xActiveMaskOptional, const aclTensor* sharedExpertXOptional, aclTensor* xOut)
{
    if (CheckType(expandX->GetDataType(), EXPANDX_LIST) && CheckType(quantExpandX->GetDataType(), QUANT_EXPANDX_LIST) &&
        CheckType(expertIds->GetDataType(), EXPERT_IDS_LIST) && CheckType(expandIdx->GetDataType(), EXPAND_IDX_LIST) &&
        CheckType(expertScales->GetDataType(), EXPERT_SCALES_LIST) &&
        CheckType(commCmdInfo->GetDataType(), COMM_CMD_INFO_LIST) &&
        CheckType(xActiveMaskOptional->GetDataType(), X_ACTIVE_MASK_OPTIONAL_LIST) &&
        CheckType(sharedExpertXOptional->GetDataType(), SHARED_EXPERT_X_OPTIONAL_LIST) &&
        CheckType(xOut->GetDataType(), X_OUT_LIST)) {
        return true;
    } else {
        return false;
    }
}

// 校验数据格式
static bool CheckInputDataFormat(const aclTensor* expandX, const aclTensor* quantExpandX, const aclTensor* expertIds,
                                  const aclTensor* expandIdx, const aclTensor* expertScales, const aclTensor* commCmdInfo,
                                  const aclTensor* xActiveMaskOptional, const aclTensor* sharedExpertXOptional, aclTensor* xOut)
{
    if (IsPrivateFormat(expandX->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, expandX format %s does not support Private Format.",
                op::ToString(expandX->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(quantExpandX->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, quantExpandX format %s does not support Private Format.",
                op::ToString(quantExpandX->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(expertIds->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, expertIds format %s does not support Private Format.",
                op::ToString(expertIds->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(expandIdx->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, expandIdx format %s does not support Private Format.",
                op::ToString(expandIdx->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(expertScales->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, expertScales format %s does not support Private Format.",
                op::ToString(expertScales->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(commCmdInfo->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, commCmdInfo format %s does not support Private Format.",
                op::ToString(commCmdInfo->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(xActiveMaskOptional->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, xActiveMaskOptional format %s does not support Private Format.",
                op::ToString(xActiveMaskOptional->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(sharedExpertXOptional->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, sharedExpertXOptional format %s does not support Private Format.",
                op::ToString(sharedExpertXOptional->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(xOut->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnMoeDistributeCombineTeardown, xOut format %s does not support Private Format.",
                op::ToString(xOut->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

// 根据API定义，列举
static aclnnStatus CheckParams(const aclTensor *expandX, const aclTensor *quantExpandX, const aclTensor *expertIds,
                               const aclTensor *expandIdx, const aclTensor *expertScales, const aclTensor *commCmdInfo,
                               const aclTensor *xActiveMaskOptional, const aclTensor *sharedExpertXOptional,
                               const char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
                               int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
                               int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
                               aclTensor *xOut)
{
    OP_LOGD("aclnn_moe_distribute_combine_teardown checkparams start");
    CHECK_RET(CheckNotNull(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMaskOptional,
                           sharedExpertXOptional, groupEp, xOut),ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckInputDataType(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMaskOptional,
                                    sharedExpertXOptional, xOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputDataFormat(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMaskOptional,
                                    sharedExpertXOptional, xOut), ACLNN_ERR_PARAM_INVALID);
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnn_moe_distribute_combine_teardown checkparams success");
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnInnerMoeDistributeCombineTeardownGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *quantExpandX, const aclTensor *expertIds, const aclTensor *expandIdx,
    const aclTensor *expertScales, const aclTensor *commCmdInfo, const aclTensor *xActiveMaskOptional,
    const aclTensor *sharedExpertXOptional, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg, aclTensor *xOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerMoeDistributeCombineTeardown(void *workspace, uint64_t workspaceSize,
                                                              aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *quantExpandX, const aclTensor *expertIds, const aclTensor *expandIdx,
    const aclTensor *expertScales, const aclTensor *commCmdInfo, const aclTensor *xActiveMaskOptional,
    const aclTensor *sharedExpertXOptional, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg, aclTensor *xOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_moe_distribute_combine_teardown get_get_workspace_size start");
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported npuArch");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto ret_param =
        CheckParams(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMaskOptional,
                    sharedExpertXOptional, groupEp, epWorldSize, epRankId, moeExpertNum, expertShardType,
                    sharedExpertNum, sharedExpertRankNum, globalBs, commQuantMode, commType, commAlg, xOut);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = aclnnInnerMoeDistributeCombineTeardownGetWorkspaceSize(
        expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMaskOptional,
        sharedExpertXOptional, groupEp, epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertNum,
        sharedExpertRankNum, globalBs, commQuantMode, commType, commAlg, xOut, workspaceSize, executor);
    OP_LOGD("aclnn_moe_distribute_combine_teardown get_workspace_size success");
    return ret;
}

extern "C" aclnnStatus aclnnMoeDistributeCombineTeardown(void *workspace, uint64_t workspaceSize,
                                                         aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnnMoeDistributeCombineTeardown start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerMoeDistributeCombineTeardown(workspace, workspaceSize, executor, stream);
    OP_LOGD("aclnnMoeDistributeCombineTeardown success");
    return ret;
}
