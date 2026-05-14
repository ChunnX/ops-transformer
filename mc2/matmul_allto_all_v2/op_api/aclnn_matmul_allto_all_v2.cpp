/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_matmul_allto_all_v2.h"
#include "securec.h"
#include "common/utils/op_mc2.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "common/utils/hccl_util.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "mc2_comm_utils.h"
#include "mc2/matmul_allto_all/op_api/matmul_allto_all_util.h"
#include "mc2/matmul_allto_all/op_api/aclnn_matmul_allto_all_helper.h"

#ifdef BUILD_OPEN_PROJECT
#include "version/hcomm_version.h"
#define HCCL_CHANNEL_SUPPORT_VERSION 90000000
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif
#endif

namespace {

using namespace matmul_allto_all_helper;

// L0层两段式接口Inner，根据算子原型op_graph/matmul_allto_all_v2_proto.h，由模板自动生成。非量化L2层接口和量化L2层接口共用一套L0层接口。
// worldSize为硬件方参数，在aclnn侧不感知。yDtype在aclnn侧不感知。这两个参数需要在Inner接口处声明，在aclnn侧通过默认值传参。
extern "C" aclnnStatus aclnnInnerMatmulAlltoAllV2GetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, char *group, int64_t worldSize, const aclIntArray *all2allAxesOptional,
    int64_t yDtype, int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype,
    bool transposeX1, bool transposeX2, int64_t groupSize, char *commModeOptional, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerMatmulAlltoAllV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);


// 非量化L2接口调用L0时需要设置较多默认值，通过InnerMatmulAlltoAllV2GetWorkspaceSize完成默认值传参和调用L0层接口
extern "C" aclnnStatus InnerMatmulAlltoAllV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                             const aclTensor *biasOptional,
                                                             const aclIntArray *alltoAllAxesOptional, const char *group,
                                                             bool transposeX1, bool transposeX2, const char *commMode,
                                                             const aclTensor *output, uint64_t *workspaceSize,
                                                             aclOpExecutor **executor)
{
    // ACL和GE的datatype枚举值对undefined定义不同，inner接口进入到算子内部，需要使用GE枚举值，因此此处使用的枚举值为28
    const int64_t GE_UNDEFINED = 28;
    // 根据算子原型定义默认值
    aclTensor *x1ScaleOptional = nullptr;
    aclTensor *x2ScaleOptional = nullptr;
    aclTensor *commScaleOptional = nullptr;
    aclTensor *x1OffsetOptional = nullptr;
    aclTensor *x2OffsetOptional = nullptr;
    const aclTensor *out = output;
    char *str_group = const_cast<char *>(group);
    int64_t worldSize = -1;
    int64_t yDtype = GE_UNDEFINED;
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    int64_t commQuantMode = 0;
    int64_t commQuantDtype = GE_UNDEFINED;
    char *str_commMode = const_cast<char *>(commMode);
    int64_t groupSize = 0;
    aclnnStatus ret = aclnnInnerMatmulAlltoAllV2GetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        str_group, worldSize, alltoAllAxesOptional, yDtype, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype,
        transposeX1, transposeX2, groupSize, str_commMode, out, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAllV2, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

// 两段式接口
extern "C" aclnnStatus aclnnMatmulAlltoAllV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                             const aclTensor *biasOptional,
                                                             const aclIntArray *alltoAllAxesOptional, const char *group,
                                                             const char *commMode, bool transposeX1, bool transposeX2,
                                                             const aclTensor *output, uint64_t *workspaceSize,
                                                             aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前只有支持转置的x2涉及该处理
    aclnnStatus checkX2Ret = CheckX2Valid(x2);
    CHECK_RET(checkX2Ret == ACLNN_SUCCESS, checkX2Ret); // 先检查x2是否合法，避免非法操作
    bool notContiguous =
        IsTransposeLastTwoDims(x2);     // notContiguous标识x2是否是非连续的，通常在pytorch经过.t()会导致x2非连续
    auto transX2 = x2;                  // 复制一个x2
    if (notContiguous && transposeX2) { // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 not contiguous, and set x2 transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous &&
        GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) { // 只有当非连续时，才会涉及到转连续等情况
        transposeX2 = !transposeX2;
        // 把非连续x2转成连续
        transX2 = TransX2Tensor(x2);
        CHECK_RET(transX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("X2 is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. After processing, transX2 "
                "dim0 is %ld, and dim1 is %ld.",
                x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1), transX2->GetViewShape().GetDim(0),
                transX2->GetViewShape().GetDim(1));
    }

    aclnnStatus retParam =
        CheckAndHandleParams(x1, transX2, biasOptional, alltoAllAxesOptional, group, transposeX1, transposeX2, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    // 校验commMode
    aclnnStatus checkCommModeRet = CheckCommModeStr(commMode);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);

    // 处理空tensor，目前非量化matmulAlltoAllV2只支持x1第一维度bs为0，空tensor作异常处理
    if (x1->GetViewShape().GetDim(0) == 0 && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return DealWithEmptyTensor(workspaceSize, executor);
    }
    aclnnStatus ret =
        InnerMatmulAlltoAllV2GetWorkspaceSize(x1, transX2, biasOptional, alltoAllAxesOptional, group, transposeX1,
                                              transposeX2, commMode, output, workspaceSize, executor);

    // 获取卡数
    uint32_t rankSize = 0;
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    auto aclnnRet = Mc2Aclnn::Mc2Context::GetMc2RankSize(group, rankSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
#endif

    if (ret == ACLNN_SUCCESS && *executor != nullptr && (NnopbaseSetHcclServerType)) {
        OP_LOGD("[MatmulAlltoAllV2] Execute commMode=%s", commMode);
        auto hcclServerType = NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU;
        if (std::strcmp(commMode, "aicpu") == 0) {
            hcclServerType = NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU;
        } else if (std::strcmp(commMode, "ccu") == 0) {
        } else {
            // default：小于等于8P走CCU，否则走AICPU
            if (rankSize <= 8) {
            } else {
                hcclServerType = NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU;
            }
        }
        NnopbaseSetHcclServerType(executor, hcclServerType);
    }

    OP_LOGD("MatmulAlltoAllV2, end ret %d", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnMatmulAlltoAllV2GetWorkspaceSize interface call failed.");
    }
    return ret;
}

extern "C" aclnnStatus aclnnMatmulAlltoAllV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerMatmulAlltoAllV2(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnInnerMatmulAlltoAllV2 interface call failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

} // namespace
