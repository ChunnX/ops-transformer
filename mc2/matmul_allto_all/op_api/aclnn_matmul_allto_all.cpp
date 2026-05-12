/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_matmul_allto_all.h"
#include "matmul_allto_all_util.h"
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
#include "aclnn_matmul_allto_all_helper.h"

namespace {

using namespace op;
using namespace matmul_allto_all_check;
using namespace matmul_allto_all_helper;

// L0层两段式接口Inner，根据算子原型op_graph/matmul_allto_all_proto.h，由模板自动生成。非量化L2层接口和量化L2层接口共用一套L0层接口。
// worldSize为硬件方参数，在aclnn侧不感知。yDtype在aclnn侧不感知。这两个参数需要在Inner接口处声明，在aclnn侧通过默认值传参。
extern "C" aclnnStatus aclnnInnerMatmulAlltoAllGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                                                                const aclTensor* x1ScaleOptional, const aclTensor* x2ScaleOptional,
                                                                const aclTensor* commScaleOptional,
                                                                const aclTensor* x1OffsetOptional, const aclTensor* x2OffsetOptional,
                                                                char* group, int64_t worldSize, const aclIntArray* all2allAxesOptional,
                                                                int64_t yDtype, int64_t x1QuantMode, int64_t x2QuantMode,
                                                                int64_t commQuantMode, int64_t commQuantDtype,
                                                                bool transposeX1, bool transposeX2, int64_t groupSize,
                                                                const aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 非量化L2接口调用L0时需要设置较多默认值，通过InnerMatmulAlltoAllGetWorkspaceSize完成默认值传参和调用L0层接口
extern "C" aclnnStatus InnerMatmulAlltoAllGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                                           const aclIntArray* alltoAllAxesOptional, const char *group,
                                                           bool transposeX1, bool transposeX2, const aclTensor *output,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // ACL和GE的datatype枚举值对undefined定义不同，inner接口进入到算子内部，需要使用GE枚举值，因此此处使用的枚举值为28
    const int64_t GE_UNDEFINED = 28;
    // 根据算子原型定义默认值
    aclTensor* x1ScaleOptional = nullptr;
    aclTensor* x2ScaleOptional = nullptr;
    aclTensor* commScaleOptional = nullptr;
    aclTensor* x1OffsetOptional = nullptr;
    aclTensor* x2OffsetOptional = nullptr;
    const aclTensor* out = output;
    char* str_group = const_cast<char*>(group);
    int64_t worldSize = -1;
    int64_t yDtype = GE_UNDEFINED;
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    int64_t commQuantMode = 0;
    int64_t commQuantDtype = GE_UNDEFINED;
    int64_t groupSize = 0;
    aclnnStatus ret = aclnnInnerMatmulAlltoAllGetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        str_group, worldSize, alltoAllAxesOptional, yDtype, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype,
        transposeX1, transposeX2, groupSize, out, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAll, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

// 两段式接口
extern "C" aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                                           const aclIntArray* alltoAllAxesOptional, const char *group,
                                                           bool transposeX1, bool transposeX2, const aclTensor *output,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前只有支持转置的x2涉及该处理
    aclnnStatus checkX2Ret = CheckX2Valid(x2);
    CHECK_RET(checkX2Ret == ACLNN_SUCCESS, checkX2Ret); // 先检查x2是否合法，避免非法操作
    bool notContiguous = IsTransposeLastTwoDims(x2);    // notContiguous标识x2是否是非连续的，通常在pytorch经过.t()会导致x2非连续
    auto transX2 = x2;    // 复制一个x2
    if (notContiguous && transposeX2) {    // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 not contiguous, and set x2 transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {    // 只有当非连续时，才会涉及到转连续等情况
        transposeX2 = !transposeX2;
        // 把非连续x2转成连续
        transX2 = TransX2Tensor(x2);
        CHECK_RET(transX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("X2 is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. After processing, transX2 dim0 is %ld, and dim1 is %ld.",
            x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1), transX2->GetViewShape().GetDim(0), transX2->GetViewShape().GetDim(1));
    }
    aclnnStatus retParam = CheckAndHandleParams(x1, transX2, biasOptional, alltoAllAxesOptional, group, transposeX1, transposeX2, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 处理空tensor，目前非量化matmulalltoall只支持x1第一维度bs为0，空tensor作异常处理
    if (x1->GetViewShape().GetDim(0) == 0 && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return DealWithEmptyTensor(workspaceSize, executor);
    }
    aclnnStatus ret = InnerMatmulAlltoAllGetWorkspaceSize(
        x1, transX2, biasOptional, alltoAllAxesOptional, group, transposeX1, transposeX2, output, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAll, end ret %d", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnMatmulAlltoAllGetWorkspaceSize interface call failed.");
    }
    return ret;
}

extern "C" aclnnStatus aclnnMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else if (GetCurrentPlatformInfo().GetSocVersion()== SocVersion::ASCEND910_93) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B){
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAlltoAll(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnInnerMatmulAlltoAll interface call failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
} // namespace
