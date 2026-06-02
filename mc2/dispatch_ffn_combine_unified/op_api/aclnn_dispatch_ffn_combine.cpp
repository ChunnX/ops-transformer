/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <vector>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_dispatch_ffn_combine.h"

using namespace Ops::Transformer;
using namespace op;
static constexpr int64_t INT4_NUMS_IN_INT32 = 8; // 一个int32包含8个int4

#ifdef __cplusplus
extern "C" {
#endif

// 将int32的输入dtype修改为int4, 同时ViewShape和ViewStrides也从int32修改为int4所对应的
aclTensorList* ConvertTensorListToInt4(const aclTensorList* input, aclOpExecutor* executor)
{
    std::vector<aclTensor *> tensors;
    for (int i = 0; i < input->Size(); i++) {
        auto tensor = (*input)[i];
        auto viewShape = tensor->GetViewShape();
        viewShape[viewShape.GetDimNum() - 1] = viewShape[viewShape.GetDimNum() - 1] * INT4_NUMS_IN_INT32;
        auto inputTemp = executor->CreateView(tensor, viewShape, tensor->GetViewOffset());
        inputTemp->SetDataType(DataType::DT_INT4);
        tensors.push_back(inputTemp);
    }
    aclTensorList *newInput = executor->AllocTensorList(tensors.data(), tensors.size());

    OP_LOGD("The conversion from int32 to int4 is completed.");
    return newInput;
}

static void CreateEmptyTensor(aclDataType dataType, const aclTensorList *&ioList,
                              aclTensorList *&outList, aclOpExecutor *executor)
{
    if (ioList == nullptr) {
        std::vector<aclTensor*> emptyTensors;
        aclTensor *emptyTensor = executor->AllocTensor({0}, static_cast<op::DataType>(dataType));
        emptyTensors.emplace_back(emptyTensor);
        outList = executor->AllocTensorList(emptyTensors.data(), emptyTensors.size());
        ioList = outList;
    }
}

aclnnStatus aclnnDispatchFFNCombineGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2,
    const aclTensorList *weightScales1, const aclTensorList *weightScales2,
    const aclTensorList *bias1, const aclTensorList *bias2,
    const aclTensor *xActiveMask, const aclTensor *scales,
    int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType,
    int64_t combineQuantMode, char *commAlg, int64_t numMaxTokenPerRank,
    char *activation, float activationClamp, int64_t activation_out_dtype, bool transposeWeight1, bool transposeWeight2,
    int64_t weight1Interleave,
    aclTensor *yOut, aclTensor *expertTokenNumsOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_dispatch_ffn_combine WorkspaceSize start");
    OP_CHECK_NULL(context, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkWeights, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(yOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expertTokenNumsOut, return ACLNN_ERR_PARAM_NULLPTR);

    // 只在DAV_2201架构上对x1和x2进行int32到int4的转换预处理
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 && executor != nullptr) {
        auto uniqueExecutor = CREATE_EXECUTOR();
        if (weight1 != nullptr && weight1->Size() > 0 && (*weight1)[0]->GetDataType() == DataType::DT_INT32) {
            weight1 = ConvertTensorListToInt4(weight1, uniqueExecutor.get());
        }
        if (weight2 != nullptr && weight2->Size() > 0 && (*weight2)[0]->GetDataType() == DataType::DT_INT32) {
            weight2 = ConvertTensorListToInt4(weight2, uniqueExecutor.get());
        }
        uniqueExecutor.ReleaseTo(executor);
    }

    // 确保 executor 已创建，以便调用 CreateEmptyTensor
    if (*executor == nullptr) {
        auto uniqueExec = CREATE_EXECUTOR();
        uniqueExec.ReleaseTo(executor);
    }

    // 可选 DYNAMIC 参数为 nullptr 时创建带 dtype 的 dummy tensor list 满足支持列表校验
    aclTensorList *tmpBiasList = nullptr;
    aclTensorList *tmpScaleList = nullptr;
    CreateEmptyTensor(ACL_FLOAT, bias1, tmpBiasList, *executor);
    CreateEmptyTensor(ACL_FLOAT, bias2, tmpBiasList, *executor);
    CreateEmptyTensor(ACL_INT64, weightScales1, tmpScaleList, *executor);
    CreateEmptyTensor(ACL_INT64, weightScales2, tmpScaleList, *executor);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerDispatchFFNCombineGetWorkspaceSize(
        context, x, topkIds, topkWeights, weight1, weight2,
        weightScales1, weightScales2, bias1, bias2, xActiveMask, scales,
        moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum,
        dispatchQuantMode, dispatchQuantOutType, combineQuantMode,
        commAlg, numMaxTokenPerRank, activation, activationClamp, activation_out_dtype,
        transposeWeight1, transposeWeight2, weight1Interleave,
        yOut, expertTokenNumsOut,
        workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnDispatchFFNCombine(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_dispatch_ffn_combine start");
    return aclnnInnerDispatchFFNCombine(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
