/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

#include <cstdio>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmul);

namespace {
template <typename... Args>
void GmmPrint(const char *format, Args... args)
{
    std::printf("[GMM L0] ");
    std::printf(format, args...);
    std::printf("\n");
    std::fflush(stdout);
}

void PrintTensorInfo(const char *name, const aclTensor *tensor)
{
    if (tensor == nullptr) {
        GmmPrint("launch tensor %s: nullptr.", name);
        return;
    }
    GmmPrint("launch tensor %s: dtype=%s, storageFormat=%s(primary=%s), viewShape=%s, storageShape=%s, "
             "viewDim=%zu, storageDim=%zu, isEmpty=%d.",
             name, op::ToString(tensor->GetDataType()).GetString(),
             op::ToString(tensor->GetStorageFormat()).GetString(),
             op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat()))).GetString(),
             op::ToString(tensor->GetViewShape()).GetString(),
             op::ToString(tensor->GetStorageShape()).GetString(),
             tensor->GetViewShape().GetDimNum(), tensor->GetStorageShape().GetDimNum(),
             static_cast<int>(tensor->IsEmpty()));
}

void PrintTensorListInfo(const char *name, const aclTensorList *tensorList)
{
    if (tensorList == nullptr) {
        GmmPrint("launch tensorList %s: nullptr.", name);
        return;
    }
    GmmPrint("launch tensorList %s: size=%zu.", name, tensorList->Size());
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        char tensorName[128] = {};
        std::snprintf(tensorName, sizeof(tensorName), "%s[%zu]", name, i);
        PrintTensorInfo(tensorName, (*tensorList)[i]);
    }
}

void PrintLaunchInfo(const char *stage, const aclTensorList *x, const aclTensorList *weight,
                     const aclTensorList *biasOptional, const aclTensorList *scaleOptional,
                     const aclTensorList *offsetOptional, const aclTensorList *antiquantScaleOptional,
                     const aclTensorList *antiquantOffsetOptional, const aclTensor *groupListOptional,
                     const aclTensor *perTokenScaleOptional, const aclTensorList *out)
{
    GmmPrint("GroupedMatmul L0 launch tensor info %s.", stage);
    PrintTensorListInfo("x", x);
    PrintTensorListInfo("weight", weight);
    PrintTensorListInfo("biasOptional", biasOptional);
    PrintTensorListInfo("scaleOptional", scaleOptional);
    PrintTensorListInfo("offsetOptional", offsetOptional);
    PrintTensorListInfo("antiquantScaleOptional", antiquantScaleOptional);
    PrintTensorListInfo("antiquantOffsetOptional", antiquantOffsetOptional);
    PrintTensorInfo("groupListOptional", groupListOptional);
    PrintTensorInfo("perTokenScaleOptional", perTokenScaleOptional);
    PrintTensorListInfo("out", out);
}
} // namespace

const aclTensorList *GroupedMatmul(const aclTensorList *x,
                                   const aclTensorList *weight,
                                   const aclTensorList *biasOptional,
                                   const aclTensorList *scaleOptional,
                                   const aclTensorList *offsetOptional,
                                   const aclTensorList *antiquantScaleOptional,
                                   const aclTensorList *antiquantOffsetOptional,
                                   const aclTensor *groupListOptional,
                                   const aclTensor *perTokenScaleOptional,
                                   int64_t splitItem,
                                   op::DataType yDtype,
                                   bool transposeWeight,
                                   bool transposeX,
                                   int64_t groupType,
                                   int64_t groupListType,
                                   int64_t actType,
                                   const aclIntArray *tuningConfig,
                                   size_t outLength,
                                   aclOpExecutor *executor) {
    L0_DFX(GroupedMatmul, x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
           antiquantOffsetOptional, groupListOptional, perTokenScaleOptional, splitItem, yDtype,
           transposeWeight, transposeX, groupType, groupListType, actType, tuningConfig, outLength);
    GmmPrint("GroupedMatmul L0 start. splitItem=%ld, yDtype=%s, transposeWeight=%d, transposeX=%d, "
             "groupType=%ld, groupListType=%ld, actType=%ld, outLength=%zu.",
             splitItem, op::ToString(yDtype).GetString(), static_cast<int>(transposeWeight),
             static_cast<int>(transposeX), groupType, groupListType, actType, outLength);

    std::vector<const aclTensor*> tensorsVec;
    const aclTensor *x0 = x->Size() > 0 ? (*x)[0] : nullptr;
    if (x0 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "(*x)[0] is nullptr.");
        return nullptr;
    }

    for (size_t i(0); i < outLength; ++i) {
        tensorsVec.emplace_back(executor->AllocTensor(yDtype, x0->GetStorageFormat(), x0->GetOriginalFormat()));
    }

    int64_t outputDtype = yDtype == DataType::DT_INT32 ? 2 : -1;
    auto out = executor->AllocTensorList(tensorsVec.data(), outLength);

    PrintLaunchInfo("before INFER_SHAPE", x, weight, biasOptional, scaleOptional, offsetOptional,
                    antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, perTokenScaleOptional, out);
    auto ret = INFER_SHAPE(GroupedMatmul,
                           OP_INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                                    antiquantOffsetOptional, groupListOptional, perTokenScaleOptional),
                           OP_OUTPUT(out),
                           OP_ATTR(splitItem, outputDtype, transposeWeight, transposeX, groupType, groupListType, actType, tuningConfig));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    PrintLaunchInfo("after INFER_SHAPE before ADD_TO_LAUNCHER_LIST_AICORE", x, weight, biasOptional, scaleOptional,
                    offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional,
                    perTokenScaleOptional, out);
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmul,
                                      OP_INPUT(x, weight, biasOptional, scaleOptional, offsetOptional,
                                               antiquantScaleOptional, antiquantOffsetOptional, groupListOptional,
                                               perTokenScaleOptional),
                                      OP_OUTPUT(out),
                                      OP_ATTR(splitItem, outputDtype, transposeWeight, transposeX, groupType, groupListType,
                                              actType, tuningConfig));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return out;
}

}  // namespace l0op
