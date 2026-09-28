/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "pointer_grouped_matmul_add.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(PointerGroupedMatmulAdd);

aclTensorList *PointerGroupedMatmulAdd(const aclTensor *x, const aclTensor *weight, const aclTensor *groupList,
                                       aclTensorList *yRefs, bool transposeX, bool transposeWeight,
                                       int64_t groupType, int64_t groupListType, aclOpExecutor *executor)
{
    L0_DFX(PointerGroupedMatmulAdd, x, weight, groupList, yRefs);
    auto ret = INFER_SHAPE(PointerGroupedMatmulAdd, OP_INPUT(x, weight, groupList, yRefs), OP_OUTPUT(yRefs),
                           OP_ATTR(transposeX, transposeWeight, groupType, groupListType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PointerGroupedMatmulAdd InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(PointerGroupedMatmulAdd, OP_INPUT(x, weight, groupList, yRefs),
                                      OP_OUTPUT(yRefs),
                                      OP_ATTR(transposeX, transposeWeight, groupType, groupListType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PointerGroupedMatmulAdd launch registration failed.");
        return nullptr;
    }
    return yRefs;
}
} // namespace l0op
