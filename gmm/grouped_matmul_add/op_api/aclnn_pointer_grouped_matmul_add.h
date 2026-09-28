/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#ifndef OP_API_INC_ACLNN_POINTER_GROUPED_MATMUL_ADD_H
#define OP_API_INC_ACLNN_POINTER_GROUPED_MATMUL_ADD_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnPointerGroupedMatmulAddGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *groupList, aclTensorList *yRefs,
    bool transposeX, bool transposeWeight, int64_t groupType, int64_t groupListType,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnPointerGroupedMatmulAdd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
