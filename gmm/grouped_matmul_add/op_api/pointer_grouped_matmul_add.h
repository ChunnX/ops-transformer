/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#ifndef OP_API_INC_LEVEL0_OP_POINTER_GROUPED_MATMUL_ADD_H
#define OP_API_INC_LEVEL0_OP_POINTER_GROUPED_MATMUL_ADD_H

#include "opdev/op_executor.h"

namespace l0op {
aclTensorList *PointerGroupedMatmulAdd(const aclTensor *x, const aclTensor *weight, const aclTensor *groupList,
                                       aclTensorList *yRefs, bool transposeX, bool transposeWeight,
                                       int64_t groupType, int64_t groupListType, aclOpExecutor *executor);
} // namespace l0op
#endif
