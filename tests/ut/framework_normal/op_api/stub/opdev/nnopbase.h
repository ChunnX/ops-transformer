/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_NNOPBASE_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_NNOPBASE_H

#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
aclnnStatus NnopbaseSetHcomGroup(void *const executor, char *const group);
aclnnStatus NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);
}
#endif // OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_NNOPBASE_H