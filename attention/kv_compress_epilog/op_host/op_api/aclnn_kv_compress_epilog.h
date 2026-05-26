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
 * \file aclnn_kv_compress_epilog.h
 * \brief L2 ACLNN API for KvCompressEpilog
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_KV_COMPRESS_EPILOG_H_
#define OP_API_INC_LEVEL2_ACLNN_KV_COMPRESS_EPILOG_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(
    const aclTensor *kvCompressCache, const aclTensor *x, const aclTensor *slotMapping, int64_t quantGroupSize,
    int64_t quantMode, int64_t roundScale, float scaleAttr, int64_t padCount, aclTensor *output,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnKvCompressEpilog(void *workspace, uint64_t workspaceSize,
                                                                          aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_KV_COMPRESS_EPILOG_H_
