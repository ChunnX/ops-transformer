/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_LIGHTNING_INDEXER_H
#define ACLNN_LIGHTNING_INDEXER_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif


/* funtion: aclnnScatterPaKvCacheGetWorkspaceSize
 * parameters :
 * key : required
 * keyCacheRef : required
 * slotMapping : required
 * value : required
 * valueCacheRef : required
 * compressLensOptional : optional
 * compressSeqOffsetOptional : optional
 * seqLensOptional : optional
 * cacheModeOptional : optional
 * scatterModeOptional : optional
 * stridesOptional : optional
 * offsetsOptional : optional
 * kvCacheStridesOptional : optional
 * keyCacheRef : required
 * valueCacheRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(
    const aclTensor *key,
    aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    aclTensor *valueCacheRef,
    const aclTensor *compressLensOptional,
    const aclTensor *compressSeqOffsetOptional,
    const aclTensor *seqLensOptional,
    char *cacheModeOptional,
    char *scatterModeOptional,
    const aclIntArray *stridesOptional,
    const aclIntArray *offsetsOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterPaKvCache
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterPaKvCache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_LIGHTNING_INDEXER_H
