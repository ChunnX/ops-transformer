/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gather_pa_kv_cache_with_dequant.h"

#include <new>

#include "gather_pa_kv_cache_with_dequant.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT32, DataType::DT_INT64
};

static aclnnStatus CheckNullptr(
    const aclTensor *keyCache,
    const aclTensor *keyScaleCache,
    const aclTensor *valueCache,
    const aclTensor *valueScale,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    const aclTensor *keyRef,
    const aclTensor *valueRef,
    const uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(keyCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyScaleCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(valueCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(valueScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTables != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(seqLens != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(valueRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypeValid(
    const aclTensor *keyCache,
    const aclTensor *keyScaleCache,
    const aclTensor *valueCache,
    const aclTensor *valueScale,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    const aclTensor *keyRef,
    const aclTensor *valueRef,
    const aclTensor *seqOffsetOptional)
{
    // Key cache must be INT8
    OP_CHECK_DTYPE_NOT_SUPPORT(keyCache, {DataType::DT_INT8}, return ACLNN_ERR_PARAM_INVALID);
    
    // Key scale must be FLOAT
    OP_CHECK_DTYPE_NOT_SUPPORT(keyScaleCache, {DataType::DT_FLOAT}, return ACLNN_ERR_PARAM_INVALID);
    
    // Value cache must be FP8 E4M3FN
    OP_CHECK_DTYPE_NOT_SUPPORT(valueCache, {DataType::DT_FLOAT8_E4M3FN}, return ACLNN_ERR_PARAM_INVALID);
    
    // Value scale must be FLOAT
    OP_CHECK_DTYPE_NOT_SUPPORT(valueScale, {DataType::DT_FLOAT}, return ACLNN_ERR_PARAM_INVALID);
    
    // Block tables and seq_lens must be INT32 or INT64
    OP_CHECK_DTYPE_NOT_SUPPORT(blockTables, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(seqLens, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    
    // Key and value ref must be BF16
    OP_CHECK_DTYPE_NOT_SUPPORT(keyRef, {DataType::DT_BF16}, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(valueRef, {DataType::DT_BF16}, return ACLNN_ERR_PARAM_INVALID);
    
    // Optional seq_offset must be INT32 or INT64
    if (seqOffsetOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(seqOffsetOptional, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(
    const aclTensor *keyCache,
    const aclTensor *keyScaleCache,
    const aclTensor *valueCache,
    const aclTensor *valueScale,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    const aclTensor *keyRef,
    const aclTensor *valueRef)
{
    // Key cache: [num_blocks, block_size, num_heads, head_dim_k]
    auto keyCacheDimNum = keyCache->GetViewShape().GetDimNum();
    if (keyCacheDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of keyCache must be 4, but got %ld.",
                keyCacheDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Key scale: [num_blocks, block_size, num_heads]
    auto keyScaleDimNum = keyScaleCache->GetViewShape().GetDimNum();
    if (keyScaleDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of keyScaleCache must be 3, but got %ld.",
                keyScaleDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Value cache: [num_blocks, block_size, num_heads, head_dim_v]
    auto valueCacheDimNum = valueCache->GetViewShape().GetDimNum();
    if (valueCacheDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of valueCache must be 4, but got %ld.",
                valueCacheDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Value scale: [num_heads, head_dim_v]
    auto valueScaleDimNum = valueScale->GetViewShape().GetDimNum();
    if (valueScaleDimNum != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of valueScale must be 2, but got %ld.",
                valueScaleDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Block tables: [batch, block_indices]
    auto blockTablesDimNum = blockTables->GetViewShape().GetDimNum();
    if (blockTablesDimNum != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of blockTables must be 2, but got %ld.",
                blockTablesDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Seq lens: [batch] or [batch+1]
    auto seqLensDimNum = seqLens->GetViewShape().GetDimNum();
    if (seqLensDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of seqLens must be 1, but got %ld.",
                seqLensDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Key ref: [num_tokens, num_heads, head_dim_k]
    auto keyRefDimNum = keyRef->GetViewShape().GetDimNum();
    if (keyRefDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of keyRef must be 3, but got %ld.",
                keyRefDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Value ref: [num_tokens, num_heads, head_dim_v]
    auto valueRefDimNum = valueRef->GetViewShape().GetDimNum();
    if (valueRefDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of valueRef must be 3, but got %ld.",
                valueRefDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnGatherPaKvCacheWithDequantGetWorkspaceSize(
    const aclTensor *keyCache,
    const aclTensor *keyScaleCache,
    const aclTensor *valueCache,
    const aclTensor *valueScale,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    aclTensor *keyRef,
    aclTensor *valueRef,
    const aclTensor *seqOffsetOptional,
    char *cacheMode,
    bool isSeqLensCumsum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGatherPaKvCacheWithDequant,
                   DFX_IN(keyCache, keyScaleCache, valueCache, valueScale, blockTables, seqLens, keyRef, valueRef,
                          seqOffsetOptional, cacheMode, isSeqLensCumsum),
                   DFX_OUT(keyRef, valueRef));

    CHECK_RET(CheckNullptr(keyCache, keyScaleCache, valueCache, valueScale, blockTables, seqLens, keyRef, valueRef,
                           workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    bool keyCacheEmpty = keyCache->GetViewShape().GetShapeSize() == 0;
    bool valueCacheEmpty = valueCache->GetViewShape().GetShapeSize() == 0;
    bool keyRefEmpty = keyRef->GetViewShape().GetShapeSize() == 0;
    bool valueRefEmpty = valueRef->GetViewShape().GetShapeSize() == 0;
    if (keyCacheEmpty || valueCacheEmpty || keyRefEmpty || valueRefEmpty) {
        *workspaceSize = 0;
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(CheckShape(keyCache, keyScaleCache, valueCache, valueScale, blockTables, seqLens, keyRef, valueRef) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDtypeValid(keyCache, keyScaleCache, valueCache, valueScale, blockTables, seqLens, keyRef, valueRef,
                              seqOffsetOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Make all tensors contiguous
    auto keyCacheContiguous = l0op::Contiguous(keyCache, uniqueExecutor.get());
    CHECK_RET(keyCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyScaleCacheContiguous = l0op::Contiguous(keyScaleCache, uniqueExecutor.get());
    CHECK_RET(keyScaleCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueCacheContiguous = l0op::Contiguous(valueCache, uniqueExecutor.get());
    CHECK_RET(valueCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueScaleContiguous = l0op::Contiguous(valueScale, uniqueExecutor.get());
    CHECK_RET(valueScaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto blockTablesContiguous = l0op::Contiguous(blockTables, uniqueExecutor.get());
    CHECK_RET(blockTablesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto seqLensContiguous = l0op::Contiguous(seqLens, uniqueExecutor.get());
    CHECK_RET(seqLensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyRefContiguous = const_cast<aclTensor *>(l0op::Contiguous(keyRef, uniqueExecutor.get()));
    CHECK_RET(keyRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueRefContiguous = const_cast<aclTensor *>(l0op::Contiguous(valueRef, uniqueExecutor.get()));
    CHECK_RET(valueRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto seqOffsetContiguous = seqOffsetOptional;
    if (seqOffsetOptional != nullptr) {
        seqOffsetContiguous = l0op::Contiguous(seqOffsetOptional, uniqueExecutor.get());
        CHECK_RET(seqOffsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto gatherResult = l0op::GatherPaKvCacheWithDequant(
        keyCacheContiguous, keyScaleCacheContiguous, valueCacheContiguous, valueScaleContiguous,
        blockTablesContiguous, seqLensContiguous,
        keyRefContiguous, valueRefContiguous,
        seqOffsetContiguous, cacheMode, isSeqLensCumsum,
        uniqueExecutor.get());
    CHECK_RET(std::get<0>(gatherResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyRefViewCopy = l0op::ViewCopy(std::get<0>(gatherResult), keyRef, uniqueExecutor.get());
    CHECK_RET(keyRefViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueRefViewCopy = l0op::ViewCopy(std::get<1>(gatherResult), valueRef, uniqueExecutor.get());
    CHECK_RET(valueRefViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGatherPaKvCacheWithDequant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGatherPaKvCacheWithDequant);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "GatherPaKvCacheWithDequant launch aicore failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
