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
 * \file aclnn_stem_oam_prep_paged_kv.cpp
 * \brief
 */

#include "aclnn_stem_oam_prep_paged_kv.h"
#include "stem_oam_prep_paged_kv.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> KV_CACHE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> KV_SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};


inline static bool CheckNull(const aclTensor* kCache, const aclTensor* vCache, const aclTensor* kvIndices,
                             const aclTensor* kvSeqLens, const aclTensor* kScaleCache, const aclTensor* vScale)
{
    OP_CHECK_NULL(kCache, return false);
    OP_CHECK_NULL(vCache, return false);
    OP_CHECK_NULL(kvIndices, return false);
    OP_CHECK_NULL(kvSeqLens, return false);
    OP_CHECK_NULL(kScaleCache, return false);
    OP_CHECK_NULL(vScale, return false);
    return true;
}

inline static bool CheckDtype(const aclTensor* kCache, const aclTensor* vCache, const aclTensor* kScaleCache,
                                     const aclTensor* vScale)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(kCache, KV_CACHE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(vCache, KV_CACHE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(kScaleCache, KV_SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(vScale, KV_SCALE_DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static bool CheckShape(const aclTensor* kCache, const aclTensor* vCache, const aclTensor* kScaleCache,
                              const aclTensor* vScale, int64_t cacheLayout, int64_t kvBlockSize)
{
    auto kCacheShape = kCache->GetViewShape();
    auto vCacheShape = vCache->GetViewShape();
    auto kScaleCacheShape = kScaleCache->GetViewShape();
    if (kCacheShape.GetDimNum() != 4 || vCacheShape.GetDimNum() != 4 || kScaleCacheShape.GetDimNum() != 4) {
        return false;
    }
    if (cacheLayout == 0 && kScaleCache.GetDim(1) != kvBlockSize && kScaleCache.GetDim(2) != vScale.GetDim(0)){
        return false;
    } else if (cacheLayout == 1 && kScaleCache.GetDim(2) != kvBlockSize && kScaleCache.GetDim(1) != vScale.GetDim(0)) {
        return false;
    }
    return true;
}

inline static bool CheckAttr(int64_t stemBlockSize, int64_t stemStride)
{
    if (stemBlockSize % 32 != 0 || stemBlockSize > 256) {
        return false;
    }
    if (stemStride % 16 != 0 || stemStride > 64 || stemStride > stemBlockSize) {
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* kCache, const aclTensor* vCache, const aclTensor* kvIndices,
                                      const aclTensor* kvSeqLens, const aclTensor* kScaleCache, const aclTensor* vScale,
                                      int64_t cacheLayout, int64_t kvBlockSize, int64_t stemBlockSize, int64_t stemStride,
                                      const aclTensor* kFlat, const aclTensor* vBias)
{
    CHECK_RET(CheckNull(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(kCache, vCache kScaleCache, vScale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(kCache, vCache, kScaleCache, vScale, cacheLayout, kvBlockSize), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttr(stemBlockSize, stemStride), ACLNN_ERR_PARAM_INVALID);
}

aclnnStatus aclnnStemOamPrepPagedKvGetWorkspaceSize(
    const aclTensor* kCache, const aclTensor* vCache, const aclTensor* kvIndices,
    const aclTensor* kvSeqLens, const aclTensor* kScaleCache, const aclTensor* vScale, float lambdaMag,
    int64_t cacheLayout, int64_t kvBlockSize, int64_t stemBlockSize, int64_t stemStride,
    const aclTensor* kFlat, const aclTensor* vBias, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNonzero, DFX_IN(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, lambdaMag,
                   cacheLayout, kvBlockSize, stemBlockSize, stemStride), DFX_OUT(kFlat, vBias));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, kFlat, vBias);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto kCacheView = executor->CreateView(
        kCache, kCache->GetViewShape(), kCache->GetStorageShape(),
        kCache->GetViewStrides(), kCache->GetViewOffset());
    auto vCacheView = executor->CreateView(
        vCache, vCache->GetViewShape(), vCache->GetStorageShape(),
        vCache->GetViewStrides(), vCache->GetViewOffset());

    auto kvIndicesContiguous = l0op::Contiguous(kvIndices, executor);
    auto kvSeqLensContiguous = l0op::Contiguous(kvSeqLens, executor);
    auto kScaleCacheContiguous = l0op::Contiguous(kScaleCache, executor);
    auto vScaleContiguous = l0op::Contiguous(vScale, executor);

    // 调用l0算子
    auto ret = l0op::StemOamPrepPagedKv(kCacheView, vCacheView, kvIndicesContiguous, kvSeqLensContiguous, kScaleCacheContiguous,
                             kScaleCacheContiguous, vScaleContiguous, lambdaMag, cacheLayout, kvBlockSize, stemBlockSize,
                             stemStride, kFlat, vBias, uniqueExecutor.get());
    CHECK_RET(ret != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStemOamPrepPagedKv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStemOamPrepPagedKv);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif