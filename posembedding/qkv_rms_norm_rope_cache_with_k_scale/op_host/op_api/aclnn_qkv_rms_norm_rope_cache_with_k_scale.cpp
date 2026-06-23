/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"

#include <dlfcn.h>
#include <vector>

#include "qkv_rms_norm_rope_cache_with_k_scale.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_checker.h"
#include "aclnn_kernels/contiguous.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

namespace QkvRmsNormRopeCacheWithKScale {

static aclnnStatus Contiguous(const aclTensor *&qkv, const aclTensor *&qGamma, const aclTensor *&kGamma,
                              const aclTensor *&cosSin, const aclTensor *&slotMapping, const aclTensor *&cuSeqLen,
                              const aclTensor *&actualSeqLens, const aclTensor *&optionalQRotation,
                              const aclTensor *&optionalKRotation, const aclTensor *&optionalVScale,
                              aclOpExecutor *executor)
{
    qkv = l0op::Contiguous(qkv, executor);
    CHECK_RET(qkv != nullptr, ACLNN_ERR_INNER_NULLPTR);

    qGamma = l0op::Contiguous(qGamma, executor);
    CHECK_RET(qGamma != nullptr, ACLNN_ERR_INNER_NULLPTR);

    kGamma = l0op::Contiguous(kGamma, executor);
    CHECK_RET(kGamma != nullptr, ACLNN_ERR_INNER_NULLPTR);

    cosSin = l0op::Contiguous(cosSin, executor);
    CHECK_RET(cosSin != nullptr, ACLNN_ERR_INNER_NULLPTR);

    slotMapping = l0op::Contiguous(slotMapping, executor);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_INNER_NULLPTR);

    cuSeqLen = l0op::Contiguous(cuSeqLen, executor);
    CHECK_RET(cuSeqLen != nullptr, ACLNN_ERR_INNER_NULLPTR);

    actualSeqLens = l0op::Contiguous(actualSeqLens, executor);
    CHECK_RET(actualSeqLens != nullptr, ACLNN_ERR_INNER_NULLPTR);

    optionalQRotation = l0op::Contiguous(optionalQRotation, executor);
    CHECK_RET(optionalQRotation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    optionalKRotation = l0op::Contiguous(optionalKRotation, executor);
    CHECK_RET(optionalKRotation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    optionalVScale = l0op::Contiguous(optionalVScale, executor);
    CHECK_RET(optionalVScale != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

} // namespace QkvRmsNormRopeCacheWithKScale

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma, const aclTensor *cosSin,
    const aclTensor *slotMapping, aclTensor *kCache, aclTensor *vCache, aclTensor *kScaleCache,
    const aclTensor *cuSeqLen, const aclTensor *actualSeqLens, const aclTensor *optionalQRotation,
    const aclTensor *optionalKRotation, const aclTensor *optionalVScale, const aclIntArray *headNums,
    const char *inputLayout, float epsilon, aclTensor *qOut, aclTensor *qScale, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    QkvRmsNormRopeCacheWithKScaleCheck::QkvRmsNormRopeCacheWithKScaleParams params;
    params.qkv = qkv;
    params.qGamma = qGamma;
    params.kGamma = kGamma;
    params.cosSin = cosSin;
    params.slotMapping = slotMapping;
    params.kCache = kCache;
    params.vCache = vCache;
    params.kScaleCache = kScaleCache;
    params.cuSeqLen = cuSeqLen;
    params.actualSeqLens = actualSeqLens;
    params.optionalQRotation = optionalQRotation;
    params.optionalKRotation = optionalKRotation;
    params.optionalVScale = optionalVScale;
    params.headNums = headNums;
    params.inputLayout = inputLayout;
    params.epsilon = epsilon;
    params.qOut = qOut;
    params.qScale = qScale;
    params.workspaceSize = workspaceSize;
    params.executor = executor;

    L2_DFX_PHASE_1(aclnnQkvRmsNormRopeCacheWithKScale,
                   DFX_IN(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, cuSeqLen,
                          actualSeqLens, optionalQRotation, optionalKRotation, optionalVScale, headNums, inputLayout,
                          epsilon),
                   DFX_OUT(qOut, qScale, kCache, vCache, kScaleCache));

    QkvRmsNormRopeCacheWithKScaleCheck::QkvRmsNormRopeCacheWithKScaleChecker checker;
    aclnnStatus checkRet = checker.CheckParams(params);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(QkvRmsNormRopeCacheWithKScale::Contiguous(
                  params.qkv, params.qGamma, params.kGamma, params.cosSin, params.slotMapping, params.cuSeqLen,
                  params.actualSeqLens, params.optionalQRotation, params.optionalKRotation, params.optionalVScale,
                  l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto kCacheView = l0Executor->CreateView(params.kCache,
                                             params.kCache->GetViewShape(),
                                             params.kCache->GetStorageShape(),
                                             params.kCache->GetViewStrides(),
                                             params.kCache->GetViewOffset());
    CHECK_RET(kCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto vCacheView = l0Executor->CreateView(params.vCache,
                                             params.vCache->GetViewShape(),
                                             params.vCache->GetStorageShape(),
                                             params.vCache->GetViewStrides(),
                                             params.vCache->GetViewOffset());
    CHECK_RET(vCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto kScaleCacheView = l0Executor->CreateView(params.kScaleCache,
                                                  params.kScaleCache->GetViewShape(),
                                                  params.kScaleCache->GetStorageShape(),
                                                  params.kScaleCache->GetViewStrides(),
                                                  params.kScaleCache->GetViewOffset());
    CHECK_RET(kScaleCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto l0Outs = l0op::QkvRmsNormRopeCacheWithKScale(
        params.qkv, params.qGamma, params.kGamma, params.cosSin, params.slotMapping,
        kCacheView, vCacheView, kScaleCacheView, params.cuSeqLen, params.actualSeqLens,
        params.optionalQRotation, params.optionalKRotation, params.optionalVScale,
        params.headNums, params.inputLayout, params.epsilon, params.qOut, params.qScale, l0Executor);

    CHECK_RET(std::get<0>(l0Outs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<1>(l0Outs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<2>(l0Outs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<3>(l0Outs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<4>(l0Outs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQkvRmsNormRopeCacheWithKScale);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "QkvRmsNormRopeCacheWithKScale execution failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
