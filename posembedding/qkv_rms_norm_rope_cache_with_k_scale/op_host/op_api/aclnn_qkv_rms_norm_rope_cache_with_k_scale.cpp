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

using Params = QkvRmsNormRopeCacheWithKScaleCheck::QkvRmsNormRopeCacheWithKScaleParams;
using L0Outputs =
    std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *>;

static constexpr const char *DEFAULT_QKV_LAYOUT = "TND";
static constexpr const char *DEFAULT_Q_OUT_LAYOUT = "NTD";

static const char *GetLayoutQkvOrDefault(const char *layout)
{
    return layout == nullptr || layout[0] == '\0' ? DEFAULT_QKV_LAYOUT : layout;
}

static const char *GetLayoutQOutOrDefault(const char *layout)
{
    return layout == nullptr || layout[0] == '\0' ? DEFAULT_Q_OUT_LAYOUT : layout;
}

static Params MakeParams(const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma,
                         const aclTensor *cosSin, const aclTensor *slotMapping, aclTensor *kCache, aclTensor *vCache,
                         aclTensor *kScaleCache, const aclTensor *queryStartLoc, const aclTensor *seqLens,
                         const aclTensor *rotationOptional, const aclTensor *vScaleOptional,
                         const aclIntArray *headNums, const char *layoutQkv, const char *layoutQOut, float epsilon,
                         aclTensor *qOut, aclTensor *qScale, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    Params params;
    params.qkv = qkv;
    params.qGamma = qGamma;
    params.kGamma = kGamma;
    params.cosSin = cosSin;
    params.slotMapping = slotMapping;
    params.kCache = kCache;
    params.vCache = vCache;
    params.kScaleCache = kScaleCache;
    params.queryStartLoc = queryStartLoc;
    params.seqLens = seqLens;
    params.rotationOptional = rotationOptional;
    params.vScaleOptional = vScaleOptional;
    params.headNums = headNums;
    params.layoutQkv = GetLayoutQkvOrDefault(layoutQkv);
    params.layoutQOut = GetLayoutQOutOrDefault(layoutQOut);
    params.epsilon = epsilon;
    params.qOut = qOut;
    params.qScale = qScale;
    params.workspaceSize = workspaceSize;
    params.executor = executor;
    return params;
}

static aclnnStatus MakeInputContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ContiguousInputs(Params &params, aclOpExecutor *executor)
{
    CHECK_RET(MakeInputContiguous(params.qkv, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.qGamma, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.kGamma, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.cosSin, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.slotMapping, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.queryStartLoc, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.seqLens, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.rotationOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeInputContiguous(params.vScaleOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const Params &params)
{
    QkvRmsNormRopeCacheWithKScaleCheck::QkvRmsNormRopeCacheWithKScaleChecker checker;
    aclnnStatus checkRet = checker.CheckParams(params);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckL0Outputs(const L0Outputs &outputs)
{
    CHECK_RET(std::get<0>(outputs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<1>(outputs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<2>(outputs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<3>(outputs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<4>(outputs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CreateCacheViews(Params &params, aclOpExecutor *executor)
{
    params.kCache = executor->CreateView(params.kCache, params.kCache->GetViewShape(), params.kCache->GetStorageShape(),
                                         params.kCache->GetViewStrides(), params.kCache->GetViewOffset());
    CHECK_RET(params.kCache != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.vCache = executor->CreateView(params.vCache, params.vCache->GetViewShape(), params.vCache->GetStorageShape(),
                                         params.vCache->GetViewStrides(), params.vCache->GetViewOffset());
    CHECK_RET(params.vCache != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.kScaleCache = executor->CreateView(
        params.kScaleCache, params.kScaleCache->GetViewShape(), params.kScaleCache->GetStorageShape(),
        params.kScaleCache->GetViewStrides(), params.kScaleCache->GetViewOffset());
    CHECK_RET(params.kScaleCache != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus AddQkvRmsNormRopeCacheWithKScaleTask(Params &params, aclOpExecutor *executor)
{
    CHECK_RET(ContiguousInputs(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateCacheViews(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    auto l0Outs = l0op::QkvRmsNormRopeCacheWithKScale(
        params.qkv, params.qGamma, params.kGamma, params.cosSin, params.slotMapping, params.kCache, params.vCache,
        params.kScaleCache, params.queryStartLoc, params.seqLens, params.rotationOptional, params.vScaleOptional,
        params.headNums, params.layoutQkv, params.layoutQOut, params.epsilon, params.qOut, params.qScale, executor);
    return CheckL0Outputs(l0Outs);
}

} // namespace QkvRmsNormRopeCacheWithKScale

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma, const aclTensor *cosSin,
    const aclTensor *slotMapping, aclTensor *kCacheRef, aclTensor *vCacheRef, aclTensor *kScaleCacheRef,
    const aclTensor *queryStartLoc, const aclTensor *seqLens, const aclTensor *rotationOptional,
    const aclTensor *vScaleOptional, const aclIntArray *headNums, const char *layoutQkv, const char *layoutQOut,
    float epsilon, aclTensor *qOut, aclTensor *qScale, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const char *layoutQkvAttr = QkvRmsNormRopeCacheWithKScale::GetLayoutQkvOrDefault(layoutQkv);
    const char *layoutQOutAttr = QkvRmsNormRopeCacheWithKScale::GetLayoutQOutOrDefault(layoutQOut);
    L2_DFX_PHASE_1(aclnnQkvRmsNormRopeCacheWithKScale,
                   DFX_IN(qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef, kScaleCacheRef, queryStartLoc,
                          seqLens, rotationOptional, vScaleOptional, headNums, layoutQkvAttr, layoutQOutAttr,
                          epsilon),
                   DFX_OUT(qOut, qScale, kCacheRef, vCacheRef, kScaleCacheRef));

    auto params = QkvRmsNormRopeCacheWithKScale::MakeParams(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef, kScaleCacheRef, queryStartLoc, seqLens,
        rotationOptional, vScaleOptional, headNums, layoutQkvAttr, layoutQOutAttr, epsilon, qOut, qScale, workspaceSize,
        executor);
    aclnnStatus checkRet = QkvRmsNormRopeCacheWithKScale::CheckParams(params);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(QkvRmsNormRopeCacheWithKScale::AddQkvRmsNormRopeCacheWithKScaleTask(params, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQkvRmsNormRopeCacheWithKScale);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "QkvRmsNormRopeCacheWithKScale execution failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
