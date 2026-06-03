/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_qkv_rms_norm_rope_cache_with_kscale.h"

#include <dlfcn.h>
#include <new>

#include "qkv_rms_norm_rope_cache_with_kscale.h"
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

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static aclnnStatus CheckQkvRmsNormRopeCacheWithKScaleParam(
    const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma, const aclTensor *cosSin,
    const aclTensor *slotMapping, const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kScaleCache,
    const aclTensor *cuSeqLen, const aclTensor *rotation, const aclTensor *vScale, const aclIntArray *headNums,
    aclTensor *qOut, aclTensor *qScale, aclTensor *kCacheOut, aclTensor *vCacheOut, aclTensor *kScaleCacheOut,
    const uint64_t *workspaceSize, aclOpExecutor *const* executor)
{
    CHECK_RET(qkv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(qGamma != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kGamma != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(cosSin != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(vCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kScaleCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(cuSeqLen != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(rotation != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(vScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(headNums != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(qOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(qScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kCacheOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(vCacheOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kScaleCacheOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus Contiguous(const aclTensor *&qkv, const aclTensor *&qGamma, const aclTensor *&kGamma,
                               const aclTensor *&cosSin, const aclTensor *&slotMapping, aclTensor *&kCache,
                               aclTensor *&vCache, aclTensor *&kScaleCache, const aclTensor *&cuSeqLen,
                               const aclTensor *&rotation, const aclTensor *&vScale, aclOpExecutor *executor)
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

    kCache = const_cast<aclTensor *>(l0op::Contiguous(kCache, executor));
    CHECK_RET(kCache != nullptr, ACLNN_ERR_INNER_NULLPTR);

    vCache = const_cast<aclTensor *>(l0op::Contiguous(vCache, executor));
    CHECK_RET(vCache != nullptr, ACLNN_ERR_INNER_NULLPTR);

    kScaleCache = const_cast<aclTensor *>(l0op::Contiguous(kScaleCache, executor));
    CHECK_RET(kScaleCache != nullptr, ACLNN_ERR_INNER_NULLPTR);

    cuSeqLen = l0op::Contiguous(cuSeqLen, executor);
    CHECK_RET(cuSeqLen != nullptr, ACLNN_ERR_INNER_NULLPTR);

    rotation = l0op::Contiguous(rotation, executor);
    CHECK_RET(rotation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    vScale = l0op::Contiguous(vScale, executor);
    CHECK_RET(vScale != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus InputDtypeCheck(const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma,
                                    const aclTensor *cosSin, const aclTensor *slotMapping, const aclTensor *kCache,
                                    const aclTensor *vCache, const aclTensor *kScaleCache, const aclTensor *cuSeqLen,
                                    const aclTensor *rotation, const aclTensor *vScale)
{
    if (qkv->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qkv only support dtype bf16, but got %s.",
                op::ToString(qkv->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qGamma->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qGamma only support dtype float, but got %s.",
                op::ToString(qGamma->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kGamma->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kGamma only support dtype float, but got %s.",
                op::ToString(kGamma->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (cosSin->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cosSin only support dtype float, but got %s.",
                op::ToString(cosSin->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (slotMapping->GetDataType() != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "slotMapping only support dtype int32, but got %s.",
                op::ToString(slotMapping->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kCache->GetDataType() != DataType::DT_FLOAT8_E4M3FN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kCache only support dtype float8_e4m3fn, but got %s.",
                op::ToString(kCache->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (vCache->GetDataType() != DataType::DT_FLOAT8_E4M3FN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "vCache only support dtype float8_e4m3fn, but got %s.",
                op::ToString(vCache->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kScaleCache->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kScaleCache only support dtype float, but got %s.",
                op::ToString(kScaleCache->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (cuSeqLen->GetDataType() != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLen only support dtype int32, but got %s.",
                op::ToString(cuSeqLen->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (rotation->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotation only support dtype bf16, but got %s.",
                op::ToString(rotation->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (vScale->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "vScale only support dtype float, but got %s.",
                op::ToString(vScale->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static bool CheckIsEmptyTensor(const aclTensor *qkv, const aclTensor *kCache, const aclTensor *vCache)
{
    int64_t qkvShapeSize = qkv->GetViewShape().GetShapeSize();
    int64_t kCacheShapeSize = kCache->GetViewShape().GetShapeSize();
    int64_t vCacheShapeSize = vCache->GetViewShape().GetShapeSize();
    if (qkvShapeSize == 0 || kCacheShapeSize == 0 || vCacheShapeSize == 0) {
        return true;
    }
    return false;
}
}

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma, const aclTensor *cosSin,
    const aclTensor *slotMapping, aclTensor *kCache, aclTensor *vCache, aclTensor *kScaleCache,
    const aclTensor *cuSeqLen, const aclTensor *rotation, const aclTensor *vScale, const aclIntArray *headNums,
    float epsilon, aclTensor *qOut, aclTensor *qScale, aclTensor *kCacheOut, aclTensor *vCacheOut,
    aclTensor *kScaleCacheOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQkvRmsNormRopeCacheWithKScale,
                   DFX_IN(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, cuSeqLen,
                          rotation, vScale, headNums, epsilon),
                   DFX_OUT(qOut, qScale, kCacheOut, vCacheOut, kScaleCacheOut));

    aclnnStatus paramCheckRet = CheckQkvRmsNormRopeCacheWithKScaleParam(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, cuSeqLen,
        rotation, vScale, headNums, qOut, qScale, kCacheOut, vCacheOut, kScaleCacheOut,
        workspaceSize, executor);
    if (paramCheckRet != ACLNN_SUCCESS) {
        return paramCheckRet;
    }

    if (CheckIsEmptyTensor(qkv, kCache, vCache)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[QkvRmsNormRopeCacheWithKScale] do not support empty tensors.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(InputDtypeCheck(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
                              cuSeqLen, rotation, vScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(Contiguous(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
                         cuSeqLen, rotation, vScale, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto l0Outs = l0op::QkvRmsNormRopeCacheWithKScale(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, cuSeqLen,
        rotation, vScale, headNums, epsilon, qOut, qScale, kCacheOut, vCacheOut, kScaleCacheOut, l0Executor);

    CHECK_RET(l0Outs != nullptr, ACLNN_ERR_INNER_NULLPTR);

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
