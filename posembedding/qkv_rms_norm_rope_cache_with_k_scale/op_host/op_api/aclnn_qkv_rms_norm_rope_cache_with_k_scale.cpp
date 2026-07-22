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

#include <cstring>
#include <vector>

#include "aclnn_kernels/common/op_error_check.h"
#include "qkv_rms_norm_rope_cache_with_k_scale.h"
// The checker uses opbase logging macros; restore the ACLNN macros for the bridge below.
#pragma push_macro("OpLogSub")
#pragma push_macro("OpLogErrSub")
#pragma push_macro("D_OP_LOGI")
#pragma push_macro("D_OP_LOGW")
#pragma push_macro("D_OP_LOGE")
#pragma push_macro("D_OP_LOGD")
#pragma push_macro("OP_LOGI")
#pragma push_macro("OP_LOGW")
#pragma push_macro("OP_LOGE_WITHOUT_REPORT")
#pragma push_macro("OP_LOGE")
#pragma push_macro("OP_LOGD")
#include "qkv_rms_norm_rope_cache_with_k_scale_checker.h"
#pragma pop_macro("OP_LOGD")
#pragma pop_macro("OP_LOGE")
#pragma pop_macro("OP_LOGE_WITHOUT_REPORT")
#pragma pop_macro("OP_LOGW")
#pragma pop_macro("OP_LOGI")
#pragma pop_macro("D_OP_LOGD")
#pragma pop_macro("D_OP_LOGE")
#pragma pop_macro("D_OP_LOGW")
#pragma pop_macro("D_OP_LOGI")
#pragma pop_macro("OpLogErrSub")
#pragma pop_macro("OpLogSub")
#include "aclnn_kernels/contiguous.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
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
static constexpr const char *DEFAULT_Q_QUANT_MODE = "PerTokenPerHead";
static constexpr const char *MROPE_Q_QUANT_MODE = "NoQuant";
static constexpr const char *ACLNN_NAME = "aclnnQkvRmsNormRopeCacheWithKScale";

static const char *GetLayoutQkvOrDefault(const char *layout)
{
    return layout == nullptr || layout[0] == '\0' ? DEFAULT_QKV_LAYOUT : layout;
}

static const char *GetLayoutQOutOrDefault(const char *layout)
{
    return layout == nullptr || layout[0] == '\0' ? DEFAULT_Q_OUT_LAYOUT : layout;
}

static const char *GetQQuantModeOrDefault(const char *qQuantMode)
{
    return qQuantMode == nullptr || qQuantMode[0] == '\0' ? DEFAULT_Q_QUANT_MODE : qQuantMode;
}

static aclnnStatus CheckMropePositionShape(const aclTensor *qkv, const aclTensor *mropePositionOptional,
                                            const char *layoutQkv)
{
    if (mropePositionOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const auto &shape = mropePositionOptional->GetViewShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 3) {
        std::string incorrectShape;
        for (uint64_t i = 0; i < shape.GetDimNum(); ++i) {
            if (i != 0) {
                incorrectShape += ", ";
            }
            incorrectShape += std::to_string(shape.GetDim(i));
        }
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            ACLNN_NAME, "mropePositionOptional", incorrectShape.c_str(),
            "mropePositionOptional must have logical shape [T, 3] (one token per row and T/H/W in columns)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qkv != nullptr) {
        const auto &qkvShape = qkv->GetViewShape();
        const char *layout = GetLayoutQkvOrDefault(layoutQkv);
        const bool isNtd = std::strcmp(layout, "NTD") == 0;
        const bool isTnd = std::strcmp(layout, "TND") == 0;
        if (qkvShape.GetDimNum() == 3 && (isNtd || isTnd)) {
            const uint64_t tokenAxis = isNtd ? 1 : 0;
            const int64_t tokenNum = qkvShape.GetDim(tokenAxis);
            if (shape.GetDim(0) != tokenNum) {
                const std::string incorrectShape = std::to_string(shape.GetDim(0)) + ", 3";
                const std::string reason = "mropePositionOptional must have shape [" + std::to_string(tokenNum) +
                                           ", 3] to match the qkv token dimension";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "mropePositionOptional", incorrectShape.c_str(),
                                                      reason.c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (mropePositionOptional->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            ACLNN_NAME, "mropePositionOptional",
            op::ToString(mropePositionOptional->GetDataType()).GetString(),
            "the dtype of mropePositionOptional must be INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (unlikely(op::IsPrivateFormat(mropePositionOptional->GetStorageFormat()))) {
        OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "mropePositionOptional",
                                   op::ToString(mropePositionOptional->GetStorageFormat()).GetString(), "ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSceneParams(const aclTensor *qkv, const aclTensor *queryStartLocOptional,
                                    const aclTensor *seqLensOptional,
                                    const aclTensor *mropePositionOptional,
                                    const aclIntArray *mropeSectionOptional, const char *qQuantMode,
                                    const char *layoutQkv,
                                    const aclTensor *qOut, const aclTensor *qScaleOptional)
{
    const bool hasMropePosition = mropePositionOptional != nullptr;
    const bool hasMropeSection = mropeSectionOptional != nullptr;
    if (hasMropePosition != hasMropeSection) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "mropePositionOptional/mropeSectionOptional",
                                              "presence mismatch", "both parameters must be provided together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qOut == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qOut", "nullptr", "qOut can not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (hasMropePosition) {
        aclnnStatus mropePositionCheck = CheckMropePositionShape(qkv, mropePositionOptional, layoutQkv);
        if (mropePositionCheck != ACLNN_SUCCESS) {
            return mropePositionCheck;
        }
        if (queryStartLocOptional != nullptr || seqLensOptional != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ACLNN_NAME, "queryStartLocOptional/seqLensOptional", "not nullptr",
                "both parameters must be nullptr in the M-RoPE scene");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (std::strcmp(qQuantMode, MROPE_Q_QUANT_MODE) != 0 || qScaleOptional != nullptr ||
            qOut->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ACLNN_NAME, "qQuantMode/qOut/qScaleOptional", "invalid combination",
                "the M-RoPE scene requires NoQuant, a BF16 qOut and a nullptr qScaleOptional");
            return ACLNN_ERR_PARAM_INVALID;
        }
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "mropePositionOptional/mropeSectionOptional", "present",
                                              "the M-RoPE scene is not supported yet");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (std::strcmp(qQuantMode, DEFAULT_Q_QUANT_MODE) != 0 ||
        qOut->GetDataType() != op::DataType::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            ACLNN_NAME, "qQuantMode/qOut", "invalid combination",
            "the RoPE scene requires PerTokenPerHead and an FP8 E4M3FN qOut");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
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
    const aclTensor *queryStartLocOptional, const aclTensor *seqLensOptional, const aclTensor *rotationOptional,
    const aclTensor *vScaleOptional, const aclTensor *mropePositionOptional, const aclIntArray *headNums,
    const char *layoutQkv, const char *layoutQOut, float epsilon, const aclIntArray *mropeSectionOptional,
    const char *qQuantMode, aclTensor *qOut, aclTensor *qScaleOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    const char *layoutQkvAttr = QkvRmsNormRopeCacheWithKScale::GetLayoutQkvOrDefault(layoutQkv);
    const char *layoutQOutAttr = QkvRmsNormRopeCacheWithKScale::GetLayoutQOutOrDefault(layoutQOut);
    const char *qQuantModeAttr = QkvRmsNormRopeCacheWithKScale::GetQQuantModeOrDefault(qQuantMode);
    L2_DFX_PHASE_1(aclnnQkvRmsNormRopeCacheWithKScale,
                   DFX_IN(qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef, kScaleCacheRef,
                          queryStartLocOptional, seqLensOptional, rotationOptional, vScaleOptional,
                          mropePositionOptional, headNums, layoutQkvAttr, layoutQOutAttr, epsilon,
                          mropeSectionOptional, qQuantModeAttr),
                   DFX_OUT(qOut, qScaleOptional, kCacheRef, vCacheRef, kScaleCacheRef));

    aclnnStatus sceneCheckRet = QkvRmsNormRopeCacheWithKScale::CheckSceneParams(
        qkv, queryStartLocOptional, seqLensOptional, mropePositionOptional, mropeSectionOptional, qQuantModeAttr,
        layoutQkvAttr, qOut, qScaleOptional);
    CHECK_RET(sceneCheckRet == ACLNN_SUCCESS, sceneCheckRet);

    auto params = QkvRmsNormRopeCacheWithKScale::MakeParams(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef, kScaleCacheRef, queryStartLocOptional,
        seqLensOptional, rotationOptional, vScaleOptional, headNums, layoutQkvAttr, layoutQOutAttr, epsilon, qOut,
        qScaleOptional, workspaceSize, executor);
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
