/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_compressor_grad.cpp
 * \brief
 */

#include "aclnn_compressor_grad.h"
#include "l0_compressor_grad.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"

#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static inline aclnnStatus ContiguousTensorWithCheck(const aclTensor *inputTensor, const aclTensor **outTensor,
                                                    aclOpExecutor *executor)
{
    if (inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() != 0) {
        *outTensor = l0op::Contiguous(inputTensor, executor);
        CHECK_RET(*outTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus InvalidTensorDimCheck(const aclTensor *x, const aclTensor *softmaxScore,
                                                const aclTensor *kv, const aclTensor *dCmpKv,
                                                const aclTensor *dX, const aclTensor *cuSeqLensOptional,
                                                const char *layoutX,
                                                const char *layoutDCmpKv, const char *layoutSoftmaxScore,
                                                const char *layoutKv, const char *layoutDX,
                                                int64_t coff, int64_t cmpRatio)
{
    std::string layoutXStr = (layoutX != nullptr) ? std::string(layoutX) : "";
    bool isTH_X = (layoutXStr == "TH");
    std::string layoutSoftmaxScoreStr = (layoutSoftmaxScore != nullptr) ? std::string(layoutSoftmaxScore) : "";
    bool isTCD_softmaxScoreStr = (layoutSoftmaxScore == "TCD");
    std::string layoutKvStr = (layoutKv != nullptr) ? std::string(layoutKv) : "";
    bool isTCD_kvStr = (layoutKv == "TCD");
    std::string layoutDCmpKvStr = (layoutDCmpKv != nullptr) ? std::string(layoutDCmpKv) : "";
    bool isTD_DCmpKvStr = (layoutKv == "TD");
    std::string layoutDXStr = (layoutDX != nullptr) ? std::string(layoutDX) : "";
    bool isTH_DXStr = (layoutDX == "TH");

    // layout 校验
    OP_CHECK(layoutXStr == "BSH" || layoutXStr == "TH",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutX should be BSH or TH, but got %s.", layoutXStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(layoutSoftmaxScoreStr == "BSCD" || layoutSoftmaxScoreStr == "TCD",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutX should be BSCD or TCD, but got %s.", layoutSoftmaxScoreStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(layoutKvStr == "BSCD" || layoutKvStr == "TCD",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutX should be BSCD or TCD, but got %s.", layoutKvStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(layoutDCmpKvStr == "BSD" || layoutDCmpKvStr == "TD",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutX should be BSCD or TCD, but got %s.", layoutDCmpKvStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(layoutDXStr == "BSH" || layoutDXStr == "TH",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutDX should be BSH or TH, but got %s.", layoutXStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    // cmpRatio 校验：取值范围 1~128
    OP_CHECK(cmpRatio >= 2 && cmpRatio <= 128,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpRatio must be in [2, 128], but got %ld.", cmpRatio),
             return ACLNN_ERR_PARAM_INVALID);
    
    // cmpRatio 校验：取值范围 1~128
    OP_CHECK(coff == 2 && coff == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "coff must be 1 or 2, but got %ld.", coff),
             return ACLNN_ERR_PARAM_INVALID);

    // query 维度校验：BSND 4维 [B,S1,N1,D]，TND 3维 [T1,N1,D]
    auto xShape = x->GetViewShape();
    OP_CHECK(isTH_X && xShape.GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TH layout, x dim num should be 2, but got %zu.",
                     xShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(!isTH_X && xShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSH layout, x dim num should be 2, but got %zu.",
                     xShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);
    
    auto softmaxScoreShape = softmaxScore->GetViewShape();
    OP_CHECK(isTCD_softmaxScoreStr && softmaxScoreShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TCD layout, softmaxScore dim num should be 3, but got %zu.",
                     softmaxScoreShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(!isTCD_softmaxScoreStr && softmaxScoreShape.GetDimNum() == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSCD layout, softmaxScore dim num should be 4, but got %zu.",
                     softmaxScoreShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);
    
    auto kvShape = kv->GetViewShape();
    OP_CHECK(isTCD_kvStr && kvShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TCD layout, kv dim num should be 3, but got %zu.",
                     kvShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(!isTCD_kvStr && kvShape.GetDimNum() == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSCD layout, kv dim num should be 4, but got %zu.",
                     kvShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);
    
    auto dCmpKvShape = dCmpKv->GetViewShape();
    OP_CHECK(isTD_DCmpKvStr && dCmpKvShape.GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TD layout, dCmpKv dim num should be 2, but got %zu.",
                    dCmpKvShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(!isTD_DCmpKvStr && dCmpKvShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSD layout, dCmpKv dim num should be 3, but got %zu.",
                     dCmpKvShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);
    
    auto dXShape = dX->GetViewShape();
    OP_CHECK(isTH_DXStr && xShape.GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TH layout, dX dim num should be 2, but got %zu.",
                     dXShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(!isTH_DXStr && xShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSH layout, dX dim num should be 2, but got %zu.",
                     dXShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    // TND layout 下 cuSeqLensQ/K 必须传入
    OP_CHECK(isTH_X && cuSeqLensOptional != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLens must be provided when layoutX is TH."),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCompressorGradGetWorkspaceSize(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
    const aclTensor *dCmpKv, const aclTensor *softmaxScore, const aclTensor *kv,
    const aclTensor *cuSeqLensOptional, const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional, const char *layoutX, const char *layoutDCmpKv, 
    const char *layoutSoftmaxScore, const char *layoutKv,
    const char *layoutDX, int64_t coff, int64_t cmpRatio,
    const aclTensor *dX, const aclTensor *dWkv, const aclTensor *dWgate,
    const aclTensor *dApe, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCompressorGrad,
                   DFX_IN(x, wkv, wgate, dCmpKv, softmaxScore, kv,
                          cuSeqLensOptional, sequsedOptional, startPosOptional,
                          layoutX, layoutDCmpKv, layoutSoftmaxScore, layoutKv, layoutDX,
                          coff, cmpRatio),
                   DFX_OUT(dX, dWkv, dWgate, dApe));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(wkv, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(wgate, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(dCmpKv, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(softmaxScore, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(kv, return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_NULL(dX, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(dWkv, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(dWgate, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(dApe, return ACLNN_ERR_PARAM_NULLPTR);

    auto ret = InvalidTensorDimCheck(x, softmaxScore, kv, dCmpKv, dX,
                                     cuSeqLensOptional, layoutX, layoutDCmpKv,
                                     layoutSoftmaxScore, layoutKv, layoutDX, coff, cmpRatio);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *xContiguous = nullptr;
    const aclTensor *wkvContiguous = nullptr;
    const aclTensor *wgateContiguous = nullptr;
    const aclTensor *dCmpKvContiguous = nullptr;
    const aclTensor *softmaxScoreContiguous = nullptr;
    const aclTensor *kvContiguous = nullptr;
    const aclTensor *cuSeqLensContiguous = nullptr;
    const aclTensor *sequsedOptionalContiguous = nullptr;
    const aclTensor *startPosOptionalContiguous = nullptr;

    ret = ContiguousTensorWithCheck(x, &xContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(wkv, &wkvContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(wgate, &wgateContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(dCmpKv, &dCmpKvContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(softmaxScore, &softmaxScoreContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(kv, &kvContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(cuSeqLensOptional, &cuSeqLensContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(sequsedOptional, &sequsedOptionalContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(startPosOptional, &startPosOptionalContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    l0op::CompressorGrad(
        xContiguous, wkvContiguous, wgateContiguous,
        dCmpKvContiguous, softmaxScoreContiguous, kvContiguous,
        cuSeqLensContiguous, sequsedOptionalContiguous,
        startPosOptionalContiguous, layoutX, layoutDCmpKv,
        layoutSoftmaxScore, layoutKv,
        layoutDX, coff, cmpRatio, dX, dWkv, dWgate, dApe,
        uniqueExecutor.get());

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();

    OP_LOGI("aclnnCompressorGrad", "GetWorkspaceSize completed, workspaceSize=%lu.", *workspaceSize);

    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnCompressorGrad(void *workspace,
                    uint64_t workspaceSize,
                    aclOpExecutor *executor,
                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCompressorGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
