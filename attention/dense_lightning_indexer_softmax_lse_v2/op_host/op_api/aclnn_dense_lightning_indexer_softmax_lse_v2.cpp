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
 * \file aclnn_dense_lightning_indexer_softmax_lse_v2.cpp
 * \brief
 */

#include "aclnn_dense_lightning_indexer_softmax_lse_v2.h"
#include "l0_dense_lightning_indexer_softmax_lse_v2.h"

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

static inline aclnnStatus InvalidTensorDimCheck(const aclTensor *queryIndex, const aclTensor *keyIndex,
                                                [[maybe_unused]] const aclTensor *weight,
                                                const aclTensor *cuSeqLensQOptional,
                                                const aclTensor *cuSeqLensKOptional,
                                                [[maybe_unused]] const aclTensor *seqUsedQOptional,
                                                [[maybe_unused]] const aclTensor *seqUsedKOptional,
                                                [[maybe_unused]] const aclTensor *cmpResidualKOptional,
                                                const char *layoutQ, const char *layoutK,
                                                int64_t maskMode, int64_t cmpRatio)
{
    std::string layoutQStr = (layoutQ != nullptr) ? std::string(layoutQ) : "";
    std::string layoutKStr = (layoutK != nullptr) ? std::string(layoutK) : "";
    bool isTndQ = (layoutQStr == "TND");
    bool isTndK = (layoutKStr == "TND");

    // layout 校验
    OP_CHECK(layoutQStr == "BSND" || layoutQStr == "TND",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutQ should be BSND or TND, but got %s.", layoutQStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(layoutKStr == "BSND" || layoutKStr == "TND",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutK should be BSND or TND, but got %s.", layoutKStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    // maskMode 校验：仅支持 0 (defaultMask) 或 3 (rightDownCausal)
    OP_CHECK(maskMode == 0 || maskMode == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskMode only supports 0 or 3, but got %ld.", maskMode),
             return ACLNN_ERR_PARAM_INVALID);

    // cmpRatio 校验：取值范围 1~128
    OP_CHECK(cmpRatio >= 1 && cmpRatio <= 128,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpRatio must be in [1, 128], but got %ld.", cmpRatio),
             return ACLNN_ERR_PARAM_INVALID);

    // query 维度校验：BSND 4维 [B,S1,N1,D]，TND 3维 [T1,N1,D]
    auto qShape = queryIndex->GetViewShape();
    OP_CHECK(!isTndQ || qShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND layout, queryIndex dim num should be 3, but got %zu.",
                     qShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(isTndQ || qShape.GetDimNum() == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSND layout, queryIndex dim num should be 4, but got %zu.",
                     qShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    // key 维度校验：BSND 4维 [B,S2,1,D]，TND 3维 [T2,1,D]
    auto kShape = keyIndex->GetViewShape();
    OP_CHECK(!isTndK || kShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND layout, keyIndex dim num should be 3, but got %zu.",
                     kShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(isTndK || kShape.GetDimNum() == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSND layout, keyIndex dim num should be 4, but got %zu.",
                     kShape.GetDimNum()),
             return ACLNN_ERR_PARAM_INVALID);

    // N2 固定为 1
    int64_t n2Idx = isTndK ? 1 : 2;
    OP_CHECK(kShape.GetDim(n2Idx) == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "keyIndex N2 dim must be 1, but got %ld.", kShape.GetDim(n2Idx)),
             return ACLNN_ERR_PARAM_INVALID);

    // D 维一致
    int64_t qDIdx = isTndQ ? 2 : 3;
    int64_t kDIdx = isTndK ? 2 : 3;
    OP_CHECK(qShape.GetDim(qDIdx) == kShape.GetDim(kDIdx),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "queryIndex and keyIndex D dim must be equal, but got qD=%ld kD=%ld.",
                     qShape.GetDim(qDIdx), kShape.GetDim(kDIdx)),
             return ACLNN_ERR_PARAM_INVALID);

    // query 和 key dtype 一致
    OP_CHECK(queryIndex->GetDataType() == keyIndex->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "queryIndex and keyIndex dtype must be the same."),
             return ACLNN_ERR_PARAM_INVALID);

    // TND layout 下 cuSeqLensQ/K 必须传入
    OP_CHECK(!isTndQ || cuSeqLensQOptional != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLensQ must be provided when layoutQ is TND."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(!isTndK || cuSeqLensKOptional != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLensK must be provided when layoutK is TND."),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize(
    const aclTensor *queryIndex,
    const aclTensor *keyIndex,
    const aclTensor *weight,
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *metadataOptional,
    const char *layoutQ,
    const char *layoutK,
    int64_t maskMode,
    int64_t cmpRatio,
    const aclTensor *softmaxLseOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnDenseLightningIndexerSoftmaxLseV2,
                   DFX_IN(queryIndex, keyIndex, weight,
                          cuSeqLensQOptional, cuSeqLensKOptional,
                          seqUsedQOptional, seqUsedKOptional,
                          cmpResidualKOptional, metadataOptional,
                          layoutQ, layoutK, maskMode, cmpRatio),
                   DFX_OUT(softmaxLseOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(queryIndex, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(keyIndex, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(softmaxLseOut, return ACLNN_ERR_PARAM_NULLPTR);

    auto ret = InvalidTensorDimCheck(queryIndex, keyIndex, weight,
                                     cuSeqLensQOptional, cuSeqLensKOptional,
                                     seqUsedQOptional, seqUsedKOptional,
                                     cmpResidualKOptional,
                                     layoutQ, layoutK, maskMode, cmpRatio);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *queryIndexContiguous = nullptr;
    const aclTensor *keyIndexContiguous = nullptr;
    const aclTensor *weightContiguous = nullptr;
    const aclTensor *cuSeqLensQContiguous = nullptr;
    const aclTensor *cuSeqLensKContiguous = nullptr;
    const aclTensor *seqUsedQContiguous = nullptr;
    const aclTensor *seqUsedKContiguous = nullptr;
    const aclTensor *cmpResidualKContiguous = nullptr;
    const aclTensor *metadataContiguous = nullptr;

    ret = ContiguousTensorWithCheck(queryIndex, &queryIndexContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(keyIndex, &keyIndexContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(weight, &weightContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(cuSeqLensQOptional, &cuSeqLensQContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(cuSeqLensKOptional, &cuSeqLensKContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(seqUsedQOptional, &seqUsedQContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(seqUsedKOptional, &seqUsedKContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(cmpResidualKOptional, &cmpResidualKContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(metadataOptional, &metadataContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    l0op::DenseLightningIndexerSoftmaxLseV2(
        queryIndexContiguous, keyIndexContiguous, weightContiguous,
        cuSeqLensQContiguous, cuSeqLensKContiguous,
        seqUsedQContiguous, seqUsedKContiguous,
        cmpResidualKContiguous, metadataContiguous,
        layoutQ, layoutK, maskMode, cmpRatio,
        softmaxLseOut,
        uniqueExecutor.get());

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();

    OP_LOGI("aclnnDenseLightningIndexerSoftmaxLseV2", "GetWorkspaceSize completed, workspaceSize=%lu, metadata=%s.",
            *workspaceSize, metadataContiguous != nullptr ? "provided" : "nullptr");

    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnDenseLightningIndexerSoftmaxLseV2(void *workspace,
                                        uint64_t workspaceSize,
                                        aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDenseLightningIndexerSoftmaxLseV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
