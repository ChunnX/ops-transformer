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
 * \file aclnn_generic_block_sparse_attention_grad.cpp
 * \brief L2 aclnn API for GenericBlockSparseAttentionGrad.
 */

#include "aclnn_generic_block_sparse_attention_grad.h"
#include "generic_block_sparse_attention_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr int64_t GSAG_MAX_HEAD_NUM = 128;
static constexpr int64_t GSAG_SUPPORTED_MASK_TYPE = 1;
static constexpr int64_t GSAG_SUPPORTED_SOFTMAX_PRECISION = 0;

aclnnStatus CheckRequiredTensor(const aclTensor *t, const char *name)
{
    if (t == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GenericBlockSparseAttentionGrad: %s is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (t->GetViewShape().GetDimNum() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GenericBlockSparseAttentionGrad: %s is empty.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

int64_t GetHeadNumFromQuery(const aclTensor *query, const char *layout)
{
    const auto &shape = query->GetViewShape();
    if (strcmp(layout, "TND") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BNSD") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BSND") == 0) {
        return shape.GetDimNum() >= 3 ? shape.GetDim(2) : -1;
    }
    return -1;
}

int64_t GetHeadNumFromKey(const aclTensor *key, const char *layout)
{
    const auto &shape = key->GetViewShape();
    if (strcmp(layout, "TND") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BNSD") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BSND") == 0) {
        return shape.GetDimNum() >= 3 ? shape.GetDim(2) : -1;
    }
    return -1;
}

bool IsSupportedLayout(const char *layout)
{
    return layout != nullptr &&
           (strcmp(layout, "TND") == 0 || strcmp(layout, "BNSD") == 0 || strcmp(layout, "BSND") == 0);
}

aclnnStatus Validate(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout,
                     const aclTensor *out, const aclTensor *lse, const aclTensor *sparseBlockIdx,
                     const aclTensor *sparseBlockCount, const aclTensor *metadataOptional,
                     const aclTensor *attenMaskOptional, const aclIntArray *blockShape, char *layoutQ, char *layoutKv,
                     int64_t layoutSparsePattern, int64_t maskType, int64_t softmaxPrecision, int64_t winLeft,
                     int64_t winRight, int64_t residualBlockMode, bool isConsistentTopk, const aclTensor *dQuery,
                     const aclTensor *dKey, const aclTensor *dValue)
{
    const aclTensor *required[] = {query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount,
                                  metadataOptional, dQuery, dKey, dValue};
    const char *requiredNames[] = {"query", "key", "value", "dout", "out", "lse", "sparseBlockIdx",
                                  "sparseBlockCount", "metadata", "dQuery", "dKey", "dValue"};
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        aclnnStatus st = CheckRequiredTensor(required[i], requiredNames[i]);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }
    if (layoutQ == nullptr || layoutKv == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GenericBlockSparseAttentionGrad: layoutQ and layoutKv must not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strcmp(layoutQ, layoutKv) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: layoutQ(%s) must equal layoutKv(%s).", layoutQ, layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsSupportedLayout(layoutQ)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: layout must be TND/BSND/BNSD, got layoutQ=%s.", layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutSparsePattern != 1 || residualBlockMode != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: only layoutSparsePattern=1 and residualBlockMode=0 are supported, "
                "got %ld and %ld.",
                layoutSparsePattern, residualBlockMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Both TopK hints use the same per-block sparse-count implementation.
    (void)isConsistentTopk;
    if (maskType != GSAG_SUPPORTED_MASK_TYPE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: only support maskType == %ld, got %ld.",
                GSAG_SUPPORTED_MASK_TYPE, maskType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (softmaxPrecision != GSAG_SUPPORTED_SOFTMAX_PRECISION) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: only support softmaxPrecision == %ld, got %ld.",
                GSAG_SUPPORTED_SOFTMAX_PRECISION, softmaxPrecision);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (winLeft != -1 || winRight != -1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: only support winLeft == -1 and winRight == -1, got %ld, %ld.",
                winLeft, winRight);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (attenMaskOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: attenMask must be nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockShape != nullptr) {
        if (blockShape->Size() != 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "GenericBlockSparseAttentionGrad: blockShape must contain exactly two elements [x, y], got size %zu.",
                    blockShape->Size());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((*blockShape)[0] != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "GenericBlockSparseAttentionGrad: only support blockShape[0] == 1, got %ld.", (*blockShape)[0]);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((*blockShape)[1] < 128 || ((*blockShape)[1] % 64 != 0)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "GenericBlockSparseAttentionGrad: blockShape[1] must be >= 128 and 64-aligned, got %ld.",
                    (*blockShape)[1]);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t qHeadNum = GetHeadNumFromQuery(query, layoutQ);
    const int64_t kvHeadNum = GetHeadNumFromKey(key, layoutKv);
    if (qHeadNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: failed to get qHeadNum from query shape[%s] with layout=%s.",
                op::ToString(query->GetViewShape()).GetString(), layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qHeadNum > GSAG_MAX_HEAD_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: qHeadNum=%ld must be in [1, %ld], layout=%s, query shape[%s].",
                qHeadNum, GSAG_MAX_HEAD_NUM, layoutQ, op::ToString(query->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kvHeadNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: failed to get kvHeadNum from key shape[%s] with layout=%s.",
                op::ToString(key->GetViewShape()).GetString(), layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kvHeadNum > GSAG_MAX_HEAD_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: kvHeadNum=%ld must be in [1, %ld], layout=%s, key shape[%s].",
                kvHeadNum, GSAG_MAX_HEAD_NUM, layoutKv, op::ToString(key->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qHeadNum % kvHeadNum != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: qHeadNum(%ld) must be divisible by kvHeadNum(%ld).", qHeadNum,
                kvHeadNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    DataType qDtype = query->GetDataType();
    if (qDtype != ACL_FLOAT16 && qDtype != ACL_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: query dtype must be FP16 or BF16, got %s.",
                op::ToString(qDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (key->GetDataType() != qDtype || value->GetDataType() != qDtype || dout->GetDataType() != qDtype ||
        out->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: key/value/dout/out dtype must match query (%s), "
                "got key=%s value=%s dout=%s out=%s.",
                op::ToString(qDtype).GetString(), op::ToString(key->GetDataType()).GetString(),
                op::ToString(value->GetDataType()).GetString(), op::ToString(dout->GetDataType()).GetString(),
                op::ToString(out->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dQuery->GetDataType() != qDtype || dKey->GetDataType() != qDtype || dValue->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: dQuery/dKey/dValue dtype must match query (%s), "
                "got dQuery=%s dKey=%s dValue=%s.",
                op::ToString(qDtype).GetString(), op::ToString(dQuery->GetDataType()).GetString(),
                op::ToString(dKey->GetDataType()).GetString(), op::ToString(dValue->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (lse->GetDataType() != ACL_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: lse dtype must be FP32, got %s.",
                op::ToString(lse->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseBlockIdx->GetDataType() != ACL_INT32 || sparseBlockCount->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: sparseBlockIdx and sparseBlockCount dtype must be INT32, "
                "got idx=%s cnt=%s.",
                op::ToString(sparseBlockIdx->GetDataType()).GetString(),
                op::ToString(sparseBlockCount->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metadataOptional->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: metadata dtype must be INT32, got %s.",
                op::ToString(metadataOptional->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout, const aclTensor *out,
    const aclTensor *lse, const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
    const aclTensor *metadataOptional, const aclTensor *attenMaskOptional, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclIntArray *blockShape, char *layoutQ, char *layoutKv, int64_t layoutSparsePattern, double scaleValue,
    int64_t maskType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight, int64_t residualBlockMode,
    bool isConsistentTopk, aclTensor *dQuery, aclTensor *dKey, aclTensor *dValue, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnGenericBlockSparseAttentionGrad,
                   DFX_IN(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadataOptional,
                          attenMaskOptional, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional,
                          sequsedKvOptional, blockShape, layoutQ, layoutKv, layoutSparsePattern, scaleValue, maskType,
                          softmaxPrecision, winLeft, winRight, residualBlockMode, isConsistentTopk),
                   DFX_OUT(dQuery, dKey, dValue));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = Validate(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadataOptional,
                        attenMaskOptional, blockShape, layoutQ, layoutKv, layoutSparsePattern, maskType,
                        softmaxPrecision, winLeft, winRight, residualBlockMode, isConsistentTopk, dQuery, dKey, dValue);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto queryC = l0op::Contiguous(query, uniqueExecutor.get());
    auto keyC = l0op::Contiguous(key, uniqueExecutor.get());
    auto valueC = l0op::Contiguous(value, uniqueExecutor.get());
    auto doutC = l0op::Contiguous(dout, uniqueExecutor.get());
    auto outC = l0op::Contiguous(out, uniqueExecutor.get());
    auto lseC = l0op::Contiguous(lse, uniqueExecutor.get());
    auto idxC = l0op::Contiguous(sparseBlockIdx, uniqueExecutor.get());
    auto cntC = l0op::Contiguous(sparseBlockCount, uniqueExecutor.get());
    auto metaC = l0op::Contiguous(metadataOptional, uniqueExecutor.get());
    CHECK_RET(queryC && keyC && valueC && doutC && outC && lseC && idxC && cntC && metaC, ACLNN_ERR_INNER_NULLPTR);

    // attenMaskOptional already required nullptr in Validate; keep placeholder for OP_INPUT slot.
    const aclTensor *attenC = nullptr;
    const aclTensor *cuQC = nullptr;
    if (cuSeqLengthsQOptional != nullptr) {
        cuQC = l0op::Contiguous(cuSeqLengthsQOptional, uniqueExecutor.get());
        CHECK_RET(cuQC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cuKvC = nullptr;
    if (cuSeqLengthsKvOptional != nullptr) {
        cuKvC = l0op::Contiguous(cuSeqLengthsKvOptional, uniqueExecutor.get());
        CHECK_RET(cuKvC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedQC = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQC = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        CHECK_RET(sequsedQC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedKvC = nullptr;
    if (sequsedKvOptional != nullptr) {
        sequsedKvC = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
        CHECK_RET(sequsedKvC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outs = l0op::GenericBlockSparseAttentionGrad(
        queryC, keyC, valueC, doutC, outC, lseC, idxC, cntC, metaC, attenC, cuQC, cuKvC, sequsedQC, sequsedKvC,
        blockShape, layoutQ, layoutKv, layoutSparsePattern, scaleValue, maskType, softmaxPrecision, winLeft, winRight,
        residualBlockMode, isConsistentTopk, uniqueExecutor.get());
    CHECK_RET(outs[0] != nullptr && outs[1] != nullptr && outs[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dqView = l0op::ViewCopy(outs[0], dQuery, uniqueExecutor.get());
    auto dkView = l0op::ViewCopy(outs[1], dKey, uniqueExecutor.get());
    auto dvView = l0op::ViewCopy(outs[2], dValue, uniqueExecutor.get());
    CHECK_RET(dqView != nullptr && dkView != nullptr && dvView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGrad(void *workspace,
                                                                                        uint64_t workspaceSize,
                                                                                        aclOpExecutor *executor,
                                                                                        const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGenericBlockSparseAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
