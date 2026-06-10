/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_stem_oam_prep_varlen_q.h"
#include "stem_oam_prep_varlen_q.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_error_check.h"
#include "opdev/platform.h"

namespace {

aclnnStatus CheckParams(
    const aclTensor* q, const aclTensor* qSeqLens, const aclTensor* cuSeqLensQ,
    /* * optional */ const aclTensor* qScale, const aclTensor* kScale,
    int64_t stemBlockSize, int64_t stemStride,
    const aclTensor* qFlat)
{
    if (q == nullptr || qSeqLens == nullptr || cuSeqLensQ == nullptr || qFlat == nullptr) {
        OP_LOGE("StemOamPrepVarlenQ", "required input/output tensor is nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto qDtype = q->GetDataType();
    if (qDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE("StemOamPrepVarlenQ", "q dtype must be FP8_e4m3fn, got %d", static_cast<int>(qDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qFlat->GetDataType() != ge::DT_BFLOAT16) {
        OP_LOGE("StemOamPrepVarlenQ", "qFlat dtype must be BF16, got %d", static_cast<int>(qFlat->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qSeqLens->GetDataType() != ge::DT_INT32) {
        OP_LOGE("StemOamPrepVarlenQ", "qSeqLens dtype must be INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (cuSeqLensQ->GetDataType() != ge::DT_INT32) {
        OP_LOGE("StemOamPrepVarlenQ", "cuSeqLensQ dtype must be INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qScale == nullptr) {
        OP_LOGE("StemOamPrepVarlenQ", "qScale must be provided for FP8 input");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qScale->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE("StemOamPrepVarlenQ", "qScale dtype must be FP32 for FP8 input");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kScale == nullptr) {
        OP_LOGE("StemOamPrepVarlenQ", "kScale must be provided for FP8 input");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kScale->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE("StemOamPrepVarlenQ", "kScale dtype must be FP32, got %d", static_cast<int>(kScale->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto kScaleShape = kScale->GetViewShape();
    if (kScaleShape.GetDimNum() != 1) {
        OP_LOGE("StemOamPrepVarlenQ", "kScale must be 1D [H_q], got %d dims", static_cast<int>(kScaleShape.GetDimNum()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (stemBlockSize <= 0 || stemStride <= 0) {
        OP_LOGE("StemOamPrepVarlenQ", "stemBlockSize and stemStride must be positive");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (stemStride > stemBlockSize) {
        OP_LOGE("StemOamPrepVarlenQ", "stemStride must be <= stemBlockSize");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

}

aclnnStatus aclnnStemOamPrepVarlenQGetWorkspaceSize(
    const aclTensor* q, const aclTensor* qSeqLens, const aclTensor* cuSeqLensQ,
    /* * optional */ const aclTensor* qScale, const aclTensor* kScale,
    int64_t stemBlockSize, int64_t stemStride,
    aclTensor* qFlat,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    ACLNN_CHECK(CheckParams(q, qSeqLens, cuSeqLensQ, qScale, kScale, stemBlockSize, stemStride, qFlat));

    auto uniqueExecutor = CREATE_EXECUTOR();
    uniqueExecutor->AddAclInput(q);
    uniqueExecutor->AddAclInput(qSeqLens);
    uniqueExecutor->AddAclInput(cuSeqLensQ);
    uniqueExecutor->AddAclInput(qScale);
    uniqueExecutor->AddAclInput(kScale);
    uniqueExecutor->AddAclOutput(qFlat);
    uniqueExecutor->AddAttr("stemBlockSize", stemBlockSize);
    uniqueExecutor->AddAttr("stemStride", stemStride);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStemOamPrepVarlenQ(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    auto ret = executor->RunKernel(workspace, workspaceSize, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE("StemOamPrepVarlenQ", "RunKernel failed, ret=%d", static_cast<int>(ret));
        return ret;
    }
    return ACLNN_SUCCESS;
}