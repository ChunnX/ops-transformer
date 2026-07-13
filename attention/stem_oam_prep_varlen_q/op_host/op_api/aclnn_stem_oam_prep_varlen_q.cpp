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

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr int64_t DIM_QK = 128;
static constexpr int64_t MAX_BATCH_SIZE = 1024;

static aclnnStatus CheckNullptr(const aclTensor *q, const aclIntArray *qSeqLens,
                                const aclIntArray *cuSeqLensQ, const aclTensor *qFlat,
                                uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(q != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(qSeqLens != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(cuSeqLensQ != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(qFlat != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor *q, const aclIntArray *qSeqLens,
                              const aclIntArray *cuSeqLensQ, const aclTensor *qScale,
                              const aclTensor *qFlat)
{
    auto qShape = q->GetViewShape();
    if (qShape.GetDimNum() != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q dim num must be 3, but got %ld.", qShape.GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qShape.GetDim(2) != DIM_QK) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q last dim must be 128, but got %ld.", qShape.GetDim(2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (q->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q is empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    uint64_t batchSize = qSeqLens->Size();
    if (batchSize == 0 || batchSize > MAX_BATCH_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qSeqLens size must be in (0, %ld], but got %lu.",
                MAX_BATCH_SIZE, batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    uint64_t cuSeqLensQSize = cuSeqLensQ->Size();
    if (cuSeqLensQSize != batchSize + 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLensQ size must be batch+1=%lu, but got %lu.",
                batchSize + 1, cuSeqLensQSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t *cuData = cuSeqLensQ->GetData();
    if (cuData[0] != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLensQ[0] must be 0, but got %ld.", cuData[0]);
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (uint64_t i = 1; i < cuSeqLensQSize; i++) {
        if (cuData[i] < cuData[i - 1]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLensQ must be monotonically increasing.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (cuData[batchSize] != qShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLensQ[batch] must equal total_tokens, but got %ld vs %ld.",
                cuData[batchSize], qShape.GetDim(0));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qScale != nullptr) {
        auto qScaleShape = qScale->GetViewShape();
        if (qScaleShape.GetDimNum() != 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qScale dim num must be 2, but got %ld.",
                    qScaleShape.GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    auto qFlatShape = qFlat->GetViewShape();
    if (qFlatShape.GetDimNum() != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qFlat dim num must be 4, but got %ld.", qFlatShape.GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const aclTensor *q, const aclTensor *qScale, const aclTensor *qFlat)
{
    auto qDtype = q->GetDataType();
    if (qDtype != DataType::DT_FLOAT8_E4M3FN && qDtype != DataType::DT_BF16 &&
        qDtype != DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q dtype must be FLOAT8_E4M3FN, BF16 or FP16, but got %d.",
                static_cast<int>(qDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qScale != nullptr) {
        auto qScaleDtype = qScale->GetDataType();
        if (qScaleDtype != DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qScale dtype must be FLOAT, but got %d.",
                    static_cast<int>(qScaleDtype));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    auto qFlatDtype = qFlat->GetDataType();
    if (qFlatDtype != DataType::DT_BF16 && qFlatDtype != DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qFlat dtype must be BF16 or FP16, but got %d.",
                static_cast<int>(qFlatDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttr(int64_t stemBlockSize, int64_t stemStride)
{
    if (stemBlockSize % 32 != 0 || stemBlockSize > 256 || stemBlockSize <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "stemBlockSize must be positive, <= 256, and multiple of 32, but got %ld.",
                stemBlockSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (stemStride % 16 != 0 || stemStride > 64 || stemStride <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "stemStride must be positive, <= 64, and multiple of 16, but got %ld.",
                stemStride);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (stemStride > stemBlockSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "stemStride must be <= stemBlockSize, but got stride=%ld, blockSize=%ld.",
                stemStride, stemBlockSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (stemBlockSize % stemStride != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "stemBlockSize must be a multiple of stemStride, but got blockSize=%ld, stride=%ld.",
                stemBlockSize, stemStride);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

}  // namespace

aclnnStatus aclnnStemOamPrepVarlenQGetWorkspaceSize(
    const aclTensor *q,
    const aclIntArray *qSeqLens,
    const aclIntArray *cuSeqLensQ,
    const aclTensor *qScale,
    int64_t stemBlockSize,
    int64_t stemStride,
    aclTensor *qFlat,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnStemOamPrepVarlenQ,
                   DFX_IN(q, qSeqLens, cuSeqLensQ, qScale, stemBlockSize, stemStride),
                   DFX_OUT(qFlat));

    aclnnStatus ret = CheckNullptr(q, qSeqLens, cuSeqLensQ, qFlat, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = CheckAttr(stemBlockSize, stemStride);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = CheckShape(q, qSeqLens, cuSeqLensQ, qScale, qFlat);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = CheckDtype(q, qScale, qFlat);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorImpl = uniqueExecutor.get();

    q = l0op::Contiguous(q, executorImpl);
    CHECK_RET(q != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (qScale != nullptr) {
        qScale = l0op::Contiguous(qScale, executorImpl);
        CHECK_RET(qScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto output = l0op::StemOamPrepVarlenQ(q, qSeqLens, cuSeqLensQ, qScale,
                                           stemBlockSize, stemStride,
                                           executorImpl);
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(output, qFlat, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStemOamPrepVarlenQ(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStemOamPrepVarlenQ);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
