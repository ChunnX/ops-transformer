/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_generic_block_sparse_attention_grad.h"

#include <acl/acl.h>
#include <cmath>
#include <string>

#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

#include "generic_block_sparse_attention_grad.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct GenericParams {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;
    const aclTensor *dout = nullptr;
    const aclTensor *out = nullptr;
    const aclTensor *lse = nullptr;
    const aclTensor *sparseBlockIdx = nullptr;
    const aclTensor *sparseBlockCount = nullptr;
    const aclTensor *metadataOptional = nullptr;
    const aclTensor *attenMaskOptional = nullptr;
    const aclTensor *cuSeqLengthsQOptional = nullptr;
    const aclTensor *cuSeqLengthsKvOptional = nullptr;
    const aclTensor *sequsedQOptional = nullptr;
    const aclTensor *sequsedKvOptional = nullptr;
    const aclIntArray *blockShape = nullptr;
    char *layoutQ = nullptr;
    char *layoutKv = nullptr;
    int64_t isPackedGQA = 1;
    int64_t maskType = 0;
    double scaleValue = 0.0;
    int64_t softmaxPrecision = 0;
    int64_t winLeft = -1;
    int64_t winRight = -1;
    const aclTensor *dq = nullptr;
    const aclTensor *dk = nullptr;
    const aclTensor *dv = nullptr;
};

static aclnnStatus CheckRequiredTensors(const GenericParams &p)
{
    CHECK_RET(p.query != nullptr && p.key != nullptr && p.value != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(p.dout != nullptr && p.out != nullptr && p.lse != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(p.sparseBlockIdx != nullptr && p.sparseBlockCount != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(p.dq != nullptr && p.dk != nullptr && p.dv != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypeAndRank(const GenericParams &p)
{
    const auto qType = p.query->GetDataType();
    CHECK_COND(qType == ACL_FLOAT16 || qType == ACL_BF16, ACLNN_ERR_PARAM_INVALID,
               "query only supports FLOAT16 or BFLOAT16.");
    CHECK_COND(p.key->GetDataType() == qType && p.value->GetDataType() == qType,
               ACLNN_ERR_PARAM_INVALID, "query/key/value dtype must match.");
    CHECK_COND(p.dout->GetDataType() == qType && p.out->GetDataType() == qType,
               ACLNN_ERR_PARAM_INVALID, "dOut/out dtype must match query.");
    CHECK_COND(p.lse->GetDataType() == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "lse must be FLOAT32.");
    CHECK_COND(p.sparseBlockIdx->GetDataType() == ACL_INT32 &&
                   p.sparseBlockCount->GetDataType() == ACL_INT32,
               ACLNN_ERR_PARAM_INVALID, "sparse block tensors must be INT32.");
    CHECK_COND(p.sparseBlockIdx->GetViewShape().GetDimNum() == 4 &&
                   p.sparseBlockCount->GetViewShape().GetDimNum() == 3,
               ACLNN_ERR_PARAM_INVALID, "sparse block tensors must be rank 4/3.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckLayoutAndAttributes(const GenericParams &p)
{
    CHECK_COND(p.blockShape != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "blockShape must not be null.");
    CHECK_COND(p.layoutQ != nullptr && p.layoutKv != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "layoutQ/layoutKv must not be null.");
    const std::string qLayout = op::ToString(p.layoutQ).GetString();
    const std::string kvLayout = op::ToString(p.layoutKv).GetString();
    CHECK_COND((qLayout == "TND" || qLayout == "BNSD" || qLayout == "BSND") &&
                   (kvLayout == "TND" || kvLayout == "BNSD" || kvLayout == "BSND"),
               ACLNN_ERR_PARAM_INVALID, "layoutQ/layoutKv only support TND/BNSD/BSND.");
    CHECK_COND(qLayout == kvLayout, ACLNN_ERR_PARAM_INVALID,
               "layoutQ and layoutKv must match.");
    CHECK_COND(p.maskType == 0 || p.maskType == 1, ACLNN_ERR_PARAM_INVALID,
               "maskType only supports 0 or 1.");
    CHECK_COND(std::isfinite(p.scaleValue), ACLNN_ERR_PARAM_INVALID,
               "scaleValue must be finite.");
    if (p.isPackedGQA != 1 || p.softmaxPrecision != 0 || p.winLeft != -1 || p.winRight != -1) {
        OP_LOGW("Unsupported control attributes will be normalized by tiling: isPackedGQA=%ld, "
                "softmaxPrecision=%ld, winLeft=%ld, winRight=%ld.",
                p.isPackedGQA, p.softmaxPrecision, p.winLeft, p.winRight);
    }
    if (p.attenMaskOptional != nullptr) {
        OP_LOGW("attenMaskOptional is reserved and ignored in this implementation.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOptionalInputs(const GenericParams &p)
{
    if (p.metadataOptional != nullptr) {
        CHECK_COND(p.metadataOptional->GetDataType() == ACL_INT32 &&
                       p.metadataOptional->GetViewShape().GetDimNum() == 1 &&
                       p.metadataOptional->GetViewShape().GetDim(0) == 32 + 64 * 8,
                   ACLNN_ERR_PARAM_INVALID, "metadataOptional must be INT32[544].");
    }
    auto checkSequenceTensor = [](const aclTensor *value, bool isCuSeq,
                                  const char *name) -> aclnnStatus {
        if (value == nullptr) {
            return ACLNN_SUCCESS;
        }
        const op::DataType expected = isCuSeq ? op::DataType::DT_INT64 : op::DataType::DT_INT32;
        CHECK_COND(value->GetDataType() == expected && value->GetViewShape().GetDimNum() == 1 &&
                       value->GetViewShape().GetDim(0) > 0,
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must be a non-empty rank-1 tensor with the expected dtype.", name);
        return ACLNN_SUCCESS;
    };
    CHECK_RET(checkSequenceTensor(p.cuSeqLengthsQOptional, true, "cuSeqLengthsQOptional") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(checkSequenceTensor(p.cuSeqLengthsKvOptional, true, "cuSeqLengthsKvOptional") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(checkSequenceTensor(p.sequsedQOptional, false, "sequsedQOptional") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(checkSequenceTensor(p.sequsedKvOptional, false, "sequsedKvOptional") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateParams(const GenericParams &p)
{
    CHECK_RET(CheckRequiredTensors(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeAndRank(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckLayoutAndAttributes(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return CheckOptionalInputs(p);
}

static aclnnStatus MakeContiguous(GenericParams &p, aclOpExecutor *executor)
{
    p.query = l0op::Contiguous(p.query, executor);
    p.key = l0op::Contiguous(p.key, executor);
    p.value = l0op::Contiguous(p.value, executor);
    p.dout = l0op::Contiguous(p.dout, executor);
    p.out = l0op::Contiguous(p.out, executor);
    p.lse = l0op::Contiguous(p.lse, executor);
    p.sparseBlockIdx = l0op::Contiguous(p.sparseBlockIdx, executor);
    p.sparseBlockCount = l0op::Contiguous(p.sparseBlockCount, executor);
    CHECK_RET(p.query && p.key && p.value && p.dout && p.out && p.lse &&
                  p.sparseBlockIdx && p.sparseBlockCount,
              ACLNN_ERR_INNER_NULLPTR);
    if (p.metadataOptional != nullptr)
        p.metadataOptional = l0op::Contiguous(p.metadataOptional, executor);
    if (p.attenMaskOptional != nullptr)
        p.attenMaskOptional = l0op::Contiguous(p.attenMaskOptional, executor);
    if (p.cuSeqLengthsQOptional != nullptr)
        p.cuSeqLengthsQOptional = l0op::Contiguous(p.cuSeqLengthsQOptional, executor);
    if (p.cuSeqLengthsKvOptional != nullptr)
        p.cuSeqLengthsKvOptional = l0op::Contiguous(p.cuSeqLengthsKvOptional, executor);
    if (p.sequsedQOptional != nullptr)
        p.sequsedQOptional = l0op::Contiguous(p.sequsedQOptional, executor);
    if (p.sequsedKvOptional != nullptr)
        p.sequsedKvOptional = l0op::Contiguous(p.sequsedKvOptional, executor);
    return ACLNN_SUCCESS;
}
} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout,
    const aclTensor *out, const aclTensor *lse, const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount, const aclTensor *metadataOptional,
    const aclTensor *attenMaskOptional, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, const aclIntArray *blockShape,
    int64_t isPackedGQA, char *layoutQ, char *layoutKv, double scaleValue,
    int64_t maskType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight,
    aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    GenericParams p{query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount,
                    metadataOptional, attenMaskOptional, cuSeqLengthsQOptional,
                    cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, blockShape,
                    layoutQ, layoutKv, isPackedGQA, maskType, scaleValue, softmaxPrecision,
                    winLeft, winRight, dq, dk, dv};
    CHECK_RET(ValidateParams(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    // Populate thread-local L2 context (incl. aicNum/aivNum via ACL_RT_DEV_RES_CUBE_CORE/VECTOR_CORE)
    // so the tiling SoCInfo dict isn't empty; without this GetCoreNumAic() returns 0 at compile+runtime.
    L2_DFX_PHASE_1(aclnnGenericBlockSparseAttentionGrad,
                   DFX_IN(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount,
                          metadataOptional, attenMaskOptional, cuSeqLengthsQOptional,
                          cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, blockShape,
                          isPackedGQA, layoutQ, layoutKv, scaleValue, maskType, softmaxPrecision,
                          winLeft, winRight),
                   DFX_OUT(dq, dk, dv));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();
    CHECK_RET(MakeContiguous(p, executorImpl) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    const std::string qLayout = op::ToString(layoutQ).GetString();
    const std::string kvLayout = op::ToString(layoutKv).GetString();
    auto outputs = l0op::GenericBlockSparseAttentionGrad(
        p.query, p.key, p.value, p.dout, p.out, p.lse, p.sparseBlockIdx, p.sparseBlockCount,
        p.metadataOptional, p.attenMaskOptional, p.cuSeqLengthsQOptional,
        p.cuSeqLengthsKvOptional, p.sequsedQOptional, p.sequsedKvOptional, p.blockShape,
        isPackedGQA, const_cast<char *>(qLayout.c_str()), const_cast<char *>(kvLayout.c_str()),
        scaleValue, maskType, softmaxPrecision, winLeft, winRight, executorImpl);
    CHECK_RET(outputs[0] && outputs[1] && outputs[2], ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[0], dq, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[1], dk, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[2], dv, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGrad(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGenericBlockSparseAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
