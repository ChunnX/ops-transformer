/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include <cstring>
#include <string>
#include <vector>
#include "aclnn_sparse_flash_mla.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q, const aclTensor *oriKv, const aclTensor *cmpKv, const aclTensor *oriSparseIndices,
    const aclTensor *cmpSparseIndices, const aclTensor *oriBlockTable, const aclTensor *cmpBlockTable,
    const aclTensor *cuSeqLensQ, const aclTensor *cuSeqLensOriKv, const aclTensor *cuSeqLensCmpKv,
    const aclTensor *seqUsedQ, const aclTensor *seqUsedOriKv, const aclTensor *seqUsedCmpKv,
    const aclTensor *cmpResidualKv, const aclTensor *oriTopkLength, const aclTensor *cmpTopkLength,
    const aclTensor *sinks, const aclTensor *metadata, double softmaxScale, int64_t cmpRatio,
    int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, char *layoutQ,
    char *layoutKv, int64_t topkValueMode, bool returnSoftmaxLse, const aclTensor *attnOut,
    const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSparseFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            const aclrtStream stream);

class TensorHolder {
public:
    TensorHolder(const aclTensor *&output, aclDataType dataType)
    {
        inner_ = nullptr;
        if (output == nullptr) {
            std::vector<int64_t> shape = {0};
            int64_t addr = 0xff;
            inner_ = aclCreateTensor(shape.data(), shape.size(), dataType, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void *>(&addr));
            output = inner_;
        }
    }

    ~TensorHolder()
    {
        if (inner_ != nullptr) {
            aclDestroyTensor(inner_);
            inner_ = nullptr;
        }
    }

private:
    const aclTensor *inner_;
};

aclnnStatus CheckSparseFlashMlaRequired(const aclTensor *q, bool returnSoftmaxLse, const aclTensor *attnOut,
                                        const aclTensor *softmaxLse, const char *layoutQ, const char *layoutKv,
                                        int64_t cmpRatio)
{
    if (q == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "q cannot be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (attnOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "attn_out cannot be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (returnSoftmaxLse && softmaxLse == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "softmax_lse cannot be nullptr when return_softmax_lse is true.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (layoutQ == nullptr || layoutKv == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "layout_q and layout_kv cannot be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (cmpRatio < 0 || cmpRatio > 128) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be in range [0, 128], but got %ld.", cmpRatio);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q, const aclTensor *oriKv, const aclTensor *cmpKv, const aclTensor *oriSparseIndices,
    const aclTensor *cmpSparseIndices, const aclTensor *oriBlockTable, const aclTensor *cmpBlockTable,
    const aclTensor *cuSeqLensQ, const aclTensor *cuSeqLensOriKv, const aclTensor *cuSeqLensCmpKv,
    const aclTensor *seqUsedQ, const aclTensor *seqUsedOriKv, const aclTensor *seqUsedCmpKv,
    const aclTensor *cmpResidualKv, const aclTensor *oriTopkLength, const aclTensor *cmpTopkLength,
    const aclTensor *sinks, const aclTensor *metadata, double softmaxScale, int64_t cmpRatio,
    int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, char *layoutQ,
    char *layoutKv, int64_t topkValueMode, bool returnSoftmaxLse, const aclTensor *attnOut,
    const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSparseFlashMla,
                   DFX_IN(q, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                          cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv, seqUsedQ, seqUsedOriKv, seqUsedCmpKv,
                          cmpResidualKv, oriTopkLength, cmpTopkLength, sinks, metadata, softmaxScale, cmpRatio,
                          oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQ, layoutKv, topkValueMode,
                          returnSoftmaxLse),
                   DFX_OUT(attnOut, softmaxLse));

    auto ret = CheckSparseFlashMlaRequired(q, returnSoftmaxLse, attnOut, softmaxLse, layoutQ, layoutKv, cmpRatio);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t effectiveCmpRatio = (cmpRatio == 0) ? 1 : cmpRatio;
    TensorHolder softmaxLseHolder(softmaxLse, aclDataType::ACL_FLOAT);

    return aclnnInnerSparseFlashMlaGetWorkspaceSize(
        q, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqLensQ,
        cuSeqLensOriKv, cuSeqLensCmpKv, seqUsedQ, seqUsedOriKv, seqUsedCmpKv, cmpResidualKv, oriTopkLength,
        cmpTopkLength, sinks, metadata, softmaxScale, effectiveCmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
        oriWinRight, layoutQ, layoutKv, topkValueMode, returnSoftmaxLse, attnOut, softmaxLse, workspaceSize,
        executor);
}

aclnnStatus aclnnSparseFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseFlashMla);
    return aclnnInnerSparseFlashMla(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
