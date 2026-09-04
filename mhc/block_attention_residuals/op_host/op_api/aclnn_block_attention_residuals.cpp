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
 * \file aclnn_block_attention_residuals.cpp
 * \brief BlockAttentionResiduals ACLNN 两段式接口
 */
#include "aclnn_block_attention_residuals.h"
#include "block_attention_residuals.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct BlockAttentionResidualsParams {
    const aclTensor *partialBlock{nullptr};
    const aclTensor *blockRes{nullptr};
    const aclTensor *projWeight{nullptr};
    const aclTensor *normWeight{nullptr};
    int64_t validBlockNum{0};
    double normEps{1e-5};
    bool needBackward{false};
    const aclTensor *hiddenStates{nullptr};
    const aclTensor *invNorm{nullptr};
    const aclTensor *probs{nullptr};
};

// 输入/输出 hidden_states 支持 BF16 / FP16 / FP32（kernel 由编译期宏按 dtype 生成独立二进制）
static const std::initializer_list<op::DataType> BF16_TYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FLOAT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
constexpr int64_t MAX_NUM_BLOCKS = 100;

static inline bool CheckNotNull(const BlockAttentionResidualsParams &params)
{
    OP_CHECK_NULL(params.partialBlock, return false);
    OP_CHECK_NULL(params.blockRes, return false);
    OP_CHECK_NULL(params.projWeight, return false);
    OP_CHECK_NULL(params.normWeight, return false);
    OP_CHECK_NULL(params.hiddenStates, return false);
    if (params.needBackward) {
        OP_CHECK_NULL(params.invNorm, return false);
        OP_CHECK_NULL(params.probs, return false);
    }
    return true;
}

static inline bool CheckDtypeValid(const BlockAttentionResidualsParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.partialBlock, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.blockRes, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.projWeight, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.normWeight, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.hiddenStates, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(params.partialBlock, params.blockRes, return false);
    OP_CHECK_DTYPE_NOT_SAME(params.partialBlock, params.projWeight, return false);
    OP_CHECK_DTYPE_NOT_SAME(params.partialBlock, params.normWeight, return false);
    OP_CHECK_DTYPE_NOT_SAME(params.partialBlock, params.hiddenStates, return false);
    if (params.needBackward) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.invNorm, FLOAT_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.probs, FLOAT_TYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static inline bool CheckShapeValid(const BlockAttentionResidualsParams &params)
{
    const auto &partialBlockShape = params.partialBlock->GetViewShape();
    const auto &blockShape = params.blockRes->GetViewShape();
    const auto &projShape = params.projWeight->GetViewShape();
    const auto &normShape = params.normWeight->GetViewShape();
    const auto &hiddenShape = params.hiddenStates->GetViewShape();
    if (partialBlockShape.GetDimNum() != 2 || blockShape.GetDimNum() != 3 ||
        (projShape.GetDimNum() != 1 && projShape.GetDimNum() != 2) || normShape.GetDimNum() != 1 ||
        hiddenShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invalid tensor rank");
        return false;
    }
    const int64_t T = partialBlockShape.GetDim(0);
    const int64_t H = partialBlockShape.GetDim(1);
    const int64_t N = blockShape.GetDim(1);
    const int64_t B = blockShape.GetDim(1) + 1;
    if (params.validBlockNum != N) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only default valid_block_num=N is supported, got %ld", params.validBlockNum);
        return false;
    }
    if (T < 0 || H <= 0 || N < 1 || N > MAX_NUM_BLOCKS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape range requires T>=0, H>=1 and 1<=N<=%ld", MAX_NUM_BLOCKS);
        return false;
    }
    if (blockShape.GetDim(0) != T || blockShape.GetDim(2) != H) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "block_res shape must be [T,N,H]");
        return false;
    }
    const bool projShapeOk = (projShape.GetDimNum() == 1 && projShape.GetDim(0) == H) ||
                             (projShape.GetDimNum() == 2 && projShape.GetDim(0) == 1 && projShape.GetDim(1) == H);
    if (!projShapeOk || normShape.GetDim(0) != H) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "proj_weight must be [H] or [1,H], and norm_weight must be [H]");
        return false;
    }
    if (hiddenShape.GetDim(0) != T || hiddenShape.GetDim(1) != H) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hidden_states shape must be [T,H]");
        return false;
    }
    if (!params.needBackward) {
        return true;
    }
    const auto &invShape = params.invNorm->GetViewShape();
    const auto &probsShape = params.probs->GetViewShape();
    if (invShape.GetDimNum() != 2 || probsShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_norm/probs must be 2D [T,B]");
        return false;
    }
    if (invShape.GetDim(0) != T || invShape.GetDim(1) != B || probsShape.GetDim(0) != T || probsShape.GetDim(1) != B) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_norm/probs shape must be [%ld,%ld]", T, B);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(BlockAttentionResidualsParams &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(params), ACLNN_ERR_PARAM_INVALID);
    if (params.normEps <= 0.0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "normEps must be > 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("BlockAttentionResiduals check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(BlockAttentionResidualsParams &params)
{
    params.partialBlock->SetOriginalShape(params.partialBlock->GetViewShape());
    params.blockRes->SetOriginalShape(params.blockRes->GetViewShape());
    params.projWeight->SetOriginalShape(params.projWeight->GetViewShape());
    params.normWeight->SetOriginalShape(params.normWeight->GetViewShape());
    params.hiddenStates->SetOriginalShape(params.hiddenStates->GetViewShape());
    if (params.needBackward) {
        params.invNorm->SetOriginalShape(params.invNorm->GetViewShape());
        params.probs->SetOriginalShape(params.probs->GetViewShape());
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnBlockAttentionResidualsGetWorkspaceSize(const aclTensor *partialBlock, const aclTensor *blockRes,
                                                         const aclTensor *projWeight, const aclTensor *normWeight,
                                                         int64_t validBlockNum, double normEps, bool needBackward,
                                                         aclTensor *hiddenStates, aclTensor *invNorm, aclTensor *probs,
                                                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnBlockAttentionResiduals,
                   DFX_IN(partialBlock, blockRes, projWeight, normWeight, validBlockNum, normEps, needBackward),
                   DFX_OUT(hiddenStates, invNorm, probs));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    BlockAttentionResidualsParams params{partialBlock, blockRes,     projWeight,   normWeight, validBlockNum,
                                         normEps,      needBackward, hiddenStates, invNorm,    probs};
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (hiddenStates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto partialBlock_ = l0op::Contiguous(partialBlock, uniqueExecutor.get());
    CHECK_RET(partialBlock_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto blockRes_ = l0op::Contiguous(blockRes, uniqueExecutor.get());
    CHECK_RET(blockRes_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto projWeight_ = l0op::Contiguous(projWeight, uniqueExecutor.get());
    CHECK_RET(projWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto normWeight_ = l0op::Contiguous(normWeight, uniqueExecutor.get());
    CHECK_RET(normWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outRet = l0op::BlockAttentionResiduals(partialBlock_, blockRes_, projWeight_, normWeight_, validBlockNum,
                                                normEps, needBackward, uniqueExecutor.get());
    if (outRet[0] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyOut = l0op::ViewCopy(outRet[0], hiddenStates, uniqueExecutor.get());
    if (viewCopyOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (needBackward) {
        auto viewCopyInv = l0op::ViewCopy(outRet[1], invNorm, uniqueExecutor.get());
        CHECK_RET(viewCopyInv != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyProbs = l0op::ViewCopy(outRet[2], probs, uniqueExecutor.get());
        CHECK_RET(viewCopyProbs != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBlockAttentionResiduals(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockAttentionResiduals);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
