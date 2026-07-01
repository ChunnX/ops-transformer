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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

inline constexpr int64_t SLI_NO_MASK_MODE = 0;
inline constexpr int64_t SLI_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t SLI_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t SLI_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t SLI_NUM_HEADS_Q_LOWER_BOUND_A5 = 1;
inline constexpr int64_t SLI_NUM_HEADS_Q_UPPER_BOUND_A5 = 128;
inline constexpr int64_t SLI_TOPK_LOWER_BOUND_A5 = 1;
inline constexpr int64_t SLI_TOPK_UPPER_BOUND_A5 = 2048;
inline constexpr int64_t SLIKG_METADATA_SIZE = 64;

inline bool IsTensorExistSli(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumSli(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeSli(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

int64_t GetQueryBatchSizeSli(int64_t batchSize, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
    const char *layoutQOptional)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    if (IsTensorExistSli(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedQOptional 没传，使用cuSeqlensQOptional获取BatchSize
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistSli(cuSeqlensQOptional)) { // 前序校验已保证layout_q = TND时，cu_seqlens_q必须传入，此通路必达
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用batchSize
    return batchSize;
}

int64_t GetKeyBatchSizeSli(int64_t batchSize, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedKOptional,
    const char *layoutKOptional)
{
    // 1. 如果sequsedKOptional 传了，使用sequsedKOptional获取BatchSize
    if (IsTensorExistSli(sequsedKOptional)) {
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    // 如果是 TND，必须使用 cuSeqlensKOptional获取BatchSize
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistSli(cuSeqlensKOptional)) { // 前序校验已保证layout_k = TND时，cu_seqlens_k必须传入，此通路必达
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用batchSize
    return batchSize;
}

aclnnStatus CheckSingleParamSli(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, int64_t numHeadsQ,
    int64_t numHeadsK, int64_t headDim, int64_t topk, const char *layoutQOptional, const char *layoutKOptional,
    int64_t maskMode, int64_t cmpRatio, uint32_t aicCoreNum, uint32_t aivCoreNum, const std::string &socVersion)
{
    // num_heads_q 校验
    CHECK_COND(numHeadsQ >= SLI_NUM_HEADS_Q_LOWER_BOUND_A5 && numHeadsQ <= SLI_NUM_HEADS_Q_UPPER_BOUND_A5,
        ACLNN_ERR_PARAM_INVALID, "num_heads_q should be [%lld, %lld], but got %lld",
        SLI_NUM_HEADS_Q_LOWER_BOUND_A5, SLI_NUM_HEADS_Q_UPPER_BOUND_A5, numHeadsQ);
    // num_heads_k 校验
    CHECK_COND(numHeadsK == 1, ACLNN_ERR_PARAM_INVALID,
        "num_heads_kv should only be 1, but got %lld", numHeadsK);
    // head_dim 校验
    CHECK_COND(headDim == 128, ACLNN_ERR_PARAM_INVALID,
        "head_dim should only be 128, but got %lld", headDim);
    // topk 校验
    CHECK_COND(topk >= SLI_TOPK_LOWER_BOUND_A5 && topk <= SLI_TOPK_UPPER_BOUND_A5, ACLNN_ERR_PARAM_INVALID,
        "topk should be [%lld, %lld], but got %lld", SLI_TOPK_LOWER_BOUND_A5, SLI_TOPK_UPPER_BOUND_A5, topk);
    // batch_size 非负校验
    CHECK_COND(batchSize >= 0, ACLNN_ERR_PARAM_INVALID,
        "batch_size should be >= 0, but got %lld", batchSize);
    // max_seqlen_q 校验
    CHECK_COND(maxSeqlenQ >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_q should be >= 0, but got %lld", maxSeqlenQ);
    // max_seqlen_k 校验
    CHECK_COND(maxSeqlenK >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_k should be >= 0, but got %lld", maxSeqlenK);
    // mask_mode 校验
    CHECK_COND((maskMode == SLI_NO_MASK_MODE) || (maskMode == SLI_CAUSAL_MASK_MODE),
        ACLNN_ERR_PARAM_INVALID,
        "mask_mode should be %lld/%lld, but got %lld", SLI_NO_MASK_MODE, SLI_CAUSAL_MASK_MODE, maskMode);
    // cmp_ratio 校验
    CHECK_COND((cmpRatio >= SLI_CMP_RATIO_LOWER_BOUND) && (cmpRatio <= SLI_CMP_RATIO_UPPER_BOUND),
        ACLNN_ERR_PARAM_INVALID,
        "cmp_ratio should be between [%lld, %lld], but got %lld",
        SLI_CMP_RATIO_LOWER_BOUND, SLI_CMP_RATIO_UPPER_BOUND, cmpRatio);
    // layout_q 校验
    CHECK_COND((layoutQOptional != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_q is null!");
    CHECK_COND((strcmp(layoutQOptional, "TND") == 0) || (strcmp(layoutQOptional, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
        "layout_q must be TND or BSND, but got %s", layoutQOptional);
    // layout_k 校验
    CHECK_COND((layoutKOptional != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_k is null!");
    CHECK_COND((strcmp(layoutKOptional, "TND") == 0) || (strcmp(layoutKOptional, "BSND") == 0),
        ACLNN_ERR_PARAM_INVALID, "layout_k must be TND/BSND, but got %s", layoutKOptional);
    // 核心数校验
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSli(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *cmpResidualKOptional, const char *layoutQOptional,
    const char *layoutKOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistSli(cuSeqlensQOptional), ACLNN_ERR_PARAM_INVALID,
            "For layout_q TND, cu_seqlens_q must be provided!");
    }
    // cu_seqlens_k 存在性校验
    if (strcmp(layoutKOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistSli(cuSeqlensKOptional), ACLNN_ERR_PARAM_INVALID,
            "For layout_k TND, cu_seqlens_k must be provided!");
    }
    // cmp_residual_k 存在性校验
    if (cmpRatio != SLI_CMP_RATIO_LOWER_BOUND && maskMode == SLI_CAUSAL_MASK_MODE) {
        CHECK_COND(IsTensorExistSli(cmpResidualKOptional), ACLNN_ERR_PARAM_INVALID,
            "When cmp_ratio is not 1 and mask_mode is CAUSAL, cmp_residual_k must be provided!");
    }
    // metadata 存在性校验
    CHECK_COND(IsTensorExistSli(metadata), ACLNN_ERR_PARAM_INVALID,
        "Output metadata is nullptr!");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencySli(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, const char *layoutQOptional, const char *layoutKOptional,
    const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // 校验 cu_seqlens_q
    if (IsTensorExistSli(cuSeqlensQOptional)) {
        dimNum = GetDimNumSli(cuSeqlensQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSli(cuSeqlensQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistSli(cuSeqlensKOptional)) {
        dimNum = GetDimNumSli(cuSeqlensKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSli(cuSeqlensKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_q
    if (IsTensorExistSli(sequsedQOptional)) {
        dimNum = GetDimNumSli(sequsedQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSli(sequsedQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_k
    if (IsTensorExistSli(sequsedKOptional)) {
        dimNum = GetDimNumSli(sequsedKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSli(sequsedKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cmp_residual_k
    if (IsTensorExistSli(cmpResidualKOptional)) {
        dimNum = GetDimNumSli(cmpResidualKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID,
            "The dim num of cmp_residual_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSli(cmpResidualKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cmp_residual_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 metadata
    if (IsTensorExistSli(metadata)) {
        dimNum = GetDimNumSli(metadata);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSli(metadata);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of metadata must be int32, but got %d", static_cast<int32_t>(dataType));
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != SLIKG_METADATA_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %lld",
                SLIKG_METADATA_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验batch
    int64_t queryBatchSize = GetQueryBatchSizeSli(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional);
    int64_t keyBatchSize = GetKeyBatchSizeSli(batchSize, cuSeqlensKOptional, sequsedKOptional, layoutKOptional);
    CHECK_COND(queryBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
        "The batch_size obtained from query should be the same as that obtained from key, but got %lld and %lld",
        queryBatchSize, keyBatchSize);
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistSli(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(cuSeqlensQBatchSize == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_q is TND and seqused_q is passed, The batch_size obtained from cu_seqlens_q should be the "
            "same as that obtained from seqused_q, but got %lld and %lld", cuSeqlensQBatchSize, queryBatchSize);
    }
    // 校验TND场景k维度一致性
    if (strcmp(layoutKOptional, "TND") == 0 && IsTensorExistSli(sequsedKOptional)) {
        int64_t cuSeqlensKBatchSize = cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(cuSeqlensKBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_k is TND and seqused_k is passed, The batch_size obtained from cu_seqlens_k should be the "
            "same as that obtained from seqused_k, but got %lld and %lld", cuSeqlensKBatchSize, keyBatchSize);
    }
    // 校验 cmp_residual_k 元素数
    if (IsTensorExistSli(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        CHECK_COND(cmpResidualKBatch == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
            "The batch_size of cmp_residual_k should match the valid batch size, but got %lld and %lld",
            cmpResidualKBatch, queryBatchSize);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckSliA5(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, char *layoutQOptional,
    char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, const aclTensor *metadata, uint32_t aicCoreNum,
    uint32_t aivCoreNum, const std::string &socVersion)
{
    auto ret = CheckSingleParamSli(batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, topk,
        layoutQOptional, layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceSli(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, cmpResidualKOptional,
        layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencySli(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
        sequsedKOptional, cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckSli(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, char *layoutQOptional,
    char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, const aclTensor *metadata, uint32_t aicCoreNum,
    uint32_t aivCoreNum, const std::string &socVersion)
{
    // A2/A3 校验
    const std::string ascend950 = "Ascend950";
    if (socVersion.find(ascend950) == std::string::npos) {
        CHECK_RET(metadata != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        return ACLNN_SUCCESS;
    }
    // A5 校验
    return ParamsCheckSliA5(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
        cmpResidualKOptional, batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, topk,
        layoutQOptional, layoutKOptional, maskMode, cmpRatio, metadata, aicCoreNum, aivCoreNum, socVersion);
}

}

#ifdef __cplusplus
}
#endif
