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
 * \file lightning_indexer_v2_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../lightning_indexer_v2/op_kernel/lightning_indexer_v2_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *LI_V2_ACLNN_OP_NAME = "LightningIndexerV2Metadata";

inline constexpr int64_t LI_V2_NO_MASK_MODE = 0;
inline constexpr int64_t LI_V2_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t LI_V2_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t LI_V2_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t LI_V2_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t LI_V2_NUM_HEADS_Q_UPPER_BOUND = 64;
inline constexpr int64_t LI_V2_TOPK_LOWER_BOUND = 1;
inline constexpr int64_t LI_V2_TOPK_UPPER_BOUND = 8192;

inline bool IsTensorExistLiV2(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumLiV2(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeLiV2(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

int64_t GetQueryBatchSizeLiV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
    const char *layoutQOptional)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    if (IsTensorExistLiV2(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedQOptional 没传，使用cuSeqlensQOptional获取BatchSize
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistLiV2(cuSeqlensQOptional)) { // 前序校验已保证layout_q = TND时，cu_seqlens_q必须传入，此通路必达
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用batchSize
    return batchSize;
}

int64_t GetKeyBatchSizeLiV2(int64_t batchSize, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedKOptional,
    const char *layoutKOptional)
{
    // 1. 如果sequsedKOptional 传了，使用sequsedKOptional获取BatchSize
    if (IsTensorExistLiV2(sequsedKOptional)) {
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    // 如果是 TND，必须使用 cuSeqlensKOptional获取BatchSize
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistLiV2(cuSeqlensKOptional)) { // 前序校验已保证layout_k = TND时，cu_seqlens_k必须传入，此通路必达
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用batchSize
    return batchSize;
}

aclnnStatus CheckSingleParamLiV2(int64_t numHeadsQ, int64_t numHeadsK, int64_t topk, int64_t batchSize,
    int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional, const char *layoutKOptional, int64_t maskMode,
    int64_t cmpRatio, uint32_t aicCoreNum, uint32_t aivCoreNum, const std::string &socVersion)
{
    // num_heads_q 校验
    if (numHeadsQ >= LI_V2_NUM_HEADS_Q_LOWER_BOUND && numHeadsQ <= LI_V2_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "num_heads_q", std::to_string(numHeadsQ),
            "The value of num_heads_q should be in range [" + std::to_string(LI_V2_NUM_HEADS_Q_LOWER_BOUND) + ", " +
            std::to_string(LI_V2_NUM_HEADS_Q_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_k 校验
    if (numHeadsK == 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "num_heads_kv", std::to_string(numHeadsK),
            "The value of num_heads_kv should be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // topk 校验
    if (topk >= LI_V2_TOPK_LOWER_BOUND && topk <= LI_V2_TOPK_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "topk", std::to_string(topk),
            "The value of topk should be in range [" + std::to_string(LI_V2_TOPK_LOWER_BOUND) + ", " +
            std::to_string(LI_V2_TOPK_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // batch_size 非负校验
    if (batchSize >= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "batch_size", std::to_string(batchSize),
            "The value of batch_size should not be negative");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 校验
    if (maxSeqlenQ >= -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "max_seqlen_q", std::to_string(maxSeqlenQ),
            "The value of max_seqlen_q should be >= -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_k 校验
    if (maxSeqlenK >= -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "max_seqlen_k", std::to_string(maxSeqlenK),
            "The value of max_seqlen_k should be >= -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode 校验
    if ((maskMode == LI_V2_NO_MASK_MODE) || (maskMode == LI_V2_CAUSAL_MASK_MODE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "mask_mode", std::to_string(maskMode),
            "The value of mask_mode should be " + std::to_string(LI_V2_NO_MASK_MODE) + " or " +
            std::to_string(LI_V2_CAUSAL_MASK_MODE));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_ratio 校验
    if ((cmpRatio >= LI_V2_CMP_RATIO_LOWER_BOUND) && (cmpRatio <= LI_V2_CMP_RATIO_UPPER_BOUND)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmp_ratio", std::to_string(cmpRatio),
            "The value of cmp_ratio should be in range [" + std::to_string(LI_V2_CMP_RATIO_LOWER_BOUND) + ", " +
            std::to_string(LI_V2_CMP_RATIO_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    CHECK_COND((layoutQOptional != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_q is null!");
    if ((strcmp(layoutQOptional, "TND") == 0) || (strcmp(layoutQOptional, "BSND") == 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", layoutQOptional,
            "The value of layout_q must be TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k 校验
    CHECK_COND((layoutKOptional != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_k is null!");
    if ((strcmp(layoutKOptional, "TND") == 0) || (strcmp(layoutKOptional, "BSND") == 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_k", layoutKOptional,
            "The value of layout_k must be in [TND, BSND, PA_BBND]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核心数校验
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aicCoreNum <= optiling::AIC_CORE_MAX_NUM, ACLNN_ERR_PARAM_INVALID,
        "The maximum supported AIC num is %u, but got %u", optiling::AIC_CORE_MAX_NUM, aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
    CHECK_COND(aivCoreNum <= optiling::AIV_CORE_MAX_NUM, ACLNN_ERR_PARAM_INVALID,
        "The maximum supported AIV num is %u, but got %u", optiling::AIV_CORE_MAX_NUM, aivCoreNum);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceLiV2(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
    const char *layoutKOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if ((strcmp(layoutQOptional, "TND") == 0) && IsTensorExistLiV2(cuSeqlensQOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_q",
            "When layout_q is TND, the value of cu_seqlens_q must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q BSND, seqused_q 不存在时，max_seqlen_q 不能为-1
    if (strcmp(layoutQOptional, "BSND") == 0 && !IsTensorExistLiV2(sequsedQOptional) && (maxSeqlenQ > -1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "max_seqlen_q", std::to_string(maxSeqlenQ),
            "When layout_q is BSND and seqused_q is not provided, max_seqlen_q can not be -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cu_seqlens_k 存在性校验
    if ((strcmp(layoutKOptional, "TND") == 0) && IsTensorExistLiV2(cuSeqlensKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_k",
            "When layout_k is TND, the value of cu_seqlens_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // seqused_k 存在性校验
    if ((strcmp(layoutKOptional, "PA_BBND") == 0) && IsTensorExistLiV2(sequsedKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "seqused_k",
            "When layout_k is PA_BBND, the value of seqused_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k BSND, seqused_k 不存在时，max_seqlen_k 不能为-1
    if (strcmp(layoutKOptional, "BSND") == 0 && !IsTensorExistLiV2(sequsedKOptional) && maxSeqlenK > -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "max_seqlen_k", std::to_string(maxSeqlenK),
            "When layout_k is BSND and seqused_k is not provided, max_seqlen_k can not be -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_residual_k 存在性校验
    if ((cmpRatio != LI_V2_CMP_RATIO_LOWER_BOUND) && (maskMode == LI_V2_CAUSAL_MASK_MODE) &&
        IsTensorExistLiV2(cmpResidualKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_residual_k",
            "When cmp_ratio is not 1 and mask_mode is CAUSAL, cmp_residual_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // metadata 存在性校验
    CHECK_COND(IsTensorExistLiV2(metadata), ACLNN_ERR_PARAM_INVALID,
        "Output metadata is nullptr!");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyLiV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, const char *layoutQOptional, const char *layoutKOptional,
    const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // 校验 cu_seqlens_q
    if (IsTensorExistLiV2(cuSeqlensQOptional)) {
        dimNum = GetDimNumLiV2(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(MQSMLA_ACLNN_OP_NAME, "cu_seqlens_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeLiV2(cuSeqlensQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistLiV2(cuSeqlensKOptional)) {
        dimNum = GetDimNumLiV2(cuSeqlensKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeLiV2(cuSeqlensKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_q
    if (IsTensorExistLiV2(sequsedQOptional)) {
        dimNum = GetDimNumLiV2(sequsedQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %lld", dimNum);
        dataType = GetDataTypeLiV2(sequsedQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_k
    if (IsTensorExistLiV2(sequsedKOptional)) {
        dimNum = GetDimNumLiV2(sequsedKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeLiV2(sequsedKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cmp_residual_k
    if (IsTensorExistLiV2(cmpResidualKOptional)) {
        dimNum = GetDimNumLiV2(cmpResidualKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID,
            "The dim num of cmp_residual_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeLiV2(cmpResidualKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cmp_residual_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 metadata
    if (IsTensorExistLiV2(metadata)) {
        dimNum = GetDimNumLiV2(metadata);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lld", dimNum);
        dataType = GetDataTypeLiV2(metadata);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of metadata must be int32, but got %d", static_cast<int32_t>(dataType));
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::LI_V2_METADATA_TOTAL_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %lld",
                optiling::LI_V2_METADATA_TOTAL_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验batch
    int64_t queryBatchSize = GetQueryBatchSizeLiV2(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional);
    int64_t keyBatchSize = GetKeyBatchSizeLiV2(batchSize, cuSeqlensKOptional, sequsedKOptional, layoutKOptional);
    CHECK_COND(queryBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
        "The batch_size obtained from query should be the same as that obtained from key, but got %lld and %lld",
        queryBatchSize, keyBatchSize);
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistLiV2(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(cuSeqlensQBatchSize == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_q is TND and seqused_q is passed, The batch_size obtained from cu_seqlens_q should be the "
            "same as that obtained from seqused_q, but got %lld and %lld", cuSeqlensQBatchSize, queryBatchSize);
    }
    // 校验TND场景k维度一致性
    if (strcmp(layoutKOptional, "TND") == 0 && IsTensorExistLiV2(sequsedKOptional)) {
        int64_t cuSeqlensKBatchSize = cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(cuSeqlensKBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_k is TND and seqused_k is passed, The batch_size obtained from cu_seqlens_k should be the "
            "same as that obtained from seqused_k, but got %lld and %lld", cuSeqlensKBatchSize, keyBatchSize);
    }
    // 校验 cmp_residual_k 元素数
    if (IsTensorExistLiV2(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        CHECK_COND(cmpResidualKBatch == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
            "The batch_size of cmp_residual_k should match the valid batch size, but got %lld and %lld",
            cmpResidualKBatch, queryBatchSize);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckLiV2(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK,
    int64_t headDim, int64_t topk, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, char *layoutQOptional,
    char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, const aclTensor *metadata, uint32_t aicCoreNum,
    uint32_t aivCoreNum, const std::string &socVersion)
{
    auto ret = CheckSingleParamLiV2(numHeadsQ, numHeadsK, topk, batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional,
        layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceLiV2(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
        sequsedKOptional, cmpResidualKOptional, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyLiV2(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
        sequsedKOptional, cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
}

#ifdef __cplusplus
}
#endif
