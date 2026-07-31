/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_v2_metadata_check.h
 * \brief Routine parameter checks (dtype/shape/attr/existence/consistency) for aclnn layer.
 *        Tensor in-memory value checks remain in aicpu.
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *DLI_ACLNN_OP_NAME = "aclnnDenseLightningIndexerSoftmaxLseV2Metadata";

inline constexpr int64_t DLI_NO_MASK_MODE = 0;
inline constexpr int64_t DLI_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t DLI_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t DLI_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t DLI_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t DLI_NUM_HEADS_Q_UPPER_BOUND = 128;
inline constexpr int64_t DLI_HEAD_DIM_LIMIT = 128;
inline constexpr int64_t DLI_METADATA_SIZE = 64;

inline bool IsTensorExistDli(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumDli(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeDli(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

inline bool IsTensorSourceDli(const std::string &source) { return source != "batch_size"; }

inline int64_t GetRawShapeSizeDli(const std::string &source, int64_t batchValue)
{
    if (source.find("cu_seqlens") != std::string::npos) {
        return batchValue + 1;
    }
    return batchValue;
}

inline std::string GetSourceDescDli(const std::string &source)
{
    if (source == "batch_size") {
        return "batch_size";
    }
    if (source.find("cu_seqlens") != std::string::npos) {
        return "the shape size of " + source + " minus 1";
    }
    return "the shape size of " + source;
}

int64_t GetQueryBatchSizeDli(int64_t batchSize, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
                             const char *layoutQOptional, std::string *source)
{
    if (IsTensorExistDli(sequsedQOptional)) {
        *source = "seqused_q";
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistDli(cuSeqlensQOptional)) {
            *source = "cu_seqlens_q";
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

int64_t GetKeyBatchSizeDli(int64_t batchSize, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedKOptional,
                           const char *layoutKOptional, std::string *source)
{
    if (IsTensorExistDli(sequsedKOptional)) {
        *source = "seqused_k";
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistDli(cuSeqlensKOptional)) {
            *source = "cu_seqlens_k";
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

aclnnStatus CheckSingleParamDli(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, int64_t numHeadsQ,
                                int64_t numHeadsK, int64_t headDim, const char *layoutQOptional,
                                const char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, uint32_t aicCoreNum)
{
    if (aicCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "aic_core_num", std::to_string(aicCoreNum),
                                              "The value of aic_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsQ < DLI_NUM_HEADS_Q_LOWER_BOUND || numHeadsQ > DLI_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "num_heads_q", std::to_string(numHeadsQ),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(DLI_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                                                  std::to_string(DLI_NUM_HEADS_Q_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsK != 1) {
        OP_LOGE_FOR_INVALID_VALUE(DLI_ACLNN_OP_NAME, "num_heads_k", std::to_string(numHeadsK), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (headDim != DLI_HEAD_DIM_LIMIT) {
        OP_LOGE_FOR_INVALID_VALUE(DLI_ACLNN_OP_NAME, "head_dim", std::to_string(headDim), "128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsQ % numHeadsK != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(DLI_ACLNN_OP_NAME, "num_heads_q, num_heads_k",
                                               std::to_string(numHeadsQ) + ", " + std::to_string(numHeadsK),
                                               "num_heads_q must be divisible by num_heads_k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
                                              "The value of batch_size must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maxSeqlenQ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "The value of max_seqlen_q must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maxSeqlenK < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
                                              "The value of max_seqlen_k must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQOptional, "BSND") == 0 && maxSeqlenQ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "The value of max_seqlen_q must be greater than 0 for BSND layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maskMode != DLI_NO_MASK_MODE && maskMode != DLI_CAUSAL_MASK_MODE) {
        OP_LOGE_FOR_INVALID_VALUE(DLI_ACLNN_OP_NAME, "mask_mode", std::to_string(maskMode), "0 or 3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cmpRatio < DLI_CMP_RATIO_LOWER_BOUND || cmpRatio > DLI_CMP_RATIO_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(DLI_CMP_RATIO_LOWER_BOUND) + ", " +
                                                  std::to_string(DLI_CMP_RATIO_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(DLI_ACLNN_OP_NAME, "layout_q", "layout_q cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQOptional, "TND") != 0 && strcmp(layoutQOptional, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(DLI_ACLNN_OP_NAME, "layout_q", layoutQOptional, "TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutKOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(DLI_ACLNN_OP_NAME, "layout_k", "layout_k cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutKOptional, "TND") != 0 && strcmp(layoutKOptional, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(DLI_ACLNN_OP_NAME, "layout_k", layoutKOptional, "TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQOptional, layoutKOptional) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(DLI_ACLNN_OP_NAME, "layout_q, layout_k",
                                               std::string(layoutQOptional) + ", " + std::string(layoutKOptional),
                                               "The value of layout_q must be equal to that of layout_k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceDli(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
                              const aclTensor *cuSeqlensKOptional, const aclTensor *cmpResidualKOptional,
                              const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistDli(cuSeqlensQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(DLI_ACLNN_OP_NAME, "cu_seqlens_q",
                                                     "When layout_q is TND, cu_seqlens_q cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (!IsTensorExistDli(cuSeqlensKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(DLI_ACLNN_OP_NAME, "cu_seqlens_k",
                                                     "When layout_k is TND, cu_seqlens_k cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (cmpRatio != DLI_CMP_RATIO_LOWER_BOUND && maskMode == DLI_CAUSAL_MASK_MODE) {
        if (!IsTensorExistDli(cmpResidualKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(DLI_ACLNN_OP_NAME, "cmp_residual_k",
                                                     "When cmp_ratio is not 1 and mask_mode is CAUSAL, "
                                                     "cmp_residual_k cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!IsTensorExistDli(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(DLI_ACLNN_OP_NAME, "metadata", "metadata cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyDli(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
                                const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
                                const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
                                const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    if (IsTensorExistDli(cuSeqlensQOptional)) {
        dimNum = GetDimNumDli(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(DLI_ACLNN_OP_NAME, "cu_seqlens_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeDli(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(DLI_ACLNN_OP_NAME, "cu_seqlens_q", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (IsTensorExistDli(cuSeqlensKOptional)) {
        dimNum = GetDimNumDli(cuSeqlensKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(DLI_ACLNN_OP_NAME, "cu_seqlens_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeDli(cuSeqlensKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(DLI_ACLNN_OP_NAME, "cu_seqlens_k", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (IsTensorExistDli(sequsedQOptional)) {
        dimNum = GetDimNumDli(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(DLI_ACLNN_OP_NAME, "seqused_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeDli(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(DLI_ACLNN_OP_NAME, "seqused_q", ToString(dataType).GetString(),
                                                  "The dtype of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (IsTensorExistDli(sequsedKOptional)) {
        dimNum = GetDimNumDli(sequsedKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(DLI_ACLNN_OP_NAME, "seqused_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeDli(sequsedKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(DLI_ACLNN_OP_NAME, "seqused_k", ToString(dataType).GetString(),
                                                  "The dtype of seqused_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (IsTensorExistDli(cmpResidualKOptional)) {
        dimNum = GetDimNumDli(cmpResidualKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(DLI_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeDli(cmpResidualKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(DLI_ACLNN_OP_NAME, "cmp_residual_k", ToString(dataType).GetString(),
                                                  "The dtype of cmp_residual_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (IsTensorExistDli(metadata)) {
        dimNum = GetDimNumDli(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(DLI_ACLNN_OP_NAME, "metadata", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeDli(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(DLI_ACLNN_OP_NAME, "metadata", ToString(dataType).GetString(),
                                                  "The dtype of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (metadata->GetViewShape().GetDim(0) != DLI_METADATA_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(DLI_ACLNN_OP_NAME, "metadata",
                                          std::to_string(metadata->GetViewShape().GetDim(0)),
                                          std::to_string(DLI_METADATA_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    std::string querySource;
    int64_t queryBatchSize =
        GetQueryBatchSizeDli(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional, &querySource);
    std::string keySource;
    int64_t keyBatchSize =
        GetKeyBatchSizeDli(batchSize, cuSeqlensKOptional, sequsedKOptional, layoutKOptional, &keySource);
    if (queryBatchSize != keyBatchSize) {
        if (IsTensorSourceDli(querySource) && IsTensorSourceDli(keySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                DLI_ACLNN_OP_NAME, querySource + " and " + keySource,
                std::to_string(GetRawShapeSizeDli(querySource, queryBatchSize)) + " and " +
                    std::to_string(GetRawShapeSizeDli(keySource, keyBatchSize)),
                GetSourceDescDli(querySource) + " must be equal to " + GetSourceDescDli(keySource));
        } else if (IsTensorSourceDli(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                DLI_ACLNN_OP_NAME, querySource, std::to_string(GetRawShapeSizeDli(querySource, queryBatchSize)),
                GetSourceDescDli(querySource) + " must be equal to batch_size");
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                DLI_ACLNN_OP_NAME, keySource, std::to_string(GetRawShapeSizeDli(keySource, keyBatchSize)),
                GetSourceDescDli(keySource) + " must be equal to batch_size");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (queryBatchSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DLI_ACLNN_OP_NAME, "batch_size", std::to_string(queryBatchSize),
                                              "The value of batch_size must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (IsTensorExistDli(cuSeqlensQOptional) &&
        cuSeqlensQOptional->GetViewShape().GetDim(0) != queryBatchSize + 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(DLI_ACLNN_OP_NAME, "cu_seqlens_q",
                                                  std::to_string(cuSeqlensQOptional->GetViewShape().GetDim(0)),
                                                  "The shape size of cu_seqlens_q must be batch_size + 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsTensorExistDli(cuSeqlensKOptional) &&
        cuSeqlensKOptional->GetViewShape().GetDim(0) != keyBatchSize + 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(DLI_ACLNN_OP_NAME, "cu_seqlens_k",
                                                  std::to_string(cuSeqlensKOptional->GetViewShape().GetDim(0)),
                                                  "The shape size of cu_seqlens_k must be batch_size + 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsTensorExistDli(sequsedQOptional) && sequsedQOptional->GetViewShape().GetDim(0) != queryBatchSize) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(DLI_ACLNN_OP_NAME, "seqused_q",
                                                  std::to_string(sequsedQOptional->GetViewShape().GetDim(0)),
                                                  "The shape size of seqused_q must be batch_size");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsTensorExistDli(sequsedKOptional) && sequsedKOptional->GetViewShape().GetDim(0) != keyBatchSize) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(DLI_ACLNN_OP_NAME, "seqused_k",
                                                  std::to_string(sequsedKOptional->GetViewShape().GetDim(0)),
                                                  "The shape size of seqused_k must be batch_size");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsTensorExistDli(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        if (cmpResidualKBatch != queryBatchSize) {
            if (IsTensorSourceDli(querySource)) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    DLI_ACLNN_OP_NAME, "cmp_residual_k and " + querySource,
                    std::to_string(cmpResidualKBatch) + " and " +
                        std::to_string(GetRawShapeSizeDli(querySource, queryBatchSize)),
                    "The shape size of cmp_residual_k must be equal to " + GetSourceDescDli(querySource));
            } else {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(DLI_ACLNN_OP_NAME, "cmp_residual_k",
                                                          std::to_string(cmpResidualKBatch),
                                                          "The shape size of cmp_residual_k must be equal "
                                                          "to batch_size");
            }
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckDliA5(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                             const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                             const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
                             int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim,
                             char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
                             const aclTensor *metadata, uint32_t aicCoreNum)
{
    auto ret = CheckSingleParamDli(batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, layoutQOptional,
                                   layoutKOptional, maskMode, cmpRatio, aicCoreNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceDli(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, cmpResidualKOptional,
                            layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyDli(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                              cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckDli(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                           const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                           const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
                           int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim,
                           char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
                           const aclTensor *metadata, uint32_t aicCoreNum, const std::string &socVersion)
{
    const std::string ascend950 = "Ascend950";
    if (socVersion.find(ascend950) == std::string::npos) {
        CHECK_RET(metadata != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        return ACLNN_SUCCESS;
    }
    return ParamsCheckDliA5(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                            cmpResidualKOptional, batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim,
                            layoutQOptional, layoutKOptional, maskMode, cmpRatio, metadata, aicCoreNum);
}

} // namespace

#ifdef __cplusplus
}
#endif
