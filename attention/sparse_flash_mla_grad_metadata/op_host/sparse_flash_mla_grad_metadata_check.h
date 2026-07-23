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
 * \file sparse_flash_mla_grad_metadata_check.h
 * \brief
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../sparse_flash_mla_grad/op_kernel/sparse_flash_mla_grad_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char* SMLAG_ACLNN_OP_NAME = "aclnnSparseFlashMlaGradMetadata";

enum class SparseModeSmlag : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

inline constexpr int64_t SMLAG_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t SMLAG_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t SMLAG_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t SMLAG_NUM_HEADS_Q_UPPER_BOUND = 128;

inline bool IsTensorExistSmlag(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumSmlag(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeSmlag(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

aclnnStatus CheckSingleParamSmlag(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
                                  int64_t maxSeqlenCmpKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
                                  int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
                                  int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
                                  const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv,
                                  bool hasCmpKv, uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion)
{
    // batch_size >= 0
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "batch_size",
            std::to_string(batchSize).c_str(),
            "The value of batch_size must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q >= 0
    if (maxSeqlenQ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "max_seqlen_q",
            std::to_string(maxSeqlenQ).c_str(),
            "The value of max_seqlen_q must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_ori_kv >= 0
    if (maxSeqlenOriKv < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "max_seqlen_ori_kv",
            std::to_string(maxSeqlenOriKv).c_str(),
            "The value of max_seqlen_ori_kv must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_cmp_kv >= 0
    if (maxSeqlenCmpKv < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "max_seqlen_cmp_kv",
            std::to_string(maxSeqlenCmpKv).c_str(),
            "The value of max_seqlen_cmp_kv must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q [1, 128]
    if (numHeadsQ < SMLAG_NUM_HEADS_Q_LOWER_BOUND ||
        numHeadsQ > SMLAG_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "num_heads_q",
            std::to_string(numHeadsQ).c_str(),
            "The current value is not within the valid range. "
            "The valid range is [" +
            std::to_string(SMLAG_NUM_HEADS_Q_LOWER_BOUND) + ", " +
            std::to_string(SMLAG_NUM_HEADS_Q_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv: 1
    if (numHeadsKv != 1) {
        OP_LOGE_FOR_INVALID_VALUE(SMLAG_ACLNN_OP_NAME, "num_heads_kv",
            std::to_string(numHeadsKv).c_str(), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim: 512
    if (headDim != 512) {
        OP_LOGE_FOR_INVALID_VALUE(SMLAG_ACLNN_OP_NAME, "head_dim",
            std::to_string(headDim).c_str(), "512");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasOriKv) {
        // ori_topk >= 0
        if (oriTopk < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "ori_topk",
                std::to_string(oriTopk).c_str(),
                "When has_ori_kv is true, the value of ori_topk must be "
                "greater than or equal to 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_mask_mode: 0, 3, or 4
        if (oriMaskMode != static_cast<int64_t>(SparseModeSmlag::DEFAULT_MASK) &&
            oriMaskMode != static_cast<int64_t>(SparseModeSmlag::RIGHT_DOWN_CAUSAL) &&
            oriMaskMode != static_cast<int64_t>(SparseModeSmlag::BAND)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "ori_mask_mode",
                std::to_string(oriMaskMode).c_str(),
                "When has_ori_kv is true, the value of ori_mask_mode "
                "must be in [0, 3, 4]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_left >= -1
        if (oriWinLeft < -1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "ori_win_left",
                std::to_string(oriWinLeft).c_str(),
                "When has_ori_kv is true, the value of ori_win_left "
                "must be greater than or equal to -1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_right >= -1
        if (oriWinRight < -1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "ori_win_right",
                std::to_string(oriWinRight).c_str(),
                "When has_ori_kv is true, the value of ori_win_right "
                "must be greater than or equal to -1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasCmpKv) {
        // cmp_topk >= 0
        if (cmpTopk < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "cmp_topk",
                std::to_string(cmpTopk).c_str(),
                "When has_cmp_kv is true, the value of cmp_topk must be "
                "greater than or equal to 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_mask_mode: 0 or 3
        if (cmpMaskMode != static_cast<int64_t>(SparseModeSmlag::DEFAULT_MASK) &&
            cmpMaskMode != static_cast<int64_t>(SparseModeSmlag::RIGHT_DOWN_CAUSAL)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "cmp_mask_mode",
                std::to_string(cmpMaskMode).c_str(),
                "When has_cmp_kv is true, the value of cmp_mask_mode "
                "must be in [0, 3]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_ratio: 1~128
        if (cmpRatio < SMLAG_CMP_RATIO_LOWER_BOUND ||
            cmpRatio > SMLAG_CMP_RATIO_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "cmp_ratio",
                std::to_string(cmpRatio).c_str(),
                "When has_cmp_kv is true, the current value is not within "
                "the valid range. The valid range is [" +
                std::to_string(SMLAG_CMP_RATIO_LOWER_BOUND) + ", " +
                std::to_string(SMLAG_CMP_RATIO_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME, "layout_q",
            "layout_q cannot be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutKvOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME, "layout_kv",
            "layout_kv cannot be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q: BSND or TND
    if (strcmp(layoutQOptional, "TND") != 0 &&
        strcmp(layoutQOptional, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "layout_q",
            layoutQOptional,
            "The value of layout_q must be in [TND, BSND]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv: BSND or TND
    if (strcmp(layoutKvOptional, "BSND") != 0 &&
        strcmp(layoutKvOptional, "TND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "layout_kv",
            layoutKvOptional,
            "The value of layout_kv must be in [TND, BSND]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q必须与layout_kv相同
    if (strcmp(layoutQOptional, layoutKvOptional) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(SMLAG_ACLNN_OP_NAME,
            "layout_q, layout_kv",
            std::string(layoutQOptional) + ", " + std::string(layoutKvOptional),
            "The value of layout_q must be equal to that of layout_kv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核数校验
    if (aicCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "aic_core_num",
            std::to_string(aicCoreNum).c_str(),
            "The value of aic_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "aiv_core_num",
            std::to_string(aivCoreNum).c_str(),
            "The value of aiv_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSmlag(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedOriKvOptional,
                                const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
                                const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional,
                                int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
                                int64_t cmpMaskMode, bool hasOriKv, bool hasCmpKv, const char *layoutQOptional,
                                const char *layoutKvOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistSmlag(cuSeqlensQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "cu_seqlens_q",
                "When layout_q is TND, cu_seqlens_q cannot be empty.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasOriKv) {
        // cu_seqlens_ori_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistSmlag(cuSeqlensOriKvOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_ori_kv",
                    "When has_ori_kv is true and layout_kv is TND, "
                    "cu_seqlens_ori_kv cannot be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // ori_topk_length 存在性校验
        if (oriTopk != 0 && oriMaskMode == static_cast<int64_t>(SparseModeSmlag::DEFAULT_MASK)) {
            if (!IsTensorExistSmlag(oriTopkLengthOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "ori_topk_length",
                    "When has_ori_kv is true, ori_topk is not 0 and "
                    "ori_mask_mode is 0, ori_topk_length cannot be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (hasCmpKv) {
        // cu_seqlens_cmp_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistSmlag(cuSeqlensCmpKvOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_cmp_kv",
                    "When has_cmp_kv is true and layout_kv is TND, "
                    "cu_seqlens_cmp_kv cannot be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_residual_kv 存在性校验
        if (cmpRatio != 1 && cmpMaskMode == static_cast<int64_t>(SparseModeSmlag::RIGHT_DOWN_CAUSAL)) {
            if (!IsTensorExistSmlag(cmpResidualKvOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cmp_residual_kv",
                    "When has_cmp_kv is true, cmp_ratio is not 1 and "
                    "cmp_mask_mode is 3, cmp_residual_kv cannot be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_topk_length 存在性校验
        if (cmpTopk != 0 && cmpMaskMode == static_cast<int64_t>(SparseModeSmlag::DEFAULT_MASK)) {
            if (!IsTensorExistSmlag(cmpTopkLengthOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cmp_topk_length",
                    "When has_cmp_kv is true, cmp_topk is not 0 and "
                    "cmp_mask_mode is 0, cmp_topk_length cannot be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // metadata 存在性校验
    if (!IsTensorExistSmlag(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLAG_ACLNN_OP_NAME, "metadata",
            "metadata cannot be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSmlag(const aclTensor *sequsedQOptional, const aclTensor *cuSeqlensQOptional,
                               const char *layoutQOptional, int64_t batchSize)
{
    // 1. 如果 sequsedQOptional 传了，使用 sequsedQOptional 获取 BatchSize
    if (IsTensorExistSmlag(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedQOptional 没传，判断 Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistSmlag(cuSeqlensQOptional)) {
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用 batchSize
    return batchSize;
}

int64_t GetOriKvBatchSizeSmlag(const aclTensor *sequsedOriKvOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const char *layoutKvOptional, int64_t batchSize)
{
    // 1. 如果 sequsedOriKvOptional 传了，使用 sequsedOriKvOptional 获取 BatchSize
    if (IsTensorExistSmlag(sequsedOriKvOptional)) {
        return sequsedOriKvOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedOriKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        if (IsTensorExistSmlag(cuSeqlensOriKvOptional)) {
            return cuSeqlensOriKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用 batchSize
    return batchSize;
}

int64_t GetCmpKvBatchSizeSmlag(const aclTensor *sequsedCmpKvOptional, const aclTensor *cuSeqlensCmpKvOptional,
                               const char *layoutKvOptional, int64_t batchSize)
{
    // 1. 如果 sequsedCmpKvOptional 传了，使用 sequsedCmpKvOptional 获取 BatchSize
    if (IsTensorExistSmlag(sequsedCmpKvOptional)) {
        return sequsedCmpKvOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedCmpKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        if (IsTensorExistSmlag(cuSeqlensCmpKvOptional)) {
            return cuSeqlensCmpKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用 batchSize
    return batchSize;
}

aclnnStatus CheckConsistencySmlag(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                  const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                                  const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                                  const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                                  const aclTensor *cmpTopkLengthOptional, int64_t batchSize,
                                  const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv,
                                  bool hasCmpKv, const aclTensor *metadata)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    int64_t dimNum = -1;
    // 校验 cu_seqlens_q
    if (IsTensorExistSmlag(cuSeqlensQOptional)) {
        // 校验 cu_seqlens_q 维度
        dimNum = GetDimNumSmlag(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME, "cu_seqlens_q",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        dataType = GetDataTypeSmlag(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "cu_seqlens_q",
                std::to_string(static_cast<int32_t>(dataType)).c_str(),
                "The dtype of cu_seqlens_q must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistSmlag(sequsedQOptional)) {
        // 校验 seqused_q 维度
        dimNum = GetDimNumSmlag(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME, "seqused_q",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_q 数据类型
        dataType = GetDataTypeSmlag(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "seqused_q",
                std::to_string(static_cast<int32_t>(dataType)).c_str(),
                "The dtype of seqused_q must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // ori_kv部分
    if (hasOriKv) {
        // 校验 cu_seqlens_ori_kv
        if (IsTensorExistSmlag(cuSeqlensOriKvOptional)) {
            // 校验 cu_seqlens_ori_kv 维度
            dimNum = GetDimNumSmlag(cuSeqlensOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_ori_kv",
                    std::to_string(dimNum).c_str(), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_ori_kv 数据类型
            dataType = GetDataTypeSmlag(cuSeqlensOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_ori_kv",
                    std::to_string(static_cast<int32_t>(dataType)).c_str(),
                    "The dtype of cu_seqlens_ori_kv must be int32.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_ori_kv
        if (IsTensorExistSmlag(sequsedOriKvOptional)) {
            // 校验 seqused_ori_kv 维度
            dimNum = GetDimNumSmlag(sequsedOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME,
                    "seqused_ori_kv",
                    std::to_string(dimNum).c_str(), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_ori_kv 数据类型
            dataType = GetDataTypeSmlag(sequsedOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "seqused_ori_kv",
                    std::to_string(static_cast<int32_t>(dataType)).c_str(),
                    "The dtype of seqused_ori_kv must be int32.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 ori_topk_length
        if (IsTensorExistSmlag(oriTopkLengthOptional)) {
            // 校验 ori_topk_length 维度
            dimNum = GetDimNumSmlag(oriTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                        "ori_topk_length",
                        std::to_string(dimNum).c_str(),
                        "The shape dim of ori_topk_length must be 2 "
                        "when layout_q is TND.");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                        "ori_topk_length",
                        std::to_string(dimNum).c_str(),
                        "The shape dim of ori_topk_length must be 3 "
                        "when layout_q is BSND.");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 ori_topk_length 数据类型
            dataType = GetDataTypeSmlag(oriTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "ori_topk_length",
                    std::to_string(static_cast<int32_t>(dataType)).c_str(),
                    "The dtype of ori_topk_length must be int32.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // cmp_kv部分
    if (hasCmpKv) {
        // 校验 cu_seqlens_cmp_kv
        if (IsTensorExistSmlag(cuSeqlensCmpKvOptional)) {
            // 校验 cu_seqlens_cmp_kv 维度
            dimNum = GetDimNumSmlag(cuSeqlensCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_cmp_kv",
                    std::to_string(dimNum).c_str(), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_cmp_kv 数据类型
            dataType = GetDataTypeSmlag(cuSeqlensCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_cmp_kv",
                    std::to_string(static_cast<int32_t>(dataType)).c_str(),
                    "The dtype of cu_seqlens_cmp_kv must be int32.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_cmp_kv
        if (IsTensorExistSmlag(sequsedCmpKvOptional)) {
            // 校验 seqused_cmp_kv 维度
            dimNum = GetDimNumSmlag(sequsedCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME,
                    "seqused_cmp_kv",
                    std::to_string(dimNum).c_str(), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_cmp_kv 数据类型
            dataType = GetDataTypeSmlag(sequsedCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "seqused_cmp_kv",
                    std::to_string(static_cast<int32_t>(dataType)).c_str(),
                    "The dtype of seqused_cmp_kv must be int32.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_residual_kv
        if (IsTensorExistSmlag(cmpResidualKvOptional)) {
            // 校验 cmp_residual_kv 维度
            dimNum = GetDimNumSmlag(cmpResidualKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME,
                    "cmp_residual_kv",
                    std::to_string(dimNum).c_str(), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cmp_residual_kv 数据类型
            dataType = GetDataTypeSmlag(cmpResidualKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cmp_residual_kv",
                    std::to_string(static_cast<int32_t>(dataType)).c_str(),
                    "The dtype of cmp_residual_kv must be int32.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_topk_length
        if (IsTensorExistSmlag(cmpTopkLengthOptional)) {
            // 校验 cmp_topk_length 维度
            dimNum = GetDimNumSmlag(cmpTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                        "cmp_topk_length",
                        std::to_string(dimNum).c_str(),
                        "The shape dim of cmp_topk_length must be 2 "
                        "when layout_q is TND.");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                        "cmp_topk_length",
                        std::to_string(dimNum).c_str(),
                        "The shape dim of cmp_topk_length must be 3 "
                        "when layout_q is BSND.");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 cmp_topk_length 数据类型
            dataType = GetDataTypeSmlag(cmpTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cmp_topk_length",
                    std::to_string(static_cast<int32_t>(dataType)).c_str(),
                    "The dtype of cmp_topk_length must be int32.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // 校验 metadata
    if (IsTensorExistSmlag(metadata)) {
        // 校验 metadata 维度
        dimNum = GetDimNumSmlag(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLAG_ACLNN_OP_NAME, "metadata",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::SMLAG_METADATA_TOTAL_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLAG_ACLNN_OP_NAME, "metadata",
                std::to_string(metadata->GetViewShape().GetDim(0)).c_str(),
                std::to_string(optiling::SMLAG_METADATA_TOTAL_SIZE).c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 数据类型
        dataType = GetDataTypeSmlag(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLAG_ACLNN_OP_NAME, "metadata",
                std::to_string(static_cast<int32_t>(dataType)).c_str(),
                "The dtype of metadata must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 q/kv 维度一致性
    int64_t queryBatchSize = GetQueryBatchSizeSmlag(sequsedQOptional, cuSeqlensQOptional,
                                                    layoutQOptional, batchSize);
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistSmlag(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "cu_seqlens_q, seqused_q",
                std::to_string(cuSeqlensQBatchSize) + ", " +
                std::to_string(queryBatchSize),
                "When layout_q is TND and seqused_q is passed, "
                "the batch sizes of cu_seqlens_q, "
                "seqused_q must be the same.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasOriKv) {
        int64_t oriKvBatchSize = GetOriKvBatchSizeSmlag(sequsedOriKvOptional,
            cuSeqlensOriKvOptional, layoutKvOptional, batchSize);
        // 校验q与ori_kv维度一致性
        if (queryBatchSize != oriKvBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "query, ori_kv",
                std::to_string(queryBatchSize) + ", " +
                std::to_string(oriKvBatchSize),
                "When has_ori_kv is true, the batch sizes of "
                "query, ori_kv must be the same.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验TND场景ori_kv维度一致性
        if (strcmp(layoutKvOptional, "TND") == 0 && IsTensorExistSmlag(sequsedOriKvOptional)) {
            int64_t cuSeqlensOriKvBatchSize = cuSeqlensOriKvOptional->GetViewShape().GetDim(0) - 1;
            if (cuSeqlensOriKvBatchSize != oriKvBatchSize) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_ori_kv, seqused_ori_kv",
                    std::to_string(cuSeqlensOriKvBatchSize) + ", " +
                    std::to_string(oriKvBatchSize),
                    "When layout_kv is TND and seqused_ori_kv is passed, "
                    "the batch sizes of cu_seqlens_ori_kv, "
                    "seqused_ori_kv must be the same.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (hasCmpKv) {
        int64_t cmpKvBatchSize = GetCmpKvBatchSizeSmlag(sequsedCmpKvOptional,
            cuSeqlensCmpKvOptional, layoutKvOptional, batchSize);
        // 校验q与cmp_kv维度一致性
        if (queryBatchSize != cmpKvBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                "query, cmp_kv",
                std::to_string(queryBatchSize) + ", " +
                std::to_string(cmpKvBatchSize),
                "When has_cmp_kv is true, the batch sizes of "
                "query, cmp_kv must be the same.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验TND场景cmp_kv维度一致性
        if (strcmp(layoutKvOptional, "TND") == 0 && IsTensorExistSmlag(sequsedCmpKvOptional)) {
            int64_t cuSeqlensCmpKvBatchSize = cuSeqlensCmpKvOptional->GetViewShape().GetDim(0) - 1;
            if (cuSeqlensCmpKvBatchSize != cmpKvBatchSize) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cu_seqlens_cmp_kv, seqused_cmp_kv",
                    std::to_string(cuSeqlensCmpKvBatchSize) + ", " +
                    std::to_string(cmpKvBatchSize),
                    "When layout_kv is TND and seqused_cmp_kv is passed, "
                    "the batch sizes of cu_seqlens_cmp_kv, "
                    "seqused_cmp_kv must be the same.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_residual_kv 元素数
        if (IsTensorExistSmlag(cmpResidualKvOptional)) {
            if (cmpResidualKvOptional->GetViewShape().GetDim(0) != queryBatchSize) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(SMLAG_ACLNN_OP_NAME,
                    "cmp_residual_kv",
                    std::to_string(cmpResidualKvOptional->GetViewShape().GetDim(0)).c_str(),
                    "The shape size of cmp_residual_kv should be equal to "
                    "the batch size of query.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                               const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                               const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                               const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ, int64_t numHeadsKv,
                               int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
                               int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio,
                               int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
                               const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
                               uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion,
                               const aclTensor *metaData)
{
    if (CheckSingleParamSmlag(batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, numHeadsQ, numHeadsKv, headDim,
                              oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                              layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, aicCoreNum, aivCoreNum,
                              socVersion) == ACLNN_SUCCESS &&
        CheckExistenceSmlag(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedOriKvOptional,
                            sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
                            oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, hasOriKv, hasCmpKv, layoutQOptional,
                            layoutKvOptional, metaData) == ACLNN_SUCCESS &&
        CheckConsistencySmlag(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                              sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                              cmpTopkLengthOptional, batchSize, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv,
                              metaData) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}
}

#ifdef __cplusplus
}
#endif
