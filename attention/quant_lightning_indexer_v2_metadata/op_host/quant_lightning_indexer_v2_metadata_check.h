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
 * \file quant_lightning_indexer_v2_metadata_check.h
 * \brief
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char* QLI_V2_ACLNN_OP_NAME = "aclnnQuantLightningIndexerV2Metadata";

inline constexpr int64_t QLI_V2_QUANT_MODE_1 = 1;
inline constexpr int64_t QLI_V2_QUANT_MODE_2 = 2;
inline constexpr int64_t QLI_V2_QUANT_MODE_4 = 4;
inline constexpr int64_t QLI_V2_NO_MASK_MODE = 0;
inline constexpr int64_t QLI_V2_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t QLI_V2_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t QLI_V2_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t QLI_V2_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t QLI_V2_NUM_HEADS_Q_UPPER_BOUND = 64;
inline constexpr int64_t QLI_V2_TOPK_LOWER_BOUND = 1;
inline constexpr int64_t QLI_V2_A5_TOPK_UPPER_BOUND = 8192;
inline constexpr int64_t QLI_V2_A3_TOPK_UPPER_BOUND = 2048;

inline bool IsTensorExistQliV2(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumQliV2(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeQliV2(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

inline bool IsTensorSourceQliV2(const std::string &source)
{
    return source != "batch_size";
}

inline int64_t GetRawShapeSizeQliV2(const std::string &source, int64_t batchValue)
{
    if (source.find("cu_seqlens") != std::string::npos) {
        return batchValue + 1;
    }
    return batchValue;
}

inline std::string GetSourceDescQliV2(const std::string &source)
{
    if (source == "batch_size") {
        return "batch_size";
    }
    if (source.find("cu_seqlens") != std::string::npos) {
        return "the shape size of " + source + " minus 1";
    }
    return "the shape size of " + source;
}

int64_t GetQueryBatchSizeQliV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
    const aclTensor *sequsedQOptional, const char *layoutQOptional,
    std::string &source)
{
    if (IsTensorExistQliV2(sequsedQOptional)) {
        source = "seqused_q";
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistQliV2(cuSeqlensQOptional)) {
            source = "cu_seqlens_q";
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    source = "batch_size";
    return batchSize;
}

int64_t GetKeyBatchSizeQliV2(int64_t batchSize, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedKOptional, const char *layoutKOptional,
    std::string &source)
{
    if (IsTensorExistQliV2(sequsedKOptional)) {
        source = "seqused_k";
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistQliV2(cuSeqlensKOptional)) {
            source = "cu_seqlens_k";
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    source = "batch_size";
    return batchSize;
}

aclnnStatus CheckSingleParamQliV2(int64_t numHeadsQ, int64_t numHeadsK, int64_t topk, int64_t quantMode,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
    const char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, uint32_t aicCoreNum,
    uint32_t aivCoreNum, const std::string &socVersion)
{
    // num_heads_k 校验
    if (numHeadsK != 1) {
        OP_LOGE_FOR_INVALID_VALUE(QLI_V2_ACLNN_OP_NAME, "num_heads_kv",
            std::to_string(numHeadsK).c_str(), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // batch_size 非负校验
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "batch_size",
            std::to_string(batchSize).c_str(),
            "The value of batch_size must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 校验
    if (maxSeqlenQ < -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "max_seqlen_q",
            std::to_string(maxSeqlenQ).c_str(),
            "The value of max_seqlen_q must be greater than or "
            "equal to -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_k 校验
    if (maxSeqlenK < -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "max_seqlen_k",
            std::to_string(maxSeqlenK).c_str(),
            "The value of max_seqlen_k must be greater than or "
            "equal to -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode 校验
    if (maskMode != QLI_V2_NO_MASK_MODE && maskMode != QLI_V2_CAUSAL_MASK_MODE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "mask_mode",
            std::to_string(maskMode).c_str(),
            "The value of mask_mode must be in [0, 3]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "layout_q", "layout_q cannot be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQOptional, "TND") != 0 &&
        strcmp(layoutQOptional, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "layout_q",
            layoutQOptional,
            "The value of layout_q must be in [TND, BSND]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k 校验
    if (layoutKOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "layout_k", "layout_k cannot be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutKOptional, "PA_BBND") != 0 &&
        strcmp(layoutQOptional, layoutKOptional) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "layout_q, layout_k",
            std::string(layoutQOptional) + ", " + std::string(layoutKOptional),
            "The value of layout_q must be equal to that of layout_k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 A2/A3 参数
    if (socVersion.find("Ascend950") == std::string::npos) {
        // num_heads_q 校验
        if (numHeadsQ != QLI_V2_NUM_HEADS_Q_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE(QLI_V2_ACLNN_OP_NAME, "num_heads_q",
                std::to_string(numHeadsQ).c_str(),
                std::to_string(QLI_V2_NUM_HEADS_Q_UPPER_BOUND));
            return ACLNN_ERR_PARAM_INVALID;
        }
        // topk 校验
        if (topk < QLI_V2_TOPK_LOWER_BOUND ||
            topk > QLI_V2_A3_TOPK_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "topk",
                std::to_string(topk).c_str(),
                "The current value is not within the valid range. "
                "The valid range is [" +
                std::to_string(QLI_V2_TOPK_LOWER_BOUND) + ", " +
                std::to_string(QLI_V2_A3_TOPK_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // quant_mode 校验
        if (quantMode != QLI_V2_QUANT_MODE_2) {
            OP_LOGE_FOR_INVALID_VALUE(QLI_V2_ACLNN_OP_NAME, "quant_mode",
                std::to_string(quantMode).c_str(),
                std::to_string(QLI_V2_QUANT_MODE_2));
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_ratio 校验
        if (cmpRatio < QLI_V2_CMP_RATIO_LOWER_BOUND ||
            cmpRatio > QLI_V2_CMP_RATIO_UPPER_BOUND ||
            ((cmpRatio & (cmpRatio - 1)) != 0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cmp_ratio",
                std::to_string(cmpRatio).c_str(),
                "The value of cmp_ratio must be in "
                "[1, 2, 4, 8, 16, 32, 64, 128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (strcmp(layoutKOptional, "PA_BBND") != 0) {
            OP_LOGE_FOR_INVALID_VALUE(QLI_V2_ACLNN_OP_NAME, "layout_k",
                layoutKOptional, "PA_BBND");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else { // 校验 A5参数
        // num_heads_q 校验
        if (numHeadsQ < QLI_V2_NUM_HEADS_Q_LOWER_BOUND ||
            numHeadsQ > QLI_V2_NUM_HEADS_Q_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "num_heads_q",
                std::to_string(numHeadsQ).c_str(),
                "The current value is not within the valid range. "
                "The valid range is [" +
                std::to_string(QLI_V2_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                std::to_string(QLI_V2_NUM_HEADS_Q_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // topk 校验
        if (topk < QLI_V2_TOPK_LOWER_BOUND ||
            topk > QLI_V2_A5_TOPK_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "topk",
                std::to_string(topk).c_str(),
                "The current value is not within the valid range. "
                "The valid range is [" +
                std::to_string(QLI_V2_TOPK_LOWER_BOUND) + ", " +
                std::to_string(QLI_V2_A5_TOPK_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // quant_mode 校验
        if (quantMode != QLI_V2_QUANT_MODE_1 &&
            quantMode != QLI_V2_QUANT_MODE_2 &&
            quantMode != QLI_V2_QUANT_MODE_4) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "quant_mode",
                std::to_string(quantMode).c_str(),
                "The value of quant_mode must be in [1, 2, 4]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_ratio 校验
        if (cmpRatio < QLI_V2_CMP_RATIO_LOWER_BOUND ||
            cmpRatio > QLI_V2_CMP_RATIO_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cmp_ratio",
                std::to_string(cmpRatio).c_str(),
                "The current value is not within the valid range. "
                "The valid range is [" +
                std::to_string(QLI_V2_CMP_RATIO_LOWER_BOUND) + ", " +
                std::to_string(QLI_V2_CMP_RATIO_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (strcmp(layoutKOptional, "TND") != 0 &&
            strcmp(layoutKOptional, "BSND") != 0 &&
            strcmp(layoutKOptional, "PA_BBND") != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "layout_k",
                layoutKOptional,
                "The value of layout_k must be in "
                "[TND, BSND, PA_BBND]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 核心数校验
    if (aicCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "aic_core_num",
            std::to_string(aicCoreNum).c_str(),
            "The value of aic_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aicCoreNum > optiling::AIC_CORE_MAX_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "aic_core_num",
            std::to_string(aicCoreNum).c_str(),
            "The current value is not within the valid range. "
            "The valid range is [1, " +
            std::to_string(optiling::AIC_CORE_MAX_NUM) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "aiv_core_num",
            std::to_string(aivCoreNum).c_str(),
            "The value of aiv_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum > optiling::AIV_CORE_MAX_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "aiv_core_num",
            std::to_string(aivCoreNum).c_str(),
            "The current value is not within the valid range. "
            "The valid range is [1, " +
            std::to_string(optiling::AIV_CORE_MAX_NUM) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceQliV2(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
    const char *layoutKOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistQliV2(cuSeqlensQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cu_seqlens_q",
                "When layout_q is TND, cu_seqlens_q cannot be empty.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // layout_q BSND, seqused_q 不存在时，max_seqlen_q 不能为-1
    if (strcmp(layoutQOptional, "BSND") == 0 && !IsTensorExistQliV2(sequsedQOptional)) {
        if (maxSeqlenQ <= -1) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "max_seqlen_q",
                "When layout_q is BSND and seqused_q is not provided, "
                "max_seqlen_q cannot be -1.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // cu_seqlens_k 存在性校验
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (!IsTensorExistQliV2(cuSeqlensKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cu_seqlens_k",
                "When layout_k is TND, cu_seqlens_k cannot be empty.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // seqused_k 存在性校验
    if (strcmp(layoutKOptional, "PA_BBND") == 0) {
        if (!IsTensorExistQliV2(sequsedKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "seqused_k",
                "When layout_k is PA_BBND, seqused_k cannot be empty.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // layout_k BSND, seqused_k 不存在时，max_seqlen_k 不能为-1
    if (strcmp(layoutKOptional, "BSND") == 0 && !IsTensorExistQliV2(sequsedKOptional)) {
        if (maxSeqlenK <= -1) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "max_seqlen_k",
                "When layout_k is BSND and seqused_k is not provided, "
                "max_seqlen_k cannot be -1.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // cmp_residual_k 存在性校验
    if (cmpRatio != QLI_V2_CMP_RATIO_LOWER_BOUND && maskMode == QLI_V2_CAUSAL_MASK_MODE) {
        if (!IsTensorExistQliV2(cmpResidualKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cmp_residual_k",
                "When cmp_ratio is not 1 and mask_mode is CAUSAL, "
                "cmp_residual_k cannot be empty.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // metadata 存在性校验
    if (!IsTensorExistQliV2(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
            "metadata", "metadata cannot be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyQliV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, const char *layoutQOptional, const char *layoutKOptional,
    const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // 校验 cu_seqlens_q
    if (IsTensorExistQliV2(cuSeqlensQOptional)) {
        dimNum = GetDimNumQliV2(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_q",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cu_seqlens_q",
                ToString(dataType).GetString(),
                "The dtype of cu_seqlens_q must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistQliV2(cuSeqlensKOptional)) {
        dimNum = GetDimNumQliV2(cuSeqlensKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_k",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(cuSeqlensKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cu_seqlens_k",
                ToString(dataType).GetString(),
                "The dtype of cu_seqlens_k must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistQliV2(sequsedQOptional)) {
        dimNum = GetDimNumQliV2(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "seqused_q",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "seqused_q",
                ToString(dataType).GetString(),
                "The dtype of seqused_q must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_k
    if (IsTensorExistQliV2(sequsedKOptional)) {
        dimNum = GetDimNumQliV2(sequsedKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "seqused_k",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(sequsedKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "seqused_k",
                ToString(dataType).GetString(),
                "The dtype of seqused_k must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k
    if (IsTensorExistQliV2(cmpResidualKOptional)) {
        dimNum = GetDimNumQliV2(cmpResidualKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME,
                "cmp_residual_k",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(cmpResidualKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cmp_residual_k",
                ToString(dataType).GetString(),
                "The dtype of cmp_residual_k must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 metadata
    if (IsTensorExistQliV2(metadata)) {
        dimNum = GetDimNumQliV2(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "metadata",
                std::to_string(dimNum).c_str(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "metadata",
                ToString(dataType).GetString(),
                "The dtype of metadata must be int32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::QLI_V2_METADATA_TOTAL_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(QLI_V2_ACLNN_OP_NAME, "metadata",
                std::to_string(metadata->GetViewShape().GetDim(0)).c_str(),
                std::to_string(optiling::QLI_V2_METADATA_TOTAL_SIZE).c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验batch
    std::string querySource;
    int64_t queryBatchSize = GetQueryBatchSizeQliV2(batchSize, cuSeqlensQOptional,
                                                     sequsedQOptional, layoutQOptional,
                                                     querySource);
    std::string keySource;
    int64_t keyBatchSize = GetKeyBatchSizeQliV2(batchSize, cuSeqlensKOptional,
                                                 sequsedKOptional, layoutKOptional,
                                                 keySource);
    if (queryBatchSize != keyBatchSize) {
        if (IsTensorSourceQliV2(querySource) &&
            IsTensorSourceQliV2(keySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                querySource + ", " + keySource,
                std::to_string(GetRawShapeSizeQliV2(querySource, queryBatchSize)) +
                ", " + std::to_string(GetRawShapeSizeQliV2(keySource, keyBatchSize)),
                GetSourceDescQliV2(querySource) + " must be equal to " +
                GetSourceDescQliV2(keySource) + ".");
        } else if (IsTensorSourceQliV2(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                querySource,
                std::to_string(GetRawShapeSizeQliV2(querySource, queryBatchSize)).c_str(),
                GetSourceDescQliV2(querySource) +
                " must be equal to batch_size.");
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                keySource,
                std::to_string(GetRawShapeSizeQliV2(keySource, keyBatchSize)).c_str(),
                GetSourceDescQliV2(keySource) +
                " must be equal to batch_size.");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistQliV2(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cu_seqlens_q, seqused_q",
                std::to_string(cuSeqlensQOptional->GetViewShape().GetDim(0)) +
                ", " + std::to_string(sequsedQOptional->GetViewShape().GetDim(0)),
                "When layout_q is TND and seqused_q is passed, "
                "the shape size of cu_seqlens_q minus 1 must be equal to "
                "the shape size of seqused_q.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验TND场景k维度一致性
    if (strcmp(layoutKOptional, "TND") == 0 && IsTensorExistQliV2(sequsedKOptional)) {
        int64_t cuSeqlensKBatchSize = cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensKBatchSize != keyBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                "cu_seqlens_k, seqused_k",
                std::to_string(cuSeqlensKOptional->GetViewShape().GetDim(0)) +
                ", " + std::to_string(sequsedKOptional->GetViewShape().GetDim(0)),
                "When layout_k is TND and seqused_k is passed, "
                "the shape size of cu_seqlens_k minus 1 must be equal to "
                "the shape size of seqused_k.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k 元素数
    if (IsTensorExistQliV2(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        if (cmpResidualKBatch != queryBatchSize) {
            if (IsTensorSourceQliV2(querySource)) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                    "cmp_residual_k, " + querySource,
                    std::to_string(cmpResidualKBatch) +
                    ", " + std::to_string(GetRawShapeSizeQliV2(querySource, queryBatchSize)),
                    "The shape size of cmp_residual_k must be equal to " +
                    GetSourceDescQliV2(querySource) + ".");
            } else {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(QLI_V2_ACLNN_OP_NAME,
                    "cmp_residual_k",
                    std::to_string(cmpResidualKBatch).c_str(),
                    "The shape size of cmp_residual_k must be equal "
                    "to batch_size.");
            }
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckQliV2(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK,
    int64_t headDim, int64_t topk, int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK,
    char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, const aclTensor *metadata,
    uint32_t aicCoreNum, uint32_t aivCoreNum, const std::string &socVersion)
{
    auto ret = CheckSingleParamQliV2(numHeadsQ, numHeadsK, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK,
        layoutQOptional, layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceQliV2(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
        sequsedKOptional, cmpResidualKOptional, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyQliV2(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
        sequsedKOptional, cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
}

#ifdef __cplusplus
}
#endif
