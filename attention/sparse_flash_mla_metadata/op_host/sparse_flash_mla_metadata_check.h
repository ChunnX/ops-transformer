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
 * \file sparse_flash_mla_metadata_check.h.cpp
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../sparse_flash_mla/op_kernel/sparse_flash_mla_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

static bool IsPowerOfTwoInRangeSmla(int64_t value, int64_t lower, int64_t upper)
{
    return value >= lower && value <= upper && (value & (value - 1)) == 0;
}

static bool IsCmpRatioSupportSmla(bool isA5, bool hasCmpKv, int64_t cmpTopK, int64_t cmpRatio)
{
    if (isA5) {
        return cmpRatio >= 1 && cmpRatio <= 128;
    }
    if (!hasCmpKv) {
        return cmpRatio == 1;
    }
    return (cmpTopK > 0) ? (cmpRatio == 4) : (cmpRatio == 128);
}

aclnnStatus CheckSingleParamSas(int64_t batchSize, int64_t maxSeqlenQ, int64_t numHeadsQ, int64_t numHeadsKv,
                                int64_t headDim, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode,
                                int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, char* layoutQOptional,
                                char* layoutKvOptional, bool hasCmpKv, bool isA5)
{
    // batch_size 非负校验
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should not be negative, but got %ld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 非负校验
    if (maxSeqlenQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should not be negative, but got %ld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv 校验
    if (headDim != 512) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_dim should only be 512, but got %ld", headDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %ld", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsQ < 1 || numHeadsQ > 128 || numHeadsQ % numHeadsKv != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "num_heads_q should be in [1, 128] and divisible by num_heads_kv, but got %ld", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t headRatio = numHeadsQ / numHeadsKv;
    if (!isA5 && !IsPowerOfTwoInRangeSmla(headRatio, 1, 128)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "num_heads_q / num_heads_kv should be power of two in [1, 128], but got %ld", headRatio);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isA5) {
        if (oriMaskMode != 0 && oriMaskMode != 3 && oriMaskMode != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ori_mask_mode should be 0, 3 or 4, but got %ld", oriMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cmpMaskMode != 0 && cmpMaskMode != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_mask_mode should be 0 or 3, but got %ld", cmpMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (oriWinLeft < -1 || oriWinRight < -1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "ori_win_left and ori_win_right should be -1 or non-negative, but got %ld and %ld",
                    oriWinLeft, oriWinRight);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (oriMaskMode != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ori_mask_mode should only be 4 on A2/A3, but got %ld", oriMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cmpMaskMode != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_mask_mode should only be 3 on A2/A3, but got %ld", cmpMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (oriWinLeft != 127 || oriWinRight != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "ori_win_left and ori_win_right should only be 127 and 0 on A2/A3, but got %ld and %ld",
                    oriWinLeft, oriWinRight);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cmpTopK != 0 && cmpTopK != 512 && cmpTopK != 1024) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_topk should be 0, 512 or 1024, but got %ld", cmpTopK);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!IsCmpRatioSupportSmla(isA5, hasCmpKv, cmpTopK, cmpRatio)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio is not supported, got %ld", cmpRatio);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutQOptional == nullptr || layoutKvOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q and layout_kv must not be nullptr!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q check
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv 校验
    if ((strcmp(layoutKvOptional, "PA_BBND") != 0) && (strcmp(layoutKvOptional, "TND") != 0) &&
        (strcmp(layoutKvOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BBND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSas(char* layoutQOptional, char* layoutKvOptional, const aclTensor* cuSeqLensQOptional,
                              const aclTensor* cuSeqLensOriKvOptional, const aclTensor* cuSeqLensCmpKvOptional,
                              const aclTensor* sequsedOriKvOptional, bool hasCmpKv, const aclTensor* metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_q TND, cu_seqlens_q must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // seqused_ori_kv 存在性校验
    if (strcmp(layoutKvOptional, "PA_BBND") == 0) {
        auto ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_kv PA_BBND, seqused_ori_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKvOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_kv TND, cu_seqlens_ori_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKvOptional, "TND") == 0 && hasCmpKv) {
        auto ret = aclGetViewShape(cuSeqLensCmpKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_kv TND and has_cmp_kv is true, "
                "cu_seqlens_cmp_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // metadata 存在性校验
    auto ret = aclGetViewShape(metadata, &viewDims, &viewDimsNum);
    if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output metadata is nullptr");
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

static bool IsTensorSpecifiedSas(const aclTensor* tensor)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(tensor, &viewDims, &viewDimsNum);
    bool specified = ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0;
    delete[] viewDims;
    return specified;
}

static aclnnStatus CheckInt32VectorOptionalSas(const aclTensor* tensor, const char *name)
{
    if (!IsTensorSpecifiedSas(tensor)) {
        return ACLNN_SUCCESS;
    }
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(tensor, &viewDims, &viewDimsNum);
    if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of %s must be 1", name);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    ret = aclGetDataType(tensor, &dataType);
    if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of %s must be int32", name);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckNewOptionalInputsSas(const aclTensor* sequsedCmpKvOptional,
                                             const aclTensor* cmpResidualKvOptional,
                                             const aclTensor* oriTopkLengthOptional,
                                             const aclTensor* cmpTopkLengthOptional, int64_t cmpRatio,
                                             int64_t cmpMaskMode, bool hasCmpKv)
{
    if (IsTensorSpecifiedSas(oriTopkLengthOptional) || IsTensorSpecifiedSas(cmpTopkLengthOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ori_topk_length and cmp_topk_length are reserved and must be empty in current version.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasCmpKv && cmpMaskMode == 3 && cmpRatio != 1 && !IsTensorSpecifiedSas(cmpResidualKvOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "cmp_residual_kv is required when cmp_mask_mode is 3 and cmp_ratio is not 1.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckInt32VectorOptionalSas(sequsedCmpKvOptional, "seqused_cmp_kv") != ACLNN_SUCCESS ||
        CheckInt32VectorOptionalSas(cmpResidualKvOptional, "cmp_residual_kv") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSas(const aclTensor* sequsedQOptional, const aclTensor* cuSeqLensQOptional,
                             char* layoutQOptional, int64_t batchSize)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedQOptional 没传，判断 Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqLensQOptional获取BatchSize
        ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqLensQOptional 为空，使用batchSize
    delete[] viewDims;
    return batchSize;
}

int64_t GetKvBatchSizeSas(const aclTensor* sequsedOriKvOptional, const aclTensor* cuSeqLensOriKvOptional,
                          char* layoutKvOptional, int64_t batchSize)
{
    // 1. 如果sequsedOriKvOptional传了，使用sequsedOriKvOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedOriKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqLensOriKvOptional获取BatchSize
        ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqLensOriKvOptional 为空，使用 batchSize
    delete[] viewDims;
    return batchSize;
}

aclnnStatus CheckConsistencySas(const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
                                const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional,
                                const aclTensor* sequsedOriKvOptional, char* layoutQOptional, char* layoutKvOptional,
                                int64_t batchSize, const aclTensor* metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    // 校验 cu_seqlens_q
    auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 cu_seqlens_q 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        ret = aclGetDataType(cuSeqLensQOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seqlens_ori_kv
    ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 cu_seqlens_ori_kv 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_ori_kv must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_ori_kv 数据类型
        ret = aclGetDataType(cuSeqLensOriKvOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_ori_kv must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seqlens_cmp_kv
    ret = aclGetViewShape(cuSeqLensCmpKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 cu_seqlens_cmp_kv 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_cmp_kv must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_cmp_kv 数据类型
        ret = aclGetDataType(cuSeqLensCmpKvOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_cmp_kv must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 seqused_q 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_q 数据类型
        ret = aclGetDataType(sequsedQOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_ori_kv
    ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 seqused_ori_kv 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_ori_kv must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_ori_kv 数据类型
        ret = aclGetDataType(sequsedOriKvOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_ori_kv must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 metadata
    ret = aclGetViewShape(metadata, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 metadata 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (viewDims[0] != optiling::SMLA_META_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %ld",
                optiling::SMLA_META_SIZE, viewDims[0]);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 数据类型
        ret = aclGetDataType(metadata, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of metadata must be int32");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 q/kv 维度
    int64_t queryBatchSize = GetQueryBatchSizeSas(sequsedQOptional, cuSeqLensQOptional, layoutQOptional,
                                                  batchSize);
    int64_t kvBatchSize = GetKvBatchSizeSas(sequsedOriKvOptional, cuSeqLensOriKvOptional, layoutKvOptional,
                                            batchSize);
    if (queryBatchSize != kvBatchSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The batch_size obtained from q Tensor should be the same as "
                "that obtained from kv tensor, but got %ld and %ld", queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
                               const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional,
                               const aclTensor* sequsedOriKvOptional, const aclTensor* sequsedCmpKvOptional,
                               const aclTensor* cmpResidualKvOptional, const aclTensor* oriTopkLengthOptional,
                               const aclTensor* cmpTopkLengthOptional, int64_t numHeadsQ, int64_t numHeadsKv,
                               int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
                               int64_t maxSeqlenCmpKv, int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio,
                               int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
                               char* layoutQOptional, char* layoutKvOptional, bool hasOriKv, bool hasCmpKv,
                               const aclTensor* metadata, bool isA5)
{
    if (CheckSingleParamSas(batchSize, maxSeqlenQ, numHeadsQ, numHeadsKv, headDim, cmpTopK, cmpRatio, oriMaskMode,
                            cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, hasCmpKv,
                            isA5) == ACLNN_SUCCESS &&
        CheckNewOptionalInputsSas(sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                                  cmpTopkLengthOptional, cmpRatio, cmpMaskMode, hasCmpKv) == ACLNN_SUCCESS &&
        CheckExistenceSas(layoutQOptional, layoutKvOptional, cuSeqLensQOptional, cuSeqLensOriKvOptional,
                          cuSeqLensCmpKvOptional, sequsedOriKvOptional, hasCmpKv, metadata) == ACLNN_SUCCESS &&
        CheckConsistencySas(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                            sequsedOriKvOptional, layoutQOptional, layoutKvOptional, batchSize,
                            metadata) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

#ifdef __cplusplus
}
#endif
