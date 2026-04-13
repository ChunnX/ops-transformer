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
 * \file aclnn_sparse_attn_sharedkv_grad_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_attn_sharedkv_grad_metadata.h"
#include "l0_sparse_attn_sharedkv_grad_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional, 
                               const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional, 
                               const aclTensor* oriTopkLengthOptional, const aclTensor* cmpTopkLengthOptional, 
                               int64_t numHeadsQ, int64_t numHeadsKv, int64_t batchSizeOptional, 
                               int64_t maxSeqlenQOptional, int64_t oriMaskModeOptional, int64_t oriWinLeftOptional, 
                               char* layoutQOptional, char* layoutKvOptional) 
{
    if (numHeadsQ > 64) {
        return ParamsCheckSasN128(cuSeqLensQOptional, sequsedQOptional, cuSeqLensOriKvOptional, sequsedKvOptional, 
                                  oriTopkLengthOptional, cmpTopkLengthOptional, numHeadsQ, numHeadsKv, 
                                  batchSizeOptional, maxSeqlenQOptional, layoutQOptional, layoutKvOptional);
    } else {
        return ParamsCheckSasN64(cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedQOptional, sequsedKvOptional, 
                                 oriTopkLengthOptional, cmpTopkLengthOptional, numHeadsQ, numHeadsKv, 
                                 batchSizeOptional, maxSeqlenQOptional, oriMaskModeOptional, oriWinLeftOptional, 
                                 layoutQOptional, layoutKvOptional);
    }
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSas(const aclTensor* sequsedQOptional, const aclTensor* cuSeqLensQOptional, 
                             char* layoutQOptional, int64_t batchSizeOptional)
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
    // 3. 如果不是 TND，或者 cuSeqLensQOptional 为空，使用batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

int64_t GetKvBatchSizeSas(const aclTensor* sequsedKvOptional, const aclTensor* cuSeqLensOriKvOptional, 
                          char* layoutKvOptional, int64_t batchSizeOptional)
{
    // 1. 如果sequsedKvOptional传了，使用sequsedKvOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqLensOriKvOptional获取BatchSize
        ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqLensOriKvOptional 为空，使用 batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

aclnnStatus ParamsCheckSasN128(const aclTensor* cuSeqLensQOptional, const aclTensor* sequsedQOptional, 
                               const aclTensor* cuSeqLensOriKvOptional, const aclTensor* sequsedKvOptional, 
                               const aclTensor* oriTopkLengthOptional, const aclTensor* cmpTopkLengthOptional, 
                               int64_t numHeadsQ, int64_t numHeadsKv, int64_t batchSizeOptional, 
                               int64_t maxSeqlenQOptional, char* layoutQOptional, char* layoutKvOptional)
{
    if (CheckSingleParamSasN128(batchSizeOptional, maxSeqlenQOptional, numHeadsQ, numHeadsKv, layoutQOptional, 
                                layoutKvOptional) == ACLNN_SUCCESS && 
        CheckExistenceSas(layoutQOptional, layoutKvOptional, cuSeqLensQOptional, sequsedKvOptional) == ACLNN_SUCCESS && 
        CheckConsistencySas(oriTopkLengthOptional, cmpTopkLengthOptional, sequsedQOptional, cuSeqLensQOptional, 
                            layoutQOptional, sequsedKvOptional, cuSeqLensOriKvOptional, layoutKvOptional, 
                            batchSizeOptional, maxSeqlenQOptional) == ACLNN_SUCCESS &&
        CheckFeatureSas() == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

aclnnStatus CheckSingleParamSasN128(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t numHeadsQ, 
                                    int64_t numHeadsKv, char* layoutQOptional, char* layoutKvOptional)
{
    // batch_size 非负校验
    if (batchSizeOptional < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should not be negative, but got %d", batchSizeOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 非负校验
    if (maxSeqlenQOptional < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should not be negative, but got %d", maxSeqlenQOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q 校验
    if (numHeadsQ > 128) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should not be greater than 128, but got %d", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv 校验
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %d", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv 校验
    if ((strcmp(layoutKvOptional, "PA_BNBD") != 0) && (strcmp(layoutKvOptional, "TND") != 0) && 
        (strcmp(layoutKvOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BNBD!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout交叉校验
    if (strcmp(layoutQOptional, "TND") == 0 && strcmp(layoutKvOptional, "BSND") == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query TND, layout_key should be PA_BSND/TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQOptional, "BSND") == 0 && strcmp(layoutKvOptional, "TND") == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query BSND, layout_key should be PA_BSND/BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSas(char* layoutQOptional, char* layoutKvOptional, const aclTensor* cuSeqLensQOptional, 
                              const aclTensor* sequsedKvOptional)
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
    // layout_kv "PA_BNBD" 存在性校验
    if (strcmp(layoutKvOptional, "PA_BNBD") == 0) {
        auto ret = aclGetViewShape(sequsedKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_kv PA_BNBD, seqused_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencySas(const aclTensor* oriTopkLengthOptional, const aclTensor* cmpTopkLengthOptional, 
                                const aclTensor* sequsedQOptional, const aclTensor* cuSeqLensQOptional, 
                                char* layoutQOptional, const aclTensor* sequsedKvOptional, 
                                const aclTensor* cuSeqLensOriKvOptional, char* layoutKvOptional, 
                                int64_t batchSizeOptional, int64_t maxSeqlenQOptional)
{
    // 校验 q/kv 维度
    int64_t queryBatchSize = GetQueryBatchSizeSas(sequsedQOptional, cuSeqLensQOptional, layoutQOptional, 
                                                  batchSizeOptional);
    int64_t kvBatchSize = GetKvBatchSizeSas(sequsedKvOptional, cuSeqLensOriKvOptional, layoutKvOptional, 
                                            batchSizeOptional);
    if (queryBatchSize != kvBatchSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The batch_size obtained from q Tensor should be the same as "
                "that obtained from kv tensor, but got %ld and %ld", queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // 校验 oriTopkLength 维度
    auto ret = aclGetViewShape(oriTopkLengthOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 非 TND 场景
        if ((strcmp(layoutQOptional, "TND") != 0)) {
            // 校验 cmpTopkLength 的维度数量
            if (viewDimsNum != 2) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is not TND, the dim num of ori_topk_length must be 2, "
                        "but got %ld", viewDimsNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cmpTopkLength 的元素数量是否和q tensor相对应
            int64_t bsSize = batchSizeOptional * maxSeqlenQOptional;
            int64_t oriTopkLengthDims = viewDims[0] * viewDims[1];
            if (oriTopkLengthDims != bsSize) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is not TND, the size of ori_topk_length %ld should be "
                        "equal with the B*S size %ld", oriTopkLengthDims, bsSize);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 当 cmpTopkLength 的维度为[B,S]时，校验 cmpTopkLength 第一维是否和 batchSize 相等
            if (viewDims[0] != queryBatchSize) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the shape of ori_topk_length is [B, S], the first dim of "
                        "ori_topk_length(B) should be equal with the batch size %ld, but got %ld",
                        queryBatchSize, viewDims[0]);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // 校验 cmpTopkLength 维度
    ret = aclGetViewShape(cmpTopkLengthOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 非 TND 场景
        if ((strcmp(layoutQOptional, "TND") != 0)) {
            // 校验 cmpTopkLength 的维度数量
            if (viewDimsNum != 2) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is not TND, the dim num of cmp_topk_length must be 2, "
                        "but got %ld", viewDimsNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cmpTopkLength 的元素数量是否和q tensor相对应
            int64_t bsSize = batchSizeOptional * maxSeqlenQOptional;
            int64_t cmpTopkLengthDims = viewDims[0] * viewDims[1];
            if (cmpTopkLengthDims != bsSize) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is not TND, the size of cmp_topk_length %ld should be "
                        "equal with the B*S size %ld", cmpTopkLengthDims, bsSize);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 当 cmpTopkLength 的维度为[B,S]时，校验 cmpTopkLength 第一维是否和 batchSize 相等
            if (viewDims[0] != queryBatchSize) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the shape of cmp_topk_length is [B, S], the first dim of "
                        "cmp_topk_length(B) should be equal with the batch size %ld, but got %ld",
                        queryBatchSize, viewDims[0]);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFeatureSas()
{
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckSasN64(const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional, 
                              const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional, 
                              const aclTensor* oriTopkLengthOptional, const aclTensor* cmpTopkLengthOptional, 
                              int64_t numHeadsQ, int64_t numHeadsKv, int64_t batchSizeOptional, 
                              int64_t maxSeqlenQOptional, int64_t oriMaskModeOptional, int64_t oriWinLeftOptional, 
                              char* layoutQOptional, char* layoutKvOptional)
{
    if (CheckSingleParamSasN64(batchSizeOptional, maxSeqlenQOptional, numHeadsQ, numHeadsKv, oriMaskModeOptional, 
                               oriWinLeftOptional, layoutQOptional, layoutKvOptional) == ACLNN_SUCCESS && 
        CheckExistenceSas(layoutQOptional, layoutKvOptional, cuSeqLensQOptional, sequsedKvOptional) == ACLNN_SUCCESS && 
        CheckConsistencySas(oriTopkLengthOptional, cmpTopkLengthOptional, sequsedQOptional, cuSeqLensQOptional, 
                            layoutQOptional, sequsedKvOptional, cuSeqLensOriKvOptional, layoutKvOptional, 
                            batchSizeOptional, maxSeqlenQOptional) == ACLNN_SUCCESS && 
        CheckFeatureSas() == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

aclnnStatus CheckSingleParamSasN64(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t numHeadsQ, 
                                   int64_t numHeadsKv, int64_t oriMaskMode_, int64_t winLeft_, char* layoutQOptional, 
                                   char* layoutKvOptional)
{
    // batch_size 非负校验
    if (batchSizeOptional < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should not be negative, but got %d", batchSizeOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 非负校验
    if (maxSeqlenQOptional < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should not be negative, but got %d", maxSeqlenQOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q 校验
    if (numHeadsQ > 128) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should not be greater than 128, but got %d", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv 校验
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %d", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // ori_mask_mode 校验
    if (oriMaskMode_ != 0 && oriMaskMode_ != 3 && oriMaskMode_ != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ori_mask_mode should be 0, 3 or 4, but got %d", oriMaskMode_);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // ori_win_left 校验
    if (winLeft_ != 127) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ori_win_left should only be 127, but got %ld", winLeft_);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv 校验
    if ((strcmp(layoutKvOptional, "PA_BNBD") != 0) && (strcmp(layoutKvOptional, "TND") != 0) && 
        (strcmp(layoutKvOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BNBD!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout交叉校验
    if (strcmp(layoutQOptional, "TND") == 0 && strcmp(layoutKvOptional, "BSND") == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query TND, layout_key should be PA_BSND/TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQOptional, "BSND") == 0 && strcmp(layoutKvOptional, "TND") == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query BSND, layout_key should be PA_BSND/BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseAttnSharedkvGradMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional,const aclTensor* cuSeqLensOriKvOptional, 
    const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
    const aclTensor* oriTopkLengthOptional, const aclTensor* cmpTopkLengthOptional, int64_t numHeadsQ, 
    int64_t numHeadsKv, int64_t headDim, int64_t batchSizeOptional, int64_t maxSeqlenQOptional, 
    int64_t maxSeqlenKvOptional, int64_t oriTopKOptional, int64_t cmpTopKOptional, int64_t cmpRatioOptional, 
    int64_t oriMaskModeOptional, int64_t cmpMaskModeOptional, int64_t oriWinLeftOptional, int64_t oriWinRightOptional, 
    char *layoutQOptional, char *layoutKvOptional, bool hasOriKvOptional, bool hasCmpKvOptional, 
    int64_t deterministicOptional, const aclTensor* metaData, uint64_t* workspaceSize, aclOpExecutor** executor) 
{
    L2_DFX_PHASE_1(aclnnSparseAttnSharedkvGradMetadata,
                   DFX_IN(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional, 
                          sequsedKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, 
                          batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, 
                          cmpRatioOptional, oriMaskModeOptional, cmpMaskModeOptional, oriWinLeftOptional, 
                          oriWinRightOptional, layoutQOptional, layoutKvOptional, hasOriKvOptional, hasCmpKvOptional, 
                          deterministicOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedQOptional, sequsedKvOptional, 
                           oriTopkLengthOptional, cmpTopkLengthOptional, numHeadsQ, numHeadsKv, batchSizeOptional,
                           maxSeqlenQOptional, oriMaskModeOptional, oriWinLeftOptional, layoutQOptional, 
                           layoutKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();
    // 核心数校验
    if (aicCoreNum == 0 || aivCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIC num or AIV num should not be 0, but got %u and %u", aicCoreNum, aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto cuSeqLensQOptionalContiguous = l0op::Contiguous(cuSeqLensQOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLensQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cuSeqLensOriKvOptionalContiguous = l0op::Contiguous(cuSeqLensOriKvOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLensOriKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cuSeqLensCmpKvOptionalContiguous = l0op::Contiguous(cuSeqLensCmpKvOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLensCmpKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
    CHECK_RET(sequsedQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sequsedKvOptionalContiguous = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
    CHECK_RET(sequsedKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto oriTopkLengthOptionalContiguous = l0op::Contiguous(oriTopkLengthOptional, uniqueExecutor.get());
    CHECK_RET(oriTopkLengthOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cmpTopkLengthOptionalContiguous = l0op::Contiguous(cmpTopkLengthOptional, uniqueExecutor.get());
    CHECK_RET(cmpTopkLengthOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto output = l0op::SparseAttnSharedkvGradMetadata(
        cuSeqLensQOptionalContiguous, cuSeqLensOriKvOptionalContiguous, cuSeqLensCmpKvOptionalContiguous, 
        sequsedQOptionalContiguous, sequsedKvOptionalContiguous, oriTopkLengthOptionalContiguous, 
        cmpTopkLengthOptionalContiguous, numHeadsQ, numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, 
        maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, cmpRatioOptional, oriMaskModeOptional, 
        cmpMaskModeOptional, oriWinLeftOptional, oriWinRightOptional, layoutQOptional, layoutKvOptional, 
        hasOriKvOptional, hasCmpKvOptional, deterministicOptional, socVersion, aicCoreNum, aivCoreNum, metaData, 
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnSparseAttnSharedkvGradMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) 
{
    L2_DFX_PHASE_2(aclnnSparseAttnSharedkvGradMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
