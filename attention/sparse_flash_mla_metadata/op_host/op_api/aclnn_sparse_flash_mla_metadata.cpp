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
 * \file aclnn_sparse_flash_mla_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_flash_mla_metadata.h"
#include "l0_sparse_flash_mla_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensOriKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional, int64_t numHeadsQ, int64_t numHeadsKv,
    int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKvOptional, int64_t oriTopKOptional,
    int64_t oriMaskModeOptional, char *layoutQOptional, char *layoutKvOptional, bool hasOriKvOptional)
{
    auto ret = CheckSingleParamSmla(batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, numHeadsQ, numHeadsKv,
        oriTopKOptional, oriMaskModeOptional, layoutQOptional, layoutKvOptional, hasOriKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceSmla(cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedOriKvOptional, layoutQOptional,
        layoutKvOptional, hasOriKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencySmla(cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedQOptional, sequsedOriKvOptional,
        batchSizeOptional, layoutQOptional, layoutKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckFeatureSmla();
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSmla(const aclTensor* cuSeqLensQOptional, const aclTensor *sequsedQOptional,
    char *layoutQOptional, int64_t batchSizeOptional)
{
    // 1. 如果 sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedQOptional 没传，判断Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 使用batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

int64_t GetKvBatchSizeSmla(const aclTensor* cuSeqLensOriKvOptional, const aclTensor *sequsedOriKvOptional,
    char *layoutKvOptional, int64_t batchSizeOptional)
{
    // 1. 如果 sequsedOriKvOptional 传了，使用 sequsedOriKvOptional 获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedOriKvOptional 没传，判断Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 如果不是 TND，或者 cuSeqLensOriKvOptional 为空，使用 batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

aclnnStatus CheckSingleParamSmla(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t oriTopKOptional, int64_t oriMaskModeOptional,
    char *layoutQOptional, char *layoutKvOptional, bool hasOriKvOptional)
{
    // batch_size 非负校验
    CHECK_COND(batchSizeOptional >= 0, ACLNN_ERR_PARAM_INVALID,
        "batch_size should not be negative, but got %d", batchSizeOptional);
    // max_seqlen_q 非负校验
    CHECK_COND(maxSeqlenQOptional >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_q should not be negative, but got %d", maxSeqlenQOptional);
    // max_seqlen_kv 非负校验
    CHECK_COND(maxSeqlenKvOptional >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_q should not be negative, but got %d", maxSeqlenKvOptional);
    // num_heads_q 校验
    CHECK_COND(numHeadsQ >= 0, ACLNN_ERR_PARAM_INVALID,
        "num_heads_q should not be negative, but got %d", numHeadsQ);
    // num_heads_kv 校验
    CHECK_COND(numHeadsKv >= 0, ACLNN_ERR_PARAM_INVALID,
        "num_heads_kv should not be negative, but got %d", numHeadsKv);
    // ori_topk 校验
    CHECK_COND(oriTopKOptional >= 0, ACLNN_ERR_PARAM_INVALID,
        "ori_topk should not be negative, but got %d", oriTopKOptional);
    // ori_mask_mode 校验
    CHECK_COND((oriMaskModeOptional == 0) || (oriMaskModeOptional == 3), ACLNN_ERR_PARAM_INVALID,
        "ori_mask_mode should be 0 or 3, but got %d", oriMaskModeOptional);
    // layout_q 校验
    CHECK_COND((strcmp(layoutQOptional, "TND") == 0) || (strcmp(layoutQOptional, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
        "layout_q must be TND or BSND, but got %s", layoutQOptional);
    // layout_kv 校验
    CHECK_COND((strcmp(layoutKvOptional, "PA_BNBD") == 0) || (strcmp(layoutKvOptional, "TND") == 0) || \
        (strcmp(layoutQOptional, "BSND") == 0), 
        ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BNBD, but got %s", layoutKvOptional);
    // has_ori_kv 校验
    CHECK_COND(hasOriKvOptional == true,
        ACLNN_ERR_PARAM_INVALID, "has_ori_kv must be true, but got %d", hasOriKvOptional);

    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSmla(const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensOriKvOptional,
    const aclTensor *sequsedOriKvOptional, char *layoutQOptional, char *layoutKvOptional, bool hasOriKvOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // Query 存在性逻辑
    if (strcmp(layoutQOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query TND, cu_seqlens_q must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // KV 存在性逻辑
    if (strcmp(layoutKvOptional, "PA_BSND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key PA_BSND, cu_seqlens_ori_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKvOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key TND, cu_seqlens_ori_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (hasOriKvOptional) {
        auto ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        bool isCuSeqLenOriKvOptionalProvided = (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0);
        ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
        bool isSequsedOriKvOptionalProvided = (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0);

        if (!isCuSeqLenOriKvOptionalProvided && !sequsedOriKvOptional) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When has_ori_kv is true, at least one of cu_seqlens_ori_kv and seqysed_ori_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencySmla(const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional, int64_t batchSizeOptional,
    char *layoutQOptional, char *layoutKvOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

// 校验 cu_seqlens_q
    auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0) {
        // 校验 cu_seqlens_q 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %ld", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        ret = aclGetDataType(cuSeqLensQOptional, &dataType);
        if (ret == ACLNN_ERR_PARAM_INVALID || dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_q must be int32, but got %d",
                static_cast<int32_t>(dataType));
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // 校验 cu_seqlens_ori_kv
    ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0) {
        // 校验 cu_seqlens_ori_kv 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_ori_kv must be 1, but got %ld", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_ori_kv 数据类型
        ret = aclGetDataType(cuSeqLensOriKvOptional, &dataType);
        if (ret == ACLNN_ERR_PARAM_INVALID || dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_ori_kv must be int32, but got %d",
                static_cast<int32_t>(dataType));
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // 校验 seqused_q
    ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0) {
        // 校验 seqused_q 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %ld", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_q 数据类型
        ret = aclGetDataType(sequsedQOptional, &dataType);
        if (ret == ACLNN_ERR_PARAM_INVALID || dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_q must be int32, but got %d",
                static_cast<int32_t>(dataType));
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // 校验 seqused_ori_kv
    ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] > 0) {
        // 校验 seqused_ori_kv 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_ori_kv must be 1, but got %ld", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_ori_kv 数据类型
        ret = aclGetDataType(sequsedOriKvOptional, &dataType);
        if (ret == ACLNN_ERR_PARAM_INVALID || dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_ori_kv must be int32, but got %d",
                static_cast<int32_t>(dataType));
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    int64_t queryBatchSize = GetQueryBatchSizeSmla(cuSeqLensQOptional, sequsedQOptional, layoutQOptional,
        batchSizeOptional);
    int64_t kvBatchSize = GetKvBatchSizeSmla(cuSeqLensOriKvOptional, sequsedOriKvOptional, layoutKvOptional,
        batchSizeOptional);
    if (queryBatchSize != kvBatchSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The batch_size obtained from q Tensor should be the same as "
                "that obtained from kv tensor, but got %ld and %ld", queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFeatureSmla()
{
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensOriKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSizeOptional, int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKvOptional, int64_t oriTopKOptional, int64_t oriMaskModeOptional,
    char *layoutQOptional, char *layoutKvOptional, bool hasOriKvOptional,
    const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor) 
{
    L2_DFX_PHASE_1(aclnnSparseFlashMlaMetadata,
                   DFX_IN(cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedQOptional, sequsedOriKvOptional,
                          numHeadsQ, numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional,
                          oriTopKOptional, oriMaskModeOptional, layoutQOptional, layoutKvOptional, hasOriKvOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedQOptional, sequsedOriKvOptional,
        numHeadsQ, numHeadsKv, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional,
        oriMaskModeOptional, layoutQOptional, layoutKvOptional, hasOriKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();
    // 核心数校验
    CHECK_COND(aicCoreNum != 0 && aivCoreNum != 0, ACLNN_ERR_PARAM_INVALID,
        "AIC num or AIV num should not be 0, but got %u and %u", aicCoreNum, aivCoreNum);

    auto cuSeqLensQOptionalContiguous = l0op::Contiguous(cuSeqLensQOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLensQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cuSeqLensOriKvOptionalContiguous = l0op::Contiguous(cuSeqLensOriKvOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLensOriKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
    CHECK_RET(sequsedQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sequsedOriKvOptionalContiguous = l0op::Contiguous(sequsedOriKvOptional, uniqueExecutor.get());
    CHECK_RET(sequsedOriKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto output = l0op::SparseFlashMlaMetadata(
        cuSeqLensQOptionalContiguous, cuSeqLensOriKvOptionalContiguous,
        sequsedQOptionalContiguous, sequsedOriKvOptionalContiguous,
        numHeadsQ, numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional,
        oriTopKOptional, oriMaskModeOptional, layoutQOptional, layoutKvOptional,
        hasOriKvOptional, socVersion, aicCoreNum, aivCoreNum, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseFlashMlaMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
