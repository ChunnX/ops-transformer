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
 * \file quant_block_sparse_attention_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../op_kernel_aicpu/quant_block_sparse_attention_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus CheckSingleParamQbsa(int64_t batchSize, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
                                         int64_t sparseBlockSizeQ, int64_t sparseBlockSizeKv, int64_t quantMode,
                                         int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                         const char *layoutQ, const char *layoutKv,
                                         const char *layoutSparseIndices)
{
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "batchSize must be >= 0, but got %ld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsQ < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "numHeadsQ must be >= 0, but got %ld", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsKv < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "numHeadsKv must be >= 0, but got %ld", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (headDim < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "headDim must be >= 0, but got %ld", headDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maskMode != 0 && maskMode != 3) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maskMode only supports 0, 3, but got %ld", maskMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maxSeqlenQ < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maxSeqlenQ must be >= -1, but got %ld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maxSeqlenKv < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maxSeqlenKv must be >= -1, but got %ld", maxSeqlenKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // TODO: validate sparse_block_size_q range
    // TODO: validate sparse_block_size_kv range
    // TODO: validate quant_mode range (default value unclear)
    bool layoutQValid = (strcmp(layoutQ, "BSND") == 0 || strcmp(layoutQ, "TND") == 0 || strcmp(layoutQ, "BNSD") == 0);
    if (!layoutQValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutQ only supports BSND, TND, BNSD, but got %s", layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    bool layoutKvValid =
        (strcmp(layoutKv, "BSND") == 0 || strcmp(layoutKv, "TND") == 0 || strcmp(layoutKv, "BNSD") == 0 ||
         strcmp(layoutKv, "PA_BNSD") == 0 || strcmp(layoutKv, "PA_BNBD") == 0 || strcmp(layoutKv, "PA_BBND") == 0 ||
         strcmp(layoutKv, "PA_NZ") == 0);
    if (!layoutKvValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "layoutKv only supports BSND, TND, BNSD, PA_BNSD, PA_BNBD, PA_BBND, PA_NZ, but got %s", layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // TODO: validate layout_sparse_indices enumeration
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckExistenceQbsa(const aclTensor *sparseSeqLen, const aclTensor *cuSeqlensQOptional,
                                       const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                                       const aclTensor *sequsedKvOptional, int64_t batchSize,
                                       uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion,
                                       const aclTensor *metaData)
{
    if (sparseSeqLen == nullptr) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseSeqLen must not be null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseSeqLen->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseSeqLen must be 1D tensor, but got %ld dims",
                sparseSeqLen->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseSeqLen->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseSeqLen dtype must be INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSeqlensQOptional != nullptr) {
        if (cuSeqlensQOptional->GetViewShape().GetDimNum() != 1) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional must be 1D tensor, but got %ld dims",
                    cuSeqlensQOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cuSeqlensQOptional->GetDataType() != ACL_INT32) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional dtype must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (batchSize > 0 && cuSeqlensQOptional->GetViewShape().GetDim(0) != batchSize + 1) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional shape must be (batchSize+1,), but got %ld",
                    cuSeqlensQOptional->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (cuSeqlensKvOptional != nullptr) {
        if (cuSeqlensKvOptional->GetViewShape().GetDimNum() != 1) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensKvOptional must be 1D tensor, but got %ld dims",
                    cuSeqlensKvOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cuSeqlensKvOptional->GetDataType() != ACL_INT32) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensKvOptional dtype must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (batchSize > 0 && cuSeqlensKvOptional->GetViewShape().GetDim(0) != batchSize + 1) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensKvOptional shape must be (batchSize+1,), but got %ld",
                    cuSeqlensKvOptional->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (sequsedQOptional != nullptr) {
        if (sequsedQOptional->GetViewShape().GetDimNum() != 1) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedQOptional must be 1D tensor, but got %ld dims",
                    sequsedQOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sequsedQOptional->GetDataType() != ACL_INT32) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedQOptional dtype must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (batchSize > 0 && sequsedQOptional->GetViewShape().GetDim(0) != batchSize) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedQOptional shape must be (batchSize,), but got %ld",
                    sequsedQOptional->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (sequsedKvOptional != nullptr) {
        if (sequsedKvOptional->GetViewShape().GetDimNum() != 1) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedKvOptional must be 1D tensor, but got %ld dims",
                    sequsedKvOptional->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sequsedKvOptional->GetDataType() != ACL_INT32) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedKvOptional dtype must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (batchSize > 0 && sequsedKvOptional->GetViewShape().GetDim(0) != batchSize) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedKvOptional shape must be (batchSize,), but got %ld",
                    sequsedKvOptional->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // metadata
    if (metaData == nullptr) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metaData must not be null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metaData->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metaData must be 1D tensor, but got %ld dims",
                metaData->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metaData->GetViewShape().GetDim(0) != optiling::QBSA_METADATA_SIZE) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metaData shape must be (%u,), but got %ld",
                optiling::QBSA_METADATA_SIZE, metaData->GetViewShape().GetDim(0));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metaData->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metaData dtype must be INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // platform
    if (aicCoreNum <= 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aicCoreNum must be > 0, but got %u", aicCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum <= 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aivCoreNum must be > 0, but got %u", aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // TODO: validate supported SOC for socVersion
    // TODO: validate dependency: when layout_q is TND, cu_seqlens_q must be provided
    // TODO: validate dependency: when layout_kv is TND, cu_seqlens_kv must be provided
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckConsistencyQbsa(int64_t numHeadsQ, int64_t numHeadsKv)
{
    if (numHeadsKv != 0 && numHeadsQ % numHeadsKv != 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "numHeadsQ must be divisible by numHeadsKv, but got numHeadsQ=%ld, numHeadsKv=%ld", numHeadsQ,
                numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // TODO: validate Q/KV batch consistency
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *sparseSeqLen, const aclTensor *cuSeqlensQOptional,
                                const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                                const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t numHeadsQ,
                                int64_t numHeadsKv, int64_t headDim, int64_t sparseBlockSizeQ,
                                int64_t sparseBlockSizeKv, int64_t quantMode, int64_t maskMode, int64_t maxSeqlenQ,
                                int64_t maxSeqlenKv, const char *layoutQOptional, const char *layoutKvOptional,
                                const char *layoutSparseIndicesOptional, uint32_t aicCoreNum, uint32_t aivCoreNum,
                                const char *socVersion, const aclTensor *metaData)
{
    aclnnStatus ret = CheckSingleParamQbsa(batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ,
                                           sparseBlockSizeKv, quantMode, maskMode, maxSeqlenQ, maxSeqlenKv,
                                           layoutQOptional, layoutKvOptional, layoutSparseIndicesOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckExistenceQbsa(sparseSeqLen, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional,
                             sequsedKvOptional, batchSize, aicCoreNum, aivCoreNum, socVersion, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckConsistencyQbsa(numHeadsQ, numHeadsKv);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif