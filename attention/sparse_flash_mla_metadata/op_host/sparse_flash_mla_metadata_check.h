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
#include <cstring>
#include "../../sparse_flash_mla/op_kernel/sparse_flash_mla_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus CheckSingleParamSmla(int64_t batchSize, int64_t maxSeqlenQ, int64_t numHeadsQ, int64_t numHeadsKv,
                                int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode, int64_t oriWinLeft,
                                char* layoutQOptional, char* layoutKvOptional, bool hasCmpKv)
{
    (void)numHeadsQ;
    // batch_size 闈炶礋鏍￠獙
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should not be negative, but got %ld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 闈炶礋鏍￠獙
    if (maxSeqlenQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should not be negative, but got %ld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv 鏍￠獙
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %ld", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // ori_mask_mode 鏍￠獙
    if (oriMaskMode != 0 && oriMaskMode != 3 && oriMaskMode != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ori_mask_mode should be 0, 3 or 4, but got %ld", oriMaskMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // ori_win_left 鏍￠獙
    if (oriWinLeft != 127) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ori_win_left should only be 127, but got %ld", oriWinLeft);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasCmpKv) {
        // cmp_ratio 鏍￠獙
        if (cmpRatio != 4 && cmpRatio != 128) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In CFA or SCFA, cmp_ratio should only be 4 or 128, but got %ld", cmpRatio);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_topk 鏍￠獙
        if (cmpTopK != 0 && cmpTopK != 512 && cmpTopK != 1024) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_topk should be 0, 512 or 1024, but got %ld", cmpTopK);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // layout_q 鏍￠獙
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv 鏍￠獙
    if ((strcmp(layoutKvOptional, "PA_ND") != 0) && (strcmp(layoutKvOptional, "TND") != 0) &&
        (strcmp(layoutKvOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_ND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSmla(char* layoutQOptional, char* layoutKvOptional, const aclTensor* cuSeqLensQOptional,
                              const aclTensor* cuSeqLensOriKvOptional, const aclTensor* cuSeqLensCmpKvOptional,
                              const aclTensor* sequsedKvOptional, bool hasCmpKv, const aclTensor* metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // cu_seqlens_q 瀛樺湪鎬ф牎楠?
    if (strcmp(layoutQOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_q TND, cu_seqlens_q must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // seqused_kv 瀛樺湪鎬ф牎楠?
    if (strcmp(layoutKvOptional, "PA_ND") == 0) {
        auto ret = aclGetViewShape(sequsedKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_kv PA_ND, seqused_kv must be provided!");
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
    // metadata 瀛樺湪鎬ф牎楠?
    auto ret = aclGetViewShape(metadata, &viewDims, &viewDimsNum);
    if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output metadata is nullptr");
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSmla(const aclTensor* sequsedQOptional, const aclTensor* cuSeqLensQOptional,
                             char* layoutQOptional, int64_t batchSize)
{
    // 1. 濡傛灉sequsedQOptional 浼犱簡锛屼娇鐢╯equsedQOptional鑾峰彇BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedQOptional 娌′紶锛屽垽鏂?Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        // 濡傛灉鏄?TND锛屽皾璇曚娇鐢?cuSeqLensQOptional鑾峰彇BatchSize
        ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 濡傛灉涓嶆槸 TND锛屾垨鑰?cuSeqLensQOptional 涓虹┖锛屼娇鐢╞atchSize
    delete[] viewDims;
    return batchSize;
}

int64_t GetKvBatchSizeSmla(const aclTensor* sequsedKvOptional, const aclTensor* cuSeqLensOriKvOptional,
                          char* layoutKvOptional, int64_t batchSize)
{
    // 1. 濡傛灉sequsedKvOptional浼犱簡锛屼娇鐢╯equsedKvOptional鑾峰彇BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedKvOptional 娌′紶锛屽垽鏂?Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 濡傛灉鏄?TND锛屽皾璇曚娇鐢?cuSeqLensOriKvOptional鑾峰彇BatchSize
        ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 濡傛灉涓嶆槸 TND锛屾垨鑰?cuSeqLensOriKvOptional 涓虹┖锛屼娇鐢?batchSize
    delete[] viewDims;
    return batchSize;
}

aclnnStatus CheckConsistencySmla(const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
                                const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional,
                                const aclTensor* sequsedKvOptional, char* layoutQOptional, char* layoutKvOptional,
                                int64_t batchSize, const aclTensor* metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    // 鏍￠獙 cu_seqlens_q
    auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 鏍￠獙 cu_seqlens_q 缁村害
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 鏍￠獙 cu_seqlens_q 鏁版嵁绫诲瀷
        ret = aclGetDataType(cuSeqLensQOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 鏍￠獙 cu_seqlens_ori_kv
    ret = aclGetViewShape(cuSeqLensOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 鏍￠獙 cu_seqlens_ori_kv 缁村害
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_ori_kv must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 鏍￠獙 cu_seqlens_ori_kv 鏁版嵁绫诲瀷
        ret = aclGetDataType(cuSeqLensOriKvOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_ori_kv must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 鏍￠獙 cu_seqlens_cmp_kv
    ret = aclGetViewShape(cuSeqLensCmpKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 鏍￠獙 cu_seqlens_cmp_kv 缁村害
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_cmp_kv must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 鏍￠獙 cu_seqlens_cmp_kv 鏁版嵁绫诲瀷
        ret = aclGetDataType(cuSeqLensCmpKvOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_cmp_kv must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 鏍￠獙 seqused_q
    ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 鏍￠獙 seqused_q 缁村害
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 鏍￠獙 seqused_q 鏁版嵁绫诲瀷
        ret = aclGetDataType(sequsedQOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 鏍￠獙 seqused_kv
    ret = aclGetViewShape(sequsedKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 鏍￠獙 seqused_kv 缁村害
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_kv must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 鏍￠獙 seqused_kv 鏁版嵁绫诲瀷
        ret = aclGetDataType(sequsedKvOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_kv must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 鏍￠獙 metadata
    ret = aclGetViewShape(metadata, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 鏍￠獙 metadata 缁村害
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 鏍￠獙 metadata 鍏冪礌鏁?
        if (viewDims[0] != optiling::SMLA_META_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %ld",
                optiling::SMLA_META_SIZE, viewDims[0]);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 鏍￠獙 metadata 鏁版嵁绫诲瀷
        ret = aclGetDataType(metadata, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of metadata must be int32");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 鏍￠獙 q/kv 缁村害
    int64_t queryBatchSize = GetQueryBatchSizeSmla(sequsedQOptional, cuSeqLensQOptional, layoutQOptional,
                                                  batchSize);
    int64_t kvBatchSize = GetKvBatchSizeSmla(sequsedKvOptional, cuSeqLensOriKvOptional, layoutKvOptional,
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
                               const aclTensor* sequsedKvOptional, int64_t numHeadsQ, int64_t numHeadsKv,
                               int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                               int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode,
                               int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, char* layoutQOptional,
                               char* layoutKvOptional, bool hasOriKv, bool hasCmpKv, const aclTensor* metadata)
{
    (void)headDim;
    (void)maxSeqlenKv;
    (void)oriTopK;
    (void)cmpMaskMode;
    (void)oriWinRight;
    (void)hasOriKv;
    if (CheckSingleParamSmla(batchSize, maxSeqlenQ, numHeadsQ, numHeadsKv, cmpTopK, cmpRatio, oriMaskMode, oriWinLeft,
                            layoutQOptional, layoutKvOptional, hasCmpKv) == ACLNN_SUCCESS &&
        CheckExistenceSmla(layoutQOptional, layoutKvOptional, cuSeqLensQOptional, cuSeqLensOriKvOptional,
                          cuSeqLensCmpKvOptional, sequsedKvOptional, hasCmpKv, metadata) == ACLNN_SUCCESS &&
        CheckConsistencySmla(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                            sequsedKvOptional, layoutQOptional, layoutKvOptional, batchSize,
                            metadata) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

#ifdef __cplusplus
}
#endif
