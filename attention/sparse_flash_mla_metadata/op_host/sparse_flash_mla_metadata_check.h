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
 * \file sparse_flash_mla_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../op_kernel_aicpu/sparse_flash_mla_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus CheckSingleParamSmla(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
                                        int64_t maxSeqlenCmpKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t oriTopk,
                                        int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
                                        int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
                                        const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
                                        int64_t batchInvariant, uint32_t aicCoreNum, uint32_t aivCoreNum,
                                        const char *socVersion)
{
    // batch_size >= 0
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should be >= 0, but got %ld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q >= 0
    if (maxSeqlenQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should be >= 0, but got %ld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_ori_kv >= 0
    if (maxSeqlenOriKv < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_ori_kv should be >= 0, but got %ld", maxSeqlenOriKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_cmp_kv >= 0
    if (maxSeqlenCmpKv < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_cmp_kv should be >= 0, but got %ld", maxSeqlenCmpKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q >= 0
    if (numHeadsQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should be >= 0, but got %ld", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv: 1
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %ld", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasOriKv) {
        // ori_topk >= 0
        if (oriTopk < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_topk should be >= 0, but got %ld", oriTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_mask_mode: 0, 3, or 4
        if (oriMaskMode != 0 && oriMaskMode != 3 && oriMaskMode != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_mask_mode should be 0, 3 or 4, but got %ld",
                oriMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_left >= -1
        if (oriWinLeft < -1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_win_left should be >= -1, but got %ld",
                oriWinLeft);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_right >= -1
        if (oriWinRight < -1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_win_right should be >= -1, but got %ld",
                oriWinRight);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasCmpKv) {
        // cmp_topk >= 0
        if (cmpTopk < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_topk should be >= 0, but got %ld", cmpTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_mask_mode: 0 or 3
        if (cmpMaskMode != 0 && cmpMaskMode != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_mask_mode should be 0 or 3, but got %ld",
                cmpMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_ratio: 1~128
        if (cmpRatio < 1 || cmpRatio > 128) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_ratio should be in [1, 128], but got %ld",
                cmpRatio);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // layout_q: BSND or TND
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv: BSND, TND, or PA_BNBD
    if ((strcmp(layoutKvOptional, "BSND") != 0) && (strcmp(layoutKvOptional, "TND") != 0) &&
        (strcmp(layoutKvOptional, "PA_BNBD") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BNBD!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // batch_invariant: 0, 1, or 2
    if (batchInvariant != 0 && batchInvariant != 1 && batchInvariant != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_invariant should be 0, 1 or 2, but got %ld", batchInvariant);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核数校验
    if (aicCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strstr(socVersion, "Ascend950") == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "This operator supports [Ascend950], but now is on %s", socVersion);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckExistenceSmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                      const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedOriKvOptional,
                                      const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
                                      const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional,
                                      int64_t oriTopk, int64_t cmpTopk,int64_t cmpRatio, int64_t oriMaskMode,
                                      int64_t cmpMaskMode, bool hasOriKv, bool hasCmpKv, const char *layoutQOptional,
                                      const char *layoutKvOptional, const aclTensor *metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqlensQOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is TND, cu_seqlens_q must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasOriKv) {
        // cu_seqlens_ori_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            auto ret = aclGetViewShape(cuSeqlensOriKvOptional, &viewDims, &viewDimsNum);
            if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true and layout_kv is TND, "
                    "cu_seqlens_ori_kv must be provided!");
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_ori_kv 存在性校验
        if (strcmp(layoutKvOptional, "PA_BNBD") == 0) {
            auto ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
            if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true and layout_kv is PA_BNBD, "
                    "seqused_ori_kv must be provided!");
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (hasCmpKv) {
        // cu_seqlens_cmp_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            auto ret = aclGetViewShape(cuSeqlensCmpKvOptional, &viewDims, &viewDimsNum);
            if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true and layout_kv is TND, "
                    "cu_seqlens_cmp_kv must be provided!");
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_cmp_kv 存在性校验
        if (strcmp(layoutKvOptional, "PA_BNBD") == 0) {
            auto ret = aclGetViewShape(sequsedCmpKvOptional, &viewDims, &viewDimsNum);
            if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true and layout_kv is PA_BNBD, "
                    "seqused_cmp_kv must be provided!");
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_residual_k 存在性校验
        if (hasCmpKv && cmpRatio != 1 && cmpMaskMode == 3) {
            auto ret = aclGetViewShape(cmpResidualKvOptional, &viewDims, &viewDimsNum);
            if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_ratio is not 1 and mask_mode is 3, "
                    "cmp_residual_k must be provided!");
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // ori_topk_length 存在性校验
    if (hasOriKv && oriTopk != 0 && oriMaskMode == 0) {
        auto ret = aclGetViewShape(oriTopkLengthOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_topk is not 0 and ori_mask_mode is 0, "
                "ori_topk_length must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // cmp_topk_length 存在性校验
    if (hasCmpKv && cmpTopk != 0 && cmpMaskMode == 0) {
        auto ret = aclGetViewShape(cmpTopkLengthOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_topk is not 0 and cmp_mask_mode is 0, "
                "cmp_topk_length must be provided!");
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

static int64_t GetQueryBatchSizeSmla(const aclTensor *sequsedQOptional, const aclTensor *cuSeqlensQOptional,
                                     const char *layoutQOptional, int64_t batchSize)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // 1. 如果 sequsedQOptional 传了，使用 sequsedQOptional 获取 BatchSize
    auto ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedQOptional 没传，判断 Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqLensQOptional 获取 BatchSize
        ret = aclGetViewShape(cuSeqlensQOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqLensQOptional 为空，使用 batchSize
    delete[] viewDims;
    return batchSize;
}

static int64_t GetOriKvBatchSizeSmla(const aclTensor *sequsedOriKvOptional, const aclTensor *cuSeqlensOriKvOptional,
                                     const char *layoutKvOptional, int64_t batchSize)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // 1. 如果 sequsedOriKvOptional 传了，使用 sequsedOriKvOptional 获取 BatchSize
    auto ret = aclGetViewShape(sequsedOriKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedOriKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqlensOriKvOptional 获取 BatchSize
        ret = aclGetViewShape(cuSeqlensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensOriKvOptional 为空，使用 batchSize
    delete[] viewDims;
    return batchSize;
}

static int64_t GetCmpKvBatchSizeSmla(const aclTensor *sequsedCmpKvOptional, const aclTensor *cuSeqlensCmpKvOptional,
                                     const char *layoutKvOptional, int64_t batchSize)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // 1. 如果 sequsedCmpKvOptional 传了，使用 sequsedCmpKvOptional 获取 BatchSize
    auto ret = aclGetViewShape(sequsedCmpKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedCmpKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqlensCmpKvOptional 获取 BatchSize
        ret = aclGetViewShape(cuSeqlensCmpKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensCmpKvOptional 为空，使用 batchSize
    delete[] viewDims;
    return batchSize;
}

static aclnnStatus CheckConsistencySmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                        const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                                        const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                                        const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                                        const aclTensor *cmpTopkLengthOptional, int64_t batchSize,
                                        const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv,
                                        bool hasCmpKv, const aclTensor *metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    // 校验 cu_seqlens_q
    auto ret = aclGetViewShape(cuSeqlensQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 cu_seqlens_q 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        ret = aclGetDataType(cuSeqlensQOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_q must be int32");
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
    // ori_kv部分
    if (hasOriKv) {
        // 校验 cu_seqlens_ori_kv
        ret = aclGetViewShape(cuSeqlensOriKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            // 校验 cu_seqlens_ori_kv 维度
            if (viewDimsNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_ori_kv must be 1, but got %lu", viewDimsNum);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_ori_kv 数据类型
            ret = aclGetDataType(cuSeqlensOriKvOptional, &dataType);
            if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_ori_kv must be int32");
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
        // 校验 ori_topk_length
        ret = aclGetViewShape(oriTopkLengthOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            // 校验 ori_topk_length 维度
            if ((strcmp(layoutQOptional, "TND") == 0)) { // TND 场景
                if (viewDimsNum != 2) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is TND, the dim num of ori_topk_length must be 2, "
                            "but got %ld", viewDimsNum);
                    delete[] viewDims;
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else { // BSND 场景
                if (viewDimsNum != 3) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is BSND, the dim num of ori_topk_length must be 3, "
                            "but got %ld", viewDimsNum);
                    delete[] viewDims;
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 ori_topk_length 数据类型
            ret = aclGetDataType(oriTopkLengthOptional, &dataType);
            if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of ori_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // cmp_kv部分
    if (hasCmpKv) {
        // 校验 cu_seqlens_cmp_kv
        ret = aclGetViewShape(cuSeqlensCmpKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            // 校验 cu_seqlens_cmp_kv 维度
            if (viewDimsNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_cmp_kv must be 1, but got %lu",
                    viewDimsNum);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_cmp_kv 数据类型
            ret = aclGetDataType(cuSeqlensCmpKvOptional, &dataType);
            if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_cmp_kv
        ret = aclGetViewShape(sequsedCmpKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            // 校验 seqused_cmp_kv 维度
            if (viewDimsNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_cmp_kv must be 1, but got %lu", viewDimsNum);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_cmp_kv 数据类型
            ret = aclGetDataType(sequsedCmpKvOptional, &dataType);
            if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_residual_kv
        ret = aclGetViewShape(cmpResidualKvOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            // 校验 cmp_residual_kv 维度
            if (viewDimsNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cmp_residual_kv must be 1, but got %lu", viewDimsNum);
                delete[] viewDims;
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cmp_residual_kv 数据类型
            ret = aclGetDataType(cmpResidualKvOptional, &dataType);
            if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cmp_residual_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_topk_length
        ret = aclGetViewShape(cmpTopkLengthOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            // 校验 cmp_topk_length 维度
            if ((strcmp(layoutQOptional, "TND") == 0)) { // TND 场景
                if (viewDimsNum != 2) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is TND, the dim num of cmp_topk_length must be 2, "
                            "but got %ld", viewDimsNum);
                    delete[] viewDims;
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else { // BSND 场景
                if (viewDimsNum != 3) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is BSND, the dim num of cmp_topk_length must be 3, "
                            "but got %ld", viewDimsNum);
                    delete[] viewDims;
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 cmp_topk_length 数据类型
            ret = aclGetDataType(cmpTopkLengthOptional, &dataType);
            if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cmp_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
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
        if (viewDims[0] != optiling::SMLA_METADATA_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %ld",
                optiling::SMLA_METADATA_SIZE, viewDims[0]);
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
    // 校验 q/kv 维度一致性
    int64_t queryBatchSize = GetQueryBatchSizeSmla(sequsedQOptional, cuSeqlensQOptional, layoutQOptional, batchSize);
    if (hasOriKv) {
        int64_t oriKvBatchSize = GetOriKvBatchSizeSmla(sequsedOriKvOptional, cuSeqlensOriKvOptional, layoutKvOptional,
                                                       batchSize);
        if (queryBatchSize != oriKvBatchSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when has_ori_kv is true, the batch_size obtained from q Tensor should be "
                    "the same as that obtained from ori_kv tensor, but got %ld and %ld", queryBatchSize,
                    oriKvBatchSize);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasCmpKv) {
        int64_t cmpKvBatchSize = GetCmpKvBatchSizeSmla(sequsedCmpKvOptional, cuSeqlensCmpKvOptional, layoutKvOptional,
                                                       batchSize);
        if (queryBatchSize != cmpKvBatchSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when has_cmp_kv is true, the batch_size obtained from q Tensor should be "
                    "the same as that obtained from cmp_kv tensor, but got %ld and %ld", queryBatchSize,
                    cmpKvBatchSize);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    delete[] viewDims;
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
                               int64_t batchInvariant, uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion,
                               const aclTensor *metaData)
{
    if (CheckSingleParamSmla(batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, numHeadsQ, numHeadsKv, oriTopk,
                             cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional,
                             layoutKvOptional, hasOriKv, hasCmpKv, batchInvariant, aicCoreNum, aivCoreNum,
                             socVersion) == ACLNN_SUCCESS &&
        CheckExistenceSmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedOriKvOptional,
                           sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
                           oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, hasOriKv, hasCmpKv, layoutQOptional,
                           layoutKvOptional, metaData) == ACLNN_SUCCESS &&
        CheckConsistencySmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                             sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                             cmpTopkLengthOptional, batchSize, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv,
                             metaData) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

#ifdef __cplusplus
}
#endif
