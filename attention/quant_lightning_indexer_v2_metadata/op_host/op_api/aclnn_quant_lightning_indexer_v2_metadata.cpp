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
 * \file aclnn_quant_lightning_indexer_v2_metadata.cpp
 * \brief
 */

#include "aclnn_quant_lightning_indexer_v2_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "l0_quant_lightning_indexer_v2_metadata.h"
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

namespace {

int64_t QLI_PER_TOKEN_HEAD_QUANT_MODE = 2;
int64_t QLI_GROUND_SCALING_QUANT_MODE = 3;
int64_t QLI_NO_MASK_MODE = 0;
int64_t QLI_CAUSAL_MASK_MODE = 3;
int64_t QLI_CMP_RATIO_LOWER_BOUND = 1;
int64_t QLI_CMP_RATIO_UPPER_BOUND = 128;

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

int64_t GetQueryBatchSizeQliV2(int64_t batchSizeOptional, const aclTensor *cuSeqlensQOptional,
    const aclTensor *sequsedQOptional, const char *layoutQOptional)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    if (IsTensorExistQliV2(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedQOptional 没传，使用cuSeqlensQOptional获取BatchSize
    if (IsTensorExistQliV2(cuSeqlensQOptional)) {
        return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
    }
    // 如果是 TND，必须使用 cuSeqlensQOptional获取BatchSize
    if (strcmp(layoutQOptional, "TND") == 0) {
        return 0;
    }
    // 3. 如果不是 TND，或者 cuSeqlensQOptional 为空，使用batchSizeOptional
    return batchSizeOptional;
}

int64_t GetKeyBatchSizeQliV2(int64_t batchSizeOptional, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedKOptional, const char *layoutKOptional)
{
    // 1. 如果sequsedKOptional 传了，使用sequsedKOptional获取BatchSize
    if (IsTensorExistQliV2(sequsedKOptional)) {
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedKOptional 没传，使用cuSeqlensKOptional获取BatchSize
    if (IsTensorExistQliV2(cuSeqlensKOptional)) {
        return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
    }
    // 如果是 TND，必须使用 cuSeqlensKOptional获取BatchSize
    if (strcmp(layoutKOptional, "TND") == 0) {
        return 0;
    }
    // 3. 如果不是 TND，或者 cuSeqlensKOptional 为空，使用batchSizeOptional
    return batchSizeOptional;
}

aclnnStatus CheckSingleParamQliV2(int64_t numHeadsQ, int64_t numHeadsK, int64_t topk, int64_t qQuantMode,
    int64_t kQuantMode,
    int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, const char *layoutQOptional,
    const char *layoutKOptional, int64_t maskModeOptional, int64_t cmpRatioOptional,
    uint32_t aicCoreNum, uint32_t aivCoreNum, const std::string &socVersion)
{
    // num_heads_q 校验
    CHECK_COND(numHeadsQ >= 0, ACLNN_ERR_PARAM_INVALID,
        "num_heads_q should not be negative, but got %d", numHeadsQ);
    // num_heads_k 校验
    CHECK_COND(numHeadsK >= 0, ACLNN_ERR_PARAM_INVALID,
        "num_heads_kv should not be negative, but got %d", numHeadsK);
    // topk 校验
    CHECK_COND(topk >= 0, ACLNN_ERR_PARAM_INVALID,
        "topk should not be negative, but got %d", topk);
    // q_quant_mode 非负校验
    CHECK_COND((qQuantMode == QLI_PER_TOKEN_HEAD_QUANT_MODE) || (qQuantMode == QLI_GROUND_SCALING_QUANT_MODE),
        ACLNN_ERR_PARAM_INVALID, "q_quant_mode should be %d/%d, but got %d",
        QLI_PER_TOKEN_HEAD_QUANT_MODE, QLI_GROUND_SCALING_QUANT_MODE, qQuantMode);
    // k_quant_mode 非负校验
    CHECK_COND((kQuantMode == QLI_PER_TOKEN_HEAD_QUANT_MODE) || (kQuantMode == QLI_GROUND_SCALING_QUANT_MODE),
        ACLNN_ERR_PARAM_INVALID, "k_quant_mode should be %d/%d, but got %d",
        QLI_PER_TOKEN_HEAD_QUANT_MODE, QLI_GROUND_SCALING_QUANT_MODE, kQuantMode);
    // batch_size 非负校验
    CHECK_COND(batchSizeOptional >= 0, ACLNN_ERR_PARAM_INVALID,
        "batch_size should not be negative, but got %d", batchSizeOptional);
    // max_seqlen_q 非负校验
    CHECK_COND(maxSeqlenQOptional >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_q should not be negative, but got %d", maxSeqlenQOptional);
    // max_seqlen_k 非负校验
    CHECK_COND(maxSeqlenKOptional >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_k should not be negative, but got %d", maxSeqlenKOptional);
    // mask_mode 校验
    CHECK_COND((maskModeOptional == QLI_NO_MASK_MODE) || (maskModeOptional == QLI_CAUSAL_MASK_MODE),
        ACLNN_ERR_PARAM_INVALID,
        "mask_mode should be %d/%d, but got %d", QLI_NO_MASK_MODE, QLI_CAUSAL_MASK_MODE, maskModeOptional);
    // cmp_ratio 校验
    CHECK_COND((cmpRatioOptional >= QLI_CMP_RATIO_LOWER_BOUND) && (cmpRatioOptional <= QLI_CMP_RATIO_UPPER_BOUND),
        ACLNN_ERR_PARAM_INVALID,
        "cmp_ratio should be between [%d, %d], but got %d",
        QLI_CMP_RATIO_LOWER_BOUND, QLI_CMP_RATIO_UPPER_BOUND, cmpRatioOptional);
    // layout_q 校验
    CHECK_COND((strcmp(layoutQOptional, "TND") == 0) || (strcmp(layoutQOptional, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
        "layout_q must be TND or BSND, but got %s", layoutQOptional);
    // layout_k 校验
    CHECK_COND((strcmp(layoutKOptional, "TND") == 0) || (strcmp(layoutKOptional, "BSND") == 0) || \
        (strcmp(layoutKOptional, "PA_BBND") == 0),
        ACLNN_ERR_PARAM_INVALID, "layout_k must be TND/BSND/PA_BBND, but got %s", layoutKOptional);
    if ((strcmp(layoutKOptional, "PA_BBND") != 0)) {
        CHECK_COND((strcmp(layoutQOptional, layoutKOptional) == 0), ACLNN_ERR_PARAM_INVALID,
            "For layout_k != PA_BBND, layout_q and layout_k must be the same!");
    }

    // 核心数校验
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID,
        "AIC num should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_PARAM_INVALID,
        "AIV num should be larger than 0, but got %u", aivCoreNum);
    CHECK_COND(socVersion.find("Ascend950") != std::string::npos, ACLNN_ERR_PARAM_INVALID,
        "This operator supports [Ascend950], but now is on %s", socVersion.c_str());
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceQliV2(int64_t maskModeOptional, int64_t cmpRatioOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *cmpResidualKOptional,
    const char *layoutQOptional, const char *layoutKOptional)
{
    // Query 存在性逻辑
    if (strcmp(layoutQOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistQliV2(cuSeqlensQOptional), ACLNN_ERR_PARAM_INVALID,
            "For layout_q TND, cu_seqlens_q must be provided!");
    }
    // Key 存在性逻辑
    if (strcmp(layoutKOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistQliV2(cuSeqlensKOptional), ACLNN_ERR_PARAM_INVALID,
            "For layout_k TND, cu_seqlens_k must be provided!");
    }
    // cmp_residual_k 存在性逻辑
    if (cmpRatioOptional != QLI_CMP_RATIO_LOWER_BOUND && maskModeOptional == QLI_CAUSAL_MASK_MODE) {
        CHECK_COND(IsTensorExistQliV2(cmpResidualKOptional), ACLNN_ERR_PARAM_INVALID,
            "When cmp_ratio is not 1 and mask_mode is CAUSAL, cmp_residual_k must be provided!");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyQliV2(int64_t batchSizeOptional, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, const char *layoutQOptional, const char *layoutKOptional)
{
int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    int64_t batchSize = GetQueryBatchSizeQliV2(batchSizeOptional, cuSeqlensQOptional, sequsedQOptional,
        layoutQOptional);
    // 校验 cu_seqlens_q
    if (IsTensorExistQliV2(cuSeqlensQOptional)) {
        dimNum = GetDimNumQliV2(cuSeqlensQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %ld", dimNum);
        dataType = GetDataTypeQliV2(cuSeqlensQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistQliV2(cuSeqlensKOptional)) {
        dimNum = GetDimNumQliV2(cuSeqlensKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_k must be 1, but got %ld", dimNum);
        dataType = GetDataTypeQliV2(cuSeqlensKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_q
    if (IsTensorExistQliV2(sequsedQOptional)) {
        dimNum = GetDimNumQliV2(sequsedQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %ld", dimNum);
        dataType = GetDataTypeQliV2(sequsedQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_k
    if (IsTensorExistQliV2(sequsedKOptional)) {
        dimNum = GetDimNumQliV2(sequsedKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_k must be 1, but got %ld", dimNum);
        dataType = GetDataTypeQliV2(sequsedKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cmp_residual_k
    if (IsTensorExistQliV2(cmpResidualKOptional)) {
        dimNum = GetDimNumQliV2(cmpResidualKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID,
            "The dim num of cmp_residual_k must be 1, but got %ld", dimNum);
        dataType = GetDataTypeQliV2(cmpResidualKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cmp_residual_k must be int32, but got %d", static_cast<int32_t>(dataType));
        // 校验batch
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        CHECK_COND(cmpResidualKBatch == batchSize, ACLNN_ERR_PARAM_INVALID,
            "The batch_size of cmp_residual_k should match the one of q tensor, but got %ld and %ld",
            cmpResidualKBatch, batchSize);
    }

    // 校验batch
    int64_t keyBatchSize = GetKeyBatchSizeQliV2(batchSizeOptional, cuSeqlensKOptional, sequsedKOptional,
        layoutKOptional);
    CHECK_COND(batchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
        "The batch_size obtained from q Tensor should be the same as that obtained from k tensor, but got %ld and %ld",
        batchSize, keyBatchSize);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFeatureQliV2()
{
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckQliV2(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t topk, int64_t qQuantMode, int64_t kQuantMode,
    int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, const char *layoutQOptional,
    const char *layoutKOptional, int64_t maskModeOptional, int64_t cmpRatioOptional,
    uint32_t aicCoreNum, uint32_t aivCoreNum, const std::string &socVersion)
{
    auto ret = CheckSingleParamQliV2(numHeadsQ, numHeadsK, topk, qQuantMode, kQuantMode,
        batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQOptional, layoutKOptional, maskModeOptional,
        cmpRatioOptional,
        aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceQliV2(maskModeOptional, cmpRatioOptional,
        cuSeqlensQOptional, cuSeqlensKOptional, cmpResidualKOptional, layoutQOptional, layoutKOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyQliV2(batchSizeOptional, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
        sequsedKOptional, cmpResidualKOptional, layoutQOptional, layoutKOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckFeatureQliV2();
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t qQuantMode, int64_t kQuantMode,
    int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, char *layoutQOptional,
    char *layoutKOptional, int64_t maskModeOptional, int64_t cmpRatioOptional,
    const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantLightningIndexerV2Metadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                          cmpResidualKOptional,
                          numHeadsQ, numHeadsK, headDim, topk, qQuantMode, kQuantMode,
                          batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQOptional, layoutKOptional,
                          maskModeOptional, cmpRatioOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const std::string socVersion = npuInfo.GetSocLongVersion();

    auto ret = ParamsCheckQliV2(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                                cmpResidualKOptional,
                                numHeadsQ, numHeadsK, topk, qQuantMode, kQuantMode, batchSizeOptional,
                                maxSeqlenQOptional, maxSeqlenKOptional, layoutQOptional, layoutKOptional,
                                maskModeOptional, cmpRatioOptional,
                                aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *cuSeqlensQOptionalContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQOptionalContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqlensQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cuSeqlensKOptionalContiguous = nullptr;
    if (cuSeqlensKOptional != nullptr) {
        cuSeqlensKOptionalContiguous = l0op::Contiguous(cuSeqlensKOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqlensKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedQOptionalContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        CHECK_RET(sequsedQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedKOptionalContiguous = nullptr;
    if (sequsedKOptional != nullptr) {
        sequsedKOptionalContiguous = l0op::Contiguous(sequsedKOptional, uniqueExecutor.get());
        CHECK_RET(sequsedKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cmpResidualLKOptionalContiguous = nullptr;
    if (cmpResidualKOptional != nullptr) {
        cmpResidualLKOptionalContiguous = l0op::Contiguous(cmpResidualKOptional, uniqueExecutor.get());
        CHECK_RET(cmpResidualLKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto output = l0op::QuantLightningIndexerV2Metadata(
        cuSeqlensQOptionalContiguous, cuSeqlensKOptionalContiguous, sequsedQOptionalContiguous,
        sequsedKOptionalContiguous, cmpResidualLKOptionalContiguous,
        numHeadsQ, numHeadsK, headDim, topk, qQuantMode, kQuantMode,
        batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQOptional, layoutKOptional, maskModeOptional,
        cmpRatioOptional,
        aicCoreNum, aivCoreNum, socVersion.c_str(), metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantLightningIndexerV2Metadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantLightningIndexerV2Metadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
