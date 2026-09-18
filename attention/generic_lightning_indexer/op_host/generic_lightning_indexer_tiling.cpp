/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generic_lightning_indexer_tiling.h"
#include <algorithm>
#include <string>
#include "register/op_impl_registry.h"
#include "../op_kernel/generic_lightning_indexer_template_tiling_key.h"

using namespace ge;
namespace optiling {

static uint64_t RoundUp64(uint64_t val, uint64_t align)
{
    return (align == 0) ? val : (val + align - 1) / align * align;
}

static bool IsTopkValid(int64_t topk)
{
    return (topk >= 1 && topk <= 2048) || topk == 3072 || topk == 4096 || topk == 5120 || topk == 6144 ||
           topk == 7168 || topk == 8192;
}

static uint32_t QuantModeToTpl(int32_t hostQuant)
{
    if (hostQuant < 0) {
        return GLI_TPL_QUANT_NONE;
    }
    if (hostQuant == 0 || hostQuant == 1) {
        return GLI_TPL_QUANT_FP8_PTH;
    }
    return static_cast<uint32_t>(hostQuant + 1);
}

ge::graphStatus GenericLightningIndexerTiling::ParseAndCheckParams(GenericLightningIndexerTilingInfo &info)
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *topkPtr = attrs->GetInt(GLI_ATTR_TOPK);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topkPtr);
    OP_CHECK_IF(!IsTopkValid(*topkPtr), OP_LOGE(context_, "topk(%ld) invalid", *topkPtr), return ge::GRAPH_FAILED);
    info.topk = static_cast<uint32_t>(*topkPtr);
    info.sparseCount = info.topk;

    const int64_t *cmpPtr = attrs->GetInt(GLI_ATTR_CMP_RATIO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cmpPtr);
    OP_CHECK_IF(*cmpPtr < 1 || *cmpPtr > GLI_CMP_RATIO_MAX, OP_LOGE(context_, "cmp_ratio(%ld) not in [1,128]", *cmpPtr),
                return ge::GRAPH_FAILED);
    info.cmpRatio = static_cast<uint32_t>(*cmpPtr);

    const char *layoutQPtr = attrs->GetAttrPointer<char>(GLI_ATTR_LAYOUT_Q);
    const char *layoutKPtr = attrs->GetAttrPointer<char>(GLI_ATTR_LAYOUT_K);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutQPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutKPtr);
    const std::string layoutQStr(layoutQPtr);
    const std::string layoutKStr(layoutKPtr);
    if (layoutQStr == "BSND") {
        info.layoutQ = GliDataLayout::BSND;
    } else if (layoutQStr == "TND") {
        info.layoutQ = GliDataLayout::TND;
    } else {
        OP_LOGE(context_, "layout_q must be BSND or TND, got %s", layoutQStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (layoutKStr == "BSND") {
        info.layoutK = GliDataLayout::BSND;
    } else if (layoutKStr == "TND") {
        info.layoutK = GliDataLayout::TND;
    } else if (layoutKStr == "PA_BBND") {
        info.layoutK = GliDataLayout::PA_BBND;
        info.pageAttention = true;
    } else {
        OP_LOGE(context_, "layout_k must be BSND, TND or PA_BBND, got %s", layoutKStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!info.pageAttention && info.layoutQ != info.layoutK) {
        OP_LOGE(context_, "layout_q and layout_k must match unless PA_BBND");
        return ge::GRAPH_FAILED;
    }

    const int64_t *sparsePtr = attrs->GetInt(GLI_ATTR_SPARSE_MODE);
    const int64_t *actPtr = attrs->GetInt(GLI_ATTR_ACT_TYPE);
    const bool *weightedPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_HAS_WEIGHTED_SUM);
    const bool *expandPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_EXPAND_INDICES);
    const bool *appendPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_APPEND_TAIL);
    const int64_t *quantPtr = attrs->GetInt(GLI_ATTR_QUANT_MODE);
    const bool *retValPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_RETURN_VALUE);
    const int64_t *selectPtr = attrs->GetInt(GLI_ATTR_SELECT_MODE);
    const int64_t *keyReducePtr = attrs->GetInt(GLI_ATTR_KEY_REDUCE_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparsePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, actPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightedPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, appendPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, retValPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, selectPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyReducePtr);

    info.sparseMode = static_cast<int32_t>(*sparsePtr);
    info.actType = static_cast<int32_t>(*actPtr);
    info.hasWeightedSum = *weightedPtr;
    info.expandIndices = *expandPtr;
    info.appendTail = *appendPtr;
    info.quantMode = static_cast<int32_t>(*quantPtr);
    info.returnValue = *retValPtr;
    const float *scoreScalePtr = attrs->GetAttrPointer<float>(GLI_ATTR_SCORE_SCALE);
    info.scoreScale = (scoreScalePtr == nullptr) ? 1.0f : *scoreScalePtr;

    OP_CHECK_IF(info.sparseMode != 0 && info.sparseMode != 3 && info.sparseMode != 4,
                OP_LOGE(context_, "sparse_mode must be 0/3/4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.appendTail && !info.expandIndices, OP_LOGE(context_, "append_tail requires expand_indices"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*selectPtr != 0, OP_LOGE(context_, "Step1 rejects select_mode=%ld (MSA reserved)", *selectPtr),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*keyReducePtr != 0,
                OP_LOGE(context_, "Step1 rejects key_reduce_mode=%ld (MSA reserved)", *keyReducePtr),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.actType != 0, OP_LOGE(context_, "Step1 rejects act_type=%d (identity reserved)", info.actType),
                return ge::GRAPH_FAILED);

    info.expandMode = 0;
    if (info.expandIndices) {
        info.expandMode = info.appendTail ? 2 : 1;
    }

    const gert::StorageShape *qShape = context_->GetInputShape(GLI_QUERY_INDEX);
    const gert::StorageShape *kShape = context_->GetInputShape(GLI_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kShape);

    const gert::Tensor *weights = context_->GetOptionalInputTensor(GLI_WEIGHTS_INDEX);
    if (info.hasWeightedSum) {
        OP_CHECK_IF(weights == nullptr, OP_LOGE(context_, "has_weighted_sum=true requires weights"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(weights != nullptr, OP_LOGE(context_, "has_weighted_sum=false forbids weights"),
                    return ge::GRAPH_FAILED);
    }

    auto qDesc = context_->GetInputDesc(GLI_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qDesc);
    info.queryDtype = qDesc->GetDataType();

    if (info.layoutQ == GliDataLayout::BSND) {
        OP_CHECK_IF(qShape->GetStorageShape().GetDimNum() != 4, OP_LOGE(context_, "BSND query must be 4D"),
                    return ge::GRAPH_FAILED);
        info.bSize = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
        info.s1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(1));
        info.n1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(2));
        info.headDim = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(3));
    } else {
        OP_CHECK_IF(qShape->GetStorageShape().GetDimNum() != 3, OP_LOGE(context_, "TND query must be 3D"),
                    return ge::GRAPH_FAILED);
        info.s1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
        info.n1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(1));
        info.headDim = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(2));
        const gert::Tensor *actQ = context_->GetOptionalInputTensor(GLI_ACTUAL_SEQ_Q_INDEX);
        OP_CHECK_IF(actQ == nullptr, OP_LOGE(context_, "TND requires actual_seq_q"), return ge::GRAPH_FAILED);
        info.bSize = static_cast<uint32_t>(actQ->GetStorageShape().GetDim(0));
    }
    OP_CHECK_IF(info.headDim != GLI_HEAD_DIM, OP_LOGE(context_, "headDim must be 128"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.n1Size == 0 || info.n1Size > GLI_N1_LIMIT, OP_LOGE(context_, "N1 must be in [1,64]"),
                return ge::GRAPH_FAILED);
    info.gSize = info.n1Size;

    if (info.pageAttention) {
        OP_CHECK_IF(kShape->GetStorageShape().GetDimNum() != 4, OP_LOGE(context_, "PA key must be 4D"),
                    return ge::GRAPH_FAILED);
        info.blockSize = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(1));
        OP_CHECK_IF(info.blockSize == 0 || (info.blockSize % GLI_BLOCK_SIZE_FACTOR) != 0 ||
                        info.blockSize > GLI_BLOCK_SIZE_LIMIT,
                    OP_LOGE(context_, "PA block_size invalid"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(kShape->GetStorageShape().GetDim(2)) != GLI_N2_FIXED,
                    OP_LOGE(context_, "key N2 must be 1"), return ge::GRAPH_FAILED);
        const gert::StorageShape *btShape = context_->GetOptionalInputShape(GLI_BLOCK_TABLE_INDEX);
        OP_CHECK_IF(btShape == nullptr, OP_LOGE(context_, "PA requires block_table"), return ge::GRAPH_FAILED);
        info.maxBlockNumPerBatch = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(1));
        info.s2Size = static_cast<int64_t>(info.maxBlockNumPerBatch) * info.blockSize;
        const int64_t *ks0 = attrs->GetAttrPointer<int64_t>(GLI_ATTR_KEY_STRIDE0);
        if (ks0 != nullptr && *ks0 >= 0) {
            info.keyStride0 = static_cast<uint32_t>(*ks0);
        } else {
            info.keyStride0 = info.blockSize * GLI_N2_FIXED * info.headDim;
        }
        OP_CHECK_IF(context_->GetOptionalInputTensor(GLI_ACTUAL_SEQ_K_INDEX) == nullptr,
                    OP_LOGE(context_, "PA requires actual_seq_k"), return ge::GRAPH_FAILED);
    } else if (info.layoutK == GliDataLayout::BSND) {
        OP_CHECK_IF(kShape->GetStorageShape().GetDimNum() != 4, OP_LOGE(context_, "BSND key must be 4D"),
                    return ge::GRAPH_FAILED);
        info.s2Size = kShape->GetStorageShape().GetDim(1);
        OP_CHECK_IF(static_cast<uint32_t>(kShape->GetStorageShape().GetDim(2)) != GLI_N2_FIXED,
                    OP_LOGE(context_, "key N2 must be 1"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(kShape->GetStorageShape().GetDimNum() != 3, OP_LOGE(context_, "TND key must be 3D"),
                    return ge::GRAPH_FAILED);
        info.s2Size = kShape->GetStorageShape().GetDim(0);
        OP_CHECK_IF(context_->GetOptionalInputTensor(GLI_ACTUAL_SEQ_K_INDEX) == nullptr,
                    OP_LOGE(context_, "TND requires actual_seq_k"), return ge::GRAPH_FAILED);
    }

    const bool needResidual = (info.cmpRatio > 1 && info.sparseMode != 0);
    const gert::Tensor *residual = context_->GetOptionalInputTensor(GLI_CMP_RESIDUAL_K_INDEX);
    OP_CHECK_IF(needResidual && residual == nullptr,
                OP_LOGE(context_, "cmp_residual_k required when cmp_ratio!=1 and sparse_mode!=0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!needResidual && residual != nullptr,
                OP_LOGE(context_, "cmp_residual_k must be omitted unless cmp_ratio!=1 and sparse_mode!=0"),
                return ge::GRAPH_FAILED);

    const bool needQuantScale = (info.quantMode >= 0);
    const gert::Tensor *qScale = context_->GetOptionalInputTensor(GLI_Q_DESCALE_INDEX);
    const gert::Tensor *kScale = context_->GetOptionalInputTensor(GLI_K_DESCALE_INDEX);
    OP_CHECK_IF(needQuantScale && (qScale == nullptr || kScale == nullptr),
                OP_LOGE(context_, "quant_mode>=0 requires q_descale and k_descale"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenericLightningIndexerTiling::CalcTilingParams(GenericLightningIndexerTilingInfo &info)
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    info.aicNum = ascendcPlatform.GetCoreNumAic();
    info.aivNum = ascendcPlatform.GetCoreNumAiv();
    info.usedCoreNum = std::max(info.aicNum, 1u);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenericLightningIndexerTiling::CalcWorkspaceSize(GenericLightningIndexerTilingInfo &info,
                                                                 uint64_t &workspaceSize)
{
    constexpr uint64_t BASE_TOPK = 2048;
    constexpr uint64_t TOPK_MAX_SIZE = 8192;
    constexpr uint64_t SPARSE_COUNT_8K = 8192;
    constexpr uint64_t S2_BASE_SIZE_ARCH22 = 512;
    constexpr uint64_t BLOCK_CUBE = 16;
    constexpr uint64_t WS_DOUBLE = 2;
    constexpr uint64_t LD_PARAM_NUM = 16;

    uint64_t offset = 0;
    uint32_t aicNum = info.aicNum;
    uint64_t s1BaseSize = (info.sparseCount > BASE_TOPK) ? (SPARSE_COUNT_8K / info.sparseCount) * 2 : 8;
    uint64_t mBaseSizeAlign = RoundUp64(s1BaseSize * info.gSize, BLOCK_CUBE);

    uint64_t scoreGmSize =
        RoundUp64(static_cast<uint64_t>(aicNum) * WS_DOUBLE * mBaseSizeAlign * S2_BASE_SIZE_ARCH22 * GLI_SCORE_T_SIZE,
                  GLI_GM_ALIGN_BYTES);
    uint64_t s1BaseSize35 = (info.sparseCount > BASE_TOPK) ? GLI_S1_BASE_SIZE_SMALL : GLI_S1_BASE_SIZE;
    uint64_t s2Size35Align =
        (static_cast<uint64_t>(std::max(info.s2Size, static_cast<int64_t>(1))) + GLI_S2_BASE_SIZE - 1) /
        GLI_S2_BASE_SIZE * GLI_S2_BASE_SIZE;
    uint64_t scoreGmSize35 =
        RoundUp64(static_cast<uint64_t>(aicNum) * s1BaseSize35 * s2Size35Align * GLI_SCORE_T_SIZE, GLI_GM_ALIGN_BYTES);
    scoreGmSize = std::max(scoreGmSize, scoreGmSize35);
    tilingData_.set_wsOffScore(offset);
    offset += scoreGmSize;

    uint64_t ldScoreSize = RoundUp64(
        static_cast<uint64_t>(aicNum) * s1BaseSize * WS_DOUBLE * WS_DOUBLE * TOPK_MAX_SIZE * GLI_SCORE_T_SIZE, 32);
    tilingData_.set_wsOffLdScore(offset);
    offset += ldScoreSize;

    uint64_t ldIdxSize =
        RoundUp64(static_cast<uint64_t>(aicNum) * s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t), 32);
    tilingData_.set_wsOffLdIdx(offset);
    offset += ldIdxSize;

    workspaceSize = offset;
    return ge::GRAPH_SUCCESS;
}

void GenericLightningIndexerTiling::SetTilingData(const GenericLightningIndexerTilingInfo &info)
{
    tilingData_.set_bSize(info.bSize);
    tilingData_.set_gSize(info.gSize);
    tilingData_.set_s1Size(info.s1Size);
    tilingData_.set_s2Size(info.s2Size);
    tilingData_.set_sparseCount(info.sparseCount);
    tilingData_.set_topk(info.topk);
    tilingData_.set_poolSize(info.cmpRatio);
    tilingData_.set_usedCoreNum(info.usedCoreNum);
    uint32_t s1Base = (info.sparseCount > 2048) ? GLI_S1_BASE_SIZE_SMALL : GLI_S1_BASE_SIZE;
    tilingData_.set_s1BaseSize(s1Base);
    tilingData_.set_mBaseSize(GLI_M_BASE_SIZE);
    tilingData_.set_mBaseSizeMax(GLI_M_BASE_SIZE);
    tilingData_.set_s2BaseSize(GLI_S2_BASE_SIZE);
    tilingData_.set_trunkLen(GLI_TRUNK_LEN_8K);
    tilingData_.set_maskMode(static_cast<uint32_t>(info.sparseMode));
    tilingData_.set_quantMode(QuantModeToTpl(info.quantMode));
    tilingData_.set_returnValue(info.returnValue ? 1 : 0);
    tilingData_.set_layoutQ(static_cast<uint32_t>(info.layoutQ));
    tilingData_.set_layoutK(static_cast<uint32_t>(info.layoutK));
    tilingData_.set_blockSize(info.blockSize);
    tilingData_.set_maxBlockNumPerBatch(info.maxBlockNumPerBatch);
    tilingData_.set_keyStride0(info.keyStride0);
    tilingData_.set_keyDequantScaleStride0(info.keyDequantScaleStride0);
    tilingData_.set_qkScale(info.scoreScale);
    tilingData_.set_expandMode(info.expandMode);
    tilingData_.set_hasWeightedSum(info.hasWeightedSum ? 1 : 0);
    tilingData_.set_actType(static_cast<uint32_t>(info.actType));
}

ge::graphStatus GenericLightningIndexerTiling::DoTiling(GenericLightningIndexerTilingInfo *tilingInfo)
{
    OP_CHECK_IF(context_ == nullptr || tilingInfo == nullptr, OP_LOGE("GenericLightningIndexer", "null context"),
                return ge::GRAPH_FAILED);
    tilingInfo->opName = context_->GetNodeName();
    OP_CHECK_IF(ParseAndCheckParams(*tilingInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseAndCheckParams failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcTilingParams(*tilingInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CalcTilingParams failed"),
                return ge::GRAPH_FAILED);

    uint64_t workspaceSize = 0;
    OP_CHECK_IF(CalcWorkspaceSize(*tilingInfo, workspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CalcWorkspaceSize failed"), return ge::GRAPH_FAILED);

    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize += ascendcPlatform.GetLibApiWorkSpaceSize();

    SetTilingData(*tilingInfo);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_, "workspace sizes is nullptr"), return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(workspaceSize);

    uint32_t tplDtQ = GLI_TPL_FP16;
    if (tilingInfo->queryDtype == ge::DT_BF16) {
        tplDtQ = GLI_TPL_BF16;
    } else if (tilingInfo->queryDtype == ge::DT_FLOAT8_E4M3FN) {
        tplDtQ = GLI_TPL_FP8;
    }
    uint32_t headReduce = tilingInfo->hasWeightedSum ? GLI_TPL_HEAD_WEIGHTED : GLI_TPL_HEAD_SUM;
    uint64_t tilingKey =
        GET_TPL_TILING_KEY(tplDtQ, tplDtQ, static_cast<uint32_t>(GLI_TPL_INT32),
                           static_cast<uint32_t>(tilingInfo->layoutQ), static_cast<uint32_t>(tilingInfo->layoutK),
                           static_cast<uint32_t>(tilingInfo->sparseMode), QuantModeToTpl(tilingInfo->quantMode),
                           static_cast<uint32_t>(tilingInfo->returnValue ? 1 : 0), headReduce, tilingInfo->expandMode);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(tilingInfo->aivNum, tilingInfo->aicNum, tilingInfo->aivNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGenericLightningIndexer(gert::TilingContext *context)
{
    GenericLightningIndexerTiling tiling(context);
    GenericLightningIndexerTilingInfo info;
    return tiling.DoTiling(&info);
}

static ge::graphStatus TilingPrepareForGenericLightningIndexer(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GenericLightningIndexer", "TilingParseContext is nullptr"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platformInfo is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<GenericLightningIndexerCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericLightningIndexer)
    .Tiling(TilingGenericLightningIndexer)
    .TilingParse<GenericLightningIndexerCompileInfo>(TilingPrepareForGenericLightningIndexer);

} // namespace optiling
