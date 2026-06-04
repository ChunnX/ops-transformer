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
 * \file quant_block_sparse_attention_tiling.cpp
 * \brief QuantBlockSparseAttention tiling: three-stage pipeline (Parse -> Check -> Tiling).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <tiling/platform/platform_ascendc.h>
#include "../op_kernel/quant_block_sparse_attention_template_tiling_key.h"
#include "quant_block_sparse_attention_tiling.h"
#include "quant_block_sparse_attention_check.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr uint32_t BSA_LAYOUT_Q_BSND_VALUE = 0U;
constexpr uint32_t BSA_LAYOUT_Q_TND_VALUE = 1U;
constexpr uint32_t BSA_LAYOUT_KV_PA_BNSD_VALUE = 1U;
constexpr uint32_t BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE = 0U;
constexpr uint8_t BSA_LAYOUT_TYPE_BSH = 1U;
constexpr uint8_t BSA_LAYOUT_TYPE_TND = 4U;
constexpr uint8_t BSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t BSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t BSA_COMPAT_MASK_LEFT_UP_CAUSAL = 1U;
constexpr uint32_t BSA_COMBINE_ALIGN_BYTES = 32U;
constexpr uint32_t BSA_K_SCALE_BYTES = sizeof(float);

uint32_t GetAicCoreNum(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return BSA_MAX_CORE_NUM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    return aicNum == 0U ? BSA_MAX_CORE_NUM : std::min<uint32_t>(aicNum, BSA_MAX_CORE_NUM);
}
} // namespace

QuantBlockSparseAttentionTiling::QuantBlockSparseAttentionTiling(gert::TilingContext *context) : context_(context)
{
}

void QuantBlockSparseAttentionTiling::FillBaseParams()
{
    const auto &info = *tilingInfo_;
    auto &baseParams = tilingData_.baseParams;
    baseParams.set_bSize(info.bSize);
    baseParams.set_n1Size(info.n1Size);
    baseParams.set_n2Size(info.n2Size);
    baseParams.set_gSize(info.gSize);
    baseParams.set_s1Size(info.s1Size);
    baseParams.set_s2Size(info.s2Size);
    baseParams.set_dSize(BSA_D_SIZE);
    baseParams.set_dSizeV(BSA_D_SIZE);
    baseParams.set_qbMax(info.qbMax);
    baseParams.set_kbMax(info.kbMax);
    baseParams.set_sparseCount(info.sparseCount);
    baseParams.set_qBlockSize(info.qBlockSizeVal);
    baseParams.set_kvBlockSize(info.kvBlockSizeVal);
    baseParams.set_maxBlockNumPerBatch(info.maxBlockNumPerBatch);
    baseParams.set_softmaxScale(info.softmaxScaleVal);
}

void QuantBlockSparseAttentionTiling::FillAttrParams()
{
    const auto &info = *tilingInfo_;
    auto &attrParams = tilingData_.attrParams;
    attrParams.set_layoutQ(info.layoutQValue);
    attrParams.set_layoutKv(BSA_LAYOUT_KV_PA_BNSD_VALUE);
    attrParams.set_layoutSparseIndices(BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE);
    attrParams.set_quantMode(info.quantModeVal);
    attrParams.set_maskMode(info.maskModeVal);
    attrParams.set_returnSoftmaxLse(info.returnSoftmaxLseVal ? 1U : 0U);
}

void QuantBlockSparseAttentionTiling::FillPaParams()
{
    const auto &info = *tilingInfo_;
    auto &paParams = tilingData_.paParams;
    paParams.set_blockSize(info.kvBlockSizeVal);
    paParams.set_blockTableDim2(info.maxBlockNumPerBatch);
    paParams.set_paBlockNumSum(info.paBlockNumSum);
    paParams.set_paLayoutType(BSA_PA_LAYOUT_TYPE_BNBD);
}

void QuantBlockSparseAttentionTiling::FillSparseParams()
{
    const auto &info = *tilingInfo_;
    auto &sparseParams = tilingData_.sparseParams;
    sparseParams.set_gS1OuterSize(info.gS1OuterSize);
    sparseParams.set_sparseSeqLenStride(info.qbMax);
    sparseParams.set_sparseIndicesStride(info.sparseCount);
}

void QuantBlockSparseAttentionTiling::FillInputParams()
{
    const auto &info = *tilingInfo_;
    const uint8_t compatMaskMode = (info.maskModeVal == 3U) ? BSA_COMPAT_MASK_LEFT_UP_CAUSAL : BSA_COMPAT_MASK_NONE;
    auto &inputParams = tilingData_.inputParamsRegbase;
    inputParams.set_bSize(info.bSize);
    inputParams.set_t1Size(info.qTokenNum);
    inputParams.set_t2Size(info.s2Size);
    inputParams.set_n2Size(info.n2Size);
    inputParams.set_gSize(info.gSize);
    inputParams.set_s1Size(info.s1Size);
    inputParams.set_s2Size(info.s2Size);
    inputParams.set_alignedS2(info.kbMax * info.kvBlockSizeVal);
    inputParams.set_dSize(BSA_D_SIZE);
    inputParams.set_dSizeV(BSA_D_SIZE);
    inputParams.set_dSizeRope(0);
    inputParams.set_keepProb(1.0F);
    inputParams.set_scaleValue(info.softmaxScaleVal);
    inputParams.set_preTokens(std::numeric_limits<int32_t>::max());
    inputParams.set_nextTokens(0);
    inputParams.set_pseS1Size(0);
    inputParams.set_pseS2Size(0);
    inputParams.set_pseBSize(0);
    inputParams.set_bandIndex(0);
    inputParams.set_layoutType(info.layoutQValue == BSA_LAYOUT_Q_TND_VALUE ? BSA_LAYOUT_TYPE_TND : BSA_LAYOUT_TYPE_BSH);
    inputParams.set_pseShapeType(0);
    inputParams.set_attenMaskShapeType(2);
    inputParams.set_attenMaskDataType(1);
    inputParams.set_attenMaskCompressMode(compatMaskMode);
    inputParams.set_implMode(0);
    inputParams.set_sparseType(0);
    inputParams.set_needDropMaskOp(0);
    inputParams.set_dropMaskOuter(0);
    inputParams.set_pseEncodeType(0);
    inputParams.set_tndSoftmaxOut(0);
    inputParams.set_remain(0);
    inputParams.set_attenMaskS2Size(info.s2Size);
    inputParams.set_pseType(0);
    inputParams.set_rsv1(0);
    inputParams.set_qStartIdx(0);
    inputParams.set_kvStartIdx(0);
    inputParams.set_s1SparseValidSize(info.s1Size);
    inputParams.set_s2SparseValidSize(info.s2Size);
    inputParams.set_seed(0);
    inputParams.set_offset(0);
    inputParams.set_keepProbUint8(255);
    inputParams.set_pseAlibiBaseS1(0);
    inputParams.set_pseAlibiBaseS2(0);
    inputParams.set_deqScaleFlag(1);
    inputParams.set_deqScale2Flag(1);
    inputParams.set_isActualSeqLengthsNull(1);
    inputParams.set_isActualSeqLengthsKVNull(1);
    inputParams.set_actualSeqLengthsSize(info.bSize);
    inputParams.set_actualSeqLengthsKVSize(info.bSize);
    inputParams.set_isKvContinuous(1);
    inputParams.set_fromFused(0);
    inputParams.set_isBSNDOut(0);
    inputParams.set_transposeLayout(0);
    inputParams.set_isGqa(info.isGqa ? 1 : 0);
    inputParams.set_isSoftMaxLseEnable(info.returnSoftmaxLseVal ? 1 : 0);
    inputParams.set_isActualSharedPrefixLenNull(1);
    inputParams.set_isQHasLeftPadding(0);
    inputParams.set_isKVHasLeftPadding(0);
    inputParams.set_ropeHeadSize(0);
    inputParams.set_prefixSeqInnerSize(0);
    inputParams.set_headNumRatio(1U);
    inputParams.set_blockSize(info.kvBlockSizeVal);
    inputParams.set_blockTableDim2(info.maxBlockNumPerBatch);
    inputParams.set_paBlockNumSum(info.paBlockNumSum);
    inputParams.set_attenMaskS1Size(info.s1Size);
    inputParams.set_paLayoutType(BSA_PA_LAYOUT_TYPE_BNBD);
    inputParams.set_isRowInvalid(0);

void QuantBlockSparseAttentionTiling::FillMultiCoreParams()
{
    const auto &info = *tilingInfo_;
    auto &multiCoreParams = tilingData_.multiCoreParamsRegbase;
    multiCoreParams.set_coreNum(static_cast<int32_t>(usedCoreNum_));
    multiCoreParams.set_totalSize(static_cast<int64_t>(totalTaskNum_));
    multiCoreParams.set_s1OuterSize(info.qbMax);
    multiCoreParams.set_splitFactorSize(0);
    multiCoreParams.set_splitFactorTailSize(0);

    uint32_t bnStartIdx[BSA_CORE_SPLIT_NUM] = {};
    int64_t sparseStartIdx[BSA_CORE_SPLIT_NUM] = {};
    for (uint32_t boundaryIdx = 0U; boundaryIdx <= usedCoreNum_; ++boundaryIdx) {
        const uint64_t taskOffset = totalTaskNum_ * boundaryIdx / usedCoreNum_;
        bnStartIdx[boundaryIdx] = static_cast<uint32_t>(taskOffset / info.gS1OuterSize);
        sparseStartIdx[boundaryIdx] = static_cast<int64_t>(taskOffset % info.gS1OuterSize);
    }
    multiCoreParams.set_bnStartIdx(bnStartIdx);
    multiCoreParams.set_sparseStartIdx(sparseStartIdx);
    multiCoreParams.set_firstFullLoadS1OuterIdx(0);
    multiCoreParams.set_splitCoreMode(0);
    uint8_t reserve[3] = {};
    multiCoreParams.set_reserve(reserve);
}

void QuantBlockSparseAttentionTiling::FillDropmaskParams()
{
    auto &dropmaskParams = tilingData_.dropmaskParamsRegbase;
    dropmaskParams.set_multiCoreFactorSize(0);
    dropmaskParams.set_baseUbCalSize(0);
    dropmaskParams.set_multiCoreTotalSize(0);
    dropmaskParams.set_shapeTotalSize(0);
    dropmaskParams.set_dropMaskAddrOffset(0);
}

void QuantBlockSparseAttentionTiling::FillInitOutputParams()
{
    const auto &info = *tilingInfo_;
    auto &initOutputParams = tilingData_.initOutputParams;
    const int64_t totalOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size * BSA_D_SIZE;
    const int64_t totalSoftmaxLseSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size;
    initOutputParams.set_singleCoreSize(0);
    initOutputParams.set_needInit(0);
    initOutputParams.set_isOneN(info.n1Size == 1U ? 1 : 0);
    uint8_t rsvd[2] = {};
    initOutputParams.set_rsvd(rsvd);
    initOutputParams.set_totalOutputSize(totalOutputSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(totalSoftmaxLseSize);
}

void QuantBlockSparseAttentionTiling::CalcTilingKey()
{
    const auto &info = *tilingInfo_;
    tilingKey_ = GET_TPL_TILING_KEY(BSA_DTYPE_FP8_E4M3FN, info.layoutQValue, BSA_KV_LAYOUT_PA_ND, info.maskModeVal,
                                    info.returnSoftmaxLseVal ? 1U : 0U);
}

void QuantBlockSparseAttentionTiling::CalcWorkspaceSize()
{
    workspaceSize_ = 0;
}

ge::graphStatus QuantBlockSparseAttentionTiling::DoOpTiling(QuantBlockSparseAttentionTilingInfo *tilingInfo)
{
    if (tilingInfo == nullptr || context_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingInfo_ = tilingInfo;

    const uint32_t bnCount = tilingInfo_->bSize * tilingInfo_->n2Size;
    totalTaskNum_ = static_cast<uint64_t>(bnCount) * tilingInfo_->gS1OuterSize;
    if (totalTaskNum_ == 0U || totalTaskNum_ > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t maxAicCoreNum = GetAicCoreNum(context_);
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(maxAicCoreNum), totalTaskNum_));

    FillBaseParams();
    FillAttrParams();
    FillPaParams();
    FillSparseParams();
    FillInputParams();
    FillMultiCoreParams();
    FillDropmaskParams();
    FillInitOutputParams();
    CalcTilingKey();
    CalcWorkspaceSize();

    auto rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingQuantBlockSparseAttention(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttentionTilingInfo tilingInfo;
    QuantBlockSparseAttentionInfoParser parser(context);
    if (parser.Parse(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttentionCheck checker(tilingInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttentionTiling tiling(context);
    return tiling.DoOpTiling(&tilingInfo);
}

} // namespace optiling

namespace ge {
graphStatus TilingPrepareQuantBlockSparseAttention(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class QuantBlockSparseAttentionTiling {
public:
    static ge::graphStatus Tiling(gert::TilingContext *context)
    {
        return optiling::TilingQuantBlockSparseAttention(context);
    }
};
} // namespace ops

namespace optiling {
IMPL_OP_OPTILING(QuantBlockSparseAttention).Tiling(TilingQuantBlockSparseAttention);
} // namespace optiling
