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
 * \file quant_block_sparse_attn_tiling.cpp
 * \brief QuantBlockSparseAttn tiling: three-stage pipeline (Parse -> Check -> Tiling).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <tiling/platform/platform_ascendc.h>
#include "../op_kernel/quant_block_sparse_attn_template_tiling_key.h"
#include "quant_block_sparse_attn_tiling.h"
#include "quant_block_sparse_attn_check.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr uint8_t QBSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL = 2U;
constexpr uint32_t QBSA_ATTEN_MASK_DEFAULT_S2_SIZE = 2048U;
constexpr uint32_t QBSA_MXFP8_VALUE_SCALE_LAST_DIM = 2U;

uint32_t GetAicCoreNum(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return QBSA_MAX_CORE_NUM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    return aicNum == 0U ? QBSA_MAX_CORE_NUM : std::min<uint32_t>(aicNum, QBSA_MAX_CORE_NUM);
}

uint32_t CalcMxScaleDSize(uint32_t dSize)
{
    // 返回 MX scale 的 D/64 维度。
    return QBSACeilDiv(dSize, QBSA_MXFP8_SCALE_GROUP_SIZE);
}

QuantBlockSparseAttnMxStrideParams MakePaBnbdStride(uint32_t n2Size, uint32_t blockSize, uint32_t lastDim)
{
    // PA BNBD：[physicalBlock,N,blockSize,lastDim]。
    QuantBlockSparseAttnMxStrideParams stride;
    stride.n2Stride = static_cast<uint64_t>(blockSize) * lastDim;
    stride.bnStride = static_cast<uint64_t>(n2Size) * stride.n2Stride;
    return stride;
}

} // namespace

QuantBlockSparseAttnTiling::QuantBlockSparseAttnTiling(gert::TilingContext *context)
    : context_(context)
{}

void QuantBlockSparseAttnTiling::FillCommonParams()
{
    const auto &info = *tilingInfo_;
    auto &common = tilingData_.commonParams;
    auto &base = common.baseParams;
    base.bSize = info.bSize;
    base.t1Size = info.qTokenNum;
    base.n2Size = info.n2Size;
    base.gSize = info.gSize;
    base.dSize = info.dSize;
    base.dSizeV = info.dSizeV;
    base.coreNum = usedCoreNum_;
    base.scaleValue = info.softmaxScaleVal;
    if (info.opParamInfo.pScale.shape != nullptr) {
        const int64_t size = info.opParamInfo.pScale.shape->GetStorageShape().GetShapeSize();
        base.pScaleShapeSize = size > 0 ? static_cast<uint32_t>(size) : 0U;
    }

    auto &pa = common.paParams;
    pa.blockTableDim2 = info.maxBlockNumPerBatch;
    pa.paBlockNumSum = info.paBlockNumSum;
    pa.paLayoutType = QBSA_PA_LAYOUT_TYPE_BNBD;
    pa.blockSize = info.paBlockSizeVal;
    pa.kvBlockSize = info.kvBlockSizeVal;
    pa.qBlockSize = info.qBlockSizeVal;
    pa.paBlockStride = info.paBlockStrideVal;
    pa.isRowInvalid = 1U;

    common.sparseParams.sparseSeqLenStride = info.qbMax;
    common.sparseParams.sparseIndicesStride = info.sparseCount;
    common.maskParams.preTokens = std::numeric_limits<int32_t>::max();
    common.maskParams.nextTokens = 0;
    common.maskParams.attenMaskS2Size = QBSA_ATTEN_MASK_DEFAULT_S2_SIZE;
    common.maskParams.attenMaskCompressMode =
        info.maskModeVal == QBSA_MASK_MODE_CAUSAL ? QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL : QBSA_COMPAT_MASK_NONE;
    common.outputParams.totalOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size * info.dSizeV;
    common.outputParams.totalSoftMaxLseOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size;
}

void QuantBlockSparseAttnTiling::FillFp8Params()
{
    const auto &info = *tilingInfo_;
    auto &fp8 = tilingData_.fp8Params;
    fp8.layoutType =
        info.layoutQValue == QBSA_LAYOUT_Q_NTD_VALUE ? QBSA_REGBASE_LAYOUT_TYPE_NTD : QBSA_REGBASE_LAYOUT_TYPE_TND;
    fp8.seqUsedQlenSize = info.bSize;
    fp8.seqUsedKvlenSize = info.bSize;
    fp8.isKvContinuous = 1U;
    fp8.isGqa = info.isGqa ? 1U : 0U;
    fp8.s1OuterSize = info.qbMax;
    for (uint32_t boundaryIdx = 0U; boundaryIdx <= usedCoreNum_; ++boundaryIdx) {
        const uint64_t taskOffset = totalTaskNum_ * boundaryIdx / usedCoreNum_;
        fp8.bnStartIdx[boundaryIdx] = static_cast<uint32_t>(taskOffset / info.gS1OuterSize);
    }
    const int64_t aivCoreNum = static_cast<int64_t>(usedCoreNum_) * 2;
    fp8.singleCoreSize =
        static_cast<uint32_t>((tilingData_.commonParams.outputParams.totalOutputSize + aivCoreNum - 1) / aivCoreNum);
    fp8.needInit = 1U;
}

void QuantBlockSparseAttnTiling::FillMxParams()
{
    const auto &info = *tilingInfo_;
    auto &mx = tilingData_.mxParams;
    mx.scaleGroupSize = QBSA_MXFP8_SCALE_GROUP_SIZE;
    mx.scaleLastDim = QBSA_MXFP8_SCALE_LAST_DIM;
    mx.queryScaleDSize = CalcMxScaleDSize(info.dSize);
    mx.keyScaleDSize = CalcMxScaleDSize(info.dSize);
    mx.valueScaleBlockSize = QBSACeilDiv(info.paBlockSizeVal, QBSA_MXFP8_SCALE_GROUP_SIZE);
    mx.valueScaleDSize = info.dSizeV * QBSA_MXFP8_VALUE_SCALE_LAST_DIM;
    mx.keyStrides = MakePaBnbdStride(info.n2Size, info.paBlockSizeVal, info.dSize);
    mx.valueStrides = MakePaBnbdStride(info.n2Size, info.paBlockSizeVal, info.dSizeV);
    mx.kScaleStrides = MakePaBnbdStride(info.n2Size, info.paBlockSizeVal, mx.keyScaleDSize * mx.scaleLastDim);
    mx.vScaleStrides = MakePaBnbdStride(info.n2Size, mx.valueScaleBlockSize, mx.valueScaleDSize);
    mx.pScaleDtype = MX_PSCALE_DTYPE_E8M0;
    if (tilingData_.commonParams.baseParams.pScaleShapeSize > 0U && info.opParamInfo.pScale.desc != nullptr &&
        info.opParamInfo.pScale.desc->GetDataType() == ge::DT_FLOAT) {
        mx.pScaleDtype = MX_PSCALE_DTYPE_FP32;
    }
}

ge::graphStatus QuantBlockSparseAttnTiling::SelectFeature()
{
    const auto &info = *tilingInfo_;
    switch (info.quantModeVal) {
        case QBSA_QUANT_MODE_FP8:
            FillFp8Params();
            tilingKey_ = GET_TPL_TILING_KEY(QBSA_DTYPE_FP8_E4M3FN, info.layoutQValue, QBSA_KV_LAYOUT_PA_BNSD,
                                            info.maskModeVal, info.returnSoftmaxLseVal ? 1U : 0U,
                                            Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128, FP8QuantMode);
            return ge::GRAPH_SUCCESS;
        case QBSA_QUANT_MODE_MXFP8_FULL_QUANT:
            FillMxParams();
            tilingKey_ = GET_TPL_TILING_KEY(QBSA_DTYPE_FP8_E4M3FN, info.layoutQValue, QBSA_KV_LAYOUT_PA_BNSD,
                                            info.maskModeVal, info.returnSoftmaxLseVal ? 1U : 0U,
                                            Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128, MXFullQuantMode);
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "quant_mode", std::to_string(info.quantModeVal),
                                                  "No tiling feature is implemented for this quant_mode");
            return ge::GRAPH_FAILED;
    }
}

void QuantBlockSparseAttnTiling::PrintTilingData()
{
    const auto &info = *tilingInfo_;
    OP_LOGD(kOpName, "quantMode:%u, B:%u, T:%u, N2:%u, G:%u, D:%u, DV:%u", info.quantModeVal, info.bSize,
            info.qTokenNum, info.n2Size, info.gSize, info.dSize, info.dSizeV);
    OP_LOGD(kOpName, "sparseBlockQ:%u, sparseBlockKV:%u, paBlockSize:%u, paBlockStride:%u", info.qBlockSizeVal,
            info.kvBlockSizeVal, info.paBlockSizeVal, info.paBlockStrideVal);
    OP_LOGD(kOpName, "usedCoreNum:%u, totalTaskNum:%lu, tilingKey:%lu, workspaceSize:%lu, tilingDataSize:%lu",
            usedCoreNum_, totalTaskNum_, tilingKey_, workspaceSize_, sizeof(tilingData_));
}

ge::graphStatus QuantBlockSparseAttnTiling::SaveTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr) {
        OP_LOGE(kOpName, "DoOpTiling: raw tiling data is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (rawTilingData->GetCapacity() < sizeof(tilingData_)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "tilingDataCapacity",
                                              std::to_string(rawTilingData->GetCapacity()),
                                              "Must be at least " + std::to_string(sizeof(tilingData_)));
        return ge::GRAPH_FAILED;
    }
    std::memcpy(rawTilingData->GetData(), &tilingData_, sizeof(tilingData_));
    rawTilingData->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnTiling::DoOpTiling(QuantBlockSparseAttnTilingInfo *tilingInfo)
{
    if (tilingInfo == nullptr || context_ == nullptr) {
        OP_LOGE(kOpName, "DoOpTiling: tilingInfo=%p or context=%p is nullptr", tilingInfo, context_);
        return ge::GRAPH_FAILED;
    }
    tilingInfo_ = tilingInfo;
    tilingData_ = {};
    const uint64_t bnCount = static_cast<uint64_t>(tilingInfo_->bSize) * tilingInfo_->n1Size;
    totalTaskNum_ = bnCount * tilingInfo_->gS1OuterSize;
    if (totalTaskNum_ == 0U || totalTaskNum_ > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "bSize * n1Size * gS1OuterSize", std::to_string(totalTaskNum_),
                                              "Total task num must be in range [1, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(GetAicCoreNum(context_), totalTaskNum_));
    FillCommonParams();
    if (SelectFeature() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    workspaceSize_ = 0;
    PrintTilingData();
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    return SaveTilingData();
}

ge::graphStatus TilingQuantBlockSparseAttn(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "Tiling: tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnTilingInfo tilingInfo;
    QuantBlockSparseAttnInfoParser parser(context);
    if (parser.Parse(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnCheck checker(tilingInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnTiling tiling(context);
    return tiling.DoOpTiling(&tilingInfo);
}

} // namespace optiling

namespace ge {
graphStatus TilingPrepareQuantBlockSparseAttn(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantBlockSparseAttn", "TilingPrepare: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class QuantBlockSparseAttnTiling {
public:
    static ge::graphStatus Tiling(gert::TilingContext *context)
    {
        return optiling::TilingQuantBlockSparseAttn(context);
    }
};
} // namespace ops

namespace optiling {
IMPL_OP_OPTILING(QuantBlockSparseAttn).Tiling(TilingQuantBlockSparseAttn);
} // namespace optiling
