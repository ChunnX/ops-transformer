/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_tiling_general_regbase.cpp
 * \brief Arch35 tiling implementation
 */

#include "dense_lightning_indexer_softmax_lse_tiling_general_regbase.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace optiling;

namespace optiling {

ge::graphStatus DenseLightningIndexerSoftmaxLseTilingBaseRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo<platform_ascendc::PlatformAscendC>();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    socVersion = platformInfo->GetSocVersion();
    aicNum = platformInfo->GetCoreNumByType(platform_ascendc::CoreType::AIC);
    aivNum = platformInfo->GetCoreNumByType(platform_ascendc::CoreType::AIV);
    l2CacheSize = platformInfo->GetL2CacheSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerSoftmaxLseTilingBaseRegbase::GetShapeAttrsInfo()
{
    const gert::StorageShape *queryShape = context_->GetInputShape(0);
    const gert::StorageShape *keyShape = context_->GetInputShape(1);
    const gert::StorageShape *weightsShape = context_->GetInputShape(2);

    if (queryShape == nullptr || keyShape == nullptr || weightsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto queryDims = queryShape->GetStorageShape().GetDims();
    auto keyDims = keyShape->GetStorageShape().GetDims();
    auto weightsDims = weightsShape->GetStorageShape().GetDims();

    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    inputLayout = attrs->GetAttrPointer<char>(0);
    sparseMode = attrs->GetAttrsIntValue(1, 0);
    preTokens = attrs->GetAttrsIntValue(2, 0);
    nextTokens = attrs->GetAttrsIntValue(3, 0);

    if (!AnalyzeLayout()) {
        return ge::GRAPH_FAILED;
    }

    if constexpr (layoutType == static_cast<uint8_t>(LayoutTypeRegbase::BSND)) {
        bSize = queryDims[0];
        s1Size = queryDims[1];
        gSize = queryDims[2];
        dSize = queryDims[3];
        s2Size = keyDims[1];
    } else {
        bSize = queryDims[0] == 0 ? 0 : 
            (actualSeqLenData.empty() ? queryDims[0] : actualSeqLenData.back());
        gSize = queryDims[1];
        dSize = queryDims[2];
        s1Size = bSize;
        s2Size = keyDims[0] == 0 ? 0 :
            (actualSeqLenKData.empty() ? keyDims[0] : actualSeqLenKData.back());
    }

    n2Size = 1;
    kSize = s2Size;

    return ge::GRAPH_SUCCESS;
}

bool DenseLightningIndexerSoftmaxLseTilingBaseRegbase::AnalyzeLayout()
{
    if (inputLayout == nullptr) {
        return false;
    }
    std::string layoutStr(inputLayout);
    if (layoutStr == "BSND" || layoutStr == "BNSD") {
        layoutType = static_cast<uint8_t>(LayoutTypeRegbase::BSND);
    } else if (layoutStr == "TND") {
        layoutType = static_cast<uint8_t>(LayoutTypeRegbase::TND);
    } else {
        return false;
    }
    return true;
}

ge::graphStatus DenseLightningIndexerSoftmaxLseTilingBaseRegbase::DoOpTiling()
{
    int64_t totalSize = CalcTotalSize();
    int64_t usedCoreNum = std::min(static_cast<int64_t>(aicNum), totalSize);
    usedCoreNum = std::max(usedCoreNum, 1);

    SetMultiCoreParamsRegbase(totalSize, usedCoreNum);

    baseParams_->set_bSize(bSize);
    baseParams_->set_n2Size(n2Size);
    baseParams_->set_gSize(gSize);
    baseParams_->set_s1Size(s1Size);
    baseParams_->set_s2Size(s2Size);
    baseParams_->set_dSize(dSize);
    baseParams_->set_kSize(kSize);
    baseParams_->set_s1BaseSize(s1BaseSize);
    baseParams_->set_s2BaseSize(s2BaseSize);
    baseParams_->set_mBaseSize(mBaseSize);
    baseParams_->set_sparseMode(sparseMode);
    baseParams_->set_preTokens(preTokens);
    baseParams_->set_nextTokens(nextTokens);
    baseParams_->set_scaleValue(scaleValue);
    baseParams_->set_layoutType(layoutType);

    return ge::GRAPH_SUCCESS;
}

int64_t DenseLightningIndexerSoftmaxLseTilingBaseRegbase::CalcTotalSize()
{
    int64_t totalSize = 0;
    if constexpr (layoutType == static_cast<uint8_t>(LayoutTypeRegbase::TND)) {
        if (actualSeqLenData.empty()) {
            totalSize = bSize * s1Size;
        } else {
            for (auto len : actualSeqLenData) {
                totalSize += len;
            }
        }
    } else {
        totalSize = bSize * s1Size;
    }
    return totalSize;
}

void DenseLightningIndexerSoftmaxLseTilingBaseRegbase::SetMultiCoreParamsRegbase(
    int64_t totalSize, int64_t coreNum)
{
    multiCoreParams_->set_coreNum(static_cast<uint32_t>(aicNum));
    multiCoreParams_->set_usedCoreNum(static_cast<uint32_t>(coreNum));
    multiCoreParams_->set_totalSize(totalSize);
    multiCoreParams_->set_splitFactorSize(s1BaseSize);

    int64_t avgSize = DenseLISoftmaxLseCeilDiv<int64_t>(totalSize, coreNum);
    int64_t remainSize = totalSize % coreNum;

    int64_t offset = 0;
    for (int64_t i = 0; i < coreNum && i < MAX_CORE_NUM_REGBASE; i++) {
        int64_t curSize = avgSize;
        if (i < remainSize) {
            curSize = avgSize + 1;
        }
        multiCoreParams_->bS1Index[i] = offset;
        offset += curSize;
    }
    if (coreNum < MAX_CORE_NUM_REGBASE) {
        multiCoreParams_->bS1Index[coreNum] = totalSize;
    }
}

ge::graphStatus DenseLightningIndexerSoftmaxLseTilingBaseRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerSoftmaxLseTilingBaseRegbase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t DenseLightningIndexerSoftmaxLseTilingBaseRegbase::GetTilingKey() const
{
    return GetDenseLISoftmaxLseTilingKeyRegbase(static_cast<LayoutTypeRegbase>(layoutType));
}

ge::graphStatus DenseLightningIndexerSoftmaxLseTilingBaseRegbase::GetWorkspaceSize()
{
    int64_t mm1ResSize = mBaseSize * s2BaseSize * sizeof(float) * 2;
    int64_t vec1ResSize = s1BaseSize * DenseLISoftmaxLseAlign<uint64_t>(
        s2Size > 128 * 1024 ? 128 * 1024 : s2Size, s2BaseSize) * sizeof(float);
    int64_t totalWorkspaceSize = aicNum * (mm1ResSize + vec1ResSize) + 32 * 1024;

    context_->SetWorkspaceSize(totalWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

void DenseLightningIndexerSoftmaxLseTilingBaseRegbase::GetActualSeqLenData(
    int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const
{
    const gert::Tensor *actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        actualLen = 0;
        return;
    }
    auto data = actualSeqLenTensor->GetValue<int64_t>();
    auto size = actualSeqLenTensor->GetShape()->GetStorageShape().GetStorageShape().GetDimNum();
    actualLen = size;
    for (int64_t i = 0; i < size; i++) {
        res.push_back(data[i]);
    }
}

} // namespace optiling

#ifdef ASCENDC_INCB
namespace optiling {

static ge::graphStatus DenseLightningIndexerSoftmaxLseTilingFuncRegbase(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("DenseLightningIndexerSoftmaxLse", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    DenseLightningIndexerSoftmaxLseTilingBaseRegbase tilingObj(context);
    if (!tilingObj.IsCapable()) {
        return ge::GRAPH_FAILED;
    }
    if (tilingObj.GetPlatformInfo() != ge::GRAPH_SUCCESS ||
        tilingObj.GetShapeAttrsInfo() != ge::GRAPH_SUCCESS ||
        tilingObj.DoOpTiling() != ge::GRAPH_SUCCESS ||
        tilingObj.DoLibApiTiling() != ge::GRAPH_SUCCESS ||
        tilingObj.PostTiling() != ge::GRAPH_SUCCESS ||
        tilingObj.GetWorkspaceSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context_->SetTilingKey(tilingObj.GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDenseLightningIndexerSoftmaxLseRegbase(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DenseLightningIndexerSoftmaxLse)
    .Tiling(DenseLightningIndexerSoftmaxLseTilingFuncRegbase)
    .TilingParse<DenseLightningIndexerSoftmaxLseCompileInfo>(TilingParseForDenseLightningIndexerSoftmaxLseRegbase);

} // namespace optiling
#endif