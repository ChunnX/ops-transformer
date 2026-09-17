/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <string>

#include <tiling/platform/platform_ascendc.h>
#include "log/log.h"
#include "qbsa_op_contract.h"
#include "qbsa_tiling_strategy.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
} // namespace

QbsaTilingStrategy::QbsaTilingStrategy(gert::TilingContext *context)
    : context_(context)
{}

ge::graphStatus QbsaTilingStrategy::Run()
{
    if (context_ == nullptr) {
        OP_LOGE(kOpName, "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    QbsaTilingContextParser parser(context_);
    if (parser.Collect(tilingInfo_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttributes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckTensorMetadata() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseShapeFacts(parser) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeRelations() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (BuildTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return SaveTilingData();
}

uint32_t QbsaTilingStrategy::GetAicCoreNum() const
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return QBSA_MAX_CORE_NUM;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = platform.GetCoreNumAic();
    return aicNum == 0U ? QBSA_MAX_CORE_NUM : std::min(aicNum, QBSA_MAX_CORE_NUM);
}

ge::graphStatus QbsaTilingStrategy::PublishTilingData(const void *data, std::size_t dataSize)
{
    if (usedCoreNum_ == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "usedCoreNum", "0", "Mode strategy must select a core");
        return ge::GRAPH_FAILED;
    }
    auto *rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr) {
        OP_LOGE(kOpName, "Raw tiling data is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (rawTilingData->GetCapacity() < dataSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "tilingDataCapacity",
                                              std::to_string(rawTilingData->GetCapacity()),
                                              "Must be at least " + std::to_string(dataSize));
        return ge::GRAPH_FAILED;
    }

    std::memcpy(rawTilingData->GetData(), data, dataSize);
    rawTilingData->SetDataSize(dataSize);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    PrintTilingData(dataSize);
    return ge::GRAPH_SUCCESS;
}

void QbsaTilingStrategy::PrintTilingData(std::size_t dataSize) const
{
    OP_LOGD(kOpName, "quantMode:%u, B:%u, T:%u, N2:%u, G:%u, D:%u, DV:%u", tilingInfo_.quantModeVal, tilingInfo_.bSize,
            tilingInfo_.qTokenNum, tilingInfo_.n2Size, tilingInfo_.gSize, tilingInfo_.dSize, tilingInfo_.dSizeV);
    OP_LOGD(kOpName, "sparseBlockQ:%u, sparseBlockKV:%u, paBlockSize:%u, paBlockStride:%u", tilingInfo_.qBlockSizeVal,
            tilingInfo_.kvBlockSizeVal, tilingInfo_.paBlockSizeVal, tilingInfo_.paBlockStrideVal);
    OP_LOGD(kOpName, "usedCoreNum:%u, totalTaskNum:%lu, tilingKey:%lu, tilingDataSize:%lu", usedCoreNum_, totalTaskNum_,
            tilingKey_, dataSize);
    OP_LOGD(kOpName, "tilingDataCapacity:%lu", context_->GetRawTilingData()->GetCapacity());
}
} // namespace optiling
