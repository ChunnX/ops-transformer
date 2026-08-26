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
 * \file checker_adapter.cpp
 * \brief Implements context adaptation for Lightning Indexer V2 checkers.
 */

#include "checker_adapter.h"

#include <limits>

namespace optiling {
namespace indexer_v2_checker {
namespace {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

CheckerLayout ParseLayout(const std::string &layout)
{
    if (layout == "BSND") {
        return CheckerLayout::BSND;
    }
    if (layout == "TND") {
        return CheckerLayout::TND;
    }
    if (layout == "PA_BBND") {
        return CheckerLayout::PA_BBND;
    }
    return CheckerLayout::INVALID;
}
} // namespace

CheckerTensor MakeRequiredTensor(const gert::TilingContext *context, uint32_t index)
{
    if (context == nullptr) {
        return {};
    }
    return {context->GetInputDesc(index), context->GetInputShape(index), nullptr,
            context->GetDynamicInputStride(index, 0)};
}

CheckerTensor MakeOptionalTensor(const gert::TilingContext *context, uint32_t index)
{
    if (context == nullptr) {
        return {};
    }
    return {context->GetOptionalInputDesc(index), context->GetOptionalInputShape(index),
            context->GetOptionalInputTensor(index), context->GetDynamicInputStride(index, 0)};
}

CheckerTensor MakeOutputTensor(const gert::TilingContext *context, uint32_t index)
{
    if (context == nullptr) {
        return {};
    }
    return {context->GetOutputDesc(index), context->GetOutputShape(index), nullptr, nullptr};
}

void PopulateDerivedInfo(IndexerV2CheckerInfo &info)
{
    info.layoutQ = ParseLayout(info.layoutQText);
    info.layoutK = ParseLayout(info.layoutKText);
    const gert::Shape *qShape = info.query.GetShape();
    const gert::Shape *kShape = info.key.GetShape();
    if (qShape == nullptr || kShape == nullptr || info.layoutQ == CheckerLayout::INVALID ||
        info.layoutK == CheckerLayout::INVALID) {
        return;
    }
    const uint32_t qRank = qShape->GetDimNum();
    const uint32_t kRank = kShape->GetDimNum();
    if ((info.layoutQ == CheckerLayout::BSND && qRank >= 4U) || (info.layoutQ == CheckerLayout::TND && qRank >= 3U)) {
        info.batch = info.layoutQ == CheckerLayout::BSND ? qShape->GetDim(DIM_0) : 0;
        info.qSeq = qShape->GetDim(info.layoutQ == CheckerLayout::BSND ? DIM_1 : DIM_0);
        info.qHeads = qShape->GetDim(info.layoutQ == CheckerLayout::BSND ? DIM_2 : DIM_1);
        info.headDim = qShape->GetDim(info.layoutQ == CheckerLayout::BSND ? DIM_3 : DIM_2);
    }
    if ((info.layoutK != CheckerLayout::TND && kRank >= 4U) || (info.layoutK == CheckerLayout::TND && kRank >= 3U)) {
        info.kSeq = kShape->GetDim(info.layoutK == CheckerLayout::TND ? DIM_0 : DIM_1);
        info.kHeads = kShape->GetDim(info.layoutK == CheckerLayout::TND ? DIM_1 : DIM_2);
        if (info.layoutK == CheckerLayout::PA_BBND) {
            info.blockSize = kShape->GetDim(DIM_1);
            const gert::Shape *blockTableShape = info.blockTable.GetShape();
            if (blockTableShape != nullptr && blockTableShape->GetDimNum() >= 2U) {
                const int64_t blockCount = blockTableShape->GetDim(DIM_1);
                if (blockCount >= 0 && info.blockSize > 0 &&
                    blockCount <= std::numeric_limits<int64_t>::max() / info.blockSize) {
                    info.kSeq = blockCount * info.blockSize;
                } else {
                    info.kSeq = -1;
                }
            }
        }
    }
    if (info.layoutQ == CheckerLayout::TND && info.cuSeqlensQ.IsPresent()) {
        const int64_t cuSeqlensQSize = info.cuSeqlensQ.GetShape()->GetShapeSize();
        info.batch = cuSeqlensQSize > 0 ? cuSeqlensQSize - 1 : -1;
    }
}

} // namespace indexer_v2_checker
} // namespace optiling
