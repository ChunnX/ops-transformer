/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN
 * Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not
 * use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#include "base_checker_indexer_v2.h"

#include <algorithm>
#include <sstream>
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling {
namespace indexer_v2_checker {
namespace {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr uint32_t BSND_RANK = 4U;
constexpr uint32_t TND_RANK = 3U;
constexpr uint32_t BLOCK_TABLE_RANK = 2U;

bool HasExpectedRank(CheckerLayout layout, uint32_t rank)
{
    return layout == CheckerLayout::TND ? rank >= TND_RANK : rank >= BSND_RANK;
}
} // namespace

ge::graphStatus IndexerV2BaseChecker::CheckParaExistence(const IndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerV2BaseChecker::CheckSinglePara(const IndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerV2BaseChecker::CheckFeature(const IndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerV2BaseChecker::CheckMultiPara(const IndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

std::string IndexerV2BaseChecker::DtypesToString(const std::vector<ge::DataType> &dtypes)
{
    std::string result;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += Ops::Base::ToString(dtypes[i]);
    }
    return result;
}

std::string IndexerV2BaseChecker::ShapeToString(const CheckerTensor &tensor)
{
    const gert::Shape *shape = tensor.GetShape();
    return shape == nullptr ? "null" : Ops::Base::ToString(*shape);
}

ge::graphStatus IndexerV2BaseChecker::CheckDtype(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                                 const char *name, const std::vector<ge::DataType> &expected) const
{
    if (!tensor.IsPresent() || tensor.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const ge::DataType actualDtype = tensor.desc->GetDataType();
    if (std::find(expected.begin(), expected.end(), actualDtype) == expected.end()) {
        const std::string actual = Ops::Base::ToString(actualDtype);
        const std::string expectedText = DtypesToString(expected);
        OP_LOGE_FOR_INVALID_DTYPE(info.opName, name, actual.c_str(), expectedText.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerV2BaseChecker::CheckRank(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                                const char *name, uint32_t expected) const
{
    const gert::Shape *shape = tensor.GetShape();
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const uint32_t actual = shape->GetDimNum();
    if (actual != expected) {
        const std::string actualText = std::to_string(actual) + "D";
        const std::string expectedText = std::to_string(expected) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(info.opName, name, actualText.c_str(), expectedText.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

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

void FinalizeCommonInfo(IndexerV2CheckerInfo &info)
{
    info.layoutQ = ParseLayout(info.layoutQText);
    info.layoutK = ParseLayout(info.layoutKText);
    const gert::Shape *qShape = info.query.GetShape();
    const gert::Shape *kShape = info.key.GetShape();
    if (qShape == nullptr || kShape == nullptr || info.layoutQ == CheckerLayout::INVALID ||
        info.layoutK == CheckerLayout::INVALID) {
        return;
    }
    if (HasExpectedRank(info.layoutQ, qShape->GetDimNum())) {
        info.batch = info.layoutQ == CheckerLayout::BSND ? qShape->GetDim(DIM_0) : 0;
        info.qSeq = qShape->GetDim(info.layoutQ == CheckerLayout::BSND ? DIM_1 : DIM_0);
        info.qHeads = qShape->GetDim(info.layoutQ == CheckerLayout::BSND ? DIM_2 : DIM_1);
        info.headDim = qShape->GetDim(info.layoutQ == CheckerLayout::BSND ? DIM_3 : DIM_2);
    }
    if (HasExpectedRank(info.layoutK, kShape->GetDimNum())) {
        info.kSeq = kShape->GetDim(info.layoutK == CheckerLayout::TND ? DIM_0 : DIM_1);
        info.kHeads = kShape->GetDim(info.layoutK == CheckerLayout::TND ? DIM_1 : DIM_2);
        if (info.layoutK == CheckerLayout::PA_BBND) {
            info.blockSize = kShape->GetDim(DIM_1);
            const gert::Shape *blockTableShape = info.blockTable.GetShape();
            if (blockTableShape != nullptr && blockTableShape->GetDimNum() >= BLOCK_TABLE_RANK) {
                info.kSeq = blockTableShape->GetDim(DIM_1) * info.blockSize;
            }
        }
    }
    if (info.layoutQ == CheckerLayout::TND && info.cuSeqlensQ.tensor != nullptr) {
        info.batch = info.cuSeqlensQ.tensor->GetShapeSize() - 1;
    }
}

} // namespace indexer_v2_checker
} // namespace optiling
