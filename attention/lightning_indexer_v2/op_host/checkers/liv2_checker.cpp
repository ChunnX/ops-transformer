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

#include "liv2_checker.h"

#include "common_checker_indexer_v2.h"
#include "paged_attention_checker_indexer_v2.h"
#include "seq_len_checker_indexer_v2.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling {
namespace indexer_v2_checker {
namespace {
class LightningIndexerV2TypeChecker : public IndexerV2BaseChecker {
public:
    ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &info) const override
    {
        if (CheckDtype(info, info.query, "q", {ge::DT_FLOAT16, ge::DT_BF16}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.key, "k", {ge::DT_FLOAT16, ge::DT_BF16}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.weights, "w", {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.sparseValues, "sparse_values", {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (info.query.desc == nullptr || info.key.desc == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const ge::DataType qType = info.query.desc->GetDataType();
        const ge::DataType kType = info.key.desc->GetDataType();
        if (qType != kType) {
            const std::string actual = Ops::Base::ToString(qType) + " and " + Ops::Base::ToString(kType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(info.opName, "q and k", actual.c_str(),
                                                   "q and k must have the same dtype");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
};
} // namespace

void LIV2Checker::RegisterCheckers()
{
    checkers_.push_back(std::make_unique<CommonChecker>());
    checkers_.push_back(std::make_unique<SeqLenChecker>());
    checkers_.push_back(std::make_unique<PagedAttentionChecker>());
    checkers_.push_back(std::make_unique<LightningIndexerV2TypeChecker>());
}

ge::graphStatus LIV2Checker::Init(const IndexerV2CheckerInfo &)
{
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2Checker::RunCheck(const CheckMethod &method, const IndexerV2CheckerInfo &info)
{
    for (const auto &checker : checkers_) {
        if (method(checker.get(), info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2Checker::CheckSinglePara(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](const IndexerV2BaseChecker *checker,
                       const IndexerV2CheckerInfo &checkerInfo) { return checker->CheckSinglePara(checkerInfo); },
                    info);
}

ge::graphStatus LIV2Checker::CheckParaExistence(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](const IndexerV2BaseChecker *checker,
                       const IndexerV2CheckerInfo &checkerInfo) { return checker->CheckParaExistence(checkerInfo); },
                    info);
}

ge::graphStatus LIV2Checker::CheckFeature(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](const IndexerV2BaseChecker *checker,
                       const IndexerV2CheckerInfo &checkerInfo) { return checker->CheckFeature(checkerInfo); },
                    info);
}

ge::graphStatus LIV2Checker::CheckMultiPara(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](const IndexerV2BaseChecker *checker,
                       const IndexerV2CheckerInfo &checkerInfo) { return checker->CheckMultiPara(checkerInfo); },
                    info);
}

ge::graphStatus LIV2Checker::Process(const IndexerV2CheckerInfo &info)
{
    if (CheckSinglePara(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckParaExistence(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFeature(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMultiPara(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace indexer_v2_checker
} // namespace optiling
