/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "qliv2_checker.h"

#include "quant_checker_qliv2.h"
#include "../../../lightning_indexer_v2/op_host/checkers/common_checker_indexer_v2.h"
#include "../../../lightning_indexer_v2/op_host/checkers/paged_attention_checker_indexer_v2.h"
#include "../../../lightning_indexer_v2/op_host/checkers/seq_len_checker_indexer_v2.h"

namespace optiling {
namespace indexer_v2_checker {

void QLIV2Checker::RegisterCheckers()
{
    checkers_.push_back(std::make_unique<CommonChecker>());
    checkers_.push_back(std::make_unique<SeqLenChecker>());
    checkers_.push_back(std::make_unique<PagedAttentionChecker>());
    checkers_.push_back(std::make_unique<QuantChecker>());
}

ge::graphStatus QLIV2Checker::Init(const IndexerV2CheckerInfo &)
{
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2Checker::RunCheck(const CheckMethod &method, const IndexerV2CheckerInfo &info)
{
    for (const auto &checker : checkers_) {
        if (method(checker.get(), info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2Checker::CheckSinglePara(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](IndexerV2BaseChecker *checker, const IndexerV2CheckerInfo &checkerInfo) {
        return checker->CheckSinglePara(checkerInfo);
    }, info);
}

ge::graphStatus QLIV2Checker::CheckParaExistence(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](IndexerV2BaseChecker *checker, const IndexerV2CheckerInfo &checkerInfo) {
        return checker->CheckParaExistence(checkerInfo);
    }, info);
}

ge::graphStatus QLIV2Checker::CheckFeature(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](IndexerV2BaseChecker *checker, const IndexerV2CheckerInfo &checkerInfo) {
        return checker->CheckFeature(checkerInfo);
    }, info);
}

ge::graphStatus QLIV2Checker::CheckMultiPara(const IndexerV2CheckerInfo &info)
{
    return RunCheck([](IndexerV2BaseChecker *checker, const IndexerV2CheckerInfo &checkerInfo) {
        return checker->CheckMultiPara(checkerInfo);
    }, info);
}

ge::graphStatus QLIV2Checker::Process(const IndexerV2CheckerInfo &info)
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
