/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef LIGHTNING_INDEXER_V2_CHECKER_H
#define LIGHTNING_INDEXER_V2_CHECKER_H

#include <functional>
#include <memory>
#include <vector>
#include "base_checker_indexer_v2.h"

namespace optiling {
namespace indexer_v2_checker {

class LIV2Checker {
public:
    ge::graphStatus Init(const IndexerV2CheckerInfo &info);
    ge::graphStatus Process(const IndexerV2CheckerInfo &info);

    ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &info);
    ge::graphStatus CheckParaExistence(const IndexerV2CheckerInfo &info);
    ge::graphStatus CheckFeature(const IndexerV2CheckerInfo &info);
    ge::graphStatus CheckMultiPara(const IndexerV2CheckerInfo &info);

private:
    using CheckMethod = std::function<ge::graphStatus(IndexerV2BaseChecker *, const IndexerV2CheckerInfo &)>;

    ge::graphStatus RunCheck(const CheckMethod &method, const IndexerV2CheckerInfo &info);
    void RegisterCheckers();
    std::vector<std::unique_ptr<IndexerV2BaseChecker>> checkers_;
};

} // namespace indexer_v2_checker
}
#endif
