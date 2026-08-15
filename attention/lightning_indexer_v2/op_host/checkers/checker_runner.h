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
 * \file checker_runner.h
 * \brief Composite checker runner shared by LightningIndexerV2 and QuantLightningIndexerV2
 */

#ifndef LIGHTNING_INDEXER_V2_CHECKER_RUNNER_H
#define LIGHTNING_INDEXER_V2_CHECKER_RUNNER_H

#include <functional>
#include <memory>
#include <vector>
#include "base_checker.h"

namespace optiling {
namespace lightning_indexer_v2 {

class LICompositeChecker {
public:
    ge::graphStatus CheckSinglePara(const IndexerCheckerInfo &info)
    {
        return RunCheck([](LIBaseChecker *checker, const IndexerCheckerInfo &checkerInfo) {
            return checker->CheckSinglePara(checkerInfo);
        },
                        info);
    }

    ge::graphStatus CheckParaExistence(const IndexerCheckerInfo &info)
    {
        return RunCheck([](LIBaseChecker *checker, const IndexerCheckerInfo &checkerInfo) {
            return checker->CheckParaExistence(checkerInfo);
        },
                        info);
    }

    ge::graphStatus CheckFeature(const IndexerCheckerInfo &info)
    {
        return RunCheck([](LIBaseChecker *checker, const IndexerCheckerInfo &checkerInfo) {
            return checker->CheckFeature(checkerInfo);
        },
                        info);
    }

    ge::graphStatus CheckMultiPara(const IndexerCheckerInfo &info)
    {
        return RunCheck([](LIBaseChecker *checker, const IndexerCheckerInfo &checkerInfo) {
            return checker->CheckMultiPara(checkerInfo);
        },
                        info);
    }

    ge::graphStatus Process(const IndexerCheckerInfo &info)
    {
        if (CheckSinglePara(info) != ge::GRAPH_SUCCESS || CheckParaExistence(info) != ge::GRAPH_SUCCESS ||
            CheckFeature(info) != ge::GRAPH_SUCCESS || CheckMultiPara(info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

protected:
    void ClearCheckers()
    {
        checkers_.clear();
    }

    template <typename CheckerType>
    void AddChecker()
    {
        checkers_.push_back(std::make_unique<CheckerType>());
    }

private:
    using CheckMethod = std::function<ge::graphStatus(LIBaseChecker *, const IndexerCheckerInfo &)>;

    ge::graphStatus RunCheck(const CheckMethod &method, const IndexerCheckerInfo &info)
    {
        for (const auto &checker : checkers_) {
            if (method(checker.get(), info) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    std::vector<std::unique_ptr<LIBaseChecker>> checkers_;
};

} // namespace lightning_indexer_v2
} // namespace optiling

#endif // LIGHTNING_INDEXER_V2_CHECKER_RUNNER_H
