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
 * \file base_checker.h
 * \brief Base checker class for LightningIndexerV2 family parameters
 */

#ifndef LIGHTNING_INDEXER_V2_BASE_CHECKER_H
#define LIGHTNING_INDEXER_V2_BASE_CHECKER_H

#include <algorithm>
#include <initializer_list>
#include <string>
#include "checker_info.h"

namespace optiling {
namespace lightning_indexer_v2 {

class LIBaseChecker {
public:
    LIBaseChecker() = default;
    virtual ~LIBaseChecker() = default;

    virtual ge::graphStatus CheckSinglePara(const IndexerCheckerInfo &info)
    {
        (void)info;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckParaExistence(const IndexerCheckerInfo &info)
    {
        (void)info;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckFeature(const IndexerCheckerInfo &info)
    {
        (void)info;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckMultiPara(const IndexerCheckerInfo &info)
    {
        (void)info;
        return ge::GRAPH_SUCCESS;
    }

protected:
    ge::graphStatus CheckRequiredTensor(const CheckerTensor &tensor, const char *name,
                                        const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckOptionalTensor(const CheckerTensor &tensor, const char *name,
                                        const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckDtype(const CheckerTensor &tensor, const char *name,
                               std::initializer_list<ge::DataType> expected, const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckFormat(const CheckerTensor &tensor, const char *name,
                                const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckRank(const CheckerTensor &tensor, const char *name, size_t expected,
                              const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckShape(const CheckerTensor &tensor, const char *name, const std::vector<int64_t> &expected,
                               const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckNoZeroDim(const CheckerTensor &tensor, const char *name,
                                   const IndexerCheckerInfo &info) const;
    static std::string DataTypeToString(ge::DataType dtype);
    static std::string ShapeToString(const std::vector<int64_t> &shape);
    static std::string LayoutToString(IndexerLayout layout);
};

} // namespace lightning_indexer_v2
} // namespace optiling

#endif // LIGHTNING_INDEXER_V2_BASE_CHECKER_H
