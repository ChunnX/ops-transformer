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

#ifndef QUANT_LIGHTNING_INDEXER_V2_CHECKER_H
#define QUANT_LIGHTNING_INDEXER_V2_CHECKER_H

#include <functional>
#include <memory>
#include <vector>
#include "../../../lightning_indexer_v2/op_host/checkers/base_checker_indexer_v2.h"

namespace optiling {
namespace indexer_v2_checker {

class QLIV2Checker {
public:
    ge::graphStatus Init(const IndexerV2CheckerInfo &info);
    ge::graphStatus Process(const IndexerV2CheckerInfo &info);

    ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &info);
    ge::graphStatus CheckParaExistence(const IndexerV2CheckerInfo &info);
    ge::graphStatus CheckFeature(const IndexerV2CheckerInfo &info);
    ge::graphStatus CheckMultiPara(const IndexerV2CheckerInfo &info);

private:
    using CheckMethod = std::function<ge::graphStatus(const IndexerV2BaseChecker *, const IndexerV2CheckerInfo &)>;

    ge::graphStatus RunCheck(const CheckMethod &method, const IndexerV2CheckerInfo &info);
    void RegisterCheckers();
    std::vector<std::unique_ptr<IndexerV2BaseChecker>> checkers_;
};

} // namespace indexer_v2_checker
} // namespace optiling
#endif
