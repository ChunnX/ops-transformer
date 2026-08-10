/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_QLI_CHECKER_H
#define QUANT_LIGHTNING_INDEXER_V2_QLI_CHECKER_H
#include <functional>
#include <memory>
#include <vector>
#include "../quant_lightning_indexer_v2_tiling.h"
#include "../../../lightning_indexer_v2/op_host/checkers/base_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
class QLIChecker {
public:
    ge::graphStatus Init(const QLIV2TilingInfo &source);
    ge::graphStatus Process(const QLIV2TilingInfo &source);

private:
    void RegisterCheckers();
    void BuildInfo(const QLIV2TilingInfo &source);
    ge::graphStatus RunCheck(const std::function<ge::graphStatus(LIBaseChecker *, const IndexerCheckerInfo &)> &method);
    IndexerCheckerInfo info_;
    std::vector<std::unique_ptr<LIBaseChecker>> checkers_;
};
} // namespace lightning_indexer_v2
} // namespace optiling
#endif
