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
 * \file li_checker.h
 * \brief Composite checker entry for LightningIndexerV2
 */

#ifndef LIGHTNING_INDEXER_V2_LI_CHECKER_H
#define LIGHTNING_INDEXER_V2_LI_CHECKER_H

#include "../lightning_indexer_v2_tiling.h"
#include "checker_runner.h"

namespace optiling {
namespace lightning_indexer_v2 {

class LIChecker : public LICompositeChecker {
public:
    ge::graphStatus Init(const LIV2TilingInfo &source);
    ge::graphStatus Process(const LIV2TilingInfo &source);

private:
    void RegisterCheckers();
    void BuildInfo(const LIV2TilingInfo &source);
    IndexerCheckerInfo info_;
};
} // namespace lightning_indexer_v2
} // namespace optiling

#endif // LIGHTNING_INDEXER_V2_LI_CHECKER_H
