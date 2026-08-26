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
 * \file common_checker_lightning_indexer_v2.h
 * \brief Declares common parameter checks shared by Lightning Indexer V2
 * operators.
 */

#ifndef COMMON_CHECKER_LIGHTNING_INDEXER_V2_H
#define COMMON_CHECKER_LIGHTNING_INDEXER_V2_H

#include "base_checker_lightning_indexer_v2.h"

namespace optiling {
namespace indexer_v2_checker {

class CommonChecker : public IndexerV2BaseChecker {
public:
    ge::graphStatus CheckParaExistence(const IndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckFeature(const IndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckMultiPara(const IndexerV2CheckerInfo &info) const override;

private:
    ge::graphStatus CheckTensorBasics(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckAttrs(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckLayouts(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckTopk(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckCompressionAttrs(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckReturnValue(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckHeadFeature(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckFeatureRanges(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckMainShapes(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckOutputShapes(const IndexerV2CheckerInfo &info) const;
};

} // namespace indexer_v2_checker
} // namespace optiling
#endif // COMMON_CHECKER_LIGHTNING_INDEXER_V2_H
