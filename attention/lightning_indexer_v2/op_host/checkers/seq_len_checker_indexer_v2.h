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

#ifndef INDEXER_V2_SEQ_LEN_CHECKER_H
#define INDEXER_V2_SEQ_LEN_CHECKER_H

#include "base_checker_indexer_v2.h"

namespace optiling {
namespace indexer_v2_checker {

class SeqLenChecker : public IndexerV2BaseChecker {
public:
    ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckParaExistence(const IndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckMultiPara(const IndexerV2CheckerInfo &info) const override;

private:
    ge::graphStatus CheckKeyDtypes(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckQueryAndOtherDtypes(const IndexerV2CheckerInfo &info) const;
    ge::graphStatus CheckOneDimTensor(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor, const char *name,
                                      int64_t expectedSize) const;
};

} // namespace indexer_v2_checker
} // namespace optiling
#endif
