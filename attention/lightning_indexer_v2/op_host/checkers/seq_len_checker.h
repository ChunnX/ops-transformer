/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LIGHTNING_INDEXER_V2_SEQ_LEN_CHECKER_H
#define LIGHTNING_INDEXER_V2_SEQ_LEN_CHECKER_H
#include "base_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
class SeqLenChecker : public LIBaseChecker {
public:
    ge::graphStatus CheckSinglePara(const IndexerCheckerInfo &info) override;
    ge::graphStatus CheckParaExistence(const IndexerCheckerInfo &info) override;
    ge::graphStatus CheckMultiPara(const IndexerCheckerInfo &info) override;

private:
    ge::graphStatus CheckSeqTensor(const CheckerTensor &tensor, const char *name, int64_t length,
                                   const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckCuValues(const CheckerTensor &tensor, int64_t total, const char *name,
                                  const IndexerCheckerInfo &info) const;
    ge::graphStatus CheckUsedValues(const CheckerTensor &used, const CheckerTensor &cu, int64_t fallbackLimit,
                                    const char *name, const IndexerCheckerInfo &info) const;
};
} // namespace lightning_indexer_v2
} // namespace optiling
#endif
