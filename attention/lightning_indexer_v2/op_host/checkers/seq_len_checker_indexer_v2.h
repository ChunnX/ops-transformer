/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
    ge::graphStatus CheckOneDimTensor(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                      const char *name, int64_t expectedSize) const;
    ge::graphStatus CheckCuSeqlensValues(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                         const char *name, int64_t total) const;
    ge::graphStatus CheckSeqUsedValues(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                       const CheckerTensor *cuSeqlens, const char *name, int64_t maximum) const;
};

} // namespace indexer_v2_checker
} // namespace optiling
#endif
