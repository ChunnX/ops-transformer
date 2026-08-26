/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef INDEXER_V2_BASE_CHECKER_H
#define INDEXER_V2_BASE_CHECKER_H

#include <cstdint>
#include <string>
#include <vector>
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace indexer_v2_checker {

enum class CheckerLayout : uint32_t { BSND = 0, TND = 1, PA_BBND = 2, INVALID = 3 };

struct CheckerTensor {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::StorageShape *storageShape = nullptr;
    const gert::Tensor *tensor = nullptr;
    const gert::Stride *stride = nullptr;

    bool IsPresent() const
    {
        return storageShape != nullptr || tensor != nullptr;
    }

    const gert::Shape *GetShape() const
    {
        if (storageShape != nullptr) {
            return &storageShape->GetStorageShape();
        }
        return tensor == nullptr ? nullptr : &tensor->GetStorageShape();
    }
};

struct IndexerV2CheckerInfo {
    const char *opName = nullptr;
    bool quantized = false;
    bool arch35 = false;

    CheckerTensor query;
    CheckerTensor key;
    CheckerTensor weights;
    CheckerTensor queryDescale;
    CheckerTensor keyDescale;
    CheckerTensor cuSeqlensQ;
    CheckerTensor cuSeqlensK;
    CheckerTensor sequsedQ;
    CheckerTensor sequsedK;
    CheckerTensor cmpResidualK;
    CheckerTensor blockTable;
    CheckerTensor outputIdxOffset;
    CheckerTensor metadata;
    CheckerTensor sparseIndices;
    CheckerTensor sparseValues;

    CheckerLayout layoutQ = CheckerLayout::INVALID;
    CheckerLayout layoutK = CheckerLayout::INVALID;
    std::string layoutQText;
    std::string layoutKText;
    int64_t topk = 0;
    int64_t quantMode = 0;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 0;

    int64_t batch = 0;
    int64_t qSeq = 0;
    int64_t kSeq = 0;
    int64_t qHeads = 0;
    int64_t kHeads = 0;
    int64_t headDim = 0;
    int64_t blockSize = 0;
};

class IndexerV2BaseChecker {
public:
    virtual ~IndexerV2BaseChecker() = default;
    virtual ge::graphStatus CheckParaExistence(const IndexerV2CheckerInfo &info) const;
    virtual ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &info) const;
    virtual ge::graphStatus CheckFeature(const IndexerV2CheckerInfo &info) const;
    virtual ge::graphStatus CheckMultiPara(const IndexerV2CheckerInfo &info) const;

protected:
    ge::graphStatus CheckFormat(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                const char *name) const;
    ge::graphStatus CheckDtype(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor, const char *name,
                               const std::vector<ge::DataType> &expected) const;
    ge::graphStatus CheckRank(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor, const char *name,
                              uint32_t expected) const;
    ge::graphStatus CheckNotEmpty(const IndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                  const char *name) const;
    static std::string DtypesToString(const std::vector<ge::DataType> &dtypes);
    static std::string ShapeToString(const CheckerTensor &tensor);
};

CheckerLayout ParseLayout(const std::string &layout);
ge::graphStatus FinalizeCommonInfo(IndexerV2CheckerInfo &info);

} // namespace indexer_v2_checker
} // namespace optiling
#endif
