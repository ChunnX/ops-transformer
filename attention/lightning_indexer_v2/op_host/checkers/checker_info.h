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
 * \file checker_info.h
 * \brief Common checker view shared by LightningIndexerV2 and QuantLightningIndexerV2
 */

#ifndef LIGHTNING_INDEXER_V2_CHECKER_INFO_H
#define LIGHTNING_INDEXER_V2_CHECKER_INFO_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
namespace lightning_indexer_v2 {

enum class IndexerLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2,
    UNKNOWN = 3
};

struct CheckerTensor {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::Tensor *tensor = nullptr;
    std::vector<int64_t> shape;
    bool present = false;
    bool shapeAvailable = false;
};

inline std::vector<int64_t> CopyShape(const gert::Shape &shape)
{
    std::vector<int64_t> dims;
    dims.reserve(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

inline CheckerTensor MakeRequiredTensor(const gert::CompileTimeTensorDesc *desc, const gert::StorageShape *shape)
{
    CheckerTensor result;
    result.desc = desc;
    result.present = (desc != nullptr && shape != nullptr);
    result.shapeAvailable = (shape != nullptr);
    if (shape != nullptr) {
        result.shape = CopyShape(shape->GetStorageShape());
    }
    return result;
}

inline CheckerTensor MakeOptionalTensor(const gert::CompileTimeTensorDesc *desc, const gert::Tensor *tensor)
{
    CheckerTensor result;
    result.desc = desc;
    result.tensor = tensor;
    result.present = (tensor != nullptr);
    result.shapeAvailable = (tensor != nullptr);
    if (tensor != nullptr) {
        result.shape = CopyShape(tensor->GetStorageShape());
    }
    return result;
}

struct IndexerCheckerInfo {
    const char *opName = nullptr;
    bool quantized = false;
    bool isArch35 = false;

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

    IndexerLayout queryLayout = IndexerLayout::UNKNOWN;
    IndexerLayout keyLayout = IndexerLayout::UNKNOWN;
    int64_t batchSize = 0;
    int64_t queryHeadNum = 0;
    int64_t keyHeadNum = 0;
    int64_t querySeqLen = 0;
    int64_t keySeqLen = 0;
    int64_t headDim = 0;
    int64_t groupSize = 0;
    int64_t blockSize = 0;
    int64_t maxBlockNumPerBatch = 0;
    int64_t topk = 0;
    int64_t quantMode = 0;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 0;
    std::vector<uint32_t> keyStrides;
    std::vector<uint32_t> keyDescaleStrides;
};

} // namespace lightning_indexer_v2
} // namespace optiling

#endif // LIGHTNING_INDEXER_V2_CHECKER_INFO_H
