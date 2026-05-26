/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NSA_COMPRESS_WITH_CACHE_TILING_UT_COMMON_H
#define NSA_COMPRESS_WITH_CACHE_TILING_UT_COMMON_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "tiling_context_faker.h"
#include "../../../op_host/nsa_compress_with_cache_tiling_common.h"

namespace ncwc_tiling_ut {

constexpr int64_t kMaxSeqLen = 1024;
constexpr uint64_t kUtCoreNum = 48U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint32_t kTilingDataSize = 8192U;

struct CaseParams {
    int64_t batchSize = 4;
    int64_t headNum = 32;
    int64_t headDim = 192;
    ge::DataType dtype = ge::DT_FLOAT16;
    std::vector<int64_t> actSeqLenList = {144, 192, 10, 45};
    std::string layout = "TND";
    int64_t compressBlockSize = 16;
    int64_t compressStride = 16;
    int64_t actSeqLenType = 1;
    int64_t pageBlockSize = 128;
    uint64_t expectTilingKey = 0;
    ge::graphStatus expectResult = ge::GRAPH_SUCCESS;
};

struct CaseShapes {
    std::vector<int64_t> input;
    std::vector<int64_t> weight;
    std::vector<int64_t> slotMapping;
    std::vector<int64_t> outputCache;
    std::vector<int64_t> blockTable;
    std::vector<int32_t> slotMappingData;
    std::vector<int64_t> actSeqLenData;
    std::vector<int32_t> blockTableData;
    optiling::NsaCompressWithCacheCompileInfo compileInfo{};
};

inline optiling::NsaCompressWithCacheCompileInfo MakeCompileInfo()
{
    optiling::NsaCompressWithCacheCompileInfo info{};
    info.aivNum = static_cast<uint32_t>(kUtCoreNum);
    info.ubSize = kUbSize;
    return info;
}

inline gert::Shape BuildGertShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

inline gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildGertShape(dims);
    shape.MutableStorageShape() = BuildGertShape(dims);
    return shape;
}

inline std::vector<int64_t> ActSeqLenList64Batch()
{
    return {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
            144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
            144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
            144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144};
}

inline std::vector<int64_t> ActSeqLenList32Batch()
{
    return {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
            144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144};
}

inline CaseShapes BuildCaseShapes(const CaseParams& param)
{
    CaseShapes shapes;
    const int64_t blockNumPerBatch = kMaxSeqLen / param.pageBlockSize;
    const int64_t totalBlockNum = blockNumPerBatch * param.batchSize;

    shapes.input = {totalBlockNum, param.pageBlockSize, param.headNum, param.headDim};
    shapes.weight = {param.compressBlockSize, param.headNum};
    shapes.outputCache = {param.batchSize, param.headNum, param.headDim};
    shapes.slotMapping = {param.batchSize};
    shapes.blockTable = {param.batchSize, blockNumPerBatch};

    shapes.slotMappingData.resize(static_cast<size_t>(param.batchSize));
    for (int64_t i = 0; i < param.batchSize; ++i) {
        shapes.slotMappingData[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }

    shapes.actSeqLenData = param.actSeqLenList;
    if (static_cast<int64_t>(shapes.actSeqLenData.size()) != param.batchSize) {
        shapes.actSeqLenData.assign(static_cast<size_t>(param.batchSize), 1);
    }

    shapes.blockTableData.resize(static_cast<size_t>(totalBlockNum));
    for (int64_t i = 0; i < totalBlockNum; ++i) {
        shapes.blockTableData[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return shapes;
}

inline gert::TilingContextPara BuildTilingContextPara(const CaseParams& param, CaseShapes& shapesOut)
{
    shapesOut = BuildCaseShapes(param);
    shapesOut.compileInfo = MakeCompileInfo();

    gert::TilingContextPara para(
        "NsaCompressWithCache",
        {
            {MakeStorageShape(shapesOut.input), param.dtype, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.weight), param.dtype, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.slotMapping), ge::DT_INT32, ge::FORMAT_ND, true,
             shapesOut.slotMappingData.data()},
            {MakeStorageShape(shapesOut.outputCache), param.dtype, ge::FORMAT_ND},
            {MakeStorageShape({static_cast<int64_t>(shapesOut.actSeqLenData.size())}), ge::DT_INT64, ge::FORMAT_ND,
             true, shapesOut.actSeqLenData.data()},
            {MakeStorageShape(shapesOut.blockTable), ge::DT_INT32, ge::FORMAT_ND, true,
             shapesOut.blockTableData.data()},
        },
        {
            {MakeStorageShape(shapesOut.outputCache), param.dtype, ge::FORMAT_ND},
        },
        {
            {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout)},
            {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.compressBlockSize)},
            {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.compressStride)},
            {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.actSeqLenType)},
            {"pageBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pageBlockSize)},
        },
        &shapesOut.compileInfo,
        "Ascend910B",
        kUtCoreNum,
        kUbSize,
        kTilingDataSize);
    return para;
}

}  // namespace ncwc_tiling_ut

#endif  // NSA_COMPRESS_WITH_CACHE_TILING_UT_COMMON_H
