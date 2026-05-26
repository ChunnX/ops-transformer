/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NSA_SELECTED_ATTENTION_INFER_TILING_UT_COMMON_H
#define NSA_SELECTED_ATTENTION_INFER_TILING_UT_COMMON_H

#include <cstdint>
#include <string>
#include <vector>
#include "tiling_context_faker.h"

namespace optiling {
struct NsaSelectAttentionCompileInfo {
    int64_t core_num;
};
}  // namespace optiling

namespace nsai_tiling_ut {

constexpr uint64_t kUtCoreNum = 48U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint32_t kTilingDataSize = 8192U;

struct CaseParams {
    int64_t batchSize = 30;
    int64_t qSeqSize = 3;
    int64_t headSize = 1;
    int64_t headDim = 192;
    int64_t maxBlockNumPerBatch = 2;
    int64_t blockSize = 128;
    int64_t seqSize = 256;
    int64_t headDimV = 128;
    int64_t headSizeV = 1;
    int64_t selectedBlockSize = 128;
    int64_t selectedBlockCount = 2;
    int64_t sparseMode = 0;
    float scaleValue = 1.0f;
    std::string inputLayout = "TND";
    ge::DataType dtype = ge::DT_FLOAT16;
    std::vector<int64_t> actualSeqQLenList;
    std::vector<int64_t> actualSeqKVLenList;
    int64_t numTokens = 90;
    ge::graphStatus expectResult = ge::GRAPH_FAILED;
    uint64_t expectTilingKey = 0;
};

struct CaseShapes {
    std::vector<int64_t> query;
    std::vector<int64_t> key;
    std::vector<int64_t> value;
    std::vector<int64_t> topkIndices;
    std::vector<int64_t> blockTable;
    std::vector<int64_t> attentionOut;
    std::vector<int64_t> actualQ;
    std::vector<int64_t> actualKv;
    std::vector<int32_t> topkData;
    std::vector<int32_t> blockTableData;
    optiling::NsaSelectAttentionCompileInfo compileInfo{};
};

inline optiling::NsaSelectAttentionCompileInfo MakeCompileInfo()
{
    optiling::NsaSelectAttentionCompileInfo info{};
    info.core_num = static_cast<int64_t>(kUtCoreNum);
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

inline std::vector<int64_t> DefaultActSeqQLen30()
{
    return {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
}

inline std::vector<int64_t> DefaultActSeqKVLen30()
{
    return {256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
            256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256};
}

inline void BuildTopkData(const CaseParams& param, std::vector<int32_t>& topkData, std::vector<int64_t>& topkShape)
{
    topkData.clear();
    if (param.inputLayout == "TND") {
        topkShape = {param.numTokens, param.headSizeV, param.selectedBlockCount};
        topkData.assign(static_cast<size_t>(param.numTokens * param.headSizeV * param.selectedBlockCount), 0);
    } else {
        topkShape = {param.batchSize, param.qSeqSize, param.headSizeV, param.selectedBlockCount};
        topkData.assign(
            static_cast<size_t>(param.batchSize * param.qSeqSize * param.headSizeV * param.selectedBlockCount), 0);
    }
}

inline CaseShapes BuildCaseShapes(const CaseParams& param)
{
    CaseShapes shapes;
    shapes.actualQ = param.actualSeqQLenList;
    shapes.actualKv = param.actualSeqKVLenList;
    if (shapes.actualQ.empty()) {
        shapes.actualQ = DefaultActSeqQLen30();
    }
    if (shapes.actualKv.empty()) {
        shapes.actualKv = DefaultActSeqKVLen30();
    }

    if (param.inputLayout == "BSND") {
        shapes.query = {param.batchSize, param.qSeqSize, param.headSize, param.headDim};
        shapes.attentionOut = {param.batchSize, param.qSeqSize, param.headSize, param.headDimV};
    } else if (param.inputLayout == "TND") {
        shapes.query = {param.numTokens, param.headSize, param.headDim};
        shapes.attentionOut = {param.numTokens, param.headSize, param.headDimV};
    } else {
        shapes.query = {param.batchSize, param.qSeqSize, param.headSize * param.headDim};
        shapes.attentionOut = {param.batchSize, param.qSeqSize, param.headSize * param.headDimV};
    }

    shapes.key = {param.maxBlockNumPerBatch, param.blockSize, param.headSizeV, param.headDim};
    shapes.value = {param.maxBlockNumPerBatch, param.blockSize, param.headSizeV, param.headDimV};
    shapes.blockTable = {param.batchSize, param.maxBlockNumPerBatch};
    BuildTopkData(param, shapes.topkData, shapes.topkIndices);

    shapes.blockTableData.assign(static_cast<size_t>(param.batchSize * param.maxBlockNumPerBatch), 0);
    return shapes;
}

inline gert::TilingContextPara BuildTilingContextPara(const CaseParams& param, CaseShapes& shapesOut)
{
    shapesOut = BuildCaseShapes(param);
    shapesOut.compileInfo = MakeCompileInfo();

    gert::TilingContextPara para(
        "NsaSelectedAttentionInfer",
        {
            {MakeStorageShape(shapesOut.query), param.dtype, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.key), param.dtype, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.value), param.dtype, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.topkIndices), ge::DT_INT32, ge::FORMAT_ND, true, shapesOut.topkData.data()},
            {MakeStorageShape({}), ge::DT_BOOL, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.blockTable), ge::DT_INT32, ge::FORMAT_ND, true,
             shapesOut.blockTableData.data()},
            {MakeStorageShape({static_cast<int64_t>(shapesOut.actualQ.size())}), ge::DT_INT64, ge::FORMAT_ND, true,
             shapesOut.actualQ.data()},
            {MakeStorageShape({static_cast<int64_t>(shapesOut.actualKv.size())}), ge::DT_INT64, ge::FORMAT_ND, true,
             shapesOut.actualKv.data()},
        },
        {
            {MakeStorageShape(shapesOut.attentionOut), param.dtype, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.inputLayout)},
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.headSize)},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.headSizeV)},
            {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.selectedBlockSize)},
            {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.selectedBlockCount)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.blockSize)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(param.scaleValue)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparseMode)},
        },
        &shapesOut.compileInfo,
        "Ascend910B",
        kUtCoreNum,
        kUbSize,
        kTilingDataSize);
    return para;
}

inline CaseParams LegacyCaseQ3Tnd()
{
    CaseParams param;
    param.inputLayout = "TND";
    param.actualSeqQLenList = DefaultActSeqQLen30();
    param.actualSeqKVLenList = DefaultActSeqKVLen30();
    param.numTokens = 90;
    return param;
}

inline CaseParams LegacyCaseQ1Bsnd()
{
    CaseParams param;
    param.qSeqSize = 1;
    param.inputLayout = "BSND";
    param.actualSeqQLenList = DefaultActSeqQLen30();
    param.actualSeqKVLenList = DefaultActSeqKVLen30();
    param.numTokens = 30;
    return param;
}

inline CaseParams LegacyCaseQ1Batch1()
{
    CaseParams param;
    param.batchSize = 1;
    param.qSeqSize = 1;
    param.inputLayout = "BSND";
    param.actualSeqQLenList = {1};
    param.actualSeqKVLenList = {256};
    param.numTokens = 1;
    return param;
}

}  // namespace nsai_tiling_ut

#endif  // NSA_SELECTED_ATTENTION_INFER_TILING_UT_COMMON_H
