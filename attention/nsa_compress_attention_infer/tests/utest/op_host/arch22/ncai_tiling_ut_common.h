/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NSA_COMPRESS_ATTENTION_INFER_TILING_UT_COMMON_H
#define NSA_COMPRESS_ATTENTION_INFER_TILING_UT_COMMON_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "tiling_context_faker.h"
#include "../../../../op_host/nsa_compress_attention_infer_tiling.h"

namespace ncai_tiling_ut {

constexpr uint64_t kUtCoreNum = 24U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint32_t kTilingDataSize = 8192U;
constexpr uint32_t kHeadSizeVo = 128U;
constexpr int64_t kDefaultBlockNum = 1000;
constexpr uint64_t kSkipTilingKeyCheck = UINT64_MAX;

struct CaseParams {
    int64_t B = 20;
    int64_t S1 = 1;
    int64_t S2 = 4096;
    int64_t numHeads = 64;
    int64_t numKeyValueHeads = 4;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 16;
    int64_t compressBlockSize = 32;
    int64_t compressStride = 16;
    int64_t pageBlockSize = 128;
    int64_t sparseMode = 0;
    float scaleValue = 1.0f;
    std::string layout = "TND";
    uint64_t expectTilingKey = 0;
};

struct CaseShapes {
    std::vector<int64_t> query;
    std::vector<int64_t> key;
    std::vector<int64_t> value;
    std::vector<int64_t> blockTable;
    std::vector<int64_t> output;
    std::vector<int64_t> topkIndicesOut;
    std::vector<int64_t> actualQ;
    std::vector<int64_t> actualCmpKv;
    std::vector<int64_t> actualSelKv;
    optiling::NsaCompressAttentionInferCompileInfo compileInfo{};
};

inline optiling::NsaCompressAttentionInferCompileInfo MakeCompileInfo()
{
    optiling::NsaCompressAttentionInferCompileInfo info{};
    info.aivNum = kUtCoreNum;
    info.coreNum = kUtCoreNum;
    info.ubSize = kUbSize;
    info.inputDataType = ge::DT_FLOAT16;
    info.inputDataByte = 2;
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

inline CaseShapes BuildCaseShapes(const CaseParams& param)
{
    CaseShapes shapes;
    const int64_t n1 = param.numHeads;
    const int64_t n2 = param.numKeyValueHeads;
    const int64_t d = 192;
    const int64_t headSizeVo = kHeadSizeVo;

    shapes.actualQ.assign(param.B, param.S1);
    shapes.actualCmpKv.assign(param.B, param.S2);
    shapes.actualSelKv.assign(param.B, (param.S2 - 1) * param.compressStride + param.compressBlockSize);

    int64_t qSeqLenSum = param.B * param.S1;
    int64_t kvSeqLenSum = 0;
    for (int64_t v : shapes.actualCmpKv) {
        kvSeqLenSum += std::max<int64_t>(v, 0);
    }

    int64_t blockNum = (kvSeqLenSum + param.pageBlockSize - 1) / param.pageBlockSize;
    int64_t maxBlockPerQuery = (param.S2 + param.pageBlockSize - 1) / param.pageBlockSize;
    if (param.S2 < 0) {
        blockNum = kDefaultBlockNum;
        maxBlockPerQuery = (param.S2 + param.pageBlockSize - 1) / param.pageBlockSize;
    }

    if (param.layout == "TND") {
        shapes.query = {qSeqLenSum, n1, d};
        shapes.output = {param.B, n1, headSizeVo};
        shapes.topkIndicesOut = {param.B, n2, param.selectBlockCount};
    } else {
        shapes.query = {param.B, param.S1, n1, d};
        shapes.output = {param.B, param.S1, n1, headSizeVo};
        shapes.topkIndicesOut = {param.B, param.S1, n2, param.selectBlockCount};
    }

    shapes.key = {blockNum, param.pageBlockSize, n2 * d};
    shapes.value = {blockNum, param.pageBlockSize, n2 * headSizeVo};
    shapes.blockTable = {param.B, maxBlockPerQuery};
    return shapes;
}

inline gert::TilingContextPara BuildTilingContextPara(const CaseParams& param, CaseShapes& shapesOut)
{
    shapesOut = BuildCaseShapes(param);
    shapesOut.compileInfo = MakeCompileInfo();

    gert::TilingContextPara para(
        "NsaCompressAttentionInfer",
        {
            {MakeStorageShape(shapesOut.query), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.key), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.value), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape({128, 128}), ge::DT_BOOL, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.blockTable), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape({static_cast<int64_t>(shapesOut.actualQ.size())}), ge::DT_INT64, ge::FORMAT_ND, true,
             shapesOut.actualQ.data()},
            {MakeStorageShape({static_cast<int64_t>(shapesOut.actualCmpKv.size())}), ge::DT_INT64, ge::FORMAT_ND, true,
             shapesOut.actualCmpKv.data()},
            {MakeStorageShape({static_cast<int64_t>(shapesOut.actualSelKv.size())}), ge::DT_INT64, ge::FORMAT_ND, true,
             shapesOut.actualSelKv.data()},
            {MakeStorageShape({20, 4096}), ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape(shapesOut.output), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapesOut.topkIndicesOut), ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"numHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numHeads)},
            {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numKeyValueHeads)},
            {"selectBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.selectBlockSize)},
            {"selectBlockCount", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.selectBlockCount)},
            {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.compressBlockSize)},
            {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.compressStride)},
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(param.scaleValue)},
            {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout)},
            {"pageBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pageBlockSize)},
            {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparseMode)},
        },
        &shapesOut.compileInfo,
        "Ascend910B",
        kUtCoreNum,
        kUbSize,
        kTilingDataSize);
    return para;
}

inline CaseParams LegacyCase0Tnd()
{
    CaseParams param;
    param.layout = "TND";
    param.S2 = 4096;
    param.expectTilingKey = 0;
    return param;
}

inline CaseParams LegacyCase1Bsnd()
{
    CaseParams param;
    param.layout = "BSND";
    param.S2 = 2048;
    param.expectTilingKey = 1;
    return param;
}

inline CaseParams LegacyCaseAclGraphTnd()
{
    CaseParams param;
    param.layout = "TND";
    param.S2 = -1;
    param.expectTilingKey = 0;
    return param;
}

inline CaseParams LegacyCaseMtpTnd()
{
    CaseParams param;
    param.layout = "TND";
    param.S1 = 2;
    param.S2 = 4096;
    param.expectTilingKey = 0;
    return param;
}

}  // namespace ncai_tiling_ut

#endif  // NSA_COMPRESS_ATTENTION_INFER_TILING_UT_COMMON_H
