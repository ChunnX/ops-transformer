/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_tiling.cpp
 * \brief Cast tiling for ascend310p, elementwise split following ops-math arch35 formula
 */

#include "cast_tiling.h"

#include <cstdint>

#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int64_t PER_CORE_MIN_BYTES = 4 * 1024;          // one core handles at least 4KB input
constexpr int64_t UB_ELEM_ALIGN = 256;                    // covers 32B alignment for all 1/2/4-byte dtypes
constexpr int64_t UB_RESERVE_BYTES = 1024;
constexpr size_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t MID_F16_BYTES = 2;                      // f16 staging buffer for TWO_CAST/DST_BOOL
constexpr int64_t BUFFER_NUM = 2;                         // double buffer for in/out queues

int64_t GetDtypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT64:
            return 8;
        default:
            return 0;
    }
}

bool Is64OrBf16(ge::DataType d)
{
    return d == ge::DT_INT64 || d == ge::DT_BF16;
}

bool NeedMidF16(ge::DataType x, ge::DataType y)
{
    bool xIsF16 = (x == ge::DT_FLOAT16);
    bool yIsF16 = (y == ge::DT_FLOAT16);
    if (xIsF16 && y != ge::DT_BOOL) {
        return false;
    }
    if (yIsF16) {
        return false;
    }
    // direct pairs without f16
    bool direct = (x == ge::DT_FLOAT && y == ge::DT_INT32) ||
                  (x == ge::DT_INT32 && (y == ge::DT_FLOAT || y == ge::DT_INT16));
    return !direct;
}
}  // namespace

static ge::graphStatus TilingForCast(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context->GetInputDesc(0);
    auto yDesc = context->GetOutputDesc(0);
    auto xShapePtr = context->GetInputShape(0);
    if (xDesc == nullptr || yDesc == nullptr || xShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType xDtype = xDesc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();
    int64_t inBytes = GetDtypeSize(xDtype);
    int64_t outBytes = GetDtypeSize(yDtype);
    if (inBytes == 0 || outBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t coreNumSys = 0;
    int64_t ubSize = 0;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNumSys = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
        if (coreNumSys <= 0) {
            coreNumSys = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        }
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    } else {
        auto compileInfo = context->GetCompileInfo<CastCompileInfo>();
        if (compileInfo == nullptr) {
            return ge::GRAPH_FAILED;
        }
        coreNumSys = static_cast<int64_t>(compileInfo->coreNum);
        ubSize = static_cast<int64_t>(compileInfo->ubSize);
    }
    if (coreNumSys <= 0 || ubSize <= UB_RESERVE_BYTES) {
        return ge::GRAPH_FAILED;
    }

    int64_t shapeSize = xShapePtr->GetStorageShape().GetShapeSize();
    if (shapeSize < 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t midBytes = NeedMidF16(xDtype, yDtype) ? MID_F16_BYTES : 0;
    if (Is64OrBf16(xDtype) || Is64OrBf16(yDtype)) {
        midBytes = 8;  // two 4-byte staging buffers (f32/i32) for i64/bf16 software paths
    }
    int64_t bytesPerElem = BUFFER_NUM * (inBytes + outBytes) + midBytes;
    int64_t ubFormer = (ubSize - UB_RESERVE_BYTES) / bytesPerElem / UB_ELEM_ALIGN * UB_ELEM_ALIGN;
    if (xDtype == ge::DT_INT64) {
        // Diagnostic and production-safe cap for int64 software paths.
        // Large per-core tiles can fault in the i64 software path.
        constexpr int64_t maxInt64TileElems = 1024;
        if (ubFormer > maxInt64TileElems) {
            ubFormer = maxInt64TileElems;
        }
    }
    if (ubFormer <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t coreNum = (shapeSize * inBytes + PER_CORE_MIN_BYTES - 1) / PER_CORE_MIN_BYTES;
    if (coreNum > coreNumSys) {
        coreNum = coreNumSys;
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }

    int64_t blockFormer = ((shapeSize + coreNum - 1) / coreNum + UB_ELEM_ALIGN - 1) / UB_ELEM_ALIGN * UB_ELEM_ALIGN;
    int64_t blockNum = (blockFormer > 0) ? ((shapeSize + blockFormer - 1) / blockFormer) : 1;
    if (blockNum <= 0) {
        blockNum = 1;
        blockFormer = (blockFormer > 0) ? blockFormer : UB_ELEM_ALIGN;
    }
    int64_t blockTail = shapeSize - (blockNum - 1) * blockFormer;
    int64_t ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
    int64_t ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    CastTilingData tilingData;
    tilingData.set_blockNum(blockNum);
    tilingData.set_ubFormer(ubFormer);
    tilingData.set_blockFormer(blockFormer);
    tilingData.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    tilingData.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    tilingData.set_ubTailOfFormerBlock(ubTailOfFormerBlock);
    tilingData.set_ubTailOfTailBlock(ubTailOfTailBlock);

    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = MINIMAL_WORKSPACE;

    context->SetTilingKey(0);
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForCast(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = context->GetCompiledInfo<CastCompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    if (compileInfo == nullptr || platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();
    if (compileInfo->coreNum == 0) {
        compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Cast).Tiling(TilingForCast).TilingParse<CastCompileInfo>(TilingPrepareForCast);
}  // namespace optiling
