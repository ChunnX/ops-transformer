/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_tiling.cpp
 * \brief MoeTutelDispatch tiling, Ascend 950 (arch35) only.
 */

#include "moe_tutel_dispatch_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "graph/types.h"
#include "securec.h"

using namespace Ops::Base;

namespace optiling {
namespace {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_GATES_INDEX = 1;
constexpr size_t INPUT_INDICES_INDEX = 2;
constexpr size_t INPUT_LOCATIONS_INDEX = 3;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t ATTR_CAPACITY_INDEX = 0;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t X_RANK = 2;
constexpr size_t ROUTE_RANK = 2;
constexpr size_t Y_RANK = 3;

constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MAX_ROUTE_BATCH = 64;
constexpr int64_t PAYLOAD_BUFFER_NUM = 2;
constexpr int64_t MAX_DATA_COPY_BLOCK_BYTES = 2097151;
constexpr int64_t MAX_DIM_VALUE = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
constexpr size_t SYSTEM_WORKSPACE_BYTES = static_cast<size_t>(16) * 1024 * 1024;

inline bool SafeMul(int64_t lhs, int64_t rhs, int64_t &out)
{
    if (lhs == 0 || rhs == 0) {
        out = 0;
        return true;
    }
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

inline int64_t AlignUp(int64_t value, int64_t align)
{
    return value == 0 ? 0 : ((value + align - 1) / align) * align;
}

inline int64_t MetadataBytes(int64_t routeBatch, int64_t dtypeBytes)
{
    return AlignUp(routeBatch * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES) +
           AlignUp(routeBatch * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES) +
           AlignUp(routeBatch * dtypeBytes, BLOCK_BYTES);
}
} // namespace

class MoeTutelDispatchTiling {
public:
    explicit MoeTutelDispatchTiling(gert::TilingContext &context)
        : context_(&context)
    {}

    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckDtypeAndFormat();
    ge::graphStatus CheckShapeRank();
    ge::graphStatus ParseDimsAndAttr();
    ge::graphStatus ComputeScatterTiling(int64_t aivCoreLimit, int64_t ubSize);
    ge::graphStatus SaveTiling(uint64_t tilingKey, int64_t usedCoreNum, int64_t routeBatch, int64_t hChunkElems);

    gert::TilingContext *context_ = nullptr;
    MoeTutelDispatchTilingData tilingData_{};

    const gert::StorageShape *xShape_ = nullptr;
    const gert::StorageShape *gatesShape_ = nullptr;
    const gert::StorageShape *yShape_ = nullptr;

    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t routeK_ = 0;
    int64_t expertE_ = 0;
    int64_t capacity_ = 0;
    int64_t routeNum_ = 0;
    int64_t yElemCnt_ = 0;
    int64_t dtypeBytes_ = 0;
};

ge::graphStatus MoeTutelDispatchTiling::CheckDtypeAndFormat()
{
    const gert::CompileTimeTensorDesc *xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    const gert::CompileTimeTensorDesc *gatesDesc = context_->GetInputDesc(INPUT_GATES_INDEX);
    const gert::CompileTimeTensorDesc *indicesDesc = context_->GetInputDesc(INPUT_INDICES_INDEX);
    const gert::CompileTimeTensorDesc *locationsDesc = context_->GetInputDesc(INPUT_LOCATIONS_INDEX);
    const gert::CompileTimeTensorDesc *yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gatesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);

    const ge::DataType xType = xDesc->GetDataType();
    if (xType != ge::DT_FLOAT16 && xType != ge::DT_FLOAT && xType != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xType).c_str(),
                                              "x must be float16, float32 or bfloat16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType gatesType = gatesDesc->GetDataType();
    const ge::DataType yType = yDesc->GetDataType();
    if (gatesType != xType || yType != xType) {
        const std::string incorrectDtypes = ToString(xType) + ", " + ToString(gatesType) + ", " + ToString(yType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x, gates, y", incorrectDtypes.c_str(),
                                               "x, gates and y must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType indicesType = indicesDesc->GetDataType();
    const ge::DataType locationsType = locationsDesc->GetDataType();
    if (indicesType != ge::DT_INT32 || locationsType != ge::DT_INT32) {
        const std::string incorrectDtypes = ToString(indicesType) + ", " + ToString(locationsType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "indices, locations", incorrectDtypes.c_str(),
                                               "indices and locations must be int32");
        return ge::GRAPH_FAILED;
    }

    const gert::CompileTimeTensorDesc *descs[] = {xDesc, gatesDesc, indicesDesc, locationsDesc, yDesc};
    const char *names[] = {"x", "gates", "indices", "locations", "y"};
    for (size_t i = 0; i < sizeof(descs) / sizeof(descs[0]); ++i) {
        const ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(descs[i]->GetStorageFormat()));
        if (format != ge::FORMAT_ND) {
            const std::string incorrectFormat = std::to_string(static_cast<int32_t>(format));
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), names[i], incorrectFormat.c_str(),
                                                   "all tensors must use ND format");
            return ge::GRAPH_FAILED;
        }
    }

    dtypeBytes_ = (xType == ge::DT_FLOAT) ? 4 : 2;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelDispatchTiling::CheckShapeRank()
{
    xShape_ = context_->GetInputShape(INPUT_X_INDEX);
    gatesShape_ = context_->GetInputShape(INPUT_GATES_INDEX);
    const gert::StorageShape *indicesShape = context_->GetInputShape(INPUT_INDICES_INDEX);
    const gert::StorageShape *locationsShape = context_->GetInputShape(INPUT_LOCATIONS_INDEX);
    yShape_ = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gatesShape_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    const gert::Shape &x = xShape_->GetStorageShape();
    const gert::Shape &gates = gatesShape_->GetStorageShape();
    const gert::Shape &y = yShape_->GetStorageShape();
    if (x.GetDimNum() != X_RANK || gates.GetDimNum() != ROUTE_RANK || y.GetDimNum() != Y_RANK) {
        const std::string incorrectDims = std::to_string(x.GetDimNum()) + ", " + std::to_string(gates.GetDimNum()) +
                                          ", " + std::to_string(y.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x, gates, y", incorrectDims.c_str(),
                                                  "the ranks must be 2, 2 and 3, respectively");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &indices = indicesShape->GetStorageShape();
    const gert::Shape &locations = locationsShape->GetStorageShape();
    if (indices.GetDimNum() != ROUTE_RANK || locations.GetDimNum() != ROUTE_RANK) {
        const std::string incorrectDims =
            std::to_string(indices.GetDimNum()) + ", " + std::to_string(locations.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "indices, locations", incorrectDims.c_str(),
                                                  "indices and locations must both be 2D");
        return ge::GRAPH_FAILED;
    }
    if (indices != gates || locations != gates) {
        const std::string incorrectShapes = ToString(gates) + ", " + ToString(indices) + ", " + ToString(locations);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "gates, indices, locations",
                                               incorrectShapes.c_str(),
                                               "gates, indices and locations must have identical shapes");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelDispatchTiling::ParseDimsAndAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *capacity = attrs->GetAttrPointer<int64_t>(ATTR_CAPACITY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, capacity);
    capacity_ = *capacity;

    const gert::Shape &x = xShape_->GetStorageShape();
    const gert::Shape &gates = gatesShape_->GetStorageShape();
    const gert::Shape &y = yShape_->GetStorageShape();
    samples_ = x.GetDim(DIM_0);
    hidden_ = x.GetDim(DIM_1);
    routeK_ = gates.GetDim(DIM_0);
    expertE_ = y.GetDim(DIM_0);

    const int64_t dims[] = {samples_, hidden_, routeK_, expertE_, capacity_};
    const char *names[] = {"S", "H", "K", "E", "capacity"};
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); ++i) {
        if (dims[i] < 0 || dims[i] > MAX_DIM_VALUE) {
            const std::string incorrectValue = std::to_string(dims[i]);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), names[i], incorrectValue.c_str(),
                                                  "the value must be in [0, INT32_MAX]");
            return ge::GRAPH_FAILED;
        }
    }

    if (gates.GetDim(DIM_1) != samples_) {
        const std::string incorrectShapes = ToString(x) + ", " + ToString(gates);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x, gates", incorrectShapes.c_str(),
                                               "gates.dim(1) must equal x.dim(0)");
        return ge::GRAPH_FAILED;
    }
    if (y.GetDim(DIM_1) != capacity_ || y.GetDim(DIM_2) != hidden_) {
        const std::string incorrectShapes = ToString(x) + ", " + ToString(y);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x, y", incorrectShapes.c_str(),
                                               "caller-provided y must have shape [E, capacity, H]");
        return ge::GRAPH_FAILED;
    }

    int64_t xElemCnt = 0;
    int64_t expertCapacity = 0;
    if (!SafeMul(routeK_, samples_, routeNum_) || !SafeMul(samples_, hidden_, xElemCnt) ||
        !SafeMul(expertE_, capacity_, expertCapacity) || !SafeMul(expertCapacity, hidden_, yElemCnt_)) {
        const std::string incorrectValues = std::to_string(samples_) + ", " + std::to_string(hidden_) + ", " +
                                            std::to_string(routeK_) + ", " + std::to_string(expertE_) + ", " +
                                            std::to_string(capacity_);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "S, H, K, E, capacity", incorrectValues.c_str(),
                                               "x, route and y element counts must fit in signed int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelDispatchTiling::ComputeScatterTiling(int64_t aivCoreLimit, int64_t ubSize)
{
    const int64_t usedCoreNum = std::min(aivCoreLimit, std::max(yElemCnt_, routeNum_));
    if (usedCoreNum < 1) {
        OP_LOGE(context_, "scatter path requires at least one core");
        return ge::GRAPH_FAILED;
    }

    const int64_t maxRoutesPerCore = routeNum_ / usedCoreNum + ((routeNum_ % usedCoreNum) == 0 ? 0 : 1);
    const int64_t rawRowBytes = hidden_ * dtypeBytes_;
    const int64_t rowStride = AlignUp(rawRowBytes, BLOCK_BYTES);
    if (rawRowBytes <= MAX_DATA_COPY_BLOCK_BYTES) {
        const int64_t maxBatch = std::min(std::min(MAX_ROUTE_BATCH, maxRoutesPerCore), samples_);
        for (int64_t batch = maxBatch; batch >= 1; --batch) {
            const int64_t payloadBytes = PAYLOAD_BUFFER_NUM * batch * rowStride;
            if (MetadataBytes(batch, dtypeBytes_) + payloadBytes <= ubSize) {
                return SaveTiling(MOE_TUTEL_DISPATCH_FULL_ROW, usedCoreNum, batch, hidden_);
            }
        }
    }

    const int64_t metadataBytes = MetadataBytes(1, dtypeBytes_);
    if (ubSize <= metadataBytes) {
        OP_LOGE(context_, "UB size %lld cannot hold route metadata", static_cast<long long>(ubSize));
        return ge::GRAPH_FAILED;
    }
    const int64_t maxChunkBytes = std::min((ubSize - metadataBytes) / PAYLOAD_BUFFER_NUM, MAX_DATA_COPY_BLOCK_BYTES);
    const int64_t chunkBytes = (maxChunkBytes / BLOCK_BYTES) * BLOCK_BYTES;
    const int64_t hChunkElems = chunkBytes / dtypeBytes_;
    if (hChunkElems < 1) {
        OP_LOGE(context_, "UB size %lld cannot hold one payload chunk", static_cast<long long>(ubSize));
        return ge::GRAPH_FAILED;
    }
    return SaveTiling(MOE_TUTEL_DISPATCH_SPLIT_H, usedCoreNum, 1, hChunkElems);
}

ge::graphStatus MoeTutelDispatchTiling::SaveTiling(uint64_t tilingKey, int64_t usedCoreNum, int64_t routeBatch,
                                                   int64_t hChunkElems)
{
    tilingData_.samples = samples_;
    tilingData_.hidden = hidden_;
    tilingData_.expertE = expertE_;
    tilingData_.capacity = capacity_;
    tilingData_.routeNum = routeNum_;
    tilingData_.yElemCnt = yElemCnt_;
    tilingData_.usedCoreNum = usedCoreNum;
    tilingData_.routeBatch = routeBatch;
    tilingData_.hChunkElems = hChunkElems;

    auto *rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    const errno_t ret =
        memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &tilingData_, sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_, "copy tiling data failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(sizeof(tilingData_));
    const uint64_t encodedTilingKey = GET_TPL_TILING_KEY(tilingKey);
    context_->SetTilingKey(encodedTilingKey);
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context_->SetScheduleMode(1);

    OP_LOGD(context_,
            "key=%llu S=%lld H=%lld K=%lld E=%lld C=%lld routeNum=%lld yElems=%lld "
            "cores=%lld batch=%lld chunk=%lld",
            static_cast<unsigned long long>(encodedTilingKey), static_cast<long long>(samples_),
            static_cast<long long>(hidden_), static_cast<long long>(routeK_), static_cast<long long>(expertE_),
            static_cast<long long>(capacity_), static_cast<long long>(routeNum_), static_cast<long long>(yElemCnt_),
            static_cast<long long>(usedCoreNum), static_cast<long long>(routeBatch),
            static_cast<long long>(hChunkElems));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelDispatchTiling::DoTiling()
{
    if (CheckDtypeAndFormat() != ge::GRAPH_SUCCESS || CheckShapeRank() != ge::GRAPH_SUCCESS ||
        ParseDimsAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYSTEM_WORKSPACE_BYTES;

    // TilingPrepare records the repository's GetAivCoreNum/GetUbSize results.
    // MIX_AIV_1_0 launches one vector block per AIV, so aivCoreLimit is the
    // blockDim limit for every key in this kernel.
    const MoeTutelDispatchCompileInfo *compileInfo = context_->GetCompileInfo<MoeTutelDispatchCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    const int64_t aivCoreLimit = compileInfo->aivCoreLimit;
    const int64_t ubSize = compileInfo->ubSize;
    if (aivCoreLimit < 1 || ubSize < 1) {
        OP_LOGE(context_, "invalid platform info: aivCoreLimit=%lld, ubSize=%lld", static_cast<long long>(aivCoreLimit),
                static_cast<long long>(ubSize));
        return ge::GRAPH_FAILED;
    }

    if (yElemCnt_ == 0) {
        return SaveTiling(MOE_TUTEL_DISPATCH_EMPTY_OUTPUT, 1, 1, 0);
    }
    if (routeNum_ == 0) {
        const int64_t usedCoreNum = std::min(aivCoreLimit, yElemCnt_);
        return SaveTiling(MOE_TUTEL_DISPATCH_ZERO_ONLY, usedCoreNum, 1, 0);
    }
    return ComputeScatterTiling(aivCoreLimit, ubSize);
}

static ge::graphStatus Tiling4MoeTutelDispatch(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return MoeTutelDispatchTiling(*context).DoTiling();
}

static ge::graphStatus TilingPrepare4MoeTutelDispatch(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    MoeTutelDispatchCompileInfo *compileInfo = context->GetCompiledInfo<MoeTutelDispatchCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->aivCoreLimit = static_cast<int64_t>(GetAivCoreNum(context));
    compileInfo->ubSize = static_cast<int64_t>(GetUbSize(context));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTutelDispatch)
    .Tiling(Tiling4MoeTutelDispatch)
    .TilingParse<MoeTutelDispatchCompileInfo>(TilingPrepare4MoeTutelDispatch);
} // namespace optiling
