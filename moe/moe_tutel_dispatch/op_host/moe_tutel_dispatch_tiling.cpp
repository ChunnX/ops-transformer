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
 * \file moe_tutel_dispatch_tiling.cpp
 * \brief MoeTutelDispatch tiling, Ascend950 (arch35) only
 */

#include "moe_tutel_dispatch_tiling.h"

#include <limits>
#include <string>

using namespace Ops::Base;

namespace optiling {
namespace {
constexpr size_t IDX_X = 0;
constexpr size_t IDX_GATES = 1;
constexpr size_t IDX_INDICES = 2;
constexpr size_t IDX_LOCATIONS = 3;
constexpr size_t IDX_Y = 0;
constexpr size_t ATTR_CAPACITY = 0;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;

constexpr size_t X_DIM_NUM = 2;
constexpr size_t ROUTE_DIM_NUM = 2;
constexpr size_t Y_DIM_NUM = 3;

constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t BUFFER_NUM = 2; // TQueBind 双缓冲
// 预留给 TPipe 的栈区：阶段 A 的 InitGlobalMemory 走 PopStackBuffer 取「未被 InitBuffer 占用的
// UB 尾部」，留得太少会让清零退化成极小分块的 DataCopyPad。32KB 可让单次搬运达到 32KB。
constexpr int64_t UB_RESERVE_BYTES = 32768;
constexpr int64_t MAX_TILE_ROUTES = 64;
constexpr int64_t MIN_ROUTES_PER_CORE = 8;
constexpr int64_t MIN_ELEMS_PER_CORE = 4096;
// 每条路由的元数据：indices(int32) + locations(int32) + gates(<= 4B)
constexpr int64_t META_BYTES_PER_ROUTE = 12;
constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;

inline bool MulOverflow(int64_t a, int64_t b, int64_t &out)
{
    if (a == 0 || b == 0) {
        out = 0;
        return false;
    }
    if (a > std::numeric_limits<int64_t>::max() / b) {
        return true;
    }
    out = a * b;
    return false;
}
} // namespace

class MoeTutelDispatchTiling {
public:
    explicit MoeTutelDispatchTiling(gert::TilingContext *context)
        : context_(context)
    {}

    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckShapeRank();
    ge::graphStatus ParseDimsAndAttr();
    ge::graphStatus CheckDtype();
    ge::graphStatus ComputeTiling();
    void SaveTiling(uint64_t tilingKey) const;

    gert::TilingContext *context_ = nullptr;
    mutable MoeTutelDispatchTilingData tilingData_;

    const gert::StorageShape *xShape_ = nullptr;
    const gert::StorageShape *gatesShape_ = nullptr;
    const gert::StorageShape *yShape_ = nullptr;

    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t expertNum_ = 0;
    int64_t capacity_ = 0;
    int64_t routeNum_ = 0;
    int64_t yElemCnt_ = 0;
    int64_t dtypeSize_ = 0;

    int64_t usedCoreNum_ = 1;
    int64_t initPerCore_ = 0;
    int64_t routePerCore_ = 0;
    int64_t tileRoutes_ = 1;
    int64_t hChunkElems_ = 0;
};

ge::graphStatus MoeTutelDispatchTiling::CheckDtype()
{
    auto xDesc = context_->GetInputDesc(IDX_X);
    auto gatesDesc = context_->GetInputDesc(IDX_GATES);
    auto indicesDesc = context_->GetInputDesc(IDX_INDICES);
    auto locationsDesc = context_->GetInputDesc(IDX_LOCATIONS);
    auto yDesc = context_->GetOutputDesc(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gatesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);

    const ge::DataType xType = xDesc->GetDataType();
    if (xType != ge::DT_FLOAT16 && xType != ge::DT_FLOAT && xType != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ToString(xType).c_str(),
                                  "float16, float32 or bfloat16");
        return ge::GRAPH_FAILED;
    }
    if (gatesDesc->GetDataType() != xType) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gates", ToString(gatesDesc->GetDataType()).c_str(),
                                  ToString(xType).c_str());
        return ge::GRAPH_FAILED;
    }
    if (yDesc->GetDataType() != xType) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", ToString(yDesc->GetDataType()).c_str(),
                                  ToString(xType).c_str());
        return ge::GRAPH_FAILED;
    }
    if (indicesDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "indices", ToString(indicesDesc->GetDataType()).c_str(),
                                  "int32");
        return ge::GRAPH_FAILED;
    }
    if (locationsDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "locations", ToString(locationsDesc->GetDataType()).c_str(),
                                  "int32");
        return ge::GRAPH_FAILED;
    }

    dtypeSize_ = (xType == ge::DT_FLOAT) ? 4 : 2;
    return ge::GRAPH_SUCCESS;
}

// 只做「拿得到、rank 对、三份路由张量 shape 一致」这一层
ge::graphStatus MoeTutelDispatchTiling::CheckShapeRank()
{
    xShape_ = context_->GetInputShape(IDX_X);
    gatesShape_ = context_->GetInputShape(IDX_GATES);
    auto indicesShape = context_->GetInputShape(IDX_INDICES);
    auto locationsShape = context_->GetInputShape(IDX_LOCATIONS);
    yShape_ = context_->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gatesShape_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    const gert::Shape &x = xShape_->GetStorageShape();
    const gert::Shape &gates = gatesShape_->GetStorageShape();
    const gert::Shape &y = yShape_->GetStorageShape();

    if (x.GetDimNum() != X_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(x.GetDimNum()).c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    if (gates.GetDimNum() != ROUTE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gates", std::to_string(gates.GetDimNum()).c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    if (y.GetDimNum() != Y_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y", std::to_string(y.GetDimNum()).c_str(), "3D");
        return ge::GRAPH_FAILED;
    }
    if (indicesShape->GetStorageShape() != gates) {
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "indices",
                                  ToString(indicesShape->GetStorageShape()).c_str(), ToString(gates).c_str());
        return ge::GRAPH_FAILED;
    }
    if (locationsShape->GetStorageShape() != gates) {
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "locations",
                                  ToString(locationsShape->GetStorageShape()).c_str(), ToString(gates).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 取 capacity 与各维，做维间关系、非负、乘法溢出三类校验
ge::graphStatus MoeTutelDispatchTiling::ParseDimsAndAttr()
{
    const gert::Shape &x = xShape_->GetStorageShape();
    const gert::Shape &gates = gatesShape_->GetStorageShape();
    const gert::Shape &y = yShape_->GetStorageShape();

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *capacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_CAPACITY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, capacityPtr);
    capacity_ = *capacityPtr;
    if (capacity_ < 0) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "capacity", std::to_string(capacity_).c_str(),
                                  "a non-negative integer");
        return ge::GRAPH_FAILED;
    }

    samples_ = x.GetDim(DIM_0);
    hidden_ = x.GetDim(DIM_1);
    expertNum_ = gates.GetDim(DIM_0);
    // 负维在本算子无意义，且会让后续的 CeilDiv / 区间切分失去意义，直接拒收
    if (samples_ < 0 || hidden_ < 0 || expertNum_ < 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x and gates", ToString(x).c_str(),
                                              "all dims must be non-negative");
        return ge::GRAPH_FAILED;
    }
    if (gates.GetDim(DIM_1) != samples_) {
        const std::string reason = "the 1st dim of gates must equal the 0th dim of x " + ToString(x);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gates", ToString(gates).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (y.GetDim(DIM_0) != expertNum_ || y.GetDim(DIM_1) != capacity_ || y.GetDim(DIM_2) != hidden_) {
        const std::string expect = "[" + std::to_string(expertNum_) + "," + std::to_string(capacity_) + "," +
                                   std::to_string(hidden_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "y", ToString(y).c_str(), expect.c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t tmp = 0;
    if (MulOverflow(expertNum_, samples_, routeNum_) || MulOverflow(samples_, hidden_, tmp) ||
        MulOverflow(expertNum_, capacity_, tmp) || MulOverflow(tmp, hidden_, yElemCnt_)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "y", ToString(y).c_str(),
                                              "element count overflows int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelDispatchTiling::ComputeTiling()
{
    const int64_t coreNum = static_cast<int64_t>(GetAivCoreNum(context_));
    const int64_t ubSize = static_cast<int64_t>(GetUbSize(context_));
    if (coreNum <= 0 || ubSize <= UB_RESERVE_BYTES) {
        OP_LOGE(context_, "invalid platform info: coreNum %ld, ubSize %ld", coreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    // dtypeSize_ 由 CheckDtype 置为 2 或 4；这里兜一层，保证下面的除法不会除零
    if (dtypeSize_ <= 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dtypeSize_).c_str(),
                                              "element size must be 2 or 4 bytes");
        return ge::GRAPH_FAILED;
    }

    const int64_t routeCoreNeed = CeilDiv(routeNum_, MIN_ROUTES_PER_CORE);
    const int64_t initCoreNeed = CeilDiv(yElemCnt_, MIN_ELEMS_PER_CORE);
    usedCoreNum_ = std::max(routeCoreNeed, initCoreNeed);
    usedCoreNum_ = std::max(usedCoreNum_, static_cast<int64_t>(1));
    usedCoreNum_ = std::min(usedCoreNum_, coreNum);

    // 阶段 A：每核清零区间按 32B 对齐，保证起点对齐且区间不重叠
    const int64_t elemsPerBlock = BLOCK_BYTES / dtypeSize_;
    initPerCore_ = CeilAlign(CeilDiv(yElemCnt_, usedCoreNum_), elemsPerBlock);
    // 阶段 B：路由均分
    routePerCore_ = CeilDiv(routeNum_, usedCoreNum_);

    // 阶段 B 的 UB 反推
    const int64_t rowBytes = CeilAlign(hidden_ * dtypeSize_, BLOCK_BYTES);
    const int64_t metaBytesMax = CeilAlign(MAX_TILE_ROUTES * META_BYTES_PER_ROUTE, BLOCK_BYTES);
    const int64_t availBytes = ubSize - UB_RESERVE_BYTES - metaBytesMax;
    if (availBytes < BLOCK_BYTES * BUFFER_NUM) {
        OP_LOGE(context_, "UB size %ld is too small for MoeTutelDispatch", ubSize);
        return ge::GRAPH_FAILED;
    }

    if (rowBytes > 0 && rowBytes * BUFFER_NUM <= availBytes) {
        tileRoutes_ = std::min(availBytes / (rowBytes * BUFFER_NUM), MAX_TILE_ROUTES);
        tileRoutes_ = std::max(tileRoutes_, static_cast<int64_t>(1));
        hChunkElems_ = hidden_;
        SaveTiling(MOE_TUTEL_DISPATCH_TILING_KEY_FULL_ROW);
    } else {
        tileRoutes_ = 1;
        hChunkElems_ = FloorAlign(availBytes / (BUFFER_NUM * dtypeSize_), BLOCK_BYTES / dtypeSize_);
        if (hChunkElems_ <= 0) {
            OP_LOGE(context_, "UB size %ld cannot hold one aligned block", ubSize);
            return ge::GRAPH_FAILED;
        }
        SaveTiling(MOE_TUTEL_DISPATCH_TILING_KEY_SPLIT_H);
    }
    return ge::GRAPH_SUCCESS;
}

void MoeTutelDispatchTiling::SaveTiling(uint64_t tilingKey) const
{
    tilingData_.set_samples(samples_);
    tilingData_.set_hidden(hidden_);
    tilingData_.set_expertNum(expertNum_);
    tilingData_.set_capacity(capacity_);
    tilingData_.set_routeNum(routeNum_);
    tilingData_.set_yElemCnt(yElemCnt_);
    tilingData_.set_initPerCore(initPerCore_);
    tilingData_.set_routePerCore(routePerCore_);
    tilingData_.set_tileRoutes(tileRoutes_);
    tilingData_.set_hChunkElems(hChunkElems_);

    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    OP_LOGD(context_,
            "tilingKey %lu, samples %ld, hidden %ld, expertNum %ld, capacity %ld, routeNum %ld, yElemCnt %ld, "
            "blockDim %ld, initPerCore %ld, routePerCore %ld, tileRoutes %ld, hChunkElems %ld",
            tilingKey, samples_, hidden_, expertNum_, capacity_, routeNum_, yElemCnt_, usedCoreNum_, initPerCore_,
            routePerCore_, tileRoutes_, hChunkElems_);
}

ge::graphStatus MoeTutelDispatchTiling::DoTiling()
{
    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "dtype check failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeRank() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "shape rank check failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseDimsAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "shape/attr check failed"),
                return ge::GRAPH_FAILED);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    if (yElemCnt_ == 0) {
        // y 为空张量：没有任何可写字节，也不存在有效路由（capacity==0 时 0<=loc<0 恒假）
        usedCoreNum_ = 1;
        initPerCore_ = 0;
        routePerCore_ = 0;
        tileRoutes_ = 1;
        hChunkElems_ = 0;
        SaveTiling(MOE_TUTEL_DISPATCH_TILING_KEY_EMPTY);
        return ge::GRAPH_SUCCESS;
    }
    return ComputeTiling();
}

static ge::graphStatus Tiling4MoeTutelDispatch(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    MoeTutelDispatchTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4MoeTutelDispatch(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<MoeTutelDispatchCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->coreNum = static_cast<int64_t>(GetAivCoreNum(context));
    compileInfo->ubSize = static_cast<int64_t>(GetUbSize(context));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTutelDispatch)
    .Tiling(Tiling4MoeTutelDispatch)
    .TilingParse<MoeTutelDispatchCompileInfo>(TilingPrepare4MoeTutelDispatch);
} // namespace optiling
