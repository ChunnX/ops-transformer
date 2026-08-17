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
 * \file moe_tutel_combine_x_tiling.cpp
 * \brief MoeTutelCombineX arch35(Ascend950 / regbase) tiling.
 */
#include "moe_tutel_combine_x_tiling.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Y_GRAD_IDX = 0;
constexpr size_t INPUT_GATES_IDX = 1;
constexpr size_t INPUT_INDICES_IDX = 2;
constexpr size_t INPUT_LOCATIONS_IDX = 3;
constexpr size_t OUTPUT_X_GRAD_IDX = 0;

constexpr size_t Y_GRAD_DIM_NUM = 3;
constexpr size_t ROUTE_DIM_NUM = 2;
constexpr size_t X_GRAD_DIM_NUM = 2;
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;

constexpr int64_t TILING_KEY_MAIN = 10000;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t FP32_SIZE = 4;
constexpr int64_t FP32_PER_BLOCK = 8; // 32B / sizeof(float)
constexpr int64_t ROUTE_ALIGN = 16;   // 路由片行跨距对齐，同时满足 int32(8) 与 bf16/fp16(16) 的 32B 对齐
constexpr int64_t UB_BLOCK_SIZE = 32;
// VF 的 acc/行数据按整个 VREG（256B）加载，尾块会多读最多一个 VREG；预留一段余量保证
// 这段"读越界但不写"的行为始终落在 UB 内。
constexpr int64_t UB_RESERVED = 1024;
constexpr int64_t MAX_ROW_FACTOR = 64;
constexpr int64_t KFACTOR_FIX_LOOP = 16; // CeilAlign(kF,8) 引入的取整回退上限
constexpr int64_t WORKSPACE_RESERVED = 16 * 1024 * 1024;

std::string ShapeToStr(const gert::Shape &shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        oss << (i == 0 ? "" : ",") << shape.GetDim(i);
    }
    oss << "]";
    return oss.str();
}
} // namespace

bool MoeTutelCombineXTiling::IsCapable()
{
    return true;
}

ge::graphStatus MoeTutelCombineXTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    usableUb_ = static_cast<int64_t>(ubSizePlatform) - UB_RESERVED;
    OP_CHECK_IF(totalCoreNum_ <= 0 || usableUb_ <= 0,
                OP_LOGE(context_->GetNodeName(), "Get platform info failed, coreNum=%ld ubSize=%ld.", totalCoreNum_,
                        usableUb_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTiling::CheckDtype()
{
    auto yGradDesc = context_->GetRequiredInputDesc(INPUT_Y_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yGradDesc);
    auto gatesDesc = context_->GetRequiredInputDesc(INPUT_GATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gatesDesc);
    auto indicesDesc = context_->GetRequiredInputDesc(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    auto locationsDesc = context_->GetRequiredInputDesc(INPUT_LOCATIONS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsDesc);
    auto xGradDesc = context_->GetOutputDesc(OUTPUT_X_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xGradDesc);

    dtype_ = yGradDesc->GetDataType();
    OP_CHECK_IF(dtype_ != ge::DT_FLOAT && dtype_ != ge::DT_FLOAT16 && dtype_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y_grad",
                                          ge::TypeUtils::DataTypeToSerialString(dtype_),
                                          "FLOAT32, FLOAT16 or BFLOAT16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gatesDesc->GetDataType() != dtype_,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gates",
                                          ge::TypeUtils::DataTypeToSerialString(gatesDesc->GetDataType()),
                                          ge::TypeUtils::DataTypeToSerialString(dtype_)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xGradDesc->GetDataType() != dtype_,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x_grad",
                                          ge::TypeUtils::DataTypeToSerialString(xGradDesc->GetDataType()),
                                          ge::TypeUtils::DataTypeToSerialString(dtype_)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "indices",
                                          ge::TypeUtils::DataTypeToSerialString(indicesDesc->GetDataType()), "INT32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(locationsDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "locations",
                                          ge::TypeUtils::DataTypeToSerialString(locationsDesc->GetDataType()), "INT32"),
                return ge::GRAPH_FAILED);

    dtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(dtype_));
    OP_CHECK_IF(dtypeSize_ <= 0,
                OP_LOGE(context_->GetNodeName(), "Get size of y_grad dtype failed."),
                return ge::GRAPH_FAILED);
    blockElemT_ = UB_BLOCK_SIZE / dtypeSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTiling::CheckShape()
{
    auto yGradShapePtr = context_->GetRequiredInputShape(INPUT_Y_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yGradShapePtr);
    auto gatesShapePtr = context_->GetRequiredInputShape(INPUT_GATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gatesShapePtr);
    auto indicesShapePtr = context_->GetRequiredInputShape(INPUT_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto locationsShapePtr = context_->GetRequiredInputShape(INPUT_LOCATIONS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsShapePtr);
    auto xGradShapePtr = context_->GetOutputShape(OUTPUT_X_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xGradShapePtr);

    const gert::Shape &yGradShape = yGradShapePtr->GetStorageShape();
    const gert::Shape &gatesShape = gatesShapePtr->GetStorageShape();
    const gert::Shape &indicesShape = indicesShapePtr->GetStorageShape();
    const gert::Shape &locationsShape = locationsShapePtr->GetStorageShape();
    const gert::Shape &xGradShape = xGradShapePtr->GetStorageShape();

    OP_CHECK_IF(yGradShape.GetDimNum() != Y_GRAD_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_grad", ShapeToStr(yGradShape),
                                                      "y_grad must be a 3D tensor of [E, C, H]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gatesShape.GetDimNum() != ROUTE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gates", ShapeToStr(gatesShape),
                                                      "gates must be a 2D tensor of [K, S]"),
                return ge::GRAPH_FAILED);
    // K 取 gates.dim(0)，与同族 MoeTutelDispatch 的口径一致；三张路由表必须严格同 shape，
    // 否则跨 K 归约的下标就没有唯一解释。
    OP_CHECK_IF(indicesShape != gatesShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "indices", ShapeToStr(indicesShape),
                                                      "indices must have the same shape as gates " +
                                                          ShapeToStr(gatesShape)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(locationsShape != gatesShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "locations", ShapeToStr(locationsShape),
                                                      "locations must have the same shape as gates " +
                                                          ShapeToStr(gatesShape)),
                return ge::GRAPH_FAILED);

    expertNum_ = yGradShape.GetDim(DIM_0);
    capacity_ = yGradShape.GetDim(DIM_1);
    hidden_ = yGradShape.GetDim(DIM_2);
    topK_ = gatesShape.GetDim(DIM_0);
    samples_ = gatesShape.GetDim(DIM_1);

    OP_CHECK_IF(expertNum_ < 0 || capacity_ < 0 || hidden_ < 0 || topK_ < 0 || samples_ < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_grad/gates", ShapeToStr(yGradShape),
                                                      "all dims must be non-negative"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xGradShape.GetDimNum() != X_GRAD_DIM_NUM || xGradShape.GetDim(DIM_0) != samples_ ||
                    xGradShape.GetDim(DIM_1) != hidden_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x_grad", ShapeToStr(xGradShape),
                                                      "x_grad must be [gates.dim(1), y_grad.dim(2)]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTiling::GetShapeAttrsInfo()
{
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckShape();
}

int64_t MoeTutelCombineXTiling::CalcUbBytes(int64_t hChunk, int64_t rowF, int64_t kF) const
{
    const int64_t alignT = Ops::Base::CeilAlign(hChunk, blockElemT_);
    const int64_t alignF32 = Ops::Base::CeilAlign(hChunk, FP32_PER_BLOCK);
    const int64_t routeStride = Ops::Base::CeilAlign(rowF, ROUTE_ALIGN);
    const bool isFp32 = (dtypeSize_ == FP32_SIZE);

    // fp32 时累加器直接复用 VECOUT 队列（免一次 UB 搬运），故不再单独占 accBuf。
    const int64_t accBytes = isFp32 ? 0 : rowF * alignF32 * FP32_SIZE;
    const int64_t outBytes = DOUBLE_BUFFER * rowF * alignT * dtypeSize_;
    const int64_t rowBytes = DOUBLE_BUFFER * kF * alignT * dtypeSize_;
    const int64_t gateF32Bytes = isFp32 ? 0 : kF * routeStride * FP32_SIZE;
    const int64_t routeBytes = kF * routeStride * (INT32_SIZE + INT32_SIZE + dtypeSize_) + gateF32Bytes +
                               Ops::Base::CeilAlign(kF, FP32_PER_BLOCK) * FP32_SIZE;
    return accBytes + outBytes + rowBytes + routeBytes;
}

ge::graphStatus MoeTutelCombineXTiling::DoOpTiling()
{
    // 空输出：S==0 或 H==0 时 x_grad 没有元素，直接派 1 个核走空循环，不报错（与 A2 一致）。
    if (samples_ == 0 || hidden_ == 0) {
        usedCoreNum_ = 1;
        rowOfFormerBlock_ = 0;
        rowOfTailBlock_ = 0;
        rowFactor_ = 1;
        kFactor_ = 1;
        hFactor_ = (hidden_ > 0) ? hidden_ : 1;
        return ge::GRAPH_SUCCESS;
    }

    // 核间：按 token 均分，输出区间互不重叠，无原子累加。
    rowOfFormerBlock_ = Ops::Base::CeilDiv(samples_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(samples_, rowOfFormerBlock_);
    rowOfTailBlock_ = samples_ - (usedCoreNum_ - 1) * rowOfFormerBlock_;

    // H 分块：先按最小配置（rowFactor=1, kFactor=1）解出能放下的最大 hFactor，再取 min(hidden)。
    const bool isFp32 = (dtypeSize_ == FP32_SIZE);
    const int64_t bytesPerElemMin =
        isFp32 ? (DOUBLE_BUFFER * FP32_SIZE + DOUBLE_BUFFER * FP32_SIZE) : (FP32_SIZE + DOUBLE_BUFFER * dtypeSize_ + DOUBLE_BUFFER * dtypeSize_);
    const int64_t routeFixed = CalcUbBytes(0, 1, 1);
    int64_t hCap = (usableUb_ - routeFixed) / bytesPerElemMin;
    hCap = Ops::Base::FloorAlign(hCap, blockElemT_);
    hFactor_ = std::min(hidden_, std::max(hCap, blockElemT_));
    while (hFactor_ > blockElemT_ && CalcUbBytes(hFactor_, 1, 1) > usableUb_) {
        hFactor_ -= blockElemT_;
    }
    OP_CHECK_IF(CalcUbBytes(hFactor_, 1, 1) > usableUb_,
                OP_LOGE(context_->GetNodeName(), "UB is too small to hold one hidden block, ubSize=%ld.", usableUb_),
                return ge::GRAPH_FAILED);

    // token 分块：只有 H 整行全载时才批多个 token（H 被切开时批 token 收益有限）。
    // 留一半 UB 给 gather 行缓冲，避免把 kFactor 挤到 1。
    rowFactor_ = 1;
    if (hFactor_ == hidden_) {
        const int64_t maxRow = std::min(MAX_ROW_FACTOR, rowOfFormerBlock_);
        while (rowFactor_ * DOUBLE_BUFFER <= maxRow &&
               CalcUbBytes(hFactor_, rowFactor_ * DOUBLE_BUFFER, 1) * DOUBLE_BUFFER <= usableUb_) {
            rowFactor_ *= DOUBLE_BUFFER;
        }
    }

    // k 分块：用剩余 UB 反推，再按 CeilAlign(kF,8) 的取整误差回退。
    const int64_t alignT = Ops::Base::CeilAlign(hFactor_, blockElemT_);
    const int64_t routeStride = Ops::Base::CeilAlign(rowFactor_, ROUTE_ALIGN);
    const int64_t bytesPerK = DOUBLE_BUFFER * alignT * dtypeSize_ +
                              routeStride * (INT32_SIZE + INT32_SIZE + dtypeSize_ + (isFp32 ? 0 : FP32_SIZE)) +
                              FP32_SIZE;
    const int64_t maxK = std::max<int64_t>(topK_, 1);
    kFactor_ = (usableUb_ - CalcUbBytes(hFactor_, rowFactor_, 0)) / bytesPerK;
    kFactor_ = std::min(std::max<int64_t>(kFactor_, 1), maxK);
    for (int64_t i = 0; i < KFACTOR_FIX_LOOP && kFactor_ > 1; i++) {
        if (CalcUbBytes(hFactor_, rowFactor_, kFactor_) <= usableUb_) {
            break;
        }
        kFactor_--;
    }
    OP_CHECK_IF(CalcUbBytes(hFactor_, rowFactor_, kFactor_) > usableUb_,
                OP_LOGE(context_->GetNodeName(), "Solve UB tiling failed, ubSize=%ld hFactor=%ld rowFactor=%ld.",
                        usableUb_, hFactor_, rowFactor_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeTutelCombineXTiling::GetTilingKey() const
{
    // dtype 不进 tiling key，靠 kernel 侧 DTYPE_Y_GRAD 模板参数区分。
    return static_cast<uint64_t>(TILING_KEY_MAIN);
}

ge::graphStatus MoeTutelCombineXTiling::GetWorkspaceSize()
{
    // 本算子按 token 独占输出，无需用户 workspace，只留系统 workspace。
    workspaceSize_ = WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineXTiling::PostTiling()
{
    tilingData_.set_capacity(capacity_);
    tilingData_.set_hidden(hidden_);
    tilingData_.set_topK(topK_);
    tilingData_.set_samples(samples_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_kFactor(kFactor_);
    tilingData_.set_hFactor(hFactor_);
    tilingData_.set_hChunkAlignT(Ops::Base::CeilAlign(hFactor_, blockElemT_));
    tilingData_.set_hChunkAlignF32(Ops::Base::CeilAlign(hFactor_, FP32_PER_BLOCK));
    tilingData_.set_routeStride(Ops::Base::CeilAlign(rowFactor_, ROUTE_ALIGN));

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeTutelCombineXTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "E: " << expertNum_ << ", C: " << capacity_ << ", H: " << hidden_ << ", K: " << topK_
         << ", S: " << samples_ << ", usedCoreNum: " << usedCoreNum_ << ", rowOfFormerBlock: " << rowOfFormerBlock_
         << ", rowOfTailBlock: " << rowOfTailBlock_ << ", rowFactor: " << rowFactor_ << ", kFactor: " << kFactor_
         << ", hFactor: " << hFactor_ << ", ubBytes: " << CalcUbBytes(hFactor_, rowFactor_, kFactor_)
         << ", tilingKey: " << GetTilingKey();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

static ge::graphStatus Tiling4MoeTutelCombineX(gert::TilingContext *context)
{
    MoeTutelCombineXTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4MoeTutelCombineX(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTutelCombineX)
    .Tiling(Tiling4MoeTutelCombineX)
    .TilingParse<MoeTutelCombineXCompileInfo>(TilingPrepare4MoeTutelCombineX);
} // namespace optiling
