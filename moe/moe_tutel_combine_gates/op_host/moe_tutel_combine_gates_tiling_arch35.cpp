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
 * \file moe_tutel_combine_gates_tiling_arch35.cpp
 * \brief MoeTutelCombineGates arch35（Ascend950）tiling 实现
 */

#include <algorithm>
#include <securec.h>
#include "moe_tutel_combine_gates_tiling.h"

using namespace ge;

namespace optiling {
namespace {
constexpr size_t X_DIM_NUM = 2;
constexpr size_t Y_GRAD_DIM_NUM = 3;
constexpr size_t ROUTE_DIM_NUM = 2;
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t IN_BUFFER_NUM = 2; // x 与 y 各一条输入流水
constexpr int64_t UB_RESERVE_BYTES = 1024;
// CUT_H 的部分和缓冲上限：4KB / 4B = 1024 个分块。按最小 hFactor 折算，H 上限远超实际使用范围。
constexpr int64_t PARTIAL_RESERVE_BYTES = 4096;
constexpr int64_t MAX_NUM_CHUNKS = PARTIAL_RESERVE_BYTES / static_cast<int64_t>(sizeof(float));
constexpr int64_t S_BLOCK_INIT = 64;
constexpr size_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

inline int64_t CeilAlign(int64_t value, int64_t factor)
{
    return (factor <= 0) ? value : ((value + factor - 1) / factor * factor);
}

inline int64_t CeilDiv(int64_t value, int64_t factor)
{
    return (factor <= 0) ? 0 : ((value + factor - 1) / factor);
}
} // namespace

ge::graphStatus MoeTutelCombineGatesTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MoeTutelCombineGatesCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    vlFp32_ = static_cast<int64_t>(compileInfo->vectorLength) / static_cast<int64_t>(sizeof(float));
    ubBlockSize_ = static_cast<int64_t>(compileInfo->ubBlockSize);

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }
    OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0 || vlFp32_ <= 0 || ubBlockSize_ <= 0,
                OP_LOGE(context_->GetNodeName(),
                        "invalid platform info: coreNum %lu, ubSize %lu, vlFp32 %ld, ubBlockSize %ld", coreNum_,
                        ubSize_, vlFp32_, ubBlockSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::CheckDtype()
{
    auto xDesc = context_->GetInputDesc(MTCG_INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto yDesc = context_->GetInputDesc(MTCG_INPUT_Y_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto indicesDesc = context_->GetInputDesc(MTCG_INPUT_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    auto locationsDesc = context_->GetInputDesc(MTCG_INPUT_LOCATIONS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsDesc);
    auto outDesc = context_->GetOutputDesc(MTCG_OUTPUT_GATES_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);

    const ge::DataType xType = xDesc->GetDataType();
    // 三个浮点张量强制同 dtype：canndev OpDef 只铺同 dtype 组合，950 不擅自开放混合 dtype。
    OP_CHECK_IF(xType != ge::DT_FLOAT && xType != ge::DT_FLOAT16 && xType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xType),
                                                      "dtype of x must be float32, float16 or bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDesc->GetDataType() != xType || outDesc->GetDataType() != xType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, y_grad, gates_grad",
                    Ops::Base::ToString(xType) + ", " + Ops::Base::ToString(yDesc->GetDataType()) + ", " +
                        Ops::Base::ToString(outDesc->GetDataType()),
                    "dtype of x, y_grad and gates_grad must be identical"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT32 || locationsDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "indices, locations",
                    Ops::Base::ToString(indicesDesc->GetDataType()) + ", " +
                        Ops::Base::ToString(locationsDesc->GetDataType()),
                    "dtype of indices and locations must be int32"),
                return ge::GRAPH_FAILED);

    dtypeSize_ = (xType == ge::DT_FLOAT) ? static_cast<int64_t>(sizeof(float)) : static_cast<int64_t>(sizeof(int16_t));
    return ge::GRAPH_SUCCESS;
}

// 校验各输入的 rank。抽出来是为了让 GetShapeAndCheck 保持在 CodeCheck 的 50 行阈值内。
ge::graphStatus MoeTutelCombineGatesTiling::CheckRank(const gert::Shape &xShape, const gert::Shape &yShape,
                                                      const gert::Shape &indicesShape,
                                                      const gert::Shape &locationsShape,
                                                      const gert::Shape &outShape) const
{
    OP_CHECK_IF(xShape.GetDimNum() != X_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                         std::to_string(xShape.GetDimNum()),
                                                         "rank of x must be 2, in the form of [S, H]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape.GetDimNum() != Y_GRAD_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "y_grad",
                                                         std::to_string(yShape.GetDimNum()),
                                                         "rank of y_grad must be 3, in the form of [E, C, H]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape.GetDimNum() != ROUTE_DIM_NUM || locationsShape.GetDimNum() != ROUTE_DIM_NUM ||
                    outShape.GetDimNum() != ROUTE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "indices, locations, gates_grad",
                    std::to_string(indicesShape.GetDimNum()) + ", " + std::to_string(locationsShape.GetDimNum()) +
                        ", " + std::to_string(outShape.GetDimNum()),
                    "rank of indices, locations and gates_grad must all be 2, in the form of [E, S]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验路由张量与输出的形状：三者必须同为 [E, S]。
// canndev 的 taskNum = samples * y_grad.shape[0]，GM buffer 也按 S*E 设定，从不看 indices 的实际 shape；
// K != E 在旧实现上是对 gates_grad 的真实越界写，950 判非法而不复制这个未定义行为。
ge::graphStatus MoeTutelCombineGatesTiling::CheckRouteShape(const gert::Shape &indicesShape,
                                                            const gert::Shape &locationsShape,
                                                            const gert::Shape &outShape) const
{
    const bool routeShapeOk = (indicesShape.GetDim(DIM_0) == experts_) && (indicesShape.GetDim(DIM_1) == samples_) &&
                              (locationsShape.GetDim(DIM_0) == experts_) &&
                              (locationsShape.GetDim(DIM_1) == samples_) && (outShape.GetDim(DIM_0) == experts_) &&
                              (outShape.GetDim(DIM_1) == samples_);
    OP_CHECK_IF(!routeShapeOk,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "indices, locations, gates_grad",
                    Ops::Base::ToString(indicesShape) + ", " + Ops::Base::ToString(locationsShape) + ", " +
                        Ops::Base::ToString(outShape),
                    "shape of indices, locations and gates_grad must all be [E, S] = [" + std::to_string(experts_) +
                        ", " + std::to_string(samples_) + "], where E is y_grad.shape[0] and S is x.shape[0]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::GetShapeAndCheck()
{
    auto xShapePtr = context_->GetInputShape(MTCG_INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto yShapePtr = context_->GetInputShape(MTCG_INPUT_Y_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto indicesShapePtr = context_->GetInputShape(MTCG_INPUT_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto locationsShapePtr = context_->GetInputShape(MTCG_INPUT_LOCATIONS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsShapePtr);
    auto outShapePtr = context_->GetOutputShape(MTCG_OUTPUT_GATES_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShapePtr);

    const gert::Shape &xShape = xShapePtr->GetStorageShape();
    const gert::Shape &yShape = yShapePtr->GetStorageShape();
    const gert::Shape &indicesShape = indicesShapePtr->GetStorageShape();
    const gert::Shape &locationsShape = locationsShapePtr->GetStorageShape();
    const gert::Shape &outShape = outShapePtr->GetStorageShape();

    if (CheckRank(xShape, yShape, indicesShape, locationsShape, outShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    samples_ = xShape.GetDim(DIM_0);
    hidden_ = xShape.GetDim(DIM_1);
    experts_ = yShape.GetDim(DIM_0);
    capacity_ = yShape.GetDim(DIM_1);

    OP_CHECK_IF(samples_ < 0 || hidden_ < 0 || experts_ < 0 || capacity_ < 0 || yShape.GetDim(DIM_2) < 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "x, y_grad",
                    Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape),
                    "every dim of x and y_grad must be non-negative"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape.GetDim(DIM_2) != hidden_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "x, y_grad",
                    Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape),
                    "x.shape[1] must equal y_grad.shape[2], they are the contracted hidden axis H"),
                return ge::GRAPH_FAILED);

    return CheckRouteShape(indicesShape, locationsShape, outShape);
}

ge::graphStatus MoeTutelCombineGatesTiling::SplitCore()
{
    const int64_t coreNum = static_cast<int64_t>(coreNum_);
    // S、E 可能为 0（空 Tensor），此处只保证核数至少为 1，实际循环长度由 perCoreBase/remain 决定。
    const int64_t sCoreNum = std::max<int64_t>(1, std::min<int64_t>(coreNum, std::max<int64_t>(1, samples_)));
    const int64_t kCoreNum =
        std::max<int64_t>(1, std::min<int64_t>(coreNum / sCoreNum, std::max<int64_t>(1, experts_)));

    td_.sCoreNum = static_cast<uint64_t>(sCoreNum);
    td_.sPerCoreBase = static_cast<uint64_t>(samples_ / sCoreNum);
    td_.sCoreRemain = static_cast<uint64_t>(samples_ % sCoreNum);
    td_.kPerCoreBase = static_cast<uint64_t>(experts_ / kCoreNum);
    td_.kCoreRemain = static_cast<uint64_t>(experts_ % kCoreNum);
    blockDim_ = static_cast<uint32_t>(sCoreNum * kCoreNum);
    return ge::GRAPH_SUCCESS;
}

// UB 行长：H（或 H 分块）按 VL(fp32) 向上对齐后的字节数。
int64_t MoeTutelCombineGatesTiling::RowBytesOf(int64_t len) const
{
    return CeilAlign(len, vlFp32_) * dtypeSize_;
}

// 输出 tile 的字节数。每个元素同时占一份 fp32 中间结果与一份输出 dtype。
// tile 每行按 VL(fp32) 对齐：CastRowVF 里的 LoadOneTensorForDtypeT 会整段读满一个 VREG，
// 只按 32B 对齐时最后一行会读到缓冲区之外。VL 对齐同时满足 DataCopyPad 对 UB 源地址的 32B 要求。
int64_t MoeTutelCombineGatesTiling::TileBytesOf(int64_t sBlock, int64_t kBlock) const
{
    return kBlock * CeilAlign(sBlock, vlFp32_) * (dtypeSize_ + static_cast<int64_t>(sizeof(float)));
}

bool MoeTutelCombineGatesTiling::TryNotCutH(int64_t avail, int64_t &sBlock, int64_t &kBlock) const
{
    while (true) {
        if (RowBytesOf(hidden_) * IN_BUFFER_NUM * DOUBLE_BUFFER + TileBytesOf(sBlock, kBlock) <= avail) {
            return true;
        }
        if (sBlock > 1) {
            sBlock = std::max<int64_t>(1, sBlock / 2);
            continue;
        }
        if (kBlock > 1) {
            kBlock = std::max<int64_t>(1, kBlock / 2);
            continue;
        }
        return false;
    }
}

ge::graphStatus MoeTutelCombineGatesTiling::SolveCutH(int64_t avail, int64_t &sBlock, int64_t &kBlock)
{
    sBlock = 1;
    kBlock = 1;
    const int64_t budget = avail - TileBytesOf(sBlock, kBlock) - PARTIAL_RESERVE_BYTES;
    OP_CHECK_IF(budget < vlFp32_ * dtypeSize_ * IN_BUFFER_NUM * DOUBLE_BUFFER,
                OP_LOGE(context_->GetNodeName(), "ub size %lu is too small for cut-h template", ubSize_),
                return ge::GRAPH_FAILED);
    int64_t hFactor = budget / (IN_BUFFER_NUM * DOUBLE_BUFFER * dtypeSize_) / vlFp32_ * vlFp32_;
    hFactor = std::max<int64_t>(vlFp32_, hFactor);
    const int64_t numChunks = CeilDiv(hidden_, hFactor);
    OP_CHECK_IF(numChunks > MAX_NUM_CHUNKS,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(hidden_),
                    "hidden axis H is too large: it would need " + std::to_string(numChunks) +
                        " chunks, exceeding the supported limit " + std::to_string(MAX_NUM_CHUNKS)),
                return ge::GRAPH_FAILED);
    td_.hFactor = static_cast<uint64_t>(hFactor);
    td_.numChunks = static_cast<uint64_t>(numChunks);
    td_.hTail = static_cast<uint64_t>(hidden_ - (numChunks - 1) * hFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::SplitUb()
{
    const int64_t avail = static_cast<int64_t>(ubSize_) - UB_RESERVE_BYTES;
    OP_CHECK_IF(avail <= 0, OP_LOGE(context_->GetNodeName(), "ub size %lu is too small", ubSize_),
                return ge::GRAPH_FAILED);

    const int64_t kBlockMax = static_cast<int64_t>(td_.kPerCoreBase) + (td_.kCoreRemain > 0 ? 1 : 0);
    int64_t sBlock = std::max<int64_t>(1, std::min<int64_t>(S_BLOCK_INIT, std::max<int64_t>(1, samples_)));
    int64_t kBlock = std::max<int64_t>(1, kBlockMax);

    if (TryNotCutH(avail, sBlock, kBlock)) {
        tilingKey_ = MTCG_TILING_KEY_NOT_CUT_H;
        td_.hFactor = static_cast<uint64_t>(hidden_);
        td_.numChunks = 1;
        td_.hTail = static_cast<uint64_t>(hidden_);
    } else if (SolveCutH(avail, sBlock, kBlock) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    } else {
        tilingKey_ = MTCG_TILING_KEY_CUT_H;
    }

    td_.samples = static_cast<uint64_t>(samples_);
    td_.hidden = static_cast<uint64_t>(hidden_);
    td_.experts = static_cast<uint64_t>(experts_);
    td_.capacity = static_cast<uint64_t>(capacity_);
    td_.sBlock = static_cast<uint64_t>(sBlock);
    td_.sBlockAlign = static_cast<uint64_t>(CeilAlign(sBlock, vlFp32_));
    td_.kBlock = static_cast<uint64_t>(kBlock);
    td_.hFactorAlign = static_cast<uint64_t>(CeilAlign(static_cast<int64_t>(td_.hFactor), vlFp32_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::PostTiling()
{
    context_->SetBlockDim(blockDim_);
    context_->SetTilingKey(tilingKey_);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    // 本算子无跨核归约，不需要用户 workspace；只保留框架要求的系统 workspace。
    workspaces[0] = SYS_WORKSPACE_SIZE;

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "tiling data size %zu exceeds capacity %zu", sizeof(td_),
                        rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    void *ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    OP_CHECK_IF(memcpy_s(ptrData, rawTilingData->GetCapacity(), &td_, sizeof(td_)) != EOK,
                OP_LOGE(context_->GetNodeName(), "copy tiling data failed"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));

    OP_LOGI(context_->GetNodeName(),
            "MoeTutelCombineGates tiling: S=%lu H=%lu E=%lu C=%lu sCoreNum=%lu sBlock=%lu "
            "sBlockAlign=%lu kBlock=%lu hFactor=%lu hFactorAlign=%lu numChunks=%lu hTail=%lu tilingKey=%lu "
            "blockDim=%u",
            td_.samples, td_.hidden, td_.experts, td_.capacity, td_.sCoreNum, td_.sBlock,
            td_.sBlockAlign, td_.kBlock, td_.hFactor, td_.hFactorAlign, td_.numChunks, td_.hTail, tilingKey_,
            blockDim_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::DoTiling()
{
    OP_CHECK_IF(memset_s(&td_, sizeof(td_), 0, sizeof(td_)) != EOK,
                OP_LOGE(context_->GetNodeName(), "init tiling data failed"), return ge::GRAPH_FAILED);
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SplitCore() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SplitUb() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return PostTiling();
}

} // namespace optiling
