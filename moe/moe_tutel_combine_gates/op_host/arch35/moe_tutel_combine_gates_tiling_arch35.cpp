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
 * \brief MoeTutelCombineGates Ascend 950 tiling implementation.
 */

#include <algorithm>
#include <limits>
#include <securec.h>
#include <string>

#include "log/log.h"
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
constexpr int64_t INPUT_BUFFER_NUM = 2;
constexpr int64_t CUT_H_X_BUFFER_NUM = 1;
constexpr int64_t CUT_H_Y_BUFFER_NUM = 2;
constexpr int64_t ROUTE_BUFFER_NUM = 2;
constexpr int64_t POWER_OF_TWO_RADIX = 2;
constexpr int64_t TILE_SHRINK_FACTOR = 2;
constexpr int64_t UB_RESERVE_BYTES = 1024;
constexpr int64_t MAX_CHUNK_TREE_BYTES = 4096;
constexpr int64_t MAX_CHUNK_COUNT = MAX_CHUNK_TREE_BYTES / static_cast<int64_t>(sizeof(float));
constexpr int64_t MAX_DATACOPY_BLOCK_LEN = 2097151;
constexpr int64_t S_BLOCK_INIT = 64;
constexpr int64_t K_BLOCK_INIT = 64;

bool CheckedAdd(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedMul(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedCeilDiv(int64_t value, int64_t factor, int64_t &result)
{
    if (value < 0 || factor <= 0) {
        return false;
    }
    result = value / factor + static_cast<int64_t>(value % factor != 0);
    return true;
}

bool CheckedAlignUp(int64_t value, int64_t factor, int64_t &result)
{
    int64_t quotient = 0;
    return CheckedCeilDiv(value, factor, quotient) && CheckedMul(quotient, factor, result);
}

bool CheckedNextPowerOfTwo(int64_t value, int64_t minimum, int64_t &result)
{
    if (value <= 0 || minimum <= 0) {
        return false;
    }
    result = minimum;
    while (result < value) {
        if (result > std::numeric_limits<int64_t>::max() / POWER_OF_TWO_RADIX) {
            return false;
        }
        result *= POWER_OF_TWO_RADIX;
    }
    return true;
}

std::string ShapeValues(int64_t samples, int64_t hidden, int64_t experts, int64_t capacity, int64_t routes)
{
    return "S=" + std::to_string(samples) + ", H=" + std::to_string(hidden) + ", E=" + std::to_string(experts) +
           ", C=" + std::to_string(capacity) + ", K=" + std::to_string(routes);
}
} // namespace

const char *MoeTutelCombineGatesTiling::GetSafeNodeName() const
{
    if (context_ == nullptr) {
        return "MoeTutelCombineGates";
    }
    const char *nodeName = context_->GetNodeName();
    return nodeName == nullptr ? "MoeTutelCombineGates" : nodeName;
}

ge::graphStatus MoeTutelCombineGatesTiling::GetPlatformInfo()
{
    auto compileInfo = static_cast<const MoeTutelCombineGatesCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(GetSafeNodeName(), "compile info is null"), return ge::GRAPH_FAILED);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    workspaceSize_ = compileInfo->workspaceSize;
    vlFp32_ = static_cast<int64_t>(compileInfo->vectorLength) / static_cast<int64_t>(sizeof(float));
    ubBlockSize_ = static_cast<int64_t>(compileInfo->ubBlockSize);

    const bool invalidPlatform =
        coreNum_ == 0 || coreNum_ > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) || ubSize_ == 0 ||
        ubSize_ > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        compileInfo->vectorLength % sizeof(float) != 0 || vlFp32_ <= 0 || ubBlockSize_ <= 0 ||
        (vlFp32_ & (vlFp32_ - 1)) != 0;
    OP_CHECK_IF(
        invalidPlatform,
        OP_LOGE(GetSafeNodeName(), "invalid platform info: coreNum=%lu, ubSize=%lu, vlFp32=%ld, ubBlockSize=%ld",
                coreNum_, ubSize_, vlFp32_, ubBlockSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::CheckFormat() const
{
    const gert::CompileTimeTensorDesc *descs[] = {
        context_->GetInputDesc(MTCG_INPUT_X_INDEX), context_->GetInputDesc(MTCG_INPUT_Y_GRAD_INDEX),
        context_->GetInputDesc(MTCG_INPUT_INDICES_INDEX), context_->GetInputDesc(MTCG_INPUT_LOCATIONS_INDEX),
        context_->GetOutputDesc(MTCG_OUTPUT_GATES_GRAD_INDEX)};
    const char *names[] = {"x", "y_grad", "indices", "locations", "gates_grad"};
    for (size_t i = 0; i < sizeof(descs) / sizeof(descs[0]); ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, descs[i]);
        const ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(descs[i]->GetStorageFormat()));
        OP_CHECK_IF(
            format != ge::FORMAT_ND,
            OP_LOGE_FOR_INVALID_FORMAT(GetSafeNodeName(), names[i], Ops::Base::ToString(format).c_str(), "FORMAT_ND"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::CheckDtype()
{
    auto xDesc = context_->GetInputDesc(MTCG_INPUT_X_INDEX);
    auto yDesc = context_->GetInputDesc(MTCG_INPUT_Y_GRAD_INDEX);
    auto indicesDesc = context_->GetInputDesc(MTCG_INPUT_INDICES_INDEX);
    auto locationsDesc = context_->GetInputDesc(MTCG_INPUT_LOCATIONS_INDEX);
    auto outDesc = context_->GetOutputDesc(MTCG_OUTPUT_GATES_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);

    const ge::DataType xType = xDesc->GetDataType();
    OP_CHECK_IF(xType != ge::DT_FLOAT && xType != ge::DT_FLOAT16 && xType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetSafeNodeName(), "x", Ops::Base::ToString(xType),
                                                      "dtype of x must be float32, float16 or bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDesc->GetDataType() != xType || outDesc->GetDataType() != xType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(GetSafeNodeName(), "x, y_grad, gates_grad",
                                                       Ops::Base::ToString(xType) + ", " +
                                                           Ops::Base::ToString(yDesc->GetDataType()) + ", " +
                                                           Ops::Base::ToString(outDesc->GetDataType()),
                                                       "dtype of x, y_grad and gates_grad must be identical"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT32 || locationsDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(GetSafeNodeName(), "indices, locations",
                                                       Ops::Base::ToString(indicesDesc->GetDataType()) + ", " +
                                                           Ops::Base::ToString(locationsDesc->GetDataType()),
                                                       "dtype of indices and locations must be int32"),
                return ge::GRAPH_FAILED);

    dtypeSize_ = xType == ge::DT_FLOAT ? static_cast<int64_t>(sizeof(float)) : static_cast<int64_t>(sizeof(int16_t));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::CheckRank(const gert::Shape &xShape, const gert::Shape &yShape,
                                                      const gert::Shape &indicesShape,
                                                      const gert::Shape &locationsShape,
                                                      const gert::Shape &outShape) const
{
    OP_CHECK_IF(xShape.GetDimNum() != X_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetSafeNodeName(), "x", std::to_string(xShape.GetDimNum()),
                                                         "rank of x must be 2, in the form of [S, H]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        yShape.GetDimNum() != Y_GRAD_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetSafeNodeName(), "y_grad", std::to_string(yShape.GetDimNum()),
                                                 "rank of y_grad must be 3, in the form of [E, C, H]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape.GetDimNum() != ROUTE_DIM_NUM || locationsShape.GetDimNum() != ROUTE_DIM_NUM ||
                    outShape.GetDimNum() != ROUTE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    GetSafeNodeName(), "indices, locations, gates_grad",
                    std::to_string(indicesShape.GetDimNum()) + ", " + std::to_string(locationsShape.GetDimNum()) +
                        ", " + std::to_string(outShape.GetDimNum()),
                    "rank of indices, locations and gates_grad must all be 2, in the form of [K, S]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::CheckRouteShape(const gert::Shape &indicesShape,
                                                            const gert::Shape &locationsShape,
                                                            const gert::Shape &outShape)
{
    routes_ = indicesShape.GetDim(DIM_0);
    const bool routeShapeOk = indicesShape.GetDim(DIM_1) == samples_ && locationsShape.GetDim(DIM_0) == routes_ &&
                              locationsShape.GetDim(DIM_1) == samples_ && outShape.GetDim(DIM_0) == routes_ &&
                              outShape.GetDim(DIM_1) == samples_;
    OP_CHECK_IF(!routeShapeOk,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    GetSafeNodeName(), "indices, locations, gates_grad",
                    Ops::Base::ToString(indicesShape) + ", " + Ops::Base::ToString(locationsShape) + ", " +
                        Ops::Base::ToString(outShape),
                    "shape of indices, locations and gates_grad must all be [K, S], and S must equal x.shape[0]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::CheckTensorSizes() const
{
    int64_t xElements = 0;
    int64_t expertRows = 0;
    int64_t yElements = 0;
    int64_t routeElements = 0;
    int64_t bytes = 0;
    OP_CHECK_IF(!CheckedMul(samples_, hidden_, xElements),
                OP_LOGE(GetSafeNodeName(), "overflow computing xElements: S=%ld, H=%ld", samples_, hidden_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckedMul(experts_, capacity_, expertRows),
                OP_LOGE(GetSafeNodeName(), "overflow computing expertRows: E=%ld, C=%ld", experts_, capacity_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckedMul(expertRows, hidden_, yElements),
                OP_LOGE(GetSafeNodeName(), "overflow computing yElements: expertRows=%ld, H=%ld", expertRows, hidden_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckedMul(routes_, samples_, routeElements),
                OP_LOGE(GetSafeNodeName(), "overflow computing routeElements: K=%ld, S=%ld", routes_, samples_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckedMul(xElements, dtypeSize_, bytes),
        OP_LOGE(GetSafeNodeName(), "overflow computing xBytes: xElements=%ld, dtypeSize=%ld", xElements, dtypeSize_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckedMul(yElements, dtypeSize_, bytes),
        OP_LOGE(GetSafeNodeName(), "overflow computing yBytes: yElements=%ld, dtypeSize=%ld", yElements, dtypeSize_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckedMul(routeElements, static_cast<int64_t>(sizeof(int32_t)), bytes),
                OP_LOGE(GetSafeNodeName(), "overflow computing routeBytes: routeElements=%ld, dtypeSize=%zu",
                        routeElements, sizeof(int32_t)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckedMul(routeElements, dtypeSize_, bytes),
                OP_LOGE(GetSafeNodeName(), "overflow computing outputBytes: routeElements=%ld, dtypeSize=%ld",
                        routeElements, dtypeSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::GetShapeAndCheck()
{
    auto xShapePtr = context_->GetInputShape(MTCG_INPUT_X_INDEX);
    auto yShapePtr = context_->GetInputShape(MTCG_INPUT_Y_GRAD_INDEX);
    auto indicesShapePtr = context_->GetInputShape(MTCG_INPUT_INDICES_INDEX);
    auto locationsShapePtr = context_->GetInputShape(MTCG_INPUT_LOCATIONS_INDEX);
    auto outShapePtr = context_->GetOutputShape(MTCG_OUTPUT_GATES_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, locationsShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShapePtr);

    const gert::Shape &xShape = xShapePtr->GetStorageShape();
    const gert::Shape &yShape = yShapePtr->GetStorageShape();
    const gert::Shape &indicesShape = indicesShapePtr->GetStorageShape();
    const gert::Shape &locationsShape = locationsShapePtr->GetStorageShape();
    const gert::Shape &outShape = outShapePtr->GetStorageShape();
    if (CheckRank(xShape, yShape, indicesShape, locationsShape, outShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckShapeValues(xShape, yShape, indicesShape, locationsShape, outShape);
}

ge::graphStatus MoeTutelCombineGatesTiling::CheckShapeValues(const gert::Shape &xShape, const gert::Shape &yShape,
                                                             const gert::Shape &indicesShape,
                                                             const gert::Shape &locationsShape,
                                                             const gert::Shape &outShape)
{
    samples_ = xShape.GetDim(DIM_0);
    hidden_ = xShape.GetDim(DIM_1);
    experts_ = yShape.GetDim(DIM_0);
    capacity_ = yShape.GetDim(DIM_1);
    routes_ = indicesShape.GetDim(DIM_0);
    const int64_t yHidden = yShape.GetDim(DIM_2);
    const bool allConcrete = samples_ >= 0 && hidden_ >= 0 && experts_ >= 0 && capacity_ >= 0 && routes_ >= 0 &&
                             yHidden >= 0 && indicesShape.GetDim(DIM_1) >= 0 && locationsShape.GetDim(DIM_0) >= 0 &&
                             locationsShape.GetDim(DIM_1) >= 0 && outShape.GetDim(DIM_0) >= 0 &&
                             outShape.GetDim(DIM_1) >= 0;
    OP_CHECK_IF(!allConcrete,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    GetSafeNodeName(), "x, y_grad, indices, locations, gates_grad",
                    Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape) + ", " +
                        Ops::Base::ToString(indicesShape) + ", " + Ops::Base::ToString(locationsShape) + ", " +
                        Ops::Base::ToString(outShape),
                    "Tiling requires concrete non-negative dimensions"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(experts_ == 0 || capacity_ == 0 || routes_ == 0 || routes_ > experts_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GetSafeNodeName(), "y_grad, indices, locations, gates_grad",
                                                       ShapeValues(samples_, hidden_, experts_, capacity_, routes_),
                                                       "E, C and K must be positive and K must satisfy 1 <= K <= E"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yHidden != hidden_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    GetSafeNodeName(), "x, y_grad", Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape),
                    "x.shape[1] must equal y_grad.shape[2], the contracted hidden axis H"),
                return ge::GRAPH_FAILED);
    if (CheckRouteShape(indicesShape, locationsShape, outShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckTensorSizes();
}

ge::graphStatus MoeTutelCombineGatesTiling::SplitCore()
{
    const int64_t coreNum = static_cast<int64_t>(coreNum_);
    if (samples_ == 0) {
        td_.sCoreNum = 1;
        td_.sPerCoreBase = 0;
        td_.sCoreRemain = 0;
        td_.kPerCoreBase = static_cast<uint64_t>(routes_);
        td_.kCoreRemain = 0;
        blockDim_ = 1;
        return ge::GRAPH_SUCCESS;
    }

    const int64_t maxSCoreNum = std::min(coreNum, samples_);
    int64_t sCoreNum = 1;
    int64_t kCoreNum = std::min(coreNum, routes_);
    int64_t blockDim = kCoreNum;
    int64_t maxSPerCore = 0;
    int64_t maxKPerCore = 0;
    int64_t maxCoreLoad = 0;
    OP_CHECK_IF(!CheckedCeilDiv(samples_, sCoreNum, maxSPerCore) || !CheckedCeilDiv(routes_, kCoreNum, maxKPerCore) ||
                    !CheckedMul(maxSPerCore, maxKPerCore, maxCoreLoad),
                OP_LOGE(GetSafeNodeName(),
                        "overflow computing initial core load: S=%ld, K=%ld, sCoreNum=%ld, "
                        "kCoreNum=%ld",
                        samples_, routes_, sCoreNum, kCoreNum),
                return ge::GRAPH_FAILED);
    for (int64_t candidateSCoreNum = 2; candidateSCoreNum <= maxSCoreNum; ++candidateSCoreNum) {
        const int64_t candidateKCoreNum = std::min(coreNum / candidateSCoreNum, routes_);
        int64_t candidateBlockDim = 0;
        if (!CheckedMul(candidateSCoreNum, candidateKCoreNum, candidateBlockDim)) {
            OP_LOGE(GetSafeNodeName(), "overflow computing candidateBlockDim: sCoreNum=%ld, kCoreNum=%ld",
                    candidateSCoreNum, candidateKCoreNum);
            return ge::GRAPH_FAILED;
        }
        int64_t candidateMaxSPerCore = 0;
        int64_t candidateMaxKPerCore = 0;
        int64_t candidateMaxCoreLoad = 0;
        OP_CHECK_IF(!CheckedCeilDiv(samples_, candidateSCoreNum, candidateMaxSPerCore) ||
                        !CheckedCeilDiv(routes_, candidateKCoreNum, candidateMaxKPerCore) ||
                        !CheckedMul(candidateMaxSPerCore, candidateMaxKPerCore, candidateMaxCoreLoad),
                    OP_LOGE(GetSafeNodeName(),
                            "overflow computing candidate core load: S=%ld, K=%ld, sCoreNum=%ld, kCoreNum=%ld",
                            samples_, routes_, candidateSCoreNum, candidateKCoreNum),
                    return ge::GRAPH_FAILED);
        const bool hasMoreCores = candidateBlockDim > blockDim;
        const bool hasBetterLoad = candidateBlockDim == blockDim && candidateMaxCoreLoad < maxCoreLoad;
        const bool hasBetterReuse =
            candidateBlockDim == blockDim && candidateMaxCoreLoad == maxCoreLoad && candidateSCoreNum > sCoreNum;
        if (hasMoreCores || hasBetterLoad || hasBetterReuse) {
            sCoreNum = candidateSCoreNum;
            kCoreNum = candidateKCoreNum;
            blockDim = candidateBlockDim;
            maxCoreLoad = candidateMaxCoreLoad;
        }
    }
    OP_CHECK_IF(kCoreNum <= 0 || blockDim <= 0 || blockDim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(GetSafeNodeName(), "invalid core split: sCoreNum=%ld, kCoreNum=%ld", sCoreNum, kCoreNum),
                return ge::GRAPH_FAILED);

    td_.sCoreNum = static_cast<uint64_t>(sCoreNum);
    td_.sPerCoreBase = static_cast<uint64_t>(samples_ / sCoreNum);
    td_.sCoreRemain = static_cast<uint64_t>(samples_ % sCoreNum);
    td_.kPerCoreBase = static_cast<uint64_t>(routes_ / kCoreNum);
    td_.kCoreRemain = static_cast<uint64_t>(routes_ % kCoreNum);
    blockDim_ = static_cast<uint32_t>(blockDim);
    return ge::GRAPH_SUCCESS;
}

bool MoeTutelCombineGatesTiling::GetRowBytes(int64_t len, int64_t &bytes) const
{
    int64_t aligned = 0;
    return CheckedAlignUp(len, vlFp32_, aligned) && CheckedMul(aligned, dtypeSize_, bytes);
}

bool MoeTutelCombineGatesTiling::GetTileBytes(int64_t sBlock, int64_t kBlock, int64_t &bytes) const
{
    int64_t sAlign = 0;
    int64_t elements = 0;
    int64_t elementBytes = 0;
    return CheckedAlignUp(sBlock, vlFp32_, sAlign) && CheckedMul(kBlock, sAlign, elements) &&
           CheckedAdd(dtypeSize_, static_cast<int64_t>(sizeof(float)), elementBytes) &&
           CheckedAdd(elementBytes, ROUTE_BUFFER_NUM * static_cast<int64_t>(sizeof(int32_t)), elementBytes) &&
           CheckedMul(elements, elementBytes, bytes);
}

bool MoeTutelCombineGatesTiling::GetTreeLayout(int64_t hidden, int64_t &partialCount, int64_t &partialAlign,
                                               int64_t &bytes) const
{
    if (hidden <= 0 || !CheckedCeilDiv(hidden, vlFp32_, partialCount) ||
        !CheckedNextPowerOfTwo(partialCount, vlFp32_, partialAlign)) {
        return false;
    }
    return CheckedMul(partialAlign, static_cast<int64_t>(sizeof(float)), bytes);
}

bool MoeTutelCombineGatesTiling::TryNotCutH(int64_t avail, int64_t &sBlock, int64_t &kBlock)
{
    int64_t copyBytes = 0;
    int64_t rowBytes = 0;
    int64_t partialCount = 0;
    int64_t partialAlign = 0;
    int64_t treeBytes = 0;
    if (!CheckedMul(hidden_, dtypeSize_, copyBytes) || copyBytes > MAX_DATACOPY_BLOCK_LEN ||
        !GetRowBytes(hidden_, rowBytes) || !GetTreeLayout(hidden_, partialCount, partialAlign, treeBytes) ||
        partialCount > static_cast<int64_t>(std::numeric_limits<uint16_t>::max()) ||
        partialAlign > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    int64_t inputBytes = 0;
    if (!CheckedMul(rowBytes, INPUT_BUFFER_NUM * DOUBLE_BUFFER, inputBytes)) {
        return false;
    }
    while (true) {
        int64_t tileBytes = 0;
        int64_t totalBytes = 0;
        if (GetTileBytes(sBlock, kBlock, tileBytes) && CheckedAdd(inputBytes, tileBytes, totalBytes) &&
            CheckedAdd(totalBytes, treeBytes, totalBytes) && totalBytes <= avail) {
            td_.vlPartialAlign = static_cast<uint64_t>(partialAlign);
            return true;
        }
        if (sBlock > 1) {
            sBlock = std::max<int64_t>(1, sBlock / TILE_SHRINK_FACTOR);
        } else if (kBlock > 1) {
            kBlock = std::max<int64_t>(1, kBlock / TILE_SHRINK_FACTOR);
        } else {
            return false;
        }
    }
}

bool MoeTutelCombineGatesTiling::FindCutHFactor(int64_t budget, int64_t &hFactor, int64_t &partialAlign) const
{
    const int64_t bytesPerElement = dtypeSize_ * INPUT_BUFFER_NUM * DOUBLE_BUFFER;
    const int64_t budgetFactor = budget / bytesPerElement / vlFp32_ * vlFp32_;
    hFactor = std::min(hidden_, budgetFactor) / vlFp32_ * vlFp32_;
    for (; hFactor >= vlFp32_; hFactor -= vlFp32_) {
        int64_t copyBytes = 0;
        int64_t rowBytes = 0;
        int64_t inputBytes = 0;
        int64_t partialCount = 0;
        int64_t treeBytes = 0;
        int64_t totalBytes = 0;
        const bool safe = CheckedMul(hFactor, dtypeSize_, copyBytes) && copyBytes <= MAX_DATACOPY_BLOCK_LEN &&
                          GetRowBytes(hFactor, rowBytes) &&
                          CheckedMul(rowBytes, INPUT_BUFFER_NUM * DOUBLE_BUFFER, inputBytes) &&
                          GetTreeLayout(hFactor, partialCount, partialAlign, treeBytes) &&
                          CheckedAdd(inputBytes, treeBytes, totalBytes) && totalBytes <= budget &&
                          partialCount <= static_cast<int64_t>(std::numeric_limits<uint16_t>::max()) &&
                          partialAlign <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        if (safe) {
            return true;
        }
    }
    return false;
}

ge::graphStatus MoeTutelCombineGatesTiling::SolveCutH(int64_t avail, int64_t &sBlock, int64_t &kBlock)
{
    const int64_t kBlockLimit = kBlock;
    sBlock = 1;
    kBlock = 1;
    int64_t tileBytes = 0;
    int64_t fixedBytes = 0;
    OP_CHECK_IF(!GetTileBytes(sBlock, kBlock, tileBytes) || !CheckedAdd(tileBytes, MAX_CHUNK_TREE_BYTES, fixedBytes) ||
                    fixedBytes >= avail,
                OP_LOGE(GetSafeNodeName(), "UB size %lu is too small for CUT_H output and tree buffers", ubSize_),
                return ge::GRAPH_FAILED);
    const int64_t rowAndTreeBudget = avail - fixedBytes;
    int64_t hFactor = 0;
    int64_t partialAlign = 0;
    OP_CHECK_IF(!FindCutHFactor(rowAndTreeBudget, hFactor, partialAlign),
                OP_LOGE(GetSafeNodeName(), "cannot find a safe CUT_H factor for UB size %lu", ubSize_),
                return ge::GRAPH_FAILED);

    int64_t numChunks = 0;
    OP_CHECK_IF(!CheckedCeilDiv(hidden_, hFactor, numChunks) || numChunks <= 0 || numChunks > MAX_CHUNK_COUNT,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    GetSafeNodeName(), "x", std::to_string(hidden_),
                    "hidden axis H requires too many CUT_H chunks for the deterministic tree buffer"),
                return ge::GRAPH_FAILED);
    int64_t chunkPartialAlign = 0;
    int64_t chunkTreeBytes = 0;
    int64_t vlTreeBytes = 0;
    OP_CHECK_IF(!CheckedNextPowerOfTwo(numChunks, vlFp32_, chunkPartialAlign) ||
                    !CheckedMul(chunkPartialAlign, static_cast<int64_t>(sizeof(float)), chunkTreeBytes) ||
                    !CheckedMul(partialAlign, static_cast<int64_t>(sizeof(float)), vlTreeBytes) ||
                    chunkTreeBytes > MAX_CHUNK_TREE_BYTES,
                OP_LOGE(GetSafeNodeName(), "invalid CUT_H chunk tree layout: chunks=%ld", numChunks),
                return ge::GRAPH_FAILED);
    int64_t usedHidden = 0;
    OP_CHECK_IF(!CheckedMul(numChunks - 1, hFactor, usedHidden) || usedHidden > hidden_,
                OP_LOGE(GetSafeNodeName(), "CUT_H tail calculation overflow"), return ge::GRAPH_FAILED);

    const int64_t candidateKBlock = std::min(kBlockLimit, static_cast<int64_t>(optiling::MTCG_CUT_H_ROUTE_BATCH));
    if (candidateKBlock > 1) {
        int64_t rowBytes = 0;
        int64_t inputBytes = 0;
        int64_t candidateTileBytes = 0;
        int64_t candidateChunkTreeBytes = 0;
        int64_t candidateTotalBytes = 0;
        const bool batchFits = GetRowBytes(hFactor, rowBytes) &&
                               CheckedMul(rowBytes, CUT_H_X_BUFFER_NUM + CUT_H_Y_BUFFER_NUM, inputBytes) &&
                               GetTileBytes(sBlock, candidateKBlock, candidateTileBytes) &&
                               CheckedMul(chunkTreeBytes, candidateKBlock, candidateChunkTreeBytes) &&
                               CheckedAdd(inputBytes, vlTreeBytes, candidateTotalBytes) &&
                               CheckedAdd(candidateTotalBytes, candidateTileBytes, candidateTotalBytes) &&
                               CheckedAdd(candidateTotalBytes, candidateChunkTreeBytes, candidateTotalBytes) &&
                               candidateTotalBytes <= avail;
        if (batchFits) {
            kBlock = candidateKBlock;
        }
    }

    td_.hFactor = static_cast<uint64_t>(hFactor);
    td_.numChunks = static_cast<uint64_t>(numChunks);
    td_.hTail = static_cast<uint64_t>(hidden_ - usedHidden);
    td_.vlPartialAlign = static_cast<uint64_t>(partialAlign);
    td_.chunkPartialAlign = static_cast<uint64_t>(chunkPartialAlign);
    return ge::GRAPH_SUCCESS;
}

void MoeTutelCombineGatesTiling::SetShapeTilingData()
{
    td_.samples = static_cast<uint64_t>(samples_);
    td_.hidden = static_cast<uint64_t>(hidden_);
    td_.experts = static_cast<uint64_t>(experts_);
    td_.capacity = static_cast<uint64_t>(capacity_);
    td_.routes = static_cast<uint64_t>(routes_);
}

ge::graphStatus MoeTutelCombineGatesTiling::SolveZeroH(int64_t avail, int64_t &sBlock, int64_t &kBlock) const
{
    int64_t tileBytes = 0;
    while ((!GetTileBytes(sBlock, kBlock, tileBytes) || tileBytes > avail) && (sBlock > 1 || kBlock > 1)) {
        if (sBlock > 1) {
            sBlock = std::max<int64_t>(1, sBlock / TILE_SHRINK_FACTOR);
        } else {
            kBlock = std::max<int64_t>(1, kBlock / TILE_SHRINK_FACTOR);
        }
    }
    OP_CHECK_IF(!GetTileBytes(sBlock, kBlock, tileBytes) || tileBytes > avail,
                OP_LOGE(GetSafeNodeName(), "UB is too small for H=0 output tile"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::SplitUb()
{
    OP_CHECK_IF(ubSize_ <= static_cast<uint64_t>(UB_RESERVE_BYTES),
                OP_LOGE(GetSafeNodeName(), "UB size %lu is too small", ubSize_), return ge::GRAPH_FAILED);
    const int64_t avail = static_cast<int64_t>(ubSize_) - UB_RESERVE_BYTES;

    SetShapeTilingData();

    if (samples_ == 0) {
        td_.sBlock = 1;
        td_.sBlockAlign = static_cast<uint64_t>(vlFp32_);
        td_.kBlock = 1;
        return ge::GRAPH_SUCCESS;
    }

    const int64_t kBlockMax = static_cast<int64_t>(td_.kPerCoreBase) + static_cast<int64_t>(td_.kCoreRemain > 0);
    int64_t sBlock = std::min(S_BLOCK_INIT, samples_);
    int64_t kBlock = std::min(K_BLOCK_INIT, kBlockMax);
    if (hidden_ == 0) {
        if (SolveZeroH(avail, sBlock, kBlock) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (TryNotCutH(avail, sBlock, kBlock)) {
        td_.hFactor = static_cast<uint64_t>(hidden_);
        td_.numChunks = 1;
        td_.hTail = static_cast<uint64_t>(hidden_);
    } else {
        schMode_ = MTCG_TILING_KEY_CUT_H;
        kBlock = std::min(K_BLOCK_INIT, kBlockMax);
        if (SolveCutH(avail, sBlock, kBlock) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    int64_t sBlockAlign = 0;
    int64_t hFactorAlign = 0;
    int64_t outputElements = 0;
    OP_CHECK_IF(!CheckedAlignUp(sBlock, vlFp32_, sBlockAlign),
                OP_LOGE(GetSafeNodeName(), "overflow computing sBlockAlign: sBlock=%ld, vlFp32=%ld", sBlock, vlFp32_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckedAlignUp(static_cast<int64_t>(td_.hFactor), vlFp32_, hFactorAlign),
        OP_LOGE(GetSafeNodeName(), "overflow computing hFactorAlign: hFactor=%lu, vlFp32=%ld", td_.hFactor, vlFp32_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckedMul(kBlock, sBlockAlign, outputElements),
                OP_LOGE(GetSafeNodeName(), "overflow computing outputElements: kBlock=%ld, sBlockAlign=%ld", kBlock,
                        sBlockAlign),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputElements > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                OP_LOGE(GetSafeNodeName(), "outputElements=%ld exceeds kernel int32 limit=%d", outputElements,
                        std::numeric_limits<int32_t>::max()),
                return ge::GRAPH_FAILED);
    if (schMode_ == MTCG_TILING_KEY_CUT_H) {
        int64_t chunkPartialElements = 0;
        OP_CHECK_IF(
            !CheckedMul(kBlock, static_cast<int64_t>(td_.chunkPartialAlign), chunkPartialElements),
            OP_LOGE(GetSafeNodeName(), "overflow computing chunkPartialElements: kBlock=%ld, chunkPartialAlign=%lu",
                    kBlock, td_.chunkPartialAlign),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(chunkPartialElements > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                    OP_LOGE(GetSafeNodeName(), "chunkPartialElements=%ld exceeds kernel int32 limit=%d",
                            chunkPartialElements, std::numeric_limits<int32_t>::max()),
                    return ge::GRAPH_FAILED);
    }
    td_.sBlock = static_cast<uint64_t>(sBlock);
    td_.sBlockAlign = static_cast<uint64_t>(sBlockAlign);
    td_.kBlock = static_cast<uint64_t>(kBlock);
    td_.hFactorAlign = static_cast<uint64_t>(hFactorAlign);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::PostTiling()
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode_);
    OP_CHECK_IF(tilingKey == INVALID_TILING_KEY,
                OP_LOGE(GetSafeNodeName(), "failed to encode template tiling key for schedule mode %lu", schMode_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS,
                OP_LOGE(GetSafeNodeName(), "set block dim failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(GetSafeNodeName(), "set tiling key failed"), return ge::GRAPH_FAILED);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = static_cast<size_t>(workspaceSize_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(GetSafeNodeName(), "tiling data size %zu exceeds capacity %zu", sizeof(td_),
                        rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    void *data = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, data);
    OP_CHECK_IF(memcpy_s(data, rawTilingData->GetCapacity(), &td_, sizeof(td_)) != EOK,
                OP_LOGE(GetSafeNodeName(), "copy tiling data failed"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));

    OP_LOGI(GetSafeNodeName(),
            "MoeTutelCombineGates tiling: S=%lu H=%lu E=%lu C=%lu K=%lu sCore=%lu sBlock=%lu "
            "kBlock=%lu hFactor=%lu chunks=%lu vlPartialAlign=%lu chunkPartialAlign=%lu key=%lu blockDim=%u",
            td_.samples, td_.hidden, td_.experts, td_.capacity, td_.routes, td_.sCoreNum, td_.sBlock, td_.kBlock,
            td_.hFactor, td_.numChunks, td_.vlPartialAlign, td_.chunkPartialAlign, tilingKey, blockDim_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTutelCombineGatesTiling::DoTiling()
{
    OP_CHECK_IF(memset_s(&td_, sizeof(td_), 0, sizeof(td_)) != EOK,
                OP_LOGE(GetSafeNodeName(), "initialize tiling data failed"), return ge::GRAPH_FAILED);
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS || CheckFormat() != ge::GRAPH_SUCCESS ||
        CheckDtype() != ge::GRAPH_SUCCESS || GetShapeAndCheck() != ge::GRAPH_SUCCESS ||
        SplitCore() != ge::GRAPH_SUCCESS || SplitUb() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return PostTiling();
}

} // namespace optiling
