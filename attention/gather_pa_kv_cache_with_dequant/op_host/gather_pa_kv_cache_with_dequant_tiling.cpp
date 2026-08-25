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
 * \file gather_pa_kv_cache_with_dequant_tiling.cpp
 * \brief Tiling implementation for fused GatherPaKvCache + Dequant
 */

#include <cmath>
#include <array>
#include "op_common/log/log.h"
#include "platform/platform_info_def.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "gather_pa_kv_cache_with_dequant_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr int64_t INDEX_ATTR_CACHE_MODE = 0;
constexpr int64_t INDEX_ATTR_IS_SEQ_LENS_CUMSUM = 1;
constexpr int64_t INDEX_INPUT_KEY_CACHE = 0;
constexpr int64_t INDEX_INPUT_KEY_SCALE_CACHE = 1;
constexpr int64_t INDEX_INPUT_VALUE_CACHE = 2;
constexpr int64_t INDEX_INPUT_VALUE_SCALE = 3;
constexpr int64_t INDEX_INPUT_BLOCK_TABLES = 4;
constexpr int64_t INDEX_INPUT_SEQ_LENS = 5;
constexpr int64_t INDEX_INPUT_KEY = 6;
constexpr int64_t INDEX_INPUT_VALUE = 7;
constexpr int64_t INDEX_OPT_INPUT_SEQ_OFFSETS = 8;
constexpr int64_t INDEX_OUTPUT_KEY = 0;
constexpr int64_t INDEX_OUTPUT_VALUE = 1;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t WORKSPACE_SIZE = 32;
constexpr uint32_t UB_REVERSE = 1024;
constexpr uint32_t SEQ_LEN_ACC_SPACE = 1024;
constexpr uint32_t DOUBLE_BUFFER = 2;

static const std::map<ge::DataType, uint32_t> tilingDataTypeByteTable = {
    {ge::DT_INT64, 8}, {ge::DT_INT32, 4}, {ge::DT_UINT32, 4}, {ge::DT_FLOAT, 4},
    {ge::DT_INT16, 2}, {ge::DT_UINT16, 2}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2},
    {ge::DT_INT8, 1}, {ge::DT_UINT8, 1}, {ge::DT_HIFLOAT8, 1},
    {ge::DT_FLOAT8_E5M2, 1}, {ge::DT_FLOAT8_E4M3FN, 1},
};

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const GatherPaKvCacheWithDequantCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetAttrs()
{
    auto *attrs = context_->GetAttrs();
    
    const bool *isSeqLensCumsumPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_IS_SEQ_LENS_CUMSUM);
    if (isSeqLensCumsumPtr == nullptr) {
        isSeqLenCumSum_ = true;
    } else {
        isSeqLenCumSum_ = *isSeqLensCumsumPtr;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetInputKeyCache()
{
    auto kCacheDesc = context_->GetInputDesc(INDEX_INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheDesc);
    
    auto kCacheShape = context_->GetInputShape(INDEX_INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheShape);
    
    if (kCacheShape->GetStorageShape().GetDimNum() != DIM_FOUR) {
        OP_LOGE(context_, "key_cache dimension must be 4, but got %zu", kCacheShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    
    numBlocks_ = kCacheShape->GetStorageShape().GetDim(0);
    blockSize_ = kCacheShape->GetStorageShape().GetDim(1);
    numHeadsK_ = kCacheShape->GetStorageShape().GetDim(2);
    headDimK_ = kCacheShape->GetStorageShape().GetDim(3);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetInputValueCache()
{
    auto vCacheDesc = context_->GetInputDesc(INDEX_INPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheDesc);
    
    auto vCacheShape = context_->GetInputShape(INDEX_INPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheShape);
    
    if (vCacheShape->GetStorageShape().GetDimNum() != DIM_FOUR) {
        OP_LOGE(context_, "value_cache dimension must be 4, but got %zu", vCacheShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    
    numHeadsV_ = vCacheShape->GetStorageShape().GetDim(2);
    headDimV_ = vCacheShape->GetStorageShape().GetDim(3);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetInputBlockTables()
{
    auto blockTablesShape = context_->GetInputShape(INDEX_INPUT_BLOCK_TABLES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockTablesShape);
    
    if (blockTablesShape->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE(context_, "block_tables dimension must be 2, but got %zu", blockTablesShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    
    batchCount_ = blockTablesShape->GetStorageShape().GetDim(0);
    blockTableWidth_ = blockTablesShape->GetStorageShape().GetDim(1);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetInputSeqLens()
{
    auto seqLensShape = context_->GetInputShape(INDEX_INPUT_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLensShape);
    
    if (seqLensShape->GetStorageShape().GetDimNum() != DIM_ONE) {
        OP_LOGE(context_, "seq_lens dimension must be 1, but got %zu", seqLensShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetInputOutputKey()
{
    auto keyShape = context_->GetInputShape(INDEX_INPUT_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    
    if (keyShape->GetStorageShape().GetDimNum() != DIM_THREE) {
        OP_LOGE(context_, "key dimension must be 3, but got %zu", keyShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    
    numTokens_ = keyShape->GetStorageShape().GetDim(0);
    hiddenSizeK_ = keyShape->GetStorageShape().GetDim(1) * keyShape->GetStorageShape().GetDim(2);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetInputOutputValue()
{
    auto valueShape = context_->GetInputShape(INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueShape);
    
    if (valueShape->GetStorageShape().GetDimNum() != DIM_THREE) {
        OP_LOGE(context_, "value dimension must be 3, but got %zu", valueShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    
    hiddenSizeV_ = valueShape->GetStorageShape().GetDim(1) * valueShape->GetStorageShape().GetDim(2);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetInputSeqOffset()
{
    auto seqOffsetDesc = context_->GetOptionalInputDesc(INDEX_OPT_INPUT_SEQ_OFFSETS);
    if (seqOffsetDesc == nullptr) {
        hasSeqOffset_ = false;
        return ge::GRAPH_SUCCESS;
    }
    hasSeqOffset_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("GatherPaKvCacheWithDequantTiling", "context is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get attrs failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputKeyCache() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input key_cache failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputValueCache() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input value_cache failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputBlockTables() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input block_tables failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputSeqLens() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input seq_lens failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputOutputKey() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input/output key failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputOutputValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input/output value failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputSeqOffset() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get optional input seq_offset failed."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

bool GatherPaKvCacheWithDequantTiling::IsCapable()
{
    return true;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::DoOpTiling()
{
    int64_t batchPerCore = Ops::Base::CeilDiv(batchCount_, coreNum_);
    needCoreNum_ = static_cast<uint32_t>(std::min(Ops::Base::CeilDiv(batchCount_, batchPerCore), coreNum_));
    
    uint32_t tileBase = BLOCK_SIZE;
    uint32_t seqLenAccumSize = 1024;
    
    // Calculate UB buffer sizes
    // Key: INT8 (1 byte per element), Value: FP8 (1 byte per element)
    uint64_t cacheBlockK = static_cast<uint64_t>(blockSize_) * hiddenSizeK_;
    uint64_t cacheBlockV = static_cast<uint64_t>(blockSize_) * hiddenSizeV_;
    uint32_t indexByteSize = 4; // INT32
    
    uint32_t factor = (ubSize_ - UB_REVERSE - seqLenAccumSize * DOUBLE_BUFFER * indexByteSize - BLOCK_SIZE) /
                      (tileBase * DOUBLE_BUFFER);
    
    uint64_t maxUbHiddenSizeK = std::min(static_cast<uint64_t>(factor) * tileBase, cacheBlockK);
    uint64_t maxUbHiddenSizeV = std::min(static_cast<uint64_t>(factor) * tileBase, cacheBlockV);
    uint64_t maxUbHiddenSize = std::max(maxUbHiddenSizeK, maxUbHiddenSizeV);
    maxUbHiddenSize = Ops::Base::CeilAlign(maxUbHiddenSize, static_cast<uint64_t>(tileBase));
    
    if (maxUbHiddenSizeK == cacheBlockK || maxUbHiddenSizeV == cacheBlockV) {
        uint32_t spareBuffer = ubSize_ - UB_REVERSE - maxUbHiddenSize * DOUBLE_BUFFER - BLOCK_SIZE;
        seqLenAccumSize = Ops::Base::CeilDiv(spareBuffer / DOUBLE_BUFFER, BLOCK_SIZE) * BLOCK_SIZE / indexByteSize;
    }
    
    // Configure tiling data
    tilingData_.set_batchCount(batchCount_);
    tilingData_.set_batchPerCore(batchPerCore);
    tilingData_.set_needCoreNum(needCoreNum_);
    tilingData_.set_seqLenAccumSize(seqLenAccumSize);
    tilingData_.set_blockTableWidth(blockTableWidth_);
    tilingData_.set_numBlocks(numBlocks_);
    tilingData_.set_hiddenSizeK(hiddenSizeK_);
    tilingData_.set_hiddenSizeV(hiddenSizeV_);
    tilingData_.set_numTokens(numTokens_);
    tilingData_.set_maxUbHiddenSizeK(maxUbHiddenSizeK);
    tilingData_.set_maxUbHiddenSizeV(maxUbHiddenSizeV);
    tilingData_.set_maxUbHiddenSize(maxUbHiddenSize);
    tilingData_.set_kvCacheBlockSize(blockSize_);
    tilingData_.set_numHeadsK(numHeadsK_);
    tilingData_.set_numHeadsV(numHeadsV_);
    tilingData_.set_headDimK(headDimK_);
    tilingData_.set_headDimV(headDimV_);
    
    // Set tiling key based on attributes
    for (const auto &item : tilingKeyTable) {
        if (item.isSeqLenCumSum == isSeqLenCumSum_ && item.hasSeqOffset == hasSeqOffset_) {
            tilingKey_ = item.tilingKey;
            break;
        }
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GatherPaKvCacheWithDequantTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheWithDequantTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(needCoreNum_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGatherPaKvCacheWithDequant(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TilingForGatherPaKvCacheWithDequant", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingForGatherPaKvCacheWithDequant enter.");
    
    GatherPaKvCacheWithDequantTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForGatherPaKvCacheWithDequant(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForGatherPaKvCacheWithDequant", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForGatherPaKvCacheWithDequant enter");
    
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GatherPaKvCacheWithDequant)
    .Tiling(TilingForGatherPaKvCacheWithDequant)
    .TilingParse<GatherPaKvCacheWithDequantCompileInfo>(TilingPrepareForGatherPaKvCacheWithDequant);

} // namespace optiling
