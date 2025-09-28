/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_tiling_310_general.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_310_general.h"
#include "op_mc2.h"
namespace optiling {
bool MatmulAllReduceTiling310General::IsCapable()
{
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }
    OP_LOGI(opName_, "start with MatmulAllReduceTiling310General tiling.");
    return true;
}

ge::graphStatus MatmulAllReduceTiling310General::DoOpTiling()
{
    OP_LOGI(opName_, "In 310p, isA16W8_[%d], isA16W4_[%d]", isA16W8_ ? 1 : 0, isA16W4_ ? 1 : 0);
    if (isA16W8_ || isA16W4_) {
        // ND场景的伪量化校验
        GE_ASSERT_GRAPH_SUCCESS(CheckA16W8());
    } else {
        // 非量化校验
        OP_LOGI(opName_, "In 310p, check not quant scenario.");
        GE_ASSERT_GRAPH_SUCCESS(CheckA16W16());
    }
    DoRCSTiling();
    DoSplitMTiling();
    if (isKZero_) {
        tilingData_.matmulTiling.set_M(args_.orgMValue);
        tilingData_.matmulTiling.set_isBias(args_.isBias);
        tilingData_.matmulTiling.set_usedCoreNum(1);
        DoAllReduceTiling();
        return ge::GRAPH_SUCCESS;
    }

    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(
        platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mmTile(ascendcPlatform);
    enableL2Cache_ = true;
    if (args_.enableSplitK) {
        DoMatmulTiling310(mmTile, MutableTCubeTileTilingData(), tilingData_.tileL2cacheTiling);
    } else {
        args_.mValue = tileMValue_;
        DoMatmulTiling310(mmTile, MutableTCubeTileTilingData(), tilingData_.tileL2cacheTiling);
        if (MutableRCSTilingData().get_tailCnt() > 0) {
            args_.mValue = tailMValue_;
            matmul_tiling::MultiCoreMatmulTiling mmTail(ascendcPlatform);
            DoMatmulTiling310(mmTail, MutableTCubeTailTilingData(), tilingData_.tailL2cacheTiling);
        }
        args_.mValue = tileMValue_;
    }
    int32_t nAligSize = 16;
    OP_LOGI(context_->GetNodeName(), "DoOpTiling isWeightQuant = %d", isWeightQuant_);
    if (args_.orgNValue % nAligSize != 0 && isWeightQuant_) {
        enableL2Cache_ = false;
    }
    DoAllReduceTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTiling310General::GetTilingKey() const
{
    if (isKZero_) {
        const uint64_t emptyTensorKey = 2100000;
        OP_LOGI(opName_, "MatmulAllReduceTiling310General get tilingKey %lu", emptyTensorKey);
        return emptyTensorKey;
    }
    // L2cache, transB, isWeightQuant, AntiQuantType, hasAntiQuantOffset, 310Version
    const uint64_t tilingKey = RecursiveSum(
        enableL2Cache_, isTransB_, isWeightQuant_, static_cast<int32_t>(antiQuantT_), hasAntiQuantOffset_, isKZero_,
        static_cast<uint64_t>(MatmulAllReduceTiling::ALL_REDUCE_GENERAL_310));
    OP_LOGI(opName_, "MatmulAllReduceTiling310General get tilingKey %lu", tilingKey);
    return tilingKey;
}

void MatmulAllReduceTiling310General::DoMatmulTiling310(
    matmul_tiling::MultiCoreMatmulTiling& mm1, TCubeTiling& cubeTiling, L2cacheTilePara& l2cacheTiling)
{
    DoMatmulTiling(mm1, cubeTiling);
    SetTransLength(mm1, cubeTiling);
    DoL2CacheTiling(l2cacheTiling);
    DoWeightAntiQuantTiling();
}

void MatmulAllReduceTiling310General::DoWeightAntiQuantTiling()
{
    const auto weight = context_->GetInputDesc(static_cast<size_t>(ops::MC2InputIdx::K_X2));
    OP_TILING_CHECK(
        weight == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "Invalid weight tensor."), return );
    if (weight->GetDataType() != ge::DT_INT8) {
        OP_LOGD(context_->GetNodeName(), "No anti quant for weight data type %u.", weight->GetDataType());
        return;
    }

    const auto scale = context_->GetOptionalInputShape(static_cast<size_t>(ops::MC2InputIdx::K_SCALE));
    if (scale == nullptr) {
        OP_LOGD(context_->GetNodeName(), "No anti quant scale.");
        return;
    }
    antiQuantT_ = GetAntiQuantType();
    OP_LOGD(context_->GetNodeName(), "anti quant method %d.", static_cast<int32_t>(antiQuantT_));

    hasAntiQuantOffset_ = HasAntiQuantOffset();
    OP_LOGD(context_->GetNodeName(), "Offset flag %u.", hasAntiQuantOffset_);

    isTransB_ = args_.isBTrans;
    OP_LOGD(context_->GetNodeName(), "Weight trans flag %u.", isTransB_);

    isWeightQuant_ = true;
}

void MatmulAllReduceTiling310General::GetL2CacheParm(
    uint64_t& l2CacheSize, uint64_t& singleMatrixSize, uint32_t& tileSize, uint32_t& tileLimit, bool useNewPara)
{
    (void)useNewPara;
    constexpr uint64_t L2CACHE_SIZE_310 = 16;
    l2CacheSize = L2CACHE_SIZE_310;
    constexpr uint64_t SINGLE_MATRIX_SIZE_310 = 12;
    singleMatrixSize = SINGLE_MATRIX_SIZE_310;
    auto weightTensor = context_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
    OP_TILING_CHECK(
        weightTensor == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "weight tensor is invalid"),
        return );
    if (weightTensor->GetStorageFormat() == ge::Format::FORMAT_FRACTAL_NZ) {
        constexpr uint32_t TILE_SIZE_310_NZ = 5;
        tileSize = TILE_SIZE_310_NZ;
    } else {
        constexpr uint32_t TILE_SIZE_310 = 8;
        tileSize = TILE_SIZE_310;
    }
    OP_LOGI(context_->GetNodeName(), "GetL2CacheParm tileSize = %u", tileSize);
    constexpr uint32_t TILE_LIMIT_310 = 1;
    tileLimit = TILE_LIMIT_310;
}

void MatmulAllReduceTiling310General::SetTransLength(matmul_tiling::MultiCoreMatmulTiling& mm1, TCubeTiling& cubeTiling)
{
    (void)mm1;
    // mdy after api supoort vec nd2nz shareub size cal.
    uint32_t ubTransLen = 128 * 1024;
    cubeTiling.set_transLength(ubTransLen);
    cubeTiling.set_shareUbSize(0);
}
} // namespace optiling
