/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_prepare_vector_s_tile_tiling.h"

#include <vector>

#include "log/log.h"

namespace {
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = -100;
} // namespace

namespace optiling {
namespace {
constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint64_t BF16_BYTES = sizeof(uint16_t);
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t REG_ELEMS = 64UL;
constexpr uint64_t STAT_UB_ELEMS = 2UL * (REG_ELEMS + 8UL);
constexpr uint64_t STAT_BUFFER_NUM = 4UL;
constexpr uint64_t NORM_CACHE_BYTES = REG_ELEMS * FP32_BYTES;
constexpr uint64_t Q_BUFFER_NUM = 4UL;
constexpr uint64_t O_BUFFER_NUM = 2UL;
} // namespace

bool BlockAttnResPrepareVectorSTileTiling::IsCapable()
{
    if (!inputBf16_ || totalS_ < 8U || totalT_ < aivCoreNum_ / 2U || totalD_ < 256U ||
        totalD_ % REG_ELEMS != 0U || ubSize_ <= UB_RESERVED_BYTES) {
        return false;
    }
    const uint64_t dAlign = AlignUp<uint64_t>(totalD_, REG_ELEMS);
    const uint64_t vBytes = totalN_ * dAlign * BF16_BYTES;
    const uint64_t qBytes = Q_BUFFER_NUM * dAlign * BF16_BYTES;
    const uint64_t oBytes = O_BUFFER_NUM * dAlign * FP32_BYTES;
    const uint64_t statBytes = STAT_BUFFER_NUM * STAT_UB_ELEMS * FP32_BYTES;
    return vBytes + qBytes + oBytes + statBytes + NORM_CACHE_BYTES <= ubSize_ - UB_RESERVED_BYTES;
}

ge::graphStatus BlockAttnResPrepareVectorSTileTiling::DoOpTiling()
{
    const uint32_t sTileNum = (totalS_ >= 16U && totalT_ * 2U <= aivCoreNum_) ? 2U : 1U;
    WorkDistribution distribution;
    if (CalculateWorkDistribution(totalT_ * sTileNum, aivCoreNum_, "VectorSTile", distribution) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = distribution.usedCoreNum;
    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint8_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalWorkUnits = static_cast<uint32_t>(totalT_ * sTileNum);
    tilingData_.totalD = static_cast<uint32_t>(totalD_);
    tilingData_.usedCoreNum = static_cast<uint16_t>(usedCoreNum_);
    tilingData_.bigCoreNum = static_cast<uint16_t>(distribution.bigCoreNum);
    tilingData_.blockFactor = distribution.blockFactor;
    tilingData_.tailBlockFactor = distribution.tailBlockFactor;
    tilingData_.baseD = static_cast<uint32_t>(totalD_);
    tilingData_.vCacheRows = static_cast<uint8_t>(totalN_);
    tilingData_.qBufferNum = static_cast<uint8_t>(Q_BUFFER_NUM);
    tilingData_.vBufferNum = 1U;
    tilingData_.oBufferNum = static_cast<uint8_t>(O_BUFFER_NUM);
    tilingData_.sTileNum = static_cast<uint8_t>(sTileNum);
    tilingData_.statUbElems = static_cast<uint32_t>(STAT_UB_ELEMS);
    tilingData_.eps = eps_;
    return ge::GRAPH_SUCCESS;
}

uint64_t BlockAttnResPrepareVectorSTileTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_TPL_VECTOR_S_TILE_BF16);
}

BlockAttnResPrepareBaseTiling::TilingDataView BlockAttnResPrepareVectorSTileTiling::GetTilingDataView() const
{
    return {&tilingData_, sizeof(tilingData_), "VectorSTile"};
}

void BlockAttnResPrepareVectorSTileTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "BlockAttnResPrepare VectorSTile: T=%lu N=%lu S=%lu D=%lu cores=%u baseT=1 baseS=%lu baseD=%lu "
            "q/o buffers=2/2 vCacheRows=%lu",
            totalT_, totalN_, totalS_, totalD_, usedCoreNum_, CeilDiv(totalS_, static_cast<uint64_t>(tilingData_.sTileNum)),
            totalD_, totalN_);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockAttnResPrepare, BlockAttnResPrepareVectorSTileTiling, supportedNpuArch,
                                   TILING_PRIORITY);

} // namespace optiling
