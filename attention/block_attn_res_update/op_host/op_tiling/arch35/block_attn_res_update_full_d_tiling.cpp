/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_update_full_d_tiling.h"

#include <algorithm>
#include <limits>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {
namespace block_attn_res_update {
namespace {

constexpr uint32_t FP32_ALIGN_ELEMENTS = 8;  // 8 * 4 B = 32 B
constexpr uint32_t BF16_ALIGN_ELEMENTS = 16; // 16 * 2 B = 32 B
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t BF16_BYTES = 2;
constexpr uint32_t VREG_FP32_ELEMENTS = 64;
constexpr uint32_t VREG_FP32_SHIFT = 6;
constexpr uint32_t MAX_D_SIZE = 8192;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t SMALL_D_MAX_VREG_NUM = 2;
constexpr uint32_t MIN_T_FOR_BATCH = 8;
constexpr uint32_t MAX_DATA_COPY_BLOCK_COUNT = 4095;
constexpr uint32_t SYSTEM_RESERVED_UB_BYTES = 8 * 1024;
constexpr uint64_t FULL_D_TILING_KEY = 0;
// Keep the generic full-D strategy as a fallback. A preferred split-D strategy should use a numerically smaller value.
constexpr int32_t FULL_D_TILING_PRIORITY = 1000;

} // namespace

bool BlockAttnResUpdateFullDTiling::IsCapable()
{
    if (dSize_ > MAX_D_SIZE) {
        OP_LOGD(opName_, "full-D tiling only supports D <= %u, but got D=%u", MAX_D_SIZE, dSize_);
        return false;
    }
    return true;
}

ge::graphStatus BlockAttnResUpdateFullDTiling::DoOpTiling()
{
    tilingData_ = context_->GetTilingData<BlockAttnResUpdateTilingData>();
    if (tilingData_ == nullptr) {
        OP_LOGE(opName_, "Tiling data is null.");
        return ge::GRAPH_FAILED;
    }

    const uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(tSize_, coreNum_));
    const uint64_t baseTPerCore = tSize_ / usedCoreNum;
    const uint32_t extraTCoreNum = static_cast<uint32_t>(tSize_ % usedCoreNum);
    const uint64_t maxTPerCore = baseTPerCore + (extraTCoreNum == 0 ? 0U : 1U);
    if (maxTPerCore > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(opName_, "T per core exceeds the uint32_t range, T=%lu, cores=%u, maxTPerCore=%lu.",
                tSize_, usedCoreNum, maxTPerCore);
        return ge::GRAPH_FAILED;
    }

    tilingData_->baseTPerCore = static_cast<uint32_t>(baseTPerCore);
    tilingData_->extraTCoreNum = extraTCoreNum;
    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->dSize = dSize_;
    tilingData_->dAlignFp32 = Ops::Base::CeilAlign(dSize_, FP32_ALIGN_ELEMENTS);
    tilingData_->dAlignBf16 = Ops::Base::CeilAlign(dSize_, BF16_ALIGN_ELEMENTS);
    tilingData_->dVregNum = (dSize_ + VREG_FP32_ELEMENTS - 1U) >> VREG_FP32_SHIFT;
    tilingData_->eps = eps_;

    blockDim_ = usedCoreNum;
    return SelectUbTiling(maxTPerCore);
}

uint64_t BlockAttnResUpdateFullDTiling::CalcUbBytes(
    uint32_t tileT, uint32_t &statsTStride, uint8_t &phase2Mode) const
{
    const bool useBatch = tilingData_->dVregNum <= SMALL_D_MAX_VREG_NUM && tileT >= MIN_T_FOR_BATCH;
    phase2Mode = static_cast<uint8_t>(
        useBatch ? BlockAttnResUpdatePhase2Mode::BATCH : BlockAttnResUpdatePhase2Mode::TOKEN);
    const uint32_t statsTAlignElements = useBatch ? VREG_FP32_ELEMENTS : FP32_ALIGN_ELEMENTS;
    statsTStride = Ops::Base::CeilAlign(tileT, statsTAlignElements);
    const uint64_t queryBytes = static_cast<uint64_t>(tilingData_->dAlignFp32) * FP32_BYTES;
    const uint64_t partialBytes = static_cast<uint64_t>(tileT) * tilingData_->dAlignFp32 * FP32_BYTES;
    const uint64_t deltaHBytes = static_cast<uint64_t>(tileT) * tilingData_->dAlignBf16 * BF16_BYTES;
    const uint64_t numeratorBytes = partialBytes;
    const uint64_t statsBytes = static_cast<uint64_t>(3) * statsTStride * FP32_BYTES;
    const uint64_t bufferBytes = partialBytes + deltaHBytes + numeratorBytes + statsBytes;
    return queryBytes + static_cast<uint64_t>(DOUBLE_BUFFER_NUM) * bufferBytes;
}

bool BlockAttnResUpdateFullDTiling::TryUbTiling(uint64_t maxTPerCore)
{
    if (ubSize_ <= SYSTEM_RESERVED_UB_BYTES) {
        return false;
    }

    // Keep at least two tiles on the busiest core so both ping-pong buffers participate when possible.
    const uint64_t maxTileT =
        (maxTPerCore + static_cast<uint64_t>(DOUBLE_BUFFER_NUM) - 1U) / DOUBLE_BUFFER_NUM;
    const uint64_t maxCandidateT = std::min<uint64_t>(maxTileT, MAX_DATA_COPY_BLOCK_COUNT);
    const uint64_t bytesPerT =
        static_cast<uint64_t>(tilingData_->dAlignFp32) * FP32_BYTES * 2 +
        static_cast<uint64_t>(tilingData_->dAlignBf16) * BF16_BYTES + 3 * FP32_BYTES;
    const uint64_t queryBytes = static_cast<uint64_t>(tilingData_->dAlignFp32) * FP32_BYTES;
    const uint64_t usableUbBytes = ubSize_ - SYSTEM_RESERVED_UB_BYTES;
    if (usableUbBytes <= queryBytes || bytesPerT == 0) {
        return false;
    }

    uint64_t candidateT = std::min<uint64_t>(maxCandidateT,
                                             (usableUbBytes - queryBytes) /
                                                 (static_cast<uint64_t>(DOUBLE_BUFFER_NUM) * bytesPerT));
    while (candidateT > 0) {
        uint32_t statsTStride = 0;
        uint8_t phase2Mode = 0;
        const uint64_t ubBytes = CalcUbBytes(static_cast<uint32_t>(candidateT), statsTStride, phase2Mode);
        if (ubBytes <= usableUbBytes) {
            tilingData_->tileT = static_cast<uint32_t>(candidateT);
            tilingData_->statsTStride = statsTStride;
            tilingData_->phase2Mode = phase2Mode;
            selectedUbBytes_ = ubBytes;
            return true;
        }
        --candidateT;
    }
    return false;
}

ge::graphStatus BlockAttnResUpdateFullDTiling::SelectUbTiling(uint64_t maxTPerCore)
{
    if (TryUbTiling(maxTPerCore)) {
        OP_LOGI(opName_,
                "T=%lu D=%u dAlignFp32=%u dAlignBf16=%u cores=%u "
                "tileT=%u phase2Mode=%u buffers=%u "
                "UB=%lu/%lu",
                tSize_, dSize_, tilingData_->dAlignFp32, tilingData_->dAlignBf16,
                tilingData_->usedCoreNum, tilingData_->tileT,
                static_cast<uint32_t>(tilingData_->phase2Mode), DOUBLE_BUFFER_NUM, selectedUbBytes_, ubSize_);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(opName_,
            "One complete D vector does not fit in UB with double buffering (D=%u, ubSize=%lu).",
            dSize_, ubSize_);
    return ge::GRAPH_FAILED;
}

uint64_t BlockAttnResUpdateFullDTiling::GetTilingKey() const
{
    return FULL_D_TILING_KEY;
}

ge::graphStatus BlockAttnResUpdateFullDTiling::PostTiling()
{
    context_->SetBlockDim(blockDim_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockAttnResUpdate, BlockAttnResUpdateFullDTiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), FULL_D_TILING_PRIORITY);

} // namespace block_attn_res_update
} // namespace optiling
