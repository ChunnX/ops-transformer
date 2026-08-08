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
constexpr uint32_t MAX_FULL_D_COLS = 8192;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t SMALL_D_MAX_COLS_LOOPS = 2;
constexpr uint32_t ROW_BATCH_MIN_ROWS = 8;
constexpr uint32_t MAX_DATA_COPY_BLOCK_COUNT = 4095;
constexpr uint32_t SYSTEM_RESERVED_UB_BYTES = 8 * 1024;
constexpr uint64_t FULL_D_TILING_KEY = 0;
// Keep the generic full-D strategy as a fallback. A preferred split-D strategy should use a numerically smaller value.
constexpr int32_t FULL_D_TILING_PRIORITY = 1000;

} // namespace

bool BlockAttnResUpdateFullDTiling::IsCapable()
{
    if (cols_ > MAX_FULL_D_COLS) {
        OP_LOGD(opName_, "full-D tiling only supports D <= %u, but got D=%u", MAX_FULL_D_COLS, cols_);
        return false;
    }
    return true;
}

ge::graphStatus BlockAttnResUpdateFullDTiling::DoOpTiling()
{
    tilingData_ = context_->GetTilingData<BlockAttnResUpdateTilingData>();
    if (tilingData_ == nullptr) {
        OP_LOGE(opName_, "tiling data is null");
        return ge::GRAPH_FAILED;
    }

    const uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(totalRows_, coreNum_));
    const uint64_t rowsPerCoreBaseValue = totalRows_ / usedCoreNum;
    const uint32_t rowsRemainder = static_cast<uint32_t>(totalRows_ % usedCoreNum);
    const uint64_t maxRowsPerCore = rowsPerCoreBaseValue + (rowsRemainder == 0 ? 0U : 1U);
    if (maxRowsPerCore > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(opName_, "rows per core exceeds the uint32_t range, T=%lu, cores=%u, maxRowsPerCore=%lu",
                totalRows_, usedCoreNum, maxRowsPerCore);
        return ge::GRAPH_FAILED;
    }
    const uint32_t rowsPerCoreBase = static_cast<uint32_t>(rowsPerCoreBaseValue);

    tilingData_->rowsPerCoreBase = rowsPerCoreBase;
    tilingData_->rowsRemainder = rowsRemainder;
    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->cols = cols_;
    tilingData_->colsAlignFp32 = Ops::Base::CeilAlign(cols_, FP32_ALIGN_ELEMENTS);
    tilingData_->colsAlignBf16 = Ops::Base::CeilAlign(cols_, BF16_ALIGN_ELEMENTS);
    tilingData_->colsLoops = (cols_ + VREG_FP32_ELEMENTS - 1U) >> VREG_FP32_SHIFT;
    tilingData_->epsilon = epsilon_;
    tilingData_->invCols = 1.0F / static_cast<float>(cols_);

    blockDim_ = usedCoreNum;
    return SelectUbTiling(maxRowsPerCore);
}

uint64_t BlockAttnResUpdateFullDTiling::CalcUbBytes(
    uint32_t rows, uint32_t &scalarAlign, uint8_t &phase2Mode) const
{
    const bool useRowBatch = tilingData_->colsLoops <= SMALL_D_MAX_COLS_LOOPS && rows >= ROW_BATCH_MIN_ROWS;
    phase2Mode = static_cast<uint8_t>(useRowBatch ? BlockAttnResUpdatePhase2Mode::ROW_BATCH : BlockAttnResUpdatePhase2Mode::ROW_LOCAL);
    const uint32_t scalarAlignElements = useRowBatch ? VREG_FP32_ELEMENTS : FP32_ALIGN_ELEMENTS;
    scalarAlign = Ops::Base::CeilAlign(rows, scalarAlignElements);
    const uint64_t queryBytes = static_cast<uint64_t>(tilingData_->colsAlignFp32) * FP32_BYTES;
    const uint64_t partialBytes = static_cast<uint64_t>(rows) * tilingData_->colsAlignFp32 * FP32_BYTES;
    const uint64_t deltaHBytes = static_cast<uint64_t>(rows) * tilingData_->colsAlignBf16 * BF16_BYTES;
    const uint64_t numeratorBytes = partialBytes;
    const uint64_t statsBytes = static_cast<uint64_t>(3) * scalarAlign * FP32_BYTES;
    const uint64_t bufferBytes = partialBytes + deltaHBytes + numeratorBytes + statsBytes;
    return queryBytes + static_cast<uint64_t>(DOUBLE_BUFFER_NUM) * bufferBytes;
}

bool BlockAttnResUpdateFullDTiling::TryUbTiling(uint64_t maxRowsPerCore)
{
    if (ubSize_ <= SYSTEM_RESERVED_UB_BYTES) {
        return false;
    }

    // Keep at least two tiles on the busiest core so both ping-pong buffers participate when possible.
    const uint64_t maxRowsPerTile =
        (maxRowsPerCore + static_cast<uint64_t>(DOUBLE_BUFFER_NUM) - 1U) / DOUBLE_BUFFER_NUM;
    const uint64_t maxCandidate = std::min<uint64_t>(maxRowsPerTile, MAX_DATA_COPY_BLOCK_COUNT);
    const uint64_t rowBytes = static_cast<uint64_t>(tilingData_->colsAlignFp32) * FP32_BYTES * 2 +
                              static_cast<uint64_t>(tilingData_->colsAlignBf16) * BF16_BYTES + 3 * FP32_BYTES;
    const uint64_t queryBytes = static_cast<uint64_t>(tilingData_->colsAlignFp32) * FP32_BYTES;
    const uint64_t usable = ubSize_ - SYSTEM_RESERVED_UB_BYTES;
    if (usable <= queryBytes || rowBytes == 0) {
        return false;
    }

    uint64_t candidate = std::min<uint64_t>(maxCandidate,
                                            (usable - queryBytes) /
                                                (static_cast<uint64_t>(DOUBLE_BUFFER_NUM) * rowBytes));
    while (candidate > 0) {
        uint32_t scalarAlign = 0;
        uint8_t phase2Mode = 0;
        const uint64_t ubBytes = CalcUbBytes(static_cast<uint32_t>(candidate), scalarAlign, phase2Mode);
        if (ubBytes <= usable) {
            tilingData_->tileRows = static_cast<uint32_t>(candidate);
            tilingData_->scalarAlign = scalarAlign;
            tilingData_->phase2Mode = phase2Mode;
            selectedUbBytes_ = ubBytes;
            return true;
        }
        --candidate;
    }
    return false;
}

ge::graphStatus BlockAttnResUpdateFullDTiling::SelectUbTiling(uint64_t maxRowsPerCore)
{
    if (TryUbTiling(maxRowsPerCore)) {
        OP_LOGI(opName_,
                "T=%lu cols=%u colsAlignFp32=%u colsAlignBf16=%u cores=%u tileRows=%u phase2Mode=%u buffers=%u "
                "UB=%lu/%lu",
                totalRows_, cols_, tilingData_->colsAlignFp32, tilingData_->colsAlignBf16, tilingData_->usedCoreNum,
                tilingData_->tileRows, static_cast<uint32_t>(tilingData_->phase2Mode), DOUBLE_BUFFER_NUM,
                selectedUbBytes_, ubSize_);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(opName_, "one complete D row with double buffering does not fit in UB (D=%u, ubSize=%lu)", cols_,
            ubSize_);
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
