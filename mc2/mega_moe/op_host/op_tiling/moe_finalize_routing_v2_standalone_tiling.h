/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_v2_standalone_tiling.h
 * \brief Standalone tiling implementation for MoeFinalizeRoutingV2 (DROPLESS_ROW only)
 */

#ifndef MOE_FINALIZE_ROUTING_V2_STANDALONE_TILING_H_
#define MOE_FINALIZE_ROUTING_V2_STANDALONE_TILING_H_

#include <cstdint>
#include <algorithm>

#include "../../op_kernel/mega_moe_tiling_def.h"

namespace optiling {
namespace v2 {

static constexpr int64_t kTwo = 2;
static constexpr int64_t kInputBufferNum = 4;
static constexpr int64_t kOutputBufferNum = 1;
static constexpr uint64_t kFullLoadRowKHBaseTilingKey = 30000;
static constexpr uint64_t kFullLoadKHBaseTilingKey = 40000;
static constexpr uint64_t kFullLoadHBaseTilingKey = 10000;
static constexpr uint64_t kSplitHBaseTilingKey = 20000;
static constexpr uint64_t kDroplessRowTilingKey = 20;
static constexpr uint64_t kFloat16TilingKey = 1;
static constexpr uint64_t kBFloat16TilingKey = 2;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

inline int64_t CeilDivU64(uint64_t a, uint64_t b)
{
    return static_cast<int64_t>((a + b - 1) / b);
}


inline void CalcMoeFinalizeRoutingV2Tiling(
    int64_t row,
    int64_t h,
    int64_t k,
    int64_t dtypeSize,
    int64_t scaleDtypeSize,
    int64_t blockSize,
    int64_t ubSize,
    int64_t coreNum,
    bool hasX1,
    bool hasX2,
    bool hasBias,
    bool hasScales,
    bool isScaleBf16,
    MoeFinalizeRoutingV2RegbaseTilingData &tilingData,
    uint64_t &tilingKey)
{
    int64_t hAlignedByte = CeilDivU64(static_cast<uint64_t>(h * dtypeSize), static_cast<uint64_t>(blockSize)) * blockSize;
    int64_t hAligned = hAlignedByte / dtypeSize;
    int64_t hAligned32Byte = CeilDivU64(static_cast<uint64_t>(h * sizeof(float)), static_cast<uint64_t>(blockSize)) * blockSize;
    int64_t scalesAlignedByte = CeilDivU64(static_cast<uint64_t>(k * scaleDtypeSize), static_cast<uint64_t>(blockSize)) * blockSize;

    auto RowsHSize = [&](int64_t rowFactor, bool scalesInUb) -> int64_t {
        return (static_cast<int64_t>(hasX1) + static_cast<int64_t>(hasX2)) *
                   CeilDivU64(static_cast<uint64_t>(rowFactor * h * dtypeSize), static_cast<uint64_t>(blockSize)) * blockSize +
               (scalesInUb && hasScales ? 1 : 0) *
                   CeilDivU64(static_cast<uint64_t>(rowFactor * k * scaleDtypeSize), static_cast<uint64_t>(blockSize)) * blockSize +
               CeilDivU64(static_cast<uint64_t>(rowFactor * h * sizeof(float)), static_cast<uint64_t>(blockSize)) * blockSize;
    };

    auto RowsHSizeForKHFullLoad = [&](int64_t rowFactor, bool scalesInUb) -> int64_t {
        return (static_cast<int64_t>(hasX1) + static_cast<int64_t>(hasX2)) *
                   CeilDivU64(static_cast<uint64_t>(rowFactor * h * dtypeSize), static_cast<uint64_t>(blockSize)) * blockSize +
               (scalesInUb && hasScales ? 1 : 0) *
                   CeilDivU64(static_cast<uint64_t>(rowFactor * k * scaleDtypeSize), static_cast<uint64_t>(blockSize)) * blockSize +
               (hasBias ? 1 : 0) * rowFactor * k * hAligned * dtypeSize +
               rowFactor * k * hAligned * dtypeSize +
               kTwo * CeilDivU64(static_cast<uint64_t>(rowFactor * k * sizeof(int32_t)), static_cast<uint64_t>(blockSize)) * blockSize +
               CeilDivU64(static_cast<uint64_t>(rowFactor * h * sizeof(float)), static_cast<uint64_t>(blockSize)) * blockSize;
    };

    auto CalcRowFactor = [&](int64_t ubSizeRemained, bool scalesInUb) -> int64_t {
        int64_t rowFactor = 1;
        int64_t factor = 1;
        while (RowsHSize(rowFactor, scalesInUb) <= ubSizeRemained) {
            factor *= kTwo;
            rowFactor *= factor;
        }
        int64_t upper = rowFactor;
        int64_t lower = rowFactor / factor;
        while (upper - lower > 1) {
            int64_t mid = (lower + upper) / kTwo;
            int64_t size = RowsHSize(mid, scalesInUb);
            if (size <= ubSizeRemained) {
                lower = mid;
            } else {
                upper = mid;
            }
        }
        return lower;
    };

    auto CalcRowFactorForKHFullLoad = [&](int64_t ubSizeRemained, bool scalesInUb) -> int64_t {
        int64_t rowFactor = 1;
        int64_t factor = 1;
        while (RowsHSizeForKHFullLoad(rowFactor, scalesInUb) <= ubSizeRemained) {
            factor *= kTwo;
            rowFactor *= factor;
        }
        int64_t upper = rowFactor;
        int64_t lower = rowFactor / factor;
        while (upper - lower > 1) {
            int64_t mid = (lower + upper) / kTwo;
            int64_t size = RowsHSizeForKHFullLoad(mid, scalesInUb);
            if (size <= ubSizeRemained) {
                lower = mid;
            } else {
                upper = mid;
            }
        }
        return lower;
    };

    auto SetFullLoadTilingData = [&](int64_t rowOfFormerBlock, int64_t rowOfTailBlock, int64_t rowFactor) {
        int64_t rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, rowFactor);
        rowFactor = CeilDiv(rowOfFormerBlock, rowLoopOfFormerBlock);
        int64_t rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, rowFactor);
        int64_t tailRowFactorOfFormerBlock = rowOfFormerBlock - (rowLoopOfFormerBlock - 1) * rowFactor;
        int64_t tailRowFactorOfTailBlock = rowOfTailBlock - (rowLoopOfTailBlock - 1) * rowFactor;
        tilingData.rowLoopOfFormerBlock = rowLoopOfFormerBlock;
        tilingData.rowLoopOfTailBlock = rowLoopOfTailBlock;
        tilingData.rowFactor = rowFactor;
        tilingData.tailRowFactorOfFormerBlock = tailRowFactorOfFormerBlock;
        tilingData.tailRowFactorOfTailBlock = tailRowFactorOfTailBlock;
        tilingData.hLoop = 1;
        tilingData.hFactor = h;
        tilingData.tailHFactor = h;
        tilingData.kLoop = k;
        tilingData.kFactor = 1;
        tilingData.tailKFactor = 1;
        tilingData.activeNum = row * k;
    };

    int64_t expandedXAlignedByte = CeilDivU64(static_cast<uint64_t>(row * k * h * dtypeSize), static_cast<uint64_t>(blockSize)) * blockSize;
    int64_t hasBiasvalue = hasBias ? CeilDivU64(static_cast<uint64_t>(k * h * dtypeSize), static_cast<uint64_t>(blockSize)) * blockSize : 0;
    int64_t hasScalevalue = hasScales ? scalesAlignedByte : 0;
    int64_t totalSizeRowKHFullLoad =
        expandedXAlignedByte + hasBiasvalue +
        kTwo * (hasScalevalue + (static_cast<int64_t>(hasX1) + static_cast<int64_t>(hasX2)) * hAlignedByte +
                hAligned32Byte * kOutputBufferNum);
    bool rowKHFullLoad = (totalSizeRowKHFullLoad <= ubSize);

    bool kHFullLoad = false;
    if (!rowKHFullLoad) {
        int64_t kHAlignedByte = k * hAligned * dtypeSize;
        int64_t expandedRowIdxAlignedByte = CeilDivU64(static_cast<uint64_t>(k * sizeof(int32_t)), static_cast<uint64_t>(blockSize)) * blockSize;
        int64_t hasExpandedRowIdxValue = (k == 1) ? expandedRowIdxAlignedByte : 0;
        int64_t expertIdxAlignedByte = CeilDivU64(static_cast<uint64_t>(k * sizeof(int32_t)), static_cast<uint64_t>(blockSize)) * blockSize;
        int64_t hasExpertIdxValue = (hasBias && k == 1) ? expertIdxAlignedByte : 0;
        int64_t totalSizeKHFullLoad = kTwo * (kHAlignedByte + hasBiasvalue + hasScalevalue + hasExpandedRowIdxValue +
                                        hasExpertIdxValue +
                                        (static_cast<int64_t>(hasX1) + static_cast<int64_t>(hasX2)) * hAlignedByte +
                                        hAligned32Byte * kOutputBufferNum);
        kHFullLoad = (totalSizeKHFullLoad <= ubSize);
        printf("totalSizeKHFullLoad: %d\n", totalSizeKHFullLoad);
    }
    

    bool hFullLoad = false;
    if (!rowKHFullLoad && !kHFullLoad) {
        int64_t oneKAlignedByte = static_cast<int64_t>(blockSize);
        int64_t actualInputNum = kInputBufferNum - static_cast<int64_t>(!hasX1) - static_cast<int64_t>(!hasX2) -
                                 static_cast<int64_t>(!hasBias);
        int64_t totalSizeHFullLoad =
            kTwo * (hAlignedByte * actualInputNum + (hasScales ? oneKAlignedByte : 0) + hAligned32Byte * kOutputBufferNum);
        hFullLoad = (totalSizeHFullLoad <= ubSize);
    }

    int64_t rowPerCore = CeilDiv(row, coreNum);
    int64_t usedCoreNum = std::min(CeilDiv(row, rowPerCore), coreNum);
    int64_t rowOfTailBlock = row - (usedCoreNum - 1) * rowPerCore;

    tilingData.row = row;
    tilingData.e = 0;
    tilingData.c = 0;
    tilingData.h = h;
    tilingData.hAligned = hAligned;
    tilingData.k = k;
    tilingData.rowOfFormerBlock = rowPerCore;
    tilingData.rowOfTailBlock = rowOfTailBlock;

    if (rowKHFullLoad) {
        int64_t ubSizeRemained = (ubSize - expandedXAlignedByte - hasBiasvalue) / kTwo;
        int64_t rowFactor = CalcRowFactor(ubSizeRemained, true);
        SetFullLoadTilingData(rowPerCore, rowOfTailBlock, rowFactor);
    } else if (kHFullLoad) {
        int64_t rowFactor = 1;
        if (k == 1) {
            int64_t ubSizeRemained = ubSize / kTwo;
            rowFactor = CalcRowFactorForKHFullLoad(ubSizeRemained, true);
        } else {
            int64_t kHAlignedByteInner = k * hAligned * dtypeSize;
            int64_t hasBiasvalueInner = hasBias ? kHAlignedByteInner : 0;
            int64_t ubSizeRemained = ubSize / kTwo - kHAlignedByteInner - hasBiasvalueInner;
            rowFactor = CalcRowFactor(ubSizeRemained, true);
        }
        SetFullLoadTilingData(rowPerCore, rowOfTailBlock, rowFactor);
    } else if (hFullLoad) {
        int64_t expandedXAndBiasSize = (1 + static_cast<int64_t>(hasBias)) * hAligned * dtypeSize;
        int64_t kFactor = ubSize / kTwo / expandedXAndBiasSize;
        int64_t upper = kFactor;
        int64_t lower = 1;
        while (upper - lower > 1) {
            int64_t mid = (lower + upper) / kTwo;
            int64_t useUbSize =
                mid * expandedXAndBiasSize + CeilDivU64(static_cast<uint64_t>(mid * dtypeSize), static_cast<uint64_t>(blockSize)) * blockSize + RowsHSize(1, false);
            if (useUbSize > ubSize / kTwo) {
                upper = mid;
            } else {
                lower = mid;
            }
        }
        kFactor = lower;
        int64_t kLoop = CeilDiv(k, kFactor);
        kFactor = CeilDiv(k, kLoop);
        int64_t tailKFactor = k - (kLoop - 1) * kFactor;

        int64_t ubSizeRemained = ubSize / kTwo - kFactor * expandedXAndBiasSize -
                                 CeilDivU64(static_cast<uint64_t>(kFactor * dtypeSize), static_cast<uint64_t>(blockSize)) * blockSize;
        int64_t rowFactor = CalcRowFactor(ubSizeRemained, false);
        SetFullLoadTilingData(rowPerCore, rowOfTailBlock, rowFactor);
        tilingData.kLoop = kLoop;
        tilingData.kFactor = kFactor;
        tilingData.tailKFactor = tailKFactor;
    } else {
        int64_t actualInputNum = kInputBufferNum - static_cast<int64_t>(!hasX1) - static_cast<int64_t>(!hasX2) -
                                 static_cast<int64_t>(!hasBias);
        int64_t totalBufferNum = actualInputNum + kOutputBufferNum + (dtypeSize != sizeof(float) ? 1 : 0);
        int64_t hFactor = ubSize / kTwo / dtypeSize / totalBufferNum;
        int64_t hLoop = CeilDiv(h, hFactor);
        hFactor = CeilDiv(h, hLoop);
        int64_t tailHFactor = h - (hLoop - 1) * hFactor;
        tilingData.rowLoopOfFormerBlock = rowPerCore;
        tilingData.rowLoopOfTailBlock = rowOfTailBlock;
        tilingData.rowFactor = 1;
        tilingData.tailRowFactorOfFormerBlock = rowPerCore;
        tilingData.tailRowFactorOfTailBlock = rowOfTailBlock;
        tilingData.hLoop = hLoop;
        tilingData.hFactor = hFactor;
        tilingData.tailHFactor = tailHFactor;
        tilingData.activeNum = row * k;
    }

    if (rowKHFullLoad) {
        tilingKey = kFullLoadRowKHBaseTilingKey;
    } else if (kHFullLoad) {
        tilingKey = kFullLoadKHBaseTilingKey;
    } else if (hFullLoad) {
        tilingKey = kFullLoadHBaseTilingKey;
    } else {
        tilingKey = kSplitHBaseTilingKey;
    }

    tilingKey += kDroplessRowTilingKey;

    uint64_t scaleDtypeKey = 0;
    if (isScaleBf16) {
        scaleDtypeKey = kBFloat16TilingKey;
    } else if (scaleDtypeSize == sizeof(short)) {
        scaleDtypeKey = kFloat16TilingKey;
    }
    tilingKey += scaleDtypeKey;
}

} // namespace v2
} // namespace optiling

#endif
