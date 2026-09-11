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
 * \file quant_lightning_indexer_v2_common_arch35.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_COMMON_ARCH35_H
#define QUANT_LIGHTNING_INDEXER_V2_COMMON_ARCH35_H

#include <type_traits>

#if __has_include("../../lightning_indexer_v2/arch35/common/lightning_indexer_v2_base_arch35.h")
#include "../../lightning_indexer_v2/arch35/common/lightning_indexer_v2_base_arch35.h"
#else
#include "../../../lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_base_arch35.h"
#endif

using namespace AscendC;
namespace QLIV2Common {
using FP8E4M3 = fp8_e4m3fn_t;
using FP4E2M1 = fp4x2_e2m1_t;
using FP8E8M0 = fp8_e8m0_t;

constexpr uint32_t MX_SCALE_GROUP_SIZE = 32;
constexpr uint32_t FP8_TWO = 2;
constexpr uint32_t FP4_PACK_NUM = 2;

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename QK_T, typename SCORE_T, typename OUT_T,
          const bool PAGE_ATTENTION = false, LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND,
          LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BBND, typename SCALE_T = float, typename WEIGHT_T = float,
          typename... Args>
struct QLIV2Type
    : LIV2Common::LightningIndexerV2DataType<
          std::conditional_t<std::is_same_v<Q_T, FP4E2M1> && std::is_same_v<K_T, FP4E2M1> &&
                                 std::is_same_v<SCALE_T, FP8E8M0>,
                             uint8_t, Q_T>,
          std::conditional_t<std::is_same_v<Q_T, FP4E2M1> && std::is_same_v<K_T, FP4E2M1> &&
                                 std::is_same_v<SCALE_T, FP8E8M0>,
                             uint8_t, K_T>,
          QK_T, SCORE_T, OUT_T, PAGE_ATTENTION, LI_LAYOUT, Q_LAYOUT_T, K_LAYOUT_T> {
    static_assert((std::is_same_v<QK_T, float> &&
                   (std::is_same_v<SCORE_T, uint32_t> || std::is_same_v<SCORE_T, uint16_t>)) ||
                      (std::is_same_v<QK_T, bfloat16_t> && std::is_same_v<SCORE_T, uint16_t>) ||
                      (std::is_same_v<QK_T, int32_t> && std::is_same_v<SCORE_T, uint16_t>),
                  "Invalid combination of QK_T and SCORE_T");
    using rawQueryType = Q_T;
    using rawKeyType = K_T;
    using scaleType = SCALE_T;
    using weightType = WEIGHT_T;

    static constexpr bool isMxFp8 = std::is_same_v<rawQueryType, FP8E4M3> && std::is_same_v<rawKeyType, FP8E4M3> &&
                                    std::is_same_v<SCALE_T, FP8E8M0>;
    static constexpr bool isMxFp4 = std::is_same_v<rawQueryType, FP4E2M1> && std::is_same_v<rawKeyType, FP4E2M1> &&
                                    std::is_same_v<SCALE_T, FP8E8M0>;
    static constexpr bool isMx = isMxFp8 || isMxFp4;
    static constexpr bool isWeightFP16 = std::is_same_v<WEIGHT_T, half>;
};

struct RunInfo : LIV2Common::LightningIndexerV2RunInfoBase {
    uint32_t validS2Len;
    uint32_t qScaleLoop;
    uint32_t kScaleLoop;
    uint64_t tensorQScaleOffset;
    uint64_t tensorKeyScaleOffset;
};

struct ConstInfo : LIV2Common::LightningIndexerV2ConstInfoBase<LI_LAYOUT> {
    static constexpr uint32_t QLIV2_SYNC_MODE4 = SYNC_MODE4;
    static constexpr uint16_t NEG_INF_BFLOAT = 0xFF80;

    uint32_t mBaseSizeMax = 1U;
    uint64_t sparseCount;
    uint32_t cmpRatio = 1U;
    uint32_t keyDequantScaleStride0 = 0U;
    int32_t maxSeqlenQ = -1;
    uint32_t quantMode = 1U;
    uint32_t cmpResiduaKLenDims = 0U;
};

using SplitCoreInfo = LIV2Common::LightningIndexerV2SplitCoreInfo;
using LdSplitCoreInfo = LIV2Common::LightningIndexerV2LdSplitCoreInfo;
using LIV2Common::Align;
using LIV2Common::CeilDiv;
using LIV2Common::Max;
using LIV2Common::Min;

} // namespace QLIV2Common

#endif // QUANT_LIGHTNING_INDEXER_V2_COMMON_ARCH35_H
