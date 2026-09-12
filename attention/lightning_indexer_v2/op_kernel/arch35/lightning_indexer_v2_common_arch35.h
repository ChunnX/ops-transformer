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
 * \file lightning_indexer_v2_common_arch35.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_COMMON_ARCH35_H
#define LIGHTNING_INDEXER_V2_COMMON_ARCH35_H

#include "common/lightning_indexer_v2_base_arch35.h"

using namespace AscendC;
namespace LIV2Common {

// 与tiling的layout保持一致
enum class LI_V2_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename OUT_T, typename QK_T, typename SCORE_T,
          const bool PAGE_ATTENTION = false, LI_V2_LAYOUT LAYOUT_T = LI_V2_LAYOUT::BSND,
          LI_V2_LAYOUT K_LAYOUT_T = LI_V2_LAYOUT::PA_BBND, bool DT_W_FLAG = false, typename... Args>
struct LIV2Type : LightningIndexerV2DataType<Q_T, K_T, QK_T, SCORE_T, OUT_T, PAGE_ATTENTION, LI_V2_LAYOUT,
                                             LAYOUT_T, K_LAYOUT_T> {
    static constexpr bool weightsTypeFlag = DT_W_FLAG; // weight的dtype是否为FP32
};

struct RunInfo : LightningIndexerV2RunInfoBase {};

struct ConstInfo : LightningIndexerV2ConstInfoBase<LI_V2_LAYOUT> {
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t LI_SYNC_MODE4 = SYNC_MODE4;
    static constexpr uint32_t NEG_INF_FLOAT = 0xFF800000;

    uint32_t mBaseSizeAlign = 1U;
    uint64_t topk;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    int64_t cmpRatio = 1;
    int64_t cmpResidualK = 0;
    bool batchSupperFlag = false;
    bool isSparseCountOver2K = false;
    bool returnValueFlag = false;
};

using SplitCoreInfo = LightningIndexerV2SplitCoreInfo;
using LdSplitCoreInfo = LightningIndexerV2LdSplitCoreInfo;

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_COMMON_ARCH35_H
