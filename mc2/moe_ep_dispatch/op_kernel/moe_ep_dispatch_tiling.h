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
 * \file moe_ep_dispatch_tiling.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_TILING_H
#define MOE_EP_DISPATCH_TILING_H

#include "../../common/op_kernel/mc2_common_def.h"
#include "../../common/op_kernel/mc2_moe_context.h"

struct MoeEpCommonTilingData {
    uint32_t epWorldSize        = 0;
    uint32_t epRankId           = 0;
    uint32_t numExperts         = 0;
    uint32_t numLocalExperts    = 0;
    uint32_t numTokens          = 0;
    uint32_t hidden             = 0;
    uint32_t topK               = 0;
    uint32_t numMaxTokensPerRank = 0;
    uint32_t perSlotBytes       = 0;
    uint32_t expertAlignment    = 1;
};

struct MoeEpDispatchInfo {
    MoeEpCommonTilingData cfg;
    uint32_t doCpuSync          = 1;
    uint32_t isCached           = 0;
    uint32_t networkMode        = 0;
    uint32_t numScaleupRanks    = 0;
    uint32_t numScaleoutRanks   = 0;
    uint32_t numAivStage1       = 0;
    uint32_t numAivStage2       = 0;
    uint32_t aivNum             = 0;
    uint64_t hostPinnedCounterAddr = 0;
    uint64_t totalWinSizeEp     = 0;
    uint64_t totalUbSize        = 0;
};

struct MoeEpDispatchTilingData {
    MoeEpDispatchInfo moeEpDispatchInfo;
};



#endif  // MOE_EP_DISPATCH_TILING_H