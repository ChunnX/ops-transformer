/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "kernel_tiling.h"
#include "../../../op_kernel/mega_moe_tiling.h"

struct MegaMoeTilingDef {
    uint32_t bs;
    uint32_t h;
    uint32_t topk;
    uint32_t expertPerRank;
    uint32_t n;
    uint32_t epWorldSize;
    uint32_t dispatchQuantOutType;
    uint32_t coreNum;
    uint32_t ubSize;
};

extern uint8_t g_megaMoeTiling[];

void InitMegaMoeTiling(MegaMoeTilingDef& tilingDef) {
    MegaMoeTilingData* tiling = reinterpret_cast<MegaMoeTilingData*>(g_megaMoeTiling);
    
    tiling->moeInitRoutingTilingData.totalLength = tilingDef.bs * tilingDef.topk;
    tiling->moeInitRoutingTilingData.n = tilingDef.h;
    tiling->moeInitRoutingTilingData.k = tilingDef.topk;
    tiling->moeInitRoutingTilingData.cols = tilingDef.h;
    tiling->expertPerRank = tilingDef.expertPerRank;
    tiling->m = tilingDef.bs;
    tiling->k = tilingDef.h;
    tiling->n = tilingDef.n;
    tiling->epWorldSize = tilingDef.epWorldSize;
    tiling->topK = tilingDef.topk;
}