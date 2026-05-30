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
* \file mega_moe_tiling_a2a3.h
* \brief
*/

#ifndef ASCENDC_MEGA_MOE_TILING_A2A3
#define ASCENDC_MEGA_MOE_TILING_A2A3
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "moe_init_routing_v2/moe_init_routing_v2_tiling.h"

#define DISPATCH_FFN_COMBINE_QUANT_MODE_NO_QUANT 0
#define DISPATCH_FFN_COMBINE_QUANT_MODE_W4A8 1
#define DISPATCH_FFN_COMBINE_QUANT_MODE_INT8 2

#define DISPATCH_FFN_COMBINE_QUANT_OUT_TYPE_UNDEFINED 0
#define DISPATCH_FFN_COMBINE_QUANT_OUT_TYPE_E5M2 3
#define DISPATCH_FFN_COMBINE_QUANT_OUT_TYPE_E4M3FN 4

#define SOC_ASCEND910B 0
#define SOC_ASCEND910_93 1

#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

struct MegaMoeA2A3TilingData {
    uint32_t bs;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t aivNum;
    uint32_t totalUbSize;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t isWeightTensorList;

    uint32_t moeExpertNum;
    uint32_t epWorldSize;
    uint32_t cclBufferSize;
    uint32_t maxRecvTokenNum;
    uint32_t dispatchQuantMode;
    int32_t  dispatchQuantOutDtype;
    uint32_t combineQuantMode;
    uint32_t commAlgCode;
    uint32_t numMaxTokensPerRank;
    uint32_t activationCode;
    float    activationClamp;
    uint32_t isTransposeW1;
    uint32_t isTransposeW2;

    uint32_t hasBias1;
    uint32_t hasBias2;
    uint32_t hasXActiveMask;
    uint32_t hasScales;

    uint32_t isQuantRouting;

    int32_t  activationOutDtype;
    uint32_t weight1Interleave;

    uint64_t initRoutingQuantTilingKey;
};

struct MegaMoeA2A3InfoQuant {
    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
};

struct MegaMoeA2A3InfoNonQuant {
    optiling::MoeInitRoutingV2TilingData moeInitRoutingV2TilingData;
};

struct MegaMoeA2A3TilingDataQuant {
    MegaMoeA2A3InfoQuant MegaMoeA2A3Info;
};

#endif