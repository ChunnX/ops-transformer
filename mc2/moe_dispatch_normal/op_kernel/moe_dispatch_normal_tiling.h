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
 * \file moe_dispatch_normal_tiling.h
 * \brief
 */

#ifndef MOE_DISPATCH_NORMAL_TILING_H
#define MOE_DISPATCH_NORMAL_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace Moe {
struct MoeDispatchNormalInfo {
    uint32_t epWorldSize;  // epWorldSize
    uint32_t tpWorldSize;  // tpWorldSize
    uint32_t epRankId;     // epRankId
    uint32_t tpRankId;     // tpRankId
    uint32_t moeExpertNum; // moe expert number
    uint32_t quantMode;    // quant mode
    uint32_t globalBs;     // globalBs = BS * worldSize
    uint32_t bs;           // bs
    uint32_t k;            // k
    uint32_t h;            // h
    uint32_t aivNum;       // aivNum
    bool isQuant;          // whether quant or not
    bool isEnableDiagnose; // whether enable diagnose or not
    bool reserved2;        // reserved
    bool reserved3;        // reserved
    uint64_t totalUbSize;  // epWorldSize
    uint64_t totalWinSize;
};

struct MoeDispatchNormalTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    MoeDispatchNormalInfo moeDispatchNormalInfo;
};
} // namespace Moe
#endif