/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_quant_grouped_matmul_tiling_data.h
 * \brief Compact APACE tiling data for MX AllToAllV + GroupedMatMul.
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace Apace {

constexpr uint32_t A2AV_GMM_MAX_SEG_NUM = 256U;
constexpr uint32_t A2AV_GMM_MAX_RANK_NUM = A2AV_GMM_MAX_SEG_NUM;

struct A2avGmmMxTiling {
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    uint32_t kAL1{0};
    uint32_t kBL1{0};
    uint32_t scaleKAL1{0};
    uint8_t dbL0C{2};
    uint8_t l1BufferStage{2};
    uint8_t reserved[2]{0};
};

struct hcommA2avQuantGmmTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;

    uint32_t rankSize{0};
    uint32_t expertNum{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t scaleKLen{0};
    uint32_t isMxFp4{0};
    uint32_t hasPermuteOut{0};
    uint32_t localMatmul{0};
    uint64_t dataRecvOffset{0};
    uint64_t scaleRecvOffset{0};

    // expert-major: segment = expert * rankSize + peer.
    int64_t sparseGroupList[A2AV_GMM_MAX_SEG_NUM * 2U]{0};
    uint32_t segmentSendRows[A2AV_GMM_MAX_SEG_NUM]{0};

    A2avGmmMxTiling gmmTiling;

    uint32_t mmM{0};
    uint32_t mmK{0};
    uint32_t mmN{0};
    uint32_t reserved{0};
    int64_t mmGroupList[1]{0};
    A2avGmmMxTiling mmTiling;
};

} // namespace Apace
