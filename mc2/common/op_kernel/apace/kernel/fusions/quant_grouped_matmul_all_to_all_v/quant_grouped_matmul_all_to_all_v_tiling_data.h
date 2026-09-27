/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_all_to_all_v_tiling_data.h
 * \brief Compact APACE tiling data for MX QuantGroupedMatMul + AllToAllV (GMM first, then comm).
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_TILING_DATA_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_TILING_DATA_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace Apace {

// rankSize 上限与 sendCounts/recvCounts 容量对齐（SUPPORT_RANK_SIZE 最大 128，
// e * rankSize <= MAX_EXPERT_NUM = 256）
constexpr uint32_t GMM_A2AV_MAX_RANK_NUM = 128U;
constexpr uint32_t GMM_A2AV_MAX_SEG_NUM = 256U;

// 非阻塞 AllToAllV handle 上限：totalLoopCount <= TARGET_COMM_ROUNDS(4)，留余量取 8
constexpr uint32_t GMM_A2AV_MAX_BATCH_NUM = 8U;

struct GmmA2avMxTiling {
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

struct hcommGmmA2avTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;

    uint32_t rankSize{0};       // epWorldSize
    uint32_t expertNum{0};      // e：单卡专家数
    uint32_t batchExpertNum{0}; // 每批专家数（taskTilingInfo.expertNum）
    uint32_t batchCount{0};     // 总批数（taskTilingInfo.totalLoopCount）
    uint32_t n{0};              // N1
    uint32_t k{0};              // H1
    uint32_t mMax{0};           // 单专家最大 token 数（全 rank 累计）
    uint32_t isMxFp4{0};
    uint32_t localMatmul{0}; // 是否有共享专家
    uint32_t mmM{0};         // Bs
    uint32_t mmK{0};         // H2
    uint32_t mmN{0};         // N2
    uint32_t reserved{0};
    uint64_t cTableOffset{0}; // C 偏移表在 userWorkspace 中的字节偏移

    // SPARSE groupList，组序 expert-major：g = expertIdx * rankSize + rankIdx
    //   [g * 2]     = expertIdx（Blaze weightIdx，直接索引 gmmWeight/gmmWeightScale 专家维）
    //   [g * 2 + 1] = tokenCount（= sendCnt[rankIdx * e + expertIdx]，gmmX 同序行数）
    int64_t sparseGroupList[GMM_A2AV_MAX_SEG_NUM * 2U]{0};
    // recvCounts 同 sendCounts 布局 [rankIdx * e + expertIdx]（AllToAllV recv 侧）
    int32_t recvCount[GMM_A2AV_MAX_SEG_NUM]{0};

    GmmA2avMxTiling gmmTiling;

    int64_t mmGroupList[1]{0}; // 共享专家单组 OFFSET 格式：[0] = Bs
    GmmA2avMxTiling mmTiling;
};

} // namespace Apace

#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_TILING_DATA_H__
