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
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief
 */

#ifndef __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__
#define __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t MAX_EXPERT_SIZE = 16384U; // 最大通信域专家的数量
constexpr uint32_t MAX_EXPERT_PER_RANK = 1024U;

struct GmmAlltoAllvAicpuTiling {
    uint16_t* sendCnt;
    uint16_t* recvCnt;
};

struct GmmAlltoAllvCommonTilingInfo {
    uint64_t A;
    uint64_t H;
    uint64_t sharedMatmulH;
    uint64_t E_ep;
    uint64_t N1;
    uint64_t Bs;
    uint64_t N2;
    uint64_t BsK;
    uint64_t epWorldSize;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t expertGroupSize;   // G: 每组专家数（2的幂）
    uint64_t numAlltoAllvCalls; // ceil(E_ep / G): kernel 侧 AlltoAllv 调用次数
    bool isGmmWeightTrans;
    bool isMmWeightTrans;
    bool isOptionalMatmul;
    bool isOptionalSendRecvCountTensors;
};

class GroupedMatMulAlltoAllvTilingData
{
public:
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling alltoAllvCcTiling;
    GmmAlltoAllvCommonTilingInfo commonTilingInfo;
    TCubeTiling matmulTiling;
    TCubeTiling sharedExpMatmulTiling;
    GmmAlltoAllvAicpuTiling aicpuTilingInfo;
};

#endif // __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__