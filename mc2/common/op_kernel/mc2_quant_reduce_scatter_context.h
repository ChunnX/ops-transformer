/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

/*!
 * \file mc2_quant_reduce_scatter_context.h
 * \brief
 */
#ifndef MC2_QUANT_REDUCE_SCATTER_CONTEXT_H
#define MC2_QUANT_REDUCE_SCATTER_CONTEXT_H

#include <cstdint>

namespace Mc2Aclnn {

constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;

struct Mc2QuantReduceScatterContext {
    uint64_t workSpace; // client和server之间通信的地址
    uint64_t workSpaceSize; // client和server之间通信的空间大小
    uint32_t rankId; // 当前卡rankId
    uint32_t rankDim; // 总卡数
    uint64_t winSize; // ccu不使用
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM]; // ccu不使用, MTE 数据区
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM]; // ccu不使用，MTE 状态区
};

}

#endif // MC2_QUANT_REDUCE_SCATTER_CONTEXT_H