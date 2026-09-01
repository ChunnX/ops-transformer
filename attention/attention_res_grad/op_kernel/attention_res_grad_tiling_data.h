/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file attention_res_grad_tiling_data.h -- B-axis split tiling data. */
#ifndef _ATTENTION_RES_GRAD_TILING_DATA_KERNEL_H_
#define _ATTENTION_RES_GRAD_TILING_DATA_KERNEL_H_
struct AttentionResGradTilingData {
    int64_t  batchSize;               // B
    int64_t  numBlocks;               // N
    int64_t  totalBlocks;             // N+1
    int64_t  hiddenSize;              // H
    int64_t  coreBatchStart;
    int64_t  coreBatchEnd;
    int64_t  coreBatchCount;
    uint64_t gradScoreWeightWkspOff;  // this core's workspace byte offset
    int64_t  coreNum;                 // total AIV cores
    uint64_t perCoreWkspBytes;        // stride between cores' workspace regions (bytes)
};
#endif
