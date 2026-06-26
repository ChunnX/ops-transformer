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
 * \file moe_ep_combine_tiling.h
 * \brief
 */
 
#ifndef MOE_EP_COMBINE_TILING_H
#define MOE_EP_COMBINE_TILING_H

#if __has_include("../moe_ep_dispatch/moe_ep_dispatch_tiling.h")
#include "../moe_ep_dispatch/moe_ep_dispatch_tiling.h"
#else
#include "../../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"
#endif

struct MoeEpCombineInfo {
    MoeEpCommonTilingData cfg;
    uint32_t hasTopkWeights     = 0;
    uint32_t aivNum             = 0;
    uint64_t totalWinSizeEp     = 0;
    uint64_t totalUbSize        = 0;
    uint64_t localWinSizeEp     = 0;  // per-rank local workspace stride (compact, not CCL window size)
};

struct MoeEpCombineTilingData {
    MoeEpCombineInfo moeEpCombineInfo;
};



#endif  // MOE_EP_COMBINE_TILING_H