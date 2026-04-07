/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe.h
 * \brief
 */
#ifndef MEGA_MOE_H
#define MEGA_MOE_H


#include "gmmsq_base.h"
#include "kernel_gmm_swiglu_mxquant.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "tile_copy.hpp"
#include "mega_moe_tiling_def.h"
#include "hccl_shmem.h"
#include "workspace_info.h"

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

using namespace Catlass;
using namespace tla;

namespace MegaMoeImpl {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};

class MegaMoe {
public:
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR weight1, GM_ADDR weightScales1,
                                GM_ADDR weight2, GM_ADDR weightScales2, GM_ADDR probs, GM_ADDR y, GM_ADDR workspace,
                                const MegaMoeTilingData *tilingData); // TODO: buff[LCOC_MAX_RANK_SIZE] 要改成context
    __aicore__ inline void Process();
};

__aicore__ inline void MegaMoe::Init(
            GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR weight1, GM_ADDR weightScales1,
            GM_ADDR weight2, GM_ADDR weightScales2, GM_ADDR probs, GM_ADDR y, GM_ADDR workspace,
            const MegaMoeTilingData *tilingData) 
{

};


__aicore__ inline void MegaMoe::Process() {};
  // TODO: 代码结构要改
}  // namespace MegaMoeImpl
#endif