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
 * \file k_pool_kernel.h
 * \brief
 */

#ifndef K_POOL_KERNEL
#define K_POOL_KERNEL

#include "k_pool_comm.h"
#include "k_pool_template_tiling_key.h"
#include "k_pool_kernel_perf.h"
#include "k_pool_tiling_data.h"
#include "k_pool_tools.h"

using namespace AscendC;

namespace KPool {

template <typename COMP>
class KPoolKernel {
public:
    __aicore__ inline KPoolKernel(TPipe* pipe, const optiling::KPoolTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *hidden_states,
        __gm__ uint8_t *wk,
        __gm__ uint8_t *gateWeight,
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *cacheBlockTable,
        __gm__ uint8_t *seqLens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *pooledKeyOut,
        __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

    // ==============================TilingData&TPipe==============================
    TPipe* pipe_;
    const optiling::KPoolTilingData* __restrict tilingData_;
};

template <typename COMP>
__aicore__ inline void KPoolKernel<COMP>::Init(
        __gm__ uint8_t *hidden_states,
        __gm__ uint8_t *wk,
        __gm__ uint8_t *gateWeight,
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *cacheBlockTable,
        __gm__ uint8_t *seqLens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *pooledKeyOut,
        __gm__ uint8_t *workspace)
{

}


template <typename COMP>
__aicore__ inline void KPoolKernel<COMP>::Process()
{

}

} // namespace KPool

#endif // K_POOL_KERNEL
