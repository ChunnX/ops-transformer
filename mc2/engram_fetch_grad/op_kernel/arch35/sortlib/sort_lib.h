/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * \file sort_lib.h
 * \brief SortLib public API — the only header an external operator needs to include.
 *        Include this file and call SortInvoke() to use the SIMD radix sort.
 *
 * \public
 */

#ifndef SORT_LIB_H
#define SORT_LIB_H

#include "kernel_operator.h"
struct NoOpHook {
    __aicore__ inline void operator()() const {}
};
#include "sort_lib/sort_lib_core.h"
#include "sort_lib/sort_lib_one_core.h"

namespace SortLib {

/**
 * \brief SIMD radix sort entry (the only kernel-side API of SortLib).
 *        Stable LSD radix sort of x, producing sortedValues and the permutation
 *        sortedIndices (sortedValues[i] == x[sortedIndices[i]]).
 *
 * \tparam ValT        key type (int8/16/32/64, uint8/16/32/64, half, bfloat16_t, float)
 * \tparam IdxT        index output type (int32 or int64), must match the host-side indexSize
 * \tparam CountT      count/offset type (uint32 or int64), must match the host-side counter size
 * \tparam isDescending sort in descending order
 *
 * \param pipe         caller-owned TPipe instance
 * \param x            input array (GM, length p.totalElements)
 * \param sortedValues output: sorted values (GM, length p.totalElements)
 * \param sortedIndices output: sorted permutation (GM, length p.totalElements)
 * \param workspace    GM workspace, size >= workspaceBytes, 32B-aligned base address
 * \param p            SortParams, filled by the host-side tiling function
 *
 * \note p.totalElements == 0 returns immediately without touching outputs/workspace.
 * \note The sort is stable: equal keys keep their original relative order.
 * \note This API allocates UB through the given pipe. Call pipe.Reset() before and after the
 *       invocation so it can use the full UB and later code gets its UB back.
 *
 * Typical usage (host computes tiling, kernel calls the sort):
 *   [host]   Derive numTileData/tileCount/activeCores/tmpUbSize/totalElements/isSingleCore from
 *            ubSize/coreCount/element count (see engram_fetch_grad_tiling.cpp for a working example),
 *            fill them into the tiling data and allocate the GM workspace accordingly;
 *            context->SetScheduleMode(1);  // kernel uses cross-core SyncAll, needs batch mode
 *            so the op owns all required cores; concurrent multi-stream/multi-op runs may deadlock.
 *   [kernel] Fill SortParams from the tiling data;
 *            pipe.Reset();   // release UB held by earlier stages so the sort can use the full UB
 *            SortLib::SortInvoke<ValT, IdxT, CountT, false>(&pipe, x, outVal, outIdx, ws, p);
 *            pipe.Reset();   // release the sort's UB for the following stages
 */
template <typename ValT, typename IdxT, typename CountT, bool isDescending, typename HookT = NoOpHook>
__aicore__ inline void SortInvoke(AscendC::TPipe *pipe, __gm__ ValT *x, __gm__ ValT *sortedValues,
                                  __gm__ IdxT *sortedIndices, __gm__ char *workspace, const SortParams &p,
                                  HookT hook = HookT())
{
    if (p.totalElements == 0) {
        return; // empty input: nothing to sort, skip workspace clearing (avoids zero-length GM writes)
    }
    if (p.isSingleCore == 1) {
        detail::SortRadixOneCore<ValT, IdxT, isDescending> op;
        op.Init(reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(sortedValues),
                reinterpret_cast<GM_ADDR>(sortedIndices), reinterpret_cast<GM_ADDR>(workspace), p, pipe);
        op.Process();
        return;
    }

    detail::SortRadixMoreCore<ValT, IdxT, CountT, isDescending> op;
    op.Init(reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(sortedValues),
            reinterpret_cast<GM_ADDR>(sortedIndices), reinterpret_cast<GM_ADDR>(workspace), p, pipe);
    op.Process(hook);
}

} // namespace SortLib

#endif
