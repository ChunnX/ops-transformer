/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/** A5 C8 SIMT miss-slot repair, adapted from lim_simt_postprocess.h. */
#ifndef VLLM_A5_LI_MANAGE_C8_FAST_SLOT_REPAIR_H
#define VLLM_A5_LI_MANAGE_C8_FAST_SLOT_REPAIR_H

#include "simt_api/asc_simt.h"

namespace VllmA5LiManageC8FastSlotRepair {

constexpr uint32_t THREADS = 256U;

__simt_callee__ inline int32_t LookupSlot(int32_t source, __ubuf__ const int32_t *unionSources,
                                          __ubuf__ const int32_t *unionDestinations, uint32_t unionCount)
{
    uint32_t left = 0U;
    uint32_t right = unionCount;
    while (left < right) {
        const uint32_t middle = left + (right - left) / 2U;
        if (unionSources[middle] < source)
            left = middle + 1U;
        else
            right = middle;
    }
    return left < unionCount && unionSources[left] == source ? unionDestinations[left] : -1;
}

__simt_vf__ __launch_bounds__(256) inline void RepairMissSlots(__ubuf__ const int32_t *routePairs,
                                                               __ubuf__ const int32_t *unionSources,
                                                               __ubuf__ const int32_t *unionDestinations,
                                                               __ubuf__ int32_t *output, uint32_t length,
                                                               uint32_t unionCount)
{
    for (uint32_t index = threadIdx.x; index < length; index += blockDim.x)
        output[index] = LookupSlot(routePairs[index * 2U + 1U], unionSources, unionDestinations, unionCount);
}

} // namespace VllmA5LiManageC8FastSlotRepair
#endif
