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
 * \file compressor.cpp
 * \brief
 */

#if (__CCE_AICORE__ == 220)
#include "arch22/compressor_kernel.h"
#include "arch22/compressor_kernel_perf.h"
#include "arch22/compressor_kernel_full_load.h"
#else
#include "arch35/compressor_kernel.h"
#include "arch35/compressor_kernel_full_load.h"
#endif

using namespace Compressor;

#if (__CCE_AICORE__ != 220)
__aicore__ inline bool ShouldSkipInactiveReadOnlyGroupedDecode(const optiling::CompressorTilingData &tilingData,
                                                               __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                                               __gm__ uint8_t *xCache)
{
    const auto &baseParams = tilingData.baseParams;
    const auto &groupedDecodeParams = tilingData.groupedDecodeParams;
    if (xCache == nullptr || seqUsed == nullptr || startPos == nullptr || groupedDecodeParams.xCacheReadSize == 0 ||
        groupedDecodeParams.xCacheUpdateBatchSize > 0 || baseParams.cmpRatio == 0) {
        return false;
    }

    GlobalTensor<int32_t> seqUsedGm;
    GlobalTensor<int32_t> startPosGm;
    seqUsedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqUsed));
    startPosGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(startPos));

    const uint32_t cmpRatio = baseParams.cmpRatio;
    for (uint32_t batchIdx = 0; batchIdx < baseParams.batchSize; ++batchIdx) {
        const uint32_t rawSeqUsed = static_cast<uint32_t>(seqUsedGm.GetValue(batchIdx));
        const uint32_t rawStartPos = static_cast<uint32_t>(startPosGm.GetValue(batchIdx));
        if (CrossesCompressionBoundary(rawStartPos, rawSeqUsed, cmpRatio)) {
            return false;
        }
    }
    return true;
}
#endif

#define INVOKE_COMPRESSOR_GENERAL_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData); \
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens, seqUsed, \
                startPos, cmpKvOut, workspace); \
        op.Process(); \
    } while (0)

#if (__CCE_AICORE__ != 220)
#define INVOKE_COMPRESSOR_XCACHE_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData); \
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens, seqUsed, \
                startPos, xCache, xCacheSlotMapping, cmpKvOut, workspace); \
        op.Process(); \
    } while (0)
#endif

template <uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId>
__global__ __aicore__ void compressor(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                      __gm__ uint8_t *stateCache, __gm__ uint8_t *ape, __gm__ uint8_t *normWeight,
                                      __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                      __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                      __gm__ uint8_t *xCache, __gm__ uint8_t *xCacheSlotMapping,
                                      __gm__ uint8_t *cmpKvOut, __gm__ uint8_t *stateCacheOut,
                                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::CompressorTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }
#if (__CCE_AICORE__ != 220)
    // LI reuses the KV Compressor's x_cache update.  On a normal decode round
    // with no completed compression group, it has neither cache-update nor
    // projection work.  Return before constructing TPipe or either kernel
    // template; mixed-phase batches continue when any row crosses a boundary.
    if (ShouldSkipInactiveReadOnlyGroupedDecode(tilingDataIn, seqUsed, startPos, xCache)) {
        return;
    }
#endif
    // xCache/xCacheSlotMapping are optional kernel args.  Null keeps the legacy
    // direct-x path; non-null enables grouped decode composition in arch35 kernels.
    const optiling::CompressorTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    constexpr auto xLayout = static_cast<X_LAYOUT>(XLayout);
    constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
    constexpr auto coff = static_cast<COFF>(Coff);
    constexpr auto rotaryMode = static_cast<ROTARY_MODE>(RotaryMode);
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);
#if (__CCE_AICORE__ == 220)
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelFullLoad, xLayout, xDtype, coff, rotaryMode);
    } else {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelPerf, xLayout, xDtype, coff, rotaryMode);
    }
#else
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelFullLoad, xLayout, xDtype, coff, rotaryMode, cacheMode);
    } else {
        INVOKE_COMPRESSOR_XCACHE_OP_IMPL(CompressorKernel, xLayout, xDtype, coff, rotaryMode, cacheMode);
    }
#endif
}
