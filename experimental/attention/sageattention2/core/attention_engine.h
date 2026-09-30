/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "types.h"
#include "fast_math.h"
#include "device_resource.h"
#include "scheduler/snake_scheduler.h"
#include "compute/fast_quant.h"
#include "compute/simd_softmax.h"
#include "compute/out_accumulator.h"
#include "pipeline/event_handshake.h"
#include "pipeline/async_megapipe.h"

namespace sage::core {

// Unified Streaming Attention Execution Engine
template <class T, bool WIDE>
class AttentionEngine {
public:
    __aicore__ static inline __attribute__((always_inline)) void Execute(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse,
        GM_ADDR workspace, const SageAttention2TilingData &tilingData)
    {
        static_assert(sage::is_supported_input_type_v<T>, "Unsupported input dtype for SageAttention2");
        const SageParams &params = tilingData.params;

        // Stage 1: Vector Preprocessing and Quantization on AIV cores
        if ASCEND_IS_AIV {
            sage::compute::FastQuant<T, WIDE>::ProcessStats(
                k, v, workspace + tilingData.mean, workspace + tilingData.vMean, params);

            if (params.smoothK || params.smoothV) {
                AscendC::SyncAll<true>();
            }

            // Quantize Key and Value. In WIDE mode, Q is streamed on-chip directly into L1
            sage::compute::FastQuant<T, WIDE>::Quantize(
                q, k, v, workspace + tilingData.mean, workspace + tilingData.vMean,
                workspace + tilingData.vMxScale, workspace + tilingData.q8,
                workspace + tilingData.k8, workspace + tilingData.v8,
                workspace + tilingData.qScale, workspace + tilingData.kScale, params);
        }

        // Barrier: Ensure K/V are committed to memory across all AIC/AIV cores
        AscendC::SyncAll<false>();

        // Stage 2: Attention Computation Loop with 4-stage async micro-pipeline
        sage::pipeline::AsyncMegaPipe<T, WIDE>::Execute(
            q, workspace + tilingData.k8, workspace + tilingData.v8,
            workspace + tilingData.q8,
            workspace + tilingData.qScale, workspace + tilingData.kScale,
            workspace + tilingData.vMxScale, workspace + tilingData.vMean,
            out, lse, params);
    }
};

} // namespace sage::core
