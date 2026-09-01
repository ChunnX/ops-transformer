/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_DEBUG_STAGE
#define SAGE_ATTENTION2_DEBUG_STAGE 0
#endif

#include <type_traits>
#include "kernel_operator.h"
#include "sage_attention2_template_tiling_key.h"
#include "sage_attention2_common.h"
#include "sage_attention2_preprocess.h"
#include "sage_attention2_catlass.h"

using namespace AscendC;
using namespace SageAttention2Kernel;

template <bool INPUT_BF16, bool LAYOUT_NHD, bool DPAD_128, bool IS_CAUSAL, bool PER_THREAD, uint32_t ACCUM_MODE,
    bool SMOOTH_K, bool SMOOTH_V, bool RETURN_LSE>
__global__ __aicore__ void sage_attention2(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if SAGE_ATTENTION2_DEBUG_STAGE == 110
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    return;
#elif SAGE_ATTENTION2_DEBUG_STAGE == 111
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    return;
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if SAGE_ATTENTION2_DEBUG_STAGE == 100
    // Current-driver launch probe: prove that an empty MIX_AIC_1_2 kernel can enter and exit normally.
    return;
#elif SAGE_ATTENTION2_DEBUG_STAGE == 101
    // Current-driver barrier probe: isolate SyncAll from tiling loads, workspace access, and preprocessing.
    AscendC::SyncAll<false>();
    return;
#endif
    using InputT = typename std::conditional<INPUT_BF16, bfloat16_t, half>::type;
    constexpr uint32_t dPad = DPAD_128 ? 128U : 64U;

    GET_TILING_DATA_WITH_STRUCT(SageAttention2TilingData, tilingDataIn, tiling);
    const SageAttention2TilingData *__restrict tilingData = &tilingDataIn;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    // One launch is intentionally used for all phases. This removes two host launches and keeps quantized Q/K/V in
    // the same workspace allocation. Batch scheduling in host tiling is mandatory for the two global barriers.
    CatlassAttention<InputT, dPad, LAYOUT_NHD, IS_CAUSAL, PER_THREAD, ACCUM_MODE, SMOOTH_K, SMOOTH_V,
        RETURN_LSE> attention;
    Preprocess<InputT, LAYOUT_NHD, PER_THREAD, SMOOTH_K, SMOOTH_V> preprocess;
    preprocess.Init(query, key, value, user, tilingData, attention.GetResource());
    preprocess.ComputeStats();
    AscendC::GlobalTensor<uint16_t> workspaceFence;
    workspaceFence.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(user));
    if ASCEND_IS_AIV {
        if constexpr (LAYOUT_NHD) {
            AscendC::DataCacheCleanAndInvalid<uint16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                AscendC::DcciDst::CACHELINE_OUT>(workspaceFence);
        }
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        if constexpr (LAYOUT_NHD) {
            AscendC::DataCacheCleanAndInvalid<uint16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                AscendC::DcciDst::CACHELINE_OUT>(workspaceFence);
        }
    }
#if SAGE_ATTENTION2_DEBUG_STAGE == 1
    return;
#endif
    preprocess.Quantize();
    if ASCEND_IS_AIV {
        AscendC::DataCacheCleanAndInvalid<uint16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(workspaceFence);
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        if constexpr (LAYOUT_NHD) {
            AscendC::DataCacheCleanAndInvalid<uint16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                AscendC::DcciDst::CACHELINE_OUT>(workspaceFence);
        }
    }
#if SAGE_ATTENTION2_DEBUG_STAGE == 2 || SAGE_ATTENTION2_DEBUG_STAGE == 21 || \
    SAGE_ATTENTION2_DEBUG_STAGE == 22
    return;
#endif

    attention.Init(query, attentionOut, softmaxLse, user, tilingData);
#if SAGE_ATTENTION2_DEBUG_STAGE == 3
    return;
#endif
    attention.Process();
#endif
}
