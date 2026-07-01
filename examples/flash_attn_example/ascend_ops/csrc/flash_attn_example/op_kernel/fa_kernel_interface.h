/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fa_kernel_interface.h
 * \brief FA kernel — connects the host tiling data to the
 *        arch35 GQA kernel implementation.
 */

#ifndef FA_KERNEL_INTERFACE_H
#define FA_KERNEL_INTERFACE_H

#include <cstddef>

#include "kernel/fa_kernel.h"

using optiling::NoQuantTilingArch35;
using optiling::FlashAttnTilingData;

__aicore__ inline void CopyTiling(FlashAttnTilingData *tilingData, __gm__ uint8_t *tilingGM)
{
    constexpr int64_t wordCount = sizeof(FlashAttnTilingData) / sizeof(int64_t);
    constexpr int64_t tailBytes = sizeof(FlashAttnTilingData) % sizeof(int64_t);

    auto ptr = reinterpret_cast<int64_t *>(tilingData);
    auto tiling64 = reinterpret_cast<__gm__ int64_t *>(tilingGM);
    for (int64_t i = 0; i < wordCount; i++, ptr++) {
        *ptr = *(tiling64 + i);
    }
    if constexpr (tailBytes != 0) {
        auto ptr8 = reinterpret_cast<uint8_t *>(tilingData) + wordCount * sizeof(int64_t);
        auto tiling8 = tilingGM + wordCount * sizeof(int64_t);
        for (int64_t i = 0; i < tailBytes; i++) {
            ptr8[i] = tiling8[i];
        }
    }
    return;
}

template <bool hasAttnMask = false, bool isCombine = false,
    fa_kernel::config::S1TemplateType s1TemplateType = fa_kernel::config::S1TemplateType::Aligned128,
    fa_kernel::config::S2TemplateType s2TemplateType = fa_kernel::config::S2TemplateType::Aligned128>
inline __aicore__ void FaKernelInterface(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attnMask, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    FlashAttnTilingData tilingDataTemp;
    if (tiling != nullptr) {
        CopyTiling(&tilingDataTemp, tiling);
    }
    __gm__ uint8_t *faMetaData = nullptr;
    if (tiling != nullptr) {
        faMetaData = tiling + offsetof(FlashAttnTilingData, faMetaData);
    }

    TPipe tPipe;
    constexpr auto INPUT_LAYOUT = fa_kernel::layout::LayOutTypeEnum::LAYOUT_BSH;
    constexpr auto OUTPUT_LAYOUT = fa_kernel::layout::LayOutTypeEnum::LAYOUT_BSH;

    using CubeBlockType = BaseApi::FABlockCube<
        bfloat16_t, float, INPUT_LAYOUT,
        s1TemplateType, s2TemplateType,
        fa_kernel::config::DTemplateType::Aligned128,
        fa_kernel::config::DTemplateType::Aligned128>;

    using CubeBlockDummyType = BaseApi::FABlockCubeDummy<
        bfloat16_t, float, INPUT_LAYOUT,
        s1TemplateType, s2TemplateType,
        fa_kernel::config::DTemplateType::Aligned128,
        fa_kernel::config::DTemplateType::Aligned128>;

    using VecFaBlockType = BaseApi::FABlockVec<
        bfloat16_t, float, bfloat16_t, INPUT_LAYOUT, OUTPUT_LAYOUT,
        s1TemplateType, s2TemplateType,
        fa_kernel::config::DTemplateType::Aligned128,
        fa_kernel::config::DTemplateType::Aligned128,
        hasAttnMask, isCombine>;

    using VecFaBlockDummyType = BaseApi::FABlockVecDummy<
        bfloat16_t, float, bfloat16_t, INPUT_LAYOUT, OUTPUT_LAYOUT,
        s1TemplateType, s2TemplateType,
        fa_kernel::config::DTemplateType::Aligned128,
        fa_kernel::config::DTemplateType::Aligned128,
        hasAttnMask, isCombine>;

    using VecCombineBlockType = BaseApi::FaBlockVecCombine<
        bfloat16_t, float, bfloat16_t, INPUT_LAYOUT, OUTPUT_LAYOUT,
        s1TemplateType, s2TemplateType,
        fa_kernel::config::DTemplateType::Aligned128,
        fa_kernel::config::DTemplateType::Aligned128,
        hasAttnMask>;

    using VecCombineBlockDummyType = BaseApi::FaBlockVecCombineDummy<
        bfloat16_t, float, bfloat16_t, INPUT_LAYOUT, OUTPUT_LAYOUT,
        s1TemplateType, s2TemplateType,
        fa_kernel::config::DTemplateType::Aligned128,
        fa_kernel::config::DTemplateType::Aligned128,
        hasAttnMask>;

    __gm__ uint8_t *nullPtr = nullptr;
    using RealCubeBlock = typename std::conditional<g_coreType == AscendC::AIC, CubeBlockType, CubeBlockDummyType>::type;
    using RealVecFaBlock = typename std::conditional<g_coreType == AscendC::AIC, VecFaBlockDummyType, VecFaBlockType>::type;
    using RealVecCombineBlock = typename std::conditional<g_coreType == AscendC::AIV, VecCombineBlockType, VecCombineBlockDummyType>::type;
    BaseApi::FlashAttentionKernel<RealCubeBlock, RealVecFaBlock, RealVecCombineBlock> op;
    op.Init(query, key, value, attnMask, nullPtr, nullPtr, attentionOut, workspace, faMetaData,
            &tilingDataTemp.baseTiling, &tPipe);
    op.Process();
}

#endif  // FA_KERNEL_INTERFACE_H
