/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attention_score_grad_kernel_base.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_H
#define QUANT_FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_H

#include "quant_flash_attention_score_grad_common.h"

namespace QuantFagBaseApi {

template <typename ChildClass, typename CubeBlockType, typename VecBlockType, uint8_t LAYOUT>
class QuantFlashAttentionScoreGradKernelBase {
public:
    ARGS_TRAITS;
    template <bool IS_MM1_MM2 = true>
    __aicore__ inline int64_t GetQueryOffset(QuantFagRunInfo &runInfo);
    template <bool IS_MM1_MM2 = true>
    __aicore__ inline int64_t GetKeyOffset(QuantFagRunInfo &runInfo);
    __aicore__ inline void SyncALLCores();
    template <const bool IS_DENSE = false>
    __aicore__ inline ChildClass *GetDerived()
    {
        return static_cast<ChildClass *>(this);
    }

    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;

protected:
    TPipe *pipe;

    // output global memory
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buf;
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buf;

    uint32_t vBlockIdx = 0;
    uint32_t cBlockIdx = 0;
    uint32_t vSubBlockIdx = 0;
    int64_t s2CvBegin = 0;

    QuantFagTilingType tilingData;
    QuantFagConstInfo constInfo;

    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType, uint8_t LAYOUT>
template <bool IS_MM1_MM2>
__aicore__ inline int64_t
QuantFlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType, LAYOUT>::GetQueryOffset(QuantFagRunInfo &runInfo)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t n2GD = constInfo.commonConstInfo.n2GD;
    int64_t gD = constInfo.commonConstInfo.gD;
    int64_t dSize = constInfo.commonConstInfo.dSize;
    int64_t n2GS1D = constInfo.commonConstInfo.n2GS1D;
    int64_t gS1D = constInfo.commonConstInfo.gS1D;
    int64_t s1D = constInfo.commonConstInfo.s1D;
    int64_t bN2GD = constInfo.commonConstInfo.bN2GD;
    int64_t bOffsetTmp = 0;

    if (constInfo.commonConstInfo.layoutType == BNGSD) {
        bOffset = runInfo.commonRunInfo.boIdx * n2GS1D;
        n2Offset = runInfo.commonRunInfo.n2oIdx * gS1D;
        gOffset = runInfo.commonRunInfo.goIdx * s1D;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * dSize;
    } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
        bOffset = runInfo.commonRunInfo.boIdx * n2GS1D;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * n2GD;
        n2Offset = runInfo.commonRunInfo.n2oIdx * gD;
        gOffset = runInfo.commonRunInfo.goIdx * dSize;
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType, uint8_t LAYOUT>
template <bool IS_MM1_MM2>
__aicore__ inline int64_t
QuantFlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType, LAYOUT>::GetKeyOffset(QuantFagRunInfo &runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    int64_t n2D = constInfo.commonConstInfo.n2D;
    int64_t n2S2D = constInfo.commonConstInfo.n2S2D;
    int64_t dSize = constInfo.commonConstInfo.dSize;
    int64_t bN2D = constInfo.commonConstInfo.bN2D;
    int64_t s2D = constInfo.commonConstInfo.s2D;
    int64_t s1D = constInfo.commonConstInfo.s1D;
    int64_t bN2GD = constInfo.commonConstInfo.bN2GD;
    int64_t bOffsetTmp = 0;

    if (constInfo.commonConstInfo.layoutType == BNGSD) {
        bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
        n2Offset = runInfo.commonRunInfo.n2oIdx * s2D;
        s2Offset = runInfo.s2CvBegin * dSize;
        if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
            runInfo.kGmS2SplitOffset = CUBE_BASEN / NUM_TWO * dSize;
            runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * dSize;
        }
    } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
        bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
        s2Offset = runInfo.s2CvBegin * n2D;
        n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
        if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
            runInfo.kGmS2SplitOffset = CUBE_BASEN / NUM_TWO * n2D;
            runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * n2D;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType, uint8_t LAYOUT>
__aicore__ inline void
QuantFlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType, LAYOUT>::SyncALLCores()
{
    SyncAll<false>();
}

} // namespace QuantFagBaseApi
#endif