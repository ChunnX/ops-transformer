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
 * \file gmmsq_weight_quant_vcv_basic_block.h
 * \brief GMMSQ MxA8W4 VCV BasicBlock header file
 */
#ifndef GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
#define GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H

#include "kernel_cube_intf.h"
#include "op_kernel/math_util.h"
#include "../common/basic_block_config.h"
#include "../common/basic_block_vf_mx.h"
#include "weight_quant_tool.h"
#include "gmmsq_weight_quant_cube_compute.h"
#include "gmmsq_weight_quant_vec_compute.h"
#include "gmmsq_weight_quant_cube_compute_tools.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using namespace WeightQuantBatchMatmulV2::Arch35;

namespace GMMSQWeightQuant {

#define GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM                                                                        \
    template <typename xType, typename wType, typename weightScaleType, typename xScaleType, typename biasType,        \
              typename yType, typename yScaleType, bool hasBias, const WqmmConfig &wqmmConfig,                         \
              const VecAntiQuantConfig &vecConfig>

#define GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS                                                                                 \
    GMMSQWeightQuantVcvBasicBlock<xType, wType, weightScaleType, xScaleType, biasType, yType, yScaleType, hasBias,     \
                                  wqmmConfig, vecConfig>

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
class GMMSQWeightQuantVcvBasicBlock {
public:
    __aicore__ inline GMMSQWeightQuantVcvBasicBlock() = default;
    __aicore__ inline void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, __gm__ yScaleType *yScale);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight, __gm__ weightScaleType *weightScale,
                                            __gm__ biasType *bias, __gm__ xScaleType *xScale, __gm__ yType *y,
                                            __gm__ yScaleType *yScale, const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                             const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void End(const BasicBlockOffsetParam &curOffsetParam);

protected:
    __aicore__ inline void IterateNzNkWithAiv(const BasicBlockOffsetParam &curOffsetParam,
                                              const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void IterateNzNkWithKAic(const BasicBlockOffsetParam &curOffsetParam,
                                               const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void VecComputeNzNkWithStartLimit(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                                        const BasicBlockOffsetParam &curOffsetParam);

    template <const pipe_t pipe>
    __aicore__ inline void SetAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void SetAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    GMMSQWeightQuantVecCompute<xType, wType, biasType, yType, yScaleType, hasBias, wqmmConfig, vecConfig> vecCompute_;
    GMMSQWeightQuantCubeCompute<xType, biasType, weightScaleType, xScaleType, float, hasBias, wqmmConfig> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightL1DbOffset_ = 0;
    uint64_t biasL1DbOffset_ = 0;

    LocalTensor<xType> weightL1_;
    LocalTensor<biasType> biasL1_;
    LocalTensor<float> yF32Buffer_;
};

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::Init(uint64_t antiQuantGroupSize, __gm__ yType *y,
                                                            __gm__ yScaleType *yScale)
{
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;

    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;

    biasL1_ = LocalTensor<biasType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(biasType));

    constexpr uint64_t ubOffset = GetGMMSQMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
    constexpr uint64_t highBitSize = GetGMMSQMxA8W4BufferInfo<vecConfig>().weightHighBitTotalSize;
    yF32Buffer_ = LocalTensor<float>(TPosition::LCM, ubOffset, highBitSize);

    if ASCEND_IS_AIC {
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize, biasL1DbOffset_, biasL1_);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    } else {
        vecCompute_.Init(y, yScale);
    }
    cvLoopIdx_ = 0;
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::UpdateGlobalAddr(
    __gm__ xType *x, __gm__ wType *weight, __gm__ weightScaleType *weightScale, __gm__ biasType *bias,
    __gm__ xScaleType *xScale, __gm__ yType *y, __gm__ yScaleType *yScale, const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, reinterpret_cast<__gm__ float *>(y), reinterpret_cast<__gm__ biasType *>(bias),
                                      weightScale, xScale);
    } else {
        vecCompute_.UpdateGlobalAddr(weight, bias, weightL2Cacheable);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                                                         const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        IterateNzNkWithAiv(curOffsetParam, lastOffsetParam);
    } else {
        IterateNzNkWithKAic(curOffsetParam, lastOffsetParam);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithAiv(const BasicBlockOffsetParam &curOffsetParam,
                                                                          const BasicBlockOffsetParam &lastOffsetParam)
{
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;

    VecComputeNzNkWithStartLimit(kMte2Offset, Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size),
                                 curOffsetParam);

    if (curCvLoopIdx > 0) {
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

        vecCompute_.SwiGLU(yF32Buffer_, lastOffsetParam);
        vecCompute_.ComputeMaxExp(yF32Buffer_, lastOffsetParam);
        vecCompute_.ComputeScale(lastOffsetParam);
        vecCompute_.ComputeDataForQuantTargetFp8(yF32Buffer_, lastOffsetParam);
        vecCompute_.CopyOutputFromUb2Gm(lastOffsetParam, reinterpret_cast<__gm__ yType *>(lastOffsetParam.yGmAddr));
        vecCompute_.TransMxScaleLayout(lastOffsetParam);
        vecCompute_.CopyScaleFromUb2Gm(lastOffsetParam,
                                       reinterpret_cast<__gm__ yScaleType *>(lastOffsetParam.yScaleGmAddr));
    }

    VecComputeNzNkWithStartLimit(kMte2Offset, curOffsetParam.kSize, curOffsetParam);
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::VecComputeNzNkWithStartLimit(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                                             const BasicBlockOffsetParam &curOffsetParam)
{
    uint64_t kMte2BaseSize = curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM;

    for (; kMte2Offset < kMte2Limit; kMte2Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        vecCompute_.WaitVToMTE2();
        uint64_t biasL1Offset = GetSubBlockIdx() * MX_BIAS_SINGLE_VECTOR_SIZE;
        uint64_t biasGmOffset = curOffsetParam.nOffset + biasL1Offset;
        uint64_t biasRealSize = biasL1Offset >= curOffsetParam.nL1Size ? 0 :
                                biasGmOffset + MX_BIAS_SINGLE_VECTOR_SIZE >= curOffsetParam.nSize ?
                                                                         curOffsetParam.nSize - biasGmOffset :
                                                                         MX_BIAS_SINGLE_VECTOR_SIZE;
        uint64_t kL1Offset = GetSubBlockIdx() * kMte2BaseSize;
        uint64_t mte2RealK = kMte2Offset + kL1Offset >= kMte2Limit ? 0 :
                             kMte2Offset + kL1Offset + kMte2BaseSize >= kMte2Limit ?
                                                                     kMte2Limit - kMte2Offset - kL1Offset :
                                                                     kMte2BaseSize;

        vecCompute_.CopyGmToUb(mte2RealK, kMte2Offset, kL1Offset, biasGmOffset, biasRealSize, curOffsetParam);

        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        vecCompute_.WeightAntiQuantComputeNzNk(
            mte2RealK, biasRealSize, kMte2Offset,
            weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_ +
                      CeilAlign(curOffsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset],
            biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_ + biasL1Offset], curOffsetParam);
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        vecCompute_.SetVToMTE2();
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithKAic(const BasicBlockOffsetParam &curOffsetParam,
                                                                           const BasicBlockOffsetParam &lastOffsetParam)
{
    if (cvLoopIdx_ > 0) {
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        cubeCompute_.GetTensorC(yF32Buffer_, lastOffsetParam);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }
    uint64_t kMutilLoadL1Size = MX_SCALE_K_L1_SIZE;
    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize ?
                                    curOffsetParam.kSize - kbL1Offset :
                                    curOffsetParam.kbL1Size;
        if (kbL1Offset % MX_SCALE_K_L1_SIZE == 0) {
            kMutilLoadL1Size = (kbL1Offset + MX_SCALE_K_L1_SIZE) > curOffsetParam.kSize ?
                                   CeilAlign(curOffsetParam.kSize - kbL1Offset, K_ALIGNMENT64) :
                                   MX_SCALE_K_L1_SIZE;
        }
        cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);
        cubeCompute_.CopyMxScaleGmToL1(curOffsetParam, kbL1Offset);
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.CopyAAndBiasGmToL1(curOffsetParam, kbL1Offset, cvLoopIdx_);
        cubeCompute_.PadL1IfNotAlign(weightL1_, weightL1DbOffset_, cvLoopIdx_, kbL1RealSize, curOffsetParam.nL1Size);
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
        cubeCompute_.LaunchMatmul(weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_], kbL1Offset, kbL1RealSize,
                                  kMutilLoadL1Size, cvLoopIdx_, curOffsetParam);
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curOffsetParam);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::End(const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIC {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(yF32Buffer_, lastOffsetParam);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync();
    } else {
        if (cvLoopIdx_ > 0) {
            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

            vecCompute_.SwiGLU(yF32Buffer_, lastOffsetParam);
            vecCompute_.ComputeMaxExp(yF32Buffer_, lastOffsetParam);
            vecCompute_.ComputeScale(lastOffsetParam);
            vecCompute_.ComputeDataForQuantTargetFp8(yF32Buffer_, lastOffsetParam);
            vecCompute_.CopyOutputFromUb2Gm(lastOffsetParam, reinterpret_cast<__gm__ yType *>(lastOffsetParam.yGmAddr));
            vecCompute_.TransMxScaleLayout(lastOffsetParam);
            vecCompute_.CopyScaleFromUb2Gm(lastOffsetParam,
                                           reinterpret_cast<__gm__ yScaleType *>(lastOffsetParam.yScaleGmAddr));
        }
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        vecCompute_.End();
    }
}

} // namespace GMMSQWeightQuant

#endif // GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
