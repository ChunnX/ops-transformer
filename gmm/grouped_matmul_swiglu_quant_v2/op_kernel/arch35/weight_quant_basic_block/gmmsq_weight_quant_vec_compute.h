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
 * \file gmmsq_weight_quant_vec_compute.h
 * \brief
 */
#ifndef GMMSQ_WEIGHT_QUANT_VEC_COMPUTE_H
#define GMMSQ_WEIGHT_QUANT_VEC_COMPUTE_H

#include "kernel_cube_intf.h"
#include "op_kernel/math_util.h"
#include "../common/basic_block_config.h"
#include "../common/basic_block_vf_mx.h"
#include "../common/tool.h"
#include "gmmsq_mx_a8w4_vf.h"
#include "gmmsq_weight_quant_cube_compute_tools.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::TypeGet;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockOffsetParam;
using WeightQuantBatchMatmulV2::Arch35::C0_SIZE_B8;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::DataCopyPad2D;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using WeightQuantBatchMatmulV2::Arch35::MxA8W4NzParams;
using WeightQuantBatchMatmulV2::Arch35::QUADRUPLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig;
using WeightQuantBatchMatmulV2::Arch35::VEC_MAX_ELEM_B16;
using WeightQuantBatchMatmulV2::Arch35::WqmmConfig;
using WeightQuantBatchMatmulV2::Arch35::GmmsqFrAntiQuantMxA8W4NzNkVf;
using WeightQuantBatchMatmulV2::Arch35::GmmsqSwiGLUVf;
using WeightQuantBatchMatmulV2::Arch35::GmmsqComputeMaxExpVf;
using WeightQuantBatchMatmulV2::Arch35::GmmsqComputeScaleVf;
using WeightQuantBatchMatmulV2::Arch35::GmmsqComputeDataForQuantTargetFp8Vf;

namespace GMMSQ {

struct GmmsqUbBufferInfo {
    uint64_t weightHighBitBufferNum;
    uint64_t weightHighBitTotalSize;
    uint64_t weightHighBitSingleBufferSize;
    uint64_t weightLowbitTotalSize;
    uint64_t weightLowBitSingleBufferSize;
    uint64_t biasTotalSize;
    uint64_t biasSingleBufferSize;
    uint64_t biasRescaleTotalSize;
    uint64_t biasRescaleSingleBufferSize;
    uint64_t maxB16TotalSize;
    uint64_t scaleE8M0TotalSize;
    uint64_t yFp8TotalSize;
};

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr GmmsqUbBufferInfo GetGMMSQMxA8W4BufferInfo()
{
    return {
        .weightHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
        .weightHighBitTotalSize = 128 * GetKBUnit<int8_t>(),
        .weightHighBitSingleBufferSize = 128 * GetKBUnit<int8_t>() / QUADRUPLE_BUFFER_NUM,
        .weightLowbitTotalSize = 64 * GetKBUnit<int8_t>(),
        .weightLowBitSingleBufferSize = 64 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
        .biasTotalSize = 1 * GetKBUnit<half>(),
        .biasSingleBufferSize = 1 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
        .biasRescaleTotalSize = 1 * GetKBUnit<half>(),
        .biasRescaleSingleBufferSize = 1 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
        .maxB16TotalSize = 2 * GetKBUnit<half>(),
        .scaleE8M0TotalSize = 16 * GetKBUnit<fp8_e8m0_t>(),
        .yFp8TotalSize = 32 * GetKBUnit<fp8_e4m3fn_t>()
    };
}

#define GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM                                                        \
    template <typename xType, typename wType, typename biasType, typename yType,                   \
              typename yScaleType, bool hasBias, const WqmmConfig &wqmmConfig,                     \
              const VecAntiQuantConfig &vecConfig>

#define GMMSQ_WQ_VEC_COMPUTE_CLASS                                                                  \
    GMMSQWeightQuantVecCompute<xType, wType, biasType, yType, yScaleType,                          \
                               hasBias, wqmmConfig, vecConfig>

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
class GMMSQWeightQuantVecCompute {
public:
    __aicore__ inline GMMSQWeightQuantVecCompute(){};

    __aicore__ inline void Init(__gm__ yType *yFp32Addr, __gm__ yScaleType *yScaleAddr);
    __aicore__ inline void UpdateGlobalAddr(__gm__ wType *weight, __gm__ biasType *bias, const bool weightL2Cacheable);
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset,
                                      uint64_t biasGmOffset, uint64_t biasRealSize,
                                      const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WeightAntiQuantComputeNzNk(uint64_t kRealSize, uint64_t biasRealSize, uint64_t kGmOffset,
                                                      const LocalTensor<xType> &weightHighBitL1,
                                                      const LocalTensor<biasType> &biasL1,
                                                      const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SwiGLU(const LocalTensor<float> &yF32, const BasicBlockOffsetParam &param);
    __aicore__ inline void ComputeMaxExp(const LocalTensor<float> &yF32, const BasicBlockOffsetParam &param);
    __aicore__ inline void ComputeScale(const BasicBlockOffsetParam &param);
    __aicore__ inline void ComputeDataForQuantTargetFp8(const LocalTensor<float> &yF32,
                                                        const BasicBlockOffsetParam &param);
    __aicore__ inline void CopyOutputFromUb2Gm(const BasicBlockOffsetParam &param);
    __aicore__ inline void CopyScaleFromUb2Gm(const BasicBlockOffsetParam &param);
    __aicore__ inline void SetMTE3ToV();
    __aicore__ inline void End();

private:
    __aicore__ inline void AntiQuantProcessNzMxA8W4(uint64_t biasRealSize, uint64_t ubMte2KSize, uint64_t kGmOffset,
                                                    const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t antiQuantRealN, uint64_t antiQuantRealK,
                                                       const LocalTensor<xType> &weightHighBitL1);

    constexpr static GmmsqUbBufferInfo UB_BUFFER_INFO = GetGMMSQMxA8W4BufferInfo<vecConfig>();

    uint64_t weightMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;

    static constexpr uint32_t EVENT_ID_V_TO_MTE2 = 0;
    static constexpr uint32_t EVENT_ID_MTE2_TO_V = 0;
    static constexpr uint32_t EVENT_ID_MTE3_TO_V = 0;
    static constexpr uint32_t EVENT_ID_V_TO_MTE3 = 0;

    static constexpr uint32_t QUADRUPLE_BUFFER_MASK = QUADRUPLE_BUFFER_NUM - 1;

    GlobalTensor<wType> wGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yFp32Global_;
    GlobalTensor<yScaleType> yScaleGlobal_;

    LocalTensor<int8_t> weightLowBit_;
    LocalTensor<xType> weightHighBit_;
    LocalTensor<biasType> bias_;
    LocalTensor<biasType> biasRescale_;
    LocalTensor<half> maxB16Buffer_;
    LocalTensor<fp8_e8m0_t> scaleE8M0Buffer_;
    LocalTensor<fp8_e4m3fn_t> yFp8Buffer_;

    constexpr static uint32_t C0_SIZE = C0_SIZE_B8;
    constexpr static uint64_t VEC_REG_ELEM = VECTOR_REG_WIDTH;
    constexpr static uint64_t UB_AVAILABLE_SIZE = 248 * GetKBUnit<int8_t>();
};

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::Init(__gm__ yType *yFp32Addr, __gm__ yScaleType *yScaleAddr)
{
    yFp32Global_.SetGlobalBuffer(yFp32Addr);
    yScaleGlobal_.SetGlobalBuffer(yScaleAddr);

    weightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightLowbitTotalSize);
    uint64_t ubOffset = UB_BUFFER_INFO.weightLowbitTotalSize;

    weightHighBit_ = LocalTensor<xType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.weightHighBitTotalSize);
    ubOffset += UB_BUFFER_INFO.weightHighBitTotalSize;

    bias_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasTotalSize);
    ubOffset += UB_BUFFER_INFO.biasTotalSize * sizeof(biasType);

    biasRescale_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasRescaleTotalSize);
    ubOffset += UB_BUFFER_INFO.biasRescaleTotalSize * sizeof(biasType);

    maxB16Buffer_ = LocalTensor<half>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.maxB16TotalSize);
    ubOffset += UB_BUFFER_INFO.maxB16TotalSize * sizeof(half);

    scaleE8M0Buffer_ = LocalTensor<fp8_e8m0_t>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.scaleE8M0TotalSize);

    yFp8Buffer_ = weightLowBit_.template ReinterpretCast<fp8_e4m3fn_t>();

    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::UpdateGlobalAddr(__gm__ wType *weight, __gm__ biasType *bias,
                                                                    const bool weightL2Cacheable)
{
    wGlobal_.SetGlobalBuffer(weight);
    if (!weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    if constexpr (hasBias) {
        biasGlobal_.SetGlobalBuffer(bias);
    }
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::WaitVToMTE2()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::SetVToMTE2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
    weightMte2LoopIdx_++;
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset,
                                                               uint64_t kL1Offset, uint64_t biasGmOffset,
                                                               uint64_t biasRealSize,
                                                               const BasicBlockOffsetParam &offsetParam)
{
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            DataCopyPad2D(bias_[(weightMte2LoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasSingleBufferSize],
                          biasGlobal_[biasGmOffset], 1, biasRealSize, biasRealSize, biasRealSize);

            SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        }
    }

    if (offsetParam.nL1Size == 0 || ubMte2KSize == 0) {
        return;
    }

    uint64_t swishNOffset = offsetParam.nOffset * static_cast<uint64_t>(C0_SIZE);
    uint64_t gateNOffset = (offsetParam.nOffset + offsetParam.nSize / 2) * static_cast<uint64_t>(C0_SIZE);
    uint64_t kOffset = (kGmOffset + kL1Offset) * offsetParam.nAlign;

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize]
            .template ReinterpretCast<wType>(),
        wGlobal_[kOffset + swishNOffset],
        CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        128 * C0_SIZE,
        256 * C0_SIZE,
        offsetParam.nAlign * C0_SIZE);

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize +
                      128 * C0_SIZE]
            .template ReinterpretCast<wType>(),
        wGlobal_[kOffset + gateNOffset],
        CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        128 * C0_SIZE,
        256 * C0_SIZE,
        offsetParam.nAlign * C0_SIZE);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::WeightAntiQuantComputeNzNk(
    uint64_t kRealSize, uint64_t biasRealSize, uint64_t kGmOffset, const LocalTensor<xType> &weightHighBitL1,
    const LocalTensor<biasType> &biasL1, const BasicBlockOffsetParam &offsetParam)
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));

    AntiQuantProcessNzMxA8W4(biasRealSize, kRealSize, kGmOffset, offsetParam);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    if (likely(kRealSize > 0)) {
        CopyWeightHighBitForAligned(offsetParam.nL1Size, kRealSize, weightHighBitL1);
    }
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            DataCopy(
                biasL1,
                biasRescale_[(ubComputeLoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasRescaleSingleBufferSize],
                biasRealSize);
        }
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));
    ubComputeLoopIdx_++;
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::AntiQuantProcessNzMxA8W4(
    uint64_t biasRealSize, uint64_t ubMte2KSize, uint64_t kGmOffset, const BasicBlockOffsetParam &offsetParam)
{
    MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1);
    mxA8W4NzParams.nRealSizeAlign = CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType *)weightLowBit_[ubMte2BufferIdx * UB_BUFFER_INFO.weightLowBitSingleBufferSize].GetPhyAddr();

    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType *)
            weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VECTOR_REG_WIDTH]
                .GetPhyAddr();
    mxA8W4NzParams.loopKNum = CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE));
    mxA8W4NzParams.innerLoopNum =
        CeilDivide(CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                   static_cast<uint64_t>(VECTOR_REG_WIDTH));
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.weightHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;

    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            mxA8W4NzParams.biasInUbAddr = (__ubuf__ biasType *)bias_.GetPhyAddr(
                (weightMte2LoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasSingleBufferSize);
            mxA8W4NzParams.biasOutUbAddr = (__ubuf__ biasType *)biasRescale_.GetPhyAddr(
                (ubComputeLoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasRescaleSingleBufferSize);
            mxA8W4NzParams.biasLoopNum = CeilDivide(biasRealSize, VEC_MAX_ELEM_B16);
            GmmsqFrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, true>(mxA8W4NzParams);
        } else {
            GmmsqFrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
        }
    } else {
        GmmsqFrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
    }
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyWeightHighBitForAligned(
    uint64_t antiQuantRealN, uint64_t antiQuantRealK, const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(C0_SIZE)) *
                        CeilAlign(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE)) / VEC_REG_ELEM;

    params.blockLen = VEC_REG_ELEM / ONE_BLK_SIZE;
    params.srcStride = (UB_BUFFER_INFO.weightHighBitBufferNum - 1) * params.blockLen;
    params.dstStride = 0;
    DataCopy(weightHighBitL1,
             weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VEC_REG_ELEM], params);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::SwiGLU(const LocalTensor<float> &yF32,
                                                           const BasicBlockOffsetParam &param)
{
    uint64_t mRealSize = param.mL1Size;
    uint64_t nHalfSize = param.nL1Size / 2;

    __ubuf__ float *yAddr = (__ubuf__ float *)yF32.GetPhyAddr();
    __ubuf__ float *gateAddr = yAddr + mRealSize * nHalfSize;
    GmmsqSwiGLUVf(static_cast<uint16_t>(mRealSize), static_cast<uint16_t>(nHalfSize), yAddr, gateAddr);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::ComputeMaxExp(const LocalTensor<float> &yF32,
                                                                  const BasicBlockOffsetParam &param)
{
    uint64_t mRealSize = param.mL1Size;
    uint64_t nGroupCount = CeilDivide(param.nL1Size / 2, static_cast<uint64_t>(MX_GROUPSIZE));

    GmmsqComputeMaxExpVf(static_cast<uint16_t>(mRealSize), nGroupCount, (__ubuf__ float *)yF32.GetPhyAddr(),
                         (__ubuf__ half *)maxB16Buffer_.GetPhyAddr());
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::ComputeScale(const BasicBlockOffsetParam &param)
{
    uint64_t mRealSize = param.mL1Size;
    uint64_t nGroupCount = CeilDivide(param.nL1Size / 2, static_cast<uint64_t>(MX_GROUPSIZE));

    GmmsqComputeScaleVf(static_cast<uint16_t>(mRealSize), nGroupCount, (__ubuf__ half *)maxB16Buffer_.GetPhyAddr(),
                        (__ubuf__ fp8_e8m0_t *)scaleE8M0Buffer_.GetPhyAddr());
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::ComputeDataForQuantTargetFp8(const LocalTensor<float> &yF32,
                                                                                 const BasicBlockOffsetParam &param)
{
    uint64_t mRealSize = param.mL1Size;
    uint64_t nHalfSize = param.nL1Size / 2;

    GmmsqComputeDataForQuantTargetFp8Vf(static_cast<uint16_t>(mRealSize), nHalfSize,
                                        (__ubuf__ float *)yF32.GetPhyAddr(),
                                        (__ubuf__ fp8_e8m0_t *)scaleE8M0Buffer_.GetPhyAddr(),
                                        (__ubuf__ fp8_e4m3fn_t *)yFp8Buffer_.GetPhyAddr());
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyOutputFromUb2Gm(const BasicBlockOffsetParam &param)
{
    uint64_t mRealSize = param.mL1Size;
    uint64_t nHalfSize = param.nL1Size / 2;

    uint64_t yGmOffset = param.mOffset * (param.nSize / 2) + param.nOffset / 2;

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    DataCopyPad2D(yFp32Global_[yGmOffset], yFp8Buffer_, mRealSize, nHalfSize, nHalfSize, param.nSize / 2);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyScaleFromUb2Gm(const BasicBlockOffsetParam &param)
{
    uint64_t mRealSize = param.mL1Size;
    uint64_t nGroupCount = CeilDivide(param.nL1Size / 2, static_cast<uint64_t>(MX_GROUPSIZE));
    uint64_t nTotalGroupCount = CeilDivide(param.nSize / 2, static_cast<uint64_t>(MX_GROUPSIZE));

    uint64_t scaleGmOffset = param.mOffset * nTotalGroupCount + (param.nOffset / 2) / MX_GROUPSIZE;

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    DataCopyPad2D(yScaleGlobal_[scaleGmOffset], scaleE8M0Buffer_, mRealSize, nGroupCount, nGroupCount,
                  nTotalGroupCount);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::SetMTE3ToV()
{
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::End()
{
    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

}

#endif
