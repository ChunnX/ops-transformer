/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_asw_vec_compute.h
 * \brief
 */

#pragma once

#include "weight_quant_tool_asw.h"
#include "kernel_cube_intf.h"
#include "op_kernel/math_util.h"
#include "weight_quant_grouped_matmul_asw_block.h"
#include "gmmsq_quant_mx_vf.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_asw_cube_compute.h"

using AscendC::BLOCK_CUBE;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
using namespace WeightQuantTools;
using namespace WeightQuantGroupedMatmul::Arch35;

namespace GMMSQWeightQuantAsw {

struct GmmsqAswUbBufferInfo {
    uint64_t yfp32TotalSize;
    uint64_t gluResTotalSize;
    uint64_t maxExpTotalSize;
    uint64_t mxScaleTotalSize;
    uint64_t halfScaleTotalSize;
    uint64_t yFp8TotalSize;
    uint64_t scaleE8M0TotalSize;
};

static constexpr GmmsqAswUbBufferInfo ASW_UB_BUFFER_INFO = {
    .yfp32TotalSize = 256UL * 1024,
    .gluResTotalSize = 64UL * 1024,
    .maxExpTotalSize = 2UL * 1024,
    .mxScaleTotalSize = 1UL * 1024,
    .halfScaleTotalSize = 2UL * 1024,
    .yFp8TotalSize = 32UL * 1024,
    .scaleE8M0TotalSize = 8UL * 1024,
};

#define GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM template <typename yType, typename yScaleType>

#define GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS GMMSQWeightQuantASWVecCompute<yType, yScaleType>

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
class GMMSQWeightQuantASWVecCompute {
public:
    __aicore__ inline GMMSQWeightQuantASWVecCompute(){};
    __aicore__ inline void Init(LocalTensor<float> &yF32Buffer);
    __aicore__ inline void UpdateGlobalAddr(__gm__ yType *yAddr, __gm__ yScaleType *yScaleAddr);
    __aicore__ inline void SwiGLU(const ASWTilingParam &params);
    __aicore__ inline void MxQuant(const ASWTilingParam &params);
    __aicore__ inline void TransMxScaleLayout(const ASWTilingParam &params);
    __aicore__ inline void CopyOutputFromUb2Gm(const ASWTilingParam &params, const ASWOffsetParam &offset);
    __aicore__ inline void IterateNzNkWithAiv(const ASWTilingParam &params, const ASWOffsetParam &offset);

private:
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<yScaleType> yScaleGlobal_;

    LocalTensor<float> yFp32Buffer_;
    LocalTensor<half> maxB16Buffer_;
    LocalTensor<fp8_e8m0_t> scaleE8M0Buffer_;
    LocalTensor<fp8_e4m3fn_t> yFp8Buffer_;
    LocalTensor<bfloat16_t> gluRes_;
    LocalTensor<int8_t> quantScaleOutput_;
    LocalTensor<uint16_t> halfScale_;

    uint64_t nHalfSize_ = 0;
    uint32_t nDstUbAligned_ = 0;
    static constexpr uint32_t VL_FOR_HALF_NUMBER = VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    static constexpr uint32_t EVENT_ID_V_TO_MTE3 = 6;
    static constexpr uint32_t EVENT_ID_MTE3_TO_V = 7;
};

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS::Init(LocalTensor<float> &yF32Buffer)
{
    yFp32Buffer_ = yF32Buffer;

    uint64_t offset = ASW_UB_BUFFER_INFO.yfp32TotalSize;
    gluRes_ = LocalTensor<bfloat16_t>(AscendC::TPosition::LCM, offset,
                                      ASW_UB_BUFFER_INFO.gluResTotalSize / sizeof(bfloat16_t));
    offset += ASW_UB_BUFFER_INFO.gluResTotalSize;
    maxB16Buffer_ =
        LocalTensor<half>(AscendC::TPosition::LCM, offset, ASW_UB_BUFFER_INFO.maxExpTotalSize / sizeof(half));
    offset += ASW_UB_BUFFER_INFO.maxExpTotalSize;
    quantScaleOutput_ = LocalTensor<int8_t>(AscendC::TPosition::LCM, offset, ASW_UB_BUFFER_INFO.mxScaleTotalSize);
    offset += ASW_UB_BUFFER_INFO.mxScaleTotalSize;
    halfScale_ = LocalTensor<uint16_t>(AscendC::TPosition::LCM, offset,
                                       ASW_UB_BUFFER_INFO.halfScaleTotalSize / sizeof(uint16_t));
    offset += ASW_UB_BUFFER_INFO.halfScaleTotalSize;
    yFp8Buffer_ = LocalTensor<fp8_e4m3fn_t>(AscendC::TPosition::LCM, offset,
                                            ASW_UB_BUFFER_INFO.yFp8TotalSize / sizeof(fp8_e4m3fn_t));
    offset += ASW_UB_BUFFER_INFO.yFp8TotalSize;
    scaleE8M0Buffer_ = LocalTensor<fp8_e8m0_t>(AscendC::TPosition::LCM, offset,
                                               ASW_UB_BUFFER_INFO.scaleE8M0TotalSize / sizeof(fp8_e8m0_t));
}

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS::UpdateGlobalAddr(__gm__ yType *yAddr,
                                                                        __gm__ yScaleType *yScaleAddr)
{
    yGlobal_.SetGlobalBuffer(yAddr);
    yScaleGlobal_.SetGlobalBuffer(yScaleAddr);
}

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS::SwiGLU(const ASWTilingParam &params)
{
    __ubuf__ float *yAddr = (__ubuf__ float *)yFp32Buffer_.GetPhyAddr();
    __ubuf__ float *gateAddr = yAddr + nHalfSize_;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();

    uint32_t nSrcUbAligned = Ops::Base::CeilAlign(params.singleCoreN, ONE_BLK_SIZE / sizeof(float));
    constexpr uint16_t sizePerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t oneRowRepeatTimes = Ops::Base::CeilDiv(nHalfSize_, static_cast<uint64_t>(sizePerRepeat));

    GmmSwigluVf<float, false>(yAddr, gateAddr, gluResAddr, static_cast<uint16_t>(params.singleCoreM),
                              static_cast<uint16_t>(nHalfSize_), nSrcUbAligned, nDstUbAligned_, oneRowRepeatTimes,
                              sizePerRepeat);
}

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS::MxQuant(const ASWTilingParam &params)
{
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxB16Buffer_.GetPhyAddr();
    __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)yFp8Buffer_.GetPhyAddr();
    uint64_t totalDataInUb = params.singleCoreM * nDstUbAligned_;
    uint32_t totalScaleInUb = totalDataInUb / MX_GROUP_SIZE_32;
    // 32：一次载入两个VECTOR_REG_WIDTH，且每32个数取一个值，所以公式为VECTOR_REG_WIDTH / sizeof(bfloat16_t) *
    // 2 / 32 简化为 VECTOR_REG_WIDTH/32
    uint16_t elementAfterReduce = VECTOR_REG_WIDTH / 32U;
    uint16_t loopDataNum = Ops::Base::CeilDiv(totalDataInUb, static_cast<uint64_t>(VL_FOR_HALF_NUMBER) * 2);
    uint16_t loopScaleNum = Ops::Base::CeilDiv(totalScaleInUb, VL_FOR_HALF_NUMBER);
    // 0x0400: fp8的最大能取到的指数位8左移7之后的结果
    uint16_t fpEmax = 0x0400;

    ComputeMaxExpVf(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum, VL_FOR_HALF_NUMBER, elementAfterReduce);
    ComputeScaleVf(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUb, loopScaleNum, VL_FOR_HALF_NUMBER,
                   fpEmax);
    ComputeDataForQuantTargetFp8Vf<yType>(gluResAddr, halfScaleLocalAddr, outLocalAddr, totalDataInUb, loopDataNum,
                                          VL_FOR_HALF_NUMBER, elementAfterReduce);
}

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS::CopyOutputFromUb2Gm(const ASWTilingParam &params,
                                                                           const ASWOffsetParam &offset)
{
    uint64_t nGroupCount = Ops::Base::CeilDiv(params.singleCoreN / SWIGLU_N_HALF_FACTOR, MX_GROUP_SIZE_32);
    uint64_t nTotalGroupCount = Ops::Base::CeilDiv(params.n / SWIGLU_N_HALF_FACTOR, MX_GROUP_SIZE_32);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    DataCopyPad2D(yGlobal_[offset.offsetY], yFp8Buffer_, params.singleCoreM, nHalfSize_, nHalfSize_,
                  params.n / SWIGLU_N_HALF_FACTOR);
    TransMxScaleLayout(params);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    DataCopyPad2D(yScaleGlobal_[offset.offsetYScale], scaleE8M0Buffer_, params.singleCoreM, nGroupCount, nGroupCount,
                  nTotalGroupCount);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
}

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS::TransMxScaleLayout(const ASWTilingParam &params)
{
    uint64_t scaleBlockN = Ops::Base::CeilDiv(nHalfSize_, MX_SCALE_DIVISOR_SIZE) * MX_SCALE_LAST_DIM_2;

    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)scaleE8M0Buffer_.GetPhyAddr();

    TransMxScaleLayoutVf(quantScaleOutputInUbAddr, quantScaleBlockOutputInUbAddr,
                         static_cast<uint16_t>(params.singleCoreM), static_cast<uint16_t>(scaleBlockN));
}

GMMSQ_WQ_ASW_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_ASW_VEC_COMPUTE_CLASS::IterateNzNkWithAiv(const ASWTilingParam &params,
                                                                          const ASWOffsetParam &offset)
{
    nHalfSize_ = params.singleCoreN / SWIGLU_N_HALF_FACTOR;
    nDstUbAligned_ = Ops::Base::CeilAlign(nHalfSize_, static_cast<uint64_t>(ONE_BLK_SIZE));
    SwiGLU(params);
    MxQuant(params);
    CopyOutputFromUb2Gm(params, offset);
}

} // namespace GMMSQWeightQuantAsw
