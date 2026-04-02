/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_mxfp8fp4_prologue_mx_cast_w.h
 * \brief
 */
#ifndef GROUPED_MATMUL_MXFP8FP4_PROLOGUE_MX_CAST_W_H
#define GROUPED_MATMUL_MXFP8FP4_PROLOGUE_MX_CAST_W_H

#include "../tile/basic_block_vf_mx.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "tool.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyParams;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::TPosition;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::TypeGet;

using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using WeightQuantBatchMatmulV2::Arch35::C0_SIZE_B8;
using WeightQuantBatchMatmulV2::Arch35::AntiQuantMxA8W4NzNkVf;
using WeightQuantBatchMatmulV2::Arch35::CeilAlign;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::DataCopyPad2D;
using WeightQuantBatchMatmulV2::Arch35::MxA8W4NzParams;
using WeightQuantBatchMatmulV2::Arch35::MX_BIAS_SINGLE_VECTOR_SIZE;
using WeightQuantBatchMatmulV2::Arch35::VEC_MAX_ELEM_B16;

namespace Prologue {

static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;

struct PrologueMxCastWOffsetParam {
    uint64_t kSize;
    uint64_t kbL1Size;
    uint64_t nL1Size;
    uint64_t nOffset;
    uint64_t nAlign;
};

#define WQBMM_PROLOGUE_TEMPLATE_PARAM                                                                                   \
    template <class AType_, class BType_, class BiasType_>

#define WQBMM_PROLOGUE_CLASS                                                                                            \
    BlockPrologue<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, AType_, BType_, BiasType_>

struct UbConsumeConfig {
    uint64_t l1RequireVfComputeRealK;
    uint64_t l1RequireVfComputeRealN;
    uint64_t kWeightLowBitUbOffset;
    uint64_t nWeightLowBitUbOffset;
    uint64_t ubMxBiasNsize;
    bool calcMxBias = false;
    bool isBiasSingleVector = false;
};

struct L1ConsumeConfig {
    uint64_t l1SplitTwoVecExternalOffset;
    uint64_t l1RealExternalLen;
    uint64_t l1MxBiasSplitNOffset;
};

struct UbBufferInfo {
    uint64_t ubWeightOutputHighBitBufferNum;
    uint64_t weightInputLowbitUbTotalSize;
    uint64_t highBitDataUbTotalSize;
    uint64_t biasUbTotalSize;
    uint64_t biasReducedUbTotalSize;
    uint64_t weightInputLowBitUbSingleBufferSize;
    uint64_t biasUbSingleBufferSize;
    uint64_t biasReducedSingleBufferSize;
};

__aicore__ constexpr UbBufferInfo GetMxA8W4NzBufferInfo(uint64_t ubMte2BufferNum)
{
    return {.ubWeightOutputHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 64 * GetKBUnit<int8_t>(),
            .highBitDataUbTotalSize = 128 * GetKBUnit<int8_t>(),
            .biasUbTotalSize = 2 * GetKBUnit<half>(),
            .biasReducedUbTotalSize = 2 * GetKBUnit<half>(),
            .weightInputLowBitUbSingleBufferSize = 64 * GetKBUnit<int8_t>() / ubMte2BufferNum,
            .biasUbSingleBufferSize = 2 * GetKBUnit<half>() / ubMte2BufferNum,
            .biasReducedSingleBufferSize = 2 * GetKBUnit<half>() / ubMte2BufferNum};
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
class WQBMM_PROLOGUE_CLASS {
public:
    using DispatchPolicy = GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit;
    using AType = AType_;
    using BType = BType_;
    using BiasType = BiasType_;

    static constexpr bool kATrans = false;
    static constexpr bool kBTrans = true;
    static constexpr CubeFormat kWeightFormat = CubeFormat::NZ;
    static constexpr uint64_t kUbMte2BufferNum = GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit::ubMte2BufferNum;

    static_assert(!kATrans, "KernelMixDynamicKL1NTailResplit requires non-transposed input A");
    static_assert(kBTrans, "KernelMixDynamicKL1NTailResplit requires transposed weight B");
    static_assert(kWeightFormat == CubeFormat::NZ, "KernelMixDynamicKL1NTailResplit requires NZ weight format");

    struct Params {
        __gm__ BType *ptrB;
        __gm__ BiasType *ptrBias;
        uint8_t hasBias;
    };

    __aicore__ inline BlockPrologue() = delete;
    __aicore__ inline BlockPrologue(const Params& params);
    __aicore__ inline void operator()(__gm__ BType *weight, __gm__ BiasType *bias, const bool weightL2Cacheable,
                                      uint64_t mL1Size, uint64_t kSize, uint64_t nL1Size, uint64_t nOffset,
                                      uint64_t nAlign);
    __aicore__ inline ~BlockPrologue();

protected:
    __aicore__ inline uint64_t CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size) const;
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAicToAiv();
    __aicore__ inline void ComputeBasicBlockAivNdKnNzNk(const PrologueMxCastWOffsetParam &offsetParam);
    __aicore__ inline void mxBiasSetParamAndGmtoUb(const PrologueMxCastWOffsetParam &offsetParam,
                                                   L1ConsumeConfig &l1ConsumeConfig, UbConsumeConfig &ubConsumeConfig,
                                                   const uint64_t kMte2Offset, const uint64_t mte2RealK);
    __aicore__ inline void InitVectorCompute();
    __aicore__ inline void SetVectorGlobalBuffer(__gm__ BType *weight, __gm__ BiasType *bias,
                                                 const bool weightL2Cacheable);
    __aicore__ inline void WaitVectorToMTE2();
    __aicore__ inline void SetVectorToMTE2();
    __aicore__ inline void CopyWeightGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                            uint64_t ubMte2KOffset, const PrologueMxCastWOffsetParam &offsetParam);
    __aicore__ inline void CopyMxBiasGmToUb(uint64_t ubMte2MxBiasNSize, uint64_t ubMte2MxBiasNOffset);
    __aicore__ inline void WeightAntiQuantComputeNzNk(const UbConsumeConfig &ubConsumeConfig,
                                                      const LocalTensor<AType> &weightHighBitL1,
                                                      const L1ConsumeConfig &l1ConsumeConfig,
                                                      const LocalTensor<BiasType> &biasL1);
    __aicore__ inline uint64_t ComputeWeightHighBitL1Offset(uint64_t antiQuantNOffset, uint64_t antiQuantKOffset,
                                                            uint64_t nRealLen, uint64_t kRealLen,
                                                            const L1ConsumeConfig &l1ConsumeConfig);
    __aicore__ inline void AntiQuantProcessNzMxA8W4(const UbConsumeConfig &ubConsumeConfig);
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN,
                                                       uint64_t antiQuantRealK,
                                                       const LocalTensor<AType> &weightHighBitL1);
    __aicore__ inline void FinalizeVectorCompute();

    uint64_t cvLoopIdx_ = 0;

    LocalTensor<AType> weightL1_;
    LocalTensor<BiasType> biasL1_;
    uint64_t weightL1DbOffset_;
    uint64_t biasL1DbOffset_;
    bool hasBias_;

    uint64_t ubMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;
    GlobalTensor<BType> wGlobal_;
    GlobalTensor<BiasType> biasGlobal_;
    LocalTensor<int8_t> ubWeightInputLowBitTotalBuffer_;
    LocalTensor<AType> ubHighBitTotalBuffer_;
    LocalTensor<BiasType> ubBiasTotalBuffer_;
    LocalTensor<BiasType> ubBiasOutTotalBuffer_;

    constexpr static TEventID vecEventIdVToMte2_[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};
    constexpr static TEventID vecEventIdMte3ToV_[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};
    constexpr static uint32_t C0_SIZE = C0_SIZE_B8;
    constexpr static uint64_t VEC_REG_ELEM = VECTOR_REG_WIDTH;
    constexpr static UbBufferInfo UB_BUFFER_INFO = GetMxA8W4NzBufferInfo(kUbMte2BufferNum);
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;
    uint64_t mxA8W4L1KDynamicConfigMThreshold_;
};

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline WQBMM_PROLOGUE_CLASS::BlockPrologue(const Params& params)
{
    hasBias_ = params.hasBias;
    mxA8W4L1KDynamicConfigMThreshold_ = hasBias_ ? MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 :
                                                   MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<AType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(AType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;
    static constexpr uint64_t MX_BIAS_L1_SIZE = BIAS_L1_SIZE * GetKBUnit<BiasType>() * sizeof(BiasType);
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<AType>() - MXA8W4_WEIGHT_SIZE;
    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;
    biasL1_ = LocalTensor<BiasType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(BiasType));
    if (hasBias_) {
        biasL1DbOffset_ = (l1RemainSize - MX_BIAS_L1_SIZE) / sizeof(BiasType);
    }

    if ASCEND_IS_AIV {
        InitVectorCompute();
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline uint64_t WQBMM_PROLOGUE_CLASS::CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size) const
{
    return (mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ &&
            nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
               MX_A8W4_L1_K_CONFIG_512 :
               MX_A8W4_L1_K_CONFIG_256;
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::mxBiasSetParamAndGmtoUb(
    const PrologueMxCastWOffsetParam &offsetParam, L1ConsumeConfig &l1ConsumeConfig, UbConsumeConfig &ubConsumeConfig,
    const uint64_t kMte2Offset, const uint64_t mte2RealK)
{
    uint64_t ubMte2MxBiasNSize = 0;
    uint64_t ubMte2MxBiasNOffset = 0;
    if (hasBias_ && kMte2Offset == 0) {
        ubConsumeConfig.isBiasSingleVector = (mte2RealK == l1ConsumeConfig.l1RealExternalLen);
        if (ubConsumeConfig.isBiasSingleVector) {
            ubMte2MxBiasNSize = offsetParam.nL1Size;
        } else {
            uint64_t mxBiasVec0Nsize =
                offsetParam.nL1Size < MX_BIAS_SINGLE_VECTOR_SIZE ? offsetParam.nL1Size : MX_BIAS_SINGLE_VECTOR_SIZE;
            ubMte2MxBiasNSize = (GetSubBlockIdx() == 0) ? mxBiasVec0Nsize : (offsetParam.nL1Size - mxBiasVec0Nsize);
        }
        ubMte2MxBiasNOffset =
            offsetParam.nOffset + GetSubBlockIdx() * ((ubMte2MxBiasNSize != 0) ? MX_BIAS_SINGLE_VECTOR_SIZE : 0);
    }
    ubConsumeConfig.calcMxBias = (hasBias_ && (kMte2Offset == 0) && (ubMte2MxBiasNSize != 0));
    if (ubConsumeConfig.calcMxBias) {
        ubConsumeConfig.ubMxBiasNsize = ubMte2MxBiasNSize;
        l1ConsumeConfig.l1MxBiasSplitNOffset = GetSubBlockIdx() * MX_BIAS_SINGLE_VECTOR_SIZE;
        CopyMxBiasGmToUb(ubMte2MxBiasNSize, ubMte2MxBiasNOffset);
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::ComputeBasicBlockAivNdKnNzNk(
    const PrologueMxCastWOffsetParam &offsetParam)
{
    uint64_t kMte2BaseSize = offsetParam.kbL1Size >> 1;

    UbConsumeConfig ubConsumeConfig;
    L1ConsumeConfig l1ConsumeConfig;
    ubConsumeConfig.l1RequireVfComputeRealN = offsetParam.nL1Size;
    ubConsumeConfig.kWeightLowBitUbOffset = 0;
    ubConsumeConfig.nWeightLowBitUbOffset = 0;
    l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * kMte2BaseSize;

    for (uint64_t kMte2Offset = 0; kMte2Offset < offsetParam.kSize; kMte2Offset += offsetParam.kbL1Size, cvLoopIdx_++) {
        l1ConsumeConfig.l1RealExternalLen = (kMte2Offset + offsetParam.kbL1Size) > offsetParam.kSize ?
                                                offsetParam.kSize - kMte2Offset :
                                                offsetParam.kbL1Size;
        uint64_t mte2RealK = GetSubBlockIdx() == 0 ? min(kMte2BaseSize, l1ConsumeConfig.l1RealExternalLen) :
                             l1ConsumeConfig.l1RealExternalLen > kMte2BaseSize ?
                                 l1ConsumeConfig.l1RealExternalLen - kMte2BaseSize :
                                 0;

        WaitVectorToMTE2();
        mxBiasSetParamAndGmtoUb(offsetParam, l1ConsumeConfig, ubConsumeConfig, kMte2Offset, mte2RealK);
        CopyWeightGmToUb(offsetParam.nL1Size, mte2RealK, offsetParam.nOffset,
                         kMte2Offset + GetSubBlockIdx() * kMte2BaseSize, offsetParam);

        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }
        ubConsumeConfig.l1RequireVfComputeRealK = mte2RealK;
        WeightAntiQuantComputeNzNk(ubConsumeConfig, weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_],
                                   l1ConsumeConfig, biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_]);

        SetAivToAic();
        SetVectorToMTE2();
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::operator()(
    __gm__ BType *weight, __gm__ BiasType *bias, const bool weightL2Cacheable, uint64_t mL1Size, uint64_t kSize,
    uint64_t nL1Size, uint64_t nOffset, uint64_t nAlign)
{
    SetVectorGlobalBuffer(weight, bias, weightL2Cacheable);
    PrologueMxCastWOffsetParam offsetParam = {};
    offsetParam.kSize = kSize;
    offsetParam.nL1Size = nL1Size;
    offsetParam.nOffset = nOffset;
    offsetParam.kbL1Size = CalcDynamicKBlock(mL1Size, nL1Size);
    offsetParam.nAlign = nAlign;
    ComputeBasicBlockAivNdKnNzNk(offsetParam);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline WQBMM_PROLOGUE_CLASS::~BlockPrologue()
{
    if (cvLoopIdx_ > 0) {
        WaitAicToAiv();
    }
    if (cvLoopIdx_ > 1) {
        WaitAicToAiv();
    }
    FinalizeVectorCompute();
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::SetAivToAic()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::WaitAicToAiv()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::InitVectorCompute()
{
    ubWeightInputLowBitTotalBuffer_ =
        LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightInputLowbitUbTotalSize);
    ubHighBitTotalBuffer_ =
        LocalTensor<AType>(TPosition::LCM, 64 * GetKBUnit<int8_t>(), UB_BUFFER_INFO.highBitDataUbTotalSize);
    if (hasBias_) {
        ubBiasTotalBuffer_ =
            LocalTensor<BiasType>(TPosition::LCM, 192 * GetKBUnit<int8_t>(), UB_BUFFER_INFO.biasUbTotalSize);
        ubBiasOutTotalBuffer_ =
            LocalTensor<BiasType>(TPosition::LCM, 194 * GetKBUnit<int8_t>(), UB_BUFFER_INFO.biasReducedUbTotalSize);
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::SetVectorGlobalBuffer(
    __gm__ BType *weight, __gm__ BiasType *bias, const bool weightL2Cacheable)
{
    wGlobal_.SetGlobalBuffer(weight);
    if (!weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    if (hasBias_) {
        biasGlobal_.SetGlobalBuffer(bias);
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::WaitVectorToMTE2()
{
    if (likely(ubMte2LoopIdx_ > kUbMte2BufferNum - 1)) {
        if constexpr (kUbMte2BufferNum == 2 || kUbMte2BufferNum == 4) {
            WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[ubMte2LoopIdx_ & (kUbMte2BufferNum - 1)]);
        } else {
            WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[ubMte2LoopIdx_ % kUbMte2BufferNum]);
        }
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::SetVectorToMTE2()
{
    if constexpr (kUbMte2BufferNum == 2 || kUbMte2BufferNum == 4) {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[(ubMte2LoopIdx_ - 1) & (kUbMte2BufferNum - 1)]);
    } else {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[(ubMte2LoopIdx_ - 1) % kUbMte2BufferNum]);
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::CopyWeightGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
    const PrologueMxCastWOffsetParam &offsetParam)
{
    if (ubMte2NSize == 0 || ubMte2KSize == 0) {
        ubMte2LoopIdx_++;
        return;
    }

    DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % kUbMte2BufferNum) *
                                                  UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                      .template ReinterpretCast<BType>(),
                  wGlobal_[ubMte2KOffset * offsetParam.nAlign + ubMte2NOffset * static_cast<uint64_t>(C0_SIZE)],
                  CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
                  CeilAlign(ubMte2NSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                  CeilAlign(ubMte2NSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                  offsetParam.nAlign * C0_SIZE);

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    ubMte2LoopIdx_++;
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::CopyMxBiasGmToUb(
    uint64_t ubMte2MxBiasNSize, uint64_t ubMte2MxBiasNOffset)
{
    if (hasBias_) {
        DataCopyPad2D(ubBiasTotalBuffer_[(ubMte2LoopIdx_ % QUADRUPLE_BUFFER_NUM) * UB_BUFFER_INFO.biasUbSingleBufferSize],
                      biasGlobal_[ubMte2MxBiasNOffset], 1, ubMte2MxBiasNSize, ubMte2MxBiasNSize, ubMte2MxBiasNSize);
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::WeightAntiQuantComputeNzNk(
    const UbConsumeConfig &ubConsumeConfig, const LocalTensor<AType> &weightHighBitL1,
    const L1ConsumeConfig &l1ConsumeConfig, const LocalTensor<BiasType> &biasL1)
{
    if (likely(ubComputeLoopIdx_ > UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) {
        WaitFlag<HardEvent::MTE3_V>(
            vecEventIdMte3ToV_[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
    }

    AntiQuantProcessNzMxA8W4(ubConsumeConfig);

    uint64_t weightHighBitL1Offset = ComputeWeightHighBitL1Offset(
        0, 0, ubConsumeConfig.l1RequireVfComputeRealN, ubConsumeConfig.l1RequireVfComputeRealK, l1ConsumeConfig);

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    if (likely(ubConsumeConfig.l1RequireVfComputeRealK > 0)) {
        CopyWeightHighBitForAligned(weightHighBitL1Offset, ubConsumeConfig.l1RequireVfComputeRealN,
                                    ubConsumeConfig.l1RequireVfComputeRealK, weightHighBitL1);
    }
    if (ubConsumeConfig.calcMxBias) {
        DataCopy(biasL1[l1ConsumeConfig.l1MxBiasSplitNOffset],
                 ubBiasOutTotalBuffer_[((ubMte2LoopIdx_ - 1) & (kUbMte2BufferNum - 1)) *
                                       UB_BUFFER_INFO.biasReducedSingleBufferSize],
                 ubConsumeConfig.ubMxBiasNsize);
    }
    SetFlag<HardEvent::MTE3_V>(
        vecEventIdMte3ToV_[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
    ubComputeLoopIdx_++;
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline uint64_t WQBMM_PROLOGUE_CLASS::ComputeWeightHighBitL1Offset(
    uint64_t antiQuantNOffset, uint64_t antiQuantKOffset, uint64_t nRealLen, uint64_t kRealLen,
    const L1ConsumeConfig &l1ConsumeConfig)
{
    uint64_t nRealLenAlign = CeilAlign(nRealLen, static_cast<uint64_t>(BLOCK_CUBE));
    return nRealLenAlign * (antiQuantKOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) +
            antiQuantNOffset * static_cast<uint64_t>(C0_SIZE);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::AntiQuantProcessNzMxA8W4(
    const UbConsumeConfig &ubConsumeConfig)
{
    MxA8W4NzParams<AType, BType, BiasType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) & (kUbMte2BufferNum - 1);
    mxA8W4NzParams.nRealSizeAlign =
        CeilAlign(ubConsumeConfig.l1RequireVfComputeRealN, static_cast<uint64_t>(BLOCK_CUBE));
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ BType *)
            ubWeightInputLowBitTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                .GetPhyAddr();
    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ AType *)
            ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                  VECTOR_REG_WIDTH]
                .GetPhyAddr();
    mxA8W4NzParams.loopKNum = CeilDivide(ubConsumeConfig.l1RequireVfComputeRealK, static_cast<uint64_t>(C0_SIZE));
    mxA8W4NzParams.innerLoopNum = CeilDivide(
        CeilAlign(ubConsumeConfig.l1RequireVfComputeRealN, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
        static_cast<uint64_t>(VECTOR_REG_WIDTH));
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;
    if (ubConsumeConfig.calcMxBias) {
        mxA8W4NzParams.biasInUbAddr =
            (__ubuf__ BiasType *)ubBiasTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.biasUbSingleBufferSize]
                .GetPhyAddr();
        mxA8W4NzParams.biasOutUbAddr =
            (__ubuf__ BiasType *)ubBiasOutTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.biasReducedSingleBufferSize]
                .GetPhyAddr();
        if (ubConsumeConfig.isBiasSingleVector) {
            mxA8W4NzParams.biasLoopNum = CeilDivide(ubConsumeConfig.ubMxBiasNsize, VEC_MAX_ELEM_B16);
            AntiQuantMxA8W4NzNkVf<AType, BType, BiasType, true, true>(mxA8W4NzParams);
        } else {
            AntiQuantMxA8W4NzNkVf<AType, BType, BiasType, true, false>(mxA8W4NzParams);
        }
    } else {
        AntiQuantMxA8W4NzNkVf<AType, BType, BiasType, false, false>(mxA8W4NzParams);
    }
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::CopyWeightHighBitForAligned(
    uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<AType> &weightHighBitL1)
{
    DataCopyParams params;
    params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(C0_SIZE)) *
                        CeilAlign(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE)) / VEC_REG_ELEM;
    params.blockLen = VEC_REG_ELEM / ONE_BLK_SIZE;
    params.srcStride = (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1) * params.blockLen;
    params.dstStride = 0;
    DataCopy(
        weightHighBitL1[weightHighBitL1Offset],
        ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) * VEC_REG_ELEM],
        params);
}

WQBMM_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_PROLOGUE_CLASS::FinalizeVectorCompute()
{
    for (uint16_t idx = 0; idx < ubComputeLoopIdx_ && idx < UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_[idx]);
    }

    for (uint16_t idx = 0; idx < ubMte2LoopIdx_ && idx < kUbMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_[idx]);
    }
}

#undef WQBMM_PROLOGUE_CLASS
#undef WQBMM_PROLOGUE_TEMPLATE_PARAM
} // namespace Block

#endif  // GROUPED_MATMUL_MXFP8FP4_PROLOGUE_MX_CAST_W_H
