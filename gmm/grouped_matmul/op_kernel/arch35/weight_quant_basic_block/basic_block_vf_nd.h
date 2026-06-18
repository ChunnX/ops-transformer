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
 * \file basic_block_vf_nd.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_ND_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_ND_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "weight_quant_tool.h"

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType>
struct LocalAddressParam {
    __local_mem__ xType *antiQuantScaleBasePhyAddr;
    __local_mem__ xType *antiQuantScaleBasePhyAddr1;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1;
    __local_mem__ wType *weightLowBitPhyAddr0;
    __local_mem__ wType *weightLowBitPhyAddr1;
    __local_mem__ xType *weightF16PhyAddr0;
    __local_mem__ xType *weightF16PhyAddr1;
};

template <typename xType>
struct CalculateParam {
    xType offsetValue;
    xType scaleValue;
    uint64_t ubLoop;
};

static constexpr MicroAPI::CastTrait FP8_TO_FP32_TRAIT_0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                            MicroAPI::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait FP8_TO_FP32_TRAIT_2 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::UNKNOWN,
                                                            MicroAPI::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::UNKNOWN};

static constexpr MicroAPI::CastTrait FP32_TO_F16_ODD = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                        MicroAPI::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_ROUND};
static constexpr MicroAPI::CastTrait FP32_TO_F16_EVEN = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                         MicroAPI::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_ROUND};

static constexpr MicroAPI::CastTrait S8_TO_FP16_TRAIT_ODD = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait FP16_TO_BF16_TRAIT = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};

template <typename xType, typename wType, bool hasAntiQuantOffset, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdNkVfLoadScaleOffset(RegTensor<xType> &antiQuantScaleVreg,
                                                         RegTensor<xType> &antiQuantOffsetVreg,
                                                         LocalAddressParam<xType, wType> &localAddressParam,
                                                         const CalculateParam<xType> &calculateParam,
                                                         uint16_t nIdx)
{
    if constexpr (hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
            antiQuantOffsetVreg, localAddressParam.antiQuantOffsetBasePhyAddr + nIdx);
    }
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg, calculateParam.scaleValue);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg,
                                                                    localAddressParam.antiQuantScaleBasePhyAddr + nIdx);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize>
__aicore__ inline void AntiQuantFP8NdNkVfLoadWeight(RegTensor<wType> &weightF8Vreg0, RegTensor<wType> &weightF8Vreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam, uint16_t nIdx)
{
    // UNPK_B8 表示按照如下形式载入:
    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
    // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + nIdx * ubMte2InnerSize);
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + nIdx * ubMte2InnerSize);
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdNkVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg;
    RegTensor<wType> weightF8Vreg0, weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
    RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < calculateParam.ubLoop; ubLoopNIdx++) {
        AntiQuantFP8NdNkVfLoadScaleOffset<xType, wType, hasAntiQuantOffset, antiQuantType>(
            antiQuantScaleVreg, antiQuantOffsetVreg, localAddressParam, calculateParam, ubLoopNIdx);
        AntiQuantFP8NdNkVfLoadWeight<xType, wType, hasAntiQuantOffset, ubMte2InnerSize>(weightF8Vreg0, weightF8Vreg1,
                                                                                        localAddressParam, ubLoopNIdx);
        // 奇数、偶数位置分散到2个fp32寄存器存储
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);

        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);

        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg3, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg1, maskAll);

        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                MicroAPI::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                MicroAPI::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg, maskAll);
            }
        }

        MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
        MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg, maskAll);

        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset>
__aicore__ inline void AntiQuantFP8NdKnVfLoadOffset(RegTensor<xType> &antiQuantOffsetVreg0,
                                                    RegTensor<xType> &antiQuantOffsetVreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam)
{
    if constexpr (hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg0,
                                                                 localAddressParam.antiQuantOffsetBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg1,
                                                                 localAddressParam.antiQuantOffsetBasePhyAddr1);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdKnVfLoadScale(RegTensor<xType> &antiQuantScaleVreg0,
                                                   RegTensor<xType> &antiQuantScaleVreg1,
                                                   LocalAddressParam<xType, wType> &localAddressParam,
                                                   const CalculateParam<xType> &calculateParam)
{
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg0, calculateParam.scaleValue);
        MicroAPI::Duplicate(antiQuantScaleVreg1, calculateParam.scaleValue);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg0,
                                                                 localAddressParam.antiQuantScaleBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg1,
                                                                 localAddressParam.antiQuantScaleBasePhyAddr1);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize>
__aicore__ inline void AntiQuantFP8NdKnVfLoadWeight(RegTensor<wType> &weightF8Vreg0, RegTensor<wType> &weightF8Vreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam, uint16_t kIdx)
{
    // UNPK_B8 表示按照如下形式载入:
    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
    // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + kIdx * ubMte2InnerSize);
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + kIdx * ubMte2InnerSize);
}

template <typename xType, typename wType>
__aicore__ inline void AntiQuantFP8NdKnVfStoreWeight(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg2,
                                                     LocalAddressParam<xType, wType> &localAddressParam, MaskReg &mask)
{
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, mask);
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, mask);
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdKnVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg0, antiQuantScaleVreg1, antiQuantOffsetVreg0, antiQuantOffsetVreg1;
    RegTensor<wType> weightF8Vreg0, weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
    RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AntiQuantFP8NdKnVfLoadOffset<xType, wType, hasAntiQuantOffset>(antiQuantOffsetVreg0, antiQuantOffsetVreg1,
                                                                   localAddressParam);
    AntiQuantFP8NdKnVfLoadScale<xType, wType, hasAntiQuantOffset, antiQuantType>(
        antiQuantScaleVreg0, antiQuantScaleVreg1, localAddressParam, calculateParam);
    for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < calculateParam.ubLoop; ubLoopKIdx++) {
        AntiQuantFP8NdKnVfLoadWeight<xType, wType, hasAntiQuantOffset, ubMte2InnerSize>(weightF8Vreg0, weightF8Vreg1,
                                                                                        localAddressParam, ubLoopKIdx);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg1, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg3, maskAll);
        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                MicroAPI::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg0, maskAll);
                MicroAPI::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg1, maskAll);
            }
        }
        MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg0, maskAll);
        MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg1, maskAll);
        AntiQuantFP8NdKnVfStoreWeight<xType, wType>(weightF16Vreg0, weightF16Vreg2, localAddressParam, maskAll);
    }
}

template <typename xType, typename wType>
__aicore__ inline void CastS8RegTensorToF16RegTensor(RegTensor<xType> &weightF16Vreg, RegTensor<wType> &weightS8Vreg,
                                                     RegTensor<half> &weightFp16Vreg, MicroAPI::MaskReg &maskAll)
{
    // PART_EVEN 表示按照如下形式处理做cast：
    // Vn 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    // Vd 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 .....
    if constexpr (!IsSameType<typename MicroAPI::TypeGet<xType>::T, vector_f16>::value) {
        MicroAPI::Cast<half, wType, S8_TO_FP16_TRAIT_ODD>(weightFp16Vreg, weightS8Vreg, maskAll);
        MicroAPI::Cast<xType, half, FP16_TO_BF16_TRAIT>(weightF16Vreg, weightFp16Vreg, maskAll);
    } else {
        MicroAPI::Cast<xType, wType, S8_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightS8Vreg, maskAll);
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig>
__aicore__ inline void AddMulWeightF16RegTensorNdKn(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                    RegTensor<xType> &antiQuantScaleVreg,
                                                    RegTensor<xType> &antiQuantOffsetVreg,
                                                    RegTensor<xType> &antiQuantScaleVreg1,
                                                    RegTensor<xType> &antiQuantOffsetVreg1, MicroAPI::MaskReg &maskAll,
                                                    const xType &offsetValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
            MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
            MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
        } else {
            MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
            MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg1, maskAll);
        }
    }

    MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
    MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
}

template <typename xType, typename wType>
__aicore__ inline void WeightF16NdRegTensorToNzUb(__local_mem__ xType *&weightF16PhyAddr0,
                                                  __local_mem__ xType *&weightF16PhyAddr1,
                                                  RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                  MicroAPI::MaskReg &maskAll)
{
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig>
__aicore__ inline void AddMulWeightF16RegTensorNdNk(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                    RegTensor<xType> &antiQuantScaleVreg,
                                                    RegTensor<xType> &antiQuantOffsetVreg, MicroAPI::MaskReg &maskAll,
                                                    const xType &scaleValue, const xType &offsetValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
            MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
            MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
        } else {
            MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
            MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
        }
    }

    MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
    MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
}

template <typename xType, const WqmmConfig &wqmmConfig>
__aicore__ inline void NdNkLoadScaleOffset(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                           RegTensor<xType> &antiQuantScaleVreg,
                                           RegTensor<xType> &antiQuantOffsetVreg,
                                           xType scaleValue, uint64_t ubLoopNIdx)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantOffsetVreg,
                                                                    antiQuantOffsetBasePhyAddr + ubLoopNIdx);
    }
    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg, scaleValue);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg,
                                                                    antiQuantScaleBasePhyAddr + ubLoopNIdx);
    }
}

template <typename xType, const WqmmConfig &wqmmConfig>
__aicore__ inline void NdKnLoadScaleOffset(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                           __local_mem__ xType *antiQuantScaleBasePhyAddr1,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr1,
                                           RegTensor<xType> &antiQuantScaleVreg,
                                           RegTensor<xType> &antiQuantScaleVreg1,
                                           RegTensor<xType> &antiQuantOffsetVreg,
                                           RegTensor<xType> &antiQuantOffsetVreg1,
                                           xType scaleValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr1);
    }
    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg, scaleValue);
        MicroAPI::Duplicate(antiQuantScaleVreg1, scaleValue);
    } else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
        // 从 UB [G_tile, N_tile_align] 布局加载当前 group 的 scale
        // DIST_NORM: 连续加载 128 个 FP16 元素
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantB8CommonNdKn(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                             __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                             __local_mem__ wType *weightLowBitPhyAddr0,
                                             __local_mem__ xType *weightF16PhyAddr0, xType scaleValue,
                                             xType offsetValue, uint64_t ubLoopK)
{
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<xType> antiQuantScaleVreg1;
        RegTensor<xType> antiQuantOffsetVreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

        NdKnLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantScaleBasePhyAddr1,
                                               antiQuantOffsetBasePhyAddr, antiQuantOffsetBasePhyAddr1,
                                               antiQuantScaleVreg, antiQuantScaleVreg1,
                                               antiQuantOffsetVreg, antiQuantOffsetVreg1,
                                               scaleValue);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < static_cast<uint16_t>(ubLoopK); ubLoopKIdx++) {
            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopKIdx * vecConfig.ubMte2InnerSize);
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopKIdx * vecConfig.ubMte2InnerSize);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg0, weightS8Vreg0, weightFp16Vreg0, maskAll);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg1, weightS8Vreg1, weightFp16Vreg1, maskAll);
            AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(
                weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1,
                antiQuantOffsetVreg1, maskAll, offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantB8CommonNdNk(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                             __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                             __local_mem__ wType *weightLowBitPhyAddr0,
                                             __local_mem__ xType *weightF16PhyAddr0, xType scaleValue,
                                             xType offsetValue, uint64_t ubLoopN)
{
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < static_cast<uint16_t>(ubLoopN); ubLoopNIdx++) {
            NdNkLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
                                                   antiQuantScaleVreg, antiQuantOffsetVreg, scaleValue, ubLoopNIdx);
            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg0, weightS8Vreg0, weightFp16Vreg0, maskAll);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg1, weightS8Vreg1, weightFp16Vreg1, maskAll);
            AddMulWeightF16RegTensorNdNk<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, maskAll, scaleValue,
                                                                   offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdNk(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                         __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                         __local_mem__ wType *weightLowBitPhyAddr0,
                                         __local_mem__ xType *weightF16PhyAddr0, xType scaleValue, xType offsetValue,
                                         uint64_t ubLoopN)
{
    // int4每次处理128个数即为64B, 256>>2=64
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < static_cast<uint16_t>(ubLoopN); ubLoopNIdx++) {
            NdNkLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
                                                   antiQuantScaleVreg, antiQuantOffsetVreg, scaleValue, ubLoopNIdx);
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
            AddMulWeightF16RegTensorNdNk<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, maskAll, scaleValue,
                                                                   offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdKn(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                         __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                         __local_mem__ wType *weightLowBitPhyAddr0,
                                         __local_mem__ xType *weightF16PhyAddr0, xType scaleValue, xType offsetValue,
                                         uint64_t ubLoopK)
{
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1, antiQuantOffsetVreg1;
        RegTensor<int4x2_t> weightS4Vreg0, weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0, weightF16Vreg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
        NdKnLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantScaleBasePhyAddr1,
                                               antiQuantOffsetBasePhyAddr, antiQuantOffsetBasePhyAddr1,
                                               antiQuantScaleVreg, antiQuantScaleVreg1,
                                               antiQuantOffsetVreg, antiQuantOffsetVreg1,
                                               scaleValue);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < static_cast<uint16_t>(ubLoopK); ubLoopKIdx++) {
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
            AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(
                weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1,
                antiQuantOffsetVreg1, maskAll, offsetValue);

            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4PerGroupNdKn(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                                 __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                                 __local_mem__ wType *weightLowBitPhyAddr0,
                                                 __local_mem__ xType *weightF16PhyAddr0, uint16_t ubLoopK,
                                                 uint16_t groupSize, uint16_t nBubXTypeAlign)
{
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    // int4每次处理128个数即为64B, 256>>2=64
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1, antiQuantOffsetVreg1;
        RegTensor<int4x2_t> weightS4Vreg0, weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0, weightF16Vreg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                  MicroAPI::MaskMergeMode::ZEROING,
                                                                  AscendC::RoundMode::UNKNOWN};

        uint16_t groupNum = ubLoopK / groupSize;
        uint16_t groupTail = ubLoopK % groupSize;

        for (uint16_t gIdx = 0; gIdx < groupNum; gIdx++) {
            NdKnLoadScaleOffset<xType, wqmmConfig>(
                antiQuantScaleBasePhyAddr + gIdx * nBubXTypeAlign,
                antiQuantScaleBasePhyAddr1 + gIdx * nBubXTypeAlign,
                antiQuantOffsetBasePhyAddr + gIdx * nBubXTypeAlign,
                antiQuantOffsetBasePhyAddr1 + gIdx * nBubXTypeAlign, antiQuantScaleVreg, antiQuantScaleVreg1,
                antiQuantOffsetVreg, antiQuantOffsetVreg1, xType(0));

            for (uint16_t kIdx = 0; kIdx < groupSize; kIdx++) {
                uint16_t globalKIdx = gIdx * groupSize + kIdx;
                // DIST_UNPACK4_B8: Vn中一个数字4bit(0.5Byte)
                // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                // Vd 0 1 x x x x x x 2 3 x x x x x x
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg0,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 +
                                               globalKIdx * (vecConfig.ubMte2InnerSize >> 1)));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 +
                                               globalKIdx * (vecConfig.ubMte2InnerSize >> 1)));

                // PART_P0: Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
                //          Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
                AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(
                    weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1,
                    antiQuantOffsetVreg1, maskAll, xType(0));

                WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                         weightF16Vreg1, maskAll);
            }
        }

        if (groupTail > 0) {
            NdKnLoadScaleOffset<xType, wqmmConfig>(
                antiQuantScaleBasePhyAddr + groupNum * nBubXTypeAlign,
                antiQuantScaleBasePhyAddr1 + groupNum * nBubXTypeAlign,
                antiQuantOffsetBasePhyAddr + groupNum * nBubXTypeAlign,
                antiQuantOffsetBasePhyAddr1 + groupNum * nBubXTypeAlign, antiQuantScaleVreg, antiQuantScaleVreg1,
                antiQuantOffsetVreg, antiQuantOffsetVreg1, xType(0));

            for (uint16_t kIdx = 0; kIdx < groupTail; kIdx++) {
                uint16_t globalKIdx = groupNum * groupSize + kIdx;
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg0,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 +
                                               globalKIdx * (vecConfig.ubMte2InnerSize >> 1)));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 +
                                               globalKIdx * (vecConfig.ubMte2InnerSize >> 1)));

                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
                AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(
                    weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1,
                    antiQuantOffsetVreg1, maskAll, xType(0));

                WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                         weightF16Vreg1, maskAll);
            }
        }
    }
}

// ============================================================================
// NK PerGroup INT4 反量化 — gs=128 专用 VF 函数
// 每个 group 恰好 128 个 K 元素 = 1 个向量寄存器宽度
// Scale UB 布局: [G_tile, VEC_MAX_ELEM_B16]，stride = VEC_MAX_ELEM_B16
// Scale 地址: base + groupIdx * VEC_MAX_ELEM_B16 + nBubIdx
// ============================================================================
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4PerGroupNdNkGs128(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                                       __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                                       __local_mem__ wType *weightLowBitPhyAddr0,
                                                       __local_mem__ xType *weightF16PhyAddr0, uint16_t bubNLen,
                                                       uint16_t bubKLen)
{
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                                   AscendC::RoundMode::UNKNOWN};

        constexpr uint16_t GROUP_SIZE = 128;
        uint16_t groupNum = bubKLen / GROUP_SIZE;
        // N 方向 weight 步长（字节）: CeilAlign(K, 32) >> 1
        uint16_t outerStrideWeight = (bubKLen + 31) / 32 * 32 >> 1;
        // K 方向 weight 步长（字节）: 128 INT4 = 64 字节
        constexpr uint16_t innerStrideWeight = 64;
        // 输出 NZ block 间距
        uint16_t nAlign = (bubNLen + 15) / 16 * 16;
        uint16_t dataBlockStride = nAlign + 1;
        // group 间输出地址偏移
        uint16_t weightOutAddrOffset = VEC_MAX_ELEM_B16 * dataBlockStride - bubNLen * static_cast<uint16_t>(BLOCK_CUBE);

        uint32_t maskLen = bubKLen;
        MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<xType>(maskLen);

        for (uint16_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
            for (uint16_t nBubIdx = 0; nBubIdx < bubNLen; ++nBubIdx) {
                // GMM [G, N] 布局 scale 地址: base + groupIdx * VEC_MAX_ELEM_B16 + nBubIdx
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    antiQuantScaleVreg, antiQuantScaleBasePhyAddr + groupIdx * VEC_MAX_ELEM_B16 + nBubIdx);
                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr + groupIdx * VEC_MAX_ELEM_B16 + nBubIdx);
                }

                // DIST_UNPACK4_B8: 加载 128 个 INT4 元素
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg0,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + groupIdx * innerStrideWeight +
                                               nBubIdx * outerStrideWeight));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + groupIdx * innerStrideWeight +
                                               nBubIdx * outerStrideWeight));

                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);

                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
                }
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);

                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr0, weightF16Vreg0,
                                                                            dataBlockStride, 1, maskWeight);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr1, weightF16Vreg1,
                                                                            dataBlockStride, 1, maskWeight);
            }
            weightF16PhyAddr0 += weightOutAddrOffset;
            weightF16PhyAddr1 += weightOutAddrOffset;
            maskLen = bubKLen;
            maskWeight = MicroAPI::UpdateMask<xType>(maskLen);
        }
    }
}

// ============================================================================
// NK PerGroup INT4 反量化 — gs=256 专用 VF 函数
// 每个 group 256 K 元素 = 2 个向量寄存器，需 2 次 weight 加载、2 次输出
// ============================================================================
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4PerGroupNdNkGs256(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                                       __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                                       __local_mem__ wType *weightLowBitPhyAddr0,
                                                       __local_mem__ xType *weightF16PhyAddr0, uint16_t bubNLen,
                                                       uint16_t bubKLen)
{
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<int4x2_t> weightS4Vreg0_1;
        RegTensor<int4x2_t> weightS4Vreg1_1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;
        RegTensor<xType> weightF16Vreg0_1;
        RegTensor<xType> weightF16Vreg1_1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                                   AscendC::RoundMode::UNKNOWN};

        constexpr uint16_t GROUP_SIZE = 256;
        uint16_t groupNum = bubKLen / GROUP_SIZE;
        uint16_t outerStrideWeight = (bubKLen + 31) / 32 * 32 >> 1;
        // K 方向 weight 步长（字节）: 256 INT4 = 128 字节
        constexpr uint16_t innerStrideWeight = 128;
        uint16_t nAlign = (bubNLen + 15) / 16 * 16;
        uint16_t dataBlockStride = nAlign + 1;
        // 2 个向量寄存器的输出偏移
        __local_mem__ xType *weightF16PhyAddr0_1 = weightF16PhyAddr0 + dataBlockStride * VEC_MAX_ELEM_B16;
        __local_mem__ xType *weightF16PhyAddr1_1 = weightF16PhyAddr1 + dataBlockStride * VEC_MAX_ELEM_B16;
        // group 间输出地址偏移
        uint16_t weightOutAddrOffset =
            2 * VEC_MAX_ELEM_B16 * dataBlockStride - bubNLen * static_cast<uint16_t>(BLOCK_CUBE);

        uint32_t maskLen = bubKLen;
        MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<xType>(maskLen);
        MicroAPI::MaskReg maskWeight1 = MicroAPI::UpdateMask<xType>(maskLen);

        for (uint16_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
            for (uint16_t nBubIdx = 0; nBubIdx < bubNLen; ++nBubIdx) {
                // GMM [G, N] 布局 scale 地址
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    antiQuantScaleVreg, antiQuantScaleBasePhyAddr + groupIdx * VEC_MAX_ELEM_B16 + nBubIdx);
                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr + groupIdx * VEC_MAX_ELEM_B16 + nBubIdx);
                }

                // 第 1 个 128 INT4 元素
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg0,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + groupIdx * innerStrideWeight +
                                               nBubIdx * outerStrideWeight));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + groupIdx * innerStrideWeight +
                                               nBubIdx * outerStrideWeight));
                // 第 2 个 128 INT4 元素（偏移 64 字节）
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg0_1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + 64 + groupIdx * innerStrideWeight +
                                               nBubIdx * outerStrideWeight));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg1_1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + 64 + groupIdx * innerStrideWeight +
                                               nBubIdx * outerStrideWeight));

                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0_1, weightS4Vreg0_1, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1_1, weightS4Vreg1_1, maskAll);

                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg0_1, weightF16Vreg0_1, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1_1, weightF16Vreg1_1, antiQuantOffsetVreg, maskAll);
                }
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg0_1, weightF16Vreg0_1, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1_1, weightF16Vreg1_1, antiQuantScaleVreg, maskAll);

                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr0, weightF16Vreg0,
                                                                            dataBlockStride, 1, maskWeight);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr1, weightF16Vreg1,
                                                                            dataBlockStride, 1, maskWeight);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr0_1, weightF16Vreg0_1,
                                                                            dataBlockStride, 1, maskWeight1);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr1_1, weightF16Vreg1_1,
                                                                            dataBlockStride, 1, maskWeight1);
            }
            weightF16PhyAddr0 += weightOutAddrOffset;
            weightF16PhyAddr1 += weightOutAddrOffset;
            weightF16PhyAddr0_1 += weightOutAddrOffset;
            weightF16PhyAddr1_1 += weightOutAddrOffset;
            maskLen = bubKLen;
            maskWeight = MicroAPI::UpdateMask<xType>(maskLen);
            maskWeight1 = MicroAPI::UpdateMask<xType>(maskLen);
        }
    }
}

// ============================================================================
// NK PerGroup INT4 反量化 — gs=64 专用 VF 函数
// 每个 group 64 K 元素 = 半个向量寄存器，需 Select 拼接相邻 2 个 group 的 scale
// ============================================================================
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4PerGroupNdNkGs64(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                                      __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                                      __local_mem__ wType *weightLowBitPhyAddr0,
                                                      __local_mem__ xType *weightF16PhyAddr0, uint16_t bubNLen,
                                                      uint16_t bubKLen)
{
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> oriScale00;
        RegTensor<xType> oriScale01;
        RegTensor<xType> scale0;
        RegTensor<xType> oriOffset00;
        RegTensor<xType> oriOffset01;
        RegTensor<xType> offset0;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegSelect = MicroAPI::CreateMask<xType, MicroAPI::MaskPattern::H>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                                   AscendC::RoundMode::UNKNOWN};

        // K 方向按 128 元素分块（2 个 group × 64 K）
        uint16_t innerExtend = (bubKLen + 127) / 128;
        uint16_t outerExtend = bubNLen;
        uint16_t outerStrideWeight = (bubKLen + 31) / 32 * 32 >> 1;
        uint16_t nAlign = (bubNLen + 15) / 16 * 16;
        uint16_t dataBlockStride = nAlign + 1;
        // 每个 innerExtend 迭代后输出地址偏移
        uint16_t outerStride =
            VEC_MAX_ELEM_B16 * dataBlockStride - bubNLen * static_cast<uint16_t>(BLOCK_CUBE);

        uint32_t maskLen = bubKLen;
        MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<xType>(maskLen);

        for (uint16_t innerIdx = 0; innerIdx < innerExtend; ++innerIdx) {
            for (uint16_t outerIdx = 0; outerIdx < outerExtend; ++outerIdx) {
                // 从 [G, N] 加载 2 个相邻 group 的 scale，用 Select 拼接
                // group[2*innerIdx] 和 group[2*innerIdx+1] 各 64 个 K 元素
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    oriScale00, antiQuantScaleBasePhyAddr + (innerIdx * 2) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    oriScale01, antiQuantScaleBasePhyAddr + (innerIdx * 2 + 1) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::Select(scale0, oriScale00, oriScale01, maskRegSelect);

                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        oriOffset00, antiQuantOffsetBasePhyAddr + (innerIdx * 2) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        oriOffset01, antiQuantOffsetBasePhyAddr + (innerIdx * 2 + 1) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::Select(offset0, oriOffset00, oriOffset01, maskRegSelect);
                }

                // 加载 128 个 INT4 weight（覆盖 2 个 group × 64 K）
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg0,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + innerIdx * 64 +
                                               outerIdx * outerStrideWeight));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + innerIdx * 64 +
                                               outerIdx * outerStrideWeight));

                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);

                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, offset0, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, offset0, maskAll);
                }
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, scale0, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, scale0, maskAll);

                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr0, weightF16Vreg0,
                                                                            dataBlockStride, 1, maskWeight);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr1, weightF16Vreg1,
                                                                            dataBlockStride, 1, maskWeight);
            }
            weightF16PhyAddr0 += outerStride;
            weightF16PhyAddr1 += outerStride;
            maskLen = bubKLen;
            maskWeight = MicroAPI::UpdateMask<xType>(maskLen);
        }
    }
}

// ============================================================================
// NK PerGroup INT4 反量化 — gs=32 专用 VF 函数
// 每个 group 32 K 元素 = 1/4 向量寄存器
// GMM [G,N] 布局下不能用 DIST_E2B_B16，使用 8 次 DIST_BRC_B16 + 4 次 Interleave + 2 次 Select
// ============================================================================
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4PerGroupNdNkGs32(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                                      __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                                      __local_mem__ wType *weightLowBitPhyAddr0,
                                                      __local_mem__ xType *weightF16PhyAddr0, uint16_t bubNLen,
                                                      uint16_t bubKLen)
{
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        // Scale 寄存器
        RegTensor<xType> s0, s1, s2, s3, s4, s5, s6, s7;
        RegTensor<xType> scaleAB, scaleCD, scaleEF, scaleGH;
        RegTensor<xType> scale0, scale1;
        // Offset 寄存器
        RegTensor<xType> o0, o1, o2, o3, o4, o5, o6, o7;
        RegTensor<xType> offsetAB, offsetCD, offsetEF, offsetGH;
        RegTensor<xType> offset0, offset1;
        // Weight 寄存器
        RegTensor<int4x2_t> weightS4Vl0_0, weightS4Vl0_1;
        RegTensor<int4x2_t> weightS4Vl1_0, weightS4Vl1_1;
        RegTensor<xType> weightF16Vl0_0, weightF16Vl0_1;
        RegTensor<xType> weightF16Vl1_0, weightF16Vl1_1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegSelect = MicroAPI::CreateMask<xType, MicroAPI::MaskPattern::H>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                                   AscendC::RoundMode::UNKNOWN};

        // K 方向按 256 元素分块（8 groups × 32 K）
        uint16_t innerExtend = (bubKLen + 255) / 256;
        uint16_t outerExtend = bubNLen;
        uint16_t outerStrideWeight = (bubKLen + 31) / 32 * 32 >> 1;
        uint16_t nAlign = (bubNLen + 15) / 16 * 16;
        uint16_t dataBlockStride = nAlign + 1;
        // 每个 innerExtend 迭代后输出地址偏移（256 K 元素 = 2 个向量寄存器）
        uint16_t outDimOffset =
            2 * VEC_MAX_ELEM_B16 * dataBlockStride - bubNLen * static_cast<uint16_t>(BLOCK_CUBE);

        uint32_t maskLen = bubKLen;
        MicroAPI::MaskReg maskWeight0 = MicroAPI::UpdateMask<xType>(maskLen);
        MicroAPI::MaskReg maskWeight1 = MicroAPI::UpdateMask<xType>(maskLen);

        for (uint16_t innerIdx = 0; innerIdx < innerExtend; ++innerIdx) {
            for (uint16_t outerIdx = 0; outerIdx < outerExtend; ++outerIdx) {
                // 8 次广播加载 8 个 group 的 scale
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s0, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 0) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s1, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 1) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s2, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 2) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s3, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 3) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s4, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 4) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s5, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 5) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s6, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 6) * VEC_MAX_ELEM_B16 + outerIdx);
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    s7, antiQuantScaleBasePhyAddr + (innerIdx * 8 + 7) * VEC_MAX_ELEM_B16 + outerIdx);

                // Interleave 构造 64 元素对（使用独立临时寄存器避免覆盖）
                RegTensor<xType> abLow, abHigh, cdLow, cdHigh, efLow, efHigh, ghLow, ghHigh;
                MicroAPI::Interleave(abLow, abHigh, s0, s1);  // [s0×64, s1×64]
                MicroAPI::Interleave(cdLow, cdHigh, s2, s3);  // [s2×64, s3×64]
                MicroAPI::Interleave(efLow, efHigh, s4, s5);  // [s4×64, s5×64]
                MicroAPI::Interleave(ghLow, ghHigh, s6, s7);  // [s6×64, s7×64]

                // Select 拼接 128 元素寄存器
                MicroAPI::Select(scale0, abLow, cdLow, maskRegSelect);  // [s0×32, s1×32, s2×32, s3×32]
                MicroAPI::Select(scale1, efLow, ghLow, maskRegSelect);  // [s4×32, s5×32, s6×32, s7×32]

                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o0, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 0) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o1, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 1) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o2, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 2) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o3, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 3) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o4, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 4) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o5, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 5) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o6, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 6) * VEC_MAX_ELEM_B16 + outerIdx);
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        o7, antiQuantOffsetBasePhyAddr + (innerIdx * 8 + 7) * VEC_MAX_ELEM_B16 + outerIdx);

                    RegTensor<xType> oAbLow, oAbHigh, oCdLow, oCdHigh, oEfLow, oEfHigh, oGhLow, oGhHigh;
                    MicroAPI::Interleave(oAbLow, oAbHigh, o0, o1);
                    MicroAPI::Interleave(oCdLow, oCdHigh, o2, o3);
                    MicroAPI::Interleave(oEfLow, oEfHigh, o4, o5);
                    MicroAPI::Interleave(oGhLow, oGhHigh, o6, o7);

                    MicroAPI::Select(offset0, oAbLow, oCdLow, maskRegSelect);
                    MicroAPI::Select(offset1, oEfLow, oGhLow, maskRegSelect);
                }

                // 2 次 weight 加载（各 128 INT4）
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vl0_0,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + innerIdx * 128 +
                                               outerIdx * outerStrideWeight));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vl1_0,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + innerIdx * 128 +
                                               outerIdx * outerStrideWeight));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vl0_1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + 64 + innerIdx * 128 +
                                               outerIdx * outerStrideWeight));
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vl1_1,
                    (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + 64 + innerIdx * 128 +
                                               outerIdx * outerStrideWeight));

                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vl0_0, weightS4Vl0_0, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vl1_0, weightS4Vl1_0, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vl0_1, weightS4Vl0_1, maskAll);
                MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vl1_1, weightS4Vl1_1, maskAll);

                if constexpr (wqmmConfig.hasAntiQuantOffset) {
                    MicroAPI::Add(weightF16Vl0_0, weightF16Vl0_0, offset0, maskAll);
                    MicroAPI::Add(weightF16Vl1_0, weightF16Vl1_0, offset0, maskAll);
                    MicroAPI::Add(weightF16Vl0_1, weightF16Vl0_1, offset1, maskAll);
                    MicroAPI::Add(weightF16Vl1_1, weightF16Vl1_1, offset1, maskAll);
                }
                MicroAPI::Mul(weightF16Vl0_0, weightF16Vl0_0, scale0, maskAll);
                MicroAPI::Mul(weightF16Vl1_0, weightF16Vl1_0, scale0, maskAll);
                MicroAPI::Mul(weightF16Vl0_1, weightF16Vl0_1, scale1, maskAll);
                MicroAPI::Mul(weightF16Vl1_1, weightF16Vl1_1, scale1, maskAll);

                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr0, weightF16Vl0_0,
                                                                            dataBlockStride, 1, maskWeight0);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr1, weightF16Vl1_0,
                                                                            dataBlockStride, 1, maskWeight0);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr0, weightF16Vl0_1,
                                                                            dataBlockStride, 1, maskWeight1);
                MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(weightF16PhyAddr1, weightF16Vl1_1,
                                                                            dataBlockStride, 1, maskWeight1);
            }
            weightF16PhyAddr0 += outDimOffset;
            weightF16PhyAddr1 += outDimOffset;
            maskLen = bubKLen;
            maskWeight0 = MicroAPI::UpdateMask<xType>(maskLen);
            maskWeight1 = MicroAPI::UpdateMask<xType>(maskLen);
        }
    }
}

// ============================================================================
// NK PerGroup INT4 反量化 — 入口函数，按 groupSize 分派
// ============================================================================
template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4PerGroupNdNk(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                                  __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                                  __local_mem__ wType *weightLowBitPhyAddr0,
                                                  __local_mem__ xType *weightF16PhyAddr0, uint16_t bubNLen,
                                                  uint16_t groupSize, uint16_t bubKLen)
{
    if (groupSize == 32) {
        AntiQuantInt4PerGroupNdNkGs32<xType, wType, wqmmConfig, vecConfig>(
            antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
            weightLowBitPhyAddr0, weightF16PhyAddr0, bubNLen, bubKLen);
    } else if (groupSize == 64) {
        AntiQuantInt4PerGroupNdNkGs64<xType, wType, wqmmConfig, vecConfig>(
            antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
            weightLowBitPhyAddr0, weightF16PhyAddr0, bubNLen, bubKLen);
    } else if (groupSize == 128) {
        AntiQuantInt4PerGroupNdNkGs128<xType, wType, wqmmConfig, vecConfig>(
            antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
            weightLowBitPhyAddr0, weightF16PhyAddr0, bubNLen, bubKLen);
    } else if (groupSize == 256) {
        AntiQuantInt4PerGroupNdNkGs256<xType, wType, wqmmConfig, vecConfig>(
            antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
            weightLowBitPhyAddr0, weightF16PhyAddr0, bubNLen, bubKLen);
    }
}

}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_ND_H