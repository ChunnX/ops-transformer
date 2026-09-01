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
 * \file shift_w4_to_w8.h
 * \brief Convert packed W4 weights to W8 layout for vector-to-cube consumption.
 */
#pragma once

#include <type_traits>

#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze::Gemm::Tile {

template <typename OutType_, typename InType_>
struct ShiftW4ToW8Params {
    uint64_t loopKNum;
    uint64_t innerLoopNum;
    uint64_t loopKDstStride;
    uint64_t innerDstStride;
    uint64_t nRealSizeAlign;
    __ubuf__ InType_* weight4BitPhyAddr;
    __ubuf__ OutType_* weight8BitPhyAddr;
};

template <typename OutType_, typename InType_>
struct ShiftW4DnToW8Params {
    uint16_t outExtend;
    uint16_t innerExtend;
    uint32_t dataBlockStride;
    uint32_t repeatStride;
    int32_t outDimOffset;
    uint32_t maskB8Tail0;
    uint32_t maskB8Tail1;
    uint32_t inputRowStrideBytes;
    __ubuf__ int8_t* weight4BitPhyAddr;
    __ubuf__ OutType_* weight8BitPhyAddr;
    __ubuf__ OutType_* weight8BitPhyAddr1;
};

template <typename OutType, typename InType>
__simd_vf__ inline void ShiftW4ToW8Vf(ShiftW4ToW8Params<OutType, InType> params);

template <typename OutType, typename InType>
__simd_vf__ inline void ShiftW4DnToW8Vf(ShiftW4DnToW8Params<OutType, InType> params);

template <typename OutType, typename InType, typename Weight4BitTensorType, typename Weight8BitTensorType>
__aicore__ inline void ShiftW4ToW8ZnImpl(
    const Weight4BitTensorType& weight4BitTensor, const Weight8BitTensorType& weight8BitTensor)
{
    using DstLayoutPattern = AscendC::Te::GetLayoutPattern<typename Weight8BitTensorType::layoutType>;
    static_assert(
        AscendC::Std::is_same_v<DstLayoutPattern, Weight8BitZnToZnUbLayoutPtn>,
        "ZN FP4 conversion requires the MX FP8 ZN UB layout");
    static_assert(
        std::is_same_v<OutType, __fp8e4m3> || AscendC::IsSameType<OutType, fp8_e4m3fn_t>::value,
        "OutType must be fp8_e4m3fn_t");
    static_assert(
        std::is_same_v<InType, __fp4e2m1x2> || AscendC::IsSameType<InType, fp4x2_e2m1_t>::value,
        "InType must be fp4x2_e2m1_t");

    ShiftW4ToW8Params<OutType, InType> params;
    params.weight4BitPhyAddr = (__ubuf__ InType*)weight4BitTensor.Data().Get();
    params.weight8BitPhyAddr = (__ubuf__ OutType*)weight8BitTensor.Data().Get();

    // Derive kernel-loop and stride parameters directly from tensor shape/stride metadata.
    params.loopKNum = AscendC::Std::get<1>(AscendC::Std::get<0>(weight4BitTensor.Layout().Shape()));
    params.nRealSizeAlign = AscendC::Std::get<1>(AscendC::Std::get<1>(weight4BitTensor.Layout().Shape())) * BLOCK_CUBE;
    params.innerDstStride = AscendC::Std::get<1>(AscendC::Std::get<1>(weight8BitTensor.Layout().Stride()));
    params.innerLoopNum = (params.nRealSizeAlign * C0_SIZE_B8) / static_cast<uint64_t>(asc_get_vf_len());
    params.loopKDstStride = params.innerLoopNum * params.innerDstStride;

    asc_vf_call<ShiftW4ToW8Vf<OutType, InType>>(params);
}

template <typename OutType, typename InType, typename Weight4BitTensorType, typename Weight8BitTensorType>
__aicore__ inline void ShiftW4ToW8DnImpl(
    const Weight4BitTensorType& weight4BitTensor, const Weight8BitTensorType& weight8BitTensor)
{
    using DstLayoutPattern = AscendC::Te::GetLayoutPattern<typename Weight8BitTensorType::layoutType>;
    static_assert(
        AscendC::Std::is_same_v<DstLayoutPattern, Blaze::Gemm::Weight8BitDnToZnUbLayoutPtn>,
        "DN FP4 conversion requires the MX DN-to-ZN destination layout");
    static_assert(
        std::is_same_v<OutType, __fp8e4m3> || AscendC::IsSameType<OutType, fp8_e4m3fn_t>::value,
        "OutType must be fp8_e4m3fn_t");
    static_assert(
        std::is_same_v<InType, __fp4e2m1x2> || AscendC::IsSameType<InType, fp4x2_e2m1_t>::value,
        "InType must be fp4x2_e2m1_t");

    const auto& inputLayout = weight4BitTensor.Layout();
    auto inputShape = inputLayout.Shape();
    auto inputStride = inputLayout.Stride();
    uint64_t kSize = static_cast<uint64_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(inputShape)));
    uint64_t nSize = static_cast<uint64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(inputShape)));
    uint64_t nAlign = Align16(nSize);
    uint64_t kAlign32 = Align32(kSize);
    uint64_t kAlign64 = Align64(kSize);
    constexpr uint64_t VECTOR_REG_WIDTH_BYTES = static_cast<uint64_t>(asc_get_vf_len());
    constexpr uint16_t FP4_PACK_SHIFT = 1U;
    constexpr uint64_t VECTOR_REG_WIDTH_FOR_4BITS = VECTOR_REG_WIDTH_BYTES << FP4_PACK_SHIFT;

    ShiftW4DnToW8Params<OutType, InType> params;
    params.outExtend = static_cast<uint16_t>(nSize);
    params.innerExtend = static_cast<uint16_t>(CeilDiv(kAlign64, VECTOR_REG_WIDTH_FOR_4BITS));
    params.dataBlockStride = static_cast<uint32_t>(nAlign + 1U);
    params.repeatStride = params.dataBlockStride * static_cast<uint32_t>(BLOCK_CUBE);
    uint64_t outputRowSpan = static_cast<uint64_t>(params.innerExtend) * params.repeatStride * BLOCK_BYTE_SIZE;
    params.outDimOffset =
        static_cast<int32_t>(static_cast<int64_t>(BLOCK_BYTE_SIZE) - static_cast<int64_t>(outputRowSpan));
    params.maskB8Tail0 = static_cast<uint32_t>(
        Min(kAlign32 % VECTOR_REG_WIDTH_FOR_4BITS, VECTOR_REG_WIDTH_BYTES) +
        kAlign32 / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH_BYTES);
    params.maskB8Tail1 = static_cast<uint32_t>(
        Max(static_cast<int64_t>(kAlign32 % VECTOR_REG_WIDTH_FOR_4BITS) - static_cast<int64_t>(VECTOR_REG_WIDTH_BYTES),
            static_cast<int64_t>(0)) +
        kAlign32 / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH_BYTES);
    params.inputRowStrideBytes =
        static_cast<uint32_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(inputStride)) >> 1U);
    params.weight4BitPhyAddr = (__ubuf__ int8_t*)weight4BitTensor.Data().Get();
    params.weight8BitPhyAddr = (__ubuf__ OutType*)weight8BitTensor.Data().Get();
    params.weight8BitPhyAddr1 = params.weight8BitPhyAddr + VECTOR_REG_WIDTH_BYTES * params.dataBlockStride;

    asc_vf_call<ShiftW4DnToW8Vf<OutType, InType>>(params);
}

template <typename OutType, typename InType, typename Weight4BitTensorType, typename Weight8BitTensorType>
__aicore__ inline void ShiftW4ToW8(
    const Weight4BitTensorType& weight4BitTensor, const Weight8BitTensorType& weight8BitTensor)
{
    using SrcLayoutPattern = AscendC::Te::GetLayoutPattern<typename Weight4BitTensorType::layoutType>;
    using DstLayoutPattern = AscendC::Te::GetLayoutPattern<typename Weight8BitTensorType::layoutType>;
    constexpr bool IS_ZN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::ZNLayoutPtn>;
    constexpr bool IS_DN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::DNExtLayoutPtn>;
    constexpr bool IS_ZN_OUTPUT = AscendC::Std::is_same_v<DstLayoutPattern, Weight8BitZnToZnUbLayoutPtn>;
    constexpr bool IS_DN_OUTPUT =
        AscendC::Std::is_same_v<DstLayoutPattern, Blaze::Gemm::Weight8BitDnToZnUbLayoutPtn>;
    static_assert(IS_ZN_WEIGHT || IS_DN_WEIGHT, "W4-to-W8 conversion only supports ZN and DNExt source layouts");
    static_assert(
        (IS_ZN_WEIGHT && IS_ZN_OUTPUT) || (IS_DN_WEIGHT && IS_DN_OUTPUT),
        "W4-to-W8 source and destination layouts must use the matching conversion path");

    if constexpr (IS_DN_WEIGHT) {
        ShiftW4ToW8DnImpl<OutType, InType>(weight4BitTensor, weight8BitTensor);
    } else {
        ShiftW4ToW8ZnImpl<OutType, InType>(weight4BitTensor, weight8BitTensor);
    }
}

template <typename OutType, typename InType>
__simd_vf__ inline void ShiftW4ToW8Vf(ShiftW4ToW8Params<OutType, InType> params)
{
    constexpr uint32_t E2M1_SHIFT_RIGHT_SIZE = 0x2;
    constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;
    constexpr uint32_t E2M1_AND_MASK = 0x9C;
    AscendC::Reg::RegTensor<int8_t> wShrReg;
    AscendC::Reg::RegTensor<int8_t> wShlReg;
    AscendC::Reg::RegTensor<int8_t> wAndReg;
    AscendC::Reg::RegTensor<int8_t> wLoad;
    AscendC::Reg::RegTensor<int8_t> wShl;
    AscendC::Reg::RegTensor<int8_t> wShr0;
    AscendC::Reg::RegTensor<int8_t> wShr1;
    AscendC::Reg::RegTensor<int8_t> wSel;
    AscendC::Reg::RegTensor<int8_t> wAnd;

    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg pregVsel = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < params.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < params.innerLoopNum; ++innerLoopIdx) {
            // DIST_US_B8 load mode expands each packed B4 byte into lane-aligned B8 slots.
            // Packed B4 address offset (bytes) = logical element index >> 1.
            AscendC::Reg::AddrReg aregWeightB8In = AscendC::Reg::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * params.nRealSizeAlign) >> 1, innerLoopIdx, asc_get_vf_len() >> 1);
            AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_US_B8>(
                (AscendC::Reg::RegTensor<uint8_t>&)wLoad, (__ubuf__ uint8_t*&)params.weight4BitPhyAddr, aregWeightB8In);

            AscendC::Reg::ShiftRight(wShr0, wLoad, wShrReg, preg);
            AscendC::Reg::ShiftLeft(wShl, wLoad, wShlReg, preg);
            AscendC::Reg::ShiftRight(wShr1, wShl, wShrReg, preg);
            AscendC::Reg::Select(wSel, wShr1, wShr0, pregVsel);
            AscendC::Reg::And(wAnd, wSel, wAndReg, preg);

            AscendC::Reg::AddrReg aregWeightB8Out = AscendC::Reg::CreateAddrReg<uint8_t>(
                loopKIdx, params.loopKDstStride, innerLoopIdx, params.innerDstStride);
            AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t*&)params.weight8BitPhyAddr, (AscendC::Reg::RegTensor<uint8_t>&)wAnd, aregWeightB8Out,
                preg);
        }
    }
}

template <typename OutType, typename InType>
__simd_vf__ inline void ShiftW4DnToW8Vf(ShiftW4DnToW8Params<OutType, InType> params)
{
    namespace MicroAPI = AscendC::MicroAPI;
    constexpr uint32_t E2M1_EM_SHIFT_RIGHT_SIZE = 0x2;
    constexpr uint32_t E2M1_SIGN_SHIFT_LEFT_SIZE = 0x4;
    constexpr uint32_t FP4_SIGN_MASK = 0x80;
    constexpr uint32_t E2M1_EM_MASK = 0x1C;
    constexpr uint64_t VECTOR_REG_WIDTH_BYTES = static_cast<uint64_t>(asc_get_vf_len());
    __ubuf__ OutType* weightOutUbAddr = params.weight8BitPhyAddr;
    __ubuf__ OutType* weightOutUbAddr1 = params.weight8BitPhyAddr1;
    MicroAPI::RegTensor<uint8_t> wDIntlv0, wDIntlv1, wLoad0, sign0, sign1, emShr, emShl, signShl, out0, out1;
    MicroAPI::RegTensor<int8_t> emShrBits, emShlBits, signBits;
    MicroAPI::RegTensor<uint8_t> emMask, signMask;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(emShrBits, E2M1_EM_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(emShlBits, E2M1_EM_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(signBits, E2M1_SIGN_SHIFT_LEFT_SIZE, preg);
    MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(emMask, E2M1_EM_MASK, preg);
    MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(signMask, FP4_SIGN_MASK, preg);

    for (uint16_t outIdx = 0; outIdx < params.outExtend; ++outIdx) {
        uint32_t maskWeight0Tmp = params.maskB8Tail0;
        uint32_t maskWeight1Tmp = params.maskB8Tail1;
        for (uint16_t repeatIdx = 0; repeatIdx < params.innerExtend; ++repeatIdx) {
            MicroAPI::MaskReg maskB8Tail0 = MicroAPI::UpdateMask<uint8_t>(maskWeight0Tmp);
            MicroAPI::MaskReg maskB8Tail1 = MicroAPI::UpdateMask<uint8_t>(maskWeight1Tmp);
            MicroAPI::AddrReg aregWeightB8 =
                MicroAPI::CreateAddrReg<uint8_t>(outIdx, params.inputRowStrideBytes, repeatIdx, VECTOR_REG_WIDTH_BYTES);
            MicroAPI::LoadAlign(wLoad0, (__ubuf__ uint8_t*&)params.weight4BitPhyAddr, aregWeightB8);

            MicroAPI::ShiftRight(emShr, wLoad0, emShrBits, preg);
            MicroAPI::And(emShr, emShr, emMask, preg);
            MicroAPI::ShiftLeft(emShl, wLoad0, emShlBits, preg);
            MicroAPI::And(emShl, emShl, emMask, preg);
            MicroAPI::ShiftLeft(signShl, wLoad0, signBits, preg);
            MicroAPI::And(sign0, signShl, signMask, preg);
            MicroAPI::And(sign1, wLoad0, signMask, preg);
            MicroAPI::Or(out0, emShr, sign1, preg);
            MicroAPI::Or(out1, emShl, sign0, preg);
            MicroAPI::Interleave(wDIntlv0, wDIntlv1, out1, out0);
            MicroAPI::StoreAlign<
                uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ uint8_t*&)weightOutUbAddr, wDIntlv0, params.dataBlockStride, params.repeatStride,
                maskB8Tail0);
            MicroAPI::StoreAlign<
                uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ uint8_t*&)weightOutUbAddr1, wDIntlv1, params.dataBlockStride, params.repeatStride,
                maskB8Tail1);
        }
        weightOutUbAddr += params.outDimOffset;
        weightOutUbAddr1 += params.outDimOffset;
    }
}

} // namespace Blaze::Gemm::Tile
