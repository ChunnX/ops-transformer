#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"
#include "math_approximations.h"
#include "ring_buffer_layout.h"
#include "catlass_fractal_packer.h"
#include "credit_handshake.h"

#if defined(__CCE__)
#include "kernel_operator.h"

namespace sage::microkernel {

__aicore__ inline void InitMxP(memory::FixedResource &resource) {
    AscendC::InitConstValueParams<int16_t> initParams;
    initParams.repeatTimes = 1;
    initParams.dstGap = 0;
    initParams.initValue = static_cast<int16_t>(sage::core::PV_PARTIAL_E8M0_PAIR);
    initParams.blockNum = static_cast<uint16_t>(sage::core::MX_P_BLOCK_NUM);

    AscendC::InitConstValue(resource.l1Buf.template GetBufferByByte<int16_t>(sage::core::MX_P_L1), initParams);
    memory::Sync<AscendC::HardEvent::MTE2_MTE1>();

    const auto l0aByteAddr = resource.l0ABuf.template GetBufferByByte<float8_e4m3_t>(sage::core::L0A_MX_P_OFFSET).GetPhyAddr();
    const auto l1ScaleAddr = resource.l1Buf.template GetBufferByByte<float8_e8m0_t>(sage::core::MX_P_L1).GetPhyAddr();
    const uint64_t targetUnitOffset = reinterpret_cast<uint64_t>(l0aByteAddr) / 16ULL;

    load_cbuf_to_ca_mx(targetUnitOffset, (__cbuf__ void *)l1ScaleAddr, 0, 0,
                       sage::core::MX_P_M_BLOCKS, sage::core::MX_P_N_BLOCKS,
                       sage::core::MX_P_N_BLOCKS, sage::core::MX_P_N_BLOCKS);
    memory::Sync<AscendC::HardEvent::MTE1_M>();
}

template <class T, bool WIDE>
__simd_vf__ inline void QuantVRowsMxVF(__ubuf__ T *x, __ubuf__ T *maxima, __ubuf__ uint16_t *scales,
                                       __ubuf__ float8_e4m3_t *out, uint32_t dp)
{
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    auto fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    auto channelMask = AscendC::MicroAPI::UpdateMask<T>(dp);
    AscendC::MicroAPI::RegTensor<uint16_t> signMask;
    AscendC::MicroAPI::Duplicate(signMask, sage::core::FP16_ABS_MASK);

    for (uint16_t gIdx = 0; gIdx < 2; ++gIdx) {
        AscendC::MicroAPI::RegTensor<T> peakAccum;
        AscendC::MicroAPI::Duplicate(peakAccum, static_cast<T>(0));
        __ubuf__ T *groupInBase = x + (static_cast<uint32_t>(gIdx) * 32U) * dp;
        for (uint16_t r = 0; r < 32; r += 2) {
            AscendC::MicroAPI::RegTensor<T> row0, row1;
            AscendC::MicroAPI::LoadAlign(row0, groupInBase + static_cast<uint32_t>(r) * dp);
            AscendC::MicroAPI::LoadAlign(row1, groupInBase + static_cast<uint32_t>(r + 1) * dp);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t> &)row0,
                                   (AscendC::MicroAPI::RegTensor<uint16_t> &)row0, signMask, channelMask);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t> &)row1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t> &)row1, signMask, channelMask);
            AscendC::MicroAPI::Max(peakAccum, peakAccum, row0, channelMask);
            AscendC::MicroAPI::Max(peakAccum, peakAccum, row1, channelMask);
        }
        AscendC::MicroAPI::StoreAlign(maxima + static_cast<uint32_t>(gIdx) * dp, peakAccum, channelMask);
    }

    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    static constexpr AscendC::MicroAPI::CastTrait unpackTrait{
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait packTrait{
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    AscendC::MicroAPI::RegTensor<uint32_t> expMaskReg, invBiasReg;
    AscendC::MicroAPI::Duplicate(expMaskReg, sage::core::EXPONENT_MASK);
    AscendC::MicroAPI::Duplicate(invBiasReg, sage::core::RECIPROCAL_EXP_BIAS);

    for (uint16_t col = 0; col < dp; col += 64) {
        AscendC::MicroAPI::RegTensor<uint32_t> combinedScales;
        AscendC::MicroAPI::Duplicate(combinedScales, 0U);
        for (uint16_t gIdx = 0; gIdx < 2; ++gIdx) {
            AscendC::MicroAPI::RegTensor<float> colPeakF32, invScaleF32;
            AscendC::MicroAPI::RegTensor<T> rawPeakT;
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                rawPeakT, maxima + static_cast<uint32_t>(gIdx) * dp + col);
            AscendC::MicroAPI::Cast<float, T, unpackTrait>(colPeakF32, rawPeakT, fullMask);

            AscendC::MicroAPI::Muls(colPeakF32, colPeakF32, sage::core::INV_MX_FP8_MAX, fullMask);
            AscendC::MicroAPI::Maxs(colPeakF32, colPeakF32, sage::core::MIN_POSITIVE_FLOAT, fullMask);

            AscendC::MicroAPI::RegTensor<uint32_t> biasedBits, exponentField;
            AscendC::MicroAPI::Adds(biasedBits, (AscendC::MicroAPI::RegTensor<uint32_t> &)colPeakF32,
                                    sage::core::EXPONENT_ROUND_BIAS, fullMask);
            AscendC::MicroAPI::And(biasedBits, biasedBits, expMaskReg, fullMask);
            AscendC::MicroAPI::ShiftRights(exponentField, biasedBits, static_cast<int16_t>(23), fullMask);

            AscendC::MicroAPI::RegTensor<uint32_t> invBitPattern;
            AscendC::MicroAPI::Sub(invBitPattern, invBiasReg, biasedBits, fullMask);
            invScaleF32 = (AscendC::MicroAPI::RegTensor<float> &)invBitPattern;

            // [DG.6 UE8M0 Hardware Mutual Recognition]:
            // Ascend 950 hardware layout packs a pair of UE8M0 scales (one per 32 elements)
            // into a single uint16_t (64 K-elements per SF pair, MN-major order).
            // Group 0 occupies [0..7], Group 1 occupies [8..15] via ShiftLefts 8 + Or.
            // This is bit-exact with DeepGEMM-Ascend MX_SF_DIVISOR format.
            if (gIdx == 1) {
                AscendC::MicroAPI::ShiftLefts(exponentField, exponentField, static_cast<int16_t>(8), fullMask);
            }
            AscendC::MicroAPI::Or(combinedScales, combinedScales, exponentField, fullMask);

            __ubuf__ T *srcGroupPtr = x + (static_cast<uint32_t>(gIdx) * 32U) * dp + col;
            __ubuf__ float8_e4m3_t *dstGroupPtr = out + (static_cast<uint32_t>(gIdx) * 32U) * dp + col;

            for (uint16_t r = 0; r < 16; ++r) {
                AscendC::MicroAPI::RegTensor<T> inLow, inHigh;
                AscendC::MicroAPI::RegTensor<float> fLow, fHigh;
                AscendC::MicroAPI::RegTensor<float8_e4m3_t> fp8Low, fp8High;

                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    inLow, srcGroupPtr + static_cast<uint32_t>(r) * dp);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    inHigh, srcGroupPtr + static_cast<uint32_t>(r + 16U) * dp);

                AscendC::MicroAPI::Cast<float, T, unpackTrait>(fLow, inLow, fullMask);
                AscendC::MicroAPI::Cast<float, T, unpackTrait>(fHigh, inHigh, fullMask);
                AscendC::MicroAPI::Mul(fLow, fLow, invScaleF32, fullMask);
                AscendC::MicroAPI::Mul(fHigh, fHigh, invScaleF32, fullMask);
                AscendC::MicroAPI::Cast<float8_e4m3_t, float, packTrait>(fp8Low, fLow, fullMask);
                AscendC::MicroAPI::Cast<float8_e4m3_t, float, packTrait>(fp8High, fHigh, fullMask);

                const uint32_t dstOff0 = WIDE ? (static_cast<uint32_t>(r) * 2U * dp) : (static_cast<uint32_t>(r) * dp);
                const uint32_t dstOff1 = WIDE ? ((static_cast<uint32_t>(r) * 2U + 1U) * dp) : (static_cast<uint32_t>(r + 16U) * dp);

                AscendC::MicroAPI::StoreAlign<float8_e4m3_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstGroupPtr + dstOff0, fp8Low, fullMask);
                AscendC::MicroAPI::StoreAlign<float8_e4m3_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstGroupPtr + dstOff1, fp8High, fullMask);
            }
        }
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
            scales + col, (AscendC::MicroAPI::RegTensor<uint16_t> &)combinedScales, fullMask);
    }
}

} // namespace sage::microkernel

namespace sage {
using microkernel::InitMxP;
using microkernel::QuantVRowsMxVF;
}

#endif // __CCE__
