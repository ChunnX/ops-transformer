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
 * \file block_epilogue_swiglu_mx_quant.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#define BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "mega_moe_base.h"

namespace {
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t GROUP_FLAG_IDX = 2;
constexpr uint32_t WAVE_IDX = 3;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;  // 与 mega_moe_impl.h L1_TILE_M * L1_TILE_N 保持一致
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr uint32_t GROUP_PERF_VL_BF16 = 128;
constexpr uint32_t GROUP_PERF_VL_FP32 = 64;
constexpr uint32_t GROUP_PERF_SCALE_NUM = 4;
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (1U << 23U) - 1U;
} // namespace

static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_ONE = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP8_GROUP_PERF = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
// This branch keeps only the activation/swiglu_group_quant-style bf16/fp8 implementation.
// IsInterleaved_ is retained for template compatibility, but the implementation always consumes
// current mega_moe GMM1 output as [gate | up] in the selected ping/pong half.
#define BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS                                                             \
    template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_,                              \
              typename DataTypeX1Scale_, bool IsTensorList_, bool IsInterleaved_>
#define BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS                                                                   \
    DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_, IsInterleaved_

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_,
          typename DataTypeX1Scale_, bool IsTensorList_, bool IsInterleaved_ = false>
class BlockEpilogueSwigluMxQuant {
public:
    __aicore__ inline BlockEpilogueSwigluMxQuant()
    {
    }

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;
    static_assert(AscendC::IsSameType<DataTypeIn, bfloat16_t>::value,
                  "BlockEpilogueSwigluMxQuant in this branch is bf16-input only.");
    static_assert(AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value,
                  "BlockEpilogueSwigluMxQuant in this branch is fp8-output only.");

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    // pingpongIdx: 0 = ping half, 1 = pong half.
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                       uint16_t pingpongIdx = 0);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize, uint16_t pingpongIdx = 0);

    // activation/swiglu_group_quant style bf16/fp8 path. It keeps swiglu in fp32 for
    // scale and fp8 cast, but streams per row to preserve the current half-UB pingpong layout.
    __aicore__ inline void VFDoSwigluAndQuantForMXInterleavedBf16GroupPerf(
        __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
        __ubuf__ bfloat16_t *firstSrc, __ubuf__ bfloat16_t *secondSrc,
        __ubuf__ uint16_t *invScaleScratchAddr, uint16_t mSize, uint16_t nSize);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    __aicore__ inline void CopyScaleFromUb2GmCompact(uint64_t blockCount, uint64_t offset,
                                                     AscendC::LocalTensor<int8_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    __gm__ int32_t* groupFlagListGmAddr_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    // l0cOutUbSecond_：仅在 GMM1 pingpong 启用（UbFirst+UbSecond 都 ≤ UB_CAPACITY）时真正占用 128KB。
    // 当 MAX_SINGLE_MN * sizeof(DataTypeIn) * 2 > 256KB 时（L1_TILE_M=256），单块 UbFirst 已经占满整 UB，
    // pingpong 关闭。这里把 UbSecond 的 offset 折叠到 0（与 UbFirst alias，绝不会被访问），
    // 避免描述符越界。
    static constexpr uint32_t kUbSecondOffset =
        (MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U <= 256U * 1024U)
            ? (MAX_SINGLE_MN * sizeof(DataTypeIn))
            : 0U;
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, kUbSecondOffset, MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<uint16_t> halfScale_;
    const Params *params_;

    int64_t n_;
    int64_t scaleN_;
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
};

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    params_ = &params;

    // 5-only group-perf layout. Input is bf16 interleaved [gate | up] in the current ping/pong half.
    // Output and compact scale keep the already measured offsets; no old bf16 gluRes/maxExp/scale-block staging exists.
    constexpr uint32_t MAX_SINGLE_MN_ALIAS = MAX_SINGLE_MN / SWIGLU_N_HALF; // m*n_half
    constexpr uint32_t quantOutputOffset = MAX_SINGLE_MN_ALIAS * sizeof(bfloat16_t);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantOutputOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantScaleOffset = quantOutputOffset + MAX_SINGLE_MN_ALIAS * sizeof(int8_t);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, quantScaleOffset, MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t halfScaleOffset = quantScaleOffset +
        MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, halfScaleOffset, MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if constexpr(g_coreType == AscendC::AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yScaleGmAddr + Get<Y_SCALE_IDX>(baseOffset));
        groupFlagListGmAddr_ = (__gm__ int32_t *)params_->groupFlagListGmAddr +
        Get<GROUP_FLAG_IDX>(baseOffset) * INT_CACHELINE;
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<N_VALUE>(problemShape); // n/2
    scaleN_ = Ops::Base::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE))
        * MXFP_MULTI_BASE_SIZE;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount; // 128
    uint64_t nDstUbAligned = Ops::Base::CeilAlign(static_cast<uint64_t>(singleN_),
        static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
    ub2GmParams.blockLen = singleN_;
    ub2GmParams.srcStride = (nDstUbAligned - singleN_) / AscendC::ONE_BLK_SIZE;
    ub2GmParams.dstStride = n_ - singleN_;
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2GmCompact(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{0, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE))
        * MXFP_MULTI_BASE_SIZE;
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = blockScaleN;
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = scaleN_ - blockScaleN;
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(quantScaleGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMXInterleavedBf16GroupPerf(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
    __ubuf__ bfloat16_t *firstSrc, __ubuf__ bfloat16_t *secondSrc,
    __ubuf__ uint16_t *invScaleScratchAddr, uint16_t mSize, uint16_t nSize)
{
    (void)nSize;
    uint32_t maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    }
    __ubuf__ uint8_t *scaleLocalAddr = (__ubuf__ uint8_t *)scaleDst;
    __ubuf__ float *invScaleScratch = (__ubuf__ float *)invScaleScratchAddr;
    __ubuf__ DataTypeOut *outLocalAddr = (__ubuf__ DataTypeOut *)outputDst;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> x0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> x1;
        AscendC::MicroAPI::RegTensor<float> x0Layout0;
        AscendC::MicroAPI::RegTensor<float> x0Layout1;
        AscendC::MicroAPI::RegTensor<float> x1Layout0;
        AscendC::MicroAPI::RegTensor<float> x1Layout1;
        AscendC::MicroAPI::RegTensor<float> yLayout0;
        AscendC::MicroAPI::RegTensor<float> yLayout1;
        AscendC::MicroAPI::RegTensor<float> y0;
        AscendC::MicroAPI::RegTensor<float> y1;
        AscendC::MicroAPI::RegTensor<float> yMax0;
        AscendC::MicroAPI::RegTensor<float> yMax1;
        AscendC::MicroAPI::RegTensor<float> yMax1Layout0;
        AscendC::MicroAPI::RegTensor<float> yMax1Layout1;
        AscendC::MicroAPI::RegTensor<float> scale;
        AscendC::MicroAPI::RegTensor<float> clampScale;
        AscendC::MicroAPI::RegTensor<float> invScale;
        AscendC::MicroAPI::RegTensor<float> invScale0;
        AscendC::MicroAPI::RegTensor<float> invScale1;
        AscendC::MicroAPI::RegTensor<float> invScaleForY0;
        AscendC::MicroAPI::RegTensor<float> invScaleForY1;
        AscendC::MicroAPI::RegTensor<float> tmp;
        AscendC::MicroAPI::RegTensor<uint32_t> zeroU32;
        AscendC::MicroAPI::RegTensor<uint32_t> oneU32;
        AscendC::MicroAPI::RegTensor<uint32_t> coeff;
        AscendC::MicroAPI::RegTensor<uint32_t> manMask;
        AscendC::MicroAPI::RegTensor<uint32_t> expMask;
        AscendC::MicroAPI::RegTensor<uint32_t> biasU32;
        AscendC::MicroAPI::RegTensor<uint32_t> scaleExp;
        AscendC::MicroAPI::RegTensor<uint32_t> manBits;
        AscendC::MicroAPI::RegTensor<uint32_t> manNonZero;
        AscendC::MicroAPI::RegTensor<uint32_t> expScale;
        AscendC::MicroAPI::RegTensor<uint32_t> scaleU32;
        AscendC::MicroAPI::RegTensor<int32_t> invExp;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleU16;
        AscendC::MicroAPI::RegTensor<uint8_t> scaleU8;
        AscendC::MicroAPI::RegTensor<DataTypeOut> y0Fp8;
        AscendC::MicroAPI::RegTensor<DataTypeOut> y1Fp8;
        AscendC::MicroAPI::MaskReg maskB16 =
            AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskF32 =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t scaleMaskNum = GROUP_PERF_SCALE_NUM;
        uint32_t invScratchMaskNum = GROUP_PERF_SCALE_NUM * GROUP_PERF_SCALE_NUM;
        AscendC::MicroAPI::MaskReg maskScale = AscendC::MicroAPI::UpdateMask<float>(scaleMaskNum);
        AscendC::MicroAPI::MaskReg maskInvScratch = AscendC::MicroAPI::UpdateMask<float>(invScratchMaskNum);
        AscendC::MicroAPI::MaskReg manNonZeroMask;
        AscendC::MicroAPI::UnalignRegForStore scaleStoreReg;

        AscendC::MicroAPI::Duplicate(zeroU32, static_cast<uint32_t>(0), maskF32);
        AscendC::MicroAPI::Duplicate(oneU32, static_cast<uint32_t>(1), maskF32);
        AscendC::MicroAPI::Duplicate(coeff, maxValueInt, maskF32);
        AscendC::MicroAPI::Duplicate(expMask, FAST_LOG_AND_VALUE1, maskScale);
        AscendC::MicroAPI::Duplicate(manMask, FAST_LOG_AND_VALUE2, maskScale);
        AscendC::MicroAPI::Duplicate(biasU32, static_cast<uint32_t>(127), maskScale);

        for (uint16_t row = 0; row < mSize; ++row) {
            uint32_t srcOffset = static_cast<uint32_t>(row) * GROUP_PERF_VL_BF16 * SWIGLU_N_HALF;
            uint32_t dstOffset = static_cast<uint32_t>(row) * GROUP_PERF_VL_BF16;
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x0, firstSrc + srcOffset);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x1, secondSrc + srcOffset);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(x0Layout0, x0, maskB16);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ONE>(x0Layout1, x0, maskB16);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(x1Layout0, x1, maskB16);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ONE>(x1Layout1, x1, maskB16);

            AscendC::MicroAPI::Muls(tmp, x0Layout0, static_cast<float>(-1.0f), maskF32);
            AscendC::MicroAPI::Exp(tmp, tmp, maskF32);
            AscendC::MicroAPI::Adds(tmp, tmp, static_cast<float>(1.0f), maskF32);
            AscendC::MicroAPI::Div(yLayout0, x0Layout0, tmp, maskF32);
            AscendC::MicroAPI::Mul(yLayout0, yLayout0, x1Layout0, maskF32);

            AscendC::MicroAPI::Muls(tmp, x0Layout1, static_cast<float>(-1.0f), maskF32);
            AscendC::MicroAPI::Exp(tmp, tmp, maskF32);
            AscendC::MicroAPI::Adds(tmp, tmp, static_cast<float>(1.0f), maskF32);
            AscendC::MicroAPI::Div(yLayout1, x0Layout1, tmp, maskF32);
            AscendC::MicroAPI::Mul(yLayout1, yLayout1, x1Layout1, maskF32);

            AscendC::MicroAPI::Interleave(y0, y1, yLayout0, yLayout1);
            AscendC::MicroAPI::Abs(yLayout0, yLayout0, maskF32);
            AscendC::MicroAPI::Abs(yLayout1, yLayout1, maskF32);
            AscendC::MicroAPI::Max(yMax0, yLayout0, yLayout1, maskF32);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(yMax1, yMax0, maskF32);
            AscendC::MicroAPI::DeInterleave<float>(yMax1Layout0, yMax1Layout1, yMax1, yMax1);
            AscendC::MicroAPI::Max(scale, yMax1Layout0, yMax1Layout1, maskScale);
            AscendC::MicroAPI::Maxs(clampScale, scale, 0.0001f, maskScale);
            AscendC::MicroAPI::Mul(scale, clampScale, (AscendC::MicroAPI::RegTensor<float> &)coeff, maskScale);

            AscendC::MicroAPI::ShiftRights(scaleExp, (AscendC::MicroAPI::RegTensor<uint32_t> &)scale,
                                           static_cast<int16_t>(FAST_LOG_SHIFT_BITS), maskScale);
            AscendC::MicroAPI::And(scaleExp, scaleExp, expMask, maskScale);
            AscendC::MicroAPI::And(manBits, (AscendC::MicroAPI::RegTensor<uint32_t> &)scale, manMask, maskScale);
            AscendC::MicroAPI::Compare<uint32_t, AscendC::CMPMODE::NE>(
                manNonZeroMask, manBits, zeroU32, maskScale);
            AscendC::MicroAPI::Select(manNonZero, oneU32, zeroU32, manNonZeroMask);
            AscendC::MicroAPI::Sub(expScale, scaleExp, biasU32, maskScale);
            AscendC::MicroAPI::Add(expScale, expScale, manNonZero, maskScale);
            AscendC::MicroAPI::Add(scaleU32, expScale, biasU32, maskScale);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                scaleU16, scaleU32);
            AscendC::MicroAPI::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(scaleU8, scaleU16);
            AscendC::MicroAPI::StoreUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                scaleLocalAddr, scaleU8, scaleStoreReg, GROUP_PERF_SCALE_NUM);

            AscendC::MicroAPI::Sub(invExp, (AscendC::MicroAPI::RegTensor<int32_t> &)biasU32,
                                   (AscendC::MicroAPI::RegTensor<int32_t> &)expScale, maskScale);
            AscendC::MicroAPI::ShiftLefts(
                (AscendC::MicroAPI::RegTensor<int32_t> &)invScale, invExp,
                static_cast<int16_t>(FAST_LOG_SHIFT_BITS), maskScale);
            AscendC::MicroAPI::Interleave(invScale0, invScale1, invScale, invScale);
            AscendC::MicroAPI::Interleave(invScale1, invScale, invScale0, invScale0);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                invScaleScratch, invScale1, maskInvScratch);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_E2B_B32>(
                invScaleForY0, invScaleScratch);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_E2B_B32>(
                invScaleForY1, invScaleScratch + SCALE_ONCE_NUM);

            AscendC::MicroAPI::Mul(y0, y0, invScaleForY0, maskF32);
            AscendC::MicroAPI::Mul(y1, y1, invScaleForY1, maskF32);
            AscendC::MicroAPI::Cast<DataTypeOut, float, CAST_FP32_TO_FP8_GROUP_PERF>(y0Fp8, y0, maskF32);
            AscendC::MicroAPI::Cast<DataTypeOut, float, CAST_FP32_TO_FP8_GROUP_PERF>(y1Fp8, y1, maskF32);
            AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr + dstOffset, y0Fp8, maskF32);
            AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr + dstOffset + GROUP_PERF_VL_FP32, y1Fp8, maskF32);
        }
        AscendC::MicroAPI::StoreUnAlignPost<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            scaleLocalAddr, scaleStoreReg, 0);
    }
    return;
}


BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(uint16_t mSize,
                                                                                       uint16_t pingpongIdx)
{
    // 5-only path: GMM1 output is bf16 [gate | up] in the selected ping/pong half.
    constexpr uint32_t pongElemOf_DataTypeIn = MAX_SINGLE_MN;
    constexpr uint32_t pongElemOf_int8 = MAX_SINGLE_MN * sizeof(DataTypeIn);
    constexpr uint32_t pongElemOf_uint16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(uint16_t);
    const uint32_t pongMul = (pingpongIdx == 1U) ? 1U : 0U;

    __ubuf__ DataTypeIn *l0cOutUbBase =
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr() + pongMul * pongElemOf_DataTypeIn;
    __ubuf__ int8_t *quantOutputInUbAddr =
        (__ubuf__ int8_t *)quantOutput_.GetPhyAddr() + pongMul * pongElemOf_int8;
    __ubuf__ uint16_t *quantScaleOutputInUbAddr =
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *halfScaleAddr =
        (__ubuf__ uint16_t *)halfScale_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ bfloat16_t *gateSrc = (__ubuf__ bfloat16_t *)l0cOutUbBase;
    __ubuf__ bfloat16_t *upSrc = gateSrc + singleN_;

    VFDoSwigluAndQuantForMXInterleavedBf16GroupPerf(
        quantOutputInUbAddr, quantScaleOutputInUbAddr, gateSrc, upSrc, halfScaleAddr, mSize, singleN_);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(const BlockShape &blockShape,
                                                                                   const BlockCoord &blockCoord,
                                                                                   uint16_t pingpongIdx)
{
    singleM_ = Get<M_VALUE>(blockShape); // 128
    singleN_ = Get<N_VALUE>(blockShape); // 256

    if (singleM_ == 0) {
        return;
    }

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    VFDoSwigluForMX(singleM_, pingpongIdx); // switch(x)*y 计算quant quantScale
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

    constexpr uint32_t PONG_INT8_ELEMS = MAX_SINGLE_MN * sizeof(DataTypeIn);
    if (pingpongIdx == 1U) {
        AscendC::LocalTensor<int8_t> quantOutputPong = quantOutput_[PONG_INT8_ELEMS];
        AscendC::LocalTensor<int8_t> quantScalePong = quantScaleOutput_[PONG_INT8_ELEMS];
        CopyOutputFromUb2Gm(singleM_, yOffset, quantOutputPong);
        CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScalePong);
    } else {
        CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
        CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    uint64_t mLoc = Get<WAVE_IDX>(blockCoord);
    uint32_t waveIdx = static_cast<uint32_t>(mLoc / L1_TILE_M);
    AscendC::AtomicAdd((__gm__ int32_t*)(groupFlagListGmAddr_ + waveIdx * INT_CACHELINE), 1);
    return;
}

#endif // BLOCK_EPILOGUE_SWIGLU_QUANT_H
#endif
