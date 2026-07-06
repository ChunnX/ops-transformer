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

namespace SwigluQuantMsg {
enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

enum class QuantDtype : uint8_t {
    DEFAULT = 0x0U,
    FP8_E4M3FN = 0x1U,
    FP8_E5M2 = 0x1U << 1,
};
} // namespace SwigluQuantMsg

namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t GROUP_FLAG_IDX = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint32_t FLAG_VALUE_ONE = 1;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t CONST_64 = 64;
constexpr uint32_t VF_LEN_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
} // namespace

constexpr AscendC::MicroAPI::CastTrait ctInt322Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctFp322Half = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32Zero = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32One = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_ONE = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
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
        float clampLimit{0.0f};
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

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
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                       uint16_t pingpongIdx = 0);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize, uint16_t pingpongIdx = 0);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN, uint16_t pingpongIdx = 0);

    template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc = false>
    __aicore__ inline void VFDoSwigluAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                   __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                   __ubuf__ bfloat16_t *gluResAddr,
                                                   __ubuf__ uint16_t *maxExpAddr,
                                                   __ubuf__ uint16_t *halfScaleLocalAddr,
                                                   uint16_t mSize, uint16_t nSize);

    __aicore__ inline void VFDoSwigluAndQuantForMXInterleavedBf16(
        __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
        __ubuf__ bfloat16_t *firstSrc, __ubuf__ bfloat16_t *secondSrc,
        __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
        __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize, uint16_t nSize);

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);

    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    __aicore__ inline void CopyScaleFromUb2GmCompact(uint64_t blockCount, uint64_t offset,
                                                     AscendC::LocalTensor<int8_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    __gm__ int32_t* groupFlagListGmAddr_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    static constexpr uint32_t kUbSecondOffset =
        (MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U <= 256U * 1024U)
            ? (MAX_SINGLE_MN * sizeof(DataTypeIn))
            : 0U;
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, kUbSecondOffset, MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<int8_t> quantScaleBlockOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;
    const Params *params_;

    int64_t n_;
    int64_t scaleN_;
    int64_t scaleBlockN_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;
    int64_t UBBlockSize_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;

    BlockCoord blockCoord_{0, 0, 0, 0, 0, 0};
    float clampLimit_{0.0f};
};

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    params_ = &params;
    clampLimit_ = params.clampLimit;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP; // this
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    constexpr uint32_t MAX_SINGLE_MN_ALIAS = MAX_SINGLE_MN / SWIGLU_N_HALF;
    constexpr uint32_t gluResOffset = 0;
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, gluResOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantOutputOffset = gluResOffset + MAX_SINGLE_MN_ALIAS * sizeof(bfloat16_t);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantOutputOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantScaleOffset = quantOutputOffset + MAX_SINGLE_MN_ALIAS * sizeof(int8_t);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, quantScaleOffset, MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t maxExpOffset = quantScaleOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    maxExp_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, maxExpOffset, MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t halfScaleOffset = maxExpOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, halfScaleOffset, MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t quantScaleBlockOffset = halfScaleOffset +
        MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    quantScaleBlockOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, quantScaleBlockOffset, params_->baseM * AscendC::ONE_BLK_SIZE);
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
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = singleN_ >> 1;
        ub2GmParams.dstStride = (n_ - singleN_) >> 1;
        offset = offset >> 1;
    } else {
        uint64_t nDstUbAligned = Ops::Base::CeilAlign(static_cast<uint64_t>(singleN_),
            static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        ub2GmParams.blockLen = singleN_;
        ub2GmParams.srcStride = (nDstUbAligned - singleN_) / AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = n_ - singleN_;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE))
        * MXFP_MULTI_BASE_SIZE; // 256 / 32 = 8
    ub2GmParams.blockLen = blockScaleN * sizeof(int8_t); // 8
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.dstStride = (scaleN_ - blockScaleN) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], src, ub2GmParams);
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
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::MaskReg scaleMask1, scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, onceNum);
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                                       scaleMask1);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                                       scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, scaleNum);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    int64_t onceNum = QUANT_ONCE_NUM_FP4;
    int64_t onceNumMxScale = CONST_64;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, sharedExp, scaleValue, scaleBias, halfScale;
        AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_);
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor, zeroRegTensor, nanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, onceNum);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, onceNumMxScale, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, onceNum, preMaskScale);
        }
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    (void)totalCountInUB;
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8One, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8Zero, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8One, maskAllB8);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, onceXNum, maskAllB8);
        }
    }
    return;
}


BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    int64_t onceYNum = OUT_ELE_NUM_ONE_BLK;
    static constexpr AscendC::MicroAPI::CastTrait castTrait = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP4, vdExp1FP4;
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP4, onceYNum, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP4, onceYNum, dataMask1);
        }
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc>
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr,
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t nSrcUbAligned;
    if constexpr (IsInterleavedSrc) {
        nSrcUbAligned = static_cast<uint32_t>(nSize) * 2U;
    } else {
        nSrcUbAligned = Ops::Base::CeilAlign(static_cast<uint32_t>(nSize),
            static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn)));
    }
    uint32_t nDstUbAligned = Ops::Base::CeilAlign(static_cast<uint32_t>(nSize),
        static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    uint16_t dim0VfTimes = mSize;
    uint16_t dim1VfTimes = nSize / VF_LEN_FP32;
    uint32_t dim1Tail = nSize % VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    __ubuf__ DataTypeIn *firstTailAddr = firstSrc;
    __ubuf__ DataTypeIn *secondTailAddr = secondSrc;
    __ubuf__ bfloat16_t *swigluTailAddr1 = gluResAddr;
    __ubuf__ bfloat16_t *swigluTailAddr2 = gluResAddr;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = nDstUbAligned - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        uint32_t offsetAlign = dim1VfTimes * VF_LEN_FP32;
        firstTailAddr = firstSrc + offsetAlign;
        secondTailAddr = secondSrc + offsetAlign;
        swigluTailAddr1 = gluResAddr + offsetAlign;
        swigluTailAddr2 = gluResAddr + offsetAlign + dim1TailTimes * VF_LEN_FP32;
    }
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset =
                    AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, nSrcUbAligned,
                                                                 dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask);
                AscendC::MicroAPI::Exp(expReg, negReg, mask);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset =
                    AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, nDstUbAligned,
                                                                  dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, nDstUbAligned);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    swigluTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(swigluTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned; // 128*256
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE; // 128*256 / 32 = 128 * 8
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2); // 128
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_; // 8
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum); // 获取最大值
    ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum); // 计算scale和halfScale
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum); // 计算量化后的值
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMXInterleavedBf16(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
    __ubuf__ bfloat16_t *firstSrc, __ubuf__ bfloat16_t *secondSrc,
    __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
    __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize, uint16_t nSize)
{
    uint32_t alignDim1In = static_cast<uint32_t>(nSize) * SWIGLU_N_HALF;
    uint32_t alignDim1Out = Ops::Base::CeilAlign(static_cast<uint32_t>(nSize),
        static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    uint16_t dim0VfTimes = mSize;
    uint16_t dim1VfTimes = nSize / VF_LEN_FP32;
    uint32_t dim1Tail = nSize % VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    auto x1UbAddr1 = firstSrc;
    auto x2UbAddr1 = secondSrc;
    auto swigluUbAddr1 = gluResAddr;
    auto swigluUbAddr2 = gluResAddr;
    bfloat16_t numZero = 0;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = alignDim1Out - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        uint32_t offsetAlign = dim1VfTimes * VF_LEN_FP32;
        x1UbAddr1 = firstSrc + offsetAlign;
        x2UbAddr1 = secondSrc + offsetAlign;
        swigluUbAddr1 = gluResAddr + offsetAlign;
        swigluUbAddr2 = gluResAddr + offsetAlign + dim1TailTimes * VF_LEN_FP32;
    }
    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregX1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::MaskReg mask =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset =
                    AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                        dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask);
                AscendC::MicroAPI::Exp(expReg, negReg, mask);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset =
                    AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                        dim0vfLoopIdx, alignDim1Out, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, alignDim1In);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, x1UbAddr1, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, x2UbAddr1, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    swigluUbAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                AscendC::MicroAPI::Duplicate(vregX1, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(swigluUbAddr2, vregX1, outOffset1, mask3);
            }
        }
    }

    uint32_t totalDataInUb = mSize * alignDim1Out;
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * SWIGLU_N_HALF - 1) /
        (vlForHalfNumber_ * SWIGLU_N_HALF);
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);
    ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(uint16_t mSize,
                                                                                       uint16_t pingpongIdx)
{
    constexpr uint32_t pongElemOf_DataTypeIn = MAX_SINGLE_MN;
    constexpr uint32_t pongElemOf_bf16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(bfloat16_t);
    constexpr uint32_t pongElemOf_int8 = MAX_SINGLE_MN * sizeof(DataTypeIn);
    constexpr uint32_t pongElemOf_uint16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(uint16_t);
    const uint32_t pongMul = (IsInterleaved_ && pingpongIdx == 1U) ? 1U : 0U;

    __ubuf__ DataTypeIn *l0cOutUbBase =
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr() + pongMul * pongElemOf_DataTypeIn;
    __ubuf__ bfloat16_t *gluResAddr =
        (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr() + pongMul * pongElemOf_bf16;
    __ubuf__ int8_t *quantOutputInUbAddr =
        (__ubuf__ int8_t *)quantOutput_.GetPhyAddr() + pongMul * pongElemOf_int8;
    __ubuf__ uint16_t *quantScaleOutputInUbAddr =
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *maxExpAddr =
        (__ubuf__ uint16_t *)maxExp_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *halfScaleAddr =
        (__ubuf__ uint16_t *)halfScale_.GetPhyAddr() + pongMul * pongElemOf_uint16;

    if constexpr (IsInterleaved_) {
        __ubuf__ DataTypeIn *l0cOutUbSecondAddr = l0cOutUbBase + singleN_;
        if constexpr (AscendC::IsSameType<DataTypeIn, bfloat16_t>::value) {
            VFDoSwigluAndQuantForMXInterleavedBf16(
                quantOutputInUbAddr, quantScaleOutputInUbAddr,
                l0cOutUbBase, l0cOutUbSecondAddr,
                gluResAddr, maxExpAddr, halfScaleAddr, mSize, singleN_);
        } else {
            VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, true>(
                quantOutputInUbAddr, quantScaleOutputInUbAddr,
                l0cOutUbBase, l0cOutUbSecondAddr,
                gluResAddr, maxExpAddr, halfScaleAddr, mSize, singleN_);
        }
    } else {
        __ubuf__ DataTypeIn *l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
        VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, false>(
            quantOutputInUbAddr, quantScaleOutputInUbAddr,
            l0cOutUbBase, l0cOutUbSecondAddr,
            gluResAddr, maxExpAddr, halfScaleAddr, mSize, singleN_);
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::TransMxScaleLayout(uint16_t mSize,
                                                                                           uint16_t scaleBlockN,
                                                                                           uint16_t pingpongIdx)
{
    const uint32_t pongShift = (IsInterleaved_ && pingpongIdx == 1U) ? MAX_SINGLE_MN * sizeof(DataTypeIn) : 0U;
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr() + pongShift;
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr =
        (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr() + pongShift;
    // scale layout: (mSize*8) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) { // 128
            uint32_t elemNum = scaleBlockN; // 8
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
            AscendC::MicroAPI::RegTensor<int8_t> vreg0;
            AscendC::MicroAPI::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb); // 初始化
            AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, srcUb); // ?
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE; // mId * 32
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
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
    scaleBlockN_ = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE))
        * MXFP_MULTI_BASE_SIZE;
    blockCoord_ = blockCoord;

    if (singleM_ == 0) {
        return;
    }

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t); // 256 / 2 = 128
    UBBlockSize_ = BLOCK_SIZE; // 32
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_; // 256 / 32 = 8

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    VFDoSwigluForMX(singleM_, pingpongIdx); // switch(x)*y 计算quant quantScale
    if constexpr (!(IsInterleaved_ && AscendC::IsSameType<DataTypeIn, bfloat16_t>::value)) {
        TransMxScaleLayout(singleM_, scaleBlockN_, pingpongIdx); // 重排scale
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    if constexpr (IsInterleaved_) {
        constexpr uint32_t PONG_INT8_ELEMS = MAX_SINGLE_MN * sizeof(DataTypeIn);
        if (pingpongIdx == 1U) {
            AscendC::LocalTensor<int8_t> quantOutputPong = quantOutput_[PONG_INT8_ELEMS];
            AscendC::LocalTensor<int8_t> quantScaleBlockPong = quantScaleBlockOutput_[PONG_INT8_ELEMS];
            AscendC::LocalTensor<int8_t> quantScalePong = quantScaleOutput_[PONG_INT8_ELEMS];
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutputPong);
            if constexpr (AscendC::IsSameType<DataTypeIn, bfloat16_t>::value) {
                CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScalePong);
            } else {
                CopyScaleFromUb2Gm(singleM_, yScaleOffset, quantScaleBlockPong);
            }
        } else {
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
            if constexpr (AscendC::IsSameType<DataTypeIn, bfloat16_t>::value) {
                CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
            } else {
                CopyScaleFromUb2Gm(singleM_, yScaleOffset, quantScaleBlockOutput_);
            }
        }
    } else {
        CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
        CopyScaleFromUb2Gm(singleM_, yScaleOffset, quantScaleBlockOutput_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
    return;
}

#endif // BLOCK_EPILOGUE_SWIGLU_QUANT_H
#endif
