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
 * \file block_epilogue_gelu_mx_quant.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "math/erf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

enum class QuantAlg : uint32_t {
    OCP = 0,
    BLAS = 1,
};

enum class GeluAlg : uint8_t {
    TANH = 0,
    ERF = 1,
};

enum class ROUND_MODE_FP4 : uint8_t {
    RINT = 0,
    FLOOR = 1,
    ROUND = 2,
};

namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int64_t MX_SCALE_ALIGN_SIZE = 2;
constexpr float TANH_APPROX_FACTOR = 1.0f / 0.044715f;
constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
constexpr float ONE_OVER_SQRT_TWO = 0.707106781f;

constexpr uint32_t MAX_SINGLE_MN = 128 * 256;
constexpr uint32_t MAX_SINGLE_SCALE_NUM = MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;  // 0b0111 1111 1000 0000
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;  // 0b0111 1111 0000 0000
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
// elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;  // 0b 0000 0100 0000 0000 右移7位为8
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;  // 0b 0000 0111 1000 0000 右移7位为15
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;  // 0b 0000 0001 0000 0000 右移7位为2
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;  // 右移7位为0

constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t NUMBER_HALF = 0x00400000;
constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL = 60;
} // namespace

using namespace Gemm;

template <typename DataTypeOut_, typename DataTypeIn_>
class BlockEpilogueGeluMxQuant {
public:
    __aicore__ inline BlockEpilogueGeluMxQuant()
    {
    }

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        GeluAlg geluAlg;
        QuantAlg quantAlg;
        ROUND_MODE_FP4 fp4RoundMode;
        Params() = default;
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    // shape
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetTensor();

    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoGeluForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize);
    __aicore__ inline void VFDoGeluAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                        uint16_t mSize, uint16_t nSize);
    __aicore__ inline void GeluTanh(__ubuf__ bfloat16_t *geluResAddr, uint16_t mSize, uint16_t nSize,
                                    uint32_t nAligned);
    __aicore__ inline void GeluErf(__ubuf__ bfloat16_t *geluResAddr, uint16_t mSize, uint16_t nSize,
                                    uint32_t nAligned);
    __aicore__ inline void ComputeScaleOCP(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);
    __aicore__ inline void ComputeScalecuBLAS(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);
    __aicore__ inline void ComputeMaxExpOCP(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                        uint16_t loopNum);
    __aicore__ inline void ComputeMaxExpcuBLAS(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                        uint16_t loopNum);
    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);
    template <AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);
    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> cLocal_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<int8_t> quantScaleBlockOutput_;
    AscendC::LocalTensor<bfloat16_t> geluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    AscendC::LocalTensor<float> erfTmp_;
    AscendC::LocalTensor<float> fp32Tmp_;
    AscendC::LocalTensor<float> geluFp32Tmp_;

    const Params *params_;

    int64_t n_;
    int64_t scaleN_;
    int64_t scaleNAlign_;
    int64_t scaleBlockN_;
    uint32_t subBlockIdx_;
    uint32_t singleM_;
    uint32_t singleN_;

    int64_t UBBlockSize_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint32_t vlForFloat32Number_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;
    uint32_t dtypeMax_ = 0;

    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::Init(Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    // 量化结果的Nan值会转变成极大值
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    params_ = &params;
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
        dtypeMax_ = FP8_E4M3_MAX;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        dtypeMax_ = FP8_E5M2_MAX;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
        dtypeMax_ = 0;  // FP4不支持
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
        dtypeMax_ = 0;  // FP4不支持
    }

    // out
    constexpr uint32_t afterIn = MAX_SINGLE_MN * sizeof(DataTypeIn);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterIn, MAX_SINGLE_MN);
    constexpr uint32_t afterOut = afterIn + MAX_SINGLE_MN * sizeof(int8_t);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterOut, MAX_SINGLE_SCALE_NUM);
    // gelu res
    constexpr uint32_t afterIO = afterOut + MAX_SINGLE_SCALE_NUM * sizeof(int8_t);
    geluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, afterIO, MAX_SINGLE_MN);
    // sharedExp
    constexpr uint32_t afterIOAndGelu = afterIO + MAX_SINGLE_MN * sizeof(bfloat16_t);
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGelu, MAX_SINGLE_SCALE_NUM);
    constexpr uint32_t afterIOAndGeluExp = afterIOAndGelu + MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGeluExp, MAX_SINGLE_SCALE_NUM);
    constexpr uint32_t realScaleBlockOffset = afterIOAndGeluExp + MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    quantScaleBlockOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, realScaleBlockOffset,
        params_->baseM / AscendC::GetTaskRation() * AscendC::ONE_BLK_SIZE);
    if (params_->geluAlg == GeluAlg::ERF) {
        uint32_t ubOffset = realScaleBlockOffset
                            + params_->baseM / AscendC::GetTaskRation() * AscendC::ONE_BLK_SIZE * sizeof(int8_t);
        if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
            erfTmp_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ubOffset, params_->baseN);
            geluFp32Tmp_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,
                                                        ubOffset + params_->baseN, params_->baseN);
        } else {
            fp32Tmp_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ubOffset, params_->baseN);
            erfTmp_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,
                                                ubOffset + params_->baseN, params_->baseN);
            geluFp32Tmp_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,
                                                ubOffset + params_->baseN + params_->baseN, params_->baseN);
        }
    }
    
    quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yGmAddr);
    quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yScaleGmAddr);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateGlobalAddr(
                                                                const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGlobal_.SetGlobalBuffer(
                    (__gm__ int8_t *)params_->yGmAddr + AscendC::Te::Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer(
                    (__gm__ int8_t *)params_->yScaleGmAddr + AscendC::Te::Get<Y_SCALE_IDX>(baseOffset));
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = AscendC::Te::Get<MNK_N>(problemShape);
    scaleN_ = CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(BLOCK_SIZE));
    scaleNAlign_ = CeilAlign(scaleN_, MX_SCALE_ALIGN_SIZE);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = singleN_ * sizeof(int8_t);
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(int8_t);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = ub2GmParams.blockLen >> 1;
        ub2GmParams.dstStride = ub2GmParams.dstStride >> 1;
        offset = offset >> 1;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockLen = scaleBlockN_ * sizeof(int8_t);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (scaleNAlign_ - scaleBlockN_) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], src, ub2GmParams);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeMaxExpOCP(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;

        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg Mask =
                                AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * 2);
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16, Mask);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16, Mask);
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeMaxExpcuBLAS(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

        AscendC::MicroAPI::RegTensor<uint16_t> absMask16Bit;
        AscendC::MicroAPI::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        AscendC::MicroAPI::MaskReg Mask =
                                AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * 2);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                absMask16Bit, Mask);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1,
                absMask16Bit, Mask);
            AscendC::MicroAPI::Max(
                vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, Mask);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::MicroAPI::StoreUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeScalecuBLAS(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::RegTensor<uint32_t> vdMaxExp32;
        AscendC::MicroAPI::RegTensor<uint32_t> exp32;
        AscendC::MicroAPI::RegTensor<uint32_t> man32;
        AscendC::MicroAPI::RegTensor<uint32_t> normalExp32;
        AscendC::MicroAPI::RegTensor<uint32_t> expAddOne32;
        AscendC::MicroAPI::RegTensor<uint32_t> extractExp;
        AscendC::MicroAPI::RegTensor<uint16_t> expOut;
        AscendC::MicroAPI::RegTensor<uint32_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> recExpOut;

        AscendC::MicroAPI::RegTensor<uint32_t> invMax;
        AscendC::MicroAPI::Duplicate(invMax, dtypeMax_);
        AscendC::MicroAPI::RegTensor<uint32_t> manMaskFP32;
        AscendC::MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        AscendC::MicroAPI::RegTensor<uint32_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
        AscendC::MicroAPI::RegTensor<uint32_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        AscendC::MicroAPI::RegTensor<uint32_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        uint32_t SixtyFour = 64;
        AscendC::MicroAPI::MaskReg dataMaskB16Half = AscendC::MicroAPI::UpdateMask<uint16_t>(SixtyFour);
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<uint32_t>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumScale; i++) {
            AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                        AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vdMaxExp, maxExpAddr, vlForFloat32Number_);

            AscendC::MicroAPI::Cast<float, T, castTraitHalf2Float>((AscendC::MicroAPI::RegTensor<float>&)vdMaxExp32,
                                                                (AscendC::MicroAPI::RegTensor<T>&)vdMaxExp, mask);
            AscendC::MicroAPI::Compare<uint32_t, AscendC::CMPMODE::LT>(cmpResult, vdMaxExp32, expMask, mask);
            AscendC::MicroAPI::Compare<uint32_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp32, zeroRegTensor32, mask);

            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)vdMaxExp32, (AscendC::MicroAPI::RegTensor<float>&)vdMaxExp32,
                (AscendC::MicroAPI::RegTensor<float>&)invMax, mask);
            AscendC::MicroAPI::ShiftRights(exp32, vdMaxExp32, SHR_NUM_FOR_FP32, mask);
            AscendC::MicroAPI::And(man32, vdMaxExp32, manMaskFP32, mask);

            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p0, exp32, NUMBER_ZERO, mask);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR, mask);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_ZERO, mask);
            AscendC::MicroAPI::MaskAnd(p0, p0, p1, mask);
            AscendC::MicroAPI::MaskAnd(p0, p0, p2, mask);

            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(p1, exp32, NUMBER_ZERO, mask);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_HALF, mask);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, mask);
            AscendC::MicroAPI::MaskOr(p0, p0, p1, mask);

            AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, mask);
            AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);
            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr + i * 32, expOut, dataMaskB16Half);

            AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, mask);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, mask);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);
            AscendC::MicroAPI::StoreAlign<uint16_t>(
                halfScaleLocalAddr + i * vlForFloat32Number_, recExpOut, dataMaskB16Half);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeScaleOCP(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, maskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_);
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            maskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vlForHalfNumber_);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                       maskScale); // INF\nAN
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, maskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       maskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask); // 大于emax取emax
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, maskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, maskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber_ >> 1, maskScale);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       maskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, maskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, maskScale);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    uint32_t totalCountInUB2 = totalCountInUB * 2;
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1, vdExp0Convert, vdExp1Convert;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask3 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
            dataMask4 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce_);

            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
            AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
    return;
}


template <typename DataTypeOut_, typename DataTypeIn_>
template <AscendC::RoundMode roundMode>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::MaskReg dataMask2;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<T> vdExp0Convert;
        AscendC::MicroAPI::RegTensor<T> vdExp1Convert;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

        AscendC::MicroAPI::RegTensor<U> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP4;

        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * 2);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, elementAfterReduce_);

            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask2);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::GeluTanh(__ubuf__ bfloat16_t *geluResAddr,
                                                            uint16_t mSize, uint16_t nSize, uint32_t nAligned)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(float);  // 需要转换成float32计算
    uint16_t OneRowRepeatTimes = CeilDiv(nSize, sizePerRepeat); // 计算为64位对齐

    __ubuf__ DataTypeIn *src = (__ubuf__ DataTypeIn *)cLocal_.GetPhyAddr();
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInput;
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInputSqr;
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInputCub;
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregOutput;
    AscendC::MicroAPI::RegTensor<bfloat16_t, AscendC::MicroAPI::RegTraitNumOne> vregOutput16;  // gelu总是输出bfloat16
    static constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32Zero = {AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait ctFp32toBf16 = {AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    AscendC::MicroAPI::MaskReg mask;
    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        __VEC_SCOPE__
        {
            // 每行计算一次
            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
                uint32_t count = nSize;
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(count);
                    uint32_t offset = mIdx * nAligned + vfBlockIdx * sizePerRepeat;
                    AscendC::MicroAPI::DataCopy(vregInput, src + offset);
                    AscendC::MicroAPI::Mul(vregInputSqr, vregInput, vregInput, mask);
                    AscendC::MicroAPI::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                    AscendC::MicroAPI::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                    AscendC::MicroAPI::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                    AscendC::MicroAPI::Exp(vregInputCub, vregInputCub, mask);
                    AscendC::MicroAPI::Adds(vregInputCub, vregInputCub, 1.0f, mask);
                    AscendC::MicroAPI::Div(vregOutput, vregInput, vregInputCub, mask);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        geluResAddr + offset, vregOutput16, mask);
                }
            }
        }
    } else {
        AscendC::MicroAPI::RegTensor<DataTypeIn, AscendC::MicroAPI::RegTraitNumOne> vregInput16;
        __VEC_SCOPE__
        {
            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) { // 需要计算m次
                uint32_t count = nSize;
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(count);
                    uint32_t offset = mIdx * nAligned + vfBlockIdx * sizePerRepeat;
                    AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                                    src + offset);
                    AscendC::MicroAPI::Cast<float, DataTypeIn, ctHalf2Fp32Zero>(vregInput, vregInput16, mask);
                    AscendC::MicroAPI::Mul(vregInputSqr, vregInput, vregInput, mask);
                    AscendC::MicroAPI::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                    AscendC::MicroAPI::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                    AscendC::MicroAPI::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                    AscendC::MicroAPI::Exp(vregInputCub, vregInputCub, mask);
                    AscendC::MicroAPI::Adds(vregInputCub, vregInputCub, 1.0f, mask);
                    AscendC::MicroAPI::Div(vregOutput, vregInput, vregInputCub, mask);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        geluResAddr + offset, vregOutput16, mask);
                }
            }
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::GeluErf(__ubuf__ bfloat16_t *geluResAddr,
                                                        uint16_t mSize, uint16_t nSize, uint32_t nAligned)
{
    // 0.5*x*(1+erf(x/√2)
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t OneRowRepeatTimes = CeilDiv(nSize, sizePerRepeat); // 计算为64位对齐

    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInput1;
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInput2;
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInputAdds;
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInputMuls;
    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregOutput;
    AscendC::MicroAPI::RegTensor<bfloat16_t, AscendC::MicroAPI::RegTraitNumOne> vregOutput16;  // gelu总是输出bfloat16
    AscendC::MicroAPI::MaskReg mask;
    static constexpr AscendC::MicroAPI::CastTrait ctFp32toBf16 = {AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        for (uint32_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Muls(geluFp32Tmp_, cLocal_[mIdx * nAligned], ONE_OVER_SQRT_TWO, nSize);
            AscendC::Erf(erfTmp_, geluFp32Tmp_, nSize);
            __ubuf__ float *src = (__ubuf__ float *)cLocal_.GetPhyAddr();
            __ubuf__ float* erfAddr = (__ubuf__ float*)erfTmp_.GetPhyAddr();
            uint32_t count = nSize;
            __VEC_SCOPE__
            {
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(count);
                    uint32_t mnOffset = mIdx * nAligned + vfBlockIdx * sizePerRepeat;
                    AscendC::MicroAPI::DataCopy(vregInput1, (__ubuf__ float*)(erfAddr + vfBlockIdx * sizePerRepeat));
                    AscendC::MicroAPI::DataCopy(vregInput2, (__ubuf__ float*)(src + mnOffset));
                    AscendC::MicroAPI::Adds(vregInputAdds, vregInput1, (float)1.0, mask);
                    AscendC::MicroAPI::Muls(vregInputMuls, vregInput2, (float)0.5, mask);
                    AscendC::MicroAPI::Mul(vregOutput, vregInputAdds, vregInputMuls, mask);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                                geluResAddr + mnOffset, vregOutput16, mask);
                }
            }
        }
    } else {
        for (uint32_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Cast(fp32Tmp_, cLocal_[mIdx * nAligned], AscendC::RoundMode::CAST_NONE, nSize);
            AscendC::Muls(geluFp32Tmp_, fp32Tmp_, ONE_OVER_SQRT_TWO, nSize);
            AscendC::Erf(erfTmp_, geluFp32Tmp_, nSize);
            __ubuf__ float *src = (__ubuf__ float *)fp32Tmp_.GetPhyAddr();
            __ubuf__ float* erfAddr = (__ubuf__ float*)erfTmp_.GetPhyAddr();
            uint32_t count = nSize;
            __VEC_SCOPE__
            {
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(count);
                    uint32_t nOffset = vfBlockIdx * sizePerRepeat;
                    uint32_t mnOffset = mIdx * nAligned + nOffset;
                    AscendC::MicroAPI::DataCopy(vregInput1, (__ubuf__ float*)(erfAddr + nOffset));
                    AscendC::MicroAPI::DataCopy(vregInput2, (__ubuf__ float*)(src + nOffset));
                    AscendC::MicroAPI::Adds(vregInputAdds, vregInput1, (float)1.0, mask);
                    AscendC::MicroAPI::Muls(vregInputMuls, vregInput2, (float)0.5, mask);
                    AscendC::MicroAPI::Mul(vregOutput, vregInputAdds, vregInputMuls, mask);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                                geluResAddr + mnOffset, vregOutput16, mask);
                }
            }
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::VFDoGeluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, uint16_t mSize, uint16_t nSize)
{
    AscendC::Duplicate(geluRes_, static_cast<bfloat16_t>(0.0), MAX_SINGLE_MN);  // 清除残余数据
    uint32_t nAligned = Align32(static_cast<uint32_t>(nSize));  // 输入为32位对齐

    __ubuf__ bfloat16_t *geluResAddr = (__ubuf__ bfloat16_t *)geluRes_.GetPhyAddr();
    if (params_->geluAlg == GeluAlg::ERF) {
        GeluErf(geluResAddr, mSize, nSize, nAligned);
    } else {
        GeluTanh(geluResAddr, mSize, nSize, nAligned);
    }

    uint32_t totalDataInUb = mSize * nAligned;
    uint32_t totalScaleInUb = totalDataInUb / BLOCK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
    __ubuf__ uint16_t *halfScaleLocalAddr;
    if (params_->quantAlg == QuantAlg::OCP) {
        uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
        ComputeMaxExpOCP(geluResAddr, maxExpAddr, loopDataNum);
        halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
        ComputeScaleOCP(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    } else {
        uint16_t loopScaleNum = (totalScaleInUb + vlForFloat32Number_ - 1) / vlForFloat32Number_;
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
        ComputeMaxExpcuBLAS(geluResAddr, maxExpAddr, loopDataNum);
        halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
        ComputeScalecuBLAS(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    }

    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(geluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        if (params_->fp4RoundMode == ROUND_MODE_FP4::FLOOR) {
            ComputeDataForQuantTargetFp4<AscendC::RoundMode::CAST_FLOOR>(geluResAddr, halfScaleLocalAddr,
                outputDst, totalDataInUb, loopDataNum);
        } else if ((params_->fp4RoundMode == ROUND_MODE_FP4::ROUND)) {
            ComputeDataForQuantTargetFp4<AscendC::RoundMode::CAST_ROUND>(geluResAddr, halfScaleLocalAddr,
                outputDst, totalDataInUb, loopDataNum);
        } else {  // 默认rint
            ComputeDataForQuantTargetFp4<AscendC::RoundMode::CAST_RINT>(geluResAddr, halfScaleLocalAddr,
                outputDst, totalDataInUb, loopDataNum);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::VFDoGeluForMX(uint16_t mSize)
{
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr();
    __ubuf__ uint16_t *quantScaleOutputInUbAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    VFDoGeluAndQuantForMX(quantOutputInUbAddr, quantScaleOutputInUbAddr, mSize, singleN_);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::TransMxScaleLayout(uint16_t mSize)
{
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr();
    // scale layout: (mSize*8) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elemNum = scaleBlockN_;
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
            AscendC::MicroAPI::RegTensor<int8_t> vreg0;
            AscendC::MicroAPI::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN_;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb);
            AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline auto BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::GetTensor()
{
    return cLocal_;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::operator()(const BlockShape &blockShape,
                                                                        const BlockCoord &blockCoord)
{
    singleM_ = AscendC::Te::Get<MNK_M>(blockShape);
    singleN_ = AscendC::Te::Get<MNK_N>(blockShape);
    scaleBlockN_ = CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(BLOCK_SIZE));
    blockCoord_ = blockCoord;
    auto halfSingleM = CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    vlForFloat32Number_ = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    UBBlockSize_ = AscendC::ONE_BLK_SIZE;
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;

    VFDoGeluForMX(singleMInVec);
    uint64_t yOffset = AscendC::Te::Get<Y_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * n_;
    uint64_t yScaleOffset = AscendC::Te::Get<Y_SCALE_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * scaleNAlign_;
    TransMxScaleLayout(singleMInVec);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset, quantOutput_);
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset, quantScaleBlockOutput_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    return;
}
} // namespace Block
} // namespace Epilogue
} // namespace Blaze
