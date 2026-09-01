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
    DYN_DTYPE_RANGE = 2,
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
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80; // 0b0111 1111 1000 0000
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00; // 0b0111 1111 0000 0000
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t BF16_ADD_VALUE_MAN1 = 0x003f; // dst_TypeMax=0.0或6.0时使用
constexpr uint16_t BF16_ADD_VALUE_MAN2 = 0x001f; // dst_TypeMax=7.0时使用
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
// elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // 0b 0000 0100 0000 0000 右移7位为8
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780; // 0b 0000 0111 1000 0000 右移7位为15
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100; // 0b 0000 0001 0000 0000 右移7位为2
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000; // 右移7位为0

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
constexpr float DIGIT_ZERO_FLOAT = 0.0;
constexpr float DIGIT_SIX_FLOAT = 6.0;
constexpr float DIGIT_SEVEN_FLOAT = 7.0;

template <typename DataTypeOut_, typename DataTypeIn_>
class BlockEpilogueGeluMxQuant {
public:
    __aicore__ inline BlockEpilogueGeluMxQuant() {}

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        GeluAlg geluAlg;
        QuantAlg quantAlg;
        ROUND_MODE_FP4 fp4RoundMode;
        float dtypeMax = 0.0;
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
    __aicore__ inline void Init(Params const& params);

    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);

private:
    __aicore__ inline static auto MakeNDExtLayout(int64_t rows, int64_t cols, int64_t rowPitch)
    {
        auto shape = AscendC::Te::MakeShape(AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows),
                                            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, cols));
        auto stride = AscendC::Te::MakeStride(AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, rowPitch),
                                              AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        return AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(
            shape, stride);
    }

    template <class T>
    __aicore__ inline static __ubuf__ T* GetUbAddr(uint64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + byteOffset);
    }

    __aicore__ inline void SetupUbLayout();

    __aicore__ inline void VFDoGeluForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize);
    __aicore__ inline void TransFp4MxOutLayout(uint16_t mSize);
    __aicore__ inline void VFDoGeluAndQuantForMX(__ubuf__ int8_t* outputDst, __ubuf__ uint16_t* scaleDst,
                                                 uint16_t mSize, uint16_t nSize);
    __aicore__ inline void GeluTanh(__ubuf__ bfloat16_t* geluResAddr, uint16_t mSize, uint16_t nSize,
                                    uint32_t nAligned);
    __aicore__ inline void GeluErf(__ubuf__ bfloat16_t* geluResAddr, uint16_t mSize, uint16_t nSize, uint32_t nAligned);
    __aicore__ inline void ComputeScaleOCP(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                           __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                           uint16_t loopNumScale);
    template <typename DstTypeMaxType>
    __aicore__ inline void ComputeScalecuBLAS(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                              __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                              uint16_t loopNumScale);
    __aicore__ inline void ComputeScaleDynDtypeRange(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                                     __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                                     uint16_t loopNumScale);
    __aicore__ inline void ComputeMaxExpOCP(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr,
                                            uint16_t loopNum);
    __aicore__ inline void ComputeMaxExpcuBLASOrDynDtypeRange(__ubuf__ bfloat16_t* srcAddr,
                                                              __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum);
    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t* srcAddr,
                                                        __ubuf__ uint16_t* halfScaleLocalAddr,
                                                        __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);
    template <AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t* srcAddr,
                                                        __ubuf__ uint16_t* halfScaleLocalAddr,
                                                        __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);
    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, int64_t gmOffset);

    // ---- Params ----
    const Params* params_{nullptr};

    // ---- GM base pointers (set via UpdateGlobalAddr) ----
    __gm__ int8_t* quantOutputGmAddr_{nullptr};
    __gm__ int8_t* quantScaleGmAddr_{nullptr};

    // ---- UB byte offsets (set in SetupUbLayout) ----
    uint64_t quantOutputUbOffset_{0};
    uint64_t quantScaleOutputUbOffset_{0};
    uint64_t quantScaleBlockOutputUbOffset_{0};
    uint64_t geluResUbOffset_{0};
    uint64_t maxExpUbOffset_{0};
    uint64_t halfScaleUbOffset_{0};
    uint64_t erfTmpUbOffset_{0};
    uint64_t fp32TmpUbOffset_{0};
    uint64_t geluFp32TmpUbOffset_{0};

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
    float dstTypeMax_ = 0.0;
    uint32_t addValueBit_ = 0; // 场景2，进位附加值

    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::Init(Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    // 量化结果的Nan值会转变成极大值
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    params_ = &params;
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr);
    quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
        dtypeMax_ = FP8_E4M3_MAX;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        dtypeMax_ = FP8_E5M2_MAX;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
        dstTypeMax_ = params_->dtypeMax;
        if (params_->dtypeMax == 0.0f || params_->dtypeMax == 6.0f) {
            addValueBit_ = BF16_ADD_VALUE_MAN1;
        } else if (params_->dtypeMax == 7.0f) {
            addValueBit_ = BF16_ADD_VALUE_MAN2;
        }
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
        dstTypeMax_ = params_->dtypeMax;
    }
    SetupUbLayout();
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::SetupUbLayout()
{
    constexpr uint32_t afterIn = MAX_SINGLE_MN * sizeof(DataTypeIn);
    quantOutputUbOffset_ = afterIn;
    constexpr uint32_t afterOut = afterIn + MAX_SINGLE_MN * sizeof(int8_t);
    quantScaleOutputUbOffset_ = afterOut;
    constexpr uint32_t afterIO = afterOut + MAX_SINGLE_SCALE_NUM * sizeof(int8_t);
    geluResUbOffset_ = afterIO;
    constexpr uint32_t afterIOAndGelu = afterIO + MAX_SINGLE_MN * sizeof(bfloat16_t);
    maxExpUbOffset_ = afterIOAndGelu;
    constexpr uint32_t afterIOAndGeluExp = afterIOAndGelu + MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    halfScaleUbOffset_ = afterIOAndGeluExp;
    constexpr uint32_t realScaleBlockOffset = afterIOAndGeluExp + MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    quantScaleBlockOutputUbOffset_ = realScaleBlockOffset;
    if (params_->geluAlg == GeluAlg::ERF) {
        uint32_t ubOffset = realScaleBlockOffset +
                            params_->baseM / AscendC::GetTaskRation() * AscendC::ONE_BLK_SIZE * sizeof(int8_t);
        if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
            erfTmpUbOffset_ = ubOffset;
            geluFp32TmpUbOffset_ = ubOffset + params_->baseN * sizeof(float);
        } else {
            fp32TmpUbOffset_ = ubOffset;
            erfTmpUbOffset_ = ubOffset + params_->baseN * sizeof(float);
            geluFp32TmpUbOffset_ = ubOffset + params_->baseN * sizeof(float) * 2;
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr) + AscendC::Te::Get<Y_IDX>(baseOffset);
        quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) +
                            AscendC::Te::Get<Y_SCALE_IDX>(baseOffset);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = AscendC::Te::Get<Gemm::MNK_N>(problemShape);
    scaleN_ = Gemm::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(BLOCK_SIZE));
    scaleNAlign_ = Gemm::CeilAlign(scaleN_, MX_SCALE_ALIGN_SIZE);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyOutputFromUb2Gm(uint64_t blockCount,
                                                                                                int64_t gmOffset)
{
    int64_t nValid = static_cast<int64_t>(singleN_);
    int64_t gmRowPitch = n_;

    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        nValid = nValid >> 1;
        gmRowPitch = gmRowPitch >> 1;
        gmOffset = gmOffset >> 1;
    }
    int64_t nUbAligned = static_cast<int64_t>(Gemm::Align32(static_cast<uint64_t>(nValid)));

    auto ubLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, nUbAligned);
    auto gmLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, gmRowPitch);
    auto outUb = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(quantOutputUbOffset_), ubLayout);
    if (singleN_ < OUT_ELE_NUM_ONE_BLK) {
        outUb = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(geluResUbOffset_),
                                        ubLayout);
    }
    auto outGm = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(quantOutputGmAddr_ + gmOffset), gmLayout);

    auto copyUB2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
    AscendC::Te::Copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyScaleFromUb2Gm(uint64_t blockCount,
                                                                                               int64_t gmOffset)
{
    int64_t nValid = static_cast<int64_t>(scaleBlockN_);
    int64_t nUbAligned = static_cast<int64_t>(AscendC::ONE_BLK_SIZE);
    int64_t gmRowPitch = scaleNAlign_;

    auto ubLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, nUbAligned);
    auto gmLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, gmRowPitch);
    auto outUb = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(quantScaleBlockOutputUbOffset_), ubLayout);
    auto outGm = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(quantScaleGmAddr_ + gmOffset), gmLayout);

    auto copyUB2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
    AscendC::Te::Copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeMaxExpOCP(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract0;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract1;

        AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
        AscendC::Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::MaskReg Mask = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                            vlForHalfNumber_ * 2);
            AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, Mask);
            AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, Mask);
            AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            AscendC::Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeMaxExpcuBLASOrDynDtypeRange(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1;
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;

        AscendC::Reg::RegTensor<uint16_t> absMask16Bit;
        AscendC::Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        AscendC::Reg::MaskReg Mask = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                            vlForHalfNumber_ * 2);
            AscendC::Reg::And((AscendC::Reg::RegTensor<uint16_t>&)vdExp0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0,
                              absMask16Bit, Mask);
            AscendC::Reg::And((AscendC::Reg::RegTensor<uint16_t>&)vdExp1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1,
                              absMask16Bit, Mask);
            AscendC::Reg::Max(vdMaxExp, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0,
                              (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, Mask);
            AscendC::Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
template <typename DstTypeMaxType>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeScalecuBLAS(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    using T = bfloat16_t;
    DstTypeMaxType dtypeMax;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        dtypeMax = dtypeMax_;
    } else {
        dtypeMax = dstTypeMax_;
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::RegTensor<uint32_t> vdMaxExp32;
        AscendC::Reg::RegTensor<uint32_t> exp32;
        AscendC::Reg::RegTensor<uint32_t> man32;
        AscendC::Reg::RegTensor<uint32_t> normalExp32;
        AscendC::Reg::RegTensor<uint32_t> expAddOne32;
        AscendC::Reg::RegTensor<uint32_t> extractExp;
        AscendC::Reg::RegTensor<uint16_t> expOut;
        AscendC::Reg::RegTensor<uint32_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> recExpOut;

        AscendC::Reg::RegTensor<DstTypeMaxType> invMax;
        AscendC::Reg::Duplicate(invMax, dtypeMax);
        AscendC::Reg::RegTensor<uint32_t> manMaskFP32;
        AscendC::Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        AscendC::Reg::RegTensor<uint32_t> expMask;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::Reg::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::Reg::Duplicate(zeroRegTensor32, 0);
        AscendC::Reg::RegTensor<uint32_t> scaleBias;
        AscendC::Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::Reg::RegTensor<uint32_t> nanRegTensor;
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        AscendC::Reg::RegTensor<uint32_t> fp8NanRegTensor;
        AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);

        AscendC::Reg::MaskReg cmpResult;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg p0;
        AscendC::Reg::MaskReg p1;
        AscendC::Reg::MaskReg p2;
        uint32_t SixtyFour = 64;
        AscendC::Reg::MaskReg dataMaskB16Half = AscendC::Reg::UpdateMask<uint16_t>(SixtyFour);
        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<uint32_t>();

        static constexpr AscendC::Reg::CastTrait castTraitHalf2Float = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumScale; i++) {
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vdMaxExp, maxExpAddr, vlForFloat32Number_);

            AscendC::Reg::Cast<float, T, castTraitHalf2Float>((AscendC::Reg::RegTensor<float>&)vdMaxExp32,
                                                              (AscendC::Reg::RegTensor<T>&)vdMaxExp, mask);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(cmpResult, vdMaxExp32, expMask, mask);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp32, zeroRegTensor32, mask);

            AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)vdMaxExp32, (AscendC::Reg::RegTensor<float>&)vdMaxExp32,
                              (AscendC::Reg::RegTensor<float>&)invMax, mask);
            AscendC::Reg::ShiftRights(exp32, vdMaxExp32, SHR_NUM_FOR_FP32, mask);
            AscendC::Reg::And(man32, vdMaxExp32, manMaskFP32, mask);

            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p0, exp32, NUMBER_ZERO, mask);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR, mask);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_ZERO, mask);
            AscendC::Reg::MaskAnd(p0, p0, p1, mask);
            AscendC::Reg::MaskAnd(p0, p0, p2, mask);

            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(p1, exp32, NUMBER_ZERO, mask);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_HALF, mask);
            AscendC::Reg::MaskAnd(p1, p1, p2, mask);
            AscendC::Reg::MaskOr(p0, p0, p1, mask);

            AscendC::Reg::Adds(expAddOne32, exp32, 1, mask);
            AscendC::Reg::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(expOut, extractExp);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr + i * 32,
                                                                                       expOut, dataMaskB16Half);

            AscendC::Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, mask);
            AscendC::Reg::Sub(halfScale, scaleBias, extractExp, mask);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(recExpOut, halfScale);
            AscendC::Reg::StoreAlign<uint16_t>(halfScaleLocalAddr + i * vlForFloat32Number_, recExpOut,
                                               dataMaskB16Half);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeScaleOCP(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, maskScale;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::Reg::Duplicate(maxExpValue, fpEmax_);
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
        AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            maskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                          vlForHalfNumber_);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                  maskScale); // INF\nAN
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, maskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, maskScale);
            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask); // 大于emax取emax
            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, maskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, maskScale);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                           vlForHalfNumber_ >> 1, maskScale);

            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, maskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, maskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, halfScale,
                                                                                          vlForHalfNumber_, maskScale);
        }
    }
    return;
}

/**
 * @brief 计算动态数据类型范围(DynDtypeRange)下的MX量化缩放因子
 *
 * 该函数基于ComputeMaxExpcuBLASOrDynDtypeRange计算的最大指数值，生成MX量化格式所需的缩放因子。
 * 主要用于FP4(E2M1/E1M2)格式的动态范围量化场景。
 *
 * 算法流程:
 * 1. 加载最大指数值到寄存器
 * 2. 提取指数位并检查INF/NaN特殊情况
 * 3. 添加进位附加值(addValueBit_)处理进位情况
 * 4. 计算共享指数: sharedExp = xMaxExpAdd - maxExpValue
 * 5. 生成MX缩放因子: scaleValue = sharedExp >> 7 (右移7位转换为BF16 E8M7格式)
 * 6. 生成半缩放因子: halfScale = BF16_EXP_BIAS - sharedExp
 * 7. 处理特殊情况: INF/NaN返回特殊值，零值返回0
 *
 * @param maxExpAddr 输入最大指数值的UB地址
 * @param mxScaleLocalAddr 输出MX缩放因子的UB地址，存储BF16 E8M7格式的缩放值
 * @param halfScaleLocalAddr 输出半缩放因子的UB地址，用于后续数据量化计算
 * @param totalScaleInUB 当前UB中的总缩放因子数量
 * @param loopNumScale 循环次数，根据vlForHalfNumber_计算
 *
 * 特殊值处理:
 * - INF/NaN: scaleValue返回0x00FF (MAX_EXP_FOR_FP8)
 * - 零值: scaleValue返回0
 * - 特殊指数(==BF16_EXP_BIAS): halfScale返回0x0040 (SPECIAL_EXP_THRESHOLD)
 */
template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeScaleDynDtypeRange(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> xMaxExp;
        AscendC::Reg::RegTensor<uint16_t> sharedExp;
        AscendC::Reg::RegTensor<uint16_t> scaleValue;
        AscendC::Reg::RegTensor<uint16_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> xMaxExpAdd;
        AscendC::Reg::RegTensor<uint16_t> xMaxExpOnly;

        AscendC::Reg::RegTensor<uint16_t> expMask;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> addValue;
        AscendC::Reg::Duplicate(addValue, addValueBit_);
        AscendC::Reg::RegTensor<uint16_t> maxExpValue;
        AscendC::Reg::Duplicate(maxExpValue, FP4_E2M1_MAX_EXP);
        AscendC::Reg::RegTensor<uint16_t> scaleBias;
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::RegTensor<uint16_t> fp8NanU16;
        AscendC::Reg::Duplicate(fp8NanU16, MAX_EXP_FOR_FP8);
        AscendC::Reg::RegTensor<uint16_t> zeroU16;
        AscendC::Reg::Duplicate(zeroU16, 0);
        AscendC::Reg::RegTensor<uint16_t> nanU16;
        AscendC::Reg::Duplicate(nanU16, NAN_CUSTOMIZATION);
        AscendC::Reg::RegTensor<uint16_t> specialExpU16;
        AscendC::Reg::Duplicate(specialExpU16, SPECIAL_EXP_THRESHOLD);

        AscendC::Reg::MaskReg cmpResult;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg invalidDataMask;
        AscendC::Reg::MaskReg specialDataMask;
        AscendC::Reg::MaskReg preMaskScale;

        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr,
                                                                                          vlForHalfNumber_);

            AscendC::Reg::And(xMaxExpOnly, xMaxExp, expMask, preMaskScale); // 提取指数位
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, xMaxExpOnly, expMask,
                                                                  preMaskScale); // INF/NAN
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LT>(invalidDataMask, xMaxExpOnly, maxExpValue,
                                                                  preMaskScale);

            AscendC::Reg::Add(xMaxExpAdd, xMaxExp, addValue, preMaskScale);   // 进位后的结果
            AscendC::Reg::And(xMaxExpAdd, xMaxExpAdd, expMask, preMaskScale); // 提取进位结果的指数位
            AscendC::Reg::Select<uint16_t>(xMaxExpAdd, maxExpValue, xMaxExpAdd, invalidDataMask);
            AscendC::Reg::Sub(sharedExp, xMaxExpAdd, maxExpValue, preMaskScale);

            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                           vlForHalfNumber_ >> 1, preMaskScale);

            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, sharedExp, zeroU16, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanU16, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroU16, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpU16, halfScale, specialDataMask);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    uint32_t totalCountInUB2 = totalCountInUB * 2;
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<float> floatScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0, vdExp1, vdExp0Convert, vdExp1Convert;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        AscendC::Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::Reg::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

        static constexpr AscendC::Reg::CastTrait castTraitZero = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::Reg::CastTrait castTraitOne = {
            AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::Reg::CastTrait castTrait32to8 = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            dataMask3 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);
            dataMask4 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);
            AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                         elementAfterReduce_);

            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
            AscendC::Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
            AscendC::Reg::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
            AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
            AscendC::Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
            AscendC::Reg::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
            AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
template <AscendC::RoundMode roundMode>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg dataMask1;
        AscendC::Reg::MaskReg dataMask2;
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0;
        AscendC::Reg::RegTensor<T> vdExp1;
        AscendC::Reg::RegTensor<T> vdExp0Convert;
        AscendC::Reg::RegTensor<T> vdExp1Convert;

        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1BF16;

        AscendC::Reg::RegTensor<U> vdExp0FP4;
        AscendC::Reg::RegTensor<U> vdExp1FP4;

        static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO,
                                                              AscendC::Reg::SatMode::UNKNOWN,
                                                              AscendC::Reg::MaskMergeMode::ZEROING, roundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                            vlForHalfNumber_ * 2);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                         elementAfterReduce_);

            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask2);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::GeluTanh(__ubuf__ bfloat16_t* geluResAddr,
                                                                                     uint16_t mSize, uint16_t nSize,
                                                                                     uint32_t nAligned)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(float); // 需要转换成float32计算
    uint16_t OneRowRepeatTimes = Gemm::CeilDiv(nSize, sizePerRepeat);             // 计算为64位对齐

    __ubuf__ DataTypeIn* src = GetUbAddr<DataTypeIn>(0);
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInput;
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInputSqr;
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInputCub;
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregOutput;
    AscendC::Reg::RegTensor<bfloat16_t, AscendC::Reg::RegTraitNumOne> vregOutput16; // gelu总是输出bfloat16
    static constexpr AscendC::Reg::CastTrait ctHalf2Fp32Zero = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait ctFp32toBf16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    AscendC::Reg::MaskReg mask;
    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        __VEC_SCOPE__
        {
            // 每行计算一次
            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
                uint32_t count = nSize;
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                    uint32_t offset = mIdx * nAligned + vfBlockIdx * sizePerRepeat;
                    AscendC::Reg::DataCopy(vregInput, src + offset);
                    AscendC::Reg::Mul(vregInputSqr, vregInput, vregInput, mask);
                    AscendC::Reg::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                    AscendC::Reg::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                    AscendC::Reg::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                    AscendC::Reg::Exp(vregInputCub, vregInputCub, mask);
                    AscendC::Reg::Adds(vregInputCub, vregInputCub, 1.0f, mask);
                    AscendC::Reg::Div(vregOutput, vregInput, vregInputCub, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(geluResAddr + offset,
                                                                                               vregOutput16, mask);
                }
            }
        }
    } else {
        AscendC::Reg::RegTensor<DataTypeIn, AscendC::Reg::RegTraitNumOne> vregInput16;
        __VEC_SCOPE__
        {
            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) { // 需要计算m次
                uint32_t count = nSize;
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                    uint32_t offset = mIdx * nAligned + vfBlockIdx * sizePerRepeat;
                    AscendC::Reg::DataCopy<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                                                src + offset);
                    AscendC::Reg::Cast<float, DataTypeIn, ctHalf2Fp32Zero>(vregInput, vregInput16, mask);
                    AscendC::Reg::Mul(vregInputSqr, vregInput, vregInput, mask);
                    AscendC::Reg::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                    AscendC::Reg::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                    AscendC::Reg::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                    AscendC::Reg::Exp(vregInputCub, vregInputCub, mask);
                    AscendC::Reg::Adds(vregInputCub, vregInputCub, 1.0f, mask);
                    AscendC::Reg::Div(vregOutput, vregInput, vregInputCub, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(geluResAddr + offset,
                                                                                               vregOutput16, mask);
                }
            }
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::GeluErf(__ubuf__ bfloat16_t* geluResAddr,
                                                                                    uint16_t mSize, uint16_t nSize,
                                                                                    uint32_t nAligned)
{
    // 0.5*x*(1+erf(x/√2)
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t OneRowRepeatTimes = Gemm::CeilDiv(nSize, sizePerRepeat); // 计算为64位对齐

    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInput1;
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInput2;
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInputAdds;
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInputMuls;
    AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregOutput;
    AscendC::Reg::RegTensor<bfloat16_t, AscendC::Reg::RegTraitNumOne> vregOutput16; // gelu总是输出bfloat16
    AscendC::Reg::MaskReg mask;
    static constexpr AscendC::Reg::CastTrait ctFp32toBf16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    static constexpr AscendC::ErfConfig erfConfig = {AscendC::ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION};

    __ubuf__ DataTypeIn* srcInput = GetUbAddr<DataTypeIn>(0);
    __ubuf__ float* erfAddr = GetUbAddr<float>(erfTmpUbOffset_);
    __ubuf__ float* fp32Addr = GetUbAddr<float>(fp32TmpUbOffset_);

    AscendC::LocalTensor<DataTypeIn> srcLocal{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<float> erfLocal{AscendC::TPosition::VECCALC, static_cast<uint32_t>(erfTmpUbOffset_),
                                         params_->baseN};
    AscendC::LocalTensor<float> geluFp32Local{AscendC::TPosition::VECCALC, static_cast<uint32_t>(geluFp32TmpUbOffset_),
                                              params_->baseN};
    AscendC::LocalTensor<float> fp32Local{AscendC::TPosition::VECCALC, static_cast<uint32_t>(fp32TmpUbOffset_),
                                          params_->baseN};

    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        __ubuf__ float* src = (__ubuf__ float*)srcInput;
        for (uint32_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Muls(geluFp32Local, srcLocal[mIdx * nAligned], ONE_OVER_SQRT_TWO, nSize);
            AscendC::Erf<float, false, erfConfig>(erfLocal, geluFp32Local, nSize);
            uint32_t count = nSize;
            __VEC_SCOPE__
            {
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                    uint32_t mnOffset = mIdx * nAligned + vfBlockIdx * sizePerRepeat;
                    AscendC::Reg::DataCopy(vregInput1, (__ubuf__ float*)(erfAddr + vfBlockIdx * sizePerRepeat));
                    AscendC::Reg::DataCopy(vregInput2, (__ubuf__ float*)(src + mnOffset));
                    AscendC::Reg::Adds(vregInputAdds, vregInput1, (float)1.0, mask);
                    AscendC::Reg::Muls(vregInputMuls, vregInput2, (float)0.5, mask);
                    AscendC::Reg::Mul(vregOutput, vregInputAdds, vregInputMuls, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(geluResAddr + mnOffset,
                                                                                               vregOutput16, mask);
                }
            }
        }
    } else {
        for (uint32_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Cast(fp32Local, srcLocal[mIdx * nAligned], AscendC::RoundMode::CAST_NONE, nSize);
            AscendC::Muls(geluFp32Local, fp32Local, ONE_OVER_SQRT_TWO, nSize);
            AscendC::Erf<float, false, erfConfig>(erfLocal, geluFp32Local, nSize);
            uint32_t count = nSize;
            __VEC_SCOPE__
            {
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                    uint32_t nOffset = vfBlockIdx * sizePerRepeat;
                    uint32_t mnOffset = mIdx * nAligned + nOffset;
                    AscendC::Reg::DataCopy(vregInput1, (__ubuf__ float*)(erfAddr + nOffset));
                    AscendC::Reg::DataCopy(vregInput2, (__ubuf__ float*)(fp32Addr + nOffset));
                    AscendC::Reg::Adds(vregInputAdds, vregInput1, (float)1.0, mask);
                    AscendC::Reg::Muls(vregInputMuls, vregInput2, (float)0.5, mask);
                    AscendC::Reg::Mul(vregOutput, vregInputAdds, vregInputMuls, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, ctFp32toBf16>(vregOutput16, vregOutput, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(geluResAddr + mnOffset,
                                                                                               vregOutput16, mask);
                }
            }
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::VFDoGeluAndQuantForMX(
    __ubuf__ int8_t* outputDst, __ubuf__ uint16_t* scaleDst, uint16_t mSize, uint16_t nSize)
{
    uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize)); // 输入为32位对齐
    __ubuf__ bfloat16_t* geluResAddr = GetUbAddr<bfloat16_t>(geluResUbOffset_);
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<bfloat16_t> zeroReg;
            AscendC::Reg::Duplicate(zeroReg, static_cast<bfloat16_t>(0.0));
            constexpr uint32_t bf16Vl = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
            uint32_t remainingElements = mSize * nAligned;
            uint32_t zeroOffset = 0;
            while (remainingElements > 0) {
                AscendC::Reg::MaskReg zeroMask = AscendC::Reg::UpdateMask<bfloat16_t>(remainingElements);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(geluResAddr + zeroOffset,
                                                                                           zeroReg, zeroMask);
                zeroOffset += bf16Vl;
            }
        }
    }
    if (params_->geluAlg == GeluAlg::ERF) {
        GeluErf(geluResAddr, mSize, nSize, nAligned);
    } else {
        GeluTanh(geluResAddr, mSize, nSize, nAligned);
    }

    uint32_t totalDataInUb = mSize * nAligned;
    uint32_t totalScaleInUb = totalDataInUb / BLOCK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
    __ubuf__ uint16_t* halfScaleLocalAddr;
    if (params_->quantAlg == QuantAlg::OCP) {
        uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
        __ubuf__ uint16_t* maxExpAddr = GetUbAddr<uint16_t>(maxExpUbOffset_);
        ComputeMaxExpOCP(geluResAddr, maxExpAddr, loopDataNum);
        halfScaleLocalAddr = GetUbAddr<uint16_t>(halfScaleUbOffset_);
        ComputeScaleOCP(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    } else if (params_->quantAlg == QuantAlg::BLAS) {
        uint16_t loopScaleNum = (totalScaleInUb + vlForFloat32Number_ - 1) / vlForFloat32Number_;
        __ubuf__ uint16_t* maxExpAddr = GetUbAddr<uint16_t>(maxExpUbOffset_);
        ComputeMaxExpcuBLASOrDynDtypeRange(geluResAddr, maxExpAddr, loopDataNum);
        halfScaleLocalAddr = GetUbAddr<uint16_t>(halfScaleUbOffset_);
        ComputeScalecuBLAS<uint32_t>(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    } else {
        if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT || dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
            uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
            __ubuf__ uint16_t* maxExpAddr = GetUbAddr<uint16_t>(maxExpUbOffset_);
            ComputeMaxExpcuBLASOrDynDtypeRange(geluResAddr, maxExpAddr, loopDataNum);
            halfScaleLocalAddr = GetUbAddr<uint16_t>(halfScaleUbOffset_);
            ComputeScaleDynDtypeRange(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
        } else {
            uint16_t loopScaleNum = (totalScaleInUb + vlForFloat32Number_ - 1) / vlForFloat32Number_;
            __ubuf__ uint16_t* maxExpAddr = GetUbAddr<uint16_t>(maxExpUbOffset_);
            ComputeMaxExpcuBLASOrDynDtypeRange(geluResAddr, maxExpAddr, loopDataNum);
            halfScaleLocalAddr = GetUbAddr<uint16_t>(halfScaleUbOffset_);
            ComputeScalecuBLAS<float>(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
        }
    }

    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(geluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        if (params_->fp4RoundMode == ROUND_MODE_FP4::FLOOR) {
            ComputeDataForQuantTargetFp4<AscendC::RoundMode::CAST_FLOOR>(geluResAddr, halfScaleLocalAddr, outputDst,
                                                                         totalDataInUb, loopDataNum);
        } else if ((params_->fp4RoundMode == ROUND_MODE_FP4::ROUND)) {
            ComputeDataForQuantTargetFp4<AscendC::RoundMode::CAST_ROUND>(geluResAddr, halfScaleLocalAddr, outputDst,
                                                                         totalDataInUb, loopDataNum);
        } else { // 默认rint
            ComputeDataForQuantTargetFp4<AscendC::RoundMode::CAST_RINT>(geluResAddr, halfScaleLocalAddr, outputDst,
                                                                        totalDataInUb, loopDataNum);
        }
    }
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::VFDoGeluForMX(uint16_t mSize)
{
    __ubuf__ int8_t* quantOutputInUbAddr = GetUbAddr<int8_t>(quantOutputUbOffset_);
    __ubuf__ uint16_t* quantScaleOutputInUbAddr = GetUbAddr<uint16_t>(quantScaleOutputUbOffset_);
    VFDoGeluAndQuantForMX(quantOutputInUbAddr, quantScaleOutputInUbAddr, mSize, singleN_);
}

/**
 * @brief 转换FP4 MX量化输出的数据布局
 *
 * 该函数将FP4量化输出从线性布局(mSize*16)转换为块布局(mSize,32)，
 * 以满足MX量化格式对数据存储的要求。
 *
 * 布局转换说明:
 * - 输入: 线性存储，每行数据连续排列，每行占singleN_/2字节(FP4是2个元素打包为1字节)
 * - 输出: 块对齐存储，每行对齐到32字节(AscendC::ONE_BLK_SIZE)，便于DMA传输
 *
 * 算法流程:
 * 1. 计算尾部行间距(tailLineStride): 当singleN_ < 64时需要填充对齐
 * 2. 对每一行数据:
 *    a. 计算有效元素数量: elemNum = singleN_ / 2 (FP4打包后的字节数)
 *    b. 从源地址加载数据到寄存器(支持非对齐加载)
 *    c. 将数据存储到目标地址，使用DIST_NORM_B8模式进行字节级存储
 *
 * @param mSize 处理的行数
 *
 * 注意: 该函数仅在DataTypeOut为fp4x2_e2m1_t或fp4x2_e1m2_t时被调用
 */
template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::TransFp4MxOutLayout(uint16_t mSize)
{
    __ubuf__ int8_t* quantOutputInUbAddr = GetUbAddr<int8_t>(quantOutputUbOffset_);
    __ubuf__ int8_t* quantBlockOutputInUbAddr = GetUbAddr<int8_t>(geluResUbOffset_);
    // yOut layout: (mSize*16) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elemNum = singleN_ / 2;
            AscendC::Reg::MaskReg maskOutN = AscendC::Reg::UpdateMask<int8_t>(elemNum);
            AscendC::Reg::RegTensor<int8_t> vreg0;
            AscendC::Reg::UnalignReg u0, u1;
            auto srcUb = quantOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE / 2;
            AscendC::Reg::DataCopyUnAlignPre(u0, srcUb);
            AscendC::Reg::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskOutN);
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::TransMxScaleLayout(uint16_t mSize)
{
    __ubuf__ int8_t* quantScaleOutputInUbAddr = GetUbAddr<int8_t>(quantScaleOutputUbOffset_);
    __ubuf__ int8_t* quantScaleBlockOutputInUbAddr = GetUbAddr<int8_t>(quantScaleBlockOutputUbOffset_);
    // scale layout: (mSize*8) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elemNum = scaleBlockN_;
            AscendC::Reg::MaskReg maskScaleN = AscendC::Reg::UpdateMask<int8_t>(elemNum);
            AscendC::Reg::RegTensor<int8_t> vreg0;
            AscendC::Reg::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN_;
            AscendC::Reg::DataCopyUnAlignPre(u0, srcUb);
            AscendC::Reg::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::operator()(const BlockShape& blockShape,
                                                                                       const BlockCoord& blockCoord)
{
    singleM_ = AscendC::Te::Get<Gemm::MNK_M>(blockShape);
    singleN_ = AscendC::Te::Get<Gemm::MNK_N>(blockShape);
    scaleBlockN_ = Gemm::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(BLOCK_SIZE));
    blockCoord_ = blockCoord;
    auto halfSingleM = Gemm::CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
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
    int64_t yOffset = static_cast<int64_t>(AscendC::Te::Get<Y_IDX>(blockCoord)) +
                      static_cast<int64_t>(subBlockIdx_ * halfSingleM * n_);
    int64_t yScaleOffset = static_cast<int64_t>(AscendC::Te::Get<Y_SCALE_IDX>(blockCoord)) +
                           static_cast<int64_t>(subBlockIdx_ * halfSingleM * scaleNAlign_);
    if (singleN_ < OUT_ELE_NUM_ONE_BLK) {
        TransFp4MxOutLayout(singleMInVec);
    }
    TransMxScaleLayout(singleMInVec);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset);
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    return;
}
} // namespace Block
} // namespace Epilogue
} // namespace Blaze
