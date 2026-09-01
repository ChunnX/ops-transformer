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

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

namespace Constant {
constexpr uint32_t NUM_TWO = 2;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint64_t MX_QUANT_COMPUTE_ALIGN = 64UL;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 64 * 256;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
} // namespace Constant

#ifdef __CCE_AICORE__
constexpr AscendC::Reg::CastTrait CT_FP32_TO_BF16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait CT_BF16_TO_FP32_ZERO = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait CT_BF16_TO_FP32_ONE = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait CT_FP32_TO_FP8_SAT = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait CT_FP4_RINT = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                 AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::DivSpecificMode DIV_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};
#endif // __CCE_AICORE__

template <typename DataTypeOut_, typename DataTypeIn_ = float, typename DataTypeScale_ = fp8_e8m0_t>
class BlockEpilogueSwigluMxQuant {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeScale = DataTypeScale_;
    static constexpr uint64_t INPUT_UB_TILE_ELEMENTS = Constant::MAX_SINGLE_MN;
    static constexpr uint64_t INPUT_UB_BUFFER_BYTES = INPUT_UB_TILE_ELEMENTS * sizeof(DataTypeIn);
    static constexpr uint64_t OUTPUT_C0_SIZE = AscendC::Te::C0_ELEMENT<DataTypeIn>;
    static constexpr uint64_t SPLIT_M_ALIGN = Constant::NUM_TWO;

    enum class L0c2UbTensorType : uint8_t {
        SWISH_INPUT = 0,
        GATE_INPUT = 1,
    };

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;

    struct OutputOffsets {
        int64_t yOffset{0};
        int64_t yScaleOffset{0};
    };

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        Params() = default;
    };

    __aicore__ inline BlockEpilogueSwigluMxQuant()
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline ~BlockEpilogueSwigluMxQuant()
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline auto GetL0c2UbTensor(int64_t rows, int64_t cols, L0c2UbTensorType tensorType)
    {
        // The Split-M Fixpipe instruction requires an even mSize. The epilogue still receives the original
        // logical row count and ignores the padding row on sub-block 1.
        const uint64_t copyRows = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(rows), SPLIT_M_ALIGN);
        const auto
            layoutOutUb = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<OUTPUT_C0_SIZE>>(
                copyRows, static_cast<uint64_t>(cols));
        const uint64_t ubOffset = static_cast<uint64_t>(tensorType) * INPUT_UB_BUFFER_BYTES;
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, DataTypeIn>(ubOffset),
                                       layoutOutUb);
    }

    __aicore__ inline auto GetConcatL0c2UbTensor(int64_t rows, int64_t cols)
    {
        const uint64_t copyRows = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(rows), SPLIT_M_ALIGN);
        const auto
            layoutOutUb = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<OUTPUT_C0_SIZE>>(
                copyRows, static_cast<uint64_t>(cols) * Constant::NUM_TWO);
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, DataTypeIn>(0), layoutOutUb);
    }

    __aicore__ inline void Init(Params const& params);
    __aicore__ inline void operator()(const BlockShape& blockShape, const OutputOffsets& outputOffsets);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);
    __aicore__ inline void UpdateGlobalAddr(const OutputOffsets& baseOffsets);

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

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    __aicore__ inline void ComputeSwiglu(uint16_t mSize);
    __aicore__ inline void ComputeMxQuant(uint16_t mSize);

    __aicore__ inline void SetupUbLayout();

    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                        __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t* srcAddr,
                                                        __ubuf__ uint16_t* halfScaleLocalAddr,
                                                        __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    // ---- Params ----
    const Params* params_{nullptr};

    // ---- GM base pointers (set via UpdateGlobalAddr) ----
    __gm__ int8_t* quantOutputGmAddr_{nullptr};
    __gm__ int8_t* quantScaleGmAddr_{nullptr};

    // ---- UB byte offsets (set in SetupUbLayout) ----
    uint64_t quantOutputUbOffset_{0};
    uint64_t quantScaleOutputUbOffset_{0};
    uint64_t quantScaleBlockOutputUbOffset_{0};
    uint64_t gluResUbOffset_{0};
    uint64_t maxExpUbOffset_{0};
    uint64_t halfScaleUbOffset_{0};

    // ---- Dimensions ----
    int64_t n_{0};
    int64_t scaleN_{0};
    int64_t scaleBlockN_{0};
    uint32_t subBlockIdx_{0};
    uint32_t singleM_{0};
    uint32_t singleN_{0};

    uint32_t vlForHalfNumber_{0};
    uint16_t elementAfterReduce_{0};
    uint16_t fpEmax_{0};
};

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::Init(Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    subBlockIdx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());

    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = Constant::FP8_E4M3_MAX_EXP;
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = Constant::FP8_E5M2_MAX_EXP;
    }
    SetupUbLayout();
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::SetupUbLayout()
{
    constexpr uint32_t afterIn = Constant::NUM_TWO * Constant::MAX_SINGLE_MN * sizeof(DataTypeIn);
    quantOutputUbOffset_ = afterIn;
    quantScaleOutputUbOffset_ = afterIn + Constant::MAX_SINGLE_MN * sizeof(int8_t);
    constexpr uint32_t afterIO = afterIn + Constant::MAX_SINGLE_MN * sizeof(int8_t) +
                                 Constant::MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    gluResUbOffset_ = afterIO;
    constexpr uint32_t afterIOAndGlu = afterIO + Constant::MAX_SINGLE_MN * sizeof(bfloat16_t);
    maxExpUbOffset_ = afterIOAndGlu;
    constexpr uint32_t afterIOAndGluExp = afterIOAndGlu +
                                          Constant::MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScaleUbOffset_ = afterIOAndGluExp;
    quantScaleBlockOutputUbOffset_ = afterIOAndGluExp +
                                     Constant::MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::UpdateGlobalAddr(
    const OutputOffsets& baseOffsets)
{
    if ASCEND_IS_AIV {
        quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr) + baseOffsets.yOffset;
        quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) + baseOffsets.yScaleOffset;
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape);
    scaleN_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(n_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
              Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::operator()(
    const BlockShape& blockShape, const OutputOffsets& outputOffsets)
{
    if ASCEND_IS_AIC {
        return;
    }
    singleM_ = static_cast<uint32_t>(AscendC::Te::Get<Blaze::Gemm::MNK_M>(blockShape));
    singleN_ = static_cast<uint32_t>(AscendC::Te::Get<Blaze::Gemm::MNK_N>(blockShape));
    scaleBlockN_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
                   Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

    auto halfSingleM = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleM_),
                                            static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = (subBlockIdx_ == 1) ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / Constant::BLOCK_SIZE;

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);

    ComputeSwiglu(static_cast<uint16_t>(singleMInVec));
    ComputeMxQuant(static_cast<uint16_t>(singleMInVec));

    int64_t yOffset = outputOffsets.yOffset + static_cast<int64_t>(mOffset) * n_;
    int64_t yScaleOffset = outputOffsets.yScaleOffset + static_cast<int64_t>(mOffset) * scaleN_;

    TransMxScaleLayout(static_cast<uint16_t>(singleMInVec), static_cast<uint16_t>(scaleBlockN_));

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(static_cast<uint64_t>(singleMInVec), yOffset);
    CopyScaleFromUb2Gm(static_cast<uint64_t>(singleMInVec), yScaleOffset);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::CopyOutputFromUb2Gm(
    uint64_t blockCount, int64_t gmOffset)
{
    int64_t nValid;
    int64_t nUbAligned;
    int64_t gmRowPitch;

    nValid = static_cast<int64_t>(singleN_);
    nUbAligned = static_cast<int64_t>(Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_)));
    gmRowPitch = n_;

    auto ubLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, nUbAligned);
    auto gmLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, gmRowPitch);
    auto outUb = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(quantOutputUbOffset_), ubLayout);
    auto outGm = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(quantOutputGmAddr_ + gmOffset), gmLayout);

    auto copyUB2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
    AscendC::Te::Copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::CopyScaleFromUb2Gm(
    uint64_t blockCount, int64_t gmOffset)
{
    int64_t blockScaleN = static_cast<int64_t>(
        Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
        Blaze::Gemm::MXFP_MULTI_BASE_SIZE);

    auto ubLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), blockScaleN,
                                    static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
    auto gmLayout = MakeNDExtLayout(static_cast<int64_t>(blockCount), blockScaleN, scaleN_);
    auto outUb = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(quantScaleBlockOutputUbOffset_), ubLayout);
    auto outGm = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(quantScaleGmAddr_ + gmOffset), gmLayout);

    auto copyUB2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
    AscendC::Te::Copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::TransMxScaleLayout(
    uint16_t mSize, uint16_t scaleBlockN)
{
    __ubuf__ int8_t* quantScaleOutputInUbAddr = GetUbAddr<int8_t>(quantScaleOutputUbOffset_);
    __ubuf__ int8_t* quantScaleBlockOutputInUbAddr = GetUbAddr<int8_t>(quantScaleBlockOutputUbOffset_);

    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elemNum = scaleBlockN;
            AscendC::Reg::MaskReg maskScaleN = AscendC::Reg::UpdateMask<int8_t>(elemNum);
            AscendC::Reg::RegTensor<int8_t> vreg0;
            AscendC::Reg::UnalignReg u0;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
            AscendC::Reg::DataCopyUnAlignPre(u0, srcUb);
            AscendC::Reg::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::ComputeSwiglu(
    uint16_t mSize)
{
    __ubuf__ DataTypeIn* swishInputUbAddr = GetUbAddr<DataTypeIn>(0);
    __ubuf__ bfloat16_t* gluResAddr = GetUbAddr<bfloat16_t>(gluResUbOffset_);

    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    const uint16_t oneRowRepeatTimes = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_),
                                                            static_cast<uint64_t>(sizePerRepeat));
    const uint32_t nSrcUbAligned = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(singleN_),
                                                          AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    const uint32_t concatNSrcUbAligned = nSrcUbAligned * Constant::NUM_TWO;
    const uint32_t nDstUbAligned64 = Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_));

    const float scalarOne = 1.0f;

    // Zero-initialize gluRes when N tail (non-64-aligned)
    if (__builtin_expect((singleN_ % Constant::MX_QUANT_COMPUTE_ALIGN) != 0, 0)) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<bfloat16_t> zeroReg;
            AscendC::Reg::Duplicate(zeroReg, static_cast<bfloat16_t>(0));
            constexpr uint32_t bf16Vl = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
            uint32_t remainingElements = static_cast<uint32_t>(mSize) * nDstUbAligned64;
            uint32_t zeroOffset = 0;
            while (remainingElements > 0) {
                AscendC::Reg::MaskReg zeroMask = AscendC::Reg::UpdateMask<bfloat16_t>(remainingElements);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(gluResAddr + zeroOffset,
                                                                                           zeroReg, zeroMask);
                zeroOffset += bf16Vl;
            }
        }
    }

    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = singleN_;
            AscendC::Reg::MaskReg mask;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < oneRowRepeatTimes; vfBlockIdx++) {
                mask = AscendC::Reg::UpdateMask<DataTypeIn>(elementNum);

                AscendC::Reg::RegTensor<bfloat16_t> verg7;
                AscendC::Reg::RegTensor<float> swishInput, gateInput;
                AscendC::Reg::RegTensor<float> verg1, verg2, verg3, verg4, verg5, verg6, swishOutput;

                uint32_t l0cOutOffset = mIdx * concatNSrcUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::Reg::DataCopy(swishInput, swishInputUbAddr + l0cOutOffset);

                // Swish: x / (1 + exp(-x))
                AscendC::Reg::Muls(verg2, swishInput, -(scalarOne), mask);
                AscendC::Reg::Exp(verg3, verg2, mask);
                AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);

                // Load gate data
                AscendC::Reg::DataCopy(gateInput, swishInputUbAddr + l0cOutOffset + nSrcUbAligned);

                // SwiGLU = Swish(act) * gate
                AscendC::Reg::Mul(verg6, swishOutput, gateInput, mask);

                AscendC::Reg::Cast<bfloat16_t, float, CT_FP32_TO_BF16>(verg7, verg6, mask);

                uint32_t dstUbOffset = mIdx * nDstUbAligned64 + vfBlockIdx * sizePerRepeat;
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(gluResAddr + dstUbOffset,
                                                                                           verg7, mask);
            }
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::ComputeMxQuant(
    uint16_t mSize)
{
    __ubuf__ bfloat16_t* gluResAddr = GetUbAddr<bfloat16_t>(gluResUbOffset_);
    __ubuf__ int8_t* quantOutputInUbAddr = GetUbAddr<int8_t>(quantOutputUbOffset_);
    __ubuf__ uint16_t* quantScaleOutputInUbAddr = GetUbAddr<uint16_t>(quantScaleOutputUbOffset_);

    const uint32_t nDstUbAligned64 = Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_));
    const uint32_t totalDataInUb = mSize * nDstUbAligned64;
    const uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    const uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * Constant::NUM_TWO - 1) /
                                 (vlForHalfNumber_ * Constant::NUM_TWO);
    const uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;

    __ubuf__ uint16_t* maxExpAddr = GetUbAddr<uint16_t>(maxExpUbOffset_);
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);

    __ubuf__ uint16_t* halfScaleLocalAddr = GetUbAddr<uint16_t>(halfScaleUbOffset_);
    ComputeScale(maxExpAddr, quantScaleOutputInUbAddr, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, quantOutputInUbAddr, totalDataInUb, loopDataNum);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::ComputeMaxExp(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract0;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract1;

        AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
        AscendC::Reg::Duplicate(expMaskBF16, Constant::MAX_EXP_FOR_BF16);

        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::MaskReg scaleMask1;
        AscendC::Reg::MaskReg scaleMask2;
        AscendC::Reg::UnalignReg u1;

        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            scaleMask2 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::Reg::MaskDeInterleave<bfloat16_t>(scaleMask1, scaleMask2, scaleMask1, scaleMask2);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                            vlForHalfNumber_ * Constant::NUM_TWO);
            AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
            AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask2);

            AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, static_cast<uint32_t>(elementAfterReduce_));
        }
        AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::ComputeScale(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue;
        AscendC::Reg::RegTensor<uint16_t> scaleBias, halfScale, fp8NanRegTensor;
        AscendC::Reg::Duplicate(expMask, Constant::MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask;
        AscendC::Reg::MaskReg preMaskScale;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::Reg::Duplicate(maxExpValue, fpEmax_);
        AscendC::Reg::Duplicate(scaleBias, Constant::BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8NanRegTensor, Constant::MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::Duplicate(nanRegTensor, Constant::NAN_CUSTOMIZATION);
        AscendC::Reg::Duplicate(specialExpRegTensor, Constant::SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                          vlForHalfNumber_);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, Constant::SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                           vlForHalfNumber_ >> 1, preMaskScale);

            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    uint32_t totalCountInUB2 = totalCountInUB * Constant::NUM_TWO;
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
        AscendC::Reg::MaskReg dataMaskEven, dataMaskOdd;
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
        AscendC::Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::Reg::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            dataMask3 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);
            dataMask4 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);
            AscendC::Reg::MaskDeInterleave<T>(dataMaskEven, dataMaskOdd, dataMask1, dataMask2);
            AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                            vlForHalfNumber_ * Constant::NUM_TWO);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                         elementAfterReduce_);

            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMaskEven);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMaskOdd);
            AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);

            AscendC::Reg::Cast<float, T, CT_BF16_TO_FP32_ZERO>(vdExp0FP32Zero, vdExp0, dataMask1);
            AscendC::Reg::Cast<float, T, CT_BF16_TO_FP32_ONE>(vdExp0FP32One, vdExp0, dataMask1);
            AscendC::Reg::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
            AscendC::Reg::Cast<DataTypeOut, float, CT_FP32_TO_FP8_SAT>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::Reg::Cast<DataTypeOut, float, CT_FP32_TO_FP8_SAT>(vdExp0FP8One, vdExp0FP32One, dataMask3);

            AscendC::Reg::Cast<float, T, CT_BF16_TO_FP32_ZERO>(vdExp1FP32Zero, vdExp1, dataMask2);
            AscendC::Reg::Cast<float, T, CT_BF16_TO_FP32_ONE>(vdExp1FP32One, vdExp1, dataMask2);
            AscendC::Reg::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
            AscendC::Reg::Cast<DataTypeOut, float, CT_FP32_TO_FP8_SAT>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::Reg::Cast<DataTypeOut, float, CT_FP32_TO_FP8_SAT>(vdExp1FP8One, vdExp1FP32One, dataMask4);

            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8Zero, Constant::OUT_ELE_NUM_ONE_BLK,
                dataMask3);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8One, Constant::OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP8Zero, Constant::OUT_ELE_NUM_ONE_BLK,
                dataMask4);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP8One, Constant::OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
