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
 * \file gmm_swiglu_quant_mx_vec_compute.h
 * \brief GMMSwigluQuant MxA8W4 VecCompute
 */

#ifndef GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_H
#define GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace GmmSwigluQuantMxA8W4 {

constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint64_t KB_SIZE = 1024;
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
constexpr uint32_t MXFP_DIVISOR_SIZE = 64;
constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2;

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

/*!
 * \brief UB buffer layout information for GmmSwigluQuantMxVecCompute.
 *
 * Defines fixed byte offsets and total sizes for each UB tensor,
 * aligned with SRS_GMMSQ.md section 8.1.
 */
struct GmmSwigluQuantMxUbBufferInfo {
    uint64_t weight8BitOffset;
    uint64_t weight8BitTotalSize;
    uint64_t yF32BufferOffset;
    uint64_t yF32BufferTotalSize;
    uint64_t biasOffset;
    uint64_t biasTotalSize;
    uint64_t biasRescaleOffset;
    uint64_t biasRescaleTotalSize;
    uint64_t maxB16BufferOffset;
    uint64_t maxB16BufferTotalSize;
    uint64_t scaleE8M0BufferOffset;
    uint64_t scaleE8M0BufferTotalSize;
    uint64_t yFp8BufferOffset;
    uint64_t yFp8BufferTotalSize;
    uint64_t gluResInYF32Offset;
    uint64_t gluResInYF32TotalSize;
    uint64_t mxScaleInYF32Offset;
    uint64_t mxScaleInYF32TotalSize;
    uint64_t halfScaleInYF32Offset;
    uint64_t halfScaleInYF32TotalSize;
};

/*!
 * \brief Returns the fixed UB buffer layout configuration.
 * \return GmmSwigluQuantMxUbBufferInfo instance with preset offsets and sizes.
 */
__aicore__ constexpr GmmSwigluQuantMxUbBufferInfo GetGmmSwigluQuantMxUbBufferInfo()
{
    return {
        .weight8BitOffset = 0 * KB_SIZE,
        .weight8BitTotalSize = 64 * KB_SIZE,
        .yF32BufferOffset = 64 * KB_SIZE,
        .yF32BufferTotalSize = 128 * KB_SIZE,
        .biasOffset = 192 * KB_SIZE,
        .biasTotalSize = 1 * KB_SIZE,
        .biasRescaleOffset = 193 * KB_SIZE,
        .biasRescaleTotalSize = 1 * KB_SIZE,
        .maxB16BufferOffset = 194 * KB_SIZE,
        .maxB16BufferTotalSize = 2 * KB_SIZE,
        .scaleE8M0BufferOffset = 196 * KB_SIZE,
        .scaleE8M0BufferTotalSize = 16 * KB_SIZE,
        .yFp8BufferOffset = 212 * KB_SIZE,
        .yFp8BufferTotalSize = 32 * KB_SIZE,
        .gluResInYF32Offset = 64 * KB_SIZE,
        .gluResInYF32TotalSize = 32 * KB_SIZE,
        .mxScaleInYF32Offset = 96 * KB_SIZE,
        .mxScaleInYF32TotalSize = 1 * KB_SIZE,
        .halfScaleInYF32Offset = 97 * KB_SIZE,
        .halfScaleInYF32TotalSize = 1 * KB_SIZE,
    };
}

/*!
 * \brief Converts byte size to element count for a given data type.
 * \tparam T Data type.
 * \param[in] byteSize Size in bytes.
 * \return Number of elements of type T.
 */
template <typename T>
__aicore__ constexpr uint64_t ByteSizeToElementCount(uint64_t byteSize)
{
    return byteSize / sizeof(T);
}

/*!
 * \brief Template parameter macro for GmmSwigluQuantMxVecCompute class methods.
 */
#define GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM \
    template <typename DataTypeIn, typename DataTypeOut, typename ScaleType>

/*!
 * \brief Class name macro for GmmSwigluQuantMxVecCompute class methods.
 */
#define GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS \
    GmmSwigluQuantMxVecCompute<DataTypeIn, DataTypeOut, ScaleType>

GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
/*!
 * \brief Vector compute class for GMM SwiGLU + MX Quantization.
 *
 * Handles SwiGLU activation, max-exp extraction, scale computation,
 * and FP8/FP4 quantization on the Ascend NPU vector unit.
 *
 * \tparam DataTypeIn  Input data type (e.g., float, bfloat16_t).
 * \tparam DataTypeOut Output quantized data type (e.g., fp8_e4m3fn_t, fp4x2_e2m1_t).
 * \tparam ScaleType   Scale data type (not used directly in this class).
 */
class GmmSwigluQuantMxVecCompute {
public:
    __aicore__ inline GmmSwigluQuantMxVecCompute();
    __aicore__ inline void Init();
    __aicore__ inline void UpdateGlobalAddr(__gm__ int8_t *y, __gm__ int8_t *yScale);
    __aicore__ inline void UpdateOutputShape(int64_t n, int64_t scaleN);
    __aicore__ inline void VFDoSwigluAndQuantForMX(AscendC::LocalTensor<DataTypeIn> &l0cOutUbFirst,
                                                   AscendC::LocalTensor<DataTypeIn> &l0cOutUbSecond,
                                                   uint16_t mSize, uint16_t nSize);
    __aicore__ inline void TransMxScaleLayoutVf(uint16_t mSize, uint16_t scaleBlockN);
    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset);

private:
    __aicore__ inline void GmmSwigluVf(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                       uint16_t mSize, uint16_t nSize,
                                       uint32_t nSrcUbAligned, uint32_t nDstUbAligned,
                                       uint16_t oneRowRepeatTimes, uint16_t sizePerRepeat);
    __aicore__ inline void ComputeMaxExpVf(uint32_t totalCountInUB, uint16_t loopNum);
    __aicore__ inline void ComputeScaleVf(uint32_t totalScaleInUB, uint16_t loopNumScale);
    __aicore__ inline void ComputeDataForQuantTargetFp8Vf(uint32_t totalCountInUB, uint16_t loopNum);

    constexpr static GmmSwigluQuantMxUbBufferInfo UB_BUFFER_INFO = GetGmmSwigluQuantMxUbBufferInfo();

    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;

    // UB ADDR
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<int8_t> quantScaleBlockOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    int64_t n_ = 0;
    int64_t scaleN_ = 0;
    int64_t scaleBlockN_ = 0;
    uint32_t singleN_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;
};

/*!
 * \brief Default constructor.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::GmmSwigluQuantMxVecCompute()
{}

/*!
 * \brief Initializes UB tensor allocations and sets fpEmax_ based on output type.
 *
 * UB buffer layout follows SRS_GMMSQ.md section 8.1.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::Init()
{
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    // UB buffer allocation follows SRS_GMMSQ.md section 8.1
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(
        AscendC::TPosition::VECCALC, UB_BUFFER_INFO.gluResInYF32Offset,
        ByteSizeToElementCount<bfloat16_t>(UB_BUFFER_INFO.gluResInYF32TotalSize));

    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, UB_BUFFER_INFO.mxScaleInYF32Offset,
        ByteSizeToElementCount<int8_t>(UB_BUFFER_INFO.mxScaleInYF32TotalSize));

    halfScale_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, UB_BUFFER_INFO.halfScaleInYF32Offset,
        ByteSizeToElementCount<uint16_t>(UB_BUFFER_INFO.halfScaleInYF32TotalSize));

    maxExp_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, UB_BUFFER_INFO.maxB16BufferOffset,
        ByteSizeToElementCount<uint16_t>(UB_BUFFER_INFO.maxB16BufferTotalSize));

    quantScaleBlockOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, UB_BUFFER_INFO.scaleE8M0BufferOffset,
        ByteSizeToElementCount<int8_t>(UB_BUFFER_INFO.scaleE8M0BufferTotalSize));

    quantOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, UB_BUFFER_INFO.yFp8BufferOffset,
        ByteSizeToElementCount<int8_t>(UB_BUFFER_INFO.yFp8BufferTotalSize));
}

/*!
 * \brief Updates global memory addresses for output and scale.
 * \param[in] y      Pointer to output global memory.
 * \param[in] yScale Pointer to scale global memory.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::UpdateGlobalAddr(__gm__ int8_t *y, __gm__ int8_t *yScale)
{
    quantOutputGlobal_.SetGlobalBuffer(y);
    quantScaleGlobal_.SetGlobalBuffer(yScale);
}

/*!
 * \brief Updates output tensor shape dimensions for GM stride calculation.
 * \param[in] n      Total N dimension size of output Y.
 * \param[in] scaleN Total N dimension size of output scale.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::UpdateOutputShape(int64_t n, int64_t scaleN)
{
    n_ = n;
    scaleN_ = scaleN;
}

/*!
 * \brief Computes SwiGLU activation using vector instructions.
 *
 * The result is stored into the internal `gluRes_` UB buffer.
 * UB addresses for inputs come from caller; output address is fetched
 * internally via `gluRes_.GetPhyAddr()`.
 *
 * \param[in] firstSrc          Swish branch input in UB.
 * \param[in] secondSrc         Gate branch input in UB.
 * \param[in] mSize             M dimension size.
 * \param[in] nSize             N dimension size.
 * \param[in] nSrcUbAligned     Aligned N stride for source UB.
 * \param[in] nDstUbAligned     Aligned N stride for destination UB.
 * \param[in] oneRowRepeatTimes Number of vector repeats per row.
 * \param[in] sizePerRepeat     Elements processed per vector repeat.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::GmmSwigluVf(__ubuf__ DataTypeIn *firstSrc,
                                                                          __ubuf__ DataTypeIn *secondSrc,
                                                                          uint16_t mSize, uint16_t nSize,
                                                                          uint32_t nSrcUbAligned, uint32_t nDstUbAligned,
                                                                          uint16_t oneRowRepeatTimes,
                                                                          uint16_t sizePerRepeat)
{
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();
    const float scalarOne = 1.0;
    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        uint32_t elementNum = nSize;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < oneRowRepeatTimes; vfBlockIdx++) {
            AscendC::MicroAPI::RegTensor<bfloat16_t> verg7;
            AscendC::MicroAPI::RegTensor<float> swishInput, gateInput;
            AscendC::MicroAPI::RegTensor<float> verg2, verg3, verg4, verg6, swishOutput;

            uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;

            // 1. Load Swish branch
            AscendC::MicroAPI::DataCopy(swishInput, firstSrc + l0cOutOffset);

            // 2. Swish: x / (1 + exp(-x))
            AscendC::MicroAPI::Muls(verg2, swishInput, -(scalarOne), mask);
            AscendC::MicroAPI::Exp(verg3, verg2, mask);
            AscendC::MicroAPI::Adds(verg4, verg3, scalarOne, mask);
            AscendC::MicroAPI::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);

            // 3. Load Gate branch
            AscendC::MicroAPI::DataCopy(gateInput, secondSrc + l0cOutOffset);

            // 4. SwiGLU: Swish(x) * gate
            AscendC::MicroAPI::Mul(verg6, swishOutput, gateInput, mask);

            // 5. Cast to BF16 and store
            AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, mask);
            uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                gluResAddr + dstUbOffset, verg7, mask);
        }
    }
}

/*!
 * \brief Extracts maximum exponent per 32-byte block from `gluRes_`.
 *
 * Source and destination UB addresses are fetched internally from
 * `gluRes_` and `maxExp_` member tensors.
 *
 * \param[in] totalCountInUB Total element count in UB to process.
 * \param[in] loopNum        Number of vector loops required.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::ComputeMaxExpVf(uint32_t totalCountInUB,
                                                                              uint16_t loopNum)
{
    __ubuf__ bfloat16_t *srcAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
    AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
    AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
    AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::MaskReg scaleMask1;
    AscendC::MicroAPI::UnalignReg u1;

    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);

        AscendC::MicroAPI::DataCopy<bfloat16_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * 2);

        AscendC::MicroAPI::And(vdExpExtract0,
                               (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);
        AscendC::MicroAPI::And(vdExpExtract1,
                               (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);

        AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

        AscendC::MicroAPI::DataCopyUnAlign<uint16_t,
                                           AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
}

/*!
 * \brief Computes MX scale (E8M0) and half-scale from max exponents.
 *
 * UB addresses for maxExp, mxScale and halfScale are fetched internally
 * from member tensors `maxExp_`, `quantScaleOutput_` and `halfScale_`.
 *
 * \param[in] totalScaleInUB Total scale element count in UB.
 * \param[in] loopNumScale   Number of vector loops for scale computation.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::ComputeScaleVf(uint32_t totalScaleInUB,
                                                                             uint16_t loopNumScale)
{
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
    AscendC::MicroAPI::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask, preMaskScale;
    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
    AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_);
    AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
    AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
    AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
    AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
    AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

    for (uint16_t i = 0; i < loopNumScale; i++) {
        preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);

        AscendC::MicroAPI::DataCopy<uint16_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            vdMaxExp, maxExpAddr, vlForHalfNumber_);

        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
            cmpResult, vdMaxExp, expMask, preMaskScale);
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
            zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(
            invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

        AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        AscendC::MicroAPI::DataCopy<uint16_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
            mxScaleLocalAddr, scaleValue, vlForHalfNumber_ >> 1, preMaskScale);

        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(
            specialDataMask, sharedExp, scaleBias, preMaskScale);

        AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        AscendC::MicroAPI::DataCopy<uint16_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
    }
}

/*!
 * \brief Quantizes FP8 data using computed half-scale.
 *
 * Source, scale and output UB addresses are fetched internally from
 * `gluRes_`, `halfScale_` and `quantOutput_` member tensors.
 *
 * \param[in] totalCountInUB Total element count in UB to quantize.
 * \param[in] loopNum        Number of vector loops required.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::ComputeDataForQuantTargetFp8Vf(
    uint32_t totalCountInUB, uint16_t loopNum)
{
    __ubuf__ bfloat16_t *srcAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr();
    uint32_t totalCountInUB2 = totalCountInUB * 2;
    using T = bfloat16_t;

    AscendC::MicroAPI::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
    AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
    AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
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

        AscendC::MicroAPI::DataCopy<T,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * 2);

        AscendC::MicroAPI::DataCopy<uint16_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
            halfScaleForMul, halfScaleLocalAddr, elementAfterReduce_);

        AscendC::MicroAPI::Mul(vdExp0, vdExp0,
                               (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::MicroAPI::Mul(vdExp1, vdExp1,
                               (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);

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

        AscendC::MicroAPI::DataCopy<int8_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero,
            OUT_ELE_NUM_ONE_BLK, dataMask3);
        AscendC::MicroAPI::DataCopy<int8_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8One,
            OUT_ELE_NUM_ONE_BLK, dataMask3);
        AscendC::MicroAPI::DataCopy<int8_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8Zero,
            OUT_ELE_NUM_ONE_BLK, dataMask4);
        AscendC::MicroAPI::DataCopy<int8_t,
                                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8One,
            OUT_ELE_NUM_ONE_BLK, dataMask4);
    }
}

/*!
 * \brief Transposes MX scale layout from interleaved to block format in UB.
 * \param[in] mSize       M dimension size.
 * \param[in] scaleBlockN Number of scale elements per block.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::TransMxScaleLayoutVf(uint16_t mSize,
                                                                                    uint16_t scaleBlockN)
{
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr();

    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        uint32_t elemNum = scaleBlockN;
        AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
        AscendC::MicroAPI::RegTensor<int8_t> vreg0;
        AscendC::MicroAPI::UnalignReg u0;

        auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb);
        AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, srcUb);

        auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
        AscendC::MicroAPI::DataCopy<int8_t,
                                    AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
            dstUb, vreg0, maskScaleN);
    }
}

/*!
 * \brief Copies quantized output from UB to global memory.
 *
 * Uses the internal `quantOutput_` UB tensor as source.
 *
 * \param[in] blockCount Number of blocks to copy.
 * \param[in] offset     Offset in global memory.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::CopyOutputFromUb2Gm(uint64_t blockCount,
                                                                                  uint64_t offset)
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
    AscendC::DataCopyPad(quantOutputGlobal_[offset], quantOutput_, ub2GmParams);
}

/*!
 * \brief Copies scale data from UB to global memory.
 *
 * Uses the internal `quantScaleBlockOutput_` UB tensor as source.
 *
 * \param[in] blockCount Number of blocks to copy.
 * \param[in] offset     Offset in global memory.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::CopyScaleFromUb2Gm(uint64_t blockCount,
                                                                                 uint64_t offset)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    auto blockScaleN =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    ub2GmParams.blockLen = blockScaleN * sizeof(int8_t);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (scaleN_ - blockScaleN) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], quantScaleBlockOutput_, ub2GmParams);
}

/*!
 * \brief Performs SwiGLU activation and MX quantization in one pipeline.
 * \param[in] l0cOutUbFirst  First L0C output tensor (Swish branch).
 * \param[in] l0cOutUbSecond Second L0C output tensor (Gate branch).
 * \param[in] mSize          M dimension size for this iteration.
 * \param[in] nSize          N dimension size for this iteration.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::VFDoSwigluAndQuantForMX(
    AscendC::LocalTensor<DataTypeIn> &l0cOutUbFirst, AscendC::LocalTensor<DataTypeIn> &l0cOutUbSecond,
    uint16_t mSize, uint16_t nSize)
{
    singleN_ = nSize;
    scaleBlockN_ =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / BLOCK_SIZE;

    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t oneRowRepeatTimes =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(sizePerRepeat));
    uint32_t nSrcUbAligned = CeilAlign(singleN_, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned = CeilAlign(singleN_, AscendC::ONE_BLK_SIZE);

    __ubuf__ DataTypeIn *firstSrc = (__ubuf__ DataTypeIn *)l0cOutUbFirst.GetPhyAddr();
    __ubuf__ DataTypeIn *secondSrc = (__ubuf__ DataTypeIn *)l0cOutUbSecond.GetPhyAddr();

    // SwiGLU
    GmmSwigluVf(firstSrc, secondSrc, mSize, singleN_,
                nSrcUbAligned, nDstUbAligned, oneRowRepeatTimes, sizePerRepeat);

    // Quant
    uint32_t totalDataInUb = mSize * nDstUbAligned;
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;

    ComputeMaxExpVf(totalDataInUb, loopDataNum);

    ComputeScaleVf(totalScaleInUb, loopScaleNum);

    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        ComputeDataForQuantTargetFp8Vf(totalDataInUb, loopDataNum);
    }
}

} // namespace GmmSwigluQuantMxA8W4

#endif // GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_H
