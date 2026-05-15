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

#include "gmm_swiglu_quant_mx_vf.h"

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

/*!
 * \brief UB buffer layout information for GmmSwigluQuantMxVecCompute.
 *
 * Defines fixed byte offsets and total sizes for each UB tensor.
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
class GmmSwigluQuantMxVecCompute {
public:
    __aicore__ inline GmmSwigluQuantMxVecCompute();
    __aicore__ inline void Init();
    __aicore__ inline void UpdateGlobalAddr(__gm__ int8_t *y, __gm__ int8_t *yScale);
    __aicore__ inline void UpdateOutputShape(int64_t n, int64_t scaleN);
    __aicore__ inline void VFDoSwigluAndQuantForMX(AscendC::LocalTensor<DataTypeIn> &l0cOutUbFirst,
                                                   AscendC::LocalTensor<DataTypeIn> &l0cOutUbSecond,
                                                   uint16_t halfMSize, uint16_t halfNSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);
    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset);

private:
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
 * UB buffer layout
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
 * \brief Transposes MX scale layout from interleaved to block format in UB.
 * \param[in] mSize       M dimension size.
 * \param[in] scaleBlockN Number of scale elements per block.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::TransMxScaleLayout(uint16_t mSize,
                                                                                    uint16_t scaleBlockN)
{
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr();

    TransMxScaleLayoutVf(quantScaleOutputInUbAddr, quantScaleBlockOutputInUbAddr, mSize, scaleBlockN);
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
 * \param[in] halfMSize      Half of the M dimension size of L0C output.
 * \param[in] halfNSize      Half of the N dimension size of L0C output.
 */
GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_CLASS::VFDoSwigluAndQuantForMX(
    AscendC::LocalTensor<DataTypeIn> &l0cOutUbFirst, AscendC::LocalTensor<DataTypeIn> &l0cOutUbSecond,
    uint16_t halfMSize, uint16_t halfNSize)
{
    singleN_ = halfNSize;
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
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();
    GmmSwigluVf<DataTypeIn>(firstSrc, secondSrc, gluResAddr, halfMSize, singleN_,
                            nSrcUbAligned, nDstUbAligned, oneRowRepeatTimes, sizePerRepeat);

    // Quant
    uint32_t totalDataInUb = halfMSize * nDstUbAligned;
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = CeilDiv(static_cast<uint64_t>(totalDataInUb), static_cast<uint64_t>(vlForHalfNumber_ * 2));
    uint16_t loopScaleNum = CeilDiv(static_cast<uint64_t>(totalScaleInUb), static_cast<uint64_t>(vlForHalfNumber_));

    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr();
    ComputeMaxExpVf(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum,
                    vlForHalfNumber_, elementAfterReduce_);

    ComputeScaleVf(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUb, loopScaleNum,
                   vlForHalfNumber_, fpEmax_);

    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        ComputeDataForQuantTargetFp8Vf<DataTypeOut>(gluResAddr, halfScaleLocalAddr, outLocalAddr,
                                                     totalDataInUb, loopDataNum,
                                                     vlForHalfNumber_, elementAfterReduce_);
    }
}

} // namespace GmmSwigluQuantMxA8W4

#endif // GMM_SWIGLU_QUANT_MX_VEC_COMPUTE_H
