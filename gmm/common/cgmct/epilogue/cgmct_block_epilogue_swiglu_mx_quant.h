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

#ifndef EPILOGUE_BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#define EPILOGUE_BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/cgmct_common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Cgmct {
namespace Gemm {
namespace Block {

namespace Gmmsg {
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
} // namespace Gmmsg

namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint64_t MX_QUANT_COMPUTE_ALIGN = 64UL; // MX量化计算的对齐约束: N轴需按64对齐处理
constexpr uint64_t DATA_BLOCK_SIZE = 32UL;        // UB上一个DataBlock的大小，即32字节。
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 64 * 256;
constexpr uint32_t CGMCT_INTERLEAVED_REG_FACTOR = 2;
constexpr uint32_t PADDED_QUANT_SCALE_SIZE = 64 * 32;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t NUMBER_HALF = 0x00400000;
constexpr float FP8_E4M3_MAX_VALUE = 448.0F;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0F;
} // namespace

constexpr AscendC::Reg::CastTrait ctInt322Fp32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN,
                                                  AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait ctFp322Half = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                 AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait ctHalf2Fp32Zero = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait ctHalf2Fp32One = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::Reg::DivSpecificMode DIV_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::Reg::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, \
              typename DataTypeX1Scale_, bool IsTensorList_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
class BlockEpilogueSwigluQuant {
public:
    __aicore__ inline BlockEpilogueSwigluQuant() {}

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // V3 SwiGLU controls.  The defaults preserve the V2 standard SwiGLU path.
        int64_t swigluMode{0};
        bool activateLeft{true};
        float clampLimit{7.0F};
        float gluAlpha{1.702F};
        float gluBias{1.0F};
        uint32_t scaleAlg{0};
        float dstTypeMax{0.0F};
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

    // 输出为FP4类型(e2m1/e1m2)的编译期标记
    static constexpr bool IS_FP4_OUT =
        AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value || AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value;

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t>; // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    template <Gmmsg::QuantMode quantMode>
    __aicore__ inline void VFDoSwigluAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                   __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                   uint16_t mSize, uint16_t nSize);

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);

    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeMaxExpCublas(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                               uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeScaleCublas(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                              __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                              uint16_t loopNumScale);

    __aicore__ inline void ComputeScaleDynamicDtypeRange(__ubuf__ uint16_t *maxExpAddr,
                                                         __ubuf__ uint16_t *mxScaleLocalAddr,
                                                         __ubuf__ uint16_t *halfScaleLocalAddr,
                                                         uint32_t totalScaleInUB, uint16_t loopNumScale);

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                              AscendC::LocalTensor<AscendC::fp8_e8m0_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> quantScaleGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, MAX_SINGLE_MN * sizeof(DataTypeIn),
                                                     MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<AscendC::fp8_e8m0_t> quantScaleBlockOutput_;
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

    int64_t swigluMode_ = 0;
    bool activateLeft_ = true;
    float clampLimit_ = 7.0F;
    float gluAlpha_ = 1.702F;
    float gluBias_ = 1.0F;
    float invDstTypeMax_ = 1.0F / FP8_E4M3_MAX_VALUE;

    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(
    Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    swigluMode_ = params.swigluMode;
    activateLeft_ = params.activateLeft;
    clampLimit_ = params.clampLimit;
    gluAlpha_ = params.gluAlpha;
    gluBias_ = params.gluBias;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        invDstTypeMax_ = 1.0F / FP8_E5M2_MAX_VALUE;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    // out
    constexpr uint32_t afterIn = 2 * MAX_SINGLE_MN * sizeof(DataTypeIn); // 2: 2 block in
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterIn, MAX_SINGLE_MN);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, afterIn + MAX_SINGLE_MN * sizeof(int8_t), MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIO =
        afterIn + MAX_SINGLE_MN * sizeof(int8_t) + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    // swiglu res
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, afterIO, MAX_SINGLE_MN);
    constexpr uint32_t afterIOAndGlu = afterIO + MAX_SINGLE_MN * sizeof(bfloat16_t);
    // sharedExp
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGlu,
                                             MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIOAndGluExp = afterIOAndGlu + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGluExp,
                                                MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t realScaleBlockOffset =
        afterIOAndGluExp + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    quantScaleBlockOutput_ = AscendC::LocalTensor<AscendC::fp8_e8m0_t>(
        AscendC::TPosition::VECOUT, realScaleBlockOffset,
        params_->baseM / AscendC::GetTaskRation() * AscendC::ONE_BLK_SIZE);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer(
            (__gm__ AscendC::fp8_e8m0_t *)params_->yScaleGmAddr + Get<Y_SCALE_IDX>(baseOffset));
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<MNK_N>(problemShape);
    scaleN_ = CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(64)) * MXFP_MULTI_BASE_SIZE;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = singleN_ * sizeof(int8_t);
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(int8_t);

    if (unlikely(singleN_ % MX_QUANT_COMPUTE_ALIGN != 0)) {
        uint64_t actualDataBlockNum = CeilDiv(static_cast<uint64_t>(singleN_), DATA_BLOCK_SIZE);
        uint64_t alignedDataBlockNum = CeilDiv(Align64(static_cast<uint64_t>(singleN_)), DATA_BLOCK_SIZE);
        // UB中每行按64对齐存放，srcStride用于跳过行尾补齐的DataBlock: 对齐后块数 - 原始singleN块数
        ub2GmParams.srcStride = (alignedDataBlockNum - actualDataBlockNum);
        if constexpr (IS_FP4_OUT) {
            ub2GmParams.srcStride = ub2GmParams.srcStride >> 1;
        }
    }

    if constexpr (IS_FP4_OUT) {
        ub2GmParams.blockLen = ub2GmParams.blockLen >> 1;
        ub2GmParams.dstStride = ub2GmParams.dstStride >> 1;
        offset = offset >> 1;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<AscendC::fp8_e8m0_t> &src)
{
    // scaleBlockN_ is the UB source row pitch (it is based on Align64(singleN)
    // because ComputeMxQuant also computes the aligned tail).  It is not the
    // number of valid scale bytes in the public yScale row.  Copy only the
    // valid groups; otherwise a 32-value tile would copy its padding byte and
    // the following tile would be addressed past the compact yScale row.
    constexpr uint64_t scaleGroupSize = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;
    auto validScaleN = CeilDiv(static_cast<uint64_t>(singleN_), scaleGroupSize);
    // quantScaleBlockOutput_ uses a 32-byte UB row for every logical M row,
    // whereas yScale is compact (validScaleN bytes per row).  A multi-row
    // DataCopyPad with a sub-32-byte blockLen expresses its source stride in
    // hardware blocks, so it cannot represent this 32-byte-to-compact-row
    // conversion without rounding ambiguity.  Copy each row from its explicit
    // 32-byte UB address instead.  This keeps the public yScale rows compact
    // for all N tiles, including the common N=64 output tile.
    AscendC::DataCopyExtParams params{1, 0, 0, 0, 0};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(validScaleN * sizeof(AscendC::fp8_e8m0_t));
    for (uint64_t row = 0; row < blockCount; ++row) {
        AscendC::DataCopyPad(quantScaleGlobal_[offset + row * scaleN_],
                             src[row * AscendC::ONE_BLK_SIZE], params);
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract0;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract1;

        AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
        AscendC::Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::MaskReg scaleMask1;
        AscendC::Reg::MaskReg scaleMask2;
        AscendC::Reg::UnalignReg u1;
        static constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16 = {
            AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            scaleMask2 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);
            AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);

            AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::Reg::Duplicate(maxExpValue, fpEmax_);
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
        AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                          vlForHalfNumber_);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                  preMaskScale); // INF/NAN
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
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
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExpCublas(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> value0;
        AscendC::Reg::RegTensor<bfloat16_t> value1;
        AscendC::Reg::RegTensor<uint16_t> absMask;
        AscendC::Reg::RegTensor<uint16_t> maxValue;
        AscendC::Reg::MaskReg mask0;
        AscendC::Reg::MaskReg mask1;
        AscendC::Reg::MaskReg maskEven;
        AscendC::Reg::MaskReg maskOdd;
        AscendC::Reg::UnalignReg unalign;
        AscendC::Reg::Duplicate(absMask, 0x7fff);
        for (uint16_t i = 0; i < loopNum; ++i) {
            mask0 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            mask1 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::Reg::MaskDeInterleave<bfloat16_t>(maskEven, maskOdd, mask0, mask1);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                value0, value1, srcAddr, vlForHalfNumber_ * 2);
            AscendC::Reg::And(reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0),
                              reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0), absMask, maskEven);
            AscendC::Reg::And(reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1),
                              reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), absMask, maskOdd);
            AscendC::Reg::Max(maxValue, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0),
                              reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), mask0);
            AscendC::Reg::ReduceMaxWithDataBlock(maxValue, maxValue, mask0);
            AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, maxValue, unalign, elementAfterReduce_);
        }
        AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, unalign, 0);
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScaleCublas(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> max16;
        AscendC::Reg::RegTensor<uint32_t> max32;
        AscendC::Reg::RegTensor<uint32_t> exp32;
        AscendC::Reg::RegTensor<uint32_t> mantissa32;
        AscendC::Reg::RegTensor<uint32_t> expAddOne32;
        AscendC::Reg::RegTensor<uint32_t> extractExp;
        AscendC::Reg::RegTensor<uint16_t> expOut;
        AscendC::Reg::RegTensor<uint32_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> reciprocalExpOut;
        AscendC::Reg::RegTensor<float> invMax;
        AscendC::Reg::RegTensor<uint32_t> mantissaMask;
        AscendC::Reg::RegTensor<uint32_t> expMask;
        AscendC::Reg::RegTensor<uint32_t> zero;
        AscendC::Reg::RegTensor<uint32_t> scaleBias;
        AscendC::Reg::RegTensor<uint32_t> nan;
        AscendC::Reg::RegTensor<uint32_t> fp8Nan;
        AscendC::Reg::MaskReg finiteMask;
        AscendC::Reg::MaskReg nonzeroMask;
        AscendC::Reg::MaskReg predicate0;
        AscendC::Reg::MaskReg predicate1;
        AscendC::Reg::MaskReg predicate2;
        AscendC::Reg::MaskReg maskB16;
        AscendC::Reg::MaskReg maskFloat;
        static constexpr AscendC::Reg::CastTrait castBf16ToFloat = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
            AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

        AscendC::Reg::Duplicate(invMax, invDstTypeMax_);
        AscendC::Reg::Duplicate(mantissaMask, MAN_MASK_FLOAT);
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::Reg::Duplicate(zero, NUMBER_ZERO);
        AscendC::Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::Reg::Duplicate(nan, NAN_CUSTOMIZATION_PACK);
        AscendC::Reg::Duplicate(fp8Nan, MAX_EXP_FOR_FP8_IN_FP32);

        uint32_t remainingScale = totalScaleInUB;
        for (uint16_t i = 0; i < loopNumScale; ++i) {
            uint32_t processedScale =
                remainingScale < vlForHalfNumber_ / 2 ? remainingScale : vlForHalfNumber_ / 2;
            maskB16 = AscendC::Reg::UpdateMask<uint16_t>(processedScale);
            maskFloat = AscendC::Reg::UpdateMask<uint32_t>(processedScale);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr,
                                                                            vlForHalfNumber_ / 2);
            AscendC::Reg::Cast<float, bfloat16_t, castBf16ToFloat>(
                reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32),
                reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(max16), maskFloat);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(finiteMask, max32, expMask, maskFloat);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(nonzeroMask, max32, zero, maskFloat);
            AscendC::Reg::Mul(reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32),
                              reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32), invMax, maskFloat);
            AscendC::Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskFloat);
            AscendC::Reg::And(mantissa32, max32, mantissaMask, maskFloat);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate0, exp32, NUMBER_ZERO, maskFloat);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(predicate1, exp32,
                                                                         NUMBER_TWO_FIVE_FOUR, maskFloat);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate2, mantissa32, NUMBER_ZERO, maskFloat);
            AscendC::Reg::MaskAnd(predicate0, predicate0, predicate1, maskFloat);
            AscendC::Reg::MaskAnd(predicate0, predicate0, predicate2, maskFloat);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(predicate1, exp32, NUMBER_ZERO, maskFloat);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate2, mantissa32, NUMBER_HALF, maskFloat);
            AscendC::Reg::MaskAnd(predicate1, predicate1, predicate2, maskFloat);
            AscendC::Reg::MaskOr(predicate0, predicate0, predicate1, maskFloat);
            AscendC::Reg::Adds(expAddOne32, exp32, 1, maskFloat);
            AscendC::Reg::Select(extractExp, expAddOne32, exp32, predicate0);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, fp8Nan, finiteMask);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, zero, nonzeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(expOut, extractExp);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr + i * vlForHalfNumber_ / 2 / 2, expOut, maskB16);
            AscendC::Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskFloat);
            AscendC::Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, nan, finiteMask);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, zero, nonzeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(reciprocalExpOut, halfScale);
            AscendC::Reg::DataCopy<uint16_t>(halfScaleLocalAddr + i * vlForHalfNumber_ / 2, reciprocalExpOut,
                                              maskB16);
            remainingScale = remainingScale > processedScale ? remainingScale - processedScale : 0;
        }
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScaleDynamicDtypeRange(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> maxValue;
        AscendC::Reg::RegTensor<uint16_t> maxExpOnly;
        AscendC::Reg::RegTensor<uint16_t> roundedMaxExp;
        AscendC::Reg::RegTensor<uint16_t> sharedExp;
        AscendC::Reg::RegTensor<uint16_t> scaleValue;
        AscendC::Reg::RegTensor<uint16_t> reciprocalValue;
        AscendC::Reg::RegTensor<uint16_t> expMask;
        AscendC::Reg::RegTensor<uint16_t> addValue;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue;
        AscendC::Reg::RegTensor<uint16_t> scaleBias;
        AscendC::Reg::RegTensor<uint16_t> fp8Nan;
        AscendC::Reg::RegTensor<uint16_t> zero;
        AscendC::Reg::RegTensor<uint16_t> nan;
        AscendC::Reg::RegTensor<uint16_t> special;
        AscendC::Reg::MaskReg finiteMask;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg belowRangeMask;
        AscendC::Reg::MaskReg specialMask;
        AscendC::Reg::MaskReg mask;

        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::Duplicate(addValue, 0x003f);
        AscendC::Reg::Duplicate(maxExpValue, fpEmax_);
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8Nan, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zero, 0);
        AscendC::Reg::Duplicate(nan, NAN_CUSTOMIZATION);
        AscendC::Reg::Duplicate(special, SPECIAL_EXP_THRESHOLD);

        uint32_t remainingScale = totalScaleInUB;
        for (uint16_t i = 0; i < loopNumScale; ++i) {
            mask = AscendC::Reg::UpdateMask<uint16_t>(remainingScale);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxValue, maxExpAddr, vlForHalfNumber_);
            AscendC::Reg::And(maxExpOnly, maxValue, expMask, mask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(finiteMask, maxExpOnly, expMask, mask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LT>(belowRangeMask, maxExpOnly, maxExpValue, mask);
            AscendC::Reg::Add(roundedMaxExp, maxValue, addValue, mask);
            AscendC::Reg::And(roundedMaxExp, roundedMaxExp, expMask, mask);
            AscendC::Reg::Select<uint16_t>(roundedMaxExp, maxExpValue, roundedMaxExp, belowRangeMask);
            AscendC::Reg::Sub(sharedExp, roundedMaxExp, maxExpValue, mask);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, mask);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8Nan, finiteMask);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber_ >> 1, mask);

            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, sharedExp, zero, mask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialMask, sharedExp, scaleBias, mask);
            AscendC::Reg::Sub(reciprocalValue, scaleBias, sharedExp, mask);
            AscendC::Reg::Select<uint16_t>(reciprocalValue, reciprocalValue, nan, finiteMask);
            AscendC::Reg::Select<uint16_t>(reciprocalValue, reciprocalValue, zero, zeroMask);
            AscendC::Reg::Select<uint16_t>(reciprocalValue, special, reciprocalValue, specialMask);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, reciprocalValue, vlForHalfNumber_, mask);
            // UpdateMask consumes remainingScale by reference. Subtracting again
            // would skip scale registers when the tile spans multiple vectors.
        }
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    (void)totalCountInUB;
    using T = bfloat16_t;
    using U = DataTypeOut;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg maskAll =
            AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB8 =
            AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
        AscendC::Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::Reg::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;
        static constexpr AscendC::Reg::CastTrait castTraitZero = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::Reg::CastTrait castTraitOne = {
            AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::Reg::CastTrait castTrait32to80 = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait castTrait32to81 = {
            AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait castTrait32to82 = {
            AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait castTrait32to83 = {
            AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                            vlForHalfNumber_ * CGMCT_INTERLEAVED_REG_FACTOR);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                         elementAfterReduce_);

            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, maskAll);
            AscendC::Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, maskAll);
            AscendC::Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, maskAll);
            AscendC::Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, maskAll);
            AscendC::Reg::Cast<U, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::Reg::Cast<U, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::Reg::Cast<U, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::Reg::Cast<U, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8One, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t> &)vdExp1FP8Zero, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t> &)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t> &)vdExp1FP8One, maskAllB8);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp0FP8Zero,
                vlForHalfNumber_ * CGMCT_INTERLEAVED_REG_FACTOR, maskAllB8);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
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

        static constexpr AscendC::Reg::CastTrait castTrait = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                         elementAfterReduce_);

            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask2);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <Gmmsg::QuantMode quantMode>
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, uint16_t mSize, uint16_t nSize)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t OneRowRepeatTimes =
        CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat));
    // The cube always produces two ordinary N-wide MMAD results.  Mode 1
    // changes only the pairing of the two results in the vector epilogue.
    uint32_t nSrcUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned64 = Align64(nSize); // 64 对齐

    const float scalarOne = 1.0;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();

    if (unlikely(nSize % MX_QUANT_COMPUTE_ALIGN != 0)) {
        AscendC::Duplicate<bfloat16_t>(gluRes_, 0, mSize * nDstUbAligned64);
    }

    // swiglu
    __VEC_SCOPE__
    {
        if (swigluMode_ == 1) {
            // Mode 1 is defined on the original 2*N-wide GEMM result:
            //   act  = mm[:, 0::2]
            //   gate = mm[:, 1::2]
            // The CGMCT path has already split that result into two ordinary
            // N-wide MMAD buffers, so a vector de-interleave between the two
            // buffers is not correct.  Gather the even/odd elements from each
            // half separately and write the two output portions directly into
            // the final contiguous row.  This also handles a tile crossing the
            // boundary between the two MMAD buffers.
            // Arch35 Arange supports signed 32-bit lanes.  Gather for a
            // float source consumes the same bits as uint32 offsets, so the
            // index registers are reinterpreted at the Gather call site.
            AscendC::Reg::RegTensor<int32_t> laneIndex;
            AscendC::Reg::RegTensor<int32_t> evenIndex;
            AscendC::Reg::RegTensor<int32_t> oddIndex;
            AscendC::Reg::RegTensor<float> swishInput;
            AscendC::Reg::RegTensor<float> gateInput;
            AscendC::Reg::RegTensor<float> verg1, verg2, verg3, verg4, verg6, swishOutput;
            AscendC::Reg::RegTensor<bfloat16_t> verg7;
            const uint32_t halfN = static_cast<uint32_t>(nSize / 2);
            const uint32_t allElements = static_cast<uint32_t>(nSize);

            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
                const uint32_t srcRowOffset = static_cast<uint32_t>(mIdx) * nSrcUbAligned;
                const uint32_t dstRowOffset = static_cast<uint32_t>(mIdx) * nDstUbAligned64;
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                    const uint32_t outputOffset = static_cast<uint32_t>(vfBlockIdx) * sizePerRepeat;
                    const uint32_t outputEnd =
                        outputOffset + sizePerRepeat < allElements ? outputOffset + sizePerRepeat : allElements;
                    const uint32_t leftEnd = outputEnd < halfN ? outputEnd : halfN;
                    uint32_t leftCount = leftEnd > outputOffset ? leftEnd - outputOffset : 0;
                    const uint32_t rightStart = outputOffset > halfN ? outputOffset : halfN;
                    uint32_t rightCount = outputEnd > rightStart ? outputEnd - rightStart : 0;

                    if (leftCount != 0) {
                        auto activeMask = AscendC::Reg::UpdateMask<float>(leftCount);
                        AscendC::Reg::Arange(laneIndex, static_cast<int32_t>(0));
                        AscendC::Reg::Muls(evenIndex, laneIndex, static_cast<int32_t>(2), activeMask);
                        AscendC::Reg::Adds(evenIndex, evenIndex, static_cast<int32_t>(outputOffset * 2), activeMask);
                        AscendC::Reg::Adds(oddIndex, evenIndex, static_cast<int32_t>(1), activeMask);
                        AscendC::Reg::Gather(swishInput, firstSrc + srcRowOffset,
                                             (AscendC::Reg::RegTensor<uint32_t> &)evenIndex, activeMask);
                        AscendC::Reg::Gather(gateInput, firstSrc + srcRowOffset,
                                             (AscendC::Reg::RegTensor<uint32_t> &)oddIndex, activeMask);

                        AscendC::Reg::Mins(verg1, swishInput, clampLimit_, activeMask);
                        AscendC::Reg::Muls(verg2, verg1, -gluAlpha_, activeMask);
                        AscendC::Reg::Exp(verg3, verg2, activeMask);
                        AscendC::Reg::Adds(verg4, verg3, scalarOne, activeMask);
                        AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, verg1, verg4, activeMask);
                        AscendC::Reg::Mins(gateInput, gateInput, clampLimit_, activeMask);
                        AscendC::Reg::Maxs(gateInput, gateInput, -clampLimit_, activeMask);
                        AscendC::Reg::Adds(gateInput, gateInput, gluBias_, activeMask);
                        AscendC::Reg::Mul(verg6, swishOutput, gateInput, activeMask);
                        AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, activeMask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            gluResAddr + dstRowOffset + outputOffset, verg7, activeMask);
                    }

                    if (rightCount != 0) {
                        auto activeMask = AscendC::Reg::UpdateMask<float>(rightCount);
                        const uint32_t localOutputOffset = rightStart - halfN;
                        AscendC::Reg::Arange(laneIndex, static_cast<int32_t>(0));
                        AscendC::Reg::Muls(evenIndex, laneIndex, static_cast<int32_t>(2), activeMask);
                        AscendC::Reg::Adds(evenIndex, evenIndex, static_cast<int32_t>(localOutputOffset * 2), activeMask);
                        AscendC::Reg::Adds(oddIndex, evenIndex, static_cast<int32_t>(1), activeMask);
                        AscendC::Reg::Gather(swishInput, secondSrc + srcRowOffset,
                                             (AscendC::Reg::RegTensor<uint32_t> &)evenIndex, activeMask);
                        AscendC::Reg::Gather(gateInput, secondSrc + srcRowOffset,
                                             (AscendC::Reg::RegTensor<uint32_t> &)oddIndex, activeMask);

                        AscendC::Reg::Mins(verg1, swishInput, clampLimit_, activeMask);
                        AscendC::Reg::Muls(verg2, verg1, -gluAlpha_, activeMask);
                        AscendC::Reg::Exp(verg3, verg2, activeMask);
                        AscendC::Reg::Adds(verg4, verg3, scalarOne, activeMask);
                        AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, verg1, verg4, activeMask);
                        AscendC::Reg::Mins(gateInput, gateInput, clampLimit_, activeMask);
                        AscendC::Reg::Maxs(gateInput, gateInput, -clampLimit_, activeMask);
                        AscendC::Reg::Adds(gateInput, gateInput, gluBias_, activeMask);
                        AscendC::Reg::Mul(verg6, swishOutput, gateInput, activeMask);
                        AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, activeMask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            gluResAddr + dstRowOffset + rightStart, verg7, activeMask);
                    }
                }
            }
        } else {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) { // 需要计算m次
            AscendC::Reg::MaskReg mask;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) { // 每次计算m=1, n=64的数据大小
                uint32_t outputOffset = static_cast<uint32_t>(vfBlockIdx) * sizePerRepeat;
                uint32_t remaining = nSize - outputOffset;
                uint32_t elementNum = remaining < sizePerRepeat ? remaining : sizePerRepeat;
                // The vector computation is performed in the MMAD accumulator
                // type (FP32); keep this mask in the accumulator lane domain
                // so the PACK_B32 store uses the same register layout as the
                // existing MX quantization epilogues.
                mask = AscendC::Reg::UpdateMask<DataTypeIn>(elementNum);
                AscendC::Reg::RegTensor<bfloat16_t> verg7;
                AscendC::Reg::RegTensor<float> swishInput, gateInput;
                AscendC::Reg::RegTensor<float> swigluEvenInput, swigluOddInput;
                AscendC::Reg::RegTensor<float> verg1, verg2, verg3, verg4, verg6, swishOutput;

                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + outputOffset;
                AscendC::Reg::DataCopy(swishInput, firstSrc + l0cOutOffset);
                AscendC::Reg::DataCopy(gateInput, secondSrc + l0cOutOffset);

                if (swigluMode_ == 1) {
                    AscendC::Reg::DeInterleave(swigluEvenInput, swigluOddInput, swishInput, gateInput);
                    swishInput = swigluEvenInput;
                    gateInput = swigluOddInput;
                }

                // V3 modes 2/3 retain the split-half layout and add clamp/alpha/bias
                // around the activation as defined by SwigluMxQuant.
                if (swigluMode_ != 1 && !activateLeft_) {
                    auto temp = swishInput;
                    swishInput = gateInput;
                    gateInput = temp;
                }

                if (swigluMode_ == 0) {
                    AscendC::Reg::Muls(verg2, swishInput, -(scalarOne), mask);
                    AscendC::Reg::Exp(verg3, verg2, mask);
                    AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                    AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);
                } else {
                    if (swigluMode_ == 3) {
                        AscendC::Reg::Muls(verg2, swishInput, -(scalarOne), mask);
                        AscendC::Reg::Exp(verg3, verg2, mask);
                        AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                        AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);
                        AscendC::Reg::Mins(swishOutput, swishOutput, clampLimit_, mask);
                        AscendC::Reg::Mins(gateInput, gateInput, clampLimit_, mask);
                        AscendC::Reg::Maxs(gateInput, gateInput, -clampLimit_, mask);
                    } else {
                        AscendC::Reg::Mins(verg1, swishInput, clampLimit_, mask);
                        AscendC::Reg::Muls(verg2, verg1, -gluAlpha_, mask);
                        AscendC::Reg::Exp(verg3, verg2, mask);
                        AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                        AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, verg1, verg4, mask);
                        AscendC::Reg::Mins(gateInput, gateInput, clampLimit_, mask);
                        AscendC::Reg::Maxs(gateInput, gateInput, -clampLimit_, mask);
                        if (swigluMode_ == 1 || swigluMode_ == 2) {
                            AscendC::Reg::Adds(gateInput, gateInput, gluBias_, mask);
                        }
                    }
                }

                AscendC::Reg::Mul(verg6, swishOutput, gateInput, mask);

                AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, mask);
                uint32_t dstUbOffset = mIdx * nDstUbAligned64 + vfBlockIdx * sizePerRepeat;
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(gluResAddr + dstUbOffset,
                                                                                             verg7, mask);
            }
        }
        }
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned64; // nDstUbAligned64 aligned by 64
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    if (params_->scaleAlg == 0) {
        ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);
        ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    } else if (params_->scaleAlg == 1) {
        ComputeMaxExpCublas(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);
        const uint16_t dynamicLoopScaleNum = static_cast<uint16_t>(
            (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_);
        ComputeScaleDynamicDtypeRange(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb,
                                      dynamicLoopScaleNum);
    } else {
        ComputeMaxExpCublas(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);
        const uint16_t cublasLoopScaleNum =
            static_cast<uint16_t>((totalScaleInUb + vlForHalfNumber_ / 2 - 1) / (vlForHalfNumber_ / 2));
        ComputeScaleCublas(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, cublasLoopScaleNum);
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    if constexpr (IS_FP4_OUT) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(
    uint16_t mSize)
{
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr();
    __ubuf__ uint16_t *quantScaleOutputInUbAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbFirstAddr = (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    VFDoSwigluAndQuantForMX<Gmmsg::QuantMode::MX_PERGROUP_MODE>(quantOutputInUbAddr, quantScaleOutputInUbAddr,
                                                                l0cOutUbFirstAddr, l0cOutUbSecondAddr, mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::TransMxScaleLayout(
    uint16_t mSize, uint16_t scaleBlockN)
{
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr();
    // scale layout: (mSize*8) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            constexpr uint64_t scaleGroupSize = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;
            uint32_t elemNum = CeilDiv(static_cast<uint64_t>(singleN_), scaleGroupSize);
            AscendC::Reg::MaskReg maskScaleN = AscendC::Reg::UpdateMask<int8_t>(elemNum);
            AscendC::Reg::RegTensor<int8_t> vreg0;
            AscendC::Reg::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
            AscendC::Reg::DataCopyUnAlignPre(u0, srcUb);
            AscendC::Reg::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    // ComputeMxQuant uses Align64(singleN_) as its UB row pitch and emits one
    // packed scale byte for each 32 values. Keep the source row pitch
    // separate from the number of valid output scales copied to GM.
    // ComputeScale emits one byte per 32 values. The public MX scale row
    // packs two such bytes for every 64 output values, so the UB source row
    // pitch is ceil(Align64(N), 64) * 2 rather than ceil(N, 64).
    scaleBlockN_ = CeilDiv(Align64(static_cast<uint64_t>(singleN_)), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                   MXFP_MULTI_BASE_SIZE;
    blockCoord_ = blockCoord;
    auto halfSingleM = CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    UBBlockSize_ = BLOCK_SIZE;
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;

    VFDoSwigluForMX(singleMInVec);
    uint64_t yOffset = Get<Y_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * n_;
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * scaleN_;
    TransMxScaleLayout(singleMInVec, scaleBlockN_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset, quantOutput_);
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset, quantScaleBlockOutput_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    return;
}
} // namespace Block
} // namespace Gemm
} // namespace Cgmct

#endif // EPILOGUE_BLOCK_EPILOGUE_SWIGLU_QUANT_H
#endif
