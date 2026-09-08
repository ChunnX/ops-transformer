/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cla_gate_mx_quant_with_dual_axis.h
 * \brief Dual-axis ClaGate MX quantization implementation.
 */

#ifndef OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H
#define OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H

#include "cla_gate_mx_quant_common.h"

namespace ClaGateMxQuantWithDualAxis {

// Reusable CLA Gate fusion and row-axis quantization primitives are inherited
// from ClaGateMxQuantCommonBase. This file implements the dual-axis path.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
class ClaGateMxQuantWithDualAxisBase
    : public ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx> {
public:
    using Base = ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>;
    using Base::Base;
    using Base::Init;

    __aicore__ inline void Process();

protected:
    // Expose common primitives/data to this dual-axis implementation.
    using Base::ComputeClaGateFullTile;
    using Base::ComputeFP4FromHalf;
    using Base::castTraitBF16toFp4;
    using Base::castTraitFp32toBF16;
    using Base::castTraitFp32toYdtype;
    using Base::ComputeY1ToFP4;
    using Base::ComputeY1ToFP8;
    using Base::CopyInClaGate;
    using Base::InitParams;
    using Base::PadZeroM;
    using Base::actInQueue_;
    using Base::blockIdx_;
    using Base::claGateBuf_;
    using Base::dimN_;
    using Base::dimNeg1ScaleNum_;
    using Base::dtypeYMaxExp_;
    using Base::gateInQueue_;
    using Base::gateSideStride_;
    using Base::globalGateGm_;
    using Base::globalOutGm_;
    using Base::headDim_;
    using Base::headNum_;
    using Base::headsPerRow_;
    using Base::inHalfSize_;
    using Base::invDtypeMax_;
    using Base::localGateGm_;
    using Base::localOutGm_;
    using Base::mxScale1ReciprocalBuf_;
    using Base::mxScale2ReciprocalBuf_;
    using Base::mxScaleGm1_;
    using Base::mxScaleGm2_;
    using Base::mxScaleQueue1_;
    using Base::mxScaleQueue2_;
    using Base::oneBlockCountB16_;
    using Base::oneBlockCountB8_;
    using Base::outQueue1_;
    using Base::outQueue2_;
    using Base::pipe_;
    using Base::sigmoidBroadcastBuf_;
    using Base::tilingData_;
    using Base::ubRowLen_;
    using Base::ubRowLenTail_;
    using Base::ubRowCount_;
    using Base::yGm1_;
    using Base::yGm2_;

    __aicore__ inline void ComputeInterleave(__ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr,
                                             __ubuf__ uint8_t* src1Addr);
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                           __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                           __ubuf__ uint8_t* mxScale2Addr, __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint8_t* y1Addr, __ubuf__ uint8_t* mxScale1Addr,
                                              __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
                                              __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
                                                        __ubuf__ uint16_t* mxScale2ReciprocalAddr,
                                                        __ubuf__ uint8_t* mxScale2Addr);
    // DAV_3510 Reg models source RegTensor/MaskReg operands as mutable references;
    // these parameters are semantically read-only but cannot be const-qualified.
    __aicore__ inline void ComputeScaleCuBLASForSlot(
        __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, Reg::RegTensor<uint8_t>& scale8,
        Reg::RegTensor<uint32_t>& invMax, Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg,
        Reg::RegTensor<uint32_t>& zero32Reg, Reg::RegTensor<uint32_t>& scaleBiasReg, Reg::RegTensor<uint32_t>& nan32Reg,
        Reg::RegTensor<uint32_t>& fp8Nan32Reg, Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16);
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeY2ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset,
                                   int64_t blockCount, int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign);
};

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Process()
{
    dimNeg1ScaleNum_ = ops::CeilDiv(dimN_, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
    int64_t dimM = tilingData_->dimM;
    int64_t dimMSplit = ops::CeilDiv(dimM, ubRowCount_);
    int64_t dimMTail = dimM % ubRowCount_ == 0 ? ubRowCount_ : dimM % ubRowCount_;
    int64_t totalBlocks = dimMSplit * tilingData_->dimNBlockNum;
    int64_t totalCoreNum = tilingData_->usedCoreNum;
    if (blockIdx_ >= totalCoreNum || totalBlocks == 0) {
        return;
    }

    int64_t blockPerCore = totalBlocks / totalCoreNum;
    int64_t tailCore = totalBlocks % totalCoreNum;
    int64_t loops = blockPerCore + (blockIdx_ < tailCore ? 1 : 0);
    int64_t firstBlock = blockIdx_ * blockPerCore + (blockIdx_ < tailCore ? blockIdx_ : tailCore);

    for (int64_t i = 0; i < loops; ++i) {
        int64_t block = firstBlock + i;
        int64_t rowBlockIdx = block / tilingData_->dimNBlockNum;
        int64_t colBlockIdx = block % tilingData_->dimNBlockNum;
        int64_t absRowStart = rowBlockIdx * ubRowCount_;
        int64_t colOffset = colBlockIdx * ubRowLen_;
        int64_t calcRow = (rowBlockIdx == dimMSplit - 1) ? dimMTail : ubRowCount_;
        int64_t calcCol = (colBlockIdx == tilingData_->dimNBlockNum - 1) ? ubRowLenTail_ : ubRowLen_;

        int64_t headStart = colOffset / headDim_;
        int64_t headsThisTile = (headNum_ - headStart < headsPerRow_) ? (headNum_ - headStart) : headsPerRow_;
        CopyInClaGate(absRowStart, colOffset, calcRow, calcCol, headsThisTile);

        LocalTensor<xDtype> input = actInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> gateInput = gateInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> merged = claGateBuf_.template Get<xDtype>();
        auto globalAddr = (__ubuf__ xDtype*)input.GetPhyAddr();
        auto localAddr = (__ubuf__ xDtype*)input[inHalfSize_].GetPhyAddr();
        auto globalGateAddr = (__ubuf__ xDtype*)gateInput.GetPhyAddr();
        auto localGateAddr = (__ubuf__ xDtype*)gateInput[gateSideStride_].GetPhyAddr();
        auto mergedAddr = (__ubuf__ xDtype*)merged.GetPhyAddr();

        ComputeClaGateFullTile(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr,
                                static_cast<uint16_t>(calcRow), static_cast<uint16_t>(headsThisTile));

        // For T % 64 != 0, zero-fill merged tail rows after the merge stage so
        // the following 64-row scale/quantization path reads zero-padded rows.
        if (calcRow < ubRowCount_) {
            int64_t zeroRows = ubRowCount_ - calcRow;
            PadZeroM(mergedAddr + calcRow * ubRowLen_, zeroRows * ubRowLen_);
        }

        actInQueue_.template FreeTensor(input);
        gateInQueue_.template FreeTensor(gateInput);

        int64_t calcRowAlgin = ops::CeilDiv(calcRow, ubRowCount_) * ubRowCount_;
        int64_t calcColAlgin = ops::CeilDiv(calcCol, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;

        LocalTensor<uint8_t> scale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> scale2 = mxScaleQueue2_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> y2 = outQueue2_.template AllocTensor<uint8_t>();
        LocalTensor<uint16_t> scale1Recip = mxScale1ReciprocalBuf_.template Get<uint16_t>();
        LocalTensor<uint16_t> scale2Recip = mxScale2ReciprocalBuf_.template Get<uint16_t>();
        auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
        auto y2Addr = (__ubuf__ uint8_t*)y2.GetPhyAddr();
        auto scale1Addr = (__ubuf__ uint8_t*)scale1.GetPhyAddr();
        auto scale2Addr = (__ubuf__ uint8_t*)scale2.GetPhyAddr();
        auto scale1RecipAddr = (__ubuf__ uint16_t*)scale1Recip.GetPhyAddr();
        auto scale2RecipAddr = (__ubuf__ uint16_t*)scale2Recip.GetPhyAddr();

        int64_t calcBlockLoop = calcRowAlgin / BLOCK_SIZE;
        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            ComputeScaleCuBLAS(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRowAlgin), mergedAddr,
                               y1Addr, scale1Addr, scale1RecipAddr, scale2Addr, scale2RecipAddr);
        }

        for (int64_t blk = 0; blk < calcBlockLoop; blk++) {
            int64_t dataOffset = blk * BLOCK_SIZE * ubRowLen_;
            int64_t scale1Offset = blk * BLOCK_SIZE * ops::CeilAlign(ubRowLen_ / BLOCK_SIZE, oneBlockCountB8_);
            int64_t scale2Offset = blk * ubRowLen_;
            int64_t recip1Offset = blk * BLOCK_SIZE * ops::CeilAlign(ubRowLen_ / BLOCK_SIZE, oneBlockCountB16_);
            int64_t recip2Offset = blk * ubRowLen_;

            if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                ComputeScaleOcp(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                mergedAddr + dataOffset, scale1Addr + scale1Offset,
                                scale1RecipAddr + recip1Offset, scale2Addr + scale2Offset,
                                scale2RecipAddr + recip2Offset);
            }

            int64_t yDataOffset = dataOffset;
            if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
                yDataOffset = dataOffset / DIGIT_TWO;
                ComputeY1ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                               mergedAddr + dataOffset, scale1RecipAddr + recip1Offset, y1Addr + yDataOffset);
                ComputeY2ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                               mergedAddr + dataOffset, scale2RecipAddr + recip2Offset, y2Addr + yDataOffset);
                ComputeY2ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                               mergedAddr + dataOffset + VF_LEN_B16, scale2RecipAddr + recip2Offset + VF_LEN_B16,
                               y2Addr + yDataOffset + VF_LEN_B16 / DIGIT_TWO);
            } else {
                if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                    ComputeY1ToFP8(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                   mergedAddr + dataOffset, scale1RecipAddr + recip1Offset, y1Addr + yDataOffset);
                }
                ComputeY2ToFP8(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                               mergedAddr + dataOffset, scale2RecipAddr + recip2Offset, y2Addr + yDataOffset);
            }
        }

        if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
            for (int64_t blk = 1; blk < calcBlockLoop; blk += 2) {
                auto src0Addr = (__ubuf__ uint8_t*)scale2[(blk - 1) * ubRowLen_].GetPhyAddr();
                auto src1Addr = (__ubuf__ uint8_t*)scale2[blk * ubRowLen_].GetPhyAddr();
                auto dstAddr = (__ubuf__ uint8_t*)scale2[(blk - 1) * ubRowLen_].GetPhyAddr();
                ComputeInterleave(dstAddr, src0Addr, src1Addr);
            }
        }

        mxScaleQueue1_.template EnQue(scale1);
        outQueue1_.template EnQue(y1);
        mxScaleQueue2_.template EnQue(scale2);
        outQueue2_.template EnQue(y2);

        int64_t yOffset = absRowStart * dimN_ + colOffset;
        int64_t scale1Offset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
        int64_t scale2Offset = rowBlockIdx * DIGIT_TWO * dimN_ + colOffset * DIGIT_TWO;

        CopyOut(yOffset, scale1Offset, scale2Offset, calcRow, calcRowAlgin, calcCol, calcColAlgin);
    }
}


template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeInterleave(__ubuf__ uint8_t* dstAddr,
                                                                                    __ubuf__ uint8_t* src0Addr,
                                                                                    __ubuf__ uint8_t* src1Addr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint8_t> src0Reg;
        Reg::RegTensor<uint8_t> src1Reg;
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::LoadAlign(src0Reg, src0Addr);
        Reg::LoadAlign(src1Reg, src1Addr);
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(dstAddr, src0Reg, src1Reg, maskB8);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    int64_t localVlForHalfNumber = VF_LEN_B16;
    int64_t localUBBlockSize = UB_BLOCK_SIZE;
    uint16_t localDtypeYMaxExp = dtypeYMaxExp_;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<uint16_t> x0ExpFP16;
        Reg::RegTensor<uint16_t> x1ExpFP16;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<uint16_t> x0ExpBF16;
        Reg::RegTensor<uint16_t> x1ExpBF16;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::RegTensor<uint16_t> expMaskFP16;
        Reg::RegTensor<uint16_t> expMaxDim1;
        Reg::RegTensor<uint16_t> expMax1Dim2;
        Reg::RegTensor<uint16_t> expMax2Dim2;
        Reg::RegTensor<uint16_t> yMaxExp;
        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::RegTensor<uint16_t> mxScale1B16;
        Reg::RegTensor<uint8_t> mxScale1B8;
        Reg::RegTensor<uint16_t> reversedShareExp1;

        Reg::RegTensor<uint16_t> mxScale2ZeroB16;
        Reg::RegTensor<uint8_t> mxScale2ZeroB8;
        Reg::RegTensor<uint16_t> reversedShareExp2Zero;
        Reg::RegTensor<uint16_t> mxScale2OneB16;
        Reg::RegTensor<uint8_t> mxScale2OneB8;
        Reg::RegTensor<uint16_t> reversedShareExp2One;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg infNanDataMask0;
        Reg::MaskReg infNanDataMask1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(expMax1Dim2, static_cast<uint16_t>(0));
        Reg::Duplicate(expMax2Dim2, static_cast<uint16_t>(0));
        Reg::Duplicate(yMaxExp, localDtypeYMaxExp);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(zero, static_cast<uint16_t>(0));
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            // Interleaved load: splits blockW bf16/fp16 elements into even/odd halves
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       ONCE_ROW_LEN);
            if constexpr (IsSameType<xDtype, half>::value) {
                // FP16 path: extract exponent, check INF/NaN, cast to BF16
                Reg::And(x0ExpFP16, (Reg::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                Reg::And(x1ExpFP16, (Reg::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x0BF16, x0, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x1BF16, x1, maskAll);
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
                Reg::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                Reg::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                // BF16 path: extract exponent bits directly
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }

            // axis=-1: max of adjacent pair exponents
            Reg::Max(expMaxDim1, x0ExpBF16, x1ExpBF16, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMaxDim1, expMaxDim1, maskAll);

            // axis=-2: accumulate column-wise max exponents across rows
            Reg::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            Reg::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);

            // ---- axis=-1 scale computation ----
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim1, expMaskBF16, maskAll);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim1, zero, maskAll);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim1, yMaxExp, maskAll);
            Reg::Select<uint16_t>(expMaxDim1, yMaxExp, expMaxDim1, invalidDataMask);

            Reg::Sub(expMaxDim1, expMaxDim1, yMaxExp, maskAll);
            Reg::ShiftRights(mxScale1B16, expMaxDim1, SHR_NUM_FOR_BF16, maskAll);
            Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
            Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);

            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, mxScale1B8, UB_BLOCK_SIZE,
                                                                         maskReduceB8);
            // Compute 1/scale
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            Reg::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1ReciprocalAddr, reversedShareExp1,
                                                                          SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
        }

        // ---- axis=-2 scale computation (interleaved part 1: even rows) ----
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax1Dim2, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax1Dim2, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax1Dim2, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMax1Dim2, yMaxExp, expMax1Dim2, invalidDataMask);
        Reg::Sub(expMax1Dim2, expMax1Dim2, yMaxExp, maskAll);
        Reg::ShiftRights(mxScale2ZeroB16, expMax1Dim2, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax1Dim2, biasE8M0, maskAll); // scale计算结束

        Reg::Sub(reversedShareExp2Zero, biasE8M0, expMax1Dim2, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero,
                              invalidDataMask); // int16 scale 结束

        // ---- axis=-2 scale computation (interleaved part 2: odd rows) ----
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax2Dim2, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax2Dim2, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax2Dim2, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMax2Dim2, yMaxExp, expMax2Dim2, invalidDataMask);
        Reg::Sub(expMax2Dim2, expMax2Dim2, yMaxExp, maskAll);
        Reg::ShiftRights(mxScale2OneB16, expMax2Dim2, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);

        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax2Dim2, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp2One, biasE8M0, expMax2Dim2, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);

        // Interleaved store: merge even/odd scale and 1/scale for axis=-2
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale2ZeroB8, mxScale2OneB8, maskB8);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, reversedShareExp2Zero,
                                                                  reversedShareExp2One, maskAll);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLASForSlot(
    __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, Reg::RegTensor<uint8_t>& scale8,
    Reg::RegTensor<uint32_t>& invMax, Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg,
    Reg::RegTensor<uint32_t>& zero32Reg, Reg::RegTensor<uint32_t>& scaleBiasReg, Reg::RegTensor<uint32_t>& nan32Reg,
    Reg::RegTensor<uint32_t>& fp8Nan32Reg, Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16)
{
    Reg::RegTensor<uint16_t> max16Reg;
    Reg::RegTensor<uint32_t> max32Reg;
    Reg::RegTensor<uint32_t> exp32Reg;
    Reg::RegTensor<uint32_t> man32Reg;
    Reg::RegTensor<uint32_t> expOne32Reg;
    Reg::RegTensor<uint32_t> extractExp;
    Reg::RegTensor<uint32_t> halfScale;
    Reg::RegTensor<uint16_t> scale16;
    Reg::RegTensor<uint16_t> recip16;
    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg p0;
    Reg::MaskReg p1;

    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(max16Reg, maxReadAddr,
                                                                                                 VF_LEN_FP32);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<xDtype>&)max16Reg,
                                                  maskAll);
    Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
    Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
    Reg::Mul((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)invMax,
             maskAll32);
    Reg::ShiftRights(exp32Reg, max32Reg, SHR_NUM_FOR_FP32, maskAll32);
    Reg::And(man32Reg, max32Reg, manMaskReg, maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32Reg, EXP_254, p0);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32Reg, static_cast<uint32_t>(0), p0);
    Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32Reg, HALF_FOR_MAN, p1);
    Reg::Or(p0, p0, p1, maskAll32);
    Reg::Adds(expOne32Reg, exp32Reg, 1, maskAll32);
    Reg::Select(extractExp, expOne32Reg, exp32Reg, p0);
    Reg::Select<uint32_t>(extractExp, extractExp, fp8Nan32Reg, cmpResult);
    Reg::Select<uint32_t>(extractExp, extractExp, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale16, extractExp);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8, scale16);

    Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
    Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
    Reg::Select<uint32_t>(halfScale, halfScale, nan32Reg, cmpResult);
    Reg::Select<uint32_t>(halfScale, halfScale, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16, halfScale);
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(reciprocalWriteAddr, recip16, VF_LEN_FP32, maskB16);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLASSecondLast(
    uint16_t dataLen, uint32_t localInvDtypeMax, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* mxScale2Addr)
{
    uint16_t times = dataLen / VF_LEN_FP32;
    PipeBarrier<PIPE_V>();
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<uint32_t> manMaskReg;
        Reg::RegTensor<uint32_t> expMaskReg;
        Reg::RegTensor<uint32_t> zero32Reg;
        Reg::RegTensor<uint32_t> scaleBiasReg;
        Reg::RegTensor<uint32_t> nan32Reg;
        Reg::RegTensor<uint32_t> fp8Nan32Reg;
        Reg::RegTensor<uint8_t> scale8Slot0;
        Reg::RegTensor<uint8_t> scale8Slot1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::VL64>();
        Reg::MaskReg interleaveMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zero32Reg, static_cast<uint32_t>(0));
        Reg::Duplicate(invMax, localInvDtypeMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(fp8Nan32Reg, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::Duplicate(nan32Reg, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        for (uint16_t i = 0; i < times; i++) {
            uint16_t colOffset = i * VF_LEN_FP32;
            __ubuf__ uint16_t* slot0Addr = mxScale2ReciprocalAddr + colOffset;
            __ubuf__ uint16_t* slot1Addr = mxScale2ReciprocalAddr + dataLen + colOffset;
            ComputeScaleCuBLASForSlot(slot0Addr, slot0Addr, scale8Slot0, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            ComputeScaleCuBLASForSlot(slot1Addr, slot1Addr, scale8Slot1, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr + DIGIT_TWO * colOffset, scale8Slot0,
                                                                    scale8Slot1, interleaveMask);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLAS(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* y1Addr,
    __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    uint32_t localInvDtypeMax = invDtypeMax_;
    __ubuf__ uint16_t* tempMaxAddr = mxScale2ReciprocalAddr;
    __ubuf__ xDtype* xAddrBase = xAddr;
    __ubuf__ uint16_t* reciprocalBase = mxScale1ReciprocalAddr;
    PipeBarrier<PIPE_V>();
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<uint16_t> x0Abs;
        Reg::RegTensor<uint16_t> x1Abs;
        Reg::RegTensor<uint16_t> absMaxDim1;
        Reg::RegTensor<uint16_t> scale2Slot0Part0;
        Reg::RegTensor<uint16_t> scale2Slot0Part1;
        Reg::RegTensor<uint16_t> scale2Slot1Part0;
        Reg::RegTensor<uint16_t> scale2Slot1Part1;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> scale16;
        Reg::RegTensor<uint8_t> scale8Reg;
        Reg::RegTensor<uint8_t> scale8Row;
        Reg::RegTensor<uint16_t> recip16Reg;
        Reg::RegTensor<uint16_t> recip16Row;
        Reg::RegTensor<int8_t> extractIdx;
        Reg::RegTensor<uint16_t> absMask;
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<uint32_t> manMaskReg;
        Reg::RegTensor<uint32_t> expMaskReg;
        Reg::RegTensor<uint32_t> zeroReg32;
        Reg::RegTensor<uint32_t> scaleBiasReg;
        Reg::RegTensor<uint32_t> nanReg32;
        Reg::RegTensor<uint32_t> fp8NanReg32;
        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();
        Reg::UnalignReg ureg;

        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        Reg::Duplicate(scale2Slot0Part0, static_cast<uint16_t>(0));
        Reg::Duplicate(scale2Slot0Part1, static_cast<uint16_t>(0));
        Reg::Duplicate(scale2Slot1Part0, static_cast<uint16_t>(0));
        Reg::Duplicate(scale2Slot1Part1, static_cast<uint16_t>(0));

        // Phase 1: collect scale1 maxima compactly while preserving perf/all's
        // two independent 32-row scale2 slots.
        uint16_t slot0Rows = blockCount < BLOCK_SIZE ? blockCount : BLOCK_SIZE;
        for (uint16_t i = 0; i < slot0Rows; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       dataLen);
            Reg::And(x0Abs, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(x1Abs, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            Reg::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            Reg::Max(scale2Slot0Part0, scale2Slot0Part0, x0Abs, maskAll);
            Reg::Max(scale2Slot0Part1, scale2Slot0Part1, x1Abs, maskAll);
        }
        for (uint16_t i = slot0Rows; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       dataLen);
            Reg::And(x0Abs, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(x1Abs, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            Reg::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            Reg::Max(scale2Slot1Part0, scale2Slot1Part0, x0Abs, maskAll);
            Reg::Max(scale2Slot1Part1, scale2Slot1Part1, x1Abs, maskAll);
        }
        Reg::StoreUnAlignPost(tempMaxAddr, ureg, 0);

        Reg::Duplicate(invMax, localInvDtypeMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

        // Phase 2: convert 64 cached maxima per VF, then scatter eight rows
        // back to the original 32-byte-aligned scale/reciprocal layout.
        __ubuf__ uint16_t* readMaxAddr = mxScale2ReciprocalAddr;
        uint16_t scaleCount = dataLen / BLOCK_SIZE;
        uint16_t batchCount = ops::CeilDiv(static_cast<uint16_t>(blockCount * scaleCount),
                                           static_cast<uint16_t>(VF_LEN_FP32));
        for (uint16_t j = 0; j < batchCount; j++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                absMaxDim1, readMaxAddr, VF_LEN_FP32);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32,
                                                          (Reg::RegTensor<xDtype>&)absMaxDim1, maskAll);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMaskReg, maskAll32);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroReg32, maskAll32);
            Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)invMax,
                     maskAll32);
            Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskAll32);
            Reg::And(man32, max32, manMaskReg, maskAll32);
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32, static_cast<uint32_t>(0), maskAll32);
            Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32, EXP_254, p0);
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32, static_cast<uint32_t>(0), p0);
            Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32, static_cast<uint32_t>(0), maskAll32);
            Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32, HALF_FOR_MAN, p1);
            Reg::Or(p0, p0, p1, maskAll32);
            Reg::Adds(expAddOne32, exp32, 1, maskAll32);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp8NanReg32, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroReg32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale16, extractExp);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8Reg, scale16);
            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
            Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
            Reg::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16Reg, halfScale);

            for (uint16_t k = 0; k < 8; k++) {
                Reg::Arange(extractIdx, static_cast<int8_t>(k * 8));
                Reg::Gather(scale8Row, scale8Reg, (Reg::RegTensor<uint8_t>&)extractIdx);
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, scale8Row, UB_BLOCK_SIZE,
                                                                             maskReduceB8);
                Reg::Arange(extractIdx, static_cast<int8_t>(k * 16));
                Reg::Gather((Reg::RegTensor<uint8_t>&)recip16Row, (Reg::RegTensor<uint8_t>&)recip16Reg,
                            (Reg::RegTensor<uint8_t>&)extractIdx);
                Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    mxScale1ReciprocalAddr, recip16Row, SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
            }
        }

        // Generate y1 immediately after scale1 while staying in the same vector scope.
        Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<y1Dtype> x0ZeroFP8;
        Reg::RegTensor<y1Dtype> x0OneFP8;
        Reg::RegTensor<y1Dtype> x1ZeroFP8;
        Reg::RegTensor<y1Dtype> x1OneFP8;
        __ubuf__ xDtype* xReadAddr = xAddrBase;
        __ubuf__ uint16_t* recipReadAddr = reciprocalBase;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xReadAddr, dataLen);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, recipReadAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                Reg::Mul(x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Mul(x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            Reg::Cast<y1Dtype, float, CAST_32_TO_80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            Reg::Cast<y1Dtype, float, CAST_32_TO_82>(x0OneFP8, x0OneFP32, maskAll);
            Reg::Cast<y1Dtype, float, CAST_32_TO_81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            Reg::Cast<y1Dtype, float, CAST_32_TO_83>(x1OneFP8, x1OneFP32, maskAll);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x0OneFP8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1ZeroFP8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1OneFP8, maskAllB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0ZeroFP8, dataLen, maskAllB8);
        }

        // Reuse the temporary area for the normal axis=-2 maxima output.
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, scale2Slot0Part0,
                                                                  scale2Slot0Part1, maskAll);
        if (blockCount > BLOCK_SIZE) {
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr + dataLen,
                                                                      scale2Slot1Part0, scale2Slot1Part1, maskAll);
        }
    }
    ComputeScaleCuBLASSecondLast(dataLen, localInvDtypeMax, mxScale2ReciprocalAddr, mxScale2Addr);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY2ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
    int64_t localUbRowLen = ubRowLen_;
    constexpr uint32_t dualLoadLen = VF_LEN_B16 * DIGIT_TWO;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> xEven;
        Reg::RegTensor<xDtype> xOdd;
        Reg::RegTensor<uint16_t> reciprocalEven;
        Reg::RegTensor<uint16_t> reciprocalOdd;
        Reg::RegTensor<float> reciprocalEvenFP32Layout0;
        Reg::RegTensor<float> reciprocalEvenFP32Layout1;
        Reg::RegTensor<float> reciprocalOddFP32Layout0;
        Reg::RegTensor<float> reciprocalOddFP32Layout1;
        Reg::RegTensor<float> xEvenFP32Layout0;
        Reg::RegTensor<float> xEvenFP32Layout1;
        Reg::RegTensor<float> xOddFP32Layout0;
        Reg::RegTensor<float> xOddFP32Layout1;
        Reg::RegTensor<y1Dtype> yEvenFP8Layout0;
        Reg::RegTensor<y1Dtype> yEvenFP8Layout2;
        Reg::RegTensor<y1Dtype> yOddFP8Layout1;
        Reg::RegTensor<y1Dtype> yOddFP8Layout3;

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        // Load all 256 reciprocal values once and reuse them for all 32 rows in this scale2 slot.
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            reciprocalEven, reciprocalOdd, mxScale2ReciprocalAddr, dualLoadLen);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalEvenFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalEvenFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalOddFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalOddFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
        }

        for (uint16_t row = 0; row < blockCount; row++) {
            __ubuf__ xDtype* xCursor = xAddr + row * localUbRowLen;
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xEven, xOdd, xCursor, dualLoadLen);
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);

                Reg::Mul(xEvenFP32Layout0, xEvenFP32Layout0, reciprocalEvenFP32Layout0, maskB32);
                Reg::Mul(xEvenFP32Layout1, xEvenFP32Layout1, reciprocalEvenFP32Layout1, maskB32);
                Reg::Mul(xOddFP32Layout0, xOddFP32Layout0, reciprocalOddFP32Layout0, maskB32);
                Reg::Mul(xOddFP32Layout1, xOddFP32Layout1, reciprocalOddFP32Layout1, maskB32);
            } else {
                Reg::Mul(xEven, xEven, (Reg::RegTensor<xDtype>&)reciprocalEven, maskB16);
                Reg::Mul(xOdd, xOdd, (Reg::RegTensor<xDtype>&)reciprocalOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);
            }

            Reg::Cast<y1Dtype, float, CAST_32_TO_80>(yEvenFP8Layout0, xEvenFP32Layout0, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_82>(yEvenFP8Layout2, xEvenFP32Layout1, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_81>(yOddFP8Layout1, xOddFP32Layout0, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_83>(yOddFP8Layout3, xOddFP32Layout1, maskB32);

            Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t>&)yEvenFP8Layout2, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)yOddFP8Layout1, (Reg::RegTensor<uint8_t>&)yOddFP8Layout1,
                     (Reg::RegTensor<uint8_t>&)yOddFP8Layout3, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t>&)yOddFP8Layout1, maskB8);

            __ubuf__ uint8_t* yCursor = y2Addr + row * localUbRowLen;
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yCursor, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, dualLoadLen, maskB8);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY2ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
    int64_t localUbRowLen = ubRowLen_;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<bfloat16_t> xBF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> reversedShareExp;
        Reg::RegTensor<float> reversedShareExp0FP32;
        Reg::RegTensor<float> reversedShareExp1FP32;
        Reg::RegTensor<y1Dtype> yZeroFP4;

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        // Load per-column 1/scale
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);

        for (uint16_t j = 0; j < blockCount; j++) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * localUbRowLen);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0FP32, x, pregAll16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1FP32, x, pregAll16);

                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    reversedShareExp0FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(
                    reversedShareExp1FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

                Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                Reg::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

                ComputeFP4FromHalf(x0FP32);
                ComputeFP4FromHalf(x1FP32);

                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t>&)x0BF16, x0FP32,
                                                                  pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0BF16);

                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t>&)x1BF16, x1FP32,
                                                                  pregAll32);

                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1BF16);

                Reg::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                                   pregAll16);
            } else {
                Reg::Mul(xBF16, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4, xBF16, pregAll16);
            }

            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(y2Addr + (j * 128),
                                                                     (Reg::RegTensor<uint8_t>&)yZeroFP4, pregAll8);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::CopyOut(
    int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset, int64_t blockCount, int64_t blockCountAlign,
    int64_t dataLen, int64_t dataLenAlign)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;

    int64_t yOffsetNow = yOffset;

    // axis=-2 two rows interleaved, accounting for 32B alignment
    uint32_t scaleSrcStride = DIGIT_TWO * ops::CeilDiv(dataLen, UB_BLOCK_SIZE) -
                              ops::CeilDiv(DIGIT_TWO * dataLen, UB_BLOCK_SIZE);

    if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        // FP4: two fp4 values packed into one byte
        outBlockLen = static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((ubRowLen_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimN_ - dataLen) / DIGIT_TWO * sizeof(uint8_t));
        yOffsetNow = yOffset / DIGIT_TWO;
    } else {
        // FP8: one byte per element
        outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(y1Dtype) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimN_ - dataLen) * sizeof(uint8_t));
    }

    DataCopyExtParams yCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};

    // axis=-1 scale output: shape [M, ceil(N/blockSize)]
    uint32_t scale1OutLen = dataLenAlign / BLOCK_SIZE;

    DataCopyExtParams scale1CopyOutParams = {
        outBurst, static_cast<uint32_t>(scale1OutLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(dimN_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - scale1OutLen),
        static_cast<uint32_t>(0)};

    // axis=-2 scale output: two rows interleaved per blockSize group
    DataCopyExtParams scale2CopyOutParams = {
        static_cast<uint16_t>(blockCountAlign / DOUBLE_BLOCK_SIZE),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>(DIGIT_TWO * (dimN_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)};

    // Dequeue and copy y1
    LocalTensor<uint8_t> y1Local = outQueue1_.template DeQue<uint8_t>();
    DataCopyPad(yGm1_[yOffsetNow], y1Local, yCopyOutParams);
    outQueue1_.FreeTensor(y1Local);

    // Dequeue and copy y2
    LocalTensor<uint8_t> y2Local = outQueue2_.template DeQue<uint8_t>();
    DataCopyPad(yGm2_[yOffsetNow], y2Local, yCopyOutParams);
    outQueue2_.FreeTensor(y2Local);

    // Dequeue and copy scale1 (axis=-1)
    LocalTensor<uint8_t> mxScale1Local = mxScaleQueue1_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm1_[scale1OutOffset], mxScale1Local, scale1CopyOutParams);
    mxScaleQueue1_.FreeTensor(mxScale1Local);

    // Dequeue and copy scale2 (axis=-2)
    LocalTensor<uint8_t> mxScale2Local = mxScaleQueue2_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm2_[scale2OutOffset], mxScale2Local, scale2CopyOutParams);
    mxScaleQueue2_.FreeTensor(mxScale2Local);
}
} // namespace ClaGateMxQuantWithDualAxis

#endif // OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H
