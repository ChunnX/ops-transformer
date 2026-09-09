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
template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class ClaGateMxQuantWithDualAxisBase
    : public ClaGateMxQuantCommonBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg> {
public:
    using Base = ClaGateMxQuantCommonBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>;
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
    using Base::ComputeRowDataToFp4;
    using Base::ComputeRowDataToFp8;
    using Base::CopyInClaGate;
    using Base::InitParams;
    using Base::FillZero;
    using Base::activationInQueue_;
    using Base::blockIdx_;
    using Base::mergedClaGateBuf_;
    using Base::rowLength_;
    using Base::rowScaleElementsPerRow_;
    using Base::outputDtypeMaxExp_;
    using Base::gateInQueue_;
    using Base::gateSideOffset_;
    using Base::globalGateGm_;
    using Base::globalOutGm_;
    using Base::headSize_;
    using Base::headCount_;
    using Base::headsPerTileRow_;
    using Base::halfBufferElemCount_;
    using Base::outputDtypeInvMax_;
    using Base::localGateGm_;
    using Base::localOutGm_;
    using Base::rowScaleReciprocalBuf_;
    using Base::colScaleReciprocalBuf_;
    using Base::rowScaleGm_;
    using Base::colScaleGm_;
    using Base::rowScaleQueue_;
    using Base::colScaleQueue_;
    using Base::ubBlockElemCountB16_;
    using Base::ubBlockElemCountB8_;
    using Base::rowDataQueue_;
    using Base::colDataQueue_;
    using Base::pipe_;
    using Base::sigmoidBroadcastBuf_;
    using Base::tilingData_;
    using Base::tileRowLength_;
    using Base::tileTailLength_;
    using Base::tileRowCount_;
    using Base::rowDataGm_;
    using Base::colDataGm_;

    __aicore__ inline void ComputeInterleave(__ubuf__ uint8_t *dstAddr, __ubuf__ uint8_t *src0Addr,
                                             __ubuf__ uint8_t *src1Addr);
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                           __ubuf__ uint8_t *rowScaleAddr, __ubuf__ uint16_t *rowScaleReciprocalAddr,
                                           __ubuf__ uint8_t *colScaleAddr, __ubuf__ uint16_t *colScaleReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                              __ubuf__ uint8_t *rowDataAddr, __ubuf__ uint8_t *rowScaleAddr,
                                              __ubuf__ uint16_t *rowScaleReciprocalAddr, __ubuf__ uint8_t *colScaleAddr,
                                              __ubuf__ uint16_t *colScaleReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t outputDtypeInvMax,
                                                        __ubuf__ uint16_t *colScaleReciprocalAddr,
                                                        __ubuf__ uint8_t *colScaleAddr);
    // DAV_3510 Reg models source RegTensor/MaskReg operands as mutable references;
    // these parameters are semantically read-only but cannot be const-qualified.
    __aicore__ inline void ComputeScaleCuBLASForSlot(
        __ubuf__ uint16_t *madataReadAddr, __ubuf__ uint16_t *reciprocalWriteAddr, Reg::RegTensor<uint8_t> &scale8,
        Reg::RegTensor<uint32_t> &invMax, Reg::RegTensor<uint32_t> &manMaskReg, Reg::RegTensor<uint32_t> &expMaskReg,
        Reg::RegTensor<uint32_t> &zero32Reg, Reg::RegTensor<uint32_t> &scaleBiasReg, Reg::RegTensor<uint32_t> &nan32Reg,
        Reg::RegTensor<uint32_t> &fp8Nan32Reg, Reg::MaskReg &maskAll, Reg::MaskReg &maskAll32, Reg::MaskReg &maskB16);
    __aicore__ inline void ComputeColDataToFp8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                               __ubuf__ uint16_t *colScaleReciprocalAddr,
                                               __ubuf__ uint8_t *colDataAddr);
    __aicore__ inline void ComputeColDataToFp4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                               __ubuf__ uint16_t *colScaleReciprocalAddr,
                                               __ubuf__ uint8_t *colDataAddr);
    __aicore__ inline void CopyOut(int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t colScaleOutOffset,
                                   int64_t blockCount, int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign);
};

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::Process()
{
    rowScaleElementsPerRow_ = ops::CeilDiv(rowLength_, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
    int64_t dimM = tilingData_->dimM;
    int64_t dimMSplit = ops::CeilDiv(dimM, tileRowCount_);
    int64_t dimMTail = dimM % tileRowCount_ == 0 ? tileRowCount_ : dimM % tileRowCount_;
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
        int64_t absRowStart = rowBlockIdx * tileRowCount_;
        int64_t colOffset = colBlockIdx * tileRowLength_;
        int64_t calcRow = (rowBlockIdx == dimMSplit - 1) ? dimMTail : tileRowCount_;
        int64_t calcCol = (colBlockIdx == tilingData_->dimNBlockNum - 1) ? tileTailLength_ : tileRowLength_;

        int64_t headStart = colOffset / headSize_;
        int64_t headsThisTile =
            (headCount_ - headStart < headsPerTileRow_) ? (headCount_ - headStart) : headsPerTileRow_;
        CopyInClaGate(absRowStart, colOffset, calcRow, calcCol, headsThisTile);

        LocalTensor<xDtype> activationInput = activationInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> gateInput = gateInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> mergedResult = mergedClaGateBuf_.template Get<xDtype>();
        auto globalAddr = (__ubuf__ xDtype *)activationInput.GetPhyAddr();
        auto localAddr = (__ubuf__ xDtype *)activationInput[halfBufferElemCount_].GetPhyAddr();
        auto globalGateAddr = (__ubuf__ xDtype *)gateInput.GetPhyAddr();
        auto localGateAddr = (__ubuf__ xDtype *)gateInput[gateSideOffset_].GetPhyAddr();
        auto mergedAddr = (__ubuf__ xDtype *)mergedResult.GetPhyAddr();

        ComputeClaGateFullTile(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr,
                               static_cast<uint16_t>(calcRow), static_cast<uint16_t>(headsThisTile));

        // For T % 64 != 0, zero-fill merged tail rows after the merge stage so
        // the following 64-row scale/quantization path reads zero-padded rows.
        if (calcRow < tileRowCount_) {
            int64_t zeroRows = tileRowCount_ - calcRow;
            FillZero(mergedAddr + calcRow * tileRowLength_, zeroRows * tileRowLength_);
        }

        activationInQueue_.template FreeTensor(activationInput);
        gateInQueue_.template FreeTensor(gateInput);

        int64_t calcRowAligned = ops::CeilDiv(calcRow, tileRowCount_) * tileRowCount_;
        int64_t calcColAligned = ops::CeilDiv(calcCol, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;

        LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> colScaleTensor = colScaleQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> colDataTensor = colDataQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
        LocalTensor<uint16_t> colScaleReciprocalTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
        auto rowDataAddr = (__ubuf__ uint8_t *)rowDataTensor.GetPhyAddr();
        auto colDataAddr = (__ubuf__ uint8_t *)colDataTensor.GetPhyAddr();
        auto rowScaleAddr = (__ubuf__ uint8_t *)rowScaleTensor.GetPhyAddr();
        auto colScaleAddr = (__ubuf__ uint8_t *)colScaleTensor.GetPhyAddr();
        auto rowScaleReciprocalAddr = (__ubuf__ uint16_t *)rowScaleReciprocalTensor.GetPhyAddr();
        auto colScaleReciprocalAddr = (__ubuf__ uint16_t *)colScaleReciprocalTensor.GetPhyAddr();

        int64_t calcBlockLoop = calcRowAligned / BLOCK_SIZE;
        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            ComputeScaleCuBLAS(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRowAligned), mergedAddr,
                               rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr, colScaleAddr, colScaleReciprocalAddr);
        }

        for (int64_t blk = 0; blk < calcBlockLoop; blk++) {
            int64_t dataOffset = blk * BLOCK_SIZE * tileRowLength_;
            int64_t rowScaleOutOffset =
                blk * BLOCK_SIZE * ops::CeilAlign(tileRowLength_ / BLOCK_SIZE, ubBlockElemCountB8_);
            int64_t colScaleOutOffset = blk * tileRowLength_;
            int64_t rowScaleReciprocalOffset =
                blk * BLOCK_SIZE * ops::CeilAlign(tileRowLength_ / BLOCK_SIZE, ubBlockElemCountB16_);
            int64_t colScaleReciprocalOffset = blk * tileRowLength_;

            if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                ComputeScaleOcp(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(BLOCK_SIZE),
                                mergedAddr + dataOffset, rowScaleAddr + rowScaleOutOffset,
                                rowScaleReciprocalAddr + rowScaleReciprocalOffset, colScaleAddr + colScaleOutOffset,
                                colScaleReciprocalAddr + colScaleReciprocalOffset);
            }

            int64_t packedDataOffset = dataOffset;
            if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value ||
                          IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
                packedDataOffset = dataOffset / DIGIT_TWO;
                ComputeRowDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(BLOCK_SIZE),
                                    mergedAddr + dataOffset, rowScaleReciprocalAddr + rowScaleReciprocalOffset,
                                    rowDataAddr + packedDataOffset);
                ComputeColDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(BLOCK_SIZE),
                                    mergedAddr + dataOffset, colScaleReciprocalAddr + colScaleReciprocalOffset,
                                    colDataAddr + packedDataOffset);
                ComputeColDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(BLOCK_SIZE),
                                    mergedAddr + dataOffset + VF_LEN_B16,
                                    colScaleReciprocalAddr + colScaleReciprocalOffset + VF_LEN_B16,
                                    colDataAddr + packedDataOffset + VF_LEN_B16 / DIGIT_TWO);
            } else {
                if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                    ComputeRowDataToFp8(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(BLOCK_SIZE),
                                        mergedAddr + dataOffset, rowScaleReciprocalAddr + rowScaleReciprocalOffset,
                                        rowDataAddr + packedDataOffset);
                }
                ComputeColDataToFp8(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(BLOCK_SIZE),
                                    mergedAddr + dataOffset, colScaleReciprocalAddr + colScaleReciprocalOffset,
                                    colDataAddr + packedDataOffset);
            }
        }

        if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
            for (int64_t blk = 1; blk < calcBlockLoop; blk += 2) {
                auto src0Addr = (__ubuf__ uint8_t *)colScaleTensor[(blk - 1) * tileRowLength_].GetPhyAddr();
                auto src1Addr = (__ubuf__ uint8_t *)colScaleTensor[blk * tileRowLength_].GetPhyAddr();
                auto dstAddr = (__ubuf__ uint8_t *)colScaleTensor[(blk - 1) * tileRowLength_].GetPhyAddr();
                ComputeInterleave(dstAddr, src0Addr, src1Addr);
            }
        }

        rowScaleQueue_.template EnQue(rowScaleTensor);
        rowDataQueue_.template EnQue(rowDataTensor);
        colScaleQueue_.template EnQue(colScaleTensor);
        colDataQueue_.template EnQue(colDataTensor);

        int64_t rowDataOutOffset = absRowStart * rowLength_ + colOffset;
        int64_t rowScaleOutOffset = absRowStart * rowScaleElementsPerRow_ + colOffset / BLOCK_SIZE;
        int64_t colScaleOutOffset = rowBlockIdx * DIGIT_TWO * rowLength_ + colOffset * DIGIT_TWO;

        CopyOut(rowDataOutOffset, rowScaleOutOffset, colScaleOutOffset, calcRow, calcRowAligned, calcCol,
                calcColAligned);
    }
}

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode,
                                                      scaleAlg>::ComputeInterleave(__ubuf__ uint8_t *dstAddr,
                                                                                   __ubuf__ uint8_t *src0Addr,
                                                                                   __ubuf__ uint8_t *src1Addr)
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

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *rowScaleAddr,
    __ubuf__ uint16_t *rowScaleReciprocalAddr, __ubuf__ uint8_t *colScaleAddr,
    __ubuf__ uint16_t *colScaleReciprocalAddr)
{
    uint16_t outputDtypeMaxExp = outputDtypeMaxExp_;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> evenData;
        Reg::RegTensor<xDtype> oddData;
        Reg::RegTensor<uint16_t> evenExpFp16;
        Reg::RegTensor<uint16_t> oddExpFp16;
        Reg::RegTensor<bfloat16_t> evenBf16;
        Reg::RegTensor<bfloat16_t> oddBf16;
        Reg::RegTensor<uint16_t> evenExpBf16;
        Reg::RegTensor<uint16_t> oddExpBf16;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::RegTensor<uint16_t> expMaskFP16;
        Reg::RegTensor<uint16_t> expMaxDim1;
        Reg::RegTensor<uint16_t> colEvenMaxExp;
        Reg::RegTensor<uint16_t> colOddMaxExp;
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
        Reg::MaskReg evenInfNanMask;
        Reg::MaskReg oddInfNanMask;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(colEvenMaxExp, static_cast<uint16_t>(0));
        Reg::Duplicate(colOddMaxExp, static_cast<uint16_t>(0));
        Reg::Duplicate(yMaxExp, outputDtypeMaxExp);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(zero, static_cast<uint16_t>(0));
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            // Interleaved load: splits blockW bf16/fp16 elements into even/odd halves
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, ONCE_ROW_LEN);
            if constexpr (IsSameType<xDtype, half>::value) {
                // FP16 path: extract exponent, check INF/NaN, cast to BF16
                Reg::And(evenExpFp16, (Reg::RegTensor<uint16_t> &)evenData, expMaskFP16, maskAll);
                Reg::And(oddExpFp16, (Reg::RegTensor<uint16_t> &)oddData, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(evenInfNanMask, evenExpFp16, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(oddInfNanMask, oddExpFp16, expMaskFP16, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(evenBf16, evenData, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(oddBf16, oddData, maskAll);
                Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t> &)evenBf16, expMaskBF16, maskAll);
                Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t> &)oddBf16, expMaskBF16, maskAll);
                Reg::Select<uint16_t>(evenExpBf16, evenExpBf16, expMaskBF16, evenInfNanMask);
                Reg::Select<uint16_t>(oddExpBf16, oddExpBf16, expMaskBF16, oddInfNanMask);
            } else {
                // BF16 path: extract exponent bits directly
                Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t> &)evenData, expMaskBF16, maskAll);
                Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t> &)oddData, expMaskBF16, maskAll);
            }

            // axis=-1: max of adjacent pair exponents
            Reg::Max(expMaxDim1, evenExpBf16, oddExpBf16, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMaxDim1, expMaxDim1, maskAll);

            // axis=-2: accumulate column-wise max exponents across rows
            Reg::Max(colEvenMaxExp, colEvenMaxExp, evenExpBf16, maskAll);
            Reg::Max(colOddMaxExp, colOddMaxExp, oddExpBf16, maskAll);

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
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(rowScaleAddr, mxScale1B8, UB_BLOCK_SIZE,
                                                                         maskReduceB8);
            // Compute 1/scale
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            Reg::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                rowScaleReciprocalAddr, reversedShareExp1, ROW_SCALE_RECIPROCAL_ROW_ELEMS, maskReduceB16);
        }

        // ---- axis=-2 scale computation (interleaved part 1: even rows) ----
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, colEvenMaxExp, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, colEvenMaxExp, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, colEvenMaxExp, yMaxExp, maskAll);
        Reg::Select<uint16_t>(colEvenMaxExp, yMaxExp, colEvenMaxExp, invalidDataMask);
        Reg::Sub(colEvenMaxExp, colEvenMaxExp, yMaxExp, maskAll);
        Reg::ShiftRights(mxScale2ZeroB16, colEvenMaxExp, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, colEvenMaxExp, biasE8M0, maskAll); // scale计算结束

        Reg::Sub(reversedShareExp2Zero, biasE8M0, colEvenMaxExp, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero,
                              invalidDataMask); // int16 scale 结束

        // ---- axis=-2 scale computation (interleaved part 2: odd rows) ----
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, colOddMaxExp, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, colOddMaxExp, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, colOddMaxExp, yMaxExp, maskAll);
        Reg::Select<uint16_t>(colOddMaxExp, yMaxExp, colOddMaxExp, invalidDataMask);
        Reg::Sub(colOddMaxExp, colOddMaxExp, yMaxExp, maskAll);
        Reg::ShiftRights(mxScale2OneB16, colOddMaxExp, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);

        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, colOddMaxExp, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp2One, biasE8M0, colOddMaxExp, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);

        // Interleaved store: merge even/odd scale and 1/scale for axis=-2
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(colScaleAddr, mxScale2ZeroB8, mxScale2OneB8, maskB8);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(colScaleReciprocalAddr, reversedShareExp2Zero,
                                                                  reversedShareExp2One, maskAll);
    }
}

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::ComputeScaleCuBLASForSlot(
    __ubuf__ uint16_t *madataReadAddr, __ubuf__ uint16_t *reciprocalWriteAddr, Reg::RegTensor<uint8_t> &scale8,
    Reg::RegTensor<uint32_t> &invMax, Reg::RegTensor<uint32_t> &manMaskReg, Reg::RegTensor<uint32_t> &expMaskReg,
    Reg::RegTensor<uint32_t> &zero32Reg, Reg::RegTensor<uint32_t> &scaleBiasReg, Reg::RegTensor<uint32_t> &nan32Reg,
    Reg::RegTensor<uint32_t> &fp8Nan32Reg, Reg::MaskReg &maskAll, Reg::MaskReg &maskAll32, Reg::MaskReg &maskB16)
{
    Reg::RegTensor<uint16_t> max16Reg;
    Reg::RegTensor<uint32_t> max32Reg;
    Reg::RegTensor<uint32_t> exp32Reg;
    Reg::RegTensor<uint32_t> man32Reg;
    Reg::RegTensor<uint32_t> expOne32Reg;
    Reg::RegTensor<uint32_t> extractExp;
    Reg::RegTensor<uint32_t> halfScale;
    Reg::RegTensor<uint16_t> scaleBf16;
    Reg::RegTensor<uint16_t> recip16;
    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg p0;
    Reg::MaskReg p1;

    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
        max16Reg, madataReadAddr, VF_LEN_FP32);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float> &)max32Reg, (Reg::RegTensor<xDtype> &)max16Reg,
                                                  maskAll);
    Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
    Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
    Reg::Mul((Reg::RegTensor<float> &)max32Reg, (Reg::RegTensor<float> &)max32Reg, (Reg::RegTensor<float> &)invMax,
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
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scaleBf16, extractExp);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8, scaleBf16);

    Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
    Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
    Reg::Select<uint32_t>(halfScale, halfScale, nan32Reg, cmpResult);
    Reg::Select<uint32_t>(halfScale, halfScale, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16, halfScale);
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(reciprocalWriteAddr, recip16, VF_LEN_FP32, maskB16);
}

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::ComputeScaleCuBLASSecondLast(
    uint16_t dataLen, uint32_t outputDtypeInvMax, __ubuf__ uint16_t *colScaleReciprocalAddr,
    __ubuf__ uint8_t *colScaleAddr)
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
        Reg::Duplicate(invMax, outputDtypeInvMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(fp8Nan32Reg, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::Duplicate(nan32Reg, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        for (uint16_t i = 0; i < times; i++) {
            uint16_t colOffset = i * VF_LEN_FP32;
            __ubuf__ uint16_t *slot0Addr = colScaleReciprocalAddr + colOffset;
            __ubuf__ uint16_t *slot1Addr = colScaleReciprocalAddr + dataLen + colOffset;
            ComputeScaleCuBLASForSlot(slot0Addr, slot0Addr, scale8Slot0, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            ComputeScaleCuBLASForSlot(slot1Addr, slot1Addr, scale8Slot1, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(colScaleAddr + DIGIT_TWO * colOffset, scale8Slot0,
                                                                    scale8Slot1, interleaveMask);
        }
    }
}

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::ComputeScaleCuBLAS(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *rowDataAddr,
    __ubuf__ uint8_t *rowScaleAddr, __ubuf__ uint16_t *rowScaleReciprocalAddr, __ubuf__ uint8_t *colScaleAddr,
    __ubuf__ uint16_t *colScaleReciprocalAddr)
{
    uint32_t outputDtypeInvMax = outputDtypeInvMax_;
    __ubuf__ uint16_t *absMaxTempAddr = colScaleReciprocalAddr;
    __ubuf__ xDtype *dataStartAddr = xAddr;
    __ubuf__ uint16_t *rowScaleReciprocalStartAddr = rowScaleReciprocalAddr;
    PipeBarrier<PIPE_V>();
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> evenData;
        Reg::RegTensor<xDtype> oddData;
        Reg::RegTensor<uint16_t> evenAbs;
        Reg::RegTensor<uint16_t> oddAbs;
        Reg::RegTensor<uint16_t> absMaxDim1;
        Reg::RegTensor<uint16_t> colScaleSlot0Part0;
        Reg::RegTensor<uint16_t> colScaleSlot0Part1;
        Reg::RegTensor<uint16_t> colScaleSlot1Part0;
        Reg::RegTensor<uint16_t> colScaleSlot1Part1;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> scaleBf16;
        Reg::RegTensor<uint8_t> scaleU8Reg;
        Reg::RegTensor<uint8_t> scaleU8Row;
        Reg::RegTensor<uint16_t> reciprocalBf16Reg;
        Reg::RegTensor<uint16_t> reciprocalBf16Row;
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
        Reg::Duplicate(colScaleSlot0Part0, static_cast<uint16_t>(0));
        Reg::Duplicate(colScaleSlot0Part1, static_cast<uint16_t>(0));
        Reg::Duplicate(colScaleSlot1Part0, static_cast<uint16_t>(0));
        Reg::Duplicate(colScaleSlot1Part1, static_cast<uint16_t>(0));

        // Phase 1: collect row-scale maxima compactly while preserving the two
        // independent 32-row col-scale slots.
        uint16_t slot0Rows = blockCount < BLOCK_SIZE ? blockCount : BLOCK_SIZE;
        for (uint16_t i = 0; i < slot0Rows; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, dataLen);
            Reg::And(evenAbs, (Reg::RegTensor<uint16_t> &)evenData, absMask, maskAll);
            Reg::And(oddAbs, (Reg::RegTensor<uint16_t> &)oddData, absMask, maskAll);
            Reg::Max(absMaxDim1, evenAbs, oddAbs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(absMaxTempAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            Reg::Max(colScaleSlot0Part0, colScaleSlot0Part0, evenAbs, maskAll);
            Reg::Max(colScaleSlot0Part1, colScaleSlot0Part1, oddAbs, maskAll);
        }
        for (uint16_t i = slot0Rows; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, dataLen);
            Reg::And(evenAbs, (Reg::RegTensor<uint16_t> &)evenData, absMask, maskAll);
            Reg::And(oddAbs, (Reg::RegTensor<uint16_t> &)oddData, absMask, maskAll);
            Reg::Max(absMaxDim1, evenAbs, oddAbs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(absMaxTempAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            Reg::Max(colScaleSlot1Part0, colScaleSlot1Part0, evenAbs, maskAll);
            Reg::Max(colScaleSlot1Part1, colScaleSlot1Part1, oddAbs, maskAll);
        }
        Reg::StoreUnAlignPost(absMaxTempAddr, ureg, 0);

        Reg::Duplicate(invMax, outputDtypeInvMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

        // Phase 2: convert 64 cached maxima per VF, then scatter eight rows
        // back to the original 32-byte-aligned scale/reciprocal layout.
        __ubuf__ uint16_t *absMaxReadAddr = colScaleReciprocalAddr;
        uint16_t scaleCount = dataLen / BLOCK_SIZE;
        uint16_t batchCount =
            ops::CeilDiv(static_cast<uint16_t>(blockCount * scaleCount), static_cast<uint16_t>(VF_LEN_FP32));
        for (uint16_t j = 0; j < batchCount; j++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                absMaxDim1, absMaxReadAddr, VF_LEN_FP32);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float> &)max32,
                                                          (Reg::RegTensor<xDtype> &)absMaxDim1, maskAll);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMaskReg, maskAll32);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroReg32, maskAll32);
            Reg::Mul((Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)invMax,
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
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scaleBf16, extractExp);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scaleU8Reg, scaleBf16);
            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
            Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
            Reg::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(reciprocalBf16Reg, halfScale);

            for (uint16_t k = 0; k < 8; k++) {
                Reg::Arange(extractIdx, static_cast<int8_t>(k * 8));
                Reg::Gather(scaleU8Row, scaleU8Reg, (Reg::RegTensor<uint8_t> &)extractIdx);
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(rowScaleAddr, scaleU8Row, UB_BLOCK_SIZE,
                                                                             maskReduceB8);
                Reg::Arange(extractIdx, static_cast<int8_t>(k * 16));
                Reg::Gather((Reg::RegTensor<uint8_t> &)reciprocalBf16Row, (Reg::RegTensor<uint8_t> &)reciprocalBf16Reg,
                            (Reg::RegTensor<uint8_t> &)extractIdx);
                Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    rowScaleReciprocalAddr, reciprocalBf16Row, ROW_SCALE_RECIPROCAL_ROW_ELEMS, maskReduceB16);
            }
        }

        // Generate row_data immediately after row_scale while staying in the same vector scope.
        Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<float> evenZeroFp32;
        Reg::RegTensor<float> evenOneFp32;
        Reg::RegTensor<float> oddZeroFp32;
        Reg::RegTensor<float> oddOneFp32;
        Reg::RegTensor<rowDataDtype> evenZeroFp8;
        Reg::RegTensor<rowDataDtype> evenOneFp8;
        Reg::RegTensor<rowDataDtype> oddZeroFp8;
        Reg::RegTensor<rowDataDtype> oddOneFp8;
        __ubuf__ xDtype *dataReadAddr = dataStartAddr;
        __ubuf__ uint16_t *rowScaleReciprocalReadAddr = rowScaleReciprocalStartAddr;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, dataReadAddr, dataLen);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, rowScaleReciprocalReadAddr, ROW_SCALE_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulFP32, (Reg::RegTensor<bfloat16_t> &)scaleForMulFP16, maskAll);
                Reg::Mul(evenZeroFp32, evenZeroFp32, scaleForMulFP32, maskAll);
                Reg::Mul(evenOneFp32, evenOneFp32, scaleForMulFP32, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
                Reg::Mul(oddZeroFp32, oddZeroFp32, scaleForMulFP32, maskAll);
                Reg::Mul(oddOneFp32, oddOneFp32, scaleForMulFP32, maskAll);
            } else {
                Reg::Mul(evenData, evenData, (Reg::RegTensor<xDtype> &)scaleForMulFP16, maskAll);
                Reg::Mul(oddData, oddData, (Reg::RegTensor<xDtype> &)scaleForMulFP16, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
            }
            Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(evenZeroFp8, evenZeroFp32, maskAll);
            Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(evenOneFp8, evenOneFp32, maskAll);
            Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(oddZeroFp8, oddZeroFp32, maskAll);
            Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(oddOneFp8, oddOneFp32, maskAll);
            Reg::Add((Reg::RegTensor<uint8_t> &)evenZeroFp8, (Reg::RegTensor<uint8_t> &)evenZeroFp8,
                     (Reg::RegTensor<uint8_t> &)evenOneFp8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t> &)evenZeroFp8, (Reg::RegTensor<uint8_t> &)evenZeroFp8,
                     (Reg::RegTensor<uint8_t> &)oddZeroFp8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t> &)evenZeroFp8, (Reg::RegTensor<uint8_t> &)evenZeroFp8,
                     (Reg::RegTensor<uint8_t> &)oddOneFp8, maskAllB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                rowDataAddr, (Reg::RegTensor<uint8_t> &)evenZeroFp8, dataLen, maskAllB8);
        }

        // Reuse the temporary area for the normal axis=-2 maxima output.
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(colScaleReciprocalAddr, colScaleSlot0Part0,
                                                                  colScaleSlot0Part1, maskAll);
        if (blockCount > BLOCK_SIZE) {
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(colScaleReciprocalAddr + dataLen,
                                                                      colScaleSlot1Part0, colScaleSlot1Part1, maskAll);
        }
    }
    ComputeScaleCuBLASSecondLast(dataLen, outputDtypeInvMax, colScaleReciprocalAddr, colScaleAddr);
}

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::ComputeColDataToFp8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *colScaleReciprocalAddr,
    __ubuf__ uint8_t *colDataAddr)
{
    int64_t rowLength = tileRowLength_;
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
        Reg::RegTensor<rowDataDtype> yEvenFP8Layout0;
        Reg::RegTensor<rowDataDtype> yEvenFP8Layout2;
        Reg::RegTensor<rowDataDtype> yOddFP8Layout1;
        Reg::RegTensor<rowDataDtype> yOddFP8Layout3;

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        // Load all 256 reciprocal values once and reuse them for all 32 rows in this col-scale slot.
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            reciprocalEven, reciprocalOdd, colScaleReciprocalAddr, dualLoadLen);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalEvenFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t> &)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalEvenFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t> &)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalOddFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t> &)reciprocalOdd, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalOddFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t> &)reciprocalOdd, maskB16);
        }

        for (uint16_t row = 0; row < blockCount; row++) {
            __ubuf__ xDtype *xCursor = xAddr + row * rowLength;
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
                Reg::Mul(xEven, xEven, (Reg::RegTensor<xDtype> &)reciprocalEven, maskB16);
                Reg::Mul(xOdd, xOdd, (Reg::RegTensor<xDtype> &)reciprocalOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);
            }

            Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(yEvenFP8Layout0, xEvenFP32Layout0, maskB32);
            Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(yEvenFP8Layout2, xEvenFP32Layout1, maskB32);
            Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(yOddFP8Layout1, xOddFP32Layout0, maskB32);
            Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(yOddFP8Layout3, xOddFP32Layout1, maskB32);

            Reg::Add((Reg::RegTensor<uint8_t> &)yEvenFP8Layout0, (Reg::RegTensor<uint8_t> &)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t> &)yEvenFP8Layout2, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t> &)yOddFP8Layout1, (Reg::RegTensor<uint8_t> &)yOddFP8Layout1,
                     (Reg::RegTensor<uint8_t> &)yOddFP8Layout3, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t> &)yEvenFP8Layout0, (Reg::RegTensor<uint8_t> &)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t> &)yOddFP8Layout1, maskB8);

            __ubuf__ uint8_t *yCursor = colDataAddr + row * rowLength;
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yCursor, (Reg::RegTensor<uint8_t> &)yEvenFP8Layout0, dualLoadLen, maskB8);
        }
    }
}

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::ComputeColDataToFp4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *colScaleReciprocalAddr,
    __ubuf__ uint8_t *colDataAddr)
{
    int64_t rowLength = tileRowLength_;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> evenBf16;
        Reg::RegTensor<bfloat16_t> oddBf16;
        Reg::RegTensor<bfloat16_t> xBF16;
        Reg::RegTensor<float> zeroHalfFp32;
        Reg::RegTensor<float> oneHalfFp32;
        Reg::RegTensor<uint16_t> reversedShareExp;
        Reg::RegTensor<float> reciprocalZeroFp32;
        Reg::RegTensor<float> reciprocalOneFp32;
        Reg::RegTensor<rowDataDtype> packedFp4;

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        // Load per-column 1/scale
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExp, colScaleReciprocalAddr);

        for (uint16_t j = 0; j < blockCount; j++) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * rowLength);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(zeroHalfFp32, x, pregAll16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oneHalfFp32, x, pregAll16);

                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    reciprocalZeroFp32, (Reg::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);

                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(
                    reciprocalOneFp32, (Reg::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);

                Reg::Mul(zeroHalfFp32, zeroHalfFp32, reciprocalZeroFp32, pregAll32);
                Reg::Mul(oneHalfFp32, oneHalfFp32, reciprocalOneFp32, pregAll32);

                ComputeFP4FromHalf(zeroHalfFp32);
                ComputeFP4FromHalf(oneHalfFp32);

                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t> &)evenBf16, zeroHalfFp32,
                                                                  pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t> &)evenBf16,
                                                                        (Reg::RegTensor<uint32_t> &)evenBf16);

                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t> &)oddBf16, oneHalfFp32,
                                                                  pregAll32);

                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t> &)oddBf16,
                                                                        (Reg::RegTensor<uint32_t> &)oddBf16);

                Reg::Interleave(evenBf16, oddBf16, evenBf16, oddBf16);
                Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(
                    packedFp4, (Reg::RegTensor<bfloat16_t> &)evenBf16, pregAll16);
            } else {
                Reg::Mul(xBF16, x, (Reg::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);
                Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(packedFp4, xBF16, pregAll16);
            }

            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(colDataAddr + (j * 128),
                                                                     (Reg::RegTensor<uint8_t> &)packedFp4, pregAll8);
        }
    }
}

template <typename xDtype, typename rowDataDtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, rowDataDtype, mode, roundMode, scaleAlg>::CopyOut(
    int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t colScaleOutOffset, int64_t blockCount,
    int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;

    int64_t rowDataOutOffsetNow = rowDataOutOffset;

    // axis=-2 two rows interleaved, accounting for 32B alignment
    uint32_t scaleSrcStride =
        DIGIT_TWO * ops::CeilDiv(dataLen, UB_BLOCK_SIZE) - ops::CeilDiv(DIGIT_TWO * dataLen, UB_BLOCK_SIZE);

    if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value || IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
        // FP4: two fp4 values packed into one byte
        outBlockLen = static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t));
        rowDataOutOffsetNow = rowDataOutOffset / DIGIT_TWO;
    } else {
        // FP8: one byte per element
        outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) * sizeof(rowDataDtype) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) * sizeof(uint8_t));
    }

    DataCopyExtParams rowDataCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};

    // axis=-1 scale output: shape [M, ceil(N/blockSize)]
    uint32_t rowScaleOutLen = dataLenAlign / BLOCK_SIZE;

    DataCopyExtParams rowScaleCopyOutParams = {
        outBurst, static_cast<uint32_t>(rowScaleOutLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(rowLength_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - rowScaleOutLen),
        static_cast<uint32_t>(0)};

    // axis=-2 scale output: two rows interleaved per blockSize group
    DataCopyExtParams colScaleCopyOutParams = {
        static_cast<uint16_t>(blockCountAlign / DOUBLE_BLOCK_SIZE),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>(DIGIT_TWO * (rowLength_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)};

    // Dequeue and copy row_data
    LocalTensor<uint8_t> rowDataLocal = rowDataQueue_.template DeQue<uint8_t>();
    DataCopyPad(rowDataGm_[rowDataOutOffsetNow], rowDataLocal, rowDataCopyOutParams);
    rowDataQueue_.FreeTensor(rowDataLocal);

    // Dequeue and copy col_data
    LocalTensor<uint8_t> colDataLocal = colDataQueue_.template DeQue<uint8_t>();
    DataCopyPad(colDataGm_[rowDataOutOffsetNow], colDataLocal, rowDataCopyOutParams);
    colDataQueue_.FreeTensor(colDataLocal);

    // Dequeue and copy row_scale (axis=-1)
    LocalTensor<uint8_t> rowScaleLocal = rowScaleQueue_.template DeQue<uint8_t>();
    DataCopyPad(rowScaleGm_[rowScaleOutOffset], rowScaleLocal, rowScaleCopyOutParams);
    rowScaleQueue_.FreeTensor(rowScaleLocal);

    // Dequeue and copy col_scale (axis=-2)
    LocalTensor<uint8_t> colScaleLocal = colScaleQueue_.template DeQue<uint8_t>();
    DataCopyPad(colScaleGm_[colScaleOutOffset], colScaleLocal, colScaleCopyOutParams);
    colScaleQueue_.FreeTensor(colScaleLocal);
}
} // namespace ClaGateMxQuantWithDualAxis

#endif // OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H
