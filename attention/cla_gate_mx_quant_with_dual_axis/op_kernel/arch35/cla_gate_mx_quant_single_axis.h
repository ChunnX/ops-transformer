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
 * \file cla_gate_mx_quant_single_axis.h
 * \brief Single-axis (tail axis) ClaGate MX quantization implementation.
 */

#ifndef OPS_NN_CLA_GATE_MX_QUANT_SINGLE_AXIS_H
#define OPS_NN_CLA_GATE_MX_QUANT_SINGLE_AXIS_H

#include "cla_gate_mx_quant_common.h"

namespace ClaGateMxQuantWithDualAxis {

// Reusable CLA Gate fusion and row-axis quantization primitives are inherited
// from ClaGateMxQuantCommonBase. This file implements the single-axis path.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
class ClaGateMxQuantTailAxisBase
    : public ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx> {
public:
    using Base = ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>;
    using Base::Base;
    using Base::Init;

    __aicore__ inline void Process();

protected:
    // Expose common primitives/data to this single-axis implementation.
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

    __aicore__ inline void CopyInClaGateBatch(int64_t firstTile, int64_t tileCount);
    __aicore__ inline void ComputeClaGateFullTileSingleAxis(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                           __ubuf__ xDtype* globalGateRow, __ubuf__ xDtype* localGateRow,
                                                           __ubuf__ xDtype* outputAddr, uint16_t calcRows,
                                                           uint16_t firstColTile);
    __aicore__ inline void ProcessBatchChunk(int64_t firstTile, int64_t tileCount);
    __aicore__ inline void ComputeScaleOcpTailOnly(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                   __ubuf__ uint8_t* mxScale1Addr,
                                                   __ubuf__ uint16_t* mxScale1ReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLASSingleAxis(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                        __ubuf__ uint8_t* y1Addr, __ubuf__ uint8_t* mxScale1Addr,
                                                        __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                                        __ubuf__ uint16_t* tempMaxAddr);
    __aicore__ inline void CopyOutRowOnly(int64_t yOffset, int64_t scale1OutOffset, int64_t blockCount,
                                          int64_t dataLen, bool hasSecondAxis = false);
};

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantTailAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::Process()
{
    dimNeg1ScaleNum_ = ops::CeilDiv(dimN_, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
    int64_t coreNum = tilingData_->usedCoreNum;
    if (blockIdx_ >= coreNum || coreNum <= 0) {
        return;
    }

    int64_t rowTileNum = tilingData_->rowTileNum;
    int64_t colTileNum = tilingData_->colTileNum;
    int64_t totalTiles = rowTileNum * colTileNum;
    if (totalTiles <= 0) {
        return;
    }

    int64_t blockPerCore = totalTiles / coreNum;
    int64_t tailCore = totalTiles % coreNum;
    int64_t loops = blockPerCore + (blockIdx_ < tailCore ? 1 : 0);
    int64_t firstBlock = blockIdx_ * blockPerCore + (blockIdx_ < tailCore ? blockIdx_ : tailCore);
    if (loops <= 0) {
        return;
    }

    // UB batch parameters are precomputed by tiling and passed via tiling data.
    // Each batch contains at most 64 full 256-column tiles; the last batch may
    // be smaller and is handled as a tail.
    int64_t batchCount = tilingData_->batchCount;
    int64_t batchSize = tilingData_->batchSize;
    if (batchCount <= 0 || batchSize <= 0) {
        batchCount = 1;
        batchSize = loops < DOUBLE_BLOCK_SIZE ? loops : DOUBLE_BLOCK_SIZE;
    }

    // If all tiles owned by this core are full 256-column blocks, the whole
    // range is memory-contiguous and can be copied in batches with one DataCopy
    // per batch. The batch size is supplied by tiling.
    bool canBatchFullTiles = true;
    int64_t firstRow = firstBlock / colTileNum;
    int64_t lastRow = (firstBlock + loops - 1) / colTileNum;
    if (firstRow != lastRow) {
        canBatchFullTiles = false;
    }
    for (int64_t i = 0; i < loops; ++i) {
        int64_t tile = firstBlock + i;
        int64_t colTileIdx = tile % colTileNum;
        int64_t calcCol = (colTileIdx == colTileNum - 1) ? tilingData_->colTailSize : tilingData_->colBlockSize;
        if (calcCol != ubRowLen_) {
            canBatchFullTiles = false;
            break;
        }
    }
    if (canBatchFullTiles) {
        for (int64_t start = 0; start < loops; start += batchSize) {
            int64_t count = (loops - start < batchSize) ? (loops - start) : batchSize;
            ProcessBatchChunk(firstBlock + start, count);
        }
        return;
    }

    for (int64_t batch = 0; batch < batchCount; ++batch) {
        int64_t batchStart = batch * batchSize;
        if (batchStart >= loops) {
            break;
        }
        int64_t batchEnd = (batch + 1) * batchSize < loops ? (batch + 1) * batchSize : loops;
        for (int64_t i = batchStart; i < batchEnd; ++i) {
        int64_t tile = firstBlock + i;
        int64_t rowTileIdx = tile / colTileNum;
        int64_t colTileIdx = tile % colTileNum;

        int64_t absRowStart = rowTileIdx * tilingData_->rowBlockSize;
        int64_t colOffset = colTileIdx * tilingData_->colBlockSize;
        int64_t calcRow = (rowTileIdx == rowTileNum - 1) ? tilingData_->rowTailSize : tilingData_->rowBlockSize;
        int64_t calcCol = (colTileIdx == colTileNum - 1) ? tilingData_->colTailSize : tilingData_->colBlockSize;
        if (calcRow <= 0 || calcCol <= 0) {
            continue;
        }

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

        actInQueue_.template FreeTensor(input);
        gateInQueue_.template FreeTensor(gateInput);

        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            LocalTensor<uint8_t> scale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> scale1Recip = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            LocalTensor<uint16_t> scale2Recip = mxScale2ReciprocalBuf_.template Get<uint16_t>();
            auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
            auto scale1Addr = (__ubuf__ uint8_t*)scale1.GetPhyAddr();
            auto scale1RecipAddr = (__ubuf__ uint16_t*)scale1Recip.GetPhyAddr();
            auto tempMaxAddr = (__ubuf__ uint16_t*)scale2Recip.GetPhyAddr();
            ComputeScaleCuBLASSingleAxis(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow), mergedAddr,
                                         y1Addr, scale1Addr, scale1RecipAddr, tempMaxAddr);
            mxScaleQueue1_.template EnQue(scale1);
            outQueue1_.template EnQue(y1);

            int64_t yOffset = absRowStart * dimN_ + colOffset;
            int64_t scale1Offset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
            CopyOutRowOnly(yOffset, scale1Offset, calcRow, calcCol);
        } else {
            LocalTensor<uint8_t> scale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> scale1Recip = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
            auto scale1Addr = (__ubuf__ uint8_t*)scale1.GetPhyAddr();
            auto scale1RecipAddr = (__ubuf__ uint16_t*)scale1Recip.GetPhyAddr();
            ComputeScaleOcpTailOnly(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow), mergedAddr,
                                    scale1Addr, scale1RecipAddr);
            if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value ||
                          IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
                ComputeY1ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow),
                               mergedAddr, scale1RecipAddr, y1Addr);
            } else {
                ComputeY1ToFP8(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow),
                               mergedAddr, scale1RecipAddr, y1Addr);
            }
            mxScaleQueue1_.template EnQue(scale1);
            outQueue1_.template EnQue(y1);

            int64_t yOffset = absRowStart * dimN_ + colOffset;
            int64_t scale1Offset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
            CopyOutRowOnly(yOffset, scale1Offset, calcRow, calcCol);
        }
    }
}
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantTailAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::CopyInClaGateBatch(
    int64_t firstTile, int64_t tileCount)
{
    LocalTensor<xDtype> actInput = actInQueue_.template AllocTensor<xDtype>();
    LocalTensor<xDtype> gateInput = gateInQueue_.template AllocTensor<xDtype>();
    DataCopyPadExtParams<xDtype> pad = {false, 0, 0, 0};
    int64_t colTileNum = tilingData_->colTileNum;
    int64_t firstRow = firstTile / colTileNum;
    int64_t firstColTile = firstTile % colTileNum;
    int64_t srcOffset = firstRow * dimN_ + firstColTile * ubRowLen_;
    uint32_t totalBytes = static_cast<uint32_t>(tileCount * ubRowLen_ * sizeof(xDtype));
    DataCopyExtParams contiguousCopy = {1, totalBytes, 0, 0, 0};
    DataCopyPad(actInput, globalOutGm_[srcOffset], contiguousCopy, pad);
    DataCopyPad(actInput[inHalfSize_], localOutGm_[srcOffset], contiguousCopy, pad);

    // Gate layout is [T,N], so for one actual row the needed gate segment is
    // contiguous in GM. Copy it once per side in row-major order. The single
    // axis compute reads directly from this row-major gate buffer.
    int64_t firstHead = (firstColTile * ubRowLen_) / headDim_;
    int64_t gateCount = tileCount * headsPerRow_;
    uint32_t gateBytes = static_cast<uint32_t>(gateCount * sizeof(xDtype));
    DataCopyExtParams gateCopy = {1, gateBytes, 0, 0, 0};
    DataCopyPad(gateInput, globalGateGm_[firstRow * headNum_ + firstHead], gateCopy, pad);
    DataCopyPad(gateInput[gateSideStride_], localGateGm_[firstRow * headNum_ + firstHead], gateCopy, pad);

    actInQueue_.template EnQue(actInput);
    gateInQueue_.template EnQue(gateInput);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantTailAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeClaGateFullTileSingleAxis(
    __ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
    __ubuf__ xDtype* globalGateRow, __ubuf__ xDtype* localGateRow,
    __ubuf__ xDtype* outputAddr, uint16_t calcRows, uint16_t firstColTile)
{
    uint16_t vfsPerHead = static_cast<uint16_t>(headDim_ / VF_LEN_FP32);
    uint16_t headsPerRow = static_cast<uint16_t>(headsPerRow_);
    // For full 256-column blocks, every slot advances by headsPerRow heads.
    uint32_t baseHead = static_cast<uint32_t>(firstColTile) * headsPerRow;
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> globalReg, localReg, outputReg;
        Reg::RegTensor<xDtype> globalGateReg, localGateReg;
        Reg::RegTensor<float> gateF, tmpF, oneF, globalSigF, localSigF;
        Reg::RegTensor<float> globalRegF, localRegF, outputRegF;
        Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::AddrReg offset;

        for (uint16_t slot = 0; slot < calcRows; ++slot) {
            uint32_t rowOffset = slot * static_cast<uint32_t>(ubRowLen_);
            __ubuf__ xDtype* rowGlobal = globalAddr + rowOffset;
            __ubuf__ xDtype* rowLocal = localAddr + rowOffset;
            __ubuf__ xDtype* rowOut = outputAddr + rowOffset;
            uint32_t headStart = baseHead + slot * headsPerRow;
            for (uint16_t h = 0; h < headsPerRow; ++h) {
                uint32_t headIndex = headStart + h;

                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_BRC_B16>(globalGateReg, globalGateRow + headIndex);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, globalGateReg, allMask);
                Reg::Muls(tmpF, gateF, -1.0f, allMask);
                Reg::Exp(tmpF, tmpF, allMask);
                Reg::Adds(tmpF, tmpF, 1.0f, allMask);
                Reg::Duplicate(oneF, 1.0f, allMask);
                Reg::Div(globalSigF, oneF, tmpF, allMask);

                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_BRC_B16>(localGateReg, localGateRow + headIndex);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, localGateReg, allMask);
                Reg::Muls(tmpF, gateF, -1.0f, allMask);
                Reg::Exp(tmpF, tmpF, allMask);
                Reg::Adds(tmpF, tmpF, 1.0f, allMask);
                Reg::Div(localSigF, oneF, tmpF, allMask);

                for (uint16_t vf = 0; vf < vfsPerHead; ++vf) {
                    offset = Reg::CreateAddrReg<xDtype>(0, ONCE_ROW_LEN, h, headDim_, vf, VF_LEN_FP32);
                    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg, rowGlobal, offset);
                    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg, rowLocal, offset);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF, globalReg, allMask);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF, localReg, allMask);
                    Reg::Mul(globalRegF, globalRegF, globalSigF, allMask);
                    Reg::Mul(localRegF, localRegF, localSigF, allMask);
                    Reg::Add(outputRegF, globalRegF, localRegF, allMask);
                    Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg, outputRegF, allMask);
                    Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(rowOut, outputReg, offset, allMask);
                }
            }
        }
    }
}


template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantTailAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ProcessBatchChunk(
    int64_t firstTile, int64_t tileCount)
{
    int64_t colTileNum = tilingData_->colTileNum;
    CopyInClaGateBatch(firstTile, tileCount);

    LocalTensor<xDtype> input = actInQueue_.template DeQue<xDtype>();
    LocalTensor<xDtype> gateInput = gateInQueue_.template DeQue<xDtype>();
    LocalTensor<xDtype> merged = claGateBuf_.template Get<xDtype>();
    auto globalAddr = (__ubuf__ xDtype*)input.GetPhyAddr();
    auto localAddr = (__ubuf__ xDtype*)input[inHalfSize_].GetPhyAddr();
    auto globalGateAddr = (__ubuf__ xDtype*)gateInput.GetPhyAddr();
    auto localGateAddr = (__ubuf__ xDtype*)gateInput[gateSideStride_].GetPhyAddr();
    auto mergedAddr = (__ubuf__ xDtype*)merged.GetPhyAddr();

    int64_t firstColTile = firstTile % colTileNum;
    ComputeClaGateFullTileSingleAxis(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr,
                                     static_cast<uint16_t>(tileCount), static_cast<uint16_t>(firstColTile));
    actInQueue_.template FreeTensor(input);
    gateInQueue_.template FreeTensor(gateInput);

    for (int64_t slot = 0; slot < tileCount; ++slot) {
        int64_t tile = firstTile + slot;
        int64_t rowTileIdx = tile / colTileNum;
        int64_t colTileIdx = tile % colTileNum;
        int64_t absRowStart = rowTileIdx * tilingData_->rowBlockSize;
        int64_t colOffset = colTileIdx * tilingData_->colBlockSize;
        int64_t calcRow = 1;
        int64_t calcCol = tilingData_->colBlockSize;
        __ubuf__ xDtype* rowMerged = mergedAddr + slot * ubRowLen_;

        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            LocalTensor<uint8_t> scale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> scale1Recip = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            LocalTensor<uint16_t> scale2Recip = mxScale2ReciprocalBuf_.template Get<uint16_t>();
            auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
            auto scale1Addr = (__ubuf__ uint8_t*)scale1.GetPhyAddr();
            auto scale1RecipAddr = (__ubuf__ uint16_t*)scale1Recip.GetPhyAddr();
            auto tempMaxAddr = (__ubuf__ uint16_t*)scale2Recip.GetPhyAddr();
            ComputeScaleCuBLASSingleAxis(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow),
                                         rowMerged, y1Addr, scale1Addr, scale1RecipAddr, tempMaxAddr);
            mxScaleQueue1_.template EnQue(scale1);
            outQueue1_.template EnQue(y1);

            int64_t yOffset = absRowStart * dimN_ + colOffset;
            int64_t scale1Offset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
            CopyOutRowOnly(yOffset, scale1Offset, calcRow, calcCol);
        } else {
            LocalTensor<uint8_t> scale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> scale1Recip = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
            auto scale1Addr = (__ubuf__ uint8_t*)scale1.GetPhyAddr();
            auto scale1RecipAddr = (__ubuf__ uint16_t*)scale1Recip.GetPhyAddr();
            ComputeScaleOcpTailOnly(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow),
                                    rowMerged, scale1Addr, scale1RecipAddr);
            if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value ||
                          IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
                ComputeY1ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow),
                               rowMerged, scale1RecipAddr, y1Addr);
            } else {
                ComputeY1ToFP8(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcRow),
                               rowMerged, scale1RecipAddr, y1Addr);
            }
            mxScaleQueue1_.template EnQue(scale1);
            outQueue1_.template EnQue(y1);

            int64_t yOffset = absRowStart * dimN_ + colOffset;
            int64_t scale1Offset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
            CopyOutRowOnly(yOffset, scale1Offset, calcRow, calcCol);
        }
    }
}


template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantTailAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleOcpTailOnly(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr)
{
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
        Reg::RegTensor<uint16_t> yMaxExp;
        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::RegTensor<uint16_t> mxScale1B16;
        Reg::RegTensor<uint8_t> mxScale1B8;
        Reg::RegTensor<uint16_t> reversedShareExp1;
        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg infNanDataMask0;
        Reg::MaskReg infNanDataMask1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(yMaxExp, localDtypeYMaxExp);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(zero, static_cast<uint16_t>(0));
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       ONCE_ROW_LEN);
            if constexpr (IsSameType<xDtype, half>::value) {
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
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }
            Reg::Max(expMaxDim1, x0ExpBF16, x1ExpBF16, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMaxDim1, expMaxDim1, maskAll);

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
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);
            Reg::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1ReciprocalAddr, reversedShareExp1,
                                                                          SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantTailAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLASSingleAxis(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* y1Addr,
    __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint16_t* tempMaxAddr)
{
    uint32_t localInvDtypeMax = invDtypeMax_;
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
        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       dataLen);
            Reg::And(x0Abs, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(x1Abs, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            Reg::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
        }
        Reg::StoreUnAlignPost(tempMaxAddr, ureg, 0);

        Reg::Duplicate(invMax, localInvDtypeMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

        __ubuf__ uint16_t* readMaxAddr = tempMaxAddr;
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
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantTailAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::CopyOutRowOnly(int64_t yOffset,
                                                                                int64_t scale1OutOffset,
                                                                                int64_t blockCount,
                                                                                int64_t dataLen,
                                                                                bool hasSecondAxis)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t yOffsetNow = yOffset;

    if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        outBlockLen = static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((ubRowLen_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimN_ - dataLen) / DIGIT_TWO * sizeof(uint8_t));
        yOffsetNow = yOffset / DIGIT_TWO;
    } else {
        outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(y1Dtype) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimN_ - dataLen) * sizeof(uint8_t));
    }

    DataCopyExtParams yCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};
    int64_t dataLenAlign = ops::CeilDiv(dataLen, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
    uint32_t scale1OutLen = static_cast<uint32_t>(dataLenAlign / BLOCK_SIZE);
    DataCopyExtParams scale1CopyOutParams = {
        outBurst, scale1OutLen, static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(dimN_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - scale1OutLen),
        static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> y1Local = outQueue1_.template DeQue<uint8_t>();
    DataCopyPad(yGm1_[yOffsetNow], y1Local, yCopyOutParams);
    outQueue1_.FreeTensor(y1Local);

    LocalTensor<uint8_t> mxScale1Local = mxScaleQueue1_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm1_[scale1OutOffset], mxScale1Local, scale1CopyOutParams);
    mxScaleQueue1_.FreeTensor(mxScale1Local);

    // Single-axis mode: col_data/col_scale are empty. If the second-axis
    // queue buffers were used (scale_alg=1 fallback), release them without copy.
    if (hasSecondAxis) {
        LocalTensor<uint8_t> y2Local = outQueue2_.template DeQue<uint8_t>();
        outQueue2_.FreeTensor(y2Local);
        LocalTensor<uint8_t> mxScale2Local = mxScaleQueue2_.template DeQue<uint8_t>();
        mxScaleQueue2_.FreeTensor(mxScale2Local);
    }
}
} // namespace ClaGateMxQuantWithDualAxis

#endif // OPS_NN_CLA_GATE_MX_QUANT_SINGLE_AXIS_H
