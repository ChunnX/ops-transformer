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
 * \file cla_gate_quant_single_axis.h
 * \brief Single-axis (tail axis) ClaGate MX quantization implementation.
 */

#ifndef OPS_NN_CLA_GATE_QUANT_SINGLE_AXIS_H
#define OPS_NN_CLA_GATE_QUANT_SINGLE_AXIS_H

#include "cla_gate_quant_common.h"

namespace ClaGateQuant {

// Reusable CLA Gate fusion and row-axis quantization primitives are inherited
// from ClaGateQuantCommonBase. This file implements the single-axis path.
template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class ClaGateQuantTailAxisBase : public ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg> {
public:
    using Base = ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>;
    using Base::Base;

    __aicore__ inline void Init(GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate,
                                GM_ADDR rowData, GM_ADDR rowScale, GM_ADDR colData, GM_ADDR colScale)
    {
        Base::template Init<TPL_SINGLE_AXIS>(globalOut, localOut, globalGate, localGate, rowData, rowScale, colData,
                                             colScale);
    }

    __aicore__ inline void Process();

protected:
    // Expose common primitives/data to this single-axis implementation.
    using Base::ComputeClaGateFullTile;
    using Base::ComputeFP4FromHalf;
    using Base::castTraitBF16toFp4;
    using Base::castTraitFp32toBF16;
    using Base::castTraitFp32toYdtype;
    using Base::ComputeRowDataToFp4;
    using Base::ComputeRowDataToFp8;
    using Base::CopyInClaGate;
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
    using Base::ubBlockElemCountB16_;
    using Base::ubBlockElemCountB8_;
    using Base::rowDataQueue_;
    using Base::pipe_;
    using Base::sigmoidBroadcastBuf_;
    using Base::tilingData_;
    using Base::tileRowLength_;
    using Base::tileTailLength_;
    using Base::tileRowCount_;
    using Base::rowDataGm_;
    using Base::colDataGm_;

    __aicore__ inline void CopyInClaGateBatch(int64_t firstTile, int64_t tileCount);
    __aicore__ inline void ComputeClaGateFullTileSingleAxis(__ubuf__ xDtype *globalAddr, __ubuf__ xDtype *localAddr,
                                                            __ubuf__ xDtype *globalGateRow,
                                                            __ubuf__ xDtype *localGateRow, __ubuf__ xDtype *outputAddr,
                                                            uint16_t calcRows, uint16_t firstColTile);
    __aicore__ inline void ProcessBatchChunk(int64_t firstTile, int64_t tileCount);
    __aicore__ inline void ComputeRowScaleOcpSingleAxis(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                                        __ubuf__ uint8_t *rowScaleAddr,
                                                        __ubuf__ uint16_t *rowScaleReciprocalAddr);
    __aicore__ inline void ComputeRowScaleOcpBatchCompact(
        uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *rowScaleAddr,
        __ubuf__ uint16_t *rowScaleReciprocalAddr, __ubuf__ uint16_t *maxExpAddr);
    __aicore__ inline void ComputeRowDataToFp8Compact(uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                                      __ubuf__ uint16_t *rowScaleReciprocalAddr,
                                                      __ubuf__ uint8_t *rowDataAddr);
    __aicore__ inline void ComputeRowScaleCuBLASSingleAxis(uint16_t dataLen, uint16_t blockCount,
                                                           __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *rowDataAddr,
                                                           __ubuf__ uint8_t *rowScaleAddr,
                                                           __ubuf__ uint16_t *rowScaleReciprocalAddr,
                                                           __ubuf__ uint16_t *absMaxTempAddr,
                                                           bool compactLayout = false);
    __aicore__ inline void CopyOutRowOnly(int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t blockCount,
                                          int64_t dataLen);
    __aicore__ inline void CopyOutRowOnlyBatch(int64_t rowDataOutOffset, int64_t rowScaleOutOffset,
                                               int64_t tileCount);
};

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::Process()
{
    rowScaleElementsPerRow_ = ops::CeilDiv(rowLength_, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
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

    // UB batch capacity is derived by host tiling.  Repeated batches use a
    // complete 64-gate sigmoid-VF quantum (64 tiles for D=256, 32 for D=128),
    // while the last batch may contain fewer tiles.
    int64_t batchSize = tilingData_->batchSize;
    if (batchSize <= 0) {
        batchSize = loops < DOUBLE_BLOCK_SIZE ? loops : DOUBLE_BLOCK_SIZE;
    }

    // Partition this core's range into maximal consecutive full-width runs.
    // The tail position is deterministic (the last column tile of each row),
    // so calculate the run length directly instead of scanning every following
    // tile with another modulo and width check.
    const bool hasColTail = tilingData_->colTailSize != tilingData_->colBlockSize;
    int64_t i = 0;
    while (i < loops) {
        int64_t tile = firstBlock + i;
        int64_t colTileIdxNow = tile % colTileNum;
        bool isTailTile = hasColTail && colTileIdxNow == colTileNum - 1;
        if (!isTailTile) {
            int64_t remaining = loops - i;
            int64_t fullRun = remaining;
            if (hasColTail) {
                // Number of full tiles before this row's unique short tail.
                int64_t fullTilesBeforeTail = colTileNum - 1 - colTileIdxNow;
                fullRun = remaining < fullTilesBeforeTail ? remaining : fullTilesBeforeTail;
            }
            for (int64_t start = 0; start < fullRun; start += batchSize) {
                int64_t count = (fullRun - start < batchSize) ? (fullRun - start) : batchSize;
                ProcessBatchChunk(tile + start, count);
            }
            i += fullRun;
            continue;
        }

        int64_t rowTileIdx = tile / colTileNum;
        int64_t colTileIdx = colTileIdxNow;

        int64_t absRowStart = rowTileIdx * tilingData_->rowBlockSize;
        int64_t colOffset = colTileIdx * tilingData_->colBlockSize;
        int64_t calcRow = tilingData_->rowBlockSize; // single-axis rowBlockSize is always 1
        int64_t calcCol = tilingData_->colTailSize;  // this branch is only the short column tail

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
                                   static_cast<uint16_t>(calcRow));

            activationInQueue_.template FreeTensor(activationInput);
            gateInQueue_.template FreeTensor(gateInput);

            if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
                LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
                LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
                LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
                LocalTensor<uint16_t> colScaleReciprocalTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
                auto rowDataAddr = (__ubuf__ uint8_t *)rowDataTensor.GetPhyAddr();
                auto rowScaleAddr = (__ubuf__ uint8_t *)rowScaleTensor.GetPhyAddr();
                auto rowScaleReciprocalAddr = (__ubuf__ uint16_t *)rowScaleReciprocalTensor.GetPhyAddr();
                auto absMaxTempAddr = (__ubuf__ uint16_t *)colScaleReciprocalTensor.GetPhyAddr();
                ComputeRowScaleCuBLASSingleAxis(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow),
                                                mergedAddr, rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr,
                                                absMaxTempAddr);
                rowScaleQueue_.template EnQue(rowScaleTensor);
                rowDataQueue_.template EnQue(rowDataTensor);

                int64_t rowDataOutOffset = absRowStart * rowLength_ + colOffset;
                int64_t rowScaleOutOffset = absRowStart * rowScaleElementsPerRow_ + colOffset / BLOCK_SIZE;
                CopyOutRowOnly(rowDataOutOffset, rowScaleOutOffset, calcRow, calcCol);
            } else {
                LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
                LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
                LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
                auto rowDataAddr = (__ubuf__ uint8_t *)rowDataTensor.GetPhyAddr();
                auto rowScaleAddr = (__ubuf__ uint8_t *)rowScaleTensor.GetPhyAddr();
                auto rowScaleReciprocalAddr = (__ubuf__ uint16_t *)rowScaleReciprocalTensor.GetPhyAddr();
                ComputeRowScaleOcpSingleAxis(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow),
                                             mergedAddr, rowScaleAddr, rowScaleReciprocalAddr);
                if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value ||
                              IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
                    ComputeRowDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow),
                                        mergedAddr, rowScaleReciprocalAddr, rowDataAddr);
                } else {
                    ComputeRowDataToFp8(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow),
                                        mergedAddr, rowScaleReciprocalAddr, rowDataAddr);
                }
                rowScaleQueue_.template EnQue(rowScaleTensor);
                rowDataQueue_.template EnQue(rowDataTensor);

                int64_t rowDataOutOffset = absRowStart * rowLength_ + colOffset;
                int64_t rowScaleOutOffset = absRowStart * rowScaleElementsPerRow_ + colOffset / BLOCK_SIZE;
                CopyOutRowOnly(rowDataOutOffset, rowScaleOutOffset, calcRow, calcCol);
            }
        ++i;
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyInClaGateBatch(
    int64_t firstTile, int64_t tileCount)
{
    LocalTensor<xDtype> activationInput = activationInQueue_.template AllocTensor<xDtype>();
    LocalTensor<xDtype> gateInput = gateInQueue_.template AllocTensor<xDtype>();
    DataCopyPadExtParams<xDtype> pad = {false, 0, 0, 0};
    int64_t colTileNum = tilingData_->colTileNum;
    int64_t firstRow = firstTile / colTileNum;
    int64_t firstColTile = firstTile % colTileNum;
    int64_t srcOffset = firstRow * rowLength_ + firstColTile * tileRowLength_;
    uint32_t totalBytes = static_cast<uint32_t>(tileCount * tileRowLength_ * sizeof(xDtype));
    DataCopyExtParams contiguousCopy = {1, totalBytes, 0, 0, 0};
    DataCopyPad(activationInput, globalOutGm_[srcOffset], contiguousCopy, pad);
    DataCopyPad(activationInput[halfBufferElemCount_], localOutGm_[srcOffset], contiguousCopy, pad);

    // Gate layout is [T,N], so for one actual row the needed gate segment is
    // contiguous in GM. Copy it once per side in row-major order. The single
    // axis compute reads directly from this row-major gate buffer.
    int64_t firstHead = (firstColTile * tileRowLength_) / headSize_;
    int64_t gateCount = tileCount * headsPerTileRow_;
    uint32_t gateBytes = static_cast<uint32_t>(gateCount * sizeof(xDtype));
    DataCopyExtParams gateCopy = {1, gateBytes, 0, 0, 0};
    DataCopyPad(gateInput, globalGateGm_[firstRow * headCount_ + firstHead], gateCopy, pad);
    DataCopyPad(gateInput[gateSideOffset_], localGateGm_[firstRow * headCount_ + firstHead], gateCopy, pad);

    activationInQueue_.template EnQue(activationInput);
    gateInQueue_.template EnQue(gateInput);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeClaGateFullTileSingleAxis(
    __ubuf__ xDtype *globalAddr, __ubuf__ xDtype *localAddr, __ubuf__ xDtype *globalGateRow,
    __ubuf__ xDtype *localGateRow, __ubuf__ xDtype *outputAddr, uint16_t calcRows, uint16_t firstColTile)
{
    uint16_t vfsPerHead = static_cast<uint16_t>(headSize_ / VF_LEN_FP32);
    uint16_t headsPerRow = static_cast<uint16_t>(headsPerTileRow_);
    // CopyInClaGateBatch copies a contiguous tile range into UB. Activation and
    // gate buffers are therefore both indexed by the local slot:
    //   activation: slot * tileRowLength_ + h * headSize_ + vf * VF_LEN_FP32
    //   gate:       slot * headsPerRow + h
    // firstColTile is not a local UB offset; using it here caused cross-tile
    // batches to read the wrong gate/activation data.
    (void)firstColTile;
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> globalReg, localReg, outputReg;
        Reg::RegTensor<xDtype> globalGateReg, localGateReg;
        Reg::RegTensor<float> globalGateF, localGateF, globalTmpF, localTmpF, oneF;
        Reg::RegTensor<float> globalSigF, localSigF;
        Reg::RegTensor<float> globalRegF, localRegF, outputRegF;
        Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        // Compute the compact gate scalars vector-wise once per batch.  The
        // previous code broadcast every scalar to a full vector before Exp/Div,
        // causing up to 128 full-vector sigmoid evaluations per 64-tile batch.
        __ubuf__ uint8_t *sigmoidBytes =
            (__ubuf__ uint8_t *)sigmoidBroadcastBuf_.template Get<float>().GetPhyAddr();
        __ubuf__ float *globalSigmoidUb = (__ubuf__ float *)sigmoidBytes;
        __ubuf__ float *localSigmoidUb = (__ubuf__ float *)(sigmoidBytes + gateSideOffset_ * sizeof(float));
        uint16_t gateCount = static_cast<uint16_t>(calcRows * headsPerRow);
        uint16_t gateVecLoops = static_cast<uint16_t>(ops::CeilDiv(gateCount, static_cast<uint16_t>(VF_LEN_FP32)));

        Reg::Duplicate(oneF, 1.0f, allMask);
        for (uint16_t i = 0; i < gateVecLoops; ++i) {
            uint32_t gateOffset = i * VF_LEN_FP32;
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalGateReg, globalGateRow + gateOffset);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localGateReg, localGateRow + gateOffset);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalGateF, globalGateReg, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localGateF, localGateReg, allMask);
            Reg::Muls(globalTmpF, globalGateF, -1.0f, allMask);
            Reg::Muls(localTmpF, localGateF, -1.0f, allMask);
            Reg::Exp(globalTmpF, globalTmpF, allMask);
            Reg::Exp(localTmpF, localTmpF, allMask);
            Reg::Adds(globalTmpF, globalTmpF, 1.0f, allMask);
            Reg::Adds(localTmpF, localTmpF, 1.0f, allMask);
            Reg::Div(globalSigF, oneF, globalTmpF, allMask);
            Reg::Div(localSigF, oneF, localTmpF, allMask);
            Reg::StoreAlign<float>(globalSigmoidUb + gateOffset, globalSigF, allMask);
            Reg::StoreAlign<float>(localSigmoidUb + gateOffset, localSigF, allMask);
        }

        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

        for (uint16_t slot = 0; slot < calcRows; ++slot) {
            uint32_t tileOffset = slot * static_cast<uint32_t>(tileRowLength_);
            __ubuf__ xDtype *tileGlobal = globalAddr + tileOffset;
            __ubuf__ xDtype *tileLocal = localAddr + tileOffset;
            __ubuf__ xDtype *tileOut = outputAddr + tileOffset;
            uint32_t localHeadBase = slot * headsPerRow;
            for (uint16_t h = 0; h < headsPerRow; ++h) {
                uint32_t localHead = localHeadBase + h;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigF, globalSigmoidUb + localHead);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigF, localSigmoidUb + localHead);

                for (uint16_t vf = 0; vf < vfsPerHead; ++vf) {
                    uint32_t vfOffset = h * static_cast<uint32_t>(headSize_) + vf * VF_LEN_FP32;
                    __ubuf__ xDtype *vfGlobal = tileGlobal + vfOffset;
                    __ubuf__ xDtype *vfLocal = tileLocal + vfOffset;
                    __ubuf__ xDtype *vfOut = tileOut + vfOffset;

                    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg, vfGlobal);
                    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg, vfLocal);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF, globalReg, allMask);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF, localReg, allMask);
                    Reg::Mul(globalRegF, globalRegF, globalSigF, allMask);
                    Reg::Mul(localRegF, localRegF, localSigF, allMask);
                    Reg::Add(outputRegF, globalRegF, localRegF, allMask);
                    Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg, outputRegF, allMask);
                    Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(vfOut, outputReg, allMask);
                }
            }
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ProcessBatchChunk(
    int64_t firstTile, int64_t tileCount)
{
    int64_t colTileNum = tilingData_->colTileNum;
    CopyInClaGateBatch(firstTile, tileCount);

    LocalTensor<xDtype> activationInput = activationInQueue_.template DeQue<xDtype>();
    LocalTensor<xDtype> gateInput = gateInQueue_.template DeQue<xDtype>();
    LocalTensor<xDtype> mergedResult = mergedClaGateBuf_.template Get<xDtype>();
    auto globalAddr = (__ubuf__ xDtype *)activationInput.GetPhyAddr();
    auto localAddr = (__ubuf__ xDtype *)activationInput[halfBufferElemCount_].GetPhyAddr();
    auto globalGateAddr = (__ubuf__ xDtype *)gateInput.GetPhyAddr();
    auto localGateAddr = (__ubuf__ xDtype *)gateInput[gateSideOffset_].GetPhyAddr();
    auto mergedAddr = (__ubuf__ xDtype *)mergedResult.GetPhyAddr();

    int64_t firstColTile = firstTile % colTileNum;
    ComputeClaGateFullTileSingleAxis(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr,
                                     static_cast<uint16_t>(tileCount), static_cast<uint16_t>(firstColTile));
    activationInQueue_.template FreeTensor(activationInput);
    gateInQueue_.template FreeTensor(gateInput);

    if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
        // The full-width batch is contiguous in both merged data and logical
        // row-scale order.  Process all rows in one VF invocation so constant
        // setup, queue synchronization and GM copy-out are amortized across
        // the batch, matching the core strategy used by DynamicMxQuant.
        LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
        auto rowDataAddr = (__ubuf__ uint8_t *)rowDataTensor.GetPhyAddr();
        auto rowScaleAddr = (__ubuf__ uint8_t *)rowScaleTensor.GetPhyAddr();
        auto rowScaleReciprocalAddr = (__ubuf__ uint16_t *)rowScaleReciprocalTensor.GetPhyAddr();

        LocalTensor<uint16_t> maxExpTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
        auto maxExpAddr = (__ubuf__ uint16_t *)maxExpTensor.GetPhyAddr();
        ComputeRowScaleOcpBatchCompact(static_cast<uint16_t>(tileCount), mergedAddr, rowScaleAddr,
                                        rowScaleReciprocalAddr, maxExpAddr);
        if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value ||
                      IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
            ComputeRowDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(tileCount), mergedAddr,
                                rowScaleReciprocalAddr, rowDataAddr, ONCE_ROW_LEN / BLOCK_SIZE);
        } else {
            ComputeRowDataToFp8Compact(static_cast<uint16_t>(tileCount), mergedAddr, rowScaleReciprocalAddr,
                                       rowDataAddr);
        }

        rowScaleQueue_.template EnQue(rowScaleTensor);
        rowDataQueue_.template EnQue(rowDataTensor);

        int64_t firstRow = firstTile / colTileNum;
        int64_t rowDataOutOffset = firstRow * rowLength_ + firstColTile * tileRowLength_;
        int64_t rowScaleOutOffset =
            firstRow * rowScaleElementsPerRow_ + firstColTile * (tileRowLength_ / BLOCK_SIZE);
        CopyOutRowOnlyBatch(rowDataOutOffset, rowScaleOutOffset, tileCount);
        return;
    }

    if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
        // cuBLAS FP8 uses the same compact three-stage organization as
        // DynamicMxQuant: compact amax, batch scale/reciprocal, batch data.
        LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
        LocalTensor<uint16_t> maxTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
        auto rowDataAddr = (__ubuf__ uint8_t *)rowDataTensor.GetPhyAddr();
        auto rowScaleAddr = (__ubuf__ uint8_t *)rowScaleTensor.GetPhyAddr();
        auto rowScaleReciprocalAddr = (__ubuf__ uint16_t *)rowScaleReciprocalTensor.GetPhyAddr();
        auto maxAddr = (__ubuf__ uint16_t *)maxTensor.GetPhyAddr();
        ComputeRowScaleCuBLASSingleAxis(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(tileCount),
                                        mergedAddr, rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr, maxAddr, true);
        rowScaleQueue_.template EnQue(rowScaleTensor);
        rowDataQueue_.template EnQue(rowDataTensor);

        int64_t firstRow = firstTile / colTileNum;
        int64_t rowDataOutOffset = firstRow * rowLength_ + firstColTile * tileRowLength_;
        int64_t rowScaleOutOffset =
            firstRow * rowScaleElementsPerRow_ + firstColTile * (tileRowLength_ / BLOCK_SIZE);
        CopyOutRowOnlyBatch(rowDataOutOffset, rowScaleOutOffset, tileCount);
        return;
    }

    for (int64_t slot = 0; slot < tileCount; ++slot) {
        int64_t tile = firstTile + slot;
        int64_t rowTileIdx = tile / colTileNum;
        int64_t colTileIdx = tile % colTileNum;
        int64_t absRowStart = rowTileIdx * tilingData_->rowBlockSize;
        int64_t colOffset = colTileIdx * tilingData_->colBlockSize;
        int64_t calcRow = 1;
        int64_t calcCol = tilingData_->colBlockSize;
        __ubuf__ xDtype *rowMerged = mergedAddr + slot * tileRowLength_;

        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
            LocalTensor<uint16_t> colScaleReciprocalTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
            auto rowDataAddr = (__ubuf__ uint8_t *)rowDataTensor.GetPhyAddr();
            auto rowScaleAddr = (__ubuf__ uint8_t *)rowScaleTensor.GetPhyAddr();
            auto rowScaleReciprocalAddr = (__ubuf__ uint16_t *)rowScaleReciprocalTensor.GetPhyAddr();
            auto absMaxTempAddr = (__ubuf__ uint16_t *)colScaleReciprocalTensor.GetPhyAddr();
            ComputeRowScaleCuBLASSingleAxis(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow),
                                            rowMerged, rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr,
                                            absMaxTempAddr);
            rowScaleQueue_.template EnQue(rowScaleTensor);
            rowDataQueue_.template EnQue(rowDataTensor);

            int64_t rowDataOutOffset = absRowStart * rowLength_ + colOffset;
            int64_t rowScaleOutOffset = absRowStart * rowScaleElementsPerRow_ + colOffset / BLOCK_SIZE;
            CopyOutRowOnly(rowDataOutOffset, rowScaleOutOffset, calcRow, calcCol);
        } else {
            LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
            auto rowDataAddr = (__ubuf__ uint8_t *)rowDataTensor.GetPhyAddr();
            auto rowScaleAddr = (__ubuf__ uint8_t *)rowScaleTensor.GetPhyAddr();
            auto rowScaleReciprocalAddr = (__ubuf__ uint16_t *)rowScaleReciprocalTensor.GetPhyAddr();
            ComputeRowScaleOcpSingleAxis(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow),
                                         rowMerged, rowScaleAddr, rowScaleReciprocalAddr);
            if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value ||
                          IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
                ComputeRowDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow), rowMerged,
                                    rowScaleReciprocalAddr, rowDataAddr);
            } else {
                ComputeRowDataToFp8(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(calcRow), rowMerged,
                                    rowScaleReciprocalAddr, rowDataAddr);
            }
            rowScaleQueue_.template EnQue(rowScaleTensor);
            rowDataQueue_.template EnQue(rowDataTensor);

            int64_t rowDataOutOffset = absRowStart * rowLength_ + colOffset;
            int64_t rowScaleOutOffset = absRowStart * rowScaleElementsPerRow_ + colOffset / BLOCK_SIZE;
            CopyOutRowOnly(rowDataOutOffset, rowScaleOutOffset, calcRow, calcCol);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyOutRowOnlyBatch(
    int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t tileCount)
{
    LocalTensor<uint8_t> rowDataLocal = rowDataQueue_.template DeQue<uint8_t>();
    LocalTensor<uint8_t> rowScaleLocal = rowScaleQueue_.template DeQue<uint8_t>();

    int64_t dataBytesPerTile = tileRowLength_;
    int64_t rowDataOutOffsetNow = rowDataOutOffset;
    if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value ||
                  IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
        dataBytesPerTile /= DIGIT_TWO;
        rowDataOutOffsetNow /= DIGIT_TWO;
    }

    DataCopyExtParams rowDataCopy = {1, static_cast<uint32_t>(tileCount * dataBytesPerTile), 0, 0, 0};
    DataCopyPad(rowDataGm_[rowDataOutOffsetNow], rowDataLocal, rowDataCopy);

    constexpr uint32_t SCALE_BYTES_PER_TILE = ONCE_ROW_LEN / BLOCK_SIZE;
    // Full-width OCP batches produce dense scale bytes for both FP8 and FP4.
    DataCopyExtParams rowScaleCopy = {
        1, static_cast<uint32_t>(tileCount * SCALE_BYTES_PER_TILE), 0, 0, 0};
    DataCopyPad(rowScaleGm_[rowScaleOutOffset], rowScaleLocal, rowScaleCopy);
    PipeBarrier<PIPE_MTE3>();

    rowDataQueue_.template FreeTensor(rowDataLocal);
    rowScaleQueue_.template FreeTensor(rowScaleLocal);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowScaleOcpBatchCompact(
    uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *rowScaleAddr,
    __ubuf__ uint16_t *rowScaleReciprocalAddr, __ubuf__ uint16_t *maxExpAddr)
{
    // Phase 1: every 256 input values produce eight block maxima.  Store them
    // densely; UnalignRegForStore is required for a multi-iteration stream.
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
        Reg::RegTensor<uint16_t> expMax;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::RegTensor<uint16_t> expMaskFP16;
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg evenValid;
        Reg::MaskReg oddValid;
        Reg::UnalignRegForStore ureg;

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        for (uint16_t i = 0; i < blockCount; ++i) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, ONCE_ROW_LEN);
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::And(evenExpFp16, (Reg::RegTensor<uint16_t> &)evenData, expMaskFP16, maskAll);
                Reg::And(oddExpFp16, (Reg::RegTensor<uint16_t> &)oddData, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(evenValid, evenExpFp16, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(oddValid, oddExpFp16, expMaskFP16, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(evenBf16, evenData, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(oddBf16, oddData, maskAll);
                Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t> &)evenBf16, expMaskBF16, maskAll);
                Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t> &)oddBf16, expMaskBF16, maskAll);
                Reg::Select<uint16_t>(evenExpBf16, evenExpBf16, expMaskBF16, evenValid);
                Reg::Select<uint16_t>(oddExpBf16, oddExpBf16, expMaskBF16, oddValid);
            } else {
                Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t> &)evenData, expMaskBF16, maskAll);
                Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t> &)oddData, expMaskBF16, maskAll);
            }
            Reg::Max(expMax, evenExpBf16, oddExpBf16, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMax, expMax, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, expMax, ureg, ONCE_ROW_LEN / BLOCK_SIZE);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }

    PipeBarrier<PIPE_V>();

    // Phase 2: transform up to 128 compact maxima per VF.  Scale is packed to
    // uint8 in UB and reciprocal remains a dense uint16 stream (eight/row).
    uint32_t totalScale = static_cast<uint32_t>(blockCount) * (ONCE_ROW_LEN / BLOCK_SIZE);
    uint16_t scaleLoops = static_cast<uint16_t>(ops::CeilDiv(totalScale, static_cast<uint32_t>(VF_LEN_B16)));
    uint16_t outputDtypeMaxExp = outputDtypeMaxExp_;
    auto packedScaleAddr = (__ubuf__ uint16_t *)rowScaleAddr;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> maxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scale;
        Reg::RegTensor<uint16_t> reciprocal;
        Reg::RegTensor<uint16_t> expMask;
        Reg::RegTensor<uint16_t> dtypeMax;
        Reg::RegTensor<uint16_t> bias;
        Reg::RegTensor<uint16_t> nanScale;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> nanReciprocal;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::MaskReg validMask;
        Reg::MaskReg infMask;
        Reg::MaskReg nonZeroMask;
        Reg::MaskReg clampMask;
        Reg::MaskReg specialMask;

        Reg::Duplicate(expMask, EXP_MASK_BF16);
        Reg::Duplicate(dtypeMax, outputDtypeMaxExp);
        Reg::Duplicate(bias, BF16_EXP_BIAS);
        Reg::Duplicate(nanScale, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(zero, static_cast<uint16_t>(0));
        Reg::Duplicate(nanReciprocal, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < scaleLoops; ++i) {
            validMask = Reg::UpdateMask<uint16_t>(totalScale);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpAddr, VF_LEN_B16);
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, maxExp, expMask, validMask);
            Reg::Compare<uint16_t, CMPMODE::LE>(clampMask, maxExp, dtypeMax, validMask);
            Reg::Select<uint16_t>(maxExp, dtypeMax, maxExp, clampMask);
            Reg::Sub(sharedExp, maxExp, dtypeMax, validMask);
            Reg::ShiftRights(scale, sharedExp, SHR_NUM_FOR_BF16, validMask);
            Reg::Select<uint16_t>(scale, scale, nanScale, infMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                packedScaleAddr, scale, VF_LEN_FP32, validMask);

            Reg::Compare<uint16_t, CMPMODE::NE>(nonZeroMask, sharedExp, zero, validMask);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExp, bias, validMask);
            Reg::Sub(reciprocal, bias, sharedExp, validMask);
            Reg::Select<uint16_t>(reciprocal, reciprocal, nanReciprocal, infMask);
            Reg::Select<uint16_t>(reciprocal, reciprocal, zero, nonZeroMask);
            Reg::Select<uint16_t>(reciprocal, specialExp, reciprocal, specialMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                rowScaleReciprocalAddr, reciprocal, VF_LEN_B16, validMask);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowDataToFp8Compact(
    uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *rowScaleReciprocalAddr,
    __ubuf__ uint8_t *rowDataAddr)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<xDtype> evenData;
        Reg::RegTensor<xDtype> oddData;
        Reg::RegTensor<float> evenZeroFp32;
        Reg::RegTensor<float> evenOneFp32;
        Reg::RegTensor<float> oddZeroFp32;
        Reg::RegTensor<float> oddOneFp32;
        Reg::RegTensor<rowDataDtype> evenZeroFp8;
        Reg::RegTensor<rowDataDtype> evenOneFp8;
        Reg::RegTensor<rowDataDtype> oddZeroFp8;
        Reg::RegTensor<rowDataDtype> oddOneFp8;

        for (uint16_t i = 0; i < blockCount; ++i) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, ONCE_ROW_LEN);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, rowScaleReciprocalAddr, ONCE_ROW_LEN / BLOCK_SIZE);
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
                rowDataAddr, (Reg::RegTensor<uint8_t> &)evenZeroFp8, ONCE_ROW_LEN, maskAllB8);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowScaleOcpSingleAxis(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *rowScaleAddr,
    __ubuf__ uint16_t *rowScaleReciprocalAddr)
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
        Reg::MaskReg evenInfNanMask;
        Reg::MaskReg oddInfNanMask;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(yMaxExp, outputDtypeMaxExp);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(zero, static_cast<uint16_t>(0));
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, ONCE_ROW_LEN);
            if constexpr (IsSameType<xDtype, half>::value) {
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
                Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t> &)evenData, expMaskBF16, maskAll);
                Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t> &)oddData, expMaskBF16, maskAll);
            }
            Reg::Max(expMaxDim1, evenExpBf16, oddExpBf16, maskAll);
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
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(rowScaleAddr, mxScale1B8, UB_BLOCK_SIZE,
                                                                         maskReduceB8);
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);
            Reg::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                rowScaleReciprocalAddr, reversedShareExp1, ROW_SCALE_RECIPROCAL_ROW_ELEMS, maskReduceB16);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowScaleCuBLASSingleAxis(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *rowDataAddr,
    __ubuf__ uint8_t *rowScaleAddr, __ubuf__ uint16_t *rowScaleReciprocalAddr, __ubuf__ uint16_t *absMaxTempAddr,
    bool compactLayout)
{
    uint32_t outputDtypeInvMax = outputDtypeInvMax_;
    __ubuf__ xDtype *dataStartAddr = xAddr;
    __ubuf__ uint16_t *rowScaleReciprocalStartAddr = rowScaleReciprocalAddr;
    PipeBarrier<PIPE_V>();

    // Phase 1: collect row maxima. StoreUnAlignPost must end a vector scope:
    // keeping later vector loads in the same scope causes VEC UB alignment errors.
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> evenData;
        Reg::RegTensor<xDtype> oddData;
        Reg::RegTensor<uint16_t> evenAbs;
        Reg::RegTensor<uint16_t> oddAbs;
        Reg::RegTensor<uint16_t> absMaxDim1;
        Reg::RegTensor<uint16_t> absMask;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::UnalignRegForStore ureg;

        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, dataLen);
            Reg::And(evenAbs, (Reg::RegTensor<uint16_t> &)evenData, absMask, maskAll);
            Reg::And(oddAbs, (Reg::RegTensor<uint16_t> &)oddData, absMask, maskAll);
            Reg::Max(absMaxDim1, evenAbs, oddAbs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(absMaxTempAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
        }
        Reg::StoreUnAlignPost(absMaxTempAddr, ureg, 0);
    }

    PipeBarrier<PIPE_V>();

    // Phase 2: compute/scatter row scale and reciprocal from the compacted maxima.
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> absMaxDim1;
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
        uint32_t compactStoreElements = VF_LEN_FP32;
        Reg::MaskReg compactStoreMask = Reg::UpdateMask<uint16_t>(compactStoreElements);

        Reg::Duplicate(invMax, outputDtypeInvMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

        __ubuf__ uint16_t *absMadataReadAddr = absMaxTempAddr;
        uint16_t scaleCount = dataLen / BLOCK_SIZE;
        uint16_t batchCount =
            ops::CeilDiv(static_cast<uint16_t>(blockCount * scaleCount), static_cast<uint16_t>(VF_LEN_FP32));
        for (uint16_t j = 0; j < batchCount; j++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                absMaxDim1, absMadataReadAddr, VF_LEN_FP32);
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

            if (compactLayout) {
                auto packedScaleAddr = (__ubuf__ uint16_t *)rowScaleAddr;
                Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(
                    packedScaleAddr + j * (VF_LEN_FP32 / DIGIT_TWO), scaleBf16, compactStoreMask);
                Reg::StoreAlign<uint16_t>(rowScaleReciprocalAddr + j * VF_LEN_FP32, reciprocalBf16Reg,
                                          compactStoreMask);
            } else {
                for (uint16_t k = 0; k < 8; k++) {
                    Reg::Arange(extractIdx, static_cast<int8_t>(k * 8));
                    Reg::Gather(scaleU8Row, scaleU8Reg, (Reg::RegTensor<uint8_t> &)extractIdx);
                    Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                        rowScaleAddr, scaleU8Row, UB_BLOCK_SIZE, maskReduceB8);
                    Reg::Arange(extractIdx, static_cast<int8_t>(k * 16));
                    Reg::Gather((Reg::RegTensor<uint8_t> &)reciprocalBf16Row,
                                (Reg::RegTensor<uint8_t> &)reciprocalBf16Reg,
                                (Reg::RegTensor<uint8_t> &)extractIdx);
                    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                        rowScaleReciprocalAddr, reciprocalBf16Row, ROW_SCALE_RECIPROCAL_ROW_ELEMS, maskReduceB16);
                }
            }
        }
    }

    // Phase 3: quantize row data using the generated row-scale reciprocals.
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> evenData;
        Reg::RegTensor<xDtype> oddData;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
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
        uint16_t reciprocalStride =
            compactLayout ? static_cast<uint16_t>(ONCE_ROW_LEN / BLOCK_SIZE) : ROW_SCALE_RECIPROCAL_ROW_ELEMS;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, dataReadAddr, dataLen);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, rowScaleReciprocalReadAddr, reciprocalStride);
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
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyOutRowOnly(
    int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t blockCount, int64_t dataLen)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t rowDataOutOffsetNow = rowDataOutOffset;

    if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value || IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
        outBlockLen = static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t));
        rowDataOutOffsetNow = rowDataOutOffset / DIGIT_TWO;
    } else {
        outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) * sizeof(rowDataDtype) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) * sizeof(uint8_t));
    }

    DataCopyExtParams rowDataCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};
    int64_t dataLenAlign = ops::CeilDiv(dataLen, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
    uint32_t rowScaleOutLen = static_cast<uint32_t>(dataLenAlign / BLOCK_SIZE);
    DataCopyExtParams rowScaleCopyOutParams = {
        outBurst, rowScaleOutLen, static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(rowLength_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - rowScaleOutLen),
        static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> rowDataLocal = rowDataQueue_.template DeQue<uint8_t>();
    DataCopyPad(rowDataGm_[rowDataOutOffsetNow], rowDataLocal, rowDataCopyOutParams);
    rowDataQueue_.FreeTensor(rowDataLocal);

    LocalTensor<uint8_t> rowScaleLocal = rowScaleQueue_.template DeQue<uint8_t>();
    DataCopyPad(rowScaleGm_[rowScaleOutOffset], rowScaleLocal, rowScaleCopyOutParams);
    rowScaleQueue_.FreeTensor(rowScaleLocal);

}
} // namespace ClaGateQuant

#endif // OPS_NN_CLA_GATE_QUANT_SINGLE_AXIS_H
