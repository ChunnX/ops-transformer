/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cla_gate_quant_common.h
 * \brief Common building blocks for the ClaGate MX quantization kernel family
 *
 * The operator fuses a CLA gate with MX quantization:
 *   merged = global_attn * sigmoid(global_gate) + local_attn * sigmoid(local_gate)
 * where global_attn/local_attn have shape [T, N, D] and gate logits [T, N].
 *
 * After the gate merge, the result is quantized into:
 *   - dual_axis_flag=false (single-axis): row_data [T, N*D] and row_scale.
 *   - dual_axis_flag=true (dual-axis): also col_data [T, N*D] and col_scale.
 *
 * In the kernel each AIV core processes disjoint row/column tiles. A tile is a
 * rowCount-by-256-column chunk in UB; rows are processed in tileRowCount-row
 * batches.
 */

#ifndef OPS_NN_CLA_GATE_QUANT_COMMON_H
#define OPS_NN_CLA_GATE_QUANT_COMMON_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "cla_gate_quant_tiling_key.h"
#include "cla_gate_quant_tilingdata.h"

namespace ClaGateQuant {

using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;

constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t NAN_FOR_FP8_E8M0 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr uint32_t FP8_E5M2_INV_MAX = 0x37924925; // 1 / max finite FP8_E5M2 value
constexpr uint32_t FP8_E4M3_INV_MAX = 0x3b124925; // 1 / max finite FP8_E4M3 value
constexpr uint16_t EXP_MASK_BF16 = 0x7f80;
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;

// CuBLAS-like scale algorithm (scaleAlg=1) constants
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr uint32_t VF_LEN_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t VF_LEN_B16 = platform::GetVRegSize() / sizeof(half);
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BLOCK_SIZE = 64;
constexpr int64_t ONCE_ROW_LEN = 256;
constexpr int64_t UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t ROW_SCALE_RECIPROCAL_ROW_ELEMS = UB_BLOCK_SIZE / sizeof(uint16_t);
// DAV_3510 BrcbCommonImpl advances a DIST_E2B_B16 source by this many elements.
constexpr uint32_t E2B_B16_SOURCE_ELEMS = BRCB_BROADCAST_NUMBER;
constexpr uint32_t COL_SCALE_STORE_COUNT = ONCE_ROW_LEN / VF_LEN_FP32;
constexpr uint32_t COL_SCALE_STORE_STRIDE_BYTES = DIGIT_TWO * VF_LEN_FP32;
constexpr uint32_t COL_SCALE_ONE_STORE_BYTES = DIGIT_TWO * platform::GetVRegSize();
constexpr uint32_t COL_SCALE_BUFFER_BYTES =
    COL_SCALE_ONE_STORE_BYTES + (COL_SCALE_STORE_COUNT - 1U) * COL_SCALE_STORE_STRIDE_BYTES;

static_assert(ONCE_ROW_LEN % VF_LEN_FP32 == 0, "A full row must contain an integral number of vector registers");
static_assert(ROW_SCALE_RECIPROCAL_ROW_ELEMS * sizeof(uint16_t) == UB_BLOCK_SIZE,
              "Each row-scale reciprocal row must occupy exactly one UB block");
static_assert(E2B_B16_SOURCE_ELEMS * sizeof(uint16_t) <= UB_BLOCK_SIZE,
              "DIST_E2B_B16 must not read beyond one row-scale reciprocal row");
static_assert(COL_SCALE_BUFFER_BYTES == 896U, "Four overlapping DIST_INTLV_B8 stores must fit in the col-scale buffer");

static constexpr Reg::CastTrait CAST_X_TO_FP32_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CAST_X_TO_FP32_ONE = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CAST_HALF_TO_BF16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};

static constexpr Reg::CastTrait CAST_FP32_TO_FP16_BF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_80 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_81 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_82 = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_83 = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class ClaGateQuantCommonBase {
public:
    __aicore__ inline ClaGateQuantCommonBase(const ClaGateQuantTilingData *tilingData, TPipe *pipe)
        : tilingData_(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate,
                                GM_ADDR rowData, GM_ADDR rowScale, GM_ADDR colData, GM_ADDR colScale);

protected:
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, roundMode};

    __aicore__ inline void InitParams();
    __aicore__ inline void CopyInClaGate(int64_t absRowStart, int64_t colOffset, int64_t calcRow, int64_t calcCol,
                                         int64_t headsThisTile);
    __aicore__ inline void ComputeClaGateFullTile(__ubuf__ xDtype *globalAddr, __ubuf__ xDtype *localAddr,
                                                  __ubuf__ xDtype *globalGateAddr, __ubuf__ xDtype *localGateAddr,
                                                  __ubuf__ xDtype *outputAddr, uint16_t calcRows,
                                                  uint16_t headsThisTile);
    __aicore__ inline void FillZero(__ubuf__ xDtype *outputAddr, uint32_t elementCount);
    __aicore__ inline void ComputeRowDataToFp8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                               __ubuf__ uint16_t *rowScaleReciprocalAddr,
                                               __ubuf__ uint8_t *rowDataAddr);
    __aicore__ inline void ComputeRowDataToFp4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
                                               __ubuf__ uint16_t *rowScaleReciprocalAddr,
                                               __ubuf__ uint8_t *rowDataAddr,
                                               uint16_t reciprocalStride = ROW_SCALE_RECIPROCAL_ROW_ELEMS);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float> &value);

    // tiling data
    const ClaGateQuantTilingData *tilingData_;

    // pipe & queue & buf
    TPipe *pipe_;
    // Double-buffered input queues: activation halves and gate scalars are split
    // so their lifetimes/synchronization can be managed independently.
    TQue<QuePosition::VECIN, DB_BUFFER> activationInQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> gateInQueue_;
    TBuf<TPosition::VECCALC> mergedClaGateBuf_;
    // One full FP32 register slot per row. Sigmoid is calculated only in lane 0;
    // StoreAlign + LoadAlign(DIST_BRC_B32) expands that scalar for the D lanes.
    TBuf<TPosition::VECCALC> sigmoidBroadcastBuf_;
    TQue<QuePosition::VECOUT, DB_BUFFER> rowDataQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> colDataQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> rowScaleQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> colScaleQueue_;
    TBuf<TPosition::VECCALC> rowScaleReciprocalBuf_;
    TBuf<TPosition::VECCALC> colScaleReciprocalBuf_;

    // gm
    GlobalTensor<xDtype> globalOutGm_;
    GlobalTensor<xDtype> localOutGm_;
    GlobalTensor<xDtype> globalGateGm_;
    GlobalTensor<xDtype> localGateGm_;
    GlobalTensor<uint8_t> rowDataGm_;
    GlobalTensor<uint8_t> rowScaleGm_;
    GlobalTensor<uint8_t> colDataGm_;
    GlobalTensor<uint8_t> colScaleGm_;

    // base variables
    int64_t blockIdx_ = 0;
    int64_t tileRowLength_ = 0;          // blockW = 256
    int64_t tileTailLength_ = 0;         // dimNTail
    int64_t tileRowCount_ = 0;           // splitBlockH = 64
                                         // MX block size, fixed
    int64_t rowScaleElementsPerRow_ = 0; // ceil(dimN / blockSize)
    uint32_t outputDtypeInvMax_ = 0;
    uint16_t outputDtypeMaxExp_ = 0;
    int64_t halfBufferElemCount_ = 0; // per-half buffer size in elements (splitBlockH * blockW)
    int64_t rowLength_ = 0;
    int64_t headCount_ = 0;
    int64_t headSize_ = 0;
    int64_t headsPerTileRow_ = 0; // blockW / headDim, 1 or 2
    int64_t gateSideOffset_ = 0;  // headsPerRow * tileRowCount_

    int64_t ubBlockElemCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
    int64_t ubBlockElemCountB8_ = UB_BLOCK_SIZE / sizeof(uint8_t);
};

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    tileRowLength_ = ONCE_ROW_LEN; // 固定256
    tileTailLength_ = tilingData_->dimNTail;
    // Dual-axis keeps its 64-row algorithmic tile.  Single-axis receives the
    // UB-derived batch capacity from host tiling instead of assuming 64.
    tileRowCount_ = (tilingData_->dualAxisFlag == 0 && tilingData_->batchSize > 0) ?
                        tilingData_->batchSize :
                        DOUBLE_BLOCK_SIZE;

    rowLength_ = tilingData_->dimN;
    headCount_ = tilingData_->headNum;
    headSize_ = tilingData_->headDim;
    headsPerTileRow_ = tileRowLength_ / headSize_;
    int64_t logicalGateCount = headsPerTileRow_ * tileRowCount_;
    if (tilingData_->dualAxisFlag == 0) {
        // Single-axis batch sigmoid is generated a full FP32 VF (64 scalars)
        // at a time.  Keep each global/local side on a complete-VF boundary:
        // this satisfies the MTE alignment requirement and prevents the final
        // unmasked VF store from overwriting the opposite side.
        gateSideOffset_ =
            ops::CeilDiv(logicalGateCount, static_cast<int64_t>(VF_LEN_FP32)) * VF_LEN_FP32;
    } else {
        // Preserve the established dual-axis UB layout exactly.  Dynamic batch
        // sizing and its extra padding are single-axis-only optimizations.
        gateSideOffset_ = logicalGateCount;
    }

    // Set dtype-specific constants for MX quantization
    if constexpr (IsSameType<rowDataDtype, fp8_e4m3fn_t>::value) {
        outputDtypeMaxExp_ = FP8_E4M3_MAX_EXP;
        outputDtypeInvMax_ = FP8_E4M3_INV_MAX;
    } else if constexpr (IsSameType<rowDataDtype, fp8_e5m2_t>::value) {
        outputDtypeMaxExp_ = FP8_E5M2_MAX_EXP;
        outputDtypeInvMax_ = FP8_E5M2_INV_MAX;
    } else if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value) {
        outputDtypeMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::Init(
    GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate, GM_ADDR rowData, GM_ADDR rowScale,
    GM_ADDR colData, GM_ADDR colScale)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams();

    // Set up Global Memory tensors
    globalOutGm_.SetGlobalBuffer((__gm__ xDtype *)(globalOut));
    localOutGm_.SetGlobalBuffer((__gm__ xDtype *)(localOut));
    globalGateGm_.SetGlobalBuffer((__gm__ xDtype *)(globalGate));
    localGateGm_.SetGlobalBuffer((__gm__ xDtype *)(localGate));
    globalOutGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    localOutGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    rowDataGm_.SetGlobalBuffer((__gm__ uint8_t *)(rowData));
    rowScaleGm_.SetGlobalBuffer((__gm__ uint8_t *)(rowScale));
    colDataGm_.SetGlobalBuffer((__gm__ uint8_t *)(colData));
    colScaleGm_.SetGlobalBuffer((__gm__ uint8_t *)(colScale));

    halfBufferElemCount_ = tileRowLength_ * tileRowCount_;
    int64_t inBufferSize = halfBufferElemCount_ * static_cast<int64_t>(sizeof(xDtype));

    int64_t mxScale2BufferSize = tileRowLength_ * DIGIT_THREE;
    if constexpr (scaleAlg != TPL_SCALE_ALG_0) {
        mxScale2BufferSize = COL_SCALE_BUFFER_BYTES;
    }

    // axis=-1 scale buffer
    int64_t mxScale1BufferSize = tileRowCount_ * UB_BLOCK_SIZE;

    // axis=-2 1/scale (xDtype sized for bf16 reciprocal storage)
    int64_t tmpScale2BufferSize = tileRowLength_ * DIGIT_TWO * static_cast<int64_t>(sizeof(xDtype));
    if (tilingData_->dualAxisFlag == 0) {
        // Single-axis reuses this buffer for the compact max-exp stream:
        // eight uint16 values are produced for every 1x256 tile.
        int64_t compactScaleCount = tileRowCount_ * (tileRowLength_ / BLOCK_SIZE);
        // Phase 2 loads up to one complete B16 VF (128 maxima), so reserve the
        // aligned tail as well even when the logical batch is not a multiple
        // of 16 rows.
        int64_t compactMaxExpSize =
            ops::CeilDiv(compactScaleCount, static_cast<int64_t>(VF_LEN_B16)) * VF_LEN_B16 * sizeof(uint16_t);
        tmpScale2BufferSize = tmpScale2BufferSize > compactMaxExpSize ? tmpScale2BufferSize : compactMaxExpSize;
    }

    // Allocate queues with double buffering (DB_BUFFER=2).
    // activationInQueue_ holds the two 64x256 activation halves (global/local).
    // gateInQueue_ holds per-head gate scalars for global/local sides.
    int64_t actFrameBytes = inBufferSize * DIGIT_TWO;
    int64_t gateFrameBytes = DIGIT_TWO * gateSideOffset_ * static_cast<int64_t>(sizeof(xDtype));
    pipe_->InitBuffer(activationInQueue_, DB_BUFFER, actFrameBytes);
    pipe_->InitBuffer(gateInQueue_, DB_BUFFER, gateFrameBytes);
    pipe_->InitBuffer(mergedClaGateBuf_, inBufferSize);
    // Sigmoid scratch: one gateSideOffset_-element FP32 region per global/local side.
    pipe_->InitBuffer(sigmoidBroadcastBuf_, DIGIT_TWO * gateSideOffset_ * sizeof(float));
    pipe_->InitBuffer(rowDataQueue_, DB_BUFFER, halfBufferElemCount_);
    pipe_->InitBuffer(colDataQueue_, DB_BUFFER, halfBufferElemCount_);
    pipe_->InitBuffer(rowScaleQueue_, DB_BUFFER, mxScale1BufferSize);
    pipe_->InitBuffer(colScaleQueue_, DB_BUFFER, mxScale2BufferSize);
    pipe_->InitBuffer(rowScaleReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(colScaleReciprocalBuf_, tmpScale2BufferSize); // 后续检查下这些ub大小是否正确
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyInClaGate(
    int64_t absRowStart, int64_t colOffset, int64_t calcRow, int64_t calcCol, int64_t headsThisTile)
{
    LocalTensor<xDtype> activationInput = activationInQueue_.template AllocTensor<xDtype>();
    LocalTensor<xDtype> gateInput = gateInQueue_.template AllocTensor<xDtype>();
    DataCopyPadExtParams<xDtype> pad = {false, 0, 0, 0};
    int64_t outOffset = absRowStart * rowLength_ + colOffset;
    int64_t rowStride = rowLength_;
    int64_t headStart = colOffset / headSize_;

    // DataCopyPad strides: srcStride is in bytes for GM, dstStride is in 32B UB
    // blocks. The UB tile always stores rows at tileRowLength_ (256) elements, so a
    // partial-width tile must leave a destination gap after every copied row.
    uint32_t dstStride = 0;
    if (calcCol < tileRowLength_) {
        dstStride = static_cast<uint32_t>((tileRowLength_ - calcCol) / ubBlockElemCountB16_);
    }
    DataCopyExtParams outCopy = {static_cast<uint16_t>(calcRow), static_cast<uint32_t>(calcCol * sizeof(xDtype)),
                                 static_cast<uint32_t>((rowStride - calcCol) * sizeof(xDtype)), dstStride, 0};
    DataCopyPad(activationInput, globalOutGm_[outOffset], outCopy, pad);
    DataCopyPad(activationInput[halfBufferElemCount_], localOutGm_[outOffset], outCopy, pad);

    // Per-head gate scalars, one contiguous array per head.
    for (int64_t h = 0; h < headsThisTile; ++h) {
        int64_t gateOffset = absRowStart * headCount_ + headStart + h;
        if (headCount_ == 1) {
            // With a single head, all rows' gate scalars are contiguous in GM.
            DataCopyExtParams gateCopy = {1, static_cast<uint32_t>(calcRow * sizeof(xDtype)), 0, 0, 0};
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput[h * tileRowCount_], globalGateGm_[gateOffset],
                                                               gateCopy, pad);
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput[gateSideOffset_ + h * tileRowCount_],
                                                               localGateGm_[gateOffset], gateCopy, pad);
        } else {
            // Same strided copy works for both full and partial column tiles:
            // each burst fetches the same head's gate value from calcRow rows.
            DataCopyExtParams gateCopy = {static_cast<uint16_t>(calcRow), static_cast<uint32_t>(sizeof(xDtype)),
                                          static_cast<uint32_t>((headCount_ - 1) * sizeof(xDtype)), 0, 0};
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput[h * tileRowCount_], globalGateGm_[gateOffset],
                                                               gateCopy, pad);
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput[gateSideOffset_ + h * tileRowCount_],
                                                               localGateGm_[gateOffset], gateCopy, pad);
        }
    }

    // Do not wait here. The DeQue() in Process is the consumer-side sync point;
    // with depth=DB_BUFFER, the next MTE2 can overlap the current vector compute.
    activationInQueue_.template EnQue(activationInput);
    gateInQueue_.template EnQue(gateInput);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeClaGateFullTile(
    __ubuf__ xDtype *globalAddr, __ubuf__ xDtype *localAddr, __ubuf__ xDtype *globalGateAddr,
    __ubuf__ xDtype *localGateAddr, __ubuf__ xDtype *outputAddr, uint16_t calcRows, uint16_t headsThisTile)
{
    uint16_t vfsPerHead = static_cast<uint16_t>(headSize_ / VF_LEN_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> globalReg, localReg, gateReg;
        Reg::RegTensor<float> gateF, tmpF, oneF;
        Reg::RegTensor<float> globalSigmoidF, localSigmoidF;
        Reg::RegTensor<float> globalRegF, localRegF, outputRegF;
        Reg::RegTensor<xDtype> outputReg;
        Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        const uint16_t heads = headsThisTile;
        // Sigmoid scratch stores FP32 values. Keep global/local regions separated.
        __ubuf__ uint8_t *sigmoidBytes = (__ubuf__ uint8_t *)sigmoidBroadcastBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float *globalSigmoidUb = (__ubuf__ float *)sigmoidBytes;
        __ubuf__ float *localSigmoidUb = (__ubuf__ float *)(sigmoidBytes + gateSideOffset_ * sizeof(float));

        // Phase 1: vectorized sigmoid for each head's 64 per-row gate values.
        for (uint16_t head = 0; head < heads; ++head) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(gateReg, globalGateAddr + head * tileRowCount_);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, gateReg, allMask);
            Reg::Muls(tmpF, gateF, -1.0f, allMask);
            Reg::Exp(tmpF, tmpF, allMask);
            Reg::Adds(tmpF, tmpF, 1.0f, allMask);
            Reg::Duplicate(oneF, 1.0f, allMask);
            Reg::Div(globalSigmoidF, oneF, tmpF, allMask);
            Reg::StoreAlign<float>(globalSigmoidUb + head * tileRowCount_, globalSigmoidF, allMask);
        }

        for (uint16_t head = 0; head < heads; ++head) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(gateReg, localGateAddr + head * tileRowCount_);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, gateReg, allMask);
            Reg::Muls(tmpF, gateF, -1.0f, allMask);
            Reg::Exp(tmpF, tmpF, allMask);
            Reg::Adds(tmpF, tmpF, 1.0f, allMask);
            Reg::Duplicate(oneF, 1.0f, allMask);
            Reg::Div(localSigmoidF, oneF, tmpF, allMask);
            Reg::StoreAlign<float>(localSigmoidUb + head * tileRowCount_, localSigmoidF, allMask);
        }

        // Phase 2: broadcast each per-row/per-head FP32 sigmoid over its D-sized span.
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        Reg::AddrReg offset;
        for (uint16_t row = 0; row < calcRows; ++row) {
            for (uint16_t head = 0; head < heads; ++head) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigmoidF,
                                                                   globalSigmoidUb + head * tileRowCount_ + row);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigmoidF,
                                                                   localSigmoidUb + head * tileRowCount_ + row);

                for (uint16_t vf = 0; vf < vfsPerHead; ++vf) {
                    offset = Reg::CreateAddrReg<xDtype>(row, ONCE_ROW_LEN, head, headSize_, vf, VF_LEN_FP32);
                    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg, globalAddr, offset);
                    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg, localAddr, offset);

                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF, globalReg, allMask);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF, localReg, allMask);
                    Reg::Mul(globalRegF, globalRegF, globalSigmoidF, allMask);
                    Reg::Mul(localRegF, localRegF, localSigmoidF, allMask);
                    Reg::Add(outputRegF, globalRegF, localRegF, allMask);
                    Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg, outputRegF, allMask);
                    Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(outputAddr, outputReg, offset, allMask);
                }
            }
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::FillZero(
    __ubuf__ xDtype *outputAddr, uint32_t elementCount)
{
    uint16_t iterations = CeilDivision(elementCount, VF_LEN_B16);
    uint32_t remainingElements = elementCount;
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> zeroReg;
        AscendC::Reg::MaskReg mask;
        Reg::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < iterations; i++) {
            mask = AscendC::Reg::UpdateMask<xDtype>(remainingElements);
            Reg::AddrReg offset = Reg::CreateAddrReg<xDtype>(i, VF_LEN_B16);
            AscendC::Reg::StoreAlign(outputAddr, zeroReg, offset, mask);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowDataToFp8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *rowScaleReciprocalAddr,
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
        Reg::RegTensor<bfloat16_t> evenBf16;
        Reg::RegTensor<bfloat16_t> oddBf16;
        Reg::RegTensor<float> evenZeroFp32;
        Reg::RegTensor<float> evenOneFp32;
        Reg::RegTensor<float> oddZeroFp32;
        Reg::RegTensor<float> oddOneFp32;
        Reg::RegTensor<rowDataDtype> evenZeroFp8;
        Reg::RegTensor<rowDataDtype> evenOneFp8;
        Reg::RegTensor<rowDataDtype> oddZeroFp8;
        Reg::RegTensor<rowDataDtype> oddOneFp8;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, ONCE_ROW_LEN);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, rowScaleReciprocalAddr, ROW_SCALE_RECIPROCAL_ROW_ELEMS);

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

                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
            }
            AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(evenZeroFp8, evenZeroFp32, maskAll);
            AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(evenOneFp8, evenOneFp32, maskAll);
            AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(oddZeroFp8, oddZeroFp32, maskAll);
            AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(oddOneFp8, oddOneFp32, maskAll);

            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)evenZeroFp8,
                              (AscendC::Reg::RegTensor<uint8_t> &)evenZeroFp8,
                              (AscendC::Reg::RegTensor<uint8_t> &)evenOneFp8, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)evenZeroFp8,
                              (AscendC::Reg::RegTensor<uint8_t> &)evenZeroFp8,
                              (AscendC::Reg::RegTensor<uint8_t> &)oddZeroFp8, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t> &)evenZeroFp8,
                              (AscendC::Reg::RegTensor<uint8_t> &)evenZeroFp8,
                              (AscendC::Reg::RegTensor<uint8_t> &)oddOneFp8, maskAllB8);
            AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                rowDataAddr, (AscendC::Reg::RegTensor<uint8_t> &)evenZeroFp8, ONCE_ROW_LEN, maskAllB8);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowDataToFp4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *rowScaleReciprocalAddr,
    __ubuf__ uint8_t *rowDataAddr, uint16_t reciprocalStride)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMaskB8 = Reg::CreateMask<uint8_t>();
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<xDtype> evenData;
        Reg::RegTensor<xDtype> oddData;

        Reg::RegTensor<float> evenZeroFp32;
        Reg::RegTensor<float> evenOneFp32;
        Reg::RegTensor<float> oddZeroFp32;
        Reg::RegTensor<float> oddOneFp32;
        Reg::RegTensor<float> scaleForMulZeroFP32;
        Reg::RegTensor<float> scaleForMulOneFP32;

        Reg::RegTensor<bfloat16_t> evenZeroBf16;
        Reg::RegTensor<bfloat16_t> evenOneBf16;
        Reg::RegTensor<bfloat16_t> oddZeroBf16;
        Reg::RegTensor<bfloat16_t> oddOneBf16;

        Reg::RegTensor<rowDataDtype> evenFp4;
        Reg::RegTensor<rowDataDtype> oddFp4;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                evenData, oddData, xAddr, ONCE_ROW_LEN);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, rowScaleReciprocalAddr, reciprocalStride);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulZeroFP32, (Reg::RegTensor<bfloat16_t> &)scaleForMulFP16, dataMaskB16);

                // evenData cast to fp32 and multiply by 1/scale
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, dataMaskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, dataMaskB16);

                Reg::Mul(evenZeroFp32, scaleForMulZeroFP32, evenZeroFp32, dataMaskB32);
                Reg::Mul(evenOneFp32, scaleForMulZeroFP32, evenOneFp32, dataMaskB32);
                ComputeFP4FromHalf(evenZeroFp32);
                ComputeFP4FromHalf(evenOneFp32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(evenZeroBf16, evenZeroFp32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(evenOneBf16, evenOneFp32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t> &)evenZeroBf16,
                                                                        (Reg::RegTensor<uint32_t> &)evenZeroBf16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t> &)evenOneBf16,
                                                                        (Reg::RegTensor<uint32_t> &)evenOneBf16);
                Reg::Interleave(evenZeroBf16, evenOneBf16, evenZeroBf16, evenOneBf16);

                // oddData cast to fp32 and multiply by 1/scale
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, dataMaskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, dataMaskB16);

                Reg::Mul(oddZeroFp32, scaleForMulZeroFP32, oddZeroFp32, dataMaskB32);
                Reg::Mul(oddOneFp32, scaleForMulZeroFP32, oddOneFp32, dataMaskB32);
                ComputeFP4FromHalf(oddZeroFp32);
                ComputeFP4FromHalf(oddOneFp32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(oddZeroBf16, oddZeroFp32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(oddOneBf16, oddOneFp32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t> &)oddZeroBf16,
                                                                        (Reg::RegTensor<uint32_t> &)oddZeroBf16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t> &)oddOneBf16,
                                                                        (Reg::RegTensor<uint32_t> &)oddOneBf16);
                Reg::Interleave(oddZeroBf16, oddOneBf16, oddZeroBf16, oddOneBf16);

                // Interleave evenData and oddData, then cast to FP4
                Reg::Interleave(evenZeroBf16, oddZeroBf16, evenZeroBf16, oddZeroBf16);
                Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(evenFp4, evenZeroBf16, dataMaskB16);
                Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(oddFp4, oddZeroBf16, dataMaskB16);
            } else {
                // BF16 input path
                Reg::Mul(evenData, evenData, (Reg::RegTensor<xDtype> &)scaleForMulFP16, dataMaskB16);
                Reg::Mul(oddData, oddData, (Reg::RegTensor<xDtype> &)scaleForMulFP16, dataMaskB16);
                Reg::Interleave(evenData, oddData, evenData, oddData);
                Reg::Cast<rowDataDtype, xDtype, castTraitBF16toFp4>(evenFp4, evenData, dataMaskB16);
                Reg::Cast<rowDataDtype, xDtype, castTraitBF16toFp4>(oddFp4, oddData, dataMaskB16);
            }

            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                rowDataAddr, (Reg::RegTensor<uint8_t> &)evenFp4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                rowDataAddr, (Reg::RegTensor<uint8_t> &)oddFp4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        }
    }
    return;
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeFP4FromHalf(
    Reg::RegTensor<float> &value)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, NEG_ZERO);

    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t> &)value, negZero, pregAll32);
    if constexpr (IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
        // E1M2: multiply by 4, truncate, divide by 4
        Reg::Muls(value, value, FOUR, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, value, 0, pregAll32);
        Reg::Truncate<float, roundMode>(value, value, pregAll32);
        Reg::Muls(value, value, ONE_FOURTH, pregAll32);
    } else {
        // E2M1: bit-level exponent manipulation for correct rounding
        Reg::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t> &)value, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        Reg::Mul(value, value, (Reg::RegTensor<float> &)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, value, 0, pregAll32);
        Reg::Truncate<float, roundMode>(value, value, pregAll32);
        Reg::Mul(value, value, (Reg::RegTensor<float> &)exp0FP32, pregAll32);
    }
    // Handle negative zero
    Reg::Compares<float, CMPMODE::EQ>(zeroMask, value, 0, pregAll32);
    Reg::And(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::Or(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t> &)value, negZero, (Reg::RegTensor<int32_t> &)value, zeroMask);
}
} // namespace ClaGateQuant

#endif // OPS_NN_CLA_GATE_QUANT_COMMON_H
