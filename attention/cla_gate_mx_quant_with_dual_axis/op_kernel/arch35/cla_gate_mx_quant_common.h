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
 * \file cla_gate_mx_quant_common.h
 * \brief Common building blocks for the ClaGate MX quantization kernel family
 *
 * Input x is [M, 2N] (left half [M, N] and right half [M, N] for SwiGLU).
 * group_index is [numGroups] cumsum int64 defining group boundaries.
 * Outputs: y1, mx_scale1 (axis=-1 quantization), y2, mx_scale2 (axis=-2 quantization).
 *
 * Each task = (group, columnBlock). Within a task, iterate over splitBlockH-row chunks.
 */

#ifndef OPS_NN_CLA_GATE_MX_QUANT_COMMON_H
#define OPS_NN_CLA_GATE_MX_QUANT_COMMON_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "cla_gate_mx_quant_with_dual_axis_tiling_key.h"
#include "cla_gate_mx_quant_with_dual_axis_tilingdata.h"

namespace ClaGateMxQuantWithDualAxis {

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
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;
constexpr uint16_t EXP_MASK_BF16 = 0x7f80;
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;

// CLUB (scaleAlg=1) specific constants
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
constexpr uint32_t SCALE1_RECIPROCAL_ROW_ELEMS = UB_BLOCK_SIZE / sizeof(uint16_t);
// DAV_3510 BrcbCommonImpl advances a DIST_E2B_B16 source by this many elements.
constexpr uint32_t E2B_B16_SOURCE_ELEMS = BRCB_BROADCAST_NUMBER;
constexpr uint32_t CUBLAS_SCALE2_STORE_COUNT = ONCE_ROW_LEN / VF_LEN_FP32;
constexpr uint32_t CUBLAS_SCALE2_STORE_STRIDE_BYTES = DIGIT_TWO * VF_LEN_FP32;
constexpr uint32_t CUBLAS_SCALE2_ONE_STORE_BYTES = DIGIT_TWO * platform::GetVRegSize();
constexpr uint32_t CUBLAS_SCALE2_BUFFER_BYTES = CUBLAS_SCALE2_ONE_STORE_BYTES +
                                                (CUBLAS_SCALE2_STORE_COUNT - 1U) * CUBLAS_SCALE2_STORE_STRIDE_BYTES;

static_assert(ONCE_ROW_LEN % VF_LEN_FP32 == 0, "A full row must contain an integral number of vector registers");
static_assert(SCALE1_RECIPROCAL_ROW_ELEMS * sizeof(uint16_t) == UB_BLOCK_SIZE,
              "Each scale1 reciprocal row must occupy exactly one UB block");
static_assert(E2B_B16_SOURCE_ELEMS * sizeof(uint16_t) <= UB_BLOCK_SIZE,
              "DIST_E2B_B16 must not read beyond one scale1 reciprocal row");
static_assert(CUBLAS_SCALE2_BUFFER_BYTES == 896U,
              "Four overlapping DIST_INTLV_B8 stores must fit in the scale2 buffer");

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

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
class ClaGateMxQuantCommonBase {
public:
    __aicore__ inline ClaGateMxQuantCommonBase(const ClaGateMxQuantWithDualAxisTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate,
                                GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2);

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
    __aicore__ inline void ComputeClaGateFullTile(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                  __ubuf__ xDtype* globalGateAddr, __ubuf__ xDtype* localGateAddr,
                                                  __ubuf__ xDtype* outputAddr, uint16_t calcRows,
                                                  uint16_t headsThisTile);
    __aicore__ inline void PadZeroM(__ubuf__ xDtype* claGateOutAddr, uint32_t num);
    __aicore__ inline void ComputeY1ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY1ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);

    // tiling data
    const ClaGateMxQuantWithDualAxisTilingData* tilingData_;

    // pipe & queue & buf
    TPipe* pipe_;
    // Double-buffered input queues: activation halves and gate scalars are split
    // so their lifetimes/synchronization can be managed independently.
    TQue<QuePosition::VECIN, DB_BUFFER> actInQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> gateInQueue_;
    TBuf<TPosition::VECCALC> claGateBuf_;
    // One full FP32 register slot per row. Sigmoid is calculated only in lane 0;
    // StoreAlign + LoadAlign(DIST_BRC_B32) expands that scalar for the D lanes.
    TBuf<TPosition::VECCALC> sigmoidBroadcastBuf_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue1_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue2_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue1_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue2_;
    TBuf<TPosition::VECCALC> mxScale1ReciprocalBuf_;
    TBuf<TPosition::VECCALC> mxScale2ReciprocalBuf_;

    // gm
    GlobalTensor<xDtype> globalOutGm_;
    GlobalTensor<xDtype> localOutGm_;
    GlobalTensor<xDtype> globalGateGm_;
    GlobalTensor<xDtype> localGateGm_;
    GlobalTensor<uint8_t> yGm1_;
    GlobalTensor<uint8_t> mxScaleGm1_;
    GlobalTensor<uint8_t> yGm2_;
    GlobalTensor<uint8_t> mxScaleGm2_;

    // base variables
    int64_t blockIdx_ = 0;
    int64_t ubRowLen_ = 0;        // blockW = 256
    int64_t ubRowLenTail_ = 0;    // dimNTail
    int64_t ubRowCount_ = 0;      // splitBlockH = 64
                                  // MX block size, fixed
    int64_t dimNeg1ScaleNum_ = 0; // ceil(dimN / blockSize)
    uint32_t invDtypeMax_ = 0;
    uint16_t dtypeYMaxExp_ = 0;
    int64_t inHalfSize_ = 0; // per-half buffer size in elements (splitBlockH * blockW)
    int64_t dimN_ = 0;
    int64_t headNum_ = 0;
    int64_t headDim_ = 0;
    int64_t headsPerRow_ = 0;    // blockW / headDim, 1 or 2
    int64_t gateSideStride_ = 0; // headsPerRow * ubRowCount_

    int64_t oneBlockCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
    int64_t oneBlockCountB8_ = UB_BLOCK_SIZE / sizeof(uint8_t);
};

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    ubRowLen_ = ONCE_ROW_LEN; // 固定256
    ubRowLenTail_ = tilingData_->dimNTail;
    ubRowCount_ = DOUBLE_BLOCK_SIZE; // 固定64

    dimN_ = tilingData_->dimN;
    headNum_ = tilingData_->headNum;
    headDim_ = tilingData_->headDim;
    headsPerRow_ = ubRowLen_ / headDim_;
    gateSideStride_ = headsPerRow_ * ubRowCount_;

    // Set dtype-specific constants for MX quantization
    if constexpr (IsSameType<y1Dtype, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX;
    } else if constexpr (IsSameType<y1Dtype, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX;
    } else if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Init(
    GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams();

    // Set up Global Memory tensors
    globalOutGm_.SetGlobalBuffer((__gm__ xDtype*)(globalOut));
    localOutGm_.SetGlobalBuffer((__gm__ xDtype*)(localOut));
    globalGateGm_.SetGlobalBuffer((__gm__ xDtype*)(globalGate));
    localGateGm_.SetGlobalBuffer((__gm__ xDtype*)(localGate));
    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {

    }
    globalOutGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    localOutGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    yGm1_.SetGlobalBuffer((__gm__ uint8_t*)(y1));
    mxScaleGm1_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale1));
    yGm2_.SetGlobalBuffer((__gm__ uint8_t*)(y2));
    mxScaleGm2_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale2));

    inHalfSize_ = ubRowLen_ * ubRowCount_;
    int64_t inBufferSize = inHalfSize_ * static_cast<int64_t>(sizeof(xDtype));

    int64_t mxScale2BufferSize = ubRowLen_ * DIGIT_THREE;
    if constexpr (scaleAlg != TPL_SCALE_ALG_0) {
        mxScale2BufferSize = CUBLAS_SCALE2_BUFFER_BYTES;
    }

    // axis=-1 scale buffer
    int64_t mxScale1BufferSize = ubRowCount_ * UB_BLOCK_SIZE;

    // axis=-2 1/scale (xDtype sized for bf16 reciprocal storage)
    int64_t tmpScale2BufferSize = ubRowLen_ * DIGIT_TWO * static_cast<int64_t>(sizeof(xDtype));

    // Allocate queues with double buffering (DB_BUFFER=2).
    // actInQueue_ holds the two 64x256 activation halves (global/local).
    // gateInQueue_ holds per-head gate scalars for global/local sides.
    int64_t actFrameBytes = inBufferSize * DIGIT_TWO;
    int64_t gateFrameBytes = DIGIT_TWO * gateSideStride_ * static_cast<int64_t>(sizeof(xDtype));
    pipe_->InitBuffer(actInQueue_, DB_BUFFER, actFrameBytes);
    pipe_->InitBuffer(gateInQueue_, DB_BUFFER, gateFrameBytes);
    pipe_->InitBuffer(claGateBuf_, inBufferSize);
    // Sigmoid scratch: one gateSideStride_-element FP32 region per global/local side.
    pipe_->InitBuffer(sigmoidBroadcastBuf_, DIGIT_TWO * gateSideStride_ * sizeof(float));
    pipe_->InitBuffer(outQueue1_, DB_BUFFER, inHalfSize_);
    pipe_->InitBuffer(outQueue2_, DB_BUFFER, inHalfSize_);
    pipe_->InitBuffer(mxScaleQueue1_, DB_BUFFER, mxScale1BufferSize);
    pipe_->InitBuffer(mxScaleQueue2_, DB_BUFFER, mxScale2BufferSize);
    pipe_->InitBuffer(mxScale1ReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(mxScale2ReciprocalBuf_, tmpScale2BufferSize); // 后续检查下这些ub大小是否正确
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::CopyInClaGate(int64_t absRowStart, int64_t colOffset,
                                                                               int64_t calcRow, int64_t calcCol,
                                                                               int64_t headsThisTile)
{
    LocalTensor<xDtype> actInput = actInQueue_.template AllocTensor<xDtype>();
    LocalTensor<xDtype> gateInput = gateInQueue_.template AllocTensor<xDtype>();
    DataCopyPadExtParams<xDtype> pad = {false, 0, 0, 0};
    int64_t outOffset = absRowStart * dimN_ + colOffset;
    int64_t rowStride = dimN_;
    int64_t headStart = colOffset / headDim_;

    // DataCopyPad strides: srcStride is in bytes for GM, dstStride is in 32B UB
    // blocks. The UB tile always stores rows at ubRowLen_ (256) elements, so a
    // partial-width tile must leave a destination gap after every copied row.
    uint32_t dstStride = 0;
    if (calcCol < ubRowLen_) {
        dstStride = static_cast<uint32_t>((ubRowLen_ - calcCol) / oneBlockCountB16_);
    }
    DataCopyExtParams outCopy = {static_cast<uint16_t>(calcRow),
                                 static_cast<uint32_t>(calcCol * sizeof(xDtype)),
                                 static_cast<uint32_t>((rowStride - calcCol) * sizeof(xDtype)), dstStride, 0};
    DataCopyPad(actInput, globalOutGm_[outOffset], outCopy, pad);
    DataCopyPad(actInput[inHalfSize_], localOutGm_[outOffset], outCopy, pad);


    // Per-head gate scalars, one contiguous array per head.
    for (int64_t h = 0; h < headsThisTile; ++h) {
        int64_t gateOffset = absRowStart * headNum_ + headStart + h;
        if (headNum_ == 1) {
            // With a single head, all rows' gate scalars are contiguous in GM.
            DataCopyExtParams gateCopy = {1, static_cast<uint32_t>(calcRow * sizeof(xDtype)), 0, 0, 0};
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput[h * ubRowCount_],
                                                               globalGateGm_[gateOffset], gateCopy, pad);
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(
                gateInput[gateSideStride_ + h * ubRowCount_],
                localGateGm_[gateOffset], gateCopy, pad);
        } else {
            // Same strided copy works for both full and partial column tiles:
            // each burst fetches the same head's gate value from calcRow rows.
            DataCopyExtParams gateCopy = {static_cast<uint16_t>(calcRow),
                                          static_cast<uint32_t>(sizeof(xDtype)),
                                          static_cast<uint32_t>((headNum_ - 1) * sizeof(xDtype)), 0, 0};
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput[h * ubRowCount_],
                                                               globalGateGm_[gateOffset], gateCopy, pad);
            DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(
                gateInput[gateSideStride_ + h * ubRowCount_],
                localGateGm_[gateOffset], gateCopy, pad);
        }
    }

    // Do not wait here. The DeQue() in Process is the consumer-side sync point;
    // with depth=DB_BUFFER, the next MTE2 can overlap the current vector compute.
    actInQueue_.template EnQue(actInput);
    gateInQueue_.template EnQue(gateInput);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeClaGateFullTile(
    __ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr, __ubuf__ xDtype* globalGateAddr,
    __ubuf__ xDtype* localGateAddr, __ubuf__ xDtype* outputAddr, uint16_t calcRows,
    uint16_t headsThisTile)
{
    uint16_t vfsPerHead = static_cast<uint16_t>(headDim_ / VF_LEN_FP32);
    __VEC_SCOPE__ {
        Reg::RegTensor<xDtype> globalReg, localReg, gateReg;
        Reg::RegTensor<float> gateF, tmpF, oneF;
        Reg::RegTensor<float> globalSigmoidF, localSigmoidF;
        Reg::RegTensor<float> globalRegF, localRegF, outputRegF;
        Reg::RegTensor<xDtype> outputReg;
        Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        const uint16_t heads = headsThisTile;
        // Sigmoid scratch stores FP32 values. Keep global/local regions separated.
        __ubuf__ uint8_t* sigmoidBytes = (__ubuf__ uint8_t*)sigmoidBroadcastBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* globalSigmoidUb = (__ubuf__ float*)sigmoidBytes;
        __ubuf__ float* localSigmoidUb = (__ubuf__ float*)(sigmoidBytes + gateSideStride_ * sizeof(float));

        // Phase 1: vectorized sigmoid for each head's 64 per-row gate values.
        for (uint16_t head = 0; head < heads; ++head) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(gateReg, globalGateAddr + head * ubRowCount_);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, gateReg, allMask);
            Reg::Muls(tmpF, gateF, -1.0f, allMask);
            Reg::Exp(tmpF, tmpF, allMask);
            Reg::Adds(tmpF, tmpF, 1.0f, allMask);
            Reg::Duplicate(oneF, 1.0f, allMask);
            Reg::Div(globalSigmoidF, oneF, tmpF, allMask);
            Reg::StoreAlign<float>(globalSigmoidUb + head * ubRowCount_, globalSigmoidF, allMask);
        }

        for (uint16_t head = 0; head < heads; ++head) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(gateReg, localGateAddr + head * ubRowCount_);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, gateReg, allMask);
            Reg::Muls(tmpF, gateF, -1.0f, allMask);
            Reg::Exp(tmpF, tmpF, allMask);
            Reg::Adds(tmpF, tmpF, 1.0f, allMask);
            Reg::Duplicate(oneF, 1.0f, allMask);
            Reg::Div(localSigmoidF, oneF, tmpF, allMask);
            Reg::StoreAlign<float>(localSigmoidUb + head * ubRowCount_, localSigmoidF, allMask);
        }

        // Phase 2: broadcast each per-row/per-head FP32 sigmoid over its D-sized span.
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        Reg::AddrReg offset;
        for (uint16_t row = 0; row < calcRows; ++row) {
            for (uint16_t head = 0; head < heads; ++head) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(
                    globalSigmoidF, globalSigmoidUb + head * ubRowCount_ + row);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(
                    localSigmoidF, localSigmoidUb + head * ubRowCount_ + row);

                for (uint16_t vf = 0; vf < vfsPerHead; ++vf) {
                    offset = Reg::CreateAddrReg<xDtype>(
                        row, ONCE_ROW_LEN,
                        head, headDim_,
                        vf, VF_LEN_FP32);
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

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::PadZeroM(
    __ubuf__ xDtype* claGateOutAddr, uint32_t num)
{
    uint16_t times = CeilDivision(num, VF_LEN_B16);
    uint32_t size = num;
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> zeroReg;
        AscendC::Reg::MaskReg mask;
        Reg::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < times; i++) {
            mask = AscendC::Reg::UpdateMask<xDtype>(size);
            Reg::AddrReg offset = Reg::CreateAddrReg<xDtype>(i, VF_LEN_B16);
            AscendC::Reg::StoreAlign(claGateOutAddr, zeroReg, offset, mask);
        }
    }
}


template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY1ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    int64_t localVlForHalfNumber = VF_LEN_B16;
    int64_t localUBBlockSize = UB_BLOCK_SIZE;

    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<y1Dtype> x0ZeroFP8;
        Reg::RegTensor<y1Dtype> x0OneFP8;
        Reg::RegTensor<y1Dtype> x1ZeroFP8;
        Reg::RegTensor<y1Dtype> x1OneFP8;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       ONCE_ROW_LEN);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

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

                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_82>(x0OneFP8, x0OneFP32, maskAll);
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_83>(x1OneFP8, x1OneFP32, maskAll);

            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8, (AscendC::Reg::RegTensor<uint8_t>&)x0OneFP8,
                              maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x1ZeroFP8, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8, (AscendC::Reg::RegTensor<uint8_t>&)x1OneFP8,
                              maskAllB8);
            AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                y1Addr, (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8, ONCE_ROW_LEN, maskAllB8);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY1ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    int64_t localVlForHalfNumber = VF_LEN_B16;
    int64_t localUBBlockSize = UB_BLOCK_SIZE;

    __VEC_SCOPE__
    {
        Reg::MaskReg dataMaskB8 = Reg::CreateMask<uint8_t>();
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;

        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<float> scaleForMulZeroFP32;
        Reg::RegTensor<float> scaleForMulOneFP32;

        Reg::RegTensor<bfloat16_t> x0ZeroBF16;
        Reg::RegTensor<bfloat16_t> x0OneBF16;
        Reg::RegTensor<bfloat16_t> x1ZeroBF16;
        Reg::RegTensor<bfloat16_t> x1OneBF16;

        Reg::RegTensor<y1Dtype> x0FP4;
        Reg::RegTensor<y1Dtype> x1FP4;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       ONCE_ROW_LEN);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulZeroFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

                // x0 cast to fp32 and multiply by 1/scale
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, dataMaskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, dataMaskB16);

                Reg::Mul(x0ZeroFP32, scaleForMulZeroFP32, x0ZeroFP32, dataMaskB32);
                Reg::Mul(x0OneFP32, scaleForMulZeroFP32, x0OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x0ZeroFP32);
                ComputeFP4FromHalf(x0OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0ZeroBF16, x0ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0OneBF16, x0OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0ZeroBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0OneBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0OneBF16);
                Reg::Interleave(x0ZeroBF16, x0OneBF16, x0ZeroBF16, x0OneBF16);

                // x1 cast to fp32 and multiply by 1/scale
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, dataMaskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, dataMaskB16);

                Reg::Mul(x1ZeroFP32, scaleForMulZeroFP32, x1ZeroFP32, dataMaskB32);
                Reg::Mul(x1OneFP32, scaleForMulZeroFP32, x1OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x1ZeroFP32);
                ComputeFP4FromHalf(x1OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1ZeroBF16, x1ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1OneBF16, x1OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1ZeroBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1OneBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1OneBF16);
                Reg::Interleave(x1ZeroBF16, x1OneBF16, x1ZeroBF16, x1OneBF16);

                // Interleave x0 and x1, then cast to FP4
                Reg::Interleave(x0ZeroBF16, x1ZeroBF16, x0ZeroBF16, x1ZeroBF16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x0FP4, x0ZeroBF16, dataMaskB16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x1FP4, x1ZeroBF16, dataMaskB16);
            } else {
                // BF16 input path
                Reg::Mul(x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                Reg::Mul(x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                Reg::Interleave(x0, x1, x0, x1);
                Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x0FP4, x0, dataMaskB16);
                Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x1FP4, x1, dataMaskB16);
            }

            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        }
    }
    return;
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantCommonBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeFP4FromHalf(Reg::RegTensor<float>& Reg)
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

    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        // E1M2: multiply by 4, truncate, divide by 4
        Reg::Muls(Reg, Reg, FOUR, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        // E2M1: bit-level exponent manipulation for correct rounding
        Reg::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    // Handle negative zero
    Reg::Compares<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::And(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::Or(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}
} // namespace ClaGateMxQuantWithDualAxis

#endif // OPS_NN_CLA_GATE_MX_QUANT_COMMON_H
