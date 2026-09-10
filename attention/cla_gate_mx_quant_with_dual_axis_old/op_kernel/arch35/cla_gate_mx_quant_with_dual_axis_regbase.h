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
 * \file cla_gate_mx_quant_with_dual_axis_regbase.h
 * \brief Fused SwiGLU + grouped dual-axis MX quantization kernel for Ascend950 (regbase)
 *
 * Input x is [M, 2N] (left half [M, N] and right half [M, N] for SwiGLU).
 * group_index is [numGroups] cumsum int64 defining group boundaries.
 * Outputs: y1, mx_scale1 (axis=-1 quantization), y2, mx_scale2 (axis=-2 quantization).
 *
 * Each task = (group, columnBlock). Within a task, iterate over splitBlockH-row chunks.
 */

#ifndef OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H
#define OPS_NN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H

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
constexpr int64_t CLA_HEAD_NUM = 96;
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
class ClaGateMxQuantWithDualAxisBase {
public:
    __aicore__ inline ClaGateMxQuantWithDualAxisBase(const ClaGateMxQuantWithDualAxisTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2,
                                GM_ADDR mxScale2);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void CopyInClaGate(int64_t absRowStart, int64_t colOffset, int64_t calcRow, int64_t calcCol);
    __aicore__ inline void ComputeClaGateFullTile(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                 __ubuf__ xDtype* globalGateAddr, __ubuf__ xDtype* localGateAddr,
                                                 __ubuf__ xDtype* outputAddr);
    __aicore__ inline void PadZeroM(__ubuf__ xDtype* claGateOutAddr, uint32_t num);
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
    __aicore__ inline void ComputeY1ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY1ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeY2ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset,
                                   int64_t blockCount, int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign);
    __aicore__ inline void ComputeInterleave(__ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr,
                                             __ubuf__ uint8_t* src1Addr);

protected:
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, roundMode};

private:
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

    int64_t oneBlockCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
    int64_t oneBlockCountB8_ = UB_BLOCK_SIZE / sizeof(uint8_t);
};

// ============================================================================
// InitParams — cache tiling parameters, set dtype constants
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    ubRowLen_ = ONCE_ROW_LEN; // 固定256
    ubRowLenTail_ = tilingData_->dimNTail;
    ubRowCount_ = DOUBLE_BLOCK_SIZE; // 固定64
    
    dimN_ = tilingData_->dimN;

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

// ============================================================================
// Init — set up GlobalTensors, allocate UB buffers
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Init(
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
    // gateInQueue_ holds the two compact per-row gate vectors (global/local).
    int64_t actFrameBytes = inBufferSize * DIGIT_TWO;
    int64_t gateFrameBytes = DIGIT_TWO * ubRowCount_ * static_cast<int64_t>(sizeof(xDtype));
    pipe_->InitBuffer(actInQueue_, DB_BUFFER, actFrameBytes);
    pipe_->InitBuffer(gateInQueue_, DB_BUFFER, gateFrameBytes);
    pipe_->InitBuffer(claGateBuf_, inBufferSize);
    // Sigmoid scratch: one 64-element region per global/local side.
    pipe_->InitBuffer(sigmoidBroadcastBuf_, DIGIT_TWO * DOUBLE_BLOCK_SIZE * sizeof(xDtype));
    pipe_->InitBuffer(outQueue1_, DB_BUFFER, inHalfSize_);
    pipe_->InitBuffer(outQueue2_, DB_BUFFER, inHalfSize_);
    pipe_->InitBuffer(mxScaleQueue1_, DB_BUFFER, mxScale1BufferSize);
    pipe_->InitBuffer(mxScaleQueue2_, DB_BUFFER, mxScale2BufferSize);
    pipe_->InitBuffer(mxScale1ReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(mxScale2ReciprocalBuf_, tmpScale2BufferSize); // 后续检查下这些ub大小是否正确
}

// ============================================================================
// Process — outer loop over groups; two core-distribution scenarios:
//   Scenario 1 (nSplitNum < usedCoreNum): groups rotate across core ranges.
//     Each group's blocks are assigned starting from a rotating core offset.
//     e.g. group0: core 0..31, group1: core 32..63, group2: core 0..31, ...
//   Scenario 2 (nSplitNum >= usedCoreNum): each group distributes all its
//     blocks across all usedCoreNum cores (same as dynamic_mx_quant_with_dual_axis).
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Process()
{
    dimNeg1ScaleNum_ = ops::CeilDiv(dimN_, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
    int64_t rowBlocks = ops::CeilDiv(tilingData_->dimM, ubRowCount_);
    int64_t totalBlocks = rowBlocks * tilingData_->dimNBlockNum;
    int64_t coreCount = tilingData_->usedCoreNum;
    int64_t loops = totalBlocks / coreCount + (blockIdx_ < totalBlocks % coreCount ? 1 : 0);
    int64_t firstBlock = blockIdx_ * (totalBlocks / coreCount) +
                         (blockIdx_ < totalBlocks % coreCount ? blockIdx_ : totalBlocks % coreCount);

    // Prologue: issue the first MTE2 transfer before entering the compute loop.
    if (loops > 0) {
        int64_t firstRowBlockIdx = firstBlock / tilingData_->dimNBlockNum;
        int64_t firstColBlockIdx = firstBlock % tilingData_->dimNBlockNum;
        int64_t firstAbsRowStart = firstRowBlockIdx * ubRowCount_;
        int64_t firstColOffset = firstColBlockIdx * ubRowLen_;
        CopyInClaGate(firstAbsRowStart, firstColOffset, ubRowCount_, ubRowLen_);
    }

    for (int64_t i = 0; i < loops; ++i) {
        int64_t block = firstBlock + i;
        int64_t rowBlockIdx = block / tilingData_->dimNBlockNum;
        int64_t colBlockIdx = block % tilingData_->dimNBlockNum;
        int64_t absRowStart = rowBlockIdx * ubRowCount_;
        int64_t colOffset = colBlockIdx * ubRowLen_;

        // Consume the double-buffered input. DeQue waits only when the
        // corresponding MTE2 transfer has not completed yet.
        LocalTensor<xDtype> input = actInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> gateInput = gateInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> merged = claGateBuf_.template Get<xDtype>();
        auto globalAddr = (__ubuf__ xDtype*)input.GetPhyAddr();
        auto localAddr = (__ubuf__ xDtype*)input[inHalfSize_].GetPhyAddr();
        auto globalGateAddr = (__ubuf__ xDtype*)gateInput.GetPhyAddr();
        auto localGateAddr = (__ubuf__ xDtype*)gateInput[ubRowCount_].GetPhyAddr();
        auto mergedAddr = (__ubuf__ xDtype*)merged.GetPhyAddr();

        // Prefetch the next block while the current block is still being computed.
        if (i + 1 < loops) {
            int64_t nextBlock = firstBlock + i + 1;
            int64_t nextRowBlockIdx = nextBlock / tilingData_->dimNBlockNum;
            int64_t nextColBlockIdx = nextBlock % tilingData_->dimNBlockNum;
            int64_t nextAbsRowStart = nextRowBlockIdx * ubRowCount_;
            int64_t nextColOffset = nextColBlockIdx * ubRowLen_;
            CopyInClaGate(nextAbsRowStart, nextColOffset, ubRowCount_, ubRowLen_);
        }

        ComputeClaGateFullTile(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr);

        actInQueue_.template FreeTensor(input);
        gateInQueue_.template FreeTensor(gateInput);

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

        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            ComputeScaleCuBLAS(ONCE_ROW_LEN, DOUBLE_BLOCK_SIZE, mergedAddr, y1Addr, scale1Addr,
                               scale1RecipAddr, scale2Addr, scale2RecipAddr);
        } else {
            for (int64_t part = 0; part < DIGIT_TWO; ++part) {
                int64_t dataOffset = part * BLOCK_SIZE * ubRowLen_;
                int64_t scale1Offset = part * BLOCK_SIZE * ops::CeilAlign(ubRowLen_ / BLOCK_SIZE, oneBlockCountB8_);
                int64_t scale2Offset = part * ubRowLen_;
                int64_t recip1Offset = part * BLOCK_SIZE * ops::CeilAlign(ubRowLen_ / BLOCK_SIZE, oneBlockCountB16_);
                int64_t recip2Offset = part * ubRowLen_;
                ComputeScaleOcp(ONCE_ROW_LEN, BLOCK_SIZE, mergedAddr + dataOffset, scale1Addr + scale1Offset,
                                scale1RecipAddr + recip1Offset, scale2Addr + scale2Offset,
                                scale2RecipAddr + recip2Offset);
                ComputeY1ToFP8(ONCE_ROW_LEN, BLOCK_SIZE, mergedAddr + dataOffset,
                               scale1RecipAddr + recip1Offset, y1Addr + dataOffset);
                ComputeY2ToFP8(ONCE_ROW_LEN, BLOCK_SIZE, mergedAddr + dataOffset,
                               scale2RecipAddr + recip2Offset, y2Addr + dataOffset);
            }
            ComputeInterleave(scale2Addr, scale2Addr, scale2Addr + ubRowLen_);
        }
        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            for (int64_t part = 0; part < DIGIT_TWO; ++part) {
                int64_t dataOffset = part * BLOCK_SIZE * ubRowLen_;
                ComputeY2ToFP8(ONCE_ROW_LEN, BLOCK_SIZE, mergedAddr + dataOffset,
                               scale2RecipAddr + part * ubRowLen_, y2Addr + dataOffset);
            }
        }

        mxScaleQueue1_.template EnQue(scale1);
        outQueue1_.template EnQue(y1);
        mxScaleQueue2_.template EnQue(scale2);
        outQueue2_.template EnQue(y2);
        int64_t yOffset = absRowStart * dimN_ + colOffset;
        int64_t scale1Offset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
        int64_t scale2Offset = rowBlockIdx * DIGIT_TWO * dimN_ + colOffset * DIGIT_TWO;

        CopyOut(yOffset, scale1Offset, scale2Offset, DOUBLE_BLOCK_SIZE, DOUBLE_BLOCK_SIZE,
                ONCE_ROW_LEN, ONCE_ROW_LEN);
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
// Load two attention tiles and the two compact gate vectors (one scalar per row).
// For the supported D=256, each column tile is exactly one head, so only that
// head's 64 gate values are copied into UB.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::CopyInClaGate(int64_t absRowStart, int64_t colOffset,
                                                                               int64_t calcRow, int64_t calcCol)
{
    LocalTensor<xDtype> actInput = actInQueue_.template AllocTensor<xDtype>();
    LocalTensor<xDtype> gateInput = gateInQueue_.template AllocTensor<xDtype>();
    DataCopyExtParams outCopy = {static_cast<uint16_t>(calcRow),
                                 static_cast<uint32_t>(calcCol * sizeof(xDtype)),
                                 static_cast<uint32_t>((dimN_ - calcCol) * sizeof(xDtype)), 0, 0};
    DataCopyPadExtParams<xDtype> pad = {false, 0, 0, 0};
    int64_t outOffset = absRowStart * dimN_ + colOffset;
    // Copy only the current head's gate column. Each row contributes one
    // bf16/fp16 scalar, so the UB layout is a compact 64-element array.
    int64_t headIdx = colOffset / ONCE_ROW_LEN;
    DataCopyExtParams gateCopy = {static_cast<uint16_t>(calcRow),
                                  static_cast<uint32_t>(sizeof(xDtype)),
                                  static_cast<uint32_t>((CLA_HEAD_NUM - 1) * sizeof(xDtype)),
                                  0, 0};
    int64_t gateOffset = absRowStart * CLA_HEAD_NUM + headIdx;
    DataCopyPad(actInput, globalOutGm_[outOffset], outCopy, pad);
    DataCopyPad(actInput[inHalfSize_], localOutGm_[outOffset], outCopy, pad);
    DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput, globalGateGm_[gateOffset],
                                                       gateCopy, pad);
    DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(
        gateInput[ubRowCount_], localGateGm_[gateOffset], gateCopy, pad);
    // Do not wait here. The DeQue() in Process is the consumer-side sync point;
    // with depth=DB_BUFFER, the next MTE2 can overlap the current vector compute.
    actInQueue_.template EnQue(actInput);
    gateInQueue_.template EnQue(gateInput);
}

// merged = sigmoid(global_gate) * global_out + sigmoid(local_gate) * local_out.
// Sigmoid and accumulation run in FP32.
// CopyInClaGate stores only the current head's 64 gate scalars contiguously.
// Phase 1 loads the 64 global/local gates as one vector and computes the 64
// sigmoid values in parallel. Phase 2 broadcasts each precomputed scalar to the
// D=256 vector before the attention multiply-add.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void ClaGateMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeClaGateFullTile(
    __ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr, __ubuf__ xDtype* globalGateAddr,
    __ubuf__ xDtype* localGateAddr, __ubuf__ xDtype* outputAddr)
{
            constexpr uint16_t fullTileRows = static_cast<uint16_t>(DOUBLE_BLOCK_SIZE);
    constexpr uint16_t vfsPerRow = static_cast<uint16_t>(ONCE_ROW_LEN / VF_LEN_FP32);
    __VEC_SCOPE__ {
        Reg::RegTensor<xDtype> globalReg, localReg, gateReg;
        Reg::RegTensor<float> gateF, tmpF, oneF;
        Reg::RegTensor<float> globalSigmoidF, localSigmoidF;
        Reg::RegTensor<xDtype> globalSigmoidBf16, localSigmoidBf16;
        Reg::RegTensor<xDtype> outputReg;
        Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        // Sigmoid scratch stores bf16/fp16 values. Keep global/local regions separated.
        __ubuf__ uint8_t* sigmoidBytes = (__ubuf__ uint8_t*)sigmoidBroadcastBuf_.Get<float>().GetPhyAddr();
        __ubuf__ xDtype* globalSigmoidUb = (__ubuf__ xDtype*)sigmoidBytes;
        __ubuf__ xDtype* localSigmoidUb =
            (__ubuf__ xDtype*)(sigmoidBytes + fullTileRows * sizeof(xDtype));

        // Phase 1: vectorized sigmoid over the 64 per-row gate values.
        Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(gateReg, globalGateAddr);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, gateReg, allMask);
        Reg::Muls(tmpF, gateF, -1.0f, allMask);
        Reg::Exp(tmpF, tmpF, allMask);
        Reg::Adds(tmpF, tmpF, 1.0f, allMask);
        Reg::Duplicate(oneF, 1.0f, allMask);
        Reg::Div(globalSigmoidF, oneF, tmpF, allMask);
        Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(globalSigmoidBf16, globalSigmoidF, allMask);
        Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(globalSigmoidUb, globalSigmoidBf16, allMask);

        Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(gateReg, localGateAddr);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gateF, gateReg, allMask);
        Reg::Muls(tmpF, gateF, -1.0f, allMask);
        Reg::Exp(tmpF, tmpF, allMask);
        Reg::Adds(tmpF, tmpF, 1.0f, allMask);
        Reg::Duplicate(oneF, 1.0f, allMask);
        Reg::Div(localSigmoidF, oneF, tmpF, allMask);
        Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(localSigmoidBf16, localSigmoidF, allMask);
        Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(localSigmoidUb, localSigmoidBf16, allMask);

        // Phase 2: broadcast one precomputed sigmoid per row and accumulate attention.
        // Phase 1 stored sigmoid to UB scratch; make the store visible before VEC loads.
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        Reg::AddrReg offset;
        for (uint16_t row = 0; row < fullTileRows; ++row) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_BRC_B16>(
                globalSigmoidBf16, globalSigmoidUb + row);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_BRC_B16>(
                localSigmoidBf16, localSigmoidUb + row);

            for (uint16_t vf = 0; vf < vfsPerRow; ++vf) {
                offset = Reg::CreateAddrReg<xDtype>(row, ONCE_ROW_LEN, vf, VF_LEN_FP32);
                // Native xDtype multiply-add: use half/bf16 arithmetic directly,
                // avoiding the FP32 casts in the inner SwiGLU accumulation loop.
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg, globalAddr, offset);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg, localAddr, offset);

                Reg::Mul(globalReg, globalReg, globalSigmoidBf16, allMask);
                Reg::Mul(localReg, localReg, localSigmoidBf16, allMask);
                Reg::Add(outputReg, globalReg, localReg, allMask);

                Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(outputAddr, outputReg, offset, allMask);
            }
        }
    }
}

template <typename xDtype, typename y1Dtype, 