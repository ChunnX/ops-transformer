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
 * \file fast_strictri_inverse.h
 * \brief Cube-only C64 mixed triangular inverse prototype.
 *
 * This header is intentionally isolated under ref/. It is a clean sketch for
 * the packed diagonal inverse plus lower-triangular compose dataflow and is not
 * wired into the active operator.
 *
 * Input contract:
 * - matrix size is fixed to 64x64, row-major GM stride is 64;
 * - lPowGm holds the strictly-lower L blocks used by MCH;
 * - invGm holds I - L before this helper runs;
 * - all tested sequence lengths for this phase are 64-aligned.
 *
 * Output contract:
 * - the final 64x64 lower-triangular inverse is written back to invGm;
 * - no intermediate diagonal-inverse or compose result is materialized in GM.
 */
#ifndef CHUNK_GATED_DELTA_RULE_NEUMANN_REF_FAST_STRICTRI_INVERSE_H
#define CHUNK_GATED_DELTA_RULE_NEUMANN_REF_FAST_STRICTRI_INVERSE_H

#include "kernel_tiling/kernel_tiling.h"
#include "../op_kernel/chunk_gated_delta_rule_neumann_utils.h"

namespace ChunkGatedDeltaRuleNeumann {
using namespace AscendC;

constexpr uint32_t MIXED_INV_C64 = 64;
constexpr uint32_t MIXED_INV_C32 = 32;
constexpr uint32_t MIXED_INV_C16 = 16;
constexpr uint32_t MIXED_INV_L1_TILE_ELEMS = MIXED_INV_C64 * MIXED_INV_C64;
constexpr uint32_t MIXED_INV_RAW_HALF_FRACTAL_ELEMS = 512 / sizeof(half);
constexpr uint32_t MIXED_INV_HALF_CACHELINE_ELEMS = BLOCK_SIZE / sizeof(half);

constexpr uint8_t MIXED_INV_MMAD_NON_FINAL_UNIT_FLAG = 2;
constexpr uint8_t MIXED_INV_MMAD_FINAL_UNIT_FLAG = 3;
constexpr bool MIXED_INV_MMAD_CMATRIX_SOURCE_L0C = false;
constexpr bool MIXED_INV_MMAD_INIT_C = true;
constexpr bool MIXED_INV_MMAD_ACCUMULATE_C = false;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_FULL = 0;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_PREPARE_ONLY = 1;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_ONLY = 2;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_STORE = 3;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE = 4;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_LOAD_INITIAL = 5;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32 = 6;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE64 = 7;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_FIRST = 8;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_SECOND = 9;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_SECOND_MMAD_ONLY = 10;
constexpr uint32_t FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_SECOND_FIRST_STORE = 11;

struct PureCubeMixedInverse64Buffers {
    LocalTensor<half> scratchL1;   // full 64x64 NZ temporary
    LocalTensor<half> resultL1;    // full 64x64 NZ result, updated in place
    LocalTensor<half> powerL1;     // full 64x64 NZ power / pack temporary
    LocalTensor<half> identityL1;  // Risk: caller must prepare full 64x64 NZ identity before Compute.
    LocalTensor<half> zeroL1;      // full 64x64 NZ zero tile for L0 clearing
    LocalTensor<half> l0A;         // at least one 64x64 fp16 L0A tile
    LocalTensor<half> l0B;         // at least one 64x64 fp16 L0B tile
    LocalTensor<half> l0AAlt;      // optional ping-pong L0A tile
    LocalTensor<half> l0BAlt;      // optional ping-pong L0B tile
    LocalTensor<float> l0C;        // at least one 64x64 fp32 L0C tile
};

class PureCubeMixedInverse64 {
public:
    __aicore__ inline void Init(TPipe *pipe, const PureCubeMixedInverse64Buffers &buffers,
                                uint32_t gmStride = MIXED_INV_C64)
    {
        pipe_ = pipe;
        buffers_ = buffers;
        gmStride_ = gmStride;
    }

    __aicore__ inline void Compute(GlobalTensor<half> lPowGm, GlobalTensor<half> invGm,
                                   uint32_t paraIdx, uint64_t ccOffset,
                                   uint32_t debugMode = FAST_STRICTRI_INVERSE_MODE_FULL)
    {
        PrepareCubeState();

        const uint64_t paraOffset = static_cast<uint64_t>(paraIdx) * ccOffset;
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_PREPARE_ONLY) {
            RestoreCubeState();
            return;
        }
        BuildPackedDiagonalInverse(lPowGm[paraOffset], invGm[paraOffset]);
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_ONLY) {
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_STORE) {
            StoreResult64ToGm(invGm[paraOffset]);
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_LOAD_INITIAL) {
            LoadInitialMatrixToL1(invGm[paraOffset]);
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32) {
            LoadInitialMatrixToL1(invGm[paraOffset]);
            ComposeTwo32From16();
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_FIRST) {
            LoadInitialMatrixToL1(invGm[paraOffset]);
            ComposeTwo32From16FirstMmad();
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_SECOND) {
            LoadInitialMatrixToL1(invGm[paraOffset]);
            ComposeTwo32From16SecondMmadOnly();
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_SECOND_MMAD_ONLY) {
            LoadInitialMatrixToL1(invGm[paraOffset]);
            ComposeTwo32From16SecondMmadNoStore();
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE_TWO32_SECOND_FIRST_STORE) {
            LoadInitialMatrixToL1(invGm[paraOffset]);
            ComposeTwo32From16SecondFirstStore();
            RestoreCubeState();
            return;
        }
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE64) {
            LoadInitialMatrixToL1(invGm[paraOffset]);
            Compose64From32();
            RestoreCubeState();
            return;
        }
        ComposeLowerTriangularBlocks(invGm[paraOffset]);
        if (debugMode == FAST_STRICTRI_INVERSE_MODE_BUILD_AND_COMPOSE) {
            RestoreCubeState();
            return;
        }
        StoreResult64ToGm(invGm[paraOffset]);

        RestoreCubeState();
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline uint64_t MatrixOffset(uint32_t row, uint32_t col) const
    {
        return static_cast<uint64_t>(row) * gmStride_ + col;
    }

    __aicore__ inline uint64_t CompactMatrixNzOffset(uint32_t row, uint32_t col,
                                                     uint32_t matrixSize) const
    {
        return static_cast<uint64_t>(col / BLOCK_CUBE) * matrixSize * BLOCK_CUBE +
               static_cast<uint64_t>(row) * BLOCK_CUBE + (col % BLOCK_CUBE);
    }

    __aicore__ inline void SyncMte2ToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(eventID);
        WaitFlag<HardEvent::MTE2_MTE1>(eventID);
    }

    __aicore__ inline void SyncMToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::M_MTE1));
        SetFlag<HardEvent::M_MTE1>(eventID);
        WaitFlag<HardEvent::M_MTE1>(eventID);
    }

    __aicore__ inline void SyncMte1ToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(eventID);
        WaitFlag<HardEvent::MTE1_M>(eventID);
    }

    __aicore__ inline void SyncMToFix()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(eventID);
        WaitFlag<HardEvent::M_FIX>(eventID);
    }

    __aicore__ inline void SyncFixToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE1));
        SetFlag<HardEvent::FIX_MTE1>(eventID);
        WaitFlag<HardEvent::FIX_MTE1>(eventID);
    }

    __aicore__ inline void SyncFixToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    __aicore__ inline void SyncFixToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_M));
        SetFlag<HardEvent::FIX_M>(eventID);
        WaitFlag<HardEvent::FIX_M>(eventID);
    }

    __aicore__ inline void SyncFixToS()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_S));
        SetFlag<HardEvent::FIX_S>(eventID);
        WaitFlag<HardEvent::FIX_S>(eventID);
    }

    __aicore__ inline void PrepareCubeState()
    {
        SetLoadDataPaddingValue<uint64_t>(0);
        SetNdParaImpl(0x1);
        SetHF32Mode(HF32Mode::DISABLE);
        SetHF32TransMode(HF32TransMode::NEAREST_EVEN);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void RestoreCubeState()
    {
        SetAtomicNone();
        SetNdParaImpl(0x1);
        SetHF32Mode(HF32Mode::DISABLE);
        SetHF32TransMode(HF32TransMode::NEAREST_EVEN);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline MmadParams MakeMmadParams(uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                                bool initCMatrix, uint8_t unitFlag) const
    {
        // Checked: cmatrixSource=false keeps accumulation in L0C; initCMatrix controls overwrite vs accumulate.
        MmadParams params(static_cast<uint16_t>(mSize), static_cast<uint16_t>(nSize),
                          static_cast<uint16_t>(kSize), unitFlag,
                          MIXED_INV_MMAD_CMATRIX_SOURCE_L0C, initCMatrix);
        params.kDirectionAlign = false;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
        params.disableGemv = true;
#endif
        return params;
    }

    __aicore__ inline void CleanHalfMatrixCacheLines(GlobalTensor<half> gm,
                                                     uint32_t rows, uint32_t cols,
                                                     uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; col += MIXED_INV_HALF_CACHELINE_ELEMS) {
                Barrier();
                DataCacheCleanAndInvalid<half, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_UB>(
                    gm[row * stride + col]);
            }
        }
        Barrier();
    }

    __aicore__ inline void FlushHalfMatrixCacheLines(GlobalTensor<half> gm,
                                                     uint32_t rows, uint32_t cols,
                                                     uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; col += MIXED_INV_HALF_CACHELINE_ELEMS) {
                Barrier();
                DataCacheCleanAndInvalid<half, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    gm[row * stride + col]);
            }
        }
        Barrier();
    }

    __aicore__ inline void ZeroHalfL1(LocalTensor<half> l1, uint32_t elems)
    {
        uint32_t bytes = elems * sizeof(half);
        InitConstValueParams<half> zeroParams(
            1, static_cast<uint16_t>(AlignUp(bytes, BLOCK_SIZE) / BLOCK_SIZE), 0, half(0.0f));
        InitConstValue(l1, zeroParams);
        SyncMToMte1();
    }

    __aicore__ inline void LoadGmHalfBlockToPackedL1(GlobalTensor<half> gm, LocalTensor<half> l1,
                                                     uint32_t dstRow, uint32_t dstCol,
                                                     uint32_t blockSize, uint32_t packedSize)
    {
        // Risk: this is still the GM->L1 source-materialization boundary.
        // The diagonal and lower-block selection happens later in L1->L0 copies.
        CleanHalfMatrixCacheLines(gm, blockSize, blockSize, gmStride_);

        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = blockSize;
        copyParams.dValue = blockSize;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = gmStride_;
        copyParams.dstNzC0Stride = packedSize;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(l1[CompactMatrixNzOffset(dstRow, dstCol, packedSize)], gm, copyParams);
        SyncMte2ToMte1();
    }

    __aicore__ inline void CopyPackedL1BlockToL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                 uint32_t srcRow, uint32_t srcCol,
                                                 uint32_t blockSize, uint32_t srcPackedSize,
                                                 bool transpose)
    {
        uint32_t blockCount = AlignUp(blockSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = 1;
        loadParams.srcStride = 1;
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;

        for (uint32_t rowBlock = 0; rowBlock < blockCount; ++rowBlock) {
            for (uint32_t colBlock = 0; colBlock < blockCount; ++colBlock) {
                uint32_t row = rowBlock * BLOCK_CUBE;
                uint32_t col = colBlock * BLOCK_CUBE;
                uint32_t srcOffset = static_cast<uint32_t>(
                    CompactMatrixNzOffset(srcRow + row, srcCol + col, srcPackedSize));
                // Checked: L0 destination is compact fractal order, not the L1 NZ offset.
                uint32_t dstOffset = (rowBlock * blockCount + colBlock) *
                                     MIXED_INV_RAW_HALF_FRACTAL_ELEMS;
                LoadData(l0[dstOffset], l1[srcOffset], loadParams);
            }
        }
    }

    __aicore__ inline void CopyPackedL1BlockToL0A(LocalTensor<half> l1, LocalTensor<half> l0,
                                                  uint32_t srcRow, uint32_t srcCol,
                                                  uint32_t blockSize, uint32_t srcPackedSize)
    {
        CopyPackedL1BlockToL0(l1, l0, srcRow, srcCol, blockSize,
                              srcPackedSize, false);
    }

    __aicore__ inline void CopyPackedL1BlockToL0B(LocalTensor<half> l1, LocalTensor<half> l0,
                                                  uint32_t srcRow, uint32_t srcCol,
                                                  uint32_t blockSize, uint32_t srcPackedSize)
    {
        CopyPackedL1BlockToL0(l1, l0, srcRow, srcCol, blockSize,
                              srcPackedSize, true);
    }

    __aicore__ inline void CopyPackedL1Diagonal16ToL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                      uint32_t matrixSize, bool transpose)
    {
        // Checked: diagonal extraction is L1->L0, matching the reference extraction boundary.
        ZeroHalfL1(buffers_.zeroL1, MIXED_INV_L1_TILE_ELEMS);
        CopyPackedL1BlockToL0(buffers_.zeroL1, l0, 0, 0, matrixSize,
                              MIXED_INV_C64, false);

        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = 1;
        loadParams.srcStride = 1;
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;

        uint32_t blockCount = matrixSize / BLOCK_CUBE;
        for (uint32_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
            uint32_t base = blockIdx * BLOCK_CUBE;
            uint32_t offset = static_cast<uint32_t>(
                CompactMatrixNzOffset(base, base, matrixSize));
            LoadData(l0[offset], l1[offset], loadParams);
        }
    }

    __aicore__ inline void CopyTwo16BlocksToCompactL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                      uint32_t src0Row, uint32_t src0Col,
                                                      uint32_t src1Row, uint32_t src1Col,
                                                      uint32_t srcPackedSize, bool transpose)
    {
        // Checked: two source 16x16 blocks in L1 are packed into a compact 32x32 L0 diagonal.
        ZeroHalfL1(buffers_.zeroL1, MIXED_INV_L1_TILE_ELEMS);
        CopyPackedL1BlockToL0(buffers_.zeroL1, l0, 0, 0, MIXED_INV_C32,
                              MIXED_INV_C64, false);

        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = 1;
        loadParams.srcStride = 1;
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;

        uint32_t src0Offset = static_cast<uint32_t>(
            CompactMatrixNzOffset(src0Row, src0Col, srcPackedSize));
        uint32_t dst0Offset = static_cast<uint32_t>(
            CompactMatrixNzOffset(0, 0, MIXED_INV_C32));
        LoadData(l0[dst0Offset], l1[src0Offset], loadParams);

        uint32_t src1Offset = static_cast<uint32_t>(
            CompactMatrixNzOffset(src1Row, src1Col, srcPackedSize));
        uint32_t dst1Offset = static_cast<uint32_t>(
            CompactMatrixNzOffset(MIXED_INV_C16, MIXED_INV_C16, MIXED_INV_C32));
        LoadData(l0[dst1Offset], l1[src1Offset], loadParams);
    }

    __aicore__ inline void CopyEvenDiagonal16BlocksL1ToL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                          uint32_t srcPackedSize, bool transpose)
    {
        CopyTwo16BlocksToCompactL0(l1, l0,
                                   0, 0,
                                   MIXED_INV_C32, MIXED_INV_C32,
                                   srcPackedSize, transpose);
    }

    __aicore__ inline void CopyOddDiagonal16BlocksL1ToL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                         uint32_t srcPackedSize, bool transpose)
    {
        CopyTwo16BlocksToCompactL0(l1, l0,
                                   MIXED_INV_C16, MIXED_INV_C16,
                                   MIXED_INV_C32 + MIXED_INV_C16, MIXED_INV_C32 + MIXED_INV_C16,
                                   srcPackedSize, transpose);
    }

    __aicore__ inline void CopyOddLower16BlocksL1ToL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                      uint32_t srcPackedSize, bool transpose)
    {
        // Checked: source blocks are (16,0) and (48,32); they become compact L0 diagonal blocks.
        CopyTwo16BlocksToCompactL0(l1, l0,
                                   MIXED_INV_C16, 0,
                                   MIXED_INV_C32 + MIXED_INV_C16, MIXED_INV_C32,
                                   srcPackedSize, transpose);
    }

    __aicore__ inline void CopyCompactDiagonal16BlocksL1ToL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                             uint32_t srcPackedSize, bool transpose)
    {
        CopyTwo16BlocksToCompactL0(l1, l0,
                                   0, 0,
                                   MIXED_INV_C16, MIXED_INV_C16,
                                   srcPackedSize, transpose);
    }

    __aicore__ inline void CopyL0CMatrixToPackedL1(LocalTensor<half> l1, LocalTensor<float> l0c,
                                                   uint32_t dstRow, uint32_t dstCol,
                                                   uint32_t mSize, uint32_t nSize,
                                                   uint32_t dstPackedSize, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = nSize;
        fixParams.mSize = mSize;
        fixParams.srcStride = AlignUp(mSize, BLOCK_CUBE);
        fixParams.dstStride = AlignUp(dstPackedSize, BLOCK_CUBE) * BLOCK_CUBE *
                              sizeof(half) / BLOCK_SIZE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        Fixpipe<half, float, CFG_NZ>(
            l1[CompactMatrixNzOffset(dstRow, dstCol, dstPackedSize)], l0c, fixParams);
    }

    __aicore__ inline void CopyL0CSubMatrixToPackedL1(LocalTensor<half> l1, LocalTensor<float> l0c,
                                                      uint32_t srcRow, uint32_t srcCol,
                                                      uint32_t dstRow, uint32_t dstCol,
                                                      uint32_t subSize, uint32_t srcMatrixSize,
                                                      uint32_t dstPackedSize, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = subSize;
        fixParams.mSize = subSize;
        fixParams.srcStride = AlignUp(srcMatrixSize, BLOCK_CUBE);
        fixParams.dstStride = AlignUp(dstPackedSize, BLOCK_CUBE) * BLOCK_CUBE *
                              sizeof(half) / BLOCK_SIZE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        // Risk: L0C sub-offset follows the existing raw helper contract; verify first if sub-blocks miscompare.
        Fixpipe<half, float, CFG_NZ>(
            l1[CompactMatrixNzOffset(dstRow, dstCol, dstPackedSize)],
            l0c[CompactMatrixNzOffset(srcRow, srcCol, srcMatrixSize)], fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToGmHalf(GlobalTensor<half> gm, LocalTensor<float> l0c,
                                                 uint32_t mSize, uint32_t nSize,
                                                 uint32_t dstStride, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.mSize = mSize;
        fixParams.nSize = nSize;
        fixParams.srcStride = AlignUp(mSize, BLOCK_CUBE);
        fixParams.dstStride = dstStride;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        Fixpipe<half, float, CFG_ROW_MAJOR>(gm, l0c, fixParams);
    }

    __aicore__ inline void RawMmadDiagonalOnly(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                               LocalTensor<float> cTile, uint32_t matrixSize,
                                               bool initCMatrix, uint8_t unitFlag,
                                               uint32_t l0BufferIdx = 0)
    {
        LocalTensor<half> l0A = (l0BufferIdx == 0) ? buffers_.l0A : buffers_.l0AAlt;
        LocalTensor<half> l0B = (l0BufferIdx == 0) ? buffers_.l0B : buffers_.l0BAlt;

        SyncMToMte1();
        CopyPackedL1Diagonal16ToL0(lhsL1, l0A, matrixSize, false);
        CopyPackedL1Diagonal16ToL0(rhsL1, l0B, matrixSize, true);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, l0A, l0B,
             MakeMmadParams(matrixSize, matrixSize, matrixSize, initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void RawMmadPackedBlock(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                              LocalTensor<float> cTile,
                                              uint32_t lhsRow, uint32_t lhsCol,
                                              uint32_t rhsRow, uint32_t rhsCol,
                                              uint32_t blockSize, uint32_t packedSize,
                                              bool initCMatrix, uint8_t unitFlag,
                                              uint32_t l0BufferIdx = 0)
    {
        LocalTensor<half> l0A = (l0BufferIdx == 0) ? buffers_.l0A : buffers_.l0AAlt;
        LocalTensor<half> l0B = (l0BufferIdx == 0) ? buffers_.l0B : buffers_.l0BAlt;

        SyncMToMte1();
        CopyPackedL1BlockToL0A(lhsL1, l0A, lhsRow, lhsCol,
                               blockSize, packedSize);
        CopyPackedL1BlockToL0B(rhsL1, l0B, rhsRow, rhsCol,
                               blockSize, packedSize);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, l0A, l0B,
             MakeMmadParams(blockSize, blockSize, blockSize, initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void RawMmadTwoDiagonal16Blocks(LocalTensor<half> lhsL1,
                                                      LocalTensor<half> rhsL1,
                                                      LocalTensor<float> cTile,
                                                      uint32_t lhs0Row, uint32_t lhs0Col,
                                                      uint32_t lhs1Row, uint32_t lhs1Col,
                                                      uint32_t rhs0Row, uint32_t rhs0Col,
                                                      uint32_t rhs1Row, uint32_t rhs1Col,
                                                      uint32_t packedSize,
                                                      bool initCMatrix, uint8_t unitFlag,
                                                      uint32_t l0BufferIdx = 0)
    {
        LocalTensor<half> l0A = (l0BufferIdx == 0) ? buffers_.l0A : buffers_.l0AAlt;
        LocalTensor<half> l0B = (l0BufferIdx == 0) ? buffers_.l0B : buffers_.l0BAlt;

        SyncMToMte1();
        CopyTwo16BlocksToCompactL0(lhsL1, l0A, lhs0Row, lhs0Col,
                                   lhs1Row, lhs1Col, packedSize, false);
        CopyTwo16BlocksToCompactL0(rhsL1, l0B, rhs0Row, rhs0Col,
                                   rhs1Row, rhs1Col, packedSize, true);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, l0A, l0B,
             MakeMmadParams(MIXED_INV_C32, MIXED_INV_C32, MIXED_INV_C32,
                            initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void PowerDiagonalOnlyToL1(LocalTensor<half> lhs,
                                                 LocalTensor<half> rhs,
                                                 LocalTensor<half> dst,
                                                 uint32_t matrixSize)
    {
        RawMmadDiagonalOnly(lhs, rhs, buffers_.l0C, matrixSize,
                            MIXED_INV_MMAD_INIT_C, MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToPackedL1(dst, buffers_.l0C, 0, 0, matrixSize, matrixSize,
                                matrixSize, MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void InvAddDiagonalOnlyToL1(LocalTensor<half> inv,
                                                  LocalTensor<half> power,
                                                  uint32_t matrixSize)
    {
        RawMmadDiagonalOnly(inv, buffers_.identityL1, buffers_.l0C, matrixSize,
                            MIXED_INV_MMAD_INIT_C, MIXED_INV_MMAD_NON_FINAL_UNIT_FLAG, 0);
        RawMmadDiagonalOnly(inv, power, buffers_.l0C, matrixSize,
                            MIXED_INV_MMAD_ACCUMULATE_C, MIXED_INV_MMAD_FINAL_UNIT_FLAG, 1);
        SyncMToFix();
        CopyL0CMatrixToPackedL1(inv, buffers_.l0C, 0, 0, matrixSize, matrixSize,
                                matrixSize, MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void BuildPackedDiagonalInverse(GlobalTensor<half> lPowGm,
                                                      GlobalTensor<half> invGm)
    {
        ZeroHalfL1(buffers_.resultL1, MIXED_INV_L1_TILE_ELEMS);
        ZeroHalfL1(buffers_.powerL1, MIXED_INV_L1_TILE_ELEMS);
        ZeroHalfL1(buffers_.scratchL1, MIXED_INV_L1_TILE_ELEMS);

        // Checked: load full source tiles once; diagonal selection is done by L1->L0 extraction below.
        LoadGmHalfBlockToPackedL1(lPowGm[MatrixOffset(0, 0)], buffers_.powerL1,
                                  0, 0, MIXED_INV_C64, MIXED_INV_C64);
        LoadGmHalfBlockToPackedL1(invGm[MatrixOffset(0, 0)], buffers_.resultL1,
                                  0, 0, MIXED_INV_C64, MIXED_INV_C64);

        PowerDiagonalOnlyToL1(buffers_.powerL1, buffers_.powerL1,
                              buffers_.scratchL1, MIXED_INV_C64);
        InvAddDiagonalOnlyToL1(buffers_.resultL1, buffers_.scratchL1, MIXED_INV_C64);

        PowerDiagonalOnlyToL1(buffers_.scratchL1, buffers_.scratchL1,
                              buffers_.powerL1, MIXED_INV_C64);
        InvAddDiagonalOnlyToL1(buffers_.resultL1, buffers_.powerL1, MIXED_INV_C64);

        PowerDiagonalOnlyToL1(buffers_.powerL1, buffers_.powerL1,
                              buffers_.scratchL1, MIXED_INV_C64);
        InvAddDiagonalOnlyToL1(buffers_.resultL1, buffers_.scratchL1, MIXED_INV_C64);
    }

    __aicore__ inline void LoadInitialMatrixToL1(GlobalTensor<half> invGm)
    {
        // Risk: full initial matrix still enters from GM; all compose block selection below is L1->L0.
        LoadGmHalfBlockToPackedL1(invGm[MatrixOffset(0, 0)], buffers_.powerL1,
                                  0, 0, MIXED_INV_C64, MIXED_INV_C64);
    }

    __aicore__ inline void RawMmadOddLower16WithEvenDiag(LocalTensor<half> lowerL1,
                                                         LocalTensor<half> diagL1,
                                                         LocalTensor<float> cTile)
    {
        SyncMToMte1();
        CopyOddLower16BlocksL1ToL0(lowerL1, buffers_.l0A, MIXED_INV_C64, false);
        CopyEvenDiagonal16BlocksL1ToL0(diagL1, buffers_.l0B, MIXED_INV_C64, true);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, buffers_.l0A, buffers_.l0B,
             MakeMmadParams(MIXED_INV_C32, MIXED_INV_C32, MIXED_INV_C32,
                            MIXED_INV_MMAD_INIT_C, MIXED_INV_MMAD_FINAL_UNIT_FLAG));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void RawMmadOddDiag16WithCompactTmp(LocalTensor<half> diagL1,
                                                          LocalTensor<half> tmpL1,
                                                          LocalTensor<float> cTile)
    {
        SyncMToMte1();
        CopyOddDiagonal16BlocksL1ToL0(diagL1, buffers_.l0A, MIXED_INV_C64, false);
        CopyCompactDiagonal16BlocksL1ToL0(tmpL1, buffers_.l0B, MIXED_INV_C64, true);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, buffers_.l0A, buffers_.l0B,
             MakeMmadParams(MIXED_INV_C32, MIXED_INV_C32, MIXED_INV_C32,
                            MIXED_INV_MMAD_INIT_C, MIXED_INV_MMAD_FINAL_UNIT_FLAG));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void ComposeTwo32From16()
    {
        RawMmadOddLower16WithEvenDiag(buffers_.powerL1, buffers_.resultL1, buffers_.l0C);
        SyncMToFix();
        CopyL0CMatrixToPackedL1(buffers_.scratchL1, buffers_.l0C, 0, 0,
                                MIXED_INV_C32, MIXED_INV_C32, MIXED_INV_C64,
                                MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();

        RawMmadOddDiag16WithCompactTmp(buffers_.resultL1, buffers_.scratchL1, buffers_.l0C);
        SyncMToFix();
        CopyL0CSubMatrixToPackedL1(buffers_.resultL1, buffers_.l0C,
                                   0, 0, MIXED_INV_C16, 0,
                                   MIXED_INV_C16, MIXED_INV_C32,
                                   MIXED_INV_C64,
                                   MIXED_INV_MMAD_NON_FINAL_UNIT_FLAG);
        PipeBarrier<PIPE_FIX>();
        CopyL0CSubMatrixToPackedL1(buffers_.resultL1, buffers_.l0C,
                                   MIXED_INV_C16, MIXED_INV_C16,
                                   MIXED_INV_C32 + MIXED_INV_C16, MIXED_INV_C32,
                                   MIXED_INV_C16, MIXED_INV_C32,
                                   MIXED_INV_C64,
                                   MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void ComposeTwo32From16FirstMmad()
    {
        RawMmadOddLower16WithEvenDiag(buffers_.powerL1, buffers_.resultL1, buffers_.l0C);
        SyncMToFix();
        CopyL0CMatrixToPackedL1(buffers_.scratchL1, buffers_.l0C, 0, 0,
                                MIXED_INV_C32, MIXED_INV_C32, MIXED_INV_C64,
                                MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void ComposeTwo32From16SecondMmadOnly()
    {
        RawMmadOddDiag16WithCompactTmp(buffers_.resultL1, buffers_.scratchL1, buffers_.l0C);
        SyncMToFix();
        CopyL0CSubMatrixToPackedL1(buffers_.resultL1, buffers_.l0C,
                                   0, 0, MIXED_INV_C16, 0,
                                   MIXED_INV_C16, MIXED_INV_C32,
                                   MIXED_INV_C64,
                                   MIXED_INV_MMAD_NON_FINAL_UNIT_FLAG);
        PipeBarrier<PIPE_FIX>();
        CopyL0CSubMatrixToPackedL1(buffers_.resultL1, buffers_.l0C,
                                   MIXED_INV_C16, MIXED_INV_C16,
                                   MIXED_INV_C32 + MIXED_INV_C16, MIXED_INV_C32,
                                   MIXED_INV_C16, MIXED_INV_C32,
                                   MIXED_INV_C64,
                                   MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void ComposeTwo32From16SecondMmadNoStore()
    {
        RawMmadOddDiag16WithCompactTmp(buffers_.resultL1, buffers_.scratchL1, buffers_.l0C);
    }

    __aicore__ inline void ComposeTwo32From16SecondFirstStore()
    {
        RawMmadOddDiag16WithCompactTmp(buffers_.resultL1, buffers_.scratchL1, buffers_.l0C);
        SyncMToFix();
        CopyL0CSubMatrixToPackedL1(buffers_.resultL1, buffers_.l0C,
                                   0, 0, MIXED_INV_C16, 0,
                                   MIXED_INV_C16, MIXED_INV_C32,
                                   MIXED_INV_C64,
                                   MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void Compose64From32()
    {
        ZeroHalfL1(buffers_.scratchL1, MIXED_INV_L1_TILE_ELEMS);

        // Checked: 32x32 lower block is extracted from the full initial matrix resident in L1.
        RawMmadPackedBlock(buffers_.powerL1, buffers_.resultL1, buffers_.l0C,
                           MIXED_INV_C32, 0,
                           0, 0,
                           MIXED_INV_C32, MIXED_INV_C64,
                           MIXED_INV_MMAD_INIT_C,
                           MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToPackedL1(buffers_.scratchL1, buffers_.l0C,
                                MIXED_INV_C32, 0,
                                MIXED_INV_C32, MIXED_INV_C32,
                                MIXED_INV_C64,
                                MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();

        RawMmadPackedBlock(buffers_.resultL1, buffers_.scratchL1, buffers_.l0C,
                           MIXED_INV_C32, MIXED_INV_C32,
                           MIXED_INV_C32, 0,
                           MIXED_INV_C32, MIXED_INV_C64,
                           MIXED_INV_MMAD_INIT_C,
                           MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToPackedL1(buffers_.resultL1, buffers_.l0C,
                                MIXED_INV_C32, 0,
                                MIXED_INV_C32, MIXED_INV_C32,
                                MIXED_INV_C64,
                                MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void ComposeLowerTriangularBlocks(GlobalTensor<half> invGm)
    {
        LoadInitialMatrixToL1(invGm);
        ComposeTwo32From16();
        Compose64From32();
    }

    __aicore__ inline void StoreResult64ToGm(GlobalTensor<half> invGm)
    {
        RawMmadPackedBlock(buffers_.resultL1, buffers_.identityL1, buffers_.l0C,
                           0, 0, 0, 0,
                           MIXED_INV_C64, MIXED_INV_C64,
                           MIXED_INV_MMAD_INIT_C,
                           MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToGmHalf(invGm, buffers_.l0C, MIXED_INV_C64, MIXED_INV_C64,
                              gmStride_, MIXED_INV_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        FlushHalfMatrixCacheLines(invGm, MIXED_INV_C64, MIXED_INV_C64, gmStride_);
    }

private:
    TPipe *pipe_ = nullptr;
    PureCubeMixedInverse64Buffers buffers_;
    uint32_t gmStride_ = MIXED_INV_C64;
};
} // namespace ChunkGatedDeltaRuleNeumann

#endif // CHUNK_GATED_DELTA_RULE_NEUMANN_REF_FAST_STRICTRI_INVERSE_H
