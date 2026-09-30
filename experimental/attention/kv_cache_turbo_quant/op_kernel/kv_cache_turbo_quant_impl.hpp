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
 * \file kv_cache_turbo_quant_impl.hpp
 * \brief KvCacheTurboQuant encode kernel: online KV cache vector quantization (TurboQuant).
 *
 * Data flow (mirrors golden.py):        1. norm = ||x||_2                    (AIV)
 *   2. unit = x / max(norm, 1e-30)     3. y = rotation_matrix * unit        (AIC)
 *   4. idx  = sum_k 1[y > 0.5*(c_k+c_k+1)]  (strict greater)                (AIV)
 *   5. prim = c[idx], r = y - prim     6. rho = ||r||, gamma = norm*rho,
 *                                          r_unit = r / max(rho, 1e-30)     (AIV)
 *   7. p = r_unit @ qjl^T              8. qjl = 1[p >= 0] packed with idx   (AIV)
 *
 * 1 AIC : 2 AIV walk the same call sequence (the framework turns the AIV call into the KFC
 * client), so SyncAll must not be inserted between phases. KCTQ_BATCH_ROWS rows are pipelined
 * together.
 */

#ifndef KV_CACHE_TURBO_QUANT_IMPL_HPP
#define KV_CACHE_TURBO_QUANT_IMPL_HPP

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kv_cache_turbo_quant_tiling_data.h"

namespace NsKvCacheTurboQuant {
using namespace AscendC;
using namespace matmul;

constexpr uint32_t KCTQ_D = optiling::KCTQ_HEAD_DIM;
constexpr uint32_t KCTQ_MM_LOCAL_WS_BYTES = 8U * 1024U; // matmul framework local workspace

// 1 AIC : 2 AIV; one AIV owns rowsPerCore / KCTQ_AIV_PER_AIC rows.
constexpr uint32_t KCTQ_AIV_PER_AIC = optiling::KCTQ_AIV_PER_AIC;

// Rows per pipeline iteration (the arena is shared with the matmul C tile).
constexpr uint32_t KCTQ_BATCH_ROWS = 8U;
constexpr uint32_t KCTQ_BATCH_LANES = KCTQ_BATCH_ROWS * KCTQ_D;

// Per row scalars 16 fp32 apart → one strided DataCopyPad per batch.
constexpr uint32_t KCTQ_SCALAR_STRIDE = 16U;
constexpr uint32_t KCTQ_SCALAR_SLOTS = KCTQ_BATCH_ROWS * KCTQ_SCALAR_STRIDE; // 128 fp32

// stepGt(x) = clamp(x*S - 1, 0, 1), stepGe(x) = clamp(x*S + 1, 0, 1).
constexpr float KCTQ_STEP_SCALE = 1099511627776.0f; // 2^40

constexpr float KCTQ_MIN_DIVISOR = 1e-30f;                // max(norm, 1e-30) of the golden
constexpr uint32_t KCTQ_OFFSET_TABLES = 2U * KCTQ_D;      // gather offsets, 2 x KCTQ_D u32
constexpr uint32_t KCTQ_OFFSET_BUF_BYTES =
    (KCTQ_OFFSET_TABLES + KCTQ_BATCH_ROWS * KCTQ_D) * sizeof(int32_t); // + divisor table
static_assert(KCTQ_OFFSET_BUF_BYTES >= (KCTQ_OFFSET_TABLES + KCTQ_BATCH_LANES) * sizeof(int32_t),
              "offsetBuf_ must cover the 3bit tables and the divisor gather");
constexpr uint32_t KCTQ_STREAM3_WORDS = KCTQ_D * 3U / 4U; // 4 chunks -> 3 words
constexpr uint32_t KCTQ_ARENA_TAIL_LANES = 128U;          // codebook table + slack

// Lloyd-Max codebooks of the paper rescaled to variance 1/headDim (c = 1/sqrt(128) * table).
template <uint8_t BITS>
struct CodeBook;

template <>
struct CodeBook<2> {
    static constexpr float kLevel[4] = {
        -0.1335033178f, -0.04002048075f, 0.04002048075f, 0.1335033178f};
};

template <>
struct CodeBook<3> {
    static constexpr float kLevel[8] = {
        -0.19020693f, -0.1187859178f, -0.06682205945f, -0.02166347019f,
        0.02166347019f, 0.06682205945f, 0.1187859178f, 0.19020693f};
};

template <>
struct CodeBook<4> {
    static constexpr float kLevel[16] = {
        -0.2414890379f, -0.1828317791f, -0.1429702938f, -0.1109927073f,
        -0.08325428516f, -0.05802082643f, -0.03428063914f, -0.01134236995f,
        0.01134236995f, 0.03428063914f, 0.05802082643f, 0.08325428516f,
        0.1109927073f, 0.1429702938f, 0.1828317791f, 0.2414890379f};
};

// kChunkLanes channels fold into one LSB-first little endian chunk (< 2^24, exact in fp32).
template <uint8_t BITS>
struct PackCfg {
    static constexpr uint32_t kChunkLanes = (BITS == 1) ? 16U : 8U;
    // 2bit / 4bit chunks are wider than one 16bit half: folded as halves, recombined in int.
    static constexpr bool kSplitChunk = (BITS == 2) || (BITS == 4);
};

template <uint8_t BITS>
class KvCacheTurboQuant {
public:
    static constexpr uint32_t kLevels = 1U << BITS;
    static constexpr uint32_t kIdxBytesPerRow = KCTQ_D * BITS / 8U;
    static constexpr uint32_t kChunksPerRow = KCTQ_D / PackCfg<BITS>::kChunkLanes;
    static constexpr uint32_t kChunksPerBatch = KCTQ_BATCH_ROWS * kChunksPerRow;

    // Arena slices (lane offsets). Buffers an async store reads from have kParity copies.
    static constexpr uint32_t kParity = 4U;
    static constexpr uint32_t kSliceIdx0 = 0U;                            // 4 x idx / residual
    static constexpr uint32_t kSliceTmp = 4U * KCTQ_BATCH_LANES;          // mask, intra batch
    static constexpr uint32_t kSlicePrim = 5U * KCTQ_BATCH_LANES;
    static constexpr uint32_t kSliceWork = 6U * KCTQ_BATCH_LANES;         // reduce work area
    static constexpr uint32_t kSliceCv = 7U * KCTQ_BATCH_LANES;
    static constexpr uint32_t kSliceCi0 = 8U * KCTQ_BATCH_LANES;          // 4 x packed words
    static constexpr uint32_t kSliceRow0 = 12U * KCTQ_BATCH_LANES;        // 4 x x / unit vector
    static constexpr uint32_t kSliceRow = kSliceRow0;                     // y, p (intra batch)
    static constexpr uint32_t kSliceScratchA = 16U * KCTQ_BATCH_LANES;    // fold high half / 3bit ci+1 view
    static constexpr uint32_t kSliceScratchB = 17U * KCTQ_BATCH_LANES;    // fold high half / 3bit words
    static constexpr uint32_t kSliceCvHi = kSliceScratchA;
    static constexpr uint32_t kSliceCiHi = kSliceScratchB;
    static constexpr uint32_t kSliceDenom = 18U * KCTQ_BATCH_LANES;
    static constexpr uint32_t kSlicePrimOff = 19U * KCTQ_BATCH_LANES;    // int32 byte offsets of the prim gather
    static constexpr uint32_t kSliceLevels = 20U * KCTQ_BATCH_LANES;     // codebook table (kLevels lanes used)
    static constexpr uint32_t kArenaLanes = kSliceLevels + KCTQ_ARENA_TAIL_LANES;

    __aicore__ inline KvCacheTurboQuant() = default;

    __aicore__ inline void Init(GM_ADDR kvVectors, GM_ADDR rotationMatrix, GM_ADDR qjlMatrix,
                                GM_ADDR quantIdx, GM_ADDR quantQjl, GM_ADDR quantNorm,
                                GM_ADDR quantGamma, GM_ADDR workspace,
                                const optiling::KvCacheTurboQuantTilingData *td)
    {
        td_ = td;
        coreIdx_ = static_cast<uint32_t>(GetBlockIdx());
        rowsPerAiv_ = td->rowsPerCore / KCTQ_AIV_PER_AIC;
        rowBegin_ = coreIdx_ * rowsPerAiv_;
        const uint32_t rowEnd = rowBegin_ + rowsPerAiv_;
        rowEnd_ = (rowEnd > td->totalRows) ? td->totalRows : rowEnd;
        rowsToCompute_ = rowsPerAiv_; // padded rows feed the GEMM tile but are never stored
        wsRowBase_ = coreIdx_ * td->aivRowStride; // workspace row base of this AIV
        wsOff_ = wsRowBase_ - rowBegin_;
        wsRows_ = td->usedCoreNum * KCTQ_AIV_PER_AIC * td->aivRowStride;

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kvVectors),
                              static_cast<uint64_t>(td->totalRows) * KCTQ_D);
        const uint64_t matElems = static_cast<uint64_t>(KCTQ_D) * KCTQ_D;
        // Default interface: fp32 matrices; bf16 arrives via the matrixBf16 interface extension.
        if (td->matrixBf16 != 0U) {
            rotBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(rotationMatrix), matElems);
            qjlBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(qjlMatrix), matElems);
        } else {
            rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rotationMatrix), matElems);
            qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qjlMatrix), matElems);
        }
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantIdx),
                               static_cast<uint64_t>(td->totalRows) * kIdxBytesPerRow);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantQjl),
                                  static_cast<uint64_t>(td->totalRows) * td->qjlBytesPerRow);
        // Metadata extension: fp16 norm / gamma instead of the default bf16.
        if (td->metadataFp16 != 0U) {
            normFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(quantNorm),
                                        static_cast<uint64_t>(td->totalRows));
            gammaFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(quantGamma),
                                         static_cast<uint64_t>(td->totalRows));
        } else {
            normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantNorm),
                                    static_cast<uint64_t>(td->totalRows));
            gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantGamma),
                                     static_cast<uint64_t>(td->totalRows));
        }

        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        if (td->matrixBf16 != 0U) {
            wsRotMatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsRotMat), matElems);
            wsQjlMatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsQjlMat), matElems);
            rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsRotMat), matElems);
            qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsQjlMat), matElems);
        }
        wsUNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsU),
                                   static_cast<uint64_t>(wsRows_) * KCTQ_D);
        wsYNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsY),
                                   static_cast<uint64_t>(wsRows_) * KCTQ_D);
        wsRNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsR),
                                   static_cast<uint64_t>(wsRows_) * KCTQ_D);
        wsPNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsP),
                                   static_cast<uint64_t>(wsRows_) * KCTQ_D);
        wsNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsNorm),
                                  static_cast<uint64_t>(wsRows_));

        const uint64_t rowOffset = static_cast<uint64_t>(wsRowBase_) * KCTQ_D;
        wsUSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsU) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);
        wsYSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsY) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);
        wsRSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsR) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);
        wsPSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsP) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);

        // The C tile staging and the vector staging buffers live in disjoint pipeline phases.
        SetSysWorkspace(workspace);
        pipe_.InitBuffer(arenaBuf_, kArenaLanes * sizeof(float));
        pipe_.InitBuffer(mmLocalWsBuf_, KCTQ_MM_LOCAL_WS_BYTES);

        if (g_coreType == AIV) {
            pipe_.InitBuffer(inQue_, 1, KCTQ_BATCH_ROWS * KCTQ_D * sizeof(bfloat16_t));
            pipe_.InitBuffer(outQue_, 1, KCTQ_BATCH_ROWS * KCTQ_D * sizeof(float));
            pipe_.InitBuffer(scalarBuf_, (kParity + 2U) * KCTQ_SCALAR_SLOTS * sizeof(float));
            pipe_.InitBuffer(bf16Buf_, kParity * KCTQ_SCALAR_SLOTS * sizeof(bfloat16_t));
            pipe_.InitBuffer(weightBuf_, KCTQ_BATCH_LANES * sizeof(float));
            pipe_.InitBuffer(qjlWeightBuf_, KCTQ_BATCH_LANES * sizeof(float));
            // only the 2bit / 4bit layouts fold two halves of a chunk
            pipe_.InitBuffer(weightHiBuf_, PackCfg<BITS>::kSplitChunk ? KCTQ_BATCH_LANES * sizeof(float)
                                                                      : 32U);
            pipe_.InitBuffer(offsetBuf_, KCTQ_OFFSET_BUF_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mm_, &td_->cubeTiling);
        mm_.SetLocalWorkspace(mmLocalWsBuf_.Get<uint8_t>());

        if (g_coreType == AIV) {
            InitWeightTables();
            CastMatrices(); // interface extension: no-op unless the matrices arrive as bf16
            NormalizeRows();
        }

        Gemm(wsUSliceGm_, rotGm_, wsYSliceGm_);

        if (g_coreType == AIV) {
            QuantizeRows();
        }

        Gemm(wsRSliceGm_, qjlGm_, wsPSliceGm_);

        if (g_coreType == AIV) {
            QjlRows();
        }
    }

    // Slice index is a lane offset inside the arena (see the kSlice* constants).
    __aicore__ inline LocalTensor<float> Slice(uint32_t index) const
    {
        return const_cast<ArenaBuf &>(arenaBuf_).Get<float>()[index];
    }

    // Workspace row index of a pipeline row (aivRowStride rows are reserved per AIV).
    __aicore__ inline uint32_t WsRow(uint32_t row) const
    {
        return row + wsOff_;
    }

    // kParity way staged copy of a double buffered staging area
    __aicore__ inline LocalTensor<float> SliceP(uint32_t base, uint32_t parity) const
    {
        return Slice(base + parity * KCTQ_BATCH_LANES);
    }

    // 2^exp as a float: AICORE forbids floating/unsigned integer casts, so build it by doubling.
    __aicore__ inline static float Pow2(uint32_t exp)
    {
        float value = 1.0f;
        for (uint32_t i = 0; i < exp; ++i) {
            value *= 2.0f;
        }
        return value;
    }

    // Weighting lane l of a group by 2^(BITS*l); one 32B block sum yields the packed chunk value.
    __aicore__ inline void InitWeightTables()
    {
        LocalTensor<float> w = weightBuf_.Get<float>();
        LocalTensor<float> wh = weightHiBuf_.Get<float>();
        LocalTensor<float> wq = qjlWeightBuf_.Get<float>();
        // Scalar-write one packing group, replicate the rest with vector copies.
        constexpr uint32_t kGroup = PackCfg<BITS>::kChunkLanes;
        for (uint32_t i = 0; i < kGroup; ++i) {
            wq.SetValue(i, Pow2(i)); // qjl packs 8 sign bits into one byte
            if constexpr (PackCfg<BITS>::kSplitChunk) {
                // low half covers lanes 0..3, high half covers lanes 4..7
                w.SetValue(i, (i < 4U) ? Pow2(static_cast<uint32_t>(BITS) * i) : 0.0f);
                wh.SetValue(i, (i >= 4U) ? Pow2(static_cast<uint32_t>(BITS) * (i - 4U)) : 0.0f);
            } else {
                w.SetValue(i, Pow2(static_cast<uint32_t>(BITS) * i));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);
        for (uint32_t span = kGroup; span < KCTQ_BATCH_LANES; span *= 2U) {
            Adds(w[span], w, 0.0f, span);
            if constexpr (PackCfg<BITS>::kSplitChunk) {
                Adds(wh[span], wh, 0.0f, span);
            }
            Adds(wq[span], wq, 0.0f, span);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (BITS == 3) {
            // Gather offset tables for the 3bit stream packing (see PackStream3Bit), in bytes.
            LocalTensor<uint32_t> off = offsetBuf_.Get<uint32_t>();
            for (uint32_t i = 0; i < KCTQ_D; ++i) {
                off.SetValue(i, 4U * ((i + 1U < KCTQ_D) ? (i + 1U) : (KCTQ_D - 1U))); // +1 chunk view (tail clamped)
            }
            for (uint32_t i = 0; i < KCTQ_D; ++i) {
                // 4 chunks make 3 words: word i sits at lane i + i / 3; the tail entries are unused
                off.SetValue(KCTQ_D + i, 4U * ((i < KCTQ_STREAM3_WORDS) ? (i + i / 3U) : 0U));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID2);
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
        }
        // Per row divisors from the 16 element grid, gathered by row: row r owns lanes
        // [128 r, 128 r + 128) and reads byte 64 r.
        {
            LocalTensor<int32_t> offB = offsetBuf_.Get<int32_t>()[KCTQ_OFFSET_TABLES];
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                Duplicate(offB[r * KCTQ_D], static_cast<int32_t>(64U * r), static_cast<int32_t>(KCTQ_D));
            }
            PipeBarrier<PIPE_V>();
        }
        // level[idx] lookup table for the primary value.
        {
            LocalTensor<float> lv = Slice(kSliceLevels);
            for (uint32_t k = 0; k < kLevels; ++k) {
                lv.SetValue(k, CodeBook<BITS>::kLevel[k]);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID2);
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
        }
    }

    // ---- bits=3 packing: 4 chunks -> 3 words, assembled in place so the batch leaves in one copy ---
    // lane L < 96 holds word L: class k (L%4==k) takes ci[L] >> 8k and ci[L+1] << (24-8k); class 3 is junk.
    __aicore__ inline void PackStream3Bit(uint32_t row0, uint32_t validRows,
                                          const LocalTensor<int32_t> &ci)
    {
        LocalTensor<float> ci1F = Slice(kSliceCvHi);
        LocalTensor<int32_t> ci1 = ci1F.ReinterpretCast<int32_t>();
        LocalTensor<float> wordF = Slice(kSliceCiHi);
        LocalTensor<int32_t> word = wordF.ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> off = offsetBuf_.Get<uint32_t>();

        // ci1[L] = ci[L + 1] (the tail lanes belong to junk classes and are never used)
        Gather(ci1, ci, off, 0U, 64U, 2U, 8U);
        PipeBarrier<PIPE_V>();

        uint64_t cls0[2] = {0x1111111111111111ULL, 0x1111111111111111ULL};
        uint64_t cls1[2] = {0x2222222222222222ULL, 0x2222222222222222ULL};
        uint64_t cls2[2] = {0x4444444444444444ULL, 0x4444444444444444ULL};
        const UnaryRepeatParams up{1U, 1U, 8U, 8U};
        // Left halves (a operand) into word, right halves pushed into ci1 in place; the three masked
        // writes touch disjoint lane classes, so the source survives until the last one.
        Adds(word, ci, static_cast<int32_t>(0), cls0, 16U, up);
        ShiftRight(word, ci, static_cast<int32_t>(8), cls1, 16U, up);
        ShiftRight(word, ci, static_cast<int32_t>(16), cls2, 16U, up);
        ShiftLeft(ci1, ci1, static_cast<int32_t>(24), cls0, 16U, up);
        ShiftLeft(ci1, ci1, static_cast<int32_t>(16), cls1, 16U, up);
        ShiftLeft(ci1, ci1, static_cast<int32_t>(8), cls2, 16U, up);
        Or(word, word, ci1, KCTQ_BATCH_LANES);
        PipeBarrier<PIPE_V>();
        // packed[i] = word[i + i/3]; ci is dead here and becomes the store source.
        Gather(ci, word, off[128], 0U, 64U, 2U, 8U);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (validRows > 0U) {
            LocalTensor<uint8_t> bytes = ci.ReinterpretCast<uint8_t>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(validRows * kIdxBytesPerRow), 0, 0, 0};
            DataCopyPad(idxGm_[static_cast<uint64_t>(row0) * kIdxBytesPerRow], bytes, cp);
        }
    }

    // dst[8*k + j] = sum_l src[64*k + 8*j + l] * 2^(BITS*l): one repeat carries 8 blocks (mask 64),
    // its sums written consecutively (dstRepStride 1), so 8 chunks come out per 32B block.
    __aicore__ inline void FoldChunks(const LocalTensor<float> &dst, const LocalTensor<float> &scratch,
                                      const LocalTensor<float> &src, const LocalTensor<float> &weights,
                                      uint32_t lanes)
    {
        Mul(scratch, src, weights, lanes);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float>(dst, scratch, static_cast<int32_t>(lanes / 64U), 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    // Spread chunk words so chunk m starts at element 8*m (byte 32*m): DataCopyPad strides in whole
    // 32B blocks. One Brcb repeat copies 8 source words, so it takes chunks/8 repeats.
    __aicore__ inline void SpreadChunks(const LocalTensor<int32_t> &dst, const LocalTensor<int32_t> &src,
                                        uint32_t chunks)
    {
        Brcb(dst, src, static_cast<uint8_t>(chunks / 8U), BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
    }

    // Per row divisors from the 16 element grid: vector clamp + one gather, all on the vector pipe.
    __aicore__ inline void BroadcastDivisors(const LocalTensor<float> &dst, const LocalTensor<float> &scalars)
    {
        LocalTensor<float> clamped = Slice(kSliceCvHi);
        Maxs(clamped, scalars, KCTQ_MIN_DIVISOR, static_cast<int32_t>(KCTQ_SCALAR_SLOTS));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> offB = offsetBuf_.Get<uint32_t>()[KCTQ_OFFSET_TABLES];
        Gather(dst, clamped, offB, 0U, 64U, static_cast<uint8_t>(KCTQ_BATCH_LANES / 64U), 8U);
        PipeBarrier<PIPE_V>();
    }

    // Row sums of a whole batch: fold[i] = src[i] + src[i + 64], then a pairwise tree over
    // fold[0..64) — the two reduce intrinsics below.
    __aicore__ inline void BatchRowSum(const LocalTensor<float> &dst, const LocalTensor<float> &tmp,
                                       const LocalTensor<float> &src)
    {
        // Stage 1: fold the two 64 lane halves element wise, one repeat per row.
        LocalTensor<float> fold = tmp;
        LocalTensor<float> blk = tmp[KCTQ_BATCH_ROWS * 64U];
        uint64_t fullMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
        Add(fold, src, src[64U], fullMask, static_cast<uint8_t>(KCTQ_BATCH_ROWS),
            BinaryRepeatParams(1, 1, 1, 8, 16, 16));
        PipeBarrier<PIPE_V>();
        // Stage 2: eight block sums per folded row, one destination block per repeat.
        BlockReduceSum<float>(blk, fold, static_cast<int32_t>(KCTQ_BATCH_ROWS), 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        // Stage 3: one whole reduce per row over its eight block sums; stride 16 lands them on the
        // 16 lane grid. kBlockMask is explicit (the counting overload treats 8 as a full mask).
        const uint64_t kBlockMask[2] = {0xFFULL, 0xFFULL};
        WholeReduceSum<float>(dst, blk, kBlockMask, static_cast<int32_t>(KCTQ_BATCH_ROWS), 16, 1, 1);
        PipeBarrier<PIPE_V>();
    }

    // Store the packed chunk bytes of the 1bit and 3bit layouts: `blockBytes` bytes are taken from
    // the head of every 32B block of `src` and appended to gm. The batch is contiguous in gm as
    // well as one chunk per 32B block in UB, so a single call stores a whole batch.
    __aicore__ inline void StoreChunks(const GlobalTensor<uint8_t> &gm, uint64_t gmOffset,
                                       const LocalTensor<int32_t> &src, uint32_t nChunks,
                                       uint32_t blockBytes)
    {
        LocalTensor<uint8_t> bytes = src.ReinterpretCast<uint8_t>();
        DataCopyExtParams cp{static_cast<uint16_t>(nChunks), blockBytes, 0, 0, 0};
        DataCopyPad(gm[gmOffset], bytes, cp);
    }

    // ---- AIV: steps 1 and 2 ---------------------------------------------------------------------
    // One batch at a time: one strided load, one cast, one division and two strided stores.
    __aicore__ inline void NormalizeRows()
    {
        LocalTensor<float> sq = scalarBuf_.Get<float>();                  // ||x||^2 at [16*row]

        for (uint32_t row0 = rowBegin_; row0 < rowBegin_ + rowsToCompute_; row0 += KCTQ_BATCH_ROWS) {
            // same KCTQ_PARITY way staging as QuantizeRows
            const uint32_t pb = ((row0 - rowBegin_) / KCTQ_BATCH_ROWS) % kParity;
            LocalTensor<float> norm = scalarBuf_.Get<float>()[KCTQ_SCALAR_SLOTS * (pb + 1U)];
            const uint32_t validRows = (row0 + KCTQ_BATCH_ROWS <= rowEnd_)
                                           ? KCTQ_BATCH_ROWS
                                           : ((rowEnd_ > row0) ? (rowEnd_ - row0) : 0U);
            LocalTensor<float> x = SliceP(kSliceRow0, pb);
            if (validRows > 0U) {
                LocalTensor<bfloat16_t> xBf = inQue_.AllocTensor<bfloat16_t>();
                DataCopyExtParams cpIn{1, static_cast<uint32_t>(validRows * KCTQ_D * sizeof(bfloat16_t)),
                                       0, 0, 0};
                DataCopyPadExtParams<bfloat16_t> padIn{false, 0, 0, 0};
                DataCopyPad(xBf, kvGm_[static_cast<uint64_t>(row0) * KCTQ_D], cpIn, padIn);
                inQue_.EnQue(xBf);
                xBf = inQue_.DeQue<bfloat16_t>();
                Cast(x, xBf, RoundMode::CAST_NONE, KCTQ_BATCH_LANES);
                inQue_.FreeTensor(xBf);
            }
            if (validRows < KCTQ_BATCH_ROWS) {
                Duplicate(x[validRows * KCTQ_D], 0.0f, (KCTQ_BATCH_ROWS - validRows) * KCTQ_D);
            }

            LocalTensor<float> tmp = Slice(kSliceTmp);
            LocalTensor<float> work = Slice(kSliceWork);
            Mul(tmp, x, x, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            BatchRowSum(sq, work, tmp);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Sqrt(norm, sq, KCTQ_SCALAR_SLOTS);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> denom = Slice(kSliceDenom);
            BroadcastDivisors(denom, norm);
            Div(x, x, denom, KCTQ_BATCH_LANES); // x := unit vector

            if (td_->metadataFp16 != 0U) {
                StoreMetaRows<half>(normFp16Gm_, norm, row0, validRows, pb);
            } else {
                StoreMetaRows<bfloat16_t>(normGm_, norm, row0, validRows, pb);
            }
            if (validRows > 0U) {
                // fp32 norms for the quantize stage: 4 bytes out of every other 32B block
                DataCopyExtParams cpNormF{static_cast<uint16_t>(KCTQ_BATCH_ROWS), 4U, 1U, 0, 0};
                DataCopyPad(wsNormGm_[WsRow(row0)], norm, cpNormF);
            }
            DataCopyExtParams cpU{1, static_cast<uint32_t>(KCTQ_BATCH_LANES * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsUNormGm_[static_cast<uint64_t>(WsRow(row0)) * KCTQ_D], x, cpU);
            if (pb + 1U == kParity) {
                // drain before the parity copies are rewritten
                PipeBarrier<PIPE_ALL>();
            }
        }
        // phase end: stores must be visible to the cube before Gemm
        PipeBarrier<PIPE_ALL>();
    }

    // ---- AIV: interface extension, bf16 matrices upcast to fp32 -----------------------------------
    // Each AIV upcasts the whole matrix into the workspace; the GEMMs read the fp32 copy.
    __aicore__ inline void CastMatrices()
    {
        if (td_->matrixBf16 == 0U) {
            return;
        }
        constexpr uint32_t chunks = KCTQ_D / KCTQ_BATCH_ROWS;
        for (uint32_t chunk = 0U; chunk < chunks; ++chunk) {
            const uint32_t row0 = chunk * KCTQ_BATCH_ROWS;
            CastMatrixChunk(rotBf16Gm_, wsRotMatGm_, row0);
            CastMatrixChunk(qjlBf16Gm_, wsQjlMatGm_, row0);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CastMatrixChunk(const GlobalTensor<bfloat16_t> &srcGm,
                                           const GlobalTensor<float> &dstGm, uint32_t row0)
    {
        LocalTensor<bfloat16_t> in = inQue_.AllocTensor<bfloat16_t>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(KCTQ_BATCH_ROWS * KCTQ_D * sizeof(bfloat16_t)),
                               0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padIn{false, 0, 0, 0};
        DataCopyPad(in, srcGm[static_cast<uint64_t>(row0) * KCTQ_D], cpIn, padIn);
        inQue_.EnQue(in);
        in = inQue_.DeQue<bfloat16_t>();
        LocalTensor<float> out = outQue_.AllocTensor<float>();
        Cast(out, in, RoundMode::CAST_NONE, KCTQ_BATCH_ROWS * KCTQ_D);
        PipeBarrier<PIPE_V>();
        inQue_.FreeTensor(in);
        outQue_.EnQue(out);
        out = outQue_.DeQue<float>();
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(KCTQ_BATCH_ROWS * KCTQ_D * sizeof(float)),
                                0, 0, 0};
        DataCopyPad(dstGm[static_cast<uint64_t>(row0) * KCTQ_D], out, cpOut);
        outQue_.FreeTensor(out);
    }

    // norm / gamma store: bf16 (formal) or fp16 (extension), sharing the 16bit parity buffer.
    template <typename TOut>
    __aicore__ inline void StoreMetaRows(const GlobalTensor<TOut> &dst, const LocalTensor<float> &src,
                                         uint32_t row0, uint32_t validRows, uint32_t pb)
    {
        LocalTensor<TOut> out = bf16Buf_.Get<TOut>()[pb * KCTQ_SCALAR_SLOTS];
        Cast(out, src, RoundMode::CAST_RINT, KCTQ_SCALAR_SLOTS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (validRows > 0U) {
            // the valid scalars sit on 32B block boundaries
            DataCopyExtParams cp{static_cast<uint16_t>(validRows), 2U, 0, 0, 0};
            DataCopyPad(dst[row0], out, cp);
        }
    }

    // ---- AIC: steps 3 and 7, dense fp32 GEMM -----------------------------------------------------
    // The cube writes every C tile straight into the workspace (enSequentialWrite = true).
    __aicore__ inline void Gemm(const GlobalTensor<float> &aGm, const GlobalTensor<float> &bGm,
                                const GlobalTensor<float> &cGm)
    {
        mm_.SetOrgShape(static_cast<int32_t>(rowsToCompute_), static_cast<int32_t>(KCTQ_D),
                        static_cast<int32_t>(KCTQ_D));
        mm_.SetHF32(false); // indices are decided on fp32 values
        mm_.SetTensorA(aGm);
        mm_.SetTensorB(bGm, true);

        const uint32_t baseM = static_cast<uint32_t>(td_->cubeTiling.baseM);
        uint32_t done = 0;
        // An AIV owns only half of its core's rows: stop once they are delivered.
        while (done < rowsToCompute_) {
            if (!mm_.template Iterate<true>()) {
                break;
            }
            uint32_t rows = rowsToCompute_ - done;
            if (rows > baseM) {
                rows = baseM;
            }
            mm_.template GetTensorC<true>(cGm[static_cast<uint64_t>(done) * KCTQ_D], 0, true);
            done += rows;
        }
        mm_.End();
        PipeBarrier<PIPE_ALL>();
    }

    // ---- AIV: steps 4, 5 and 6 --------------------------------------------------------------------
    __aicore__ inline void QuantizeRows()
    {
        LocalTensor<float> sq = scalarBuf_.Get<float>();                       // rho^2 at [16*row]
        LocalTensor<float> norm = scalarBuf_.Get<float>()[KCTQ_SCALAR_SLOTS];
        LocalTensor<float> gamma = scalarBuf_.Get<float>()[KCTQ_SCALAR_SLOTS * 3U];
        LocalTensor<float> denom = Slice(kSliceDenom);
        for (uint32_t row0 = rowBegin_; row0 < rowBegin_ + rowsToCompute_; row0 += KCTQ_BATCH_ROWS) {
            // Batches alternate between the two copies of the residual slice.
            const uint32_t grow = row0;
            const uint32_t pb = ((row0 - rowBegin_) / KCTQ_BATCH_ROWS) % kParity;
            const uint32_t validRows = (row0 + KCTQ_BATCH_ROWS <= rowEnd_)
                                           ? KCTQ_BATCH_ROWS
                                           : ((rowEnd_ > row0) ? (rowEnd_ - row0) : 0U);
            const uint64_t base = static_cast<uint64_t>(WsRow(row0)) * KCTQ_D;
            LocalTensor<float> y = Slice(kSliceRow);
            DataCopyExtParams cpY{1, static_cast<uint32_t>(KCTQ_BATCH_LANES * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padY{false, 0, 0, 0};
            DataCopyPad(y, wsYNormGm_[base], cpY, padY);
            // the fp32 norms written by NormalizeRows, spread back onto 32B block boundaries
            DataCopyExtParams cpN{static_cast<uint16_t>(KCTQ_BATCH_ROWS), 4U, 0, 1U, 0};
            DataCopyPad(norm, wsNormGm_[WsRow(row0)], cpN, padY);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            LocalTensor<float> mask = Slice(kSliceTmp);
            LocalTensor<float> idx = SliceP(kSliceIdx0, pb);
            LocalTensor<float> prim = Slice(kSlicePrim);
            Duplicate(idx, 0.0f, KCTQ_BATCH_LANES);
            for (uint32_t k = 1; k < kLevels; ++k) {
                // boundary = 0.5 * (level[k-1] + level[k])
                const float boundary =
                    (CodeBook<BITS>::kLevel[k - 1] + CodeBook<BITS>::kLevel[k]) * 0.5f;
                Adds(mask, y, -boundary, KCTQ_BATCH_LANES);
                Muls(mask, mask, KCTQ_STEP_SCALE, KCTQ_BATCH_LANES);
                Adds(mask, mask, -1.0f, KCTQ_BATCH_LANES);
                Maxs(mask, mask, 0.0f, KCTQ_BATCH_LANES);
                Mins(mask, mask, 1.0f, KCTQ_BATCH_LANES);
                Add(idx, idx, mask, KCTQ_BATCH_LANES);
            }
            // prim = level[idx], an exact integer table lookup
            {
                LocalTensor<float> levels = Slice(kSliceLevels);
                LocalTensor<float> offsetsF = Slice(kSlicePrimOff);
                LocalTensor<int32_t> offsets = offsetsF.ReinterpretCast<int32_t>();
                Cast(offsets, idx, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
                ShiftLeft(offsets, offsets, 2, KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
                Gather(prim, levels, offsets.ReinterpretCast<uint32_t>(), 0U, 64U,
                       static_cast<uint8_t>(KCTQ_BATCH_LANES / 64U), 8U);
                PipeBarrier<PIPE_V>();
            }
            PackIdx(grow, validRows, idx, pb);
            const LocalTensor<float> res = idx; // the residual replaces the packed index values
            LocalTensor<float> work = Slice(kSliceWork);
            Sub(res, y, prim, KCTQ_BATCH_LANES);
            Mul(mask, res, res, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            BatchRowSum(sq, work, mask);
            // Reduce intrinsics are long latency: wait for the sums before their slices are reused.
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Sqrt(sq, sq, KCTQ_SCALAR_SLOTS);
            Mul(gamma, sq, norm, KCTQ_SCALAR_SLOTS); // gamma = norm * rho
            PipeBarrier<PIPE_V>();
            BroadcastDivisors(denom, sq);
            Div(res, res, denom, KCTQ_BATCH_LANES); // residual unit vector (0 when residual is 0)

            if (td_->metadataFp16 != 0U) {
                StoreMetaRows<half>(gammaFp16Gm_, gamma, grow, validRows, pb);
            } else {
                StoreMetaRows<bfloat16_t>(gammaGm_, gamma, grow, validRows, pb);
            }
            DataCopyExtParams cpR{1, static_cast<uint32_t>(KCTQ_BATCH_LANES * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsRNormGm_[base], res, cpR);
            if (pb + 1U == kParity) {
                // drain before the parity copies are rewritten
                PipeBarrier<PIPE_ALL>();
            }
        }
        // phase end: stores must be visible to the cube before Gemm
        PipeBarrier<PIPE_ALL>();
    }

    // ---- AIV: pack the primary indices ------------------------------------------------------------
    __aicore__ inline void PackIdx(uint32_t row0, uint32_t validRows, const LocalTensor<float> &idx,
                                   uint32_t parity)
    {
        LocalTensor<float> w = weightBuf_.Get<float>();
        LocalTensor<float> wv = Slice(kSliceTmp);
        LocalTensor<float> cv = Slice(kSliceCv);
        LocalTensor<float> ciF = SliceP(kSliceCi0, parity);
        LocalTensor<int32_t> ci = ciF.ReinterpretCast<int32_t>();

        FoldChunks(cv, wv, idx, w, KCTQ_BATCH_LANES);
        if constexpr (PackCfg<BITS>::kSplitChunk) {
            // 2bit / 4bit chunks are wider than one 16bit half: fold separately, recombine in int.
            LocalTensor<float> wh = weightHiBuf_.Get<float>();
            LocalTensor<float> cvh = Slice(kSliceCvHi);
            LocalTensor<float> wvh = Slice(kSliceWork);
            LocalTensor<float> cihF = Slice(kSliceCiHi); // intra batch only
            FoldChunks(cvh, wvh, idx, wh, KCTQ_BATCH_LANES);
            if constexpr (BITS == 4) {
                LocalTensor<int32_t> ci = ciF.ReinterpretCast<int32_t>();
                LocalTensor<int32_t> cih = cihF.ReinterpretCast<int32_t>();
                Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
                Cast(cih, cvh, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
                ShiftLeft(cih, cih, 16, KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
                Add(ci, ci, cih, KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<int16_t> ci = ciF.ReinterpretCast<int16_t>();
                LocalTensor<int16_t> cih = cihF.ReinterpretCast<int16_t>();
                Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
                Cast(cih, cvh, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
                ShiftLeft(cih, cih, static_cast<int16_t>(8), KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
                Add(ci, ci, cih, KCTQ_BATCH_LANES);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (validRows > 0U) {
                LocalTensor<uint8_t> bytes = ciF.ReinterpretCast<uint8_t>();
                DataCopyExtParams cp{1, validRows * kIdxBytesPerRow, 0, 0, 0};
                DataCopyPad(idxGm_[static_cast<uint64_t>(row0) * kIdxBytesPerRow], bytes, cp);
            }
        } else if constexpr (BITS == 3) {
            // 3bit words are assembled inside the batch, see PackStream3Bit.
            Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            PackStream3Bit(row0, validRows, ci);
        } else {
            // 1bit: spread the 2 byte chunks one per 32B block and store strided.
            LocalTensor<float> spreadF = Slice(kSliceCiHi);
            LocalTensor<int32_t> spread = spreadF.ReinterpretCast<int32_t>();
            Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            SpreadChunks(spread, ci, kChunksPerBatch);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (validRows > 0U) {
                StoreChunks(idxGm_, static_cast<uint64_t>(row0) * kIdxBytesPerRow, spread,
                            validRows * kChunksPerRow, kIdxBytesPerRow / kChunksPerRow);
            }
        }
        // No drain here: the next batch rewrites these slices only after the QuantizeRows drain.
    }

    // ---- AIV: step 8 ------------------------------------------------------------------------------
    __aicore__ inline void QjlRows()
    {
        LocalTensor<float> wq = qjlWeightBuf_.Get<float>();
        LocalTensor<float> cv = Slice(kSliceCv);
        LocalTensor<float> hF = Slice(kSliceCvHi);
        LocalTensor<half> h8 = hF.ReinterpretCast<half>();
        LocalTensor<float> uF = Slice(kSliceCiHi);
        LocalTensor<uint8_t> u8 = uF.ReinterpretCast<uint8_t>();

        for (uint32_t row0 = rowBegin_; row0 < rowBegin_ + rowsToCompute_; row0 += KCTQ_BATCH_ROWS) {
            const uint32_t validRows = (row0 + KCTQ_BATCH_ROWS <= rowEnd_)
                                           ? KCTQ_BATCH_ROWS
                                           : ((rowEnd_ > row0) ? (rowEnd_ - row0) : 0U);
            LocalTensor<float> p = Slice(kSliceRow);
            DataCopyExtParams cpP{1, static_cast<uint32_t>(KCTQ_BATCH_LANES * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padP{false, 0, 0, 0};
            DataCopyPad(p, wsPNormGm_[static_cast<uint64_t>(WsRow(row0)) * KCTQ_D], cpP, padP);
            // drain the async GM -> UB copy
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            LocalTensor<float> mask = Slice(kSliceTmp);
            LocalTensor<float> wv = Slice(kSliceWork);
            Muls(mask, p, KCTQ_STEP_SCALE, KCTQ_BATCH_LANES);
            Adds(mask, mask, 1.0f, KCTQ_BATCH_LANES);
            Maxs(mask, mask, 0.0f, KCTQ_BATCH_LANES);
            Mins(mask, mask, 1.0f, KCTQ_BATCH_LANES); // 1[p >= 0]
            // The 8 sign bits of a chunk fold into one byte.
            FoldChunks(cv, wv, mask, wq, KCTQ_BATCH_LANES);
            // A packed sign byte is 2 bytes wide: narrow through half into a contiguous byte run.
            Cast(h8, cv, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            Cast(u8, h8, RoundMode::CAST_NONE, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (validRows > 0U) {
                DataCopyExtParams cpQjl{1, validRows * td_->qjlBytesPerRow, 0, 0, 0};
                DataCopyPad(qjlOutGm_[static_cast<uint64_t>(row0) * td_->qjlBytesPerRow], u8, cpQjl);
            }
            PipeBarrier<PIPE_ALL>(); // drain the copy before the next batch reuses the slices
        }
    }

private:
    using ArenaBuf = TBuf<TPosition::VECCALC>;

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQue_;
    // fp32 staging of the bf16 matrices; the queue owns the V <-> MTE3 syncs.
    TQue<TPosition::VECOUT, 1> outQue_;
    ArenaBuf arenaBuf_;                                     // C tile + vector staging, see kSlice*
    TBuf<TPosition::VECCALC> mmLocalWsBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;                    // per row scalars and reduce work
    TBuf<TPosition::VECCALC> bf16Buf_;                      // norm / gamma store staging
    TBuf<TPosition::VECCALC> weightBuf_;                    // packing lane weights
    TBuf<TPosition::VECCALC> weightHiBuf_;
    TBuf<TPosition::VECCALC> qjlWeightBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;                      // 3bit gather tables + divisor gather

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, float, true>; // matrices are handed transposed
    using cType = MatmulType<TPosition::LCM, CubeFormat::ND, float>;
    Matmul<aType, bType, cType> mm_;

    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<bfloat16_t> rotBf16Gm_;
    GlobalTensor<bfloat16_t> qjlBf16Gm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;
    GlobalTensor<half> normFp16Gm_;
    GlobalTensor<half> gammaFp16Gm_;
    GlobalTensor<float> wsUNormGm_;
    GlobalTensor<float> wsYNormGm_;
    GlobalTensor<float> wsRNormGm_;
    GlobalTensor<float> wsPNormGm_;
    GlobalTensor<float> wsNormGm_;
    GlobalTensor<float> wsRotMatGm_;
    GlobalTensor<float> wsQjlMatGm_;
    GlobalTensor<float> wsUSliceGm_;
    GlobalTensor<float> wsYSliceGm_;
    GlobalTensor<float> wsRSliceGm_;
    GlobalTensor<float> wsPSliceGm_;

    const optiling::KvCacheTurboQuantTilingData *td_ = nullptr;
    uint32_t coreIdx_ = 0;
    uint32_t rowsPerAiv_ = 0; // rowsPerCore split over KCTQ_AIV_PER_AIC AIVs
    uint32_t rowBegin_ = 0;
    uint32_t rowEnd_ = 0;
    uint32_t rowsToCompute_ = 0;
    uint32_t wsRowBase_ = 0; // workspace row base of this AIV
    uint32_t wsOff_ = 0;     // wsRowBase_ - rowBegin_
    uint32_t wsRows_ = 0;    // rows reserved in the workspace in total
};

} // namespace NsKvCacheTurboQuant

#endif // KV_CACHE_TURBO_QUANT_IMPL_HPP
