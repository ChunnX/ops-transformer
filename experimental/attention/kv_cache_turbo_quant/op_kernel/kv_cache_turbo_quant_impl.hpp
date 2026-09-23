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
 * Data flow (mirrors the golden reference step by step):
 *   1. norm = ||x||_2                                                     (AIV)
 *   2. unit = x / max(norm, 1e-30)                                        (AIV)
 *   3. y    = rotation_matrix * unit                                      (AIC, fp32 cube)
 *   4. idx  = sum_k 1[y > 0.5*(c_k + c_k+1)]                              (AIV, strict greater)
 *   5. prim = c[idx], r = y - prim                                        (AIV)
 *   6. rho = ||r||, gamma = norm * rho, r_unit = r / max(rho, 1e-30)      (AIV)
 *   7. p    = r_unit @ qjl^T                                              (AIC, fp32 cube;
 *             the runtime transpose flag of the B operand has to be set)
 *   8. qjl  = 1[p >= 0] bit packed together with the packed idx           (AIV)
 *
 * AIC and AIV run 1:1. Both sub cores walk the same call sequence: the matmul framework turns the
 * AIV call into the KFC client and the AIC call into the serving cube work, so a global SyncAll
 * barrier must not be inserted between the phases (the cube side has to stay in the serving loop).
 * The intermediate matrices live in the user workspace. Everything is computed in fp32; bf16 is
 * produced exactly once, on the store of norm/gamma.
 *
 * KCTQ_BATCH_ROWS rows are carried through the whole row pipeline at once. Measured on the
 * gqa_prefill_t2048 case a single row costs about 8700 AIV cycles while its pure vector work is
 * only ~900 of them: the rest is per row synchronization (a full pipe drain per store plus the
 * scalar<->vector handshakes). Batching issues every vector call, barrier and store once per batch
 * instead of once per row, which is what closes that gap.
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
constexpr uint32_t KCTQ_MM_LOCAL_WS_BYTES = 100U * 1024U;

// Rows per pipeline iteration. The AIV staging buffers are sized for this many rows, and the
// arena below is shared with the matmul C tile, which is dead while the vector stages run.
constexpr uint32_t KCTQ_BATCH_ROWS = 8U;
constexpr uint32_t KCTQ_BATCH_LANES = KCTQ_BATCH_ROWS * KCTQ_D;

// Per row scalars (norm, rho, gamma) are kept 16 fp32 apart. That makes the bf16 downcast of a
// whole batch land on 32B block boundaries, so one strided DataCopyPad per batch can emit them.
constexpr uint32_t KCTQ_SCALAR_STRIDE = 16U;
constexpr uint32_t KCTQ_SCALAR_SLOTS = KCTQ_BATCH_ROWS * KCTQ_SCALAR_STRIDE; // 128 fp32

// Comparisons are expressed with plain vector arithmetic:
//   stepGt(x) = clamp(x * SCALE - 1, 0, 1)  ==  1[x > 0]
//   stepGe(x) = clamp(x * SCALE + 1, 0, 1)  ==  1[x >= 0]
// The scale only has to separate values that really occur: the operands are fp32 numbers of
// order 1e-1, so a non zero difference is >= 1e-8, while 2^40 keeps the rounding error of the
// comparison far below that and stays far away from overflow for finite inputs.
constexpr float KCTQ_STEP_SCALE = 1099511627776.0f; // 2^40

// Lloyd-Max codebooks of the TurboQuant paper rescaled to variance 1/headDim:
//   c = 1/sqrt(128) * table, transcribed digit for digit from the golden reference.
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

// Bit packing geometry. Every kChunkLanes channels are folded into one LSB first little endian
// chunk whose value stays below 2^24, hence exact in fp32, and whose bytes are exactly the packed
// output bytes. 4bit chunks are 32bit wide and are produced as low/high 16bit halves.
template <uint8_t BITS>
struct PackCfg {
    static constexpr uint32_t kChunkLanes = (BITS == 1) ? 16U : 8U;
    static constexpr uint32_t kChunkBytes = BITS * kChunkLanes / 8U;
    // 2bit and 4bit chunks are wider than one 16bit half, so they are folded as two halves which
    // are recombined in the integer domain. 2bit halves are a byte each, 4bit halves a halfword.
    static constexpr bool kSplitChunk = (BITS == 2) || (BITS == 4);
    static constexpr uint32_t kMergeShift = (BITS == 4) ? 16U : 8U;
};

template <uint8_t BITS>
class KvCacheTurboQuant {
public:
    static constexpr uint32_t kLevels = 1U << BITS;
    static constexpr uint32_t kIdxBytesPerRow = KCTQ_D * BITS / 8U;
    static constexpr uint32_t kChunksPerRow = KCTQ_D / PackCfg<BITS>::kChunkLanes;
    static constexpr uint32_t kChunksPerBatch = KCTQ_BATCH_ROWS * kChunksPerRow;

    // Arena slices, one KCTQ_BATCH_LANES (= 4KB) each. Slice 0 also holds the matmul C tile.
    static constexpr uint32_t kSliceRow = 0U;   // x / unit, y, p
    static constexpr uint32_t kSliceTmp = 1U;   // mask, fold scratch
    static constexpr uint32_t kSliceIdx = 2U;   // primary index, residual
    static constexpr uint32_t kSlicePrim = 3U;  // primary value
    static constexpr uint32_t kSliceWork = 4U;  // reduce work
    static constexpr uint32_t kSliceCv = 5U;    // folded chunk values
    static constexpr uint32_t kSliceCi = 6U;    // chunk words as int32
    static constexpr uint32_t kSliceCvHi = 7U;  // 4bit high halves
    static constexpr uint32_t kSliceCiHi = 8U;
    static constexpr uint32_t kSliceDenom = 9U;
    static constexpr uint32_t kSliceCount = 10U;

    __aicore__ inline KvCacheTurboQuant() = default;

    __aicore__ inline void Init(GM_ADDR kvVectors, GM_ADDR rotationMatrix, GM_ADDR qjlMatrix,
                                GM_ADDR quantIdx, GM_ADDR quantQjl, GM_ADDR quantNorm,
                                GM_ADDR quantGamma, GM_ADDR workspace,
                                const optiling::KvCacheTurboQuantTilingData *td)
    {
        td_ = td;
        coreIdx_ = GetBlockIdx();
        rowBegin_ = coreIdx_ * td->rowsPerCore;
        const uint32_t rowEnd = rowBegin_ + td->rowsPerCore;
        rowEnd_ = (rowEnd > td->totalRows) ? td->totalRows : rowEnd;
        rowsToCompute_ = td->rowsPerCore; // padded rows feed the GEMM tile but are never stored

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kvVectors),
                              static_cast<uint64_t>(td->totalRows) * KCTQ_D);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rotationMatrix),
                               static_cast<uint64_t>(KCTQ_D) * KCTQ_D);
        qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qjlMatrix),
                               static_cast<uint64_t>(KCTQ_D) * KCTQ_D);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantIdx),
                               static_cast<uint64_t>(td->totalRows) * kIdxBytesPerRow);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantQjl),
                                  static_cast<uint64_t>(td->totalRows) * td->qjlBytesPerRow);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantNorm),
                                static_cast<uint64_t>(td->totalRows));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantGamma),
                                 static_cast<uint64_t>(td->totalRows));

        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        wsUNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsU),
                                   static_cast<uint64_t>(td->alignedRows) * KCTQ_D);
        wsYNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsY),
                                   static_cast<uint64_t>(td->alignedRows) * KCTQ_D);
        wsRNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsR),
                                   static_cast<uint64_t>(td->alignedRows) * KCTQ_D);
        wsPNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsP),
                                   static_cast<uint64_t>(td->alignedRows) * KCTQ_D);
        wsNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsNorm),
                                  static_cast<uint64_t>(td->alignedRows));

        const uint64_t rowOffset = static_cast<uint64_t>(rowBegin_) * KCTQ_D;
        wsUSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsU) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);
        wsYSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsY) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);
        wsRSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsR) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);
        wsPSliceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + td->wsP) + rowOffset,
                                    static_cast<uint64_t>(rowsToCompute_) * KCTQ_D);

        // The matmul framework runs on both sub cores (KFC handshake), so both of them need the
        // system workspace and the C tile staging buffer. The C tile staging area and the vector
        // staging buffers are alive in disjoint phases of the pipeline (Gemm() finishes with a full
        // drain before the next vector stage starts), so they share one arena instead of each
        // claiming its own UB.
        SetSysWorkspace(workspace);
        const uint32_t tileLanes = static_cast<uint32_t>(td->cubeTiling.baseM) * KCTQ_D;
        const uint32_t arenaLanes = (tileLanes > KCTQ_BATCH_LANES * kSliceCount)
                                        ? tileLanes
                                        : KCTQ_BATCH_LANES * kSliceCount;
        pipe_.InitBuffer(arenaBuf_, arenaLanes * sizeof(float));
        pipe_.InitBuffer(mmLocalWsBuf_, KCTQ_MM_LOCAL_WS_BYTES);

        if (g_coreType == AIV) {
            pipe_.InitBuffer(inQue_, 1, KCTQ_BATCH_ROWS * KCTQ_D * sizeof(bfloat16_t));
            pipe_.InitBuffer(scalarBuf_, 512U * sizeof(float));
            pipe_.InitBuffer(bf16Buf_, KCTQ_SCALAR_SLOTS * sizeof(bfloat16_t));
            pipe_.InitBuffer(weightBuf_, KCTQ_BATCH_LANES * sizeof(float));
            pipe_.InitBuffer(weightHiBuf_, KCTQ_BATCH_LANES * sizeof(float));
            pipe_.InitBuffer(qjlWeightBuf_, KCTQ_BATCH_LANES * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mm_, &td_->cubeTiling);
        mm_.SetLocalWorkspace(mmLocalWsBuf_.Get<uint8_t>());

        if (g_coreType == AIV) {
            InitWeightTables();
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

    __aicore__ inline LocalTensor<float> Slice(uint32_t index) const
    {
        return const_cast<ArenaBuf &>(arenaBuf_).Get<float>()[index * KCTQ_BATCH_LANES];
    }

    // Distances between neighbouring levels; the primary value is rebuilt as
    //   level[0] + sum of selected deltas
    // which reproduces the table lookup of the golden reference up to the final rounding step.
    __aicore__ inline float Delta(uint32_t k) const
    {
        return CodeBook<BITS>::kLevel[k] - CodeBook<BITS>::kLevel[k - 1];
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

    // Lane weights for the vectorised packing. One reduction block (32B) holds exactly one
    // packing group, so weighting lane l of a group by 2^(BITS*l) and summing the block with
    // BlockReduceSum directly yields the packed chunk value. Values stay below 2^24 (2^16 for
    // the split 4bit halves) and are therefore exact in fp32. The tables are built once per
    // core, outside the row loop.
    __aicore__ inline void InitWeightTables()
    {
        LocalTensor<float> w = weightBuf_.Get<float>();
        LocalTensor<float> wh = weightHiBuf_.Get<float>();
        LocalTensor<float> wq = qjlWeightBuf_.Get<float>();
        // Only one packing group is written by scalar code; the rest of the batch is replicated
        // with vector copies. Writing all KCTQ_BATCH_LANES lanes scalar would cost more than the
        // whole per batch pipeline on the small decode cases.
        constexpr uint32_t kGroup = PackCfg<BITS>::kChunkLanes;
        for (uint32_t i = 0; i < kGroup; ++i) {
            wq.SetValue(i, Pow2(i)); // qjl packs 8 sign bits into one byte
            if constexpr (PackCfg<BITS>::kSplitChunk) {
                // low half covers lanes 0..3, high half covers lanes 4..7
                w.SetValue(i, (i < 4U) ? Pow2(static_cast<uint32_t>(BITS) * i) : 0.0f);
                wh.SetValue(i, (i >= 4U) ? Pow2(static_cast<uint32_t>(BITS) * (i - 4U)) : 0.0f);
            } else {
                w.SetValue(i, Pow2(static_cast<uint32_t>(BITS) * i));
                wh.SetValue(i, 0.0f);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);
        for (uint32_t span = kGroup; span < KCTQ_BATCH_LANES; span *= 2U) {
            Adds(w[span], w, 0.0f, span);
            Adds(wh[span], wh, 0.0f, span);
            Adds(wq[span], wq, 0.0f, span);
            PipeBarrier<PIPE_V>();
        }
    }

    // Weighted 8-lane fold of a whole batch: dst[8*k + j] = sum_l src[64*k + 8*j + l]*2^(BITS*l).
    // One repeat carries 8 blocks (mask = 64) and its eight sums are written consecutively
    // (dstRepStride = 1), so the folded values come out compact, eight chunks per 32B block.
    __aicore__ inline void FoldChunks(const LocalTensor<float> &dst, const LocalTensor<float> &scratch,
                                      const LocalTensor<float> &src, const LocalTensor<float> &weights,
                                      uint32_t lanes)
    {
        Mul(scratch, src, weights, lanes);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float>(dst, scratch, static_cast<int32_t>(lanes / 64U), 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    // Spread compact chunk words so that chunk m starts at element 8*m (byte 32*m). The byte wise
    // store needs that: DataCopyPad can only stride in whole 32B blocks, so the packed bytes of a
    // chunk have to start on a block boundary. Each Brcb repeat copies 8 source words, one per
    // 32B block, so it takes chunks/8 repeats.
    __aicore__ inline void SpreadChunks(const LocalTensor<int32_t> &dst, const LocalTensor<int32_t> &src,
                                        uint32_t chunks)
    {
        Brcb(dst, src, static_cast<uint8_t>(chunks / 8U), BrcbRepeatParams(1, 8));
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
    // One batch at a time: a single strided load, one cast, one division over the whole batch and
    // two strided stores replace the per row pipeline. The padded rows of the last batch feed the
    // GEMM tile but are never stored.
    __aicore__ inline void NormalizeRows()
    {
        LocalTensor<float> sq = scalarBuf_.Get<float>();                  // ||x||^2 at [16*row]
        LocalTensor<float> norm = scalarBuf_.Get<float>()[KCTQ_SCALAR_SLOTS]; // ||x|| at [16*row]
        LocalTensor<float> denom = Slice(kSliceDenom);

        for (uint32_t row0 = rowBegin_; row0 < rowBegin_ + rowsToCompute_; row0 += KCTQ_BATCH_ROWS) {
            const uint32_t validRows = (row0 + KCTQ_BATCH_ROWS <= rowEnd_)
                                           ? KCTQ_BATCH_ROWS
                                           : ((rowEnd_ > row0) ? (rowEnd_ - row0) : 0U);
            LocalTensor<float> x = Slice(kSliceRow);
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
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                ReduceSum<float>(sq[r * KCTQ_SCALAR_STRIDE], tmp[r * KCTQ_D], work,
                                 static_cast<int32_t>(KCTQ_D));
            }
            PipeBarrier<PIPE_V>();
            Sqrt(norm, sq, KCTQ_SCALAR_SLOTS);
            PipeBarrier<PIPE_V>();

            // Only the clamp stays scalar: it has to reproduce max(norm, 1e-30) exactly, which is
            // why the divisor is not derived from a reciprocal approximation.
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                const float n = norm.GetValue(r * KCTQ_SCALAR_STRIDE);
                div_[r] = (n > 1e-30f) ? n : 1e-30f;
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                Duplicate(denom[r * KCTQ_D], div_[r], KCTQ_D);
            }
            PipeBarrier<PIPE_V>();
            Div(x, x, denom, KCTQ_BATCH_LANES); // x := unit vector

            LocalTensor<bfloat16_t> nb = bf16Buf_.Get<bfloat16_t>();
            Cast(nb, norm, RoundMode::CAST_RINT, KCTQ_SCALAR_SLOTS);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (validRows > 0U) {
                // the valid norms sit on 32B block boundaries, so one call emits the row scalars
                DataCopyExtParams cpNorm{static_cast<uint16_t>(validRows), 2U, 0, 0, 0};
                DataCopyPad(normGm_[row0], nb, cpNorm);
                // fp32 norms for the quantize stage: 4 bytes out of every other 32B block
                DataCopyExtParams cpNormF{static_cast<uint16_t>(KCTQ_BATCH_ROWS), 4U, 1U, 0, 0};
                DataCopyPad(wsNormGm_[row0], norm, cpNormF);
            }
            DataCopyExtParams cpU{1, static_cast<uint32_t>(KCTQ_BATCH_LANES * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsUNormGm_[static_cast<uint64_t>(row0) * KCTQ_D], x, cpU);
            // UB -> GM copies are asynchronous: without draining them here the next batch would
            // overwrite the staging buffers while the copies that read them are still in flight.
            PipeBarrier<PIPE_ALL>();
        }
    }

    // ---- AIC: steps 3 and 7, dense fp32 GEMM ------------------------------------------------
    __aicore__ inline void Gemm(const GlobalTensor<float> &aGm, const GlobalTensor<float> &bGm,
                                const GlobalTensor<float> &cGm)
    {
        mm_.SetOrgShape(static_cast<int32_t>(rowsToCompute_), static_cast<int32_t>(KCTQ_D),
                        static_cast<int32_t>(KCTQ_D));
        mm_.SetHF32(false); // indices are decided on fp32 values, so HF32 must stay off
        mm_.SetTensorA(aGm);
        // The runtime flag is what actually drives the transpose unit; the template argument alone
        // does not switch it on.
        mm_.SetTensorB(bGm, true);

        uint32_t done = 0;
        while (mm_.template Iterate<true>()) {
            LocalTensor<float> cLocal = arenaBuf_.Get<float>();
            mm_.template GetTensorC<true>(cLocal, false, true);
            // The framework signals the arrival of the C tile on the vector pipe, so the scalar/MTE3
            // consumers of cLocal have to be ordered behind it explicitly.
            PipeBarrier<PIPE_ALL>();
            uint32_t rows = rowsToCompute_ - done;
            if (rows > static_cast<uint32_t>(td_->cubeTiling.baseM)) {
                rows = static_cast<uint32_t>(td_->cubeTiling.baseM);
            }
            if (g_coreType == AIV) {
                // Only the KFC client owns the received tile: the engine hands the C data to the
                // client UB, the cube side never fills its own staging buffer.
                DataCopyExtParams cpOut{1, static_cast<uint32_t>(rows * KCTQ_D * sizeof(float)), 0, 0, 0};
                DataCopyPad(cGm[static_cast<uint64_t>(done) * KCTQ_D], cLocal, cpOut);
            }
            done += rows;
            PipeBarrier<PIPE_ALL>();
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
            const uint32_t validRows = (row0 + KCTQ_BATCH_ROWS <= rowEnd_)
                                           ? KCTQ_BATCH_ROWS
                                           : ((rowEnd_ > row0) ? (rowEnd_ - row0) : 0U);
            const uint64_t base = static_cast<uint64_t>(row0) * KCTQ_D;
            LocalTensor<float> y = Slice(kSliceRow);
            DataCopyExtParams cpY{1, static_cast<uint32_t>(KCTQ_BATCH_LANES * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padY{false, 0, 0, 0};
            DataCopyPad(y, wsYNormGm_[base], cpY, padY);
            // the fp32 norms written by NormalizeRows, spread back onto 32B block boundaries
            DataCopyExtParams cpN{static_cast<uint16_t>(KCTQ_BATCH_ROWS), 4U, 0, 1U, 0};
            DataCopyPad(norm, wsNormGm_[row0], cpN, padY);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            LocalTensor<float> mask = Slice(kSliceTmp);
            LocalTensor<float> idx = Slice(kSliceIdx);
            LocalTensor<float> prim = Slice(kSlicePrim);
            Duplicate(idx, 0.0f, KCTQ_BATCH_LANES);
            Duplicate(prim, CodeBook<BITS>::kLevel[0], KCTQ_BATCH_LANES);
            for (uint32_t k = 1; k < kLevels; ++k) {
                // boundary = 0.5 * (level[k-1] + level[k]), computed exactly like the reference.
                // Vector instructions execute in issue order on one pipe, so the back-to-back
                // steps below need no barrier; only the scalar constant changes per segment.
                const float boundary =
                    (CodeBook<BITS>::kLevel[k - 1] + CodeBook<BITS>::kLevel[k]) * 0.5f;
                Adds(mask, y, -boundary, KCTQ_BATCH_LANES);
                Muls(mask, mask, KCTQ_STEP_SCALE, KCTQ_BATCH_LANES);
                Adds(mask, mask, -1.0f, KCTQ_BATCH_LANES);
                Maxs(mask, mask, 0.0f, KCTQ_BATCH_LANES);
                Mins(mask, mask, 1.0f, KCTQ_BATCH_LANES);
                Add(idx, idx, mask, KCTQ_BATCH_LANES);
                Axpy(prim, mask, Delta(k), KCTQ_BATCH_LANES);
            }
            PackIdx(row0, validRows, idx);

            LocalTensor<float> res = Slice(kSliceIdx);
            LocalTensor<float> work = Slice(kSliceWork);
            Sub(res, y, prim, KCTQ_BATCH_LANES);
            Mul(mask, res, res, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                ReduceSum<float>(sq[r * KCTQ_SCALAR_STRIDE], mask[r * KCTQ_D], work,
                                 static_cast<int32_t>(KCTQ_D));
            }
            PipeBarrier<PIPE_V>();
            Sqrt(sq, sq, KCTQ_SCALAR_SLOTS);
            Mul(gamma, sq, norm, KCTQ_SCALAR_SLOTS); // gamma = norm * rho
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                const float v = sq.GetValue(r * KCTQ_SCALAR_STRIDE);
                div_[r] = (v > 1e-30f) ? v : 1e-30f;
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                Duplicate(denom[r * KCTQ_D], div_[r], KCTQ_D);
            }
            PipeBarrier<PIPE_V>();
            Div(res, res, denom, KCTQ_BATCH_LANES); // residual unit vector (0 when residual is 0)

            LocalTensor<bfloat16_t> gb = bf16Buf_.Get<bfloat16_t>();
            Cast(gb, gamma, RoundMode::CAST_RINT, KCTQ_SCALAR_SLOTS);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (validRows > 0U) {
                DataCopyExtParams cpG{static_cast<uint16_t>(validRows), 2U, 0, 0, 0};
                DataCopyPad(gammaGm_[row0], gb, cpG);
            }
            DataCopyExtParams cpR{1, static_cast<uint32_t>(KCTQ_BATCH_LANES * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsRNormGm_[base], res, cpR);
            PipeBarrier<PIPE_ALL>(); // drain the copies before the next batch reuses the slices
        }
    }

    // ---- AIV: pack the primary indices ------------------------------------------------------------
    __aicore__ inline void PackIdx(uint32_t row0, uint32_t validRows, const LocalTensor<float> &idx)
    {
        LocalTensor<float> w = weightBuf_.Get<float>();
        LocalTensor<float> wv = Slice(kSliceTmp);
        LocalTensor<float> cv = Slice(kSliceCv);
        LocalTensor<float> ciF = Slice(kSliceCi);
        LocalTensor<float> spreadF = Slice(kSliceCiHi);
        LocalTensor<int32_t> ci = ciF.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> spread = spreadF.ReinterpretCast<int32_t>();

        FoldChunks(cv, wv, idx, w, KCTQ_BATCH_LANES);
        if constexpr (PackCfg<BITS>::kSplitChunk) {
            // A 2bit/4bit chunk is wider than one 16bit half, so the two halves are folded
            // separately and recombined in the integer domain. Either way the recombined words
            // carry exactly the little endian output bytes, so the batch leaves in one contiguous
            // copy (2 bytes per chunk for 2bit, 4 bytes per chunk for 4bit).
            LocalTensor<float> wh = weightHiBuf_.Get<float>();
            LocalTensor<float> cvh = Slice(kSliceCvHi);
            LocalTensor<float> wvh = Slice(kSliceWork);
            LocalTensor<float> cihF = Slice(kSliceCiHi);
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
        } else {
            // 1bit / 2bit / 3bit: the chunk width is not a power of two (or the value would not fit
            // a signed 16bit word), so the words cannot be narrowed to a packed element type;
            // spread them one per 32B block and store strided.
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
        PipeBarrier<PIPE_ALL>(); // drain the copy before the next batch reuses the slices
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
            DataCopyPad(p, wsPNormGm_[static_cast<uint64_t>(row0) * KCTQ_D], cpP, padP);
            // The GM -> UB copy is asynchronous: without draining it here the vector ops below
            // would read the previous batch (or the stale buffer content for the first one).
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            LocalTensor<float> mask = Slice(kSliceTmp);
            LocalTensor<float> wv = Slice(kSliceWork);
            Muls(mask, p, KCTQ_STEP_SCALE, KCTQ_BATCH_LANES);
            Adds(mask, mask, 1.0f, KCTQ_BATCH_LANES);
            Maxs(mask, mask, 0.0f, KCTQ_BATCH_LANES);
            Mins(mask, mask, 1.0f, KCTQ_BATCH_LANES); // 1[p >= 0]
            // The 8 sign bits of every chunk fold into one byte through the same weighted block
            // reduction the index packer uses; values stay below 2^8 and are exact in fp32.
            FoldChunks(cv, wv, mask, wq, KCTQ_BATCH_LANES);
            // A packed sign byte is a power of two bytes wide, so the batch narrows through half
            // (exact for integers this small) into a contiguous byte run and leaves in one copy.
            // Going through half is what avoids the per chunk 32B block store of the 3bit layout.
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
    ArenaBuf arenaBuf_;                                     // C tile + vector staging, see kSlice*
    TBuf<TPosition::VECCALC> mmLocalWsBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;                    // per row scalars and reduce work
    TBuf<TPosition::VECCALC> bf16Buf_;                      // norm / gamma store staging
    TBuf<TPosition::VECCALC> weightBuf_;                    // packing lane weights
    TBuf<TPosition::VECCALC> weightHiBuf_;
    TBuf<TPosition::VECCALC> qjlWeightBuf_;

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, float, true>; // matrices are handed transposed
    using cType = MatmulType<TPosition::LCM, CubeFormat::ND, float>;
    Matmul<aType, bType, cType> mm_;

    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;
    GlobalTensor<float> wsUNormGm_;
    GlobalTensor<float> wsYNormGm_;
    GlobalTensor<float> wsRNormGm_;
    GlobalTensor<float> wsPNormGm_;
    GlobalTensor<float> wsNormGm_;
    GlobalTensor<float> wsUSliceGm_;
    GlobalTensor<float> wsYSliceGm_;
    GlobalTensor<float> wsRSliceGm_;
    GlobalTensor<float> wsPSliceGm_;

    const optiling::KvCacheTurboQuantTilingData *td_ = nullptr;
    uint32_t coreIdx_ = 0;
    uint32_t rowBegin_ = 0;
    uint32_t rowEnd_ = 0;
    uint32_t rowsToCompute_ = 0;
    float div_[KCTQ_BATCH_ROWS] = {0.0f}; // per row divisors handed between scalar and vector ops
};

} // namespace NsKvCacheTurboQuant

#endif // KV_CACHE_TURBO_QUANT_IMPL_HPP
