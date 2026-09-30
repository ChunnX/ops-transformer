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
 * AIC and AIV run 1:2 (one cube core, two vector cores). Both sub kinds walk the same call sequence:
 * the matmul framework turns the AIV call into the KFC client and the AIC call into the serving
 * cube work, so a global SyncAll barrier must not be inserted between the phases (the cube side has
 * to stay in the serving loop). The two AIVs of one AIC are independent KFC clients, each with its
 * own A rows and its own C tiles. The intermediate matrices live in the user workspace. Everything
 * is computed in fp32; bf16 is produced exactly once, on the store of norm/gamma.
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
// Local workspace of the matmul framework. Its C delivery goes straight from L0C into the operand
// output (no UB staging), so nothing of the framework needs UB here. Measured on 910B3 with CANN
// 9.1.0: 8KB and 64KB behave identically (same results, same timing), i.e. the reservation is not
// used by this version; the buffer only stays in place because the interface asks for one.
constexpr uint32_t KCTQ_MM_LOCAL_WS_BYTES = 8U * 1024U;

// Vector cores per cube core. The kernel is declared as MIX_AIC_1_2, so one AIC is served by two
// AIVs and both of them are KFC clients of that AIC. Measured with a hardware probe on 910B3
// (CANN 9.1.0): the launch is blockDim AICs plus 2 * blockDim AIVs, an AIV sees its *global* index
// in GetBlockIdx() (0 .. 2 * usedCoreNum - 1) and GetSubBlockIdx() only repeats GetBlockIdx() % 2,
// so one AIV owns rowsPerCore / KCTQ_AIV_PER_AIC rows and the AIC that serves it is
// GetBlockIdx() / KCTQ_AIV_PER_AIC.
constexpr uint32_t KCTQ_AIV_PER_AIC = optiling::KCTQ_AIV_PER_AIC;

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

    // Arena slices, indexed by lane offset. The C tile no longer lives in UB (the cube hands it to
    // the workspace directly), so the arena only carries the row level staging buffers. Every buffer
    // an asynchronous store reads from has KCTQ_PARITY copies: batch k stores out of copy
    // k % KCTQ_PARITY and only the batch KCTQ_PARITY later overwrites that copy, which is what lets
    // the pipeline drain run once every KCTQ_PARITY batches instead of once every second batch.
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
    static constexpr uint32_t kArenaLanes = kSliceLevels + 128U;

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
        wsRowBase_ = coreIdx_ * td->aivRowStride; // baseM aligned row base of this AIV
        wsOff_ = wsRowBase_ - rowBegin_;
        wsRows_ = td->usedCoreNum * KCTQ_AIV_PER_AIC * td->aivRowStride;

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kvVectors),
                              static_cast<uint64_t>(td->totalRows) * KCTQ_D);
        const uint64_t matElems = static_cast<uint64_t>(KCTQ_D) * KCTQ_D;
        // Default interface: fp32 matrices. Interface extension: bf16 matrices, which this kernel
        // upcasts once into the workspace. bf16 values are exact in fp32, so the upcast reproduces
        // the reference implementation's _npu_fp32() step bit for bit.
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
            // the GEMMs read the fp32 copies, never the bf16 inputs
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

        // The matmul framework runs on both sub cores (KFC handshake), so both of them need the
        // system workspace and the C tile staging buffer. The C tile staging area and the vector
        // staging buffers are alive in disjoint phases of the pipeline (Gemm() finishes with a full
        // drain before the next vector stage starts), so they share one arena instead of each
        // claiming its own UB.
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
            pipe_.InitBuffer(offsetBuf_, 8192U);
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

    // Workspace row index of a pipeline row: the AIV rows are baseM aligned in the workspace, so
    // the row base of this AIV is not the row it starts at (see aivRowStride in the tiling).
    __aicore__ inline uint32_t WsRow(uint32_t row) const
    {
        return row + wsOff_;
    }

    // parity copy of a KCTQ_PARITY way double buffered staging area
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
            // Gather offset tables for the 3bit stream packing (see PackStream3Bit).
            // The offsets are byte offsets into the source UB buffer (verified on hardware).
            LocalTensor<uint32_t> off = offsetBuf_.Get<uint32_t>();
            for (uint32_t i = 0; i < 128U; ++i) {
                off.SetValue(i, 4U * ((i + 1U < 128U) ? (i + 1U) : 127U)); // +1 chunk view (tail clamped)
            }
            for (uint32_t i = 0; i < 128U; ++i) {
                // 4 chunks make 3 words: word i sits at lane i + i / 3; the tail entries are unused
                off.SetValue(128U + i, 4U * ((i < 96U) ? (i + i / 3U) : 0U));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID2);
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
        }
        // The per row divisors (max(rho, 1e-30) of every row) sit on the 16 element grid; the lane
        // wise divisor a whole batch needs is that grid read back by row, which is a plain gather
        // over a byte offset table (row r owns lanes [128 r, 128 r + 128) and reads byte 64 r).
        {
            LocalTensor<int32_t> offB = offsetBuf_.Get<int32_t>()[256];
            for (uint32_t r = 0; r < KCTQ_BATCH_ROWS; ++r) {
                Duplicate(offB[r * KCTQ_D], static_cast<int32_t>(64U * r), static_cast<int32_t>(KCTQ_D));
            }
            PipeBarrier<PIPE_V>();
        }
        // Codebook table for the primary value: it is looked up as level[idx] at run time, which is
        // the only way to be bit identical to the reference. Rebuilding it incrementally (level[0]
        // plus the rounded differences) drifts by one ULP - 5 of 8 levels at 3 bit, 6 of 16 at 4 bit
        // - which moves the residual and with it gamma (see the prim diagnosis in
        // docs/evidence/repro/2026-09-30_kctq_bitexact_reverse).
        {
            LocalTensor<float> lv = Slice(kSliceLevels);
            for (uint32_t k = 0; k < kLevels; ++k) {
                lv.SetValue(k, CodeBook<BITS>::kLevel[k]);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID2);
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
        }
    }

    // ---- bits=3 packing: chunks -> 3 of every 4 lanes -> one contiguous byte stream -----------------
    // A 24bit chunk has no packed container type, and storing 3 bytes out of every 32B block costs
    // about 8.7us per batch (measured: 128 sub 32B blocks). Here the 48 byte row is assembled in
    // place instead: lane L < 96 holds output word L of the batch stream (4 chunks = 3 words), so a
    // single DataCopyPad leaves with the whole batch.
    //   class 0 (L%4==0): ci[L]        | ci[L+1] << 24
    //   class 1 (L%4==1): ci[L] >> 8   | ci[L+1] << 16
    //   class 2 (L%4==2): ci[L] >> 16  | ci[L+1] << 8
    //   class 3          : junk, dropped by the compaction gather below
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
        // left halves (the a operand) into word, right halves pushed into ci1 in place: the three
        // masked writes touch disjoint lane classes, so the source survives until the last one
        Adds(word, ci, static_cast<int32_t>(0), cls0, 16U, up);
        ShiftRight(word, ci, static_cast<int32_t>(8), cls1, 16U, up);
        ShiftRight(word, ci, static_cast<int32_t>(16), cls2, 16U, up);
        ShiftLeft(ci1, ci1, static_cast<int32_t>(24), cls0, 16U, up);
        ShiftLeft(ci1, ci1, static_cast<int32_t>(16), cls1, 16U, up);
        ShiftLeft(ci1, ci1, static_cast<int32_t>(8), cls2, 16U, up);
        Or(word, word, ci1, KCTQ_BATCH_LANES);
        PipeBarrier<PIPE_V>();
        // compact 4 chunks -> 3 words: packed[i] = word[i + i/3]. ci itself is dead here and becomes
        // the store source, so the copy only reads the parity copy of the *next* batch again later.
        Gather(ci, word, off[128], 0U, 64U, 2U, 8U);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (validRows > 0U) {
            LocalTensor<uint8_t> bytes = ci.ReinterpretCast<uint8_t>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(validRows * kIdxBytesPerRow), 0, 0, 0};
            DataCopyPad(idxGm_[static_cast<uint64_t>(row0) * kIdxBytesPerRow], bytes, cp);
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

    // Lane wise divisor of a whole batch from the per row scalars on the 16 element grid. The vector
    // clamp is bit identical to the scalar comparison it replaces and the per row broadcast is one
    // gather (row r owns lanes [128 r, 128 r + 128) and reads its scalar at lane 16 r, byte 64 r).
    // Keeping both on the vector pipe removes the round trip through the scalar unit: reading the
    // sums back scalar wise (V_S, eight GetValue, S_V) exposed the completion latency of the reduce
    // and was measured at 1.1us per batch.
    __aicore__ inline void BroadcastDivisors(const LocalTensor<float> &dst, const LocalTensor<float> &scalars)
    {
        LocalTensor<float> clamped = Slice(kSliceCvHi);
        Maxs(clamped, scalars, 1e-30f, static_cast<int32_t>(KCTQ_SCALAR_SLOTS));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> offB = offsetBuf_.Get<uint32_t>()[256];
        Gather(dst, clamped, offB, 0U, 64U, static_cast<uint8_t>(KCTQ_BATCH_LANES / 64U), 8U);
        PipeBarrier<PIPE_V>();
    }

    // Row sums of a whole batch: dst[row * KCTQ_SCALAR_STRIDE] = sum of src[row * KCTQ_D .. + KCTQ_D).
    // The order below is bit exact with the golden reference (torch.linalg.vector_norm on the NPU).
    // That reduction folds a row onto one 64 lane accumulator and then reduces the accumulator with
    // a pairwise tree, i.e. exactly
    //     fold[i] = src[i] + src[i + 64]          (i < 64)
    //     sum     = pairwise tree over fold[0 .. 64)
    // The fold is stage 1 below, the tree is the two reduce intrinsics: the block sum of a 32B
    // block is the three pairwise levels of its eight elements, and the whole reduce of the eight
    // block sums of a row adds the three levels above them. Without the fold the same two
    // intrinsics differ from the reference on 5806 of 16384 rows, with it on 0 (see
    // docs/evidence/repro/2026-09-30_kctq_bitexact_reverse).
    __aicore__ inline void BatchRowSum(const LocalTensor<float> &dst, const LocalTensor<float> &tmp,
                                       const LocalTensor<float> &src)
    {
        // Stage 1: fold the two 64 lane halves of every row element wise. One repeat per row (64
        // lanes = 256B, so the full mask covers exactly one repeat) and the row strides are plain
        // block counts: 128 lanes = 16 blocks in the source, 64 lanes = 8 blocks in the target.
        LocalTensor<float> fold = tmp;
        LocalTensor<float> blk = tmp[KCTQ_BATCH_ROWS * 64U];
        uint64_t fullMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
        Add(fold, src, src[64U], fullMask, static_cast<uint8_t>(KCTQ_BATCH_ROWS),
            BinaryRepeatParams(1, 1, 1, 8, 16, 16));
        PipeBarrier<PIPE_V>();
        // Stage 2: the eight block sums of every folded row, packed one destination block per
        // repeat, so a row leaves its eight partial sums behind contiguously.
        BlockReduceSum<float>(blk, fold, static_cast<int32_t>(KCTQ_BATCH_ROWS), 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        // Stage 3: one whole reduce per row over its eight block sums. It writes one result per
        // repeat and its destination repeat stride counts in elements, so stride 16 lands the sums
        // on the 16 lane grid the rest of the pipeline expects (measured on hardware, see the probe
        // dump analysis in docs/evidence/repro/2026-09-23_kctq_v10_qstall). The mask is one block,
        // which the counting overload of this intrinsic cannot express: a count of eight is
        // special cased to a full mask there, hence the explicit bit mask.
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
    // One batch at a time: a single strided load, one cast, one division over the whole batch and
    // two strided stores replace the per row pipeline. The padded rows of the last batch feed the
    // GEMM tile but are never stored.
    __aicore__ inline void NormalizeRows()
    {
        LocalTensor<float> sq = scalarBuf_.Get<float>();                  // ||x||^2 at [16*row]

        for (uint32_t row0 = rowBegin_; row0 < rowBegin_ + rowsToCompute_; row0 += KCTQ_BATCH_ROWS) {
            // same KCTQ_PARITY way staging as QuantizeRows: the store sources alternate between the
            // copies, so the drain only has to run every KCTQ_PARITY batches
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
            // see QuantizeRows: the reduce has to complete before the sums are consumed
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
                // the copies written by the previous KCTQ_PARITY batches are about to be rewritten
                PipeBarrier<PIPE_ALL>();
            }
        }
        // phase end: the last batch's stores have to be visible to the cube before Gemm is called
        PipeBarrier<PIPE_ALL>();
    }

    // ---- AIV: interface extension, bf16 matrices upcast to fp32 -----------------------------------
    // The formal interface hands the matrices over as fp32. When they arrive as bf16 (the task book
    // allows "float32 / bfloat16" for both) the AIVs upcast them into the workspace once, so both
    // GEMMs keep running on fp32 operands and every number downstream stays identical. bf16 -> fp32
    // is exact, so the upcast cannot change a single bit of the result.
    //
    // Every AIV converts the WHOLE matrix, not a share of it, and that redundancy is the
    // synchronisation: the cube reads the matrix itself while the vector cores write it, and the
    // only ordering edge the framework gives us is "the cube waits for its own AIVs before it
    // consumes their tiles". By making each AIV publish the complete matrix before it publishes any
    // A tile, whichever pair the cube waits for has already written every row it will ever read.
    // Concurrent writers store the same value into the same address, so overlapping copies are
    // harmless. Splitting the rows over the cores instead (the first version) left rows that no AIV
    // of that cube's pair had written yet - measured as run to run varying idx/qjl mismatches. A
    // block wide barrier (SyncAll<false>) is not usable here: with 1 AIC : 2 AIV and only
    // usedCoreNum cores launched it traps the device.
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
        // downstream stages reuse kSliceCv, so the last store must have read it
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

    // norm / gamma store: bf16 on the formal interface, fp16 for the metadata extension. Both are
    // 16 bit wide, so the parity staging buffer is shared; only the element type differs and the
    // fp16 case rounds the same fp32 value to nearest even.
    template <typename TOut>
    __aicore__ inline void StoreMetaRows(const GlobalTensor<TOut> &dst, const LocalTensor<float> &src,
                                         uint32_t row0, uint32_t validRows, uint32_t pb)
    {
        LocalTensor<TOut> out = bf16Buf_.Get<TOut>()[pb * KCTQ_SCALAR_SLOTS];
        Cast(out, src, RoundMode::CAST_RINT, KCTQ_SCALAR_SLOTS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (validRows > 0U) {
            // the valid scalars sit on 32B block boundaries, so one call emits the row scalars
            DataCopyExtParams cp{static_cast<uint16_t>(validRows), 2U, 0, 0, 0};
            DataCopyPad(dst[row0], out, cp);
        }
    }

    // ---- AIC: steps 3 and 7, dense fp32 GEMM ------------------------------------------------
    // The cube writes every C tile straight into the workspace (enSequentialWrite = true, so the
    // address handed over is used as is): no UB staging and no client side copy, which keeps the
    // tile pipeline on the cube and off the vector core. The workspace rows of one AIV are baseM
    // aligned (see tiling), so the full baseM rows written for the last, partial tile can never
    // run into the next AIV's rows.
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

        const uint32_t baseM = static_cast<uint32_t>(td_->cubeTiling.baseM);
        uint32_t done = 0;
        // The framework would serve ceil(rowsPerCore / baseM) tiles per phase, but this AIV only
        // owns half of the rows of that core: stop as soon as its own rows are delivered instead
        // of running the remaining iterations as empty round trips.
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
            // Batches alternate between the two copies of the residual slice, so the store issued
            // by batch k only blocks the batch that rewrites the same copy (k + 2) - which lets the
            // pipeline drain below run on every other batch instead of on every batch.
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
            }
            // prim = level[idx], an exact table lookup. idx is a small exact integer, so the cast and
            // the byte offset scaling are exact and the gather returns the table entry itself. The
            // offsets are consumed one per element (64 per repeat, KCTQ_BATCH_LANES / 64 repeats)
            // and written back with one destination block (8 lanes) per repeat, which is the plain
            // layout of this batch. Byte offsets, as everywhere else in this kernel.
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
            // The reduce intrinsics are long latency instructions: a vector pipe barrier only orders
            // issue, so the sums have to be waited for explicitly before anything reads or rewrites
            // their slices. The level 2 ReduceSum this replaced carried the same wait internally.
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
                // the copies written by the previous KCTQ_PARITY batches are about to be rewritten
                PipeBarrier<PIPE_ALL>();
            }
        }
        // phase end: the last batch's stores have to be visible to the cube before Gemm is called
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
            // A 2bit/4bit chunk is wider than one 16bit half, so the two halves are folded
            // separately and recombined in the integer domain. Either way the recombined words
            // carry exactly the little endian output bytes, so the batch leaves in one contiguous
            // copy (2 bytes per chunk for 2bit, 4 bytes per chunk for 4bit).
            LocalTensor<float> wh = weightHiBuf_.Get<float>();
            LocalTensor<float> cvh = Slice(kSliceCvHi);
            LocalTensor<float> wvh = Slice(kSliceWork);
            LocalTensor<float> cihF = Slice(kSliceCiHi); // intra batch only, no parity copy needed
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
            // 3bit: a 24bit chunk has no packed container type and a 3 byte per block store costs
            // ~8.7us per batch (measured), so the words are assembled inside the batch and leave in
            // one contiguous copy, see PackStream3Bit.
            Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_BATCH_LANES);
            PipeBarrier<PIPE_V>();
            PackStream3Bit(row0, validRows, ci);
        } else {
            // 1bit: the 2 byte chunks are not a packed container either; spread them one per 32B
            // block and store strided.
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
        // No drain here: the store reads slices that the *next* batch only rewrites after the
        // end-of-batch drain in QuantizeRows, which is what keeps this copy-issued window overlap
        // with the residual work of the same batch.
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
    // fp32 staging of the bf16 matrices (interface extension). A queue, not a raw arena lane: the
    // queue owns the V -> MTE3 and MTE3 -> V syncs, so the next Cast can never overwrite the buffer
    // while the previous DataCopyPad is still reading it.
    TQue<TPosition::VECOUT, 1> outQue_;
    ArenaBuf arenaBuf_;                                     // C tile + vector staging, see kSlice*
    TBuf<TPosition::VECCALC> mmLocalWsBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;                    // per row scalars and reduce work
    TBuf<TPosition::VECCALC> bf16Buf_;                      // norm / gamma store staging
    TBuf<TPosition::VECCALC> weightBuf_;                    // packing lane weights
    TBuf<TPosition::VECCALC> weightHiBuf_;
    TBuf<TPosition::VECCALC> qjlWeightBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;                      // 3bit gather offset tables (2 x 128 u32)

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
    uint32_t rowsPerAiv_ = 0; // rowsPerCore of the tiling (one AIC) split over KCTQ_AIV_PER_AIC AIVs
    uint32_t rowBegin_ = 0;
    uint32_t rowEnd_ = 0;
    uint32_t rowsToCompute_ = 0;
    uint32_t wsRowBase_ = 0; // baseM aligned workspace row base of this AIV
    uint32_t wsOff_ = 0;     // wsRowBase_ - rowBegin_
    uint32_t wsRows_ = 0;    // rows reserved in the workspace in total
};

} // namespace NsKvCacheTurboQuant

#endif // KV_CACHE_TURBO_QUANT_IMPL_HPP
