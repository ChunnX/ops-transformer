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
    static constexpr bool kSplitChunk = (BITS == 4);
    static constexpr uint32_t kChunkSlots = 64;
    static constexpr uint32_t kHalfSlots = kChunkSlots / 2;
};

template <uint8_t BITS>
class KvCacheTurboQuant {
public:
    static constexpr uint32_t kLevels = 1U << BITS;
    static constexpr uint32_t kIdxBytesPerRow = KCTQ_D * BITS / 8U;
    static constexpr uint32_t kQjlChunkLanes = 8U;                      // 8 sign bits per packed byte
    static constexpr uint32_t kQjlChunkCount = KCTQ_D / kQjlChunkLanes; // 16 chunks -> 16 bytes

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
        // system workspace and the C tile staging buffer.
        SetSysWorkspace(workspace);
        pipe_.InitBuffer(cTileBuf_,
                         static_cast<uint64_t>(td->cubeTiling.baseM) * KCTQ_D * sizeof(float));
        pipe_.InitBuffer(mmLocalWsBuf_, KCTQ_MM_LOCAL_WS_BYTES);

        if (g_coreType == AIV) {
            pipe_.InitBuffer(inQue_, 1, KCTQ_D * sizeof(bfloat16_t));
            pipe_.InitBuffer(xBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(tmpBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(yBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(resBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(primBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(idxBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(workBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(denomBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(scalarBuf_, PackCfg<BITS>::kChunkSlots * sizeof(float));
            pipe_.InitBuffer(bf16Buf_, 32);
            pipe_.InitBuffer(weightBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(weightHiBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(qjlWeightBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(redBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(redHiBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(redIntBuf_, KCTQ_D * sizeof(int32_t));
            pipe_.InitBuffer(redIntHiBuf_, KCTQ_D * sizeof(int32_t));
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

private:
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
        for (uint32_t i = 0; i < KCTQ_D; ++i) {
            const uint32_t lane = i % PackCfg<BITS>::kChunkLanes;
            wq.SetValue(i, Pow2(lane)); // qjl packs 8 sign bits into one byte
            if constexpr (PackCfg<BITS>::kSplitChunk) {
                // 4bit: low half covers lanes 0..3, high half covers lanes 4..7
                w.SetValue(i, (lane < 4U) ? Pow2(4U * lane) : 0.0f);
                wh.SetValue(i, (lane >= 4U) ? Pow2(4U * (lane - 4U)) : 0.0f);
            } else {
                w.SetValue(i, Pow2(static_cast<uint32_t>(BITS) * lane));
                wh.SetValue(i, 0.0f);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);
    }

    // Weighted 8-lane fold of one fp32 vector: dst[c] = sum_l src[8c + l] * 2^(BITS*l).
    // One repeat is restricted to a single 32B block (mask = 8) and each repeat writes its single
    // sum one 32B block further (dstRepStride = 1), so chunk c lands at element 8*c. That spread
    // layout is what the byte-wise store needs: DataCopyPad can only stride in 32B blocks, so the
    // packed bytes of chunk c have to start on a block boundary.
    __aicore__ inline void FoldChunks(const LocalTensor<float> &dst, const LocalTensor<float> &scratch,
                                      const LocalTensor<float> &src, const LocalTensor<float> &weights)
    {
        Mul(scratch, src, weights, KCTQ_D);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float>(dst, scratch, KCTQ_D / PackCfg<BITS>::kChunkLanes, 8, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }

    // Store the packed chunks: nBytes are taken from the head of every 32B block of `src`
    // (one chunk per block) and appended to the row in gm. `gmOffset` selects which byte of the
    // chunk each store owns (used by the 4bit low/high halve split), and `gmGap` leaves room for
    // the bytes the other store owns.
    __aicore__ inline void StoreChunks(const GlobalTensor<uint8_t> &gm, uint64_t gmOffset,
                                       const LocalTensor<int32_t> &src, uint32_t nChunks,
                                       uint32_t nBytes, uint32_t gmGap)
    {
        LocalTensor<uint8_t> bytes = src.ReinterpretCast<uint8_t>();
        DataCopyExtParams cp{static_cast<uint16_t>(nChunks), nBytes, 0, gmGap, 0};
        DataCopyPad(gm[gmOffset], bytes, cp);
    }

    // ---- AIV: steps 1 and 2 ---------------------------------------------------------------------
    __aicore__ inline void NormalizeRows()
    {
        for (uint32_t row = rowBegin_; row < rowBegin_ + rowsToCompute_; ++row) {
            const bool valid = (row < rowEnd_);
            LocalTensor<float> x = xBuf_.Get<float>();
            if (valid) {
                LocalTensor<bfloat16_t> xBf = inQue_.AllocTensor<bfloat16_t>();
                DataCopyExtParams cpIn{1, static_cast<uint32_t>(KCTQ_D * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPadExtParams<bfloat16_t> padIn{false, 0, 0, 0};
                DataCopyPad(xBf, kvGm_[static_cast<uint64_t>(row) * KCTQ_D], cpIn, padIn);
                inQue_.EnQue(xBf);
                xBf = inQue_.DeQue<bfloat16_t>();
                Cast(x, xBf, RoundMode::CAST_NONE, KCTQ_D);
                inQue_.FreeTensor(xBf);
            } else {
                Duplicate(x, 0.0f, KCTQ_D); // padding rows only have to stay finite
            }

            LocalTensor<float> tmp = tmpBuf_.Get<float>();
            LocalTensor<float> work = workBuf_.Get<float>();
            LocalTensor<float> sc = scalarBuf_.Get<float>();
            LocalTensor<float> denom = denomBuf_.Get<float>();

            Mul(tmp, x, x, KCTQ_D);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(sc, tmp, work, static_cast<int32_t>(KCTQ_D));
            PipeBarrier<PIPE_V>();
            Sqrt(sc[8], sc, 1);             // sc[8] = ||x||
            Maxs(sc[16], sc[8], 1e-30f, 1); // clamp_min(||x||, 1e-30)
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float divisor = sc.GetValue(16);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);

            Duplicate(denom, divisor, KCTQ_D);
            PipeBarrier<PIPE_V>();
            Div(x, x, denom, KCTQ_D); // x := unit vector

            LocalTensor<bfloat16_t> nb = bf16Buf_.Get<bfloat16_t>();
            Cast(nb, sc[8], RoundMode::CAST_RINT, 1); // norm is rounded to bf16 exactly once
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (valid) {
                DataCopyExtParams cpNorm{1, static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(normGm_[row], nb, cpNorm);
                DataCopyExtParams cpNormF{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(wsNormGm_[row], sc[8], cpNormF);
            }
            DataCopyExtParams cpU{1, static_cast<uint32_t>(KCTQ_D * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsUNormGm_[static_cast<uint64_t>(row) * KCTQ_D], x, cpU);
            // UB -> GM copies are asynchronous: without draining them here the next row would
            // overwrite xBuf_/inQue_ while the copy that reads them is still in flight.
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
            LocalTensor<float> cLocal = cTileBuf_.Get<float>();
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
        for (uint32_t row = rowBegin_; row < rowEnd_; ++row) {
            LocalTensor<float> y = yBuf_.Get<float>();
            LocalTensor<float> sc = scalarBuf_.Get<float>();
            DataCopyExtParams cpY{1, static_cast<uint32_t>(KCTQ_D * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padY{false, 0, 0, 0};
            DataCopyPad(y, wsYNormGm_[static_cast<uint64_t>(row) * KCTQ_D], cpY, padY);
            DataCopyExtParams cpN{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(sc[24], wsNormGm_[row], cpN, padY);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            LocalTensor<float> mask = tmpBuf_.Get<float>();
            LocalTensor<float> idx = idxBuf_.Get<float>();
            LocalTensor<float> prim = primBuf_.Get<float>();
            Duplicate(idx, 0.0f, KCTQ_D);
            Duplicate(prim, CodeBook<BITS>::kLevel[0], KCTQ_D);
            for (uint32_t k = 1; k < kLevels; ++k) {
                // boundary = 0.5 * (level[k-1] + level[k]), computed exactly like the reference
                const float boundary =
                    (CodeBook<BITS>::kLevel[k - 1] + CodeBook<BITS>::kLevel[k]) * 0.5f;
                Adds(mask, y, -boundary, KCTQ_D);
                PipeBarrier<PIPE_V>();
                Muls(mask, mask, KCTQ_STEP_SCALE, KCTQ_D);
                Adds(mask, mask, -1.0f, KCTQ_D);
                Maxs(mask, mask, 0.0f, KCTQ_D);
                Mins(mask, mask, 1.0f, KCTQ_D);
                PipeBarrier<PIPE_V>();
                Add(idx, idx, mask, KCTQ_D);
                Axpy(prim, mask, Delta(k), KCTQ_D);
                PipeBarrier<PIPE_V>();
            }
            PackIdx(row, idx);

            LocalTensor<float> res = resBuf_.Get<float>();
            LocalTensor<float> work = workBuf_.Get<float>();
            LocalTensor<float> denom = denomBuf_.Get<float>();
            Sub(res, y, prim, KCTQ_D);
            Mul(mask, res, res, KCTQ_D);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(sc[48], mask, work, static_cast<int32_t>(KCTQ_D)); // sc[48] = rho^2
            PipeBarrier<PIPE_V>();
            Sqrt(sc[8], sc[48], 1);            // sc[8] = rho
            Mul(sc[40], sc[8], sc[24], 1);     // gamma = norm * rho
            Maxs(sc[16], sc[8], 1e-30f, 1);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float rho = sc.GetValue(16);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);

            Duplicate(denom, rho, KCTQ_D);
            PipeBarrier<PIPE_V>();
            Div(res, res, denom, KCTQ_D); // residual unit vector (exactly 0 when the residual is 0)

            LocalTensor<bfloat16_t> gb = bf16Buf_.Get<bfloat16_t>();
            Cast(gb, sc[40], RoundMode::CAST_RINT, 1);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams cpG{1, static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(gammaGm_[row], gb, cpG);
            DataCopyExtParams cpR{1, static_cast<uint32_t>(KCTQ_D * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsRNormGm_[static_cast<uint64_t>(row) * KCTQ_D], res, cpR);
            PipeBarrier<PIPE_ALL>(); // drain the copy before the next row reuses resBuf_
        }
    }

    // ---- AIV: pack the primary indices ------------------------------------------------------------
    __aicore__ inline void PackIdx(uint32_t row, const LocalTensor<float> &idx)
    {
        LocalTensor<float> w = weightBuf_.Get<float>();
        LocalTensor<float> wv = workBuf_.Get<float>();
        LocalTensor<float> cv = redBuf_.Get<float>();
        LocalTensor<int32_t> ci = redIntBuf_.Get<int32_t>();

        FoldChunks(cv, wv, idx, w);
        if constexpr (PackCfg<BITS>::kSplitChunk) {
            // A 4bit chunk needs 32 bits, which fp32 cannot hold exactly, so the two 16bit halves
            // are folded separately and recombined in the integer domain. The vector shift/or only
            // acts correctly when the element count is a multiple of 64, hence the full 128 lane
            // block is always processed.
            LocalTensor<float> wh = weightHiBuf_.Get<float>();
            LocalTensor<float> cvh = redHiBuf_.Get<float>();
            LocalTensor<float> wvh = resBuf_.Get<float>();
            LocalTensor<int32_t> cih = redIntHiBuf_.Get<int32_t>();
            FoldChunks(cvh, wvh, idx, wh);
            Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_D);
            Cast(cih, cvh, RoundMode::CAST_RINT, KCTQ_D);
            PipeBarrier<PIPE_V>();
            ShiftLeft(cih, cih, 16, KCTQ_D);
            PipeBarrier<PIPE_V>();
            Add(ci, ci, cih, KCTQ_D);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_D);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // Every chunk lives at the head of its own 32B block, so one strided store emits exactly
        // kIdxBytesPerRow packed bytes for the row.
        StoreChunks(idxGm_, static_cast<uint64_t>(row) * kIdxBytesPerRow, ci,
                    KCTQ_D / PackCfg<BITS>::kChunkLanes, kIdxBytesPerRow / (KCTQ_D / PackCfg<BITS>::kChunkLanes), 0);
        PipeBarrier<PIPE_ALL>(); // drain the copy before the next row reuses the staging buffers
    }

    // ---- AIV: step 8 ------------------------------------------------------------------------------
    __aicore__ inline void QjlRows()
    {
        for (uint32_t row = rowBegin_; row < rowEnd_; ++row) {
            LocalTensor<float> p = yBuf_.Get<float>();
            DataCopyExtParams cpP{1, static_cast<uint32_t>(KCTQ_D * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padP{false, 0, 0, 0};
            DataCopyPad(p, wsPNormGm_[static_cast<uint64_t>(row) * KCTQ_D], cpP, padP);
            // The GM -> UB copy is asynchronous: without draining it here the vector ops below
            // would read the previous row (or the stale buffer content for the first row).
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            LocalTensor<float> mask = tmpBuf_.Get<float>();
            LocalTensor<float> wv = workBuf_.Get<float>();
            LocalTensor<float> cv = redBuf_.Get<float>();
            LocalTensor<int32_t> ci = redIntBuf_.Get<int32_t>();
            LocalTensor<float> wq = qjlWeightBuf_.Get<float>();

            Muls(mask, p, KCTQ_STEP_SCALE, KCTQ_D);
            Adds(mask, mask, 1.0f, KCTQ_D);
            Maxs(mask, mask, 0.0f, KCTQ_D);
            Mins(mask, mask, 1.0f, KCTQ_D); // 1[p >= 0]
            PipeBarrier<PIPE_V>();
            // The 8 sign bits of every chunk fold into one byte through the same weighted block
            // reduction the index packer uses, values stay below 2^8 and are exact in fp32.
            FoldChunks(cv, wv, mask, wq);
            Cast(ci, cv, RoundMode::CAST_RINT, KCTQ_D);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            // one packed sign byte per chunk, one chunk per 32B block
            StoreChunks(qjlOutGm_, static_cast<uint64_t>(row) * td_->qjlBytesPerRow, ci, kQjlChunkCount, 1, 0);
            PipeBarrier<PIPE_ALL>(); // drain the copy before the next row reuses the staging buffers
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQue_;

    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> resBuf_;
    TBuf<TPosition::VECCALC> primBuf_;
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> denomBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    TBuf<TPosition::VECCALC> bf16Buf_;
    // vectorised packing: lane weight tables plus the reduction / integer staging buffers
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> weightHiBuf_;
    TBuf<TPosition::VECCALC> qjlWeightBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    TBuf<TPosition::VECCALC> redHiBuf_;
    TBuf<TPosition::VECCALC> redIntBuf_;
    TBuf<TPosition::VECCALC> redIntHiBuf_;

    TBuf<TPosition::VECCALC> cTileBuf_;
    TBuf<TPosition::VECCALC> mmLocalWsBuf_;

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
};

} // namespace NsKvCacheTurboQuant

#endif // KV_CACHE_TURBO_QUANT_IMPL_HPP
