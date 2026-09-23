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
 *   7. p    = qjl_matrix * r_unit                                         (AIC, fp32 cube)
 *   8. qjl  = 1[p >= 0] bit packed together with the packed idx           (AIV)
 *
 * AIC and AIV run 1:1; the intermediate matrices live in the user workspace and the phases are
 * separated by SyncAll barriers. Everything is computed in fp32; bf16 is produced exactly once,
 * on the store of norm/gamma.
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
    static constexpr uint32_t kChunkCount = (KCTQ_D * BITS) / 16U;
    static constexpr uint32_t kRepStride = (BITS == 1) ? 2U : 1U;
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
        // TEMP DEBUG: float views of the two byte outputs, used to read back intermediate results
        dbgIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(quantIdx), 96U);
        dbgQjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(quantQjl), 32U);
        dbgGammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(quantGamma), 4U);
        dbgNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(quantNorm), 4U);

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
            pipe_.InitBuffer(valBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(wBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(wHiBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(qjlWBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(workBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(denomBuf_, KCTQ_D * sizeof(float));
            pipe_.InitBuffer(scalarBuf_, PackCfg<BITS>::kChunkSlots * sizeof(float));
            pipe_.InitBuffer(chunkBuf_, PackCfg<BITS>::kChunkSlots * sizeof(float));
            pipe_.InitBuffer(chunkLoBuf_, PackCfg<BITS>::kChunkSlots * sizeof(int32_t));
            pipe_.InitBuffer(chunkHiBuf_, PackCfg<BITS>::kChunkSlots * sizeof(int32_t));
            pipe_.InitBuffer(bf16Buf_, 32);
            pipe_.InitBuffer(byteBuf_, 64);
            pipe_.InitBuffer(dbgBuf_, KCTQ_D * sizeof(float)); // TEMP DEBUG
            pipe_.InitBuffer(dbgStashBuf_, 32);    // TEMP DEBUG
            pipe_.InitBuffer(scalarStashBuf_, 32); // TEMP DEBUG
            BuildWeightTables();
        }
    }

    __aicore__ inline void Process()
    {
        REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mm_, &td_->cubeTiling);
        mm_.SetLocalWorkspace(mmLocalWsBuf_.Get<uint8_t>());

        if (g_coreType == AIV) {
            NormalizeRows();
        }
        // SyncAll<false>(); // TEMP experiment

        // Both sub cores walk the same call sequence: the framework turns the AIV call into the
        // KFC client and the AIC call into the serving cube work.
        Gemm(wsUSliceGm_, rotGm_, wsYSliceGm_);
        // SyncAll<false>(); // TEMP experiment

        if (g_coreType == AIV) {
            QuantizeRows();
        }
        // SyncAll<false>(); // TEMP experiment

        Gemm(wsRSliceGm_, qjlGm_, wsPSliceGm_);
        // SyncAll<false>(); // TEMP experiment

        if (g_coreType == AIV) {
            QjlRows();
            // DumpGemmOutput(); // TEMP DEBUG
        }
    }

    // TEMP DEBUG: snapshot of the GEMM result (all 16 rows) plus the values that QuantizeRows saw.
    __aicore__ inline void DumpGemmOutput()
    {
        LocalTensor<float> dbg = dbgBuf_.Get<float>();
        for (uint32_t j = 0; j < 16U; ++j) {
            DataCopyExtParams cpD{1, static_cast<uint32_t>(8U * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padD{false, 0, 0, 0};
            DataCopyPad(dbg[static_cast<uint64_t>(j) * 8U],
                        wsYNormGm_[static_cast<uint64_t>(j) * KCTQ_D], cpD, padD);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyExtParams cpHead{1, static_cast<uint32_t>(96U * sizeof(float)), 0, 0, 0};
        DataCopyPad(dbgIdxGm_, dbg, cpHead);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cpTail{1, static_cast<uint32_t>(32U * sizeof(float)), 0, 0, 0};
        DataCopyPad(dbgQjlGm_, dbg[96U], cpTail);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cpY{1, static_cast<uint32_t>(4U * sizeof(float)), 0, 0, 0};
        DataCopyPad(dbgGammaGm_, dbgStashBuf_.Get<float>(), cpY);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cpS{1, static_cast<uint32_t>(4U * sizeof(float)), 0, 0, 0};
        DataCopyPad(dbgNormGm_, scalarStashBuf_.Get<float>(), cpS);
        PipeBarrier<PIPE_ALL>();
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

    __aicore__ inline void BuildWeightTables()
    {
        LocalTensor<float> w = wBuf_.Get<float>();
        LocalTensor<float> wHi = wHiBuf_.Get<float>();
        LocalTensor<float> qw = qjlWBuf_.Get<float>();
        for (uint32_t lane = 0; lane < KCTQ_D; ++lane) {
            const uint32_t c = lane % PackCfg<BITS>::kChunkLanes;
            float lo = Pow2(static_cast<uint32_t>(BITS * c));
            float hi = 0.0f;
            if constexpr (BITS == 4) {
                lo = (c < 4) ? Pow2(4U * c) : 0.0f;
                hi = (c >= 4) ? Pow2(4U * (c - 4U)) : 0.0f;
            }
            w.SetValue(lane, lo);
            wHi.SetValue(lane, hi);
            qw.SetValue(lane, Pow2(lane % kQjlChunkLanes));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
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
            if (row == rowBegin_) { // TEMP DEBUG: y as seen by QuantizeRows (first 4 lanes)
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                LocalTensor<float> dg = dbgStashBuf_.Get<float>();
                for (uint32_t i = 0; i < 4U; ++i) {
                    dg.SetValue(i, y.GetValue(i));
                }
            }

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
            if (row == rowBegin_) { // TEMP DEBUG: rho / norm / gamma / rho^2 of the first row
                LocalTensor<float> ds = scalarStashBuf_.Get<float>();
                ds.SetValue(0, sc.GetValue(8));
                ds.SetValue(1, sc.GetValue(24));
                ds.SetValue(2, sc.GetValue(40));
                ds.SetValue(3, sc.GetValue(48));
            }
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
        LocalTensor<float> chunk = chunkBuf_.Get<float>();
        LocalTensor<int32_t> lo = chunkLoBuf_.Get<int32_t>();
        LocalTensor<int32_t> hi = chunkHiBuf_.Get<int32_t>();
        LocalTensor<uint8_t> out = byteBuf_.Get<uint8_t>();
        constexpr uint32_t CHUNK_LANES = PackCfg<BITS>::kChunkLanes;
        constexpr uint32_t CHUNK_BYTES = PackCfg<BITS>::kChunkBytes;
        constexpr uint32_t CHUNKS = KCTQ_D / CHUNK_LANES; // one packed word per chunk
        constexpr uint32_t LOW_LANES = PackCfg<BITS>::kSplitChunk ? 4U : CHUNK_LANES;

        // The lanes of one chunk are folded with plain scalar arithmetic: every intermediate value
        // stays below 2^24 (2^16 for the split 4bit chunks) and is therefore exact in fp32.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float radix = Pow2(BITS);
        for (uint32_t c = 0; c < CHUNKS; ++c) {
            float low = 0.0f;
            float w = 1.0f;
            for (uint32_t l = 0; l < LOW_LANES; ++l) {
                low += idx.GetValue(c * CHUNK_LANES + l) * w;
                w *= radix;
            }
            chunk.SetValue(c, low);
            if constexpr (PackCfg<BITS>::kSplitChunk) {
                float high = 0.0f;
                w = 1.0f;
                for (uint32_t l = LOW_LANES; l < CHUNK_LANES; ++l) {
                    high += idx.GetValue(c * CHUNK_LANES + l) * w;
                    w *= radix;
                }
                chunk.SetValue(PackCfg<BITS>::kHalfSlots + c, high);
            }
        }
        PipeBarrier<PIPE_ALL>(); // the scalar writes have to land before the vector cast reads them
        if constexpr (PackCfg<BITS>::kSplitChunk) {
            Cast(lo, chunk, RoundMode::CAST_RINT, CHUNKS);
            Cast(hi, chunk[PackCfg<BITS>::kHalfSlots], RoundMode::CAST_RINT, CHUNKS);
            PipeBarrier<PIPE_V>();
            ShiftLeft(hi, hi, static_cast<int32_t>(16), CHUNKS);
            PipeBarrier<PIPE_V>();
            Or(lo, lo, hi, CHUNKS);
        } else {
            Cast(lo, chunk, RoundMode::CAST_RINT, CHUNKS);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        for (uint32_t c = 0; c < CHUNKS; ++c) {
            const uint32_t word = static_cast<uint32_t>(lo.GetValue(c));
            for (uint32_t b = 0; b < CHUNK_BYTES; ++b) {
                out.SetValue(c * CHUNK_BYTES + b, static_cast<uint8_t>((word >> (8U * b)) & 0xFFU));
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cpIdx{1, kIdxBytesPerRow, 0, 0, 0};
        DataCopyPad(idxGm_[static_cast<uint64_t>(row) * kIdxBytesPerRow], out, cpIdx);
        PipeBarrier<PIPE_ALL>(); // drain the copy before the next row reuses byteBuf_
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
            LocalTensor<float> chunk = chunkBuf_.Get<float>();
            LocalTensor<int32_t> lo = chunkLoBuf_.Get<int32_t>();
            LocalTensor<uint8_t> out = byteBuf_.Get<uint8_t>();

            Muls(mask, p, KCTQ_STEP_SCALE, KCTQ_D);
            Adds(mask, mask, 1.0f, KCTQ_D);
            Maxs(mask, mask, 0.0f, KCTQ_D);
            Mins(mask, mask, 1.0f, KCTQ_D); // 1[p >= 0]
            PipeBarrier<PIPE_V>();
            // fold the sign bits of every chunk with exact scalar arithmetic (values stay < 2^16)
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (uint32_t c = 0; c < kQjlChunkCount; ++c) {
                float word = 0.0f;
                float w = 1.0f;
                for (uint32_t l = 0; l < kQjlChunkLanes; ++l) {
                    word += mask.GetValue(c * kQjlChunkLanes + l) * w;
                    w *= 2.0f;
                }
                chunk.SetValue(c, word);
            }
            PipeBarrier<PIPE_ALL>(); // scalar writes must land before the vector cast reads them
            Cast(lo, chunk, RoundMode::CAST_RINT, kQjlChunkCount);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            for (uint32_t c = 0; c < kQjlChunkCount; ++c) {
                const uint32_t word = static_cast<uint32_t>(lo.GetValue(c));
                out.SetValue(c, static_cast<uint8_t>(word & 0xFFU));
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams cpQjl{1, td_->qjlBytesPerRow, 0, 0, 0};
            DataCopyPad(qjlOutGm_[static_cast<uint64_t>(row) * td_->qjlBytesPerRow], out, cpQjl);
            PipeBarrier<PIPE_ALL>(); // drain the copy before the next row reuses byteBuf_
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
    TBuf<TPosition::VECCALC> valBuf_;
    TBuf<TPosition::VECCALC> wBuf_;
    TBuf<TPosition::VECCALC> wHiBuf_;
    TBuf<TPosition::VECCALC> qjlWBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> denomBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    TBuf<TPosition::VECCALC> chunkBuf_;
    TBuf<TPosition::VECCALC> chunkLoBuf_;
    TBuf<TPosition::VECCALC> chunkHiBuf_;
    TBuf<TPosition::VECCALC> bf16Buf_;
    TBuf<TPosition::VECCALC> byteBuf_;
    TBuf<TPosition::VECCALC> dbgBuf_; // TEMP DEBUG
    TBuf<TPosition::VECCALC> dbgStashBuf_;    // TEMP DEBUG
    TBuf<TPosition::VECCALC> scalarStashBuf_; // TEMP DEBUG

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
    GlobalTensor<float> dbgIdxGm_; // TEMP DEBUG
    GlobalTensor<float> dbgQjlGm_; // TEMP DEBUG
    GlobalTensor<float> dbgGammaGm_; // TEMP DEBUG
    GlobalTensor<float> dbgNormGm_;  // TEMP DEBUG
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
