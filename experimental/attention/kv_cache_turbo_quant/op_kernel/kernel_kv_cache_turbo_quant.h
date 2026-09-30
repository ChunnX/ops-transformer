/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file kernel_kv_cache_turbo_quant.h
 *
 * TurboQuant online vector-quantization encode kernel for standard MHA/GQA KV cache.
 *
 * Per (token, kv_head) vector x (head_dim = 128):
 *   1. x_norm = ||x||_2, u = x / x_norm
 *   2. y = H u                      (orthogonal rotation)
 *   3. idx[j] = mse_bits scalar quantization index of y[j] (Lloyd-Max centroids of N(0,1))
 *   4. y_hat = centroid[idx], r = y - y_hat, gamma = x_norm * ||r||_2
 *   5. qjl = sign(S * (r / ||r||_2))   (1-bit Gaussian projection sketch)
 * Outputs: bit-packed idx / qjl, bf16 x_norm and gamma.
 *
 * Implementation notes:
 *  - Pure vector kernel. IMPORTANT: on Ascend 910B (dav_m200), dependent vector
 *    instructions are NOT hardware-interlocked (neither RAW nor WAR); every
 *    dependent pair is separated with PipeBarrier<PIPE_V>().
 *  - Single phase per 16-vector tile: both matrices (H 64KB + S 64KB) stay resident
 *    in UB together with a 32KB half-width product buffer; no GM workspace is used.
 *  - The matrix-vector product y = M v is computed in two column halves. For each
 *    half, v[half] is replicated 16x (b16 buffer, 16 independent Adds) so one Mul
 *    instruction (16 repeats x 64 elems, src0 stride 16 blocks) covers 16 rows of
 *    the half-width product matrix [rows, 64]; a WholeReduceSum pass (64 elems/row,
 *    dst stride 1 element) plus one Add of the two halves yields y.
 *  - Quantization of 2 vectors is batched (256-wide ops): per boundary, one
 *    Duplicate (broadcast boundary) + Compare(GT) + Cast(u8->h->f32) chain yields
 *    an exact 0/1 indicator; idxF accumulates, y_hat accumulates via Axpy.
 *  - Bit packing is done in the fp32 domain in two half-words (lanes 0-3 / 4-7) so
 *    every group word is an exact fp32 integer even for 4-bit codes: word =
 *    lo | hi << (4*mse_bits), bytes little-endian, identical to the reference.
 *  - DataCopyPad is unsupported on Ascend 910B (dav_m200): work is assigned in
 *    chunks of 16 vectors so that every output region per chunk is a multiple of
 *    32B (idx 768B, qjl 256B, norm/gamma 32B) and flushed with plain DataCopy;
 *    the single global tail (< 16 vectors) is written element-wise via
 *    GlobalTensor::SetValue.
 */

#ifndef KV_CACHE_TURBO_QUANT_KERNEL_H_
#define KV_CACHE_TURBO_QUANT_KERNEL_H_

#include "kernel_operator.h"

using namespace AscendC;

class KernelKvCacheTurboQuant {
public:
    __aicore__ inline KernelKvCacheTurboQuant() {}

    __aicore__ inline void Init(GM_ADDR kvVectors, GM_ADDR rotationMatrix, GM_ADDR qjlMatrix, GM_ADDR quantIdx,
                                GM_ADDR quantQjl, GM_ADDR quantNorm, GM_ADDR quantGamma,
                                const KvCacheTurboQuantTilingData* tiling)
    {
        if ASCEND_IS_AIC {
            vecNum_ = 0;
            return;
        }
        totalVec_ = tiling->totalVec;
        qjlDim_ = tiling->qjlDim;
        mseBits_ = tiling->mseBits;
        nCent_ = 1 << mseBits_;
        idxBytesPerVec_ = HEAD_DIM * mseBits_ / 8;
        qjlBytesPerVec_ = qjlDim_ / 8;
        coreNum_ = tiling->coreNum;

        // work assignment in chunks of CHUNK_VEC vectors (keeps every per-chunk
        // output region a multiple of 32B, see file header notes)
        const int64_t chunkNum = (totalVec_ + CHUNK_VEC - 1) / CHUNK_VEC;
        // mix-kernel launches a 2x grid (AIC+AIV pairs); fold to the logical core id
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t base = chunkNum / static_cast<int64_t>(coreNum_);
        const int64_t rem = chunkNum % static_cast<int64_t>(coreNum_);
        int64_t myChunks = 0;
        int64_t chunkBegin = 0;
        if (blockIdx < rem) {
            myChunks = base + 1;
            chunkBegin = blockIdx * (base + 1);
        } else {
            myChunks = base;
            chunkBegin = rem * (base + 1) + (blockIdx - rem) * base;
        }
        vecBegin_ = chunkBegin * CHUNK_VEC;
        int64_t vecEnd = (chunkBegin + myChunks) * CHUNK_VEC;
        if (vecEnd > totalVec_) {
            vecEnd = totalVec_;
        }
        vecNum_ = (myChunks > 0) ? (vecEnd - vecBegin_) : 0;
        if (vecNum_ <= 0) {
            return;
        }

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(kvVectors), totalVec_ * HEAD_DIM);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotationMatrix), HEAD_DIM * HEAD_DIM);
        qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(qjlMatrix), qjlDim_ * HEAD_DIM);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(quantIdx), totalVec_ * idxBytesPerVec_);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(quantQjl), totalVec_ * qjlBytesPerVec_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(quantNorm), totalVec_);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(quantGamma), totalVec_);

        pipe_.InitBuffer(matHBuf_, HEAD_DIM * HEAD_DIM * sizeof(float));  // 64K rotation H
        pipe_.InitBuffer(matSBuf_, HEAD_DIM * HEAD_DIM * sizeof(float));  // 64K projection S
        pipe_.InitBuffer(mhBuf_, HEAD_DIM * HALF_DIM * sizeof(float));    // 32K half-width products
        pipe_.InitBuffer(zTileBuf_, TILE_VEC * HEAD_DIM * sizeof(float)); // 8K z / r / r_hat
        pipe_.InitBuffer(b16Buf_, B16_COPIES * HALF_DIM * sizeof(float)); // 4K replicated half vector
        pipe_.InitBuffer(xTileBuf_, TILE_VEC * HEAD_DIM * sizeof(bfloat16_t));
        pipe_.InitBuffer(xfBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(yBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(yhatBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(idxFBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(pBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(bvBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(wtLBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(wtHBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(indBBuf_, BATCH_VEC * HEAD_DIM * sizeof(uint8_t));
        pipe_.InitBuffer(wordIdxLoBuf_, TILE_VEC * WORDS_PER_VEC * sizeof(float));
        pipe_.InitBuffer(wordIdxHiBuf_, TILE_VEC * WORDS_PER_VEC * sizeof(float));
        pipe_.InitBuffer(t0Buf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(t1Buf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(normVecBuf_, TILE_VEC * sizeof(float));
        pipe_.InitBuffer(rnVecBuf_, TILE_VEC * sizeof(float));
        pipe_.InitBuffer(gammaVecBuf_, TILE_VEC * sizeof(float));
        pipe_.InitBuffer(normBFBuf_, TILE_VEC * sizeof(bfloat16_t));
        pipe_.InitBuffer(gammaBFBuf_, TILE_VEC * sizeof(bfloat16_t));
        pipe_.InitBuffer(idxStageBuf_, TILE_VEC * MAX_IDX_BYTES);
        pipe_.InitBuffer(qjlStageBuf_, TILE_VEC * WORDS_PER_VEC);
        pipe_.InitBuffer(redTmpBuf_, RED_TMP_ELEMS * sizeof(float));

        eventVS_ = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        eventSV_ = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        eventMte2V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        eventVMte2_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        eventVMte3_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        eventMte3V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);

        InitConstants();
    }

    __aicore__ inline void Process()
    {
        if (vecNum_ <= 0) {
            return;
        }
        DataCopy(matHBuf_.Get<float>(), rotGm_, HEAD_DIM * HEAD_DIM);
        DataCopy(matSBuf_.Get<float>(), qjlGm_, qjlDim_ * HEAD_DIM);
        SyncPipe<HardEvent::MTE2_V>();

        LocalTensor<bfloat16_t> xTile = xTileBuf_.Get<bfloat16_t>();
        LocalTensor<float> xf = xfBuf_.Get<float>();
        LocalTensor<float> zTile = zTileBuf_.Get<float>();
        LocalTensor<float> normVec = normVecBuf_.Get<float>();
        LocalTensor<float> rnVec = rnVecBuf_.Get<float>();
        LocalTensor<float> gammaVec = gammaVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> normBF = normBFBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> gammaBF = gammaBFBuf_.Get<bfloat16_t>();
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();
        LocalTensor<float> matH = matHBuf_.Get<float>();
        LocalTensor<float> matS = matSBuf_.Get<float>();

        for (int64_t tileStart = 0; tileStart < vecNum_; tileStart += TILE_VEC) {
            const int32_t nv =
                static_cast<int32_t>((vecNum_ - tileStart) < TILE_VEC ? (vecNum_ - tileStart) : TILE_VEC);
            const int64_t g0 = vecBegin_ + tileStart;

            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(xTile, kvGm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();

            // ---- pass A: squared norms (grouped) and unnormalized rotation z = H x ----
            LocalTensor<float> sqG = yBuf_.Get<float>(); // reuse: grouped squares
            for (int32_t v = 0; v < nv; ++v) {
                Cast(xf, xTile[v * HEAD_DIM], RoundMode::CAST_NONE, HEAD_DIM);
                PipeBarrier<PIPE_V>();
                Mul(sqG[(v % BATCH_VEC) * HEAD_DIM], xf, xf, HEAD_DIM);
                PipeBarrier<PIPE_V>();
                MatVecHalf(zTile[v * HEAD_DIM], matH, xf, HEAD_DIM);
                if ((v % BATCH_VEC) == (BATCH_VEC - 1) || v == nv - 1) {
                    const int32_t gb = (v % BATCH_VEC) + 1;
                    for (int32_t i = 0; i < gb; ++i) {
                        ReduceSum(normVec[v + 1 - gb + i], sqG[i * HEAD_DIM], redTmpBuf_.Get<float>()[i * 64],
                                  HEAD_DIM);
                    }
                    PipeBarrier<PIPE_V>();
                }
            }
            Sqrt(normVec, normVec, nv);

            SyncPipe<HardEvent::V_S>();
            for (int32_t v = 0; v < nv; ++v) {
                const float norm = normVec.GetValue(v);
                inv_[v] = (norm > 0.0f) ? 1.0f / ((norm < MIN_DENOM) ? MIN_DENOM : norm) : 0.0f;
            }
            SyncPipe<HardEvent::S_V>();

            // ---- pass B: y = z * inv, MSE quantize (batched), residual into zTile ----
            for (int32_t vb = 0; vb < nv; vb += BATCH_VEC) {
                const int32_t b = (nv - vb) < BATCH_VEC ? (nv - vb) : BATCH_VEC;
                QuantizeBatch(vb, b, zTile, normVec, rnVec);
            }
            Sqrt(rnVec, rnVec, nv);
            PipeBarrier<PIPE_V>();
            Mul(gammaVec, normVec, rnVec, nv);
            PipeBarrier<PIPE_V>();
            Cast(normBF, normVec, RoundMode::CAST_RINT, nv);
            PipeBarrier<PIPE_V>();
            Cast(gammaBF, gammaVec, RoundMode::CAST_RINT, nv);
            PipeBarrier<PIPE_V>();

            SyncPipe<HardEvent::V_S>();
            for (int32_t v = 0; v < nv; ++v) {
                const float rn = rnVec.GetValue(v);
                invR_[v] = (rn > 0.0f) ? 1.0f / ((rn < MIN_DENOM) ? MIN_DENOM : rn) : 0.0f;
            }
            SyncPipe<HardEvent::S_V>();

            // ---- pass C: r_hat (in place) then QJL projection + sign (batched) ----
            for (int32_t vb = 0; vb < nv; vb += BATCH_VEC) {
                const int32_t b = (nv - vb) < BATCH_VEC ? (nv - vb) : BATCH_VEC;
                for (int32_t i = 0; i < b; ++i) {
                    Muls(zTile[(vb + i) * HEAD_DIM], zTile[(vb + i) * HEAD_DIM], invR_[vb + i], HEAD_DIM);
                    PipeBarrier<PIPE_V>();
                    MatVecHalf(pBuf_.Get<float>()[i * HEAD_DIM], matS, zTile[(vb + i) * HEAD_DIM], qjlDim_);
                }
                QjlSignBatch(vb, b);
            }
            SyncPipe<HardEvent::V_S>();
            UnpackIdxAll(nv);
            SyncPipe<HardEvent::S_V>();
            SyncPipe<HardEvent::V_MTE3>();

            const int32_t nvBulk = nv & ~(CHUNK_VEC - 1);
            if (nvBulk > 0) {
                DataCopy(normGm_[g0], normBF, nvBulk);
                DataCopy(gammaGm_[g0], gammaBF, nvBulk);
                DataCopy(idxGm_[g0 * idxBytesPerVec_], idxStage, nvBulk * idxBytesPerVec_);
                DataCopy(qjlOutGm_[g0 * qjlBytesPerVec_], qjlStage, nvBulk * qjlBytesPerVec_);
            }
            if (nvBulk < nv) {
                FlushScalar(g0 + nvBulk, nvBulk, nv - nvBulk, normBF, gammaBF, idxStage, qjlStage);
            }
            SyncPipe<HardEvent::MTE3_V>();
        }
    }

private:
    static constexpr int32_t HEAD_DIM = 128;
    static constexpr int32_t TILE_VEC = 16;
    static constexpr int32_t CHUNK_VEC = 16;
    static constexpr int32_t BATCH_VEC = 2;
    static constexpr int32_t B16_COPIES = 16;
    static constexpr int32_t WORDS_PER_VEC = 16; // max packed words per vector (128 lanes / 8)
    static constexpr int32_t MAX_IDX_BYTES = 64; // 4-bit: 16 groups * 4 bytes
    static constexpr int32_t RED_TMP_ELEMS = 256;
    static constexpr int32_t HALF_DIM = HEAD_DIM / 2;
    static constexpr float MIN_DENOM = 1e-30f;
    static constexpr float IND_SCALE = 1.2676506002282294e30f; // 2^100, overflow-to-inf indicator scale

    template <HardEvent evt>
    __aicore__ inline void SyncPipe()
    {
        TEventID eventId = GetEventId<evt>();
        SetFlag<evt>(eventId);
        WaitFlag<evt>(eventId);
    }

    template <HardEvent evt>
    __aicore__ inline TEventID GetEventId()
    {
        if constexpr (evt == HardEvent::V_S) {
            return eventVS_;
        } else if constexpr (evt == HardEvent::S_V) {
            return eventSV_;
        } else if constexpr (evt == HardEvent::MTE2_V) {
            return eventMte2V_;
        } else if constexpr (evt == HardEvent::V_MTE2) {
            return eventVMte2_;
        } else if constexpr (evt == HardEvent::V_MTE3) {
            return eventVMte3_;
        } else {
            return eventMte3V_;
        }
    }

    __aicore__ inline void InitConstants()
    {
        // Lloyd-Max centroids of the standard normal distribution (fp32-exact literals,
        // identical to the reference implementation).
        static constexpr float CENT4[4] = {-0.13350332f, -0.04002048f, 0.04002048f, 0.13350332f};
        static constexpr float CENT8[8] = {-0.19020693f, -0.11878592f, -0.06682206f, -0.02166347f,
                                           0.02166347f,  0.06682206f,  0.11878592f,  0.19020693f};
        static constexpr float CENT16[16] = {-0.24148904f,  -0.18283178f,  -0.1429703f,  -0.11099271f,
                                             -0.083254285f, -0.058020826f, -0.03428064f, -0.01134237f,
                                             0.01134237f,   0.03428064f,   0.058020826f, 0.083254285f,
                                             0.11099271f,   0.1429703f,    0.18283178f,  0.24148904f};
        const float* cent = (mseBits_ == 2) ? CENT4 : (mseBits_ == 3 ? CENT8 : CENT16);
        for (int32_t i = 0; i < nCent_; ++i) {
            cent_[i] = cent[i];
        }
        // boundaries and deltas recomputed in fp32, bit-identical to the reference
        for (int32_t i = 0; i < nCent_ - 1; ++i) {
            bound_[i] = (cent_[i] + cent_[i + 1]) * 0.5f;
            delta_[i] = cent_[i + 1] - cent_[i];
        }
        // bit-pack weight vectors (BATCH_VEC copies of the 128-lane pattern):
        //   lo half: lanes 0-3 -> 2^(bits*lane); hi half: lanes 4-7 -> 2^(bits*(lane-4));
        //   qjl: 2^lane. The fp32 sums stay exact integers for all bit widths.
        LocalTensor<float> wtL = wtLBuf_.Get<float>();
        LocalTensor<float> wtH = wtHBuf_.Get<float>();
        for (int32_t d = 0; d < BATCH_VEC * HEAD_DIM; ++d) {
            const int32_t lane = d & 7;
            const float lo = (lane < 4) ? static_cast<float>(static_cast<int32_t>(1) << (mseBits_ * lane)) : 0.0f;
            const float hi =
                (lane >= 4) ? static_cast<float>(static_cast<int32_t>(1) << (mseBits_ * (lane - 4))) : 0.0f;
            wtL.SetValue(d, lo);
            wtH.SetValue(d, hi);
        }
        SyncPipe<HardEvent::S_V>();
    }

    // dst = M v for mat[rows, 128], computed as two 64-wide column halves.
    // For half h: v[h*64, h*64+64) is replicated 16x into b16 (independent Adds),
    // then one Mul call per 16 rows (16 repeats x 64 elems; mat advances 16
    // blocks/repeat, b16 8 blocks) fills the half-width products mh[rows, 64], and
    // a WholeReduceSum pass (64 elems/row, dst stride 1) yields the half row-sums.
    __aicore__ inline void MatVecHalf(const LocalTensor<float>& dst, const LocalTensor<float>& mat,
                                      const LocalTensor<float>& src, int32_t rows)
    {
        LocalTensor<float> mh = mhBuf_.Get<float>();
        LocalTensor<float> b16 = b16Buf_.Get<float>();
        LocalTensor<float> t0 = t0Buf_.Get<float>();
        LocalTensor<float> t1 = t1Buf_.Get<float>();

        const int32_t calls = (rows + B16_COPIES - 1) / B16_COPIES;
        for (int32_t h = 0; h < 2; ++h) {
            LocalTensor<float> t = (h == 0) ? t0 : t1;
            PipeBarrier<PIPE_V>(); // b16/mh reuse across halves
            for (int32_t i = 0; i < B16_COPIES; ++i) {
                Adds(b16[i * HALF_DIM], src[h * HALF_DIM], 0.0f, HALF_DIM);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t c = 0; c < calls; ++c) {
                const int32_t reps = (rows - c * B16_COPIES) < B16_COPIES ? (rows - c * B16_COPIES) : B16_COPIES;
                Mul(mh[c * B16_COPIES * HALF_DIM], mat[(c * B16_COPIES) * HEAD_DIM + h * HALF_DIM], b16, HALF_DIM, reps,
                    BinaryRepeatParams(1, 1, 1, HALF_DIM / 8, HEAD_DIM / 8, HALF_DIM / 8));
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(t, mh, HALF_DIM, rows, 1, 1, HALF_DIM / 8);
        }
        PipeBarrier<PIPE_V>();
        Add(dst, t0, t1, rows);
        PipeBarrier<PIPE_V>();
    }

    // batched MSE quantization of b vectors starting at vb: y = z*inv, boundary
    // compares, y_hat, residual into zTile slots, residual norms, idx pack words.
    __aicore__ inline void QuantizeBatch(int32_t vb, int32_t b, const LocalTensor<float>& zTile,
                                         const LocalTensor<float>& normVec, const LocalTensor<float>& rnVec)
    {
        LocalTensor<float> y = yBuf_.Get<float>();
        LocalTensor<float> yhat = yhatBuf_.Get<float>();
        LocalTensor<float> idxF = idxFBuf_.Get<float>();
        LocalTensor<float> indF = pBuf_.Get<float>(); // reuse as fp32 indicator
        const int32_t w = b * HEAD_DIM;

        for (int32_t i = 0; i < b; ++i) {
            Muls(y[i * HEAD_DIM], zTile[(vb + i) * HEAD_DIM], inv_[vb + i], HEAD_DIM);
        }
        PipeBarrier<PIPE_V>();
        Duplicate(yhat, cent_[0], w);
        PipeBarrier<PIPE_V>();
        Duplicate(idxF, 0.0f, w);
        PipeBarrier<PIPE_V>();
        for (int32_t k = 1; k < nCent_; ++k) {
            // ind = (y > bound) ? 1 : 0, exact via overflow-to-inf scaling
            Adds(indF, y, -bound_[k - 1], w);
            PipeBarrier<PIPE_V>();
            Maxs(indF, indF, 0.0f, w);
            PipeBarrier<PIPE_V>();
            Muls(indF, indF, IND_SCALE, w);
            PipeBarrier<PIPE_V>();
            Muls(indF, indF, IND_SCALE, w);
            PipeBarrier<PIPE_V>();
            Muls(indF, indF, IND_SCALE, w);
            PipeBarrier<PIPE_V>();
            Mins(indF, indF, 1.0f, w);
            PipeBarrier<PIPE_V>();
            Add(idxF, idxF, indF, w);
            PipeBarrier<PIPE_V>();
            Axpy(yhat, indF, delta_[k - 1], w);
            PipeBarrier<PIPE_V>();
        }
        Sub(zTile[vb * HEAD_DIM], y, yhat, w);
        PipeBarrier<PIPE_V>();
        // residual norms: grouped squares then independent ReduceSums
        for (int32_t i = 0; i < b; ++i) {
            Mul(y[i * HEAD_DIM], zTile[(vb + i) * HEAD_DIM], zTile[(vb + i) * HEAD_DIM], HEAD_DIM);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < b; ++i) {
            ReduceSum(rnVec[vb + i], y[i * HEAD_DIM], redTmpBuf_.Get<float>()[i * 64], HEAD_DIM);
        }
        PipeBarrier<PIPE_V>();
        // idx bit-pack: lo/hi half-words so every group word is an exact fp32 integer
        LocalTensor<float> wtL = wtLBuf_.Get<float>();
        LocalTensor<float> wtH = wtHBuf_.Get<float>();
        LocalTensor<float> wordLo = wordIdxLoBuf_.Get<float>();
        LocalTensor<float> wordHi = wordIdxHiBuf_.Get<float>();
        Mul(y, idxF, wtL, w);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(wordLo[vb * WORDS_PER_VEC], y, 8, b * WORDS_PER_VEC, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Mul(y, idxF, wtH, w);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(wordHi[vb * WORDS_PER_VEC], y, 8, b * WORDS_PER_VEC, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }

    // batched QJL sign + pack for b vectors whose projections are in pBuf slots
    __aicore__ inline void QjlSignBatch(int32_t vb, int32_t b)
    {
        LocalTensor<float> p = pBuf_.Get<float>();
        LocalTensor<float> bv = bvBuf_.Get<float>();
        LocalTensor<uint8_t> indB = indBBuf_.Get<uint8_t>();
        const int32_t w = b * HEAD_DIM;

        Duplicate(bv, 0.0f, w);
        PipeBarrier<PIPE_V>();
        // sign = (p >= 0) ? 1 : 0; Compare writes one bit per element (byte j bit i =
        // element 8j+i), which is exactly the packed qjl byte layout of the reference
        if (qjlDim_ == HEAD_DIM) {
            // direct-to-stage: batch bits are already per-vector contiguous
            Compare(qjlStageBuf_.Get<uint8_t>()[vb * qjlBytesPerVec_], p, bv, CMPMODE::GE, w);
        } else {
            Compare(indB, p, bv, CMPMODE::GE, w);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_S>();
            LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();
            for (int32_t i = 0; i < b; ++i) {
                for (int32_t by = 0; by < qjlBytesPerVec_; ++by) {
                    qjlStage.SetValue((vb + i) * qjlBytesPerVec_ + by, indB.GetValue(i * WORDS_PER_VEC + by));
                }
            }
            SyncPipe<HardEvent::S_V>();
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UnpackIdxAll(int32_t nv)
    {
        LocalTensor<float> wordLo = wordIdxLoBuf_.Get<float>();
        LocalTensor<float> wordHi = wordIdxHiBuf_.Get<float>();
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>();
        const int32_t hiShift = 4 * mseBits_;
        for (int32_t v = 0; v < nv; ++v) {
            const int32_t outBase = v * idxBytesPerVec_;
            for (int32_t g = 0; g < WORDS_PER_VEC; ++g) {
                const int32_t lo = static_cast<int32_t>(wordLo.GetValue(v * WORDS_PER_VEC + g));
                const int32_t hi = static_cast<int32_t>(wordHi.GetValue(v * WORDS_PER_VEC + g));
                const int32_t word = lo + (hi << hiShift);
                for (int32_t by = 0; by < mseBits_; ++by) {
                    idxStage.SetValue(outBase + g * mseBits_ + by, static_cast<uint8_t>(word >> (8 * by)));
                }
            }
        }
    }

    // scalar tail flush (< CHUNK_VEC vectors, only at the global end)
    __aicore__ inline void FlushScalar(int64_t g0, int32_t stageOff, int32_t nv, const LocalTensor<bfloat16_t>& normBF,
                                       const LocalTensor<bfloat16_t>& gammaBF, const LocalTensor<uint8_t>& idxStage,
                                       const LocalTensor<uint8_t>& qjlStage)
    {
        for (int32_t v = 0; v < nv; ++v) {
            const int64_t g = g0 + v;
            const int32_t s = stageOff + v;
            normGm_.SetValue(g, normBF.GetValue(s));
            gammaGm_.SetValue(g, gammaBF.GetValue(s));
            for (int32_t b = 0; b < idxBytesPerVec_; ++b) {
                idxGm_.SetValue(g * idxBytesPerVec_ + b, idxStage.GetValue(s * idxBytesPerVec_ + b));
            }
            for (int32_t b = 0; b < qjlBytesPerVec_; ++b) {
                qjlOutGm_.SetValue(g * qjlBytesPerVec_ + b, qjlStage.GetValue(s * qjlBytesPerVec_ + b));
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> matHBuf_;
    TBuf<QuePosition::VECCALC> matSBuf_;
    TBuf<QuePosition::VECCALC> mhBuf_;
    TBuf<QuePosition::VECCALC> zTileBuf_;
    TBuf<QuePosition::VECCALC> b16Buf_;
    TBuf<QuePosition::VECCALC> xTileBuf_;
    TBuf<QuePosition::VECCALC> xfBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
    TBuf<QuePosition::VECCALC> yhatBuf_;
    TBuf<QuePosition::VECCALC> idxFBuf_;
    TBuf<QuePosition::VECCALC> pBuf_;
    TBuf<QuePosition::VECCALC> bvBuf_;
    TBuf<QuePosition::VECCALC> wtLBuf_;
    TBuf<QuePosition::VECCALC> wtHBuf_;
    TBuf<QuePosition::VECCALC> indBBuf_;
    TBuf<QuePosition::VECCALC> wordIdxLoBuf_;
    TBuf<QuePosition::VECCALC> wordIdxHiBuf_;
    TBuf<QuePosition::VECCALC> t0Buf_;
    TBuf<QuePosition::VECCALC> t1Buf_;
    TBuf<QuePosition::VECCALC> normVecBuf_;
    TBuf<QuePosition::VECCALC> rnVecBuf_;
    TBuf<QuePosition::VECCALC> gammaVecBuf_;
    TBuf<QuePosition::VECCALC> normBFBuf_;
    TBuf<QuePosition::VECCALC> gammaBFBuf_;
    TBuf<QuePosition::VECCALC> idxStageBuf_;
    TBuf<QuePosition::VECCALC> qjlStageBuf_;
    TBuf<QuePosition::VECCALC> redTmpBuf_;

    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;

    int64_t totalVec_ = 0;
    int64_t qjlDim_ = 0;
    int32_t mseBits_ = 3;
    int32_t nCent_ = 8;
    int32_t idxBytesPerVec_ = 48;
    int32_t qjlBytesPerVec_ = 16;
    uint32_t coreNum_ = 1;
    int64_t vecNum_ = 0;
    int64_t vecBegin_ = 0;

    TEventID eventVS_ = 0;
    TEventID eventSV_ = 0;
    TEventID eventMte2V_ = 0;
    TEventID eventVMte2_ = 0;
    TEventID eventVMte3_ = 0;
    TEventID eventMte3V_ = 0;

    float cent_[16] = {0.0f};
    float bound_[15] = {0.0f};
    float delta_[15] = {0.0f};
    float inv_[TILE_VEC] = {0.0f};
    float invR_[TILE_VEC] = {0.0f};
};

#endif // KV_CACHE_TURBO_QUANT_KERNEL_H_
