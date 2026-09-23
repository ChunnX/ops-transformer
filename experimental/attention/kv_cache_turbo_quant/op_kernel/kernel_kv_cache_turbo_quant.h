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
 *  - Pure vector kernel; vector instructions in the same pipe execute in order with
 *    hardware dependency handling, so only cross-pipe syncs are explicit.
 *  - The matrix-vector product y = M v is computed as elementwise products
 *    P = M (x) v_tiled followed by two ReduceSum passes (lower/upper 64-wide row
 *    halves) and one Add; the input vector is first replicated 8x so one Mul
 *    instruction (16 repeats x 64 elems) covers 8 rows.
 *  - Quantization decisions use an exact 0/1 indicator: (t > 0) is detected by
 *    overflow-to-inf scaling which maps every strictly positive float to 1.0.
 *  - Bit packing is done in the fp32 domain: idx * 2^(bits*(d mod 8)) summed per
 *    8-lane group yields the exact packed word (< 2^24), then bytes are emitted
 *    scalar-wise into a staging buffer and flushed with bulk DMA.
 *  - Two phases share one 64KB matrix buffer: phase 1 (rotation H) produces the
 *    normalized residual r_hat into the GM workspace; phase 2 (projection S)
 *    consumes it. No cross-core dependency exists.
 */

#ifndef KV_CACHE_TURBO_QUANT_KERNEL_H_
#define KV_CACHE_TURBO_QUANT_KERNEL_H_

#include "kernel_operator.h"

using namespace AscendC;

class KernelKvCacheTurboQuant {
public:
    __aicore__ inline KernelKvCacheTurboQuant() {}

    __aicore__ inline void Init(GM_ADDR kvVectors, GM_ADDR rotationMatrix, GM_ADDR qjlMatrix, GM_ADDR quantIdx,
                                GM_ADDR quantQjl, GM_ADDR quantNorm, GM_ADDR quantGamma, GM_ADDR workspace,
                                const KvCacheTurboQuantTilingData *tiling)
    {
        totalVec_ = tiling->totalVec;
        qjlDim_ = tiling->qjlDim;
        mseBits_ = tiling->mseBits;
        nCent_ = 1 << mseBits_;
        idxBytesPerVec_ = HEAD_DIM * mseBits_ / 8;
        qjlBytesPerVec_ = qjlDim_ / 8;
        coreNum_ = tiling->coreNum;

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t base = totalVec_ / static_cast<int64_t>(coreNum_);
        const int64_t rem = totalVec_ % static_cast<int64_t>(coreNum_);
        if (blockIdx < rem) {
            vecNum_ = base + 1;
            vecBegin_ = blockIdx * (base + 1);
        } else {
            vecNum_ = base;
            vecBegin_ = rem * (base + 1) + (blockIdx - rem) * base;
        }
        if (vecNum_ <= 0) {
            return;
        }

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kvVectors), totalVec_ * HEAD_DIM);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rotationMatrix), HEAD_DIM * HEAD_DIM);
        qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qjlMatrix), qjlDim_ * HEAD_DIM);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantIdx), totalVec_ * idxBytesPerVec_);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantQjl), totalVec_ * qjlBytesPerVec_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantNorm), totalVec_);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantGamma), totalVec_);
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), totalVec_ * HEAD_DIM);

        pipe_.InitBuffer(matBuf_, HEAD_DIM * HEAD_DIM * sizeof(float));       // 64K: H (phase1) / S (phase2)
        pipe_.InitBuffer(mBuf_, HEAD_DIM * HEAD_DIM * sizeof(float));         // 64K products
        pipe_.InitBuffer(zTileBuf_, TILE_VEC * HEAD_DIM * sizeof(float));     // 16K rotated vec / r_hat staging
        pipe_.InitBuffer(rTileBuf_, TILE_VEC * HEAD_DIM * sizeof(float));     // 16K residual
        pipe_.InitBuffer(b8Buf_, B8_COPIES * HEAD_DIM * sizeof(float));       // 4K replicated input vector
        pipe_.InitBuffer(xTileBuf_, TILE_VEC * HEAD_DIM * sizeof(bfloat16_t));
        pipe_.InitBuffer(xfBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(sqBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(yBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(t0Buf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(t1Buf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(indBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(yhatBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(idxFBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(pBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(wtBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(w1Buf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(wordTileBuf_, TILE_VEC * WORDS_PER_VEC * sizeof(float));
        pipe_.InitBuffer(normVecBuf_, TILE_VEC * sizeof(float));
        pipe_.InitBuffer(rnVecBuf_, TILE_VEC * sizeof(float));
        pipe_.InitBuffer(gammaVecBuf_, TILE_VEC * sizeof(float));
        pipe_.InitBuffer(normBFBuf_, TILE_VEC * sizeof(bfloat16_t));
        pipe_.InitBuffer(gammaBFBuf_, TILE_VEC * sizeof(bfloat16_t));
        pipe_.InitBuffer(idxStageBuf_, TILE_VEC * MAX_IDX_BYTES);
        pipe_.InitBuffer(qjlStageBuf_, TILE_VEC * WORDS_PER_VEC);
        pipe_.InitBuffer(redTmpBuf_, RED_TMP_ELEMS * sizeof(float));

        InitConstants();
    }

    __aicore__ inline void Process()
    {
        if (vecNum_ <= 0) {
            return;
        }
        // phase 1: rotation + MSE quantize + residual norms, write r_hat to workspace
        DataCopy(matBuf_.Get<float>(), rotGm_, HEAD_DIM * HEAD_DIM);
        SyncPipe(HardEvent::MTE2_V);
        Phase1();
        // make r_hat visible in GM before phase 2 reads it back
        PipeBarrier<PIPE_ALL>();
        // phase 2: QJL projection + sign sketch
        DataCopy(matBuf_.Get<float>(), qjlGm_, qjlDim_ * HEAD_DIM);
        SyncPipe(HardEvent::MTE2_V);
        Phase2();
    }

private:
    static constexpr int32_t HEAD_DIM = 128;
    static constexpr int32_t TILE_VEC = 32;
    static constexpr int32_t B8_COPIES = 8;
    static constexpr int32_t WORDS_PER_VEC = 16;   // max packed words per vector (128 lanes / 8)
    static constexpr int32_t MAX_IDX_BYTES = 64;   // 4-bit: 16 groups * 4 bytes
    static constexpr int32_t RED_TMP_ELEMS = 512;
    static constexpr int32_t HALF_DIM = HEAD_DIM / 2;
    static constexpr float IND_SCALE = 1.2676506002282294e30f;     // 2^100, overflow-to-inf indicator scale
    static constexpr float MIN_DENOM = 1e-30f;

    __aicore__ inline void SyncPipe(HardEvent evt)
    {
        TEventID eventId = GetTPipePtr()->FetchEventID(evt);
        SetFlag(evt, eventId);
        WaitFlag(evt, eventId);
    }

    __aicore__ inline void InitConstants()
    {
        // Lloyd-Max centroids of the standard normal distribution (fp32-exact literals,
        // identical to the reference implementation).
        static constexpr float CENT4[4] = {-0.13350332f, -0.04002048f, 0.04002048f, 0.13350332f};
        static constexpr float CENT8[8] = {-0.19020693f, -0.11878592f, -0.06682206f, -0.02166347f,
                                           0.02166347f,  0.06682206f,  0.11878592f,  0.19020693f};
        static constexpr float CENT16[16] = {
            -0.24148904f, -0.18283178f, -0.1429703f,  -0.11099271f, -0.083254285f, -0.058020826f,
            -0.03428064f, -0.01134237f, 0.01134237f,  0.03428064f,  0.058020826f, 0.083254285f,
            0.11099271f,  0.1429703f,   0.18283178f,  0.24148904f};
        const float *cent = (mseBits_ == 2) ? CENT4 : (mseBits_ == 3 ? CENT8 : CENT16);
        for (int32_t i = 0; i < nCent_; ++i) {
            cent_[i] = cent[i];
        }
        // boundaries and deltas recomputed in fp32, bit-identical to the reference
        for (int32_t i = 0; i < nCent_ - 1; ++i) {
            bound_[i] = (cent_[i] + cent_[i + 1]) * 0.5f;
            delta_[i] = cent_[i + 1] - cent_[i];
        }
        // bit-pack weight vectors: idx weights 2^(bits*lane), qjl weights 2^lane
        LocalTensor<float> wt = wtBuf_.Get<float>();
        LocalTensor<float> w1 = w1Buf_.Get<float>();
        for (int32_t d = 0; d < HEAD_DIM; ++d) {
            const int32_t lane = d & 7;
            wt.SetValue(d, static_cast<float>(1u << (mseBits_ * lane)));
            w1.SetValue(d, static_cast<float>(1u << lane));
        }
        SyncPipe(HardEvent::S_V);
    }

    // dst = M v for mat[rows, 128]; v = b8 (8 replicas of the 128-dim input vector)
    __aicore__ inline void MatVec(const LocalTensor<float> &dst, const LocalTensor<float> &mat,
                                  const LocalTensor<float> &b8, int32_t rows)
    {
        LocalTensor<float> m = mBuf_.Get<float>();
        LocalTensor<float> t0 = t0Buf_.Get<float>();
        LocalTensor<float> t1 = t1Buf_.Get<float>();
        LocalTensor<float> redTmp = redTmpBuf_.Get<float>();

        const int32_t calls = rows / B8_COPIES;
        for (int32_t c = 0; c < calls; ++c) {
            // 16 repeats x 64 elems: covers 8 rows, consumes one full b8 replica chain
            Mul(m[c * B8_COPIES * HEAD_DIM], mat[c * B8_COPIES * HEAD_DIM], b8, HALF_DIM, 2 * B8_COPIES,
                BinaryRepeatParams(1, 1, 1, HALF_DIM / 8, HALF_DIM / 8, HALF_DIM / 8));
        }
        // row sums of the two 64-wide halves, then add
        ReduceSum(t0, m, redTmp, HALF_DIM, rows, HEAD_DIM / 8);
        ReduceSum(t1, m[HALF_DIM], redTmp, HALF_DIM, rows, HEAD_DIM / 8);
        Add(dst, t0, t1, rows);
    }

    __aicore__ inline void Replicate8(const LocalTensor<float> &src)
    {
        LocalTensor<float> b8 = b8Buf_.Get<float>();
        for (int32_t i = 0; i < B8_COPIES; ++i) {
            Adds(b8[i * HEAD_DIM], src, 0.0f, HEAD_DIM);
        }
    }

    __aicore__ inline void Phase1()
    {
        LocalTensor<bfloat16_t> xTile = xTileBuf_.Get<bfloat16_t>();
        LocalTensor<float> xf = xfBuf_.Get<float>();
        LocalTensor<float> sq = sqBuf_.Get<float>();
        LocalTensor<float> y = yBuf_.Get<float>();
        LocalTensor<float> b8 = b8Buf_.Get<float>();
        LocalTensor<float> zTile = zTileBuf_.Get<float>();
        LocalTensor<float> rTile = rTileBuf_.Get<float>();
        LocalTensor<float> normVec = normVecBuf_.Get<float>();
        LocalTensor<float> rnVec = rnVecBuf_.Get<float>();
        LocalTensor<float> gammaVec = gammaVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> normBF = normBFBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> gammaBF = gammaBFBuf_.Get<bfloat16_t>();
        LocalTensor<float> redTmp = redTmpBuf_.Get<float>();
        LocalTensor<float> mat = matBuf_.Get<float>();

        for (int64_t tileStart = 0; tileStart < vecNum_; tileStart += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((vecNum_ - tileStart) < TILE_VEC ? (vecNum_ - tileStart) : TILE_VEC);
            const int64_t g0 = vecBegin_ + tileStart;

            SyncPipe(HardEvent::V_MTE2);
            DataCopy(xTile, kvGm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe(HardEvent::MTE2_V);

            // ---- pass A: squared norms and unnormalized rotation z = H x ----
            for (int32_t v = 0; v < nv; ++v) {
                Cast(xf, xTile[v * HEAD_DIM], RoundMode::CAST_NONE, HEAD_DIM);
                Mul(sq, xf, xf, HEAD_DIM);
                ReduceSum(normVec[v], sq, redTmp, HEAD_DIM);
                Replicate8(xf);
                MatVec(zTile[v * HEAD_DIM], mat, b8, HEAD_DIM);
            }
            Sqrt(normVec, normVec, nv);

            SyncPipe(HardEvent::V_S);
            for (int32_t v = 0; v < nv; ++v) {
                const float norm = normVec.GetValue(v);
                inv_[v] = (norm > 0.0f) ? 1.0f / ((norm < MIN_DENOM) ? MIN_DENOM : norm) : 0.0f;
            }
            SyncPipe(HardEvent::S_V);

            // ---- pass B: y = z * inv, MSE quantize, residual, residual norm, idx words ----
            for (int32_t v = 0; v < nv; ++v) {
                Muls(y, zTile[v * HEAD_DIM], inv_[v], HEAD_DIM);
                Quantize(v, y);
            }
            Sqrt(rnVec, rnVec, nv);
            Mul(gammaVec, normVec, rnVec, nv);
            Cast(normBF, normVec, RoundMode::CAST_RINT, nv);
            Cast(gammaBF, gammaVec, RoundMode::CAST_RINT, nv);

            SyncPipe(HardEvent::V_S);
            for (int32_t v = 0; v < nv; ++v) {
                const float rn = rnVec.GetValue(v);
                invR_[v] = (rn > 0.0f) ? 1.0f / ((rn < MIN_DENOM) ? MIN_DENOM : rn) : 0.0f;
            }
            UnpackIdx(nv);
            SyncPipe(HardEvent::S_V);

            // ---- pass C: r_hat staged into zTile slots, then bulk flush ----
            for (int32_t v = 0; v < nv; ++v) {
                Muls(zTile[v * HEAD_DIM], rTile[v * HEAD_DIM], invR_[v], HEAD_DIM);
            }
            SyncPipe(HardEvent::V_MTE3);

            DataCopy(wsGm_[g0 * HEAD_DIM], zTile, nv * HEAD_DIM);
            DataCopyExtParams bfParams{static_cast<uint16_t>(1), static_cast<uint32_t>(nv * sizeof(bfloat16_t)),
                                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
            DataCopyPad(normGm_[g0], normBF, bfParams);
            DataCopyPad(gammaGm_[g0], gammaBF, bfParams);
            DataCopyExtParams idxParams{static_cast<uint16_t>(1), static_cast<uint32_t>(nv * idxBytesPerVec_),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
            DataCopyPad(idxGm_[g0 * idxBytesPerVec_], idxStageBuf_.Get<uint8_t>(), idxParams);
            SyncPipe(HardEvent::MTE3_V);
        }
    }

    // MSE quantization of y for vector v: y_hat accumulation, residual, residual
    // sum-of-squares and weighted idx words for bit packing.
    __aicore__ inline void Quantize(int32_t v, const LocalTensor<float> &y)
    {
        LocalTensor<float> sq = sqBuf_.Get<float>();
        LocalTensor<float> ind = indBuf_.Get<float>();
        LocalTensor<float> yhat = yhatBuf_.Get<float>();
        LocalTensor<float> idxF = idxFBuf_.Get<float>();
        LocalTensor<float> p = pBuf_.Get<float>();
        LocalTensor<float> wt = wtBuf_.Get<float>();
        LocalTensor<float> rTile = rTileBuf_.Get<float>();
        LocalTensor<float> rnVec = rnVecBuf_.Get<float>();
        LocalTensor<float> wordTile = wordTileBuf_.Get<float>();
        LocalTensor<float> redTmp = redTmpBuf_.Get<float>();

        Duplicate(yhat, cent_[0], HEAD_DIM);
        Duplicate(idxF, 0.0f, HEAD_DIM);
        for (int32_t k = 1; k < nCent_; ++k) {
            // ind = (y > bound) ? 1 : 0, exact via overflow-to-inf scaling
            Adds(ind, y, -bound_[k - 1], HEAD_DIM);
            Maxs(ind, ind, 0.0f, HEAD_DIM);
            Muls(ind, ind, IND_SCALE, HEAD_DIM);
            Muls(ind, ind, IND_SCALE, HEAD_DIM);
            Muls(ind, ind, IND_SCALE, HEAD_DIM);
            Mins(ind, ind, 1.0f, HEAD_DIM);
            Add(idxF, idxF, ind, HEAD_DIM);
            Axpy(yhat, ind, delta_[k - 1], HEAD_DIM);
        }
        Sub(rTile[v * HEAD_DIM], y, yhat, HEAD_DIM);
        Mul(sq, rTile[v * HEAD_DIM], rTile[v * HEAD_DIM], HEAD_DIM);
        ReduceSum(rnVec[v], sq, redTmp, HEAD_DIM);
        // idx bit-pack words: sum over 8-lane groups of idx * 2^(bits*lane)
        Mul(p, idxF, wt, HEAD_DIM);
        ReduceSum(wordTile[v * WORDS_PER_VEC], p, redTmp, 8, WORDS_PER_VEC, 1);
    }

    __aicore__ inline void UnpackIdx(int32_t nv)
    {
        LocalTensor<float> wordTile = wordTileBuf_.Get<float>();
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>();
        for (int32_t v = 0; v < nv; ++v) {
            const int32_t outBase = v * idxBytesPerVec_;
            for (int32_t g = 0; g < WORDS_PER_VEC; ++g) {
                const uint32_t word = static_cast<uint32_t>(wordTile.GetValue(v * WORDS_PER_VEC + g));
                for (int32_t b = 0; b < mseBits_; ++b) {
                    idxStage.SetValue(outBase + g * mseBits_ + b, static_cast<uint8_t>(word >> (8 * b)));
                }
            }
        }
    }

    __aicore__ inline void Phase2()
    {
        LocalTensor<float> rhatIn = zTileBuf_.Get<float>();   // reuse: r_hat tile input
        LocalTensor<float> b8 = b8Buf_.Get<float>();
        LocalTensor<float> p = pBuf_.Get<float>();
        LocalTensor<float> ind = indBuf_.Get<float>();
        LocalTensor<float> y = yBuf_.Get<float>();            // reuse as weighted scratch
        LocalTensor<float> w1 = w1Buf_.Get<float>();
        LocalTensor<float> wordTile = wordTileBuf_.Get<float>();
        LocalTensor<float> redTmp = redTmpBuf_.Get<float>();
        LocalTensor<float> mat = matBuf_.Get<float>();
        const int32_t qjlWords = qjlDim_ / 8;

        for (int64_t tileStart = 0; tileStart < vecNum_; tileStart += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((vecNum_ - tileStart) < TILE_VEC ? (vecNum_ - tileStart) : TILE_VEC);
            const int64_t g0 = vecBegin_ + tileStart;

            SyncPipe(HardEvent::V_MTE2);
            DataCopy(rhatIn, wsGm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe(HardEvent::MTE2_V);

            for (int32_t v = 0; v < nv; ++v) {
                Replicate8(rhatIn[v * HEAD_DIM]);
                MatVec(p, mat, b8, qjlDim_);
                // sign = (p >= 0) ? 1 : 0  ==  1 - (p < 0 indicator)
                Muls(ind, p, -1.0f, qjlDim_);
                Maxs(ind, ind, 0.0f, qjlDim_);
                Muls(ind, ind, IND_SCALE, qjlDim_);
                Muls(ind, ind, IND_SCALE, qjlDim_);
                Muls(ind, ind, IND_SCALE, qjlDim_);
                Mins(ind, ind, 1.0f, qjlDim_);
                Adds(ind, ind, -1.0f, qjlDim_);
                Muls(ind, ind, -1.0f, qjlDim_);
                // qjl bit-pack words: sum over 8-lane groups of sign * 2^lane
                Mul(y, ind, w1, qjlDim_);
                ReduceSum(wordTile[v * WORDS_PER_VEC], y, redTmp, 8, qjlWords, 1);
            }

            SyncPipe(HardEvent::V_S);
            UnpackQjl(nv, qjlWords);
            SyncPipe(HardEvent::S_V);
            SyncPipe(HardEvent::V_MTE3);

            DataCopyExtParams qjlParams{static_cast<uint16_t>(1), static_cast<uint32_t>(nv * qjlBytesPerVec_),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
            DataCopyPad(qjlOutGm_[g0 * qjlBytesPerVec_], qjlStageBuf_.Get<uint8_t>(), qjlParams);
            SyncPipe(HardEvent::MTE3_V);
        }
    }

    __aicore__ inline void UnpackQjl(int32_t nv, int32_t qjlWords)
    {
        LocalTensor<float> wordTile = wordTileBuf_.Get<float>();
        LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();
        for (int32_t v = 0; v < nv; ++v) {
            const int32_t outBase = v * qjlBytesPerVec_;
            for (int32_t g = 0; g < qjlWords; ++g) {
                const uint32_t word = static_cast<uint32_t>(wordTile.GetValue(v * WORDS_PER_VEC + g));
                qjlStage.SetValue(outBase + g, static_cast<uint8_t>(word));
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> matBuf_;
    TBuf<QuePosition::VECCALC> mBuf_;
    TBuf<QuePosition::VECCALC> zTileBuf_;
    TBuf<QuePosition::VECCALC> rTileBuf_;
    TBuf<QuePosition::VECCALC> b8Buf_;
    TBuf<QuePosition::VECCALC> xTileBuf_;
    TBuf<QuePosition::VECCALC> xfBuf_;
    TBuf<QuePosition::VECCALC> sqBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
    TBuf<QuePosition::VECCALC> t0Buf_;
    TBuf<QuePosition::VECCALC> t1Buf_;
    TBuf<QuePosition::VECCALC> indBuf_;
    TBuf<QuePosition::VECCALC> yhatBuf_;
    TBuf<QuePosition::VECCALC> idxFBuf_;
    TBuf<QuePosition::VECCALC> pBuf_;
    TBuf<QuePosition::VECCALC> wtBuf_;
    TBuf<QuePosition::VECCALC> w1Buf_;
    TBuf<QuePosition::VECCALC> wordTileBuf_;
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
    GlobalTensor<float> wsGm_;

    int64_t totalVec_ = 0;
    int64_t qjlDim_ = 0;
    int32_t mseBits_ = 3;
    int32_t nCent_ = 8;
    int32_t idxBytesPerVec_ = 48;
    int32_t qjlBytesPerVec_ = 16;
    uint32_t coreNum_ = 1;
    int64_t vecNum_ = 0;
    int64_t vecBegin_ = 0;

    float cent_[16] = {0.0f};
    float bound_[15] = {0.0f};
    float delta_[15] = {0.0f};
    float inv_[TILE_VEC] = {0.0f};
    float invR_[TILE_VEC] = {0.0f};
};

#endif // KV_CACHE_TURBO_QUANT_KERNEL_H_
