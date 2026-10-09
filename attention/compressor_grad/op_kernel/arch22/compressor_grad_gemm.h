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
 * \file compressor_grad_gemm.h
 * \brief CompressorGrad arch22 (A3 / ascend910_93, DAV_2201) AIC (cube) kernel:
 *        Phase2 gemm.
 *
 * Layout note (R5): both B operands load natural (K, N) tiles straight from
 * GM — GM->L1 Nd2Nz, then L1->L0B via the official load3dv2 + SetFmatrix
 * path (LoadData3DParamsV2Pro, fMatrixCtrl) — with no pre-transposed
 * workspace anywhere. The dw A operand (dn^T) uses LoadDataWithTranspose,
 * whose content transpose + m-minor placement matches L0A ZZ.
 * R6 Step 1: the dw B operand (x) now arrives pre-tiled from the cast kernel
 * in blocked-NZ layout — (H/128) strips of T_pad x 128 tiles, tile (kb, nb)
 * at GM index (nb*kTiles + kb), each 128x128 tile contiguous in exactly the
 * former Nd2Nz L1 image — so the GM->L1 hop is one contiguous 32KB burst
 * (plain DataCopy) instead of 128 strided 256B rows; the L1->L0B load3dv2
 * path is untouched.
 * R7 Step 1: the dx B operands (wkv/wgate) arrive in the same blocked-NZ
 * layout — (H/128) strips of C x 128 tiles, tile (kb, nb) at GM index
 * (nb*kTilesC + kb) — so the dx B GM->L1 hop is likewise one contiguous
 * 32KB burst per tile.
 * R8 (PX, synced from the standalone A3 project): dx branch two-level
 * pipelining — DxTerm takes the split-stage form (all of a term's L1->L0
 * copies issued back-to-back on MTE1, waits only at the consume side before
 * the Mmads), and the per-pair DxPair loop is flattened into one term stream
 * with a distance-2 GM->L1 prefetch that crosses pair boundaries (the next
 * pair's first tiles load while the current pair's last terms and FixOut
 * drain play out). DxPair is gone. Per-queue stream order, Mmad order per
 * accumulator, and FixOut order are identical to the per-pair loop, so the
 * dx results stay bitwise vs R7.
 *   d_x (T, H) = dn_kv @ wkv + dn_sc @ wgate
 *     M=T, N=H, K=C:  A = dn_kv (M,K) direct;  B = wkv/wgate blocked-NZ (R7)
 *   d_wkv (C, H) = dn_kv^T(C,T) @ x(T,H)     [written to the final output]
 *     M=C, N=H, K=T_pad:  A = dn_kv^T via transposed L0A load of natural
 *     (T,C) dn tiles;  B = x blocked-NZ tiles (R6). d_wgate likewise.
 *
 * fp16 and bf16 inputs (any ioT): dn_kv/dn_sc are HiLo dual planes (dn = hi
 * + lo, both fp16); every dn-side k-tile accumulates two Mmad steps (hi then
 * lo) into the same L0C accumulator, giving ~fp32-level operand precision on
 * the fp16 cube. (R9: bf16 switched from single plane to the same dual-plane
 * scheme — the ATK wide-range accuracy bar needs the dn residual term.)
 * dn is consumed in its natural (T_pad, C) layout (srcDValue = C <= 65535
 * always), so the R2 dnT panel mechanism is gone together with the dn
 * transpose kernels.
 *
 * R2 scheduling/dataflow (vs R1):
 * - Work unit is a tile PAIR, and units are handed to cores as contiguous
 *   balanced ranges (base = U/cores, first U%cores cores get one extra), so
 *   per-core unit counts differ by at most 1. R1's ceil()-partition left a
 *   long tail (e.g. dw nHm=56 over 12 cores -> 5 vs 1 m-tiles, ~80% idle).
 * - dx unit = n-pair {(m0,n0), (m0,n0+G_BN)}: the dn_kv/dn_sc A tiles load
 *   to L1 once per k-step and feed BOTH tiles' Mmads (one L0A load feeds
 *   two Mmads too); w B tiles stream per tile. L1 loads per k-step are 8
 *   with HiLo (each w L0B tile also feeds the hi and lo Mmad of its output
 *   tile) — both dtypes run HiLo since R9.
 * - dw unit = {(m0,n0)} x {dWkv, dWgate}: symmetric flip of the above — the
 *   x B tile (blocked-NZ 32KB burst, R6) is shared by the cW and cG Mmads of
 *   each k-step; dn A tiles (transposed L0A load) stream. L1 loads per k-step
 *   are 5 with HiLo (both dtypes since R9).
 * - L1 queues are depth 4 (A1 128KB + B1 128KB of 512KB) so one k-step's
 *   full tile set is in flight while the previous step drains.
 *
 * Kernel logic is ported verbatim from the verified standalone A3 AscendC
 * implementation; only the entry form changed (Init reads the tiling struct,
 * workspace GM pointers are sliced from the user workspace by tiling offsets).
 */

#ifndef COMPRESSOR_GRAD_GEMM_H
#define COMPRESSOR_GRAD_GEMM_H

#include "kernel_operator.h"
#include "compressor_grad_common_arch22.h"
#include "compressor_grad_tiling_data.h"

using namespace AscendC;

// GM(ND) -> L1(Nz), matrix (m, n) row-major with leading dim ld
template <typename T>
__aicore__ inline void LoadNdGmToNzL1(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t m, uint32_t n,
                                      uint32_t ld)
{
    Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = m;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    DataCopy(dst, src, params);
}

// L1(Nz) -> L0A(Zz): A tile (m, k), non-transposed (A = src)
template <typename T>
__aicore__ inline void LoadNzL1ToZzL0A(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t m, uint32_t k,
                                       uint32_t colC0Stride)
{
    LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = k;
    params.kExtension = k;
    params.mExtension = m;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    LoadData(dst, src, params);
}

// L1(Nz of matrix (k, m)) -> L0A(Zz of (m, k)) via per-fractal 16x16 transpose
// (b16 types): fractal content transposes; fractal positions stay, because the
// L1 NZ fractal id (m-blk-major: mblk*(k/16)+kblk) equals the L0A ZZ fractal id.
// src NZ layout: m-direction C0(16) blocks; block j holds (k, 16) row-major at j*k*16.
// dst ZZ layout: k-direction C0(16) blocks;  block f holds (m, 16) row-major at f*m*16.
template <typename T>
__aicore__ inline void LoadNzL1ToZzL0ATranspose(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t m,
                                                uint32_t k)
{
    constexpr uint32_t FRAC = 16;
    LoadData2dTransposeParams params;
    params.startIndex = 0;
    params.repeatTimes = k / FRAC;
    params.srcStride = 1;
    params.dstGap = 0;
    params.dstFracGap = 0;
    params.addrMode = false;
    for (uint32_t j = 0; j < m / FRAC; j++) {
        LoadDataWithTranspose(dst[j * k * FRAC], src[j * k * FRAC], params);
    }
}

// L1 NZ of a natural (K, N) B tile -> L0B ZN via the official load3dv2 +
// fmatrix path (adv_api load_to_l0b_loadInstr.h LoadDataToL0, isBTranspose=false):
// whole-tile load (offsets 0), channelSize = CeilAlign(N,16),
// extConfig = (Koff<<48)|(Noff<<32)|(CeilAlign(K,16)<<16)|CeilAlign(N,16).
// Needs one SetFmatrix(1, CeilAlign(K,16), pads, FMATRIX_RIGHT) up front
// (= Load3DSetFMatrixBCal / set_fmatrix_b, the B-operand fmatrix register).
// bk/bn are compile-time 128 here, already 16-aligned -> no padding.
template <typename T>
__aicore__ inline void LoadNzL1ToZnL0BNatural(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t bk,
                                              uint32_t bn)
{
    LoadData3DParamsV2Pro params;
    params.channelSize = bn;
    params.extConfig = (static_cast<uint64_t>(bk) << 16) | bn;
    params.fMatrixCtrl = true;
    LoadData<T>(dst, src, params);
}

// Fixpipe on-the-fly cast mode for L0C(fp32) -> GM(T)
template <typename T>
__aicore__ inline QuantMode_t FixQuantPre()
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return QuantMode_t::F322BF16;
    } else {
        return QuantMode_t::F322F16;
    }
}

constexpr uint32_t CG_GEMM_BM = 128;
constexpr uint32_t CG_GEMM_BN = 128;
constexpr uint32_t CG_GEMM_BK = 128;

// L1 A1/B1 queue depths: one full k-step tile set in flight (dx needs
// 4 A / 4 B, dw needs 4 A / 1 B) plus lookahead into the next step.
constexpr uint32_t CG_L1_A_DEPTH = 4;
constexpr uint32_t CG_L1_B_DEPTH = 4;

template <typename ioT>
class CompressorGradGemmKernel {
public:
    __aicore__ inline CompressorGradGemmKernel() {}

    __aicore__ inline void Init(GM_ADDR dX, GM_ADDR dWkv, GM_ADDR dWgate, GM_ADDR userWs,
                                const optiling::CompressorGradTilingDataA3* tilingData, TPipe* pipe)
    {
        xF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->xF16WsOff));
        wkvF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wkvF16WsOff));
        wgateF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wgateF16WsOff));
        dnKvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnKvWsOff));
        dnScGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnScWsOff));
        dXGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dX));
        dWkvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dWkv));
        dWgateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dWgate));
        tLen_ = static_cast<uint32_t>(tilingData->tLen);
        hSize_ = static_cast<uint32_t>(tilingData->hiddenSize);
        cSize_ = static_cast<uint32_t>(tilingData->totalHeadDim);
        tPad_ = static_cast<uint32_t>(tilingData->tPad);
        gDx_ = static_cast<uint32_t>(tilingData->gemmDxCores);
        // dn is stored as HiLo dual planes (dn = hi + lo, plane stride
        // T_pad*C) for both fp16 and bf16 inputs (R9: always on).
        hilo_ = true;
        dnPlaneElems_ = static_cast<uint64_t>(tPad_) * cSize_;
        pipe->InitBuffer(queA1_, CG_L1_A_DEPTH, CG_GEMM_BM * CG_GEMM_BK * sizeof(half));
        pipe->InitBuffer(queB1_, CG_L1_B_DEPTH, CG_GEMM_BK * CG_GEMM_BN * sizeof(half));
        // L0 double buffering: 2x32KB fills L0A/L0B (64KB each), 2x64KB fills
        // L0C (128KB); the pair unit keeps both L0C slots alive (cA/cB resp.
        // cW/cG) across the k-loop.
        pipe->InitBuffer(queL0A_, 2, CG_GEMM_BM * CG_GEMM_BK * sizeof(half));
        pipe->InitBuffer(queL0B_, 2, CG_GEMM_BK * CG_GEMM_BN * sizeof(half));
        pipe->InitBuffer(queL0C_, 2, CG_GEMM_BM * CG_GEMM_BN * sizeof(float));
        // One-time fmatrix setup for the natural-B load3dv2 path (official
        // Prepare semantic; constant params: K=N=128, zero padding).
        const uint8_t padList[4] = {0, 0, 0, 0};
        SetFmatrix(1, CG_GEMM_BK, padList, FmatrixMode::FMATRIX_RIGHT);
    }

    __aicore__ inline void Process()
    {
        // KERNEL_TYPE_MIX_AIC_1_2: blockDim counts AIC cores; on the AIC side
        // GetBlockIdx()/GetBlockNum() already index the AIC space.
        uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
        uint32_t aicTotal = static_cast<uint32_t>(GetBlockNum());
        if (cid < gDx_) {
            ProcessDx(cid);
        } else if (cid < aicTotal) {
            ProcessDw(cid - gDx_, aicTotal - gDx_);
        }
    }

private:
    // Contiguous balanced partition of `total` units over `n` indices:
    // index i takes [start, start+count) with counts differing by <= 1.
    __aicore__ inline void BalancedRange(uint32_t idx, uint32_t n, uint32_t total, uint32_t& start,
                                         uint32_t& count) const
    {
        uint32_t base = total / n;
        uint32_t rem = total % n;
        start = idx * base + MinU32(idx, rem);
        count = base + (idx < rem ? 1U : 0U);
    }

    // DeQue the head of queA1_ and copy it to an L0A ZZ tile (non-transposed);
    // the L1 slot is freed once the MTE1 copy is queued.
    __aicore__ inline LocalTensor<half> DeQueAToL0A()
    {
        LocalTensor<half> aL1 = queA1_.DeQue<half>();
        LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
        LoadNzL1ToZzL0A(aL0, aL1, CG_GEMM_BM, CG_GEMM_BK, CG_GEMM_BM);
        queL0A_.EnQue(aL0);
        queA1_.FreeTensor(aL1);
        return queL0A_.DeQue<half>();
    }

    // Same for the transposed x tile (NZ of (K,M)) -> L0A ZZ of (M,K).
    __aicore__ inline LocalTensor<half> DeQueAToL0ATranspose()
    {
        LocalTensor<half> aL1 = queA1_.DeQue<half>();
        LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
        LoadNzL1ToZzL0ATranspose(aL0, aL1, CG_GEMM_BM, CG_GEMM_BK);
        queL0A_.EnQue(aL0);
        queA1_.FreeTensor(aL1);
        return queL0A_.DeQue<half>();
    }

    // DeQue the head of queB1_ and copy it to an L0B ZN tile via the
    // natural-(K,N) load3dv2+fmatrix path (dx w tiles and dw x tiles).
    __aicore__ inline LocalTensor<half> DeQueBToL0Natural()
    {
        LocalTensor<half> bL1 = queB1_.DeQue<half>();
        LocalTensor<half> bL0 = queL0B_.AllocTensor<half>();
        LoadNzL1ToZnL0BNatural(bL0, bL1, CG_GEMM_BK, CG_GEMM_BN);
        queL0B_.EnQue(bL0);
        queB1_.FreeTensor(bL1);
        return queL0B_.DeQue<half>();
    }

    __aicore__ inline void DoMmad(LocalTensor<float>& cL0, const LocalTensor<half>& aL0, const LocalTensor<half>& bL0,
                                  bool init)
    {
        MmadParams mp;
        mp.m = CG_GEMM_BM;
        mp.n = CG_GEMM_BN;
        mp.k = CG_GEMM_BK;
        mp.cmatrixInitVal = init;
        mp.cmatrixSource = false;
        // DeQue on queL0A_/queL0B_ already waits MTE1_M; and per the official
        // Mmad doc, back-to-back K-accumulation into the same L0C tile needs
        // no PipeBarrier<PIPE_M> when (m/16)*(n/16) >= 10 (here 8*8=64) —
        // the cube scheduler handles the L0C write-read dependency.
        Mmad(cL0, aL0, bL0, mp);
    }

    __aicore__ inline void FixOut(const GlobalTensor<ioT>& dst, uint64_t dstOff, LocalTensor<float>& cL0,
                                  uint32_t mValid, uint32_t rowStride)
    {
        SetWaitFlag<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;
        fp.mSize = mValid;
        fp.nSize = CG_GEMM_BN;
        fp.srcStride = CG_GEMM_BM;
        fp.dstStride = rowStride;
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        fp.quantPre = FixQuantPre<ioT>();
        Fixpipe(dst[dstOff], cL0, fp);
        SetWaitFlag<HardEvent::FIX_MTE2>();
    }

    // One dx GEMM term (kv or sc) of one k-step, S3-PI split-stage form:
    // stage 1 issues all of the term's L1->L0 copies back-to-back (A, B0, B1
    // pipeline on MTE1 without intervening waits); stage 2 then takes the
    // MTE1_M waits at the consume side before the Mmads — vs R7's per-tile
    // issue->wait serialization. Queue pop order and Mmad order per
    // accumulator are identical to R7 (bitwise preserved).
    __aicore__ inline void DxTerm(LocalTensor<float>& cA, LocalTensor<float>& cB, bool hasB, bool init)
    {
        // ---- stage 1: L1 -> L0 issue
        LocalTensor<half> aL1 = queA1_.DeQue<half>();
        LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
        LoadNzL1ToZzL0A(aL0, aL1, CG_GEMM_BM, CG_GEMM_BK, CG_GEMM_BM);
        queL0A_.EnQue(aL0);
        queA1_.FreeTensor(aL1);
        LocalTensor<half> b0L1 = queB1_.DeQue<half>();
        LocalTensor<half> b0L0 = queL0B_.AllocTensor<half>();
        LoadNzL1ToZnL0BNatural(b0L0, b0L1, CG_GEMM_BK, CG_GEMM_BN);
        queL0B_.EnQue(b0L0);
        queB1_.FreeTensor(b0L1);
        LocalTensor<half> b1L1, b1L0;
        if (hasB) {
            b1L1 = queB1_.DeQue<half>();
            b1L0 = queL0B_.AllocTensor<half>();
            LoadNzL1ToZnL0BNatural(b1L0, b1L1, CG_GEMM_BK, CG_GEMM_BN);
            queL0B_.EnQue(b1L0);
            queB1_.FreeTensor(b1L1);
        }
        // ---- stage 2: wait + Mmad (kv hi[, lo] order unchanged)
        LocalTensor<half> aL0d = queL0A_.DeQue<half>();
        LocalTensor<half> b0L0d = queL0B_.DeQue<half>();
        LocalTensor<half> b1L0d;
        DoMmad(cA, aL0d, b0L0d, init);
        if (hasB) {
            b1L0d = queL0B_.DeQue<half>();
            DoMmad(cB, aL0d, b1L0d, init);
        }
        queL0A_.FreeTensor(aL0d);
        if (hilo_) {
            LocalTensor<half> aL1Lo = queA1_.DeQue<half>();
            LocalTensor<half> aL0Lo = queL0A_.AllocTensor<half>();
            LoadNzL1ToZzL0A(aL0Lo, aL1Lo, CG_GEMM_BM, CG_GEMM_BK, CG_GEMM_BM);
            queL0A_.EnQue(aL0Lo);
            queA1_.FreeTensor(aL1Lo);
            LocalTensor<half> aL0LoD = queL0A_.DeQue<half>();
            DoMmad(cA, aL0LoD, b0L0d, false);
            if (hasB) {
                DoMmad(cB, aL0LoD, b1L0d, false);
            }
            queL0A_.FreeTensor(aL0LoD);
        }
        queL0B_.FreeTensor(b0L0d);
        if (hasB) {
            queL0B_.FreeTensor(b1L0d);
        }
    }

    // Issue the GM->L1 loads of one dx term (kv or sc) of k-step k.
    // Per-queue stream order is identical to the R7 per-step enqueue
    // (A: kv hi[,lo], sc hi[,lo]; B: b0w, b1w, b0g, b1g), so the DeQue side
    // (DxTerm kv then sc) is unchanged and bitwise semantics are preserved.
    __aicore__ inline void DxEnqueueTerm(uint32_t m0, uint32_t n0, uint32_t k, bool isSc, bool hasB, uint32_t kTiles)
    {
        uint64_t aOff = static_cast<uint64_t>(m0) * cSize_ + k * CG_GEMM_BK;
        const GlobalTensor<half>& dn = isSc ? dnScGM_ : dnKvGM_;
        LocalTensor<half> a = queA1_.AllocTensor<half>();
        LoadNdGmToNzL1(a, dn[aOff], CG_GEMM_BM, CG_GEMM_BK, cSize_);
        queA1_.EnQue(a);
        if (hilo_) {
            LocalTensor<half> aLo = queA1_.AllocTensor<half>();
            LoadNdGmToNzL1(aLo, dn[aOff + dnPlaneElems_], CG_GEMM_BM, CG_GEMM_BK, cSize_);
            queA1_.EnQue(aLo);
        }
        // wkv/wgate are blocked-NZ from the cast kernel (R7): tile
        // (k, n0/128) is one contiguous 32KB burst at strip-major index
        // nb*kTilesC + k (kTilesC = C/128 = kTiles here).
        const GlobalTensor<half>& w = isSc ? wgateF16GM_ : wkvF16GM_;
        uint64_t b0Tile = (static_cast<uint64_t>(n0) / CG_GEMM_BN) * kTiles + k;
        LocalTensor<half> b0w = queB1_.AllocTensor<half>();
        DataCopy(b0w, w[b0Tile * (CG_GEMM_BK * CG_GEMM_BN)], CG_GEMM_BK * CG_GEMM_BN);
        queB1_.EnQue(b0w);
        if (hasB) {
            LocalTensor<half> b1w = queB1_.AllocTensor<half>();
            DataCopy(b1w, w[(b0Tile + kTiles) * (CG_GEMM_BK * CG_GEMM_BN)], CG_GEMM_BK * CG_GEMM_BN);
            queB1_.EnQue(b1w);
        }
    }

    // branch B unit: d_wkv and d_wgate tiles (m0,n0) of
    // d_w* (C, H) = dn_*(T,C)^T @ x(T,H) — M=C, N=H, K=T_pad, FixOut direct
    // to the final (C, H) outputs (R5 output swap; no transposed workspace).
    // The x B tile at (k0,n0) (blocked-NZ 32KB burst, R6) is shared by the
    // cW and cG Mmads of each k-step; dn A tiles (transposed L0A load) stream
    // per term.
    __aicore__ inline void DwPair(uint32_t m0, uint32_t n0)
    {
        uint32_t kTiles = CeilDivU32(tPad_, CG_GEMM_BK);
        LocalTensor<float> cW = queL0C_.AllocTensor<float>();
        LocalTensor<float> cG = queL0C_.AllocTensor<float>();
        for (uint32_t k = 0; k < kTiles; k++) {
            uint32_t k0 = k * CG_GEMM_BK;
            bool init = (k == 0);
            // ---- GM -> L1: dnKv/dnSc A tiles (natural (T,C) tiles at
            // (k0,m0), transposed on the way to L0A), then shared x B tile
            uint64_t aOff = static_cast<uint64_t>(k0) * cSize_ + m0;
            LocalTensor<half> aW = queA1_.AllocTensor<half>();
            LoadNdGmToNzL1(aW, dnKvGM_[aOff], CG_GEMM_BK, CG_GEMM_BM, cSize_);
            queA1_.EnQue(aW);
            if (hilo_) {
                LocalTensor<half> aWLo = queA1_.AllocTensor<half>();
                LoadNdGmToNzL1(aWLo, dnKvGM_[aOff + dnPlaneElems_], CG_GEMM_BK, CG_GEMM_BM, cSize_);
                queA1_.EnQue(aWLo);
            }
            LocalTensor<half> aG = queA1_.AllocTensor<half>();
            LoadNdGmToNzL1(aG, dnScGM_[aOff], CG_GEMM_BK, CG_GEMM_BM, cSize_);
            queA1_.EnQue(aG);
            if (hilo_) {
                LocalTensor<half> aGLo = queA1_.AllocTensor<half>();
                LoadNdGmToNzL1(aGLo, dnScGM_[aOff + dnPlaneElems_], CG_GEMM_BK, CG_GEMM_BM, cSize_);
                queA1_.EnQue(aGLo);
            }
            LocalTensor<half> bX = queB1_.AllocTensor<half>();
            // x is blocked-NZ from the cast kernel: tile (k, n0/128) is one
            // contiguous 32KB burst (strip-major: nb*kTiles + k).
            uint64_t bTile = static_cast<uint64_t>(n0 / CG_GEMM_BN) * kTiles + k;
            DataCopy(bX, xF16GM_[bTile * (CG_GEMM_BK * CG_GEMM_BN)], CG_GEMM_BK * CG_GEMM_BN);
            queB1_.EnQue(bX);
            // ---- L1 -> L0 -> Mmad. x L0B tile feeds every Mmad of this step;
            // A queue pops kv hi[, kv lo, sc hi, sc lo] matching enqueue order.
            LocalTensor<half> xL0 = DeQueBToL0Natural();
            LocalTensor<half> aWL0 = DeQueAToL0ATranspose();
            DoMmad(cW, aWL0, xL0, init);
            queL0A_.FreeTensor(aWL0);
            if (hilo_) {
                LocalTensor<half> aWLoL0 = DeQueAToL0ATranspose();
                DoMmad(cW, aWLoL0, xL0, false);
                queL0A_.FreeTensor(aWLoL0);
            }
            LocalTensor<half> aGL0 = DeQueAToL0ATranspose();
            DoMmad(cG, aGL0, xL0, init);
            queL0A_.FreeTensor(aGL0);
            if (hilo_) {
                LocalTensor<half> aGLoL0 = DeQueAToL0ATranspose();
                DoMmad(cG, aGLoL0, xL0, false);
                queL0A_.FreeTensor(aGLoL0);
            }
            queL0B_.FreeTensor(xL0);
        }
        // C % 128 == 0 and H % 128 == 0 (host checks) -> full tiles,
        // mValid is always G_BM.
        queL0C_.EnQue(cW);
        cW = queL0C_.DeQue<float>();
        FixOut(dWkvGM_, static_cast<uint64_t>(m0) * hSize_ + n0, cW, CG_GEMM_BM, hSize_);
        queL0C_.FreeTensor(cW);
        queL0C_.EnQue(cG);
        cG = queL0C_.DeQue<float>();
        FixOut(dWgateGM_, static_cast<uint64_t>(m0) * hSize_ + n0, cG, CG_GEMM_BM, hSize_);
        queL0C_.FreeTensor(cG);
    }

    // branch A: d_x = dn_kv @ wkv + dn_sc @ wgate  (M=T, N=H, K=C)
    // S3-PX: the unit loop is flattened into one term stream so the
    // distance-2 load prefetch also runs across pair boundaries — the next
    // pair's first tiles are fetched while the current pair's last terms and
    // the FixOut drain play out. Per-unit accumulator rotation, Mmad order,
    // and FixOut order are identical to the per-pair loop (bitwise).
    __aicore__ inline void ProcessDx(uint32_t cid)
    {
        uint32_t nM = CeilDivU32(tLen_, CG_GEMM_BM);
        uint32_t nH = CeilDivU32(hSize_, CG_GEMM_BN);
        uint32_t nPairRow = CeilDivU32(nH, 2); // tile pairs per m-row
        uint32_t total = nM * nPairRow;
        uint32_t start, count;
        BalancedRange(cid, gDx_, total, start, count);
        if (count == 0) {
            return;
        }
        uint32_t kTiles = CeilDivU32(cSize_, CG_GEMM_BK);
        uint32_t tpu = kTiles * 2U; // terms per unit (kv + sc)
        uint32_t end = start + count;
        // enqueue cursor (eU,eT) leads the consume cursor (cU,cT) by <=2 terms
        uint32_t eU = start, eT = 0;
        for (uint32_t i = 0; i < 2U && eU < end; i++) { // prologue
            uint32_t mi = eU / nPairRow, pj = eU % nPairRow;
            DxEnqueueTerm(mi * CG_GEMM_BM, pj * 2U * CG_GEMM_BN, eT >> 1, (eT & 1U) != 0U, (pj * 2U + 1U) < nH,
                          kTiles);
            eT++;
            if (eT == tpu) {
                eT = 0;
                eU++;
            }
        }
        uint32_t cU = start, cT = 0;
        uint32_t mi = cU / nPairRow, pj = cU % nPairRow;
        uint32_t cM0 = mi * CG_GEMM_BM, cN0 = pj * 2U * CG_GEMM_BN;
        bool cHasB = (pj * 2U + 1U) < nH;
        LocalTensor<float> cA = queL0C_.AllocTensor<float>();
        LocalTensor<float> cB;
        if (cHasB) {
            cB = queL0C_.AllocTensor<float>();
        }
        while (cU < end) {
            DxTerm(cA, cB, cHasB, cT == 0);
            // refill the freed slots with the next stream term (may belong to
            // the next unit — issued before this unit's FixOut on purpose)
            if (eU < end) {
                uint32_t emi = eU / nPairRow, epj = eU % nPairRow;
                DxEnqueueTerm(emi * CG_GEMM_BM, epj * 2U * CG_GEMM_BN, eT >> 1, (eT & 1U) != 0U,
                              (epj * 2U + 1U) < nH, kTiles);
                eT++;
                if (eT == tpu) {
                    eT = 0;
                    eU++;
                }
            }
            cT++;
            if (cT == tpu) { // unit done: drain + FixOut, rotate accumulators
                cT = 0;
                uint32_t mValid = MinU32(CG_GEMM_BM, tLen_ - cM0);
                queL0C_.EnQue(cA);
                cA = queL0C_.DeQue<float>();
                FixOut(dXGM_, static_cast<uint64_t>(cM0) * hSize_ + cN0, cA, mValid, hSize_);
                queL0C_.FreeTensor(cA);
                if (cHasB) {
                    queL0C_.EnQue(cB);
                    cB = queL0C_.DeQue<float>();
                    FixOut(dXGM_, static_cast<uint64_t>(cM0) * hSize_ + cN0 + CG_GEMM_BN, cB, mValid, hSize_);
                    queL0C_.FreeTensor(cB);
                }
                cU++;
                if (cU < end) {
                    mi = cU / nPairRow;
                    pj = cU % nPairRow;
                    cM0 = mi * CG_GEMM_BM;
                    cN0 = pj * 2U * CG_GEMM_BN;
                    cHasB = (pj * 2U + 1U) < nH;
                    cA = queL0C_.AllocTensor<float>();
                    if (cHasB) {
                        cB = queL0C_.AllocTensor<float>();
                    }
                }
            }
        }
    }

    // branch B: d_wkv / d_wgate (C, H) direct — one unit = (m0,n0) x {cW,cG}
    __aicore__ inline void ProcessDw(uint32_t gid, uint32_t coreNumW)
    {
        uint32_t nCn = CeilDivU32(cSize_, CG_GEMM_BM); // M tiles over C
        uint32_t nHm = CeilDivU32(hSize_, CG_GEMM_BN); // N tiles over H
        uint32_t total = nCn * nHm;
        uint32_t start, count;
        BalancedRange(gid, coreNumW, total, start, count);
        // n-fastest unit order keeps a core on the same m0 for nHm consecutive
        // units, so its dn panel re-reads hit L2.
        for (uint32_t u = start; u < start + count; u++) {
            DwPair((u / nHm) * CG_GEMM_BM, (u % nHm) * CG_GEMM_BN);
        }
    }

    GlobalTensor<half> xF16GM_, wkvF16GM_, wgateF16GM_, dnKvGM_, dnScGM_;
    GlobalTensor<ioT> dXGM_, dWkvGM_, dWgateGM_;
    uint32_t tLen_, hSize_, cSize_, tPad_, gDx_;
    uint64_t dnPlaneElems_;
    bool hilo_;
    TQue<TPosition::A1, CG_L1_A_DEPTH> queA1_;
    TQue<TPosition::B1, CG_L1_B_DEPTH> queB1_;
    TQue<TPosition::A2, 2> queL0A_;
    TQue<TPosition::B2, 2> queL0B_;
    TQue<TPosition::CO1, 2> queL0C_;
};

#endif // COMPRESSOR_GRAD_GEMM_H
