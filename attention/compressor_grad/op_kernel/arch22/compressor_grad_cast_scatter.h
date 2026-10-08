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
 * \file compressor_grad_cast_scatter.h
 * \brief CompressorGrad arch22 (A3 / ascend910_93, DAV_2201) AIV kernels:
 *        Phase0 CompressorGradCastKernel: x/wkv/wgate -> fp16 blocked-NZ workspace;
 *               dn_kv/dn_sc/d_ape zero-fill.
 *        Phase1 CompressorGradScatterKernel: per-block softmax backward +
 *               gradient routing + d_ape accumulate.
 *
 * fp16 inputs use HiLo dual-plane dn workspaces: scatter writes hi = fp16(dn)
 * to plane 0 and lo = fp16(dn - fp32(hi)) to plane 1 (plane stride T_pad*C);
 * the gemm kernel accumulates dot(hi, w) + dot(lo, w) so the dn operand keeps
 * ~fp32-level precision through the fp16 cube path. bf16 keeps single plane.
 *
 * Kernel logic is ported verbatim from the verified standalone A3 AscendC
 * implementation; only the entry form changed (Init reads the tiling struct,
 * workspace GM pointers are sliced from the user workspace by tiling offsets,
 * and cid >= usedCores guards replace the 3-launch blockDim clamps).
 */

#ifndef COMPRESSOR_GRAD_CAST_SCATTER_H
#define COMPRESSOR_GRAD_CAST_SCATTER_H

#include "kernel_operator.h"
#include "compressor_grad_common_arch22.h"
#include "compressor_grad_tiling_data.h"

using namespace AscendC;

// =========================================================================
// Phase 0 (AIV): cast + zero fill
//   - x(T,H) ioT -> x_f16 f16 (pad rows -> 0), blocked-NZ layout (R6):
//     (H/128) column strips of T_pad x 128 tiles; tile (kb, nb) sits at GM
//     tile index (nb*kTiles + kb), 128*128 elements contiguous, in-tile
//     arrangement (c/16)*2048 + r*16 + (c%16) — byte-identical to the gemm's
//     former Nd2Nz(nValue=128, dValue=128, dstNzC0Stride=128, dstNzNStride=1)
//     L1 image, so the dw B side loads one contiguous 32KB burst per tile.
//   - wkv/wgate(C,H) ioT -> f16, blocked-NZ layout (R7): (H/128) column
//     strips of C x 128 tiles, tile (kb, nb) at GM tile index
//     (nb*(C/128) + kb) within each tensor — the same in-tile arrangement
//     as x — so the dx B side also loads one contiguous 32KB burst per tile.
//     Both tensors are tiled by all cores in one shared index space.
//   - dn_kv/dn_sc(T_pad,C) f16 zero-fill; d_ape(cr,C) f32 zero-fill
// =========================================================================
constexpr uint32_t CG_CAST_VEC = 2048; // elements per vector chunk
constexpr uint32_t CG_X_TILE = 128;    // blocked-NZ tile edge (rows and cols)

template <typename ioT>
class CompressorGradCastKernel {
public:
    __aicore__ inline CompressorGradCastKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wkv, GM_ADDR wgate, GM_ADDR dApe, GM_ADDR userWs,
                                const optiling::CompressorGradTilingDataA3* tilingData, TPipe* pipe)
    {
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(x));
        wkvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(wkv));
        wgateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(wgate));
        xF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->xF16WsOff));
        wkvF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wkvF16WsOff));
        wgateF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wgateF16WsOff));
        dnKvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnKvWsOff));
        dnScGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnScWsOff));
        dApeGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dApe));
        tLen_ = static_cast<uint32_t>(tilingData->tLen);
        hSize_ = static_cast<uint32_t>(tilingData->hiddenSize);
        cSize_ = static_cast<uint32_t>(tilingData->totalHeadDim);
        crSize_ = static_cast<uint32_t>(tilingData->cmpRatio);
        tPad_ = static_cast<uint32_t>(tilingData->tPad);
        coreNum_ = static_cast<uint32_t>(tilingData->castCoreNum);
        pipe->InitBuffer(inBuf_, CG_X_TILE * CG_X_TILE * sizeof(ioT));
        pipe->InitBuffer(f32Buf_, CG_X_TILE * CG_X_TILE * sizeof(float));
        pipe->InitBuffer(f16Buf_, CG_X_TILE * CG_X_TILE * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        uint32_t cid = GetBlockIdx();
        if (cid >= coreNum_) {
            return;
        }
        CastXNz(cid);
        CastWeightNz(cid);
        ZeroDn(cid);
        if (cid == 0) {
            ZeroApe();
        }
    }

private:
    // Stage one 128x128 ioT tile (strided 256B rows from the natural x),
    // cast ioT -> f32 (full-tile contiguous, same rounding as the old row
    // path) -> f16 with per-16-column-group strided repeats that place each
    // group at its NZ slot, then flush the tile as one 32KB contiguous write.
    __aicore__ inline void CastXNz(uint32_t cid)
    {
        uint32_t kTiles = tPad_ / CG_X_TILE; // tPad % 128 == 0 (host pads)
        uint32_t nHm = hSize_ / CG_X_TILE;   // H % 128 == 0 (host checks)
        uint32_t totalTiles = kTiles * nHm;
        uint32_t tilesPerCore = CeilDivU32(totalTiles, coreNum_);
        uint32_t t0 = cid * tilesPerCore;
        uint32_t t1 = MinU32(t0 + tilesPerCore, totalTiles);
        LocalTensor<ioT> inLocal = inBuf_.Get<ioT>();
        LocalTensor<float> f32Local = f32Buf_.Get<float>();
        LocalTensor<half> nzLocal = f16Buf_.Get<half>();
        for (uint32_t t = t0; t < t1; t++) {
            uint32_t nb = t / kTiles;
            uint32_t kb = t - nb * kTiles;
            uint32_t rowBase = kb * CG_X_TILE;
            uint32_t validRows = (rowBase < tLen_) ? MinU32(CG_X_TILE, tLen_ - rowBase) : 0;
            SetWaitFlag<HardEvent::V_MTE2>(); // prior tile's casts done reading inLocal
            if (validRows < CG_X_TILE) {
                Duplicate(inLocal, static_cast<ioT>(0), static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
                SetWaitFlag<HardEvent::V_MTE2>();
            }
            if (validRows > 0) {
                DataCopyExtParams cp{static_cast<uint16_t>(validRows), static_cast<uint32_t>(CG_X_TILE * sizeof(ioT)),
                                     static_cast<uint32_t>((hSize_ - CG_X_TILE) * sizeof(ioT)), 0, 0};
                DataCopyPadExtParams<ioT> pp{false, 0, 0, static_cast<ioT>(0)};
                DataCopyPad(inLocal, xGM_[static_cast<uint64_t>(rowBase) * hSize_ + nb * CG_X_TILE], cp, pp);
            }
            SetWaitFlag<HardEvent::MTE2_V>();
            SetWaitFlag<HardEvent::MTE3_V>(); // prior tile's GM write of nzLocal drained
            Cast(f32Local, inLocal, RoundMode::CAST_NONE, static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
            // f32 tile (row-major 128x128) -> NZ slots: group g (cols 16g..)
            // lands at g*2048 with its 128 rows packed contiguously.
            UnaryRepeatParams gather{1, 1, 1, static_cast<uint8_t>(CG_X_TILE / 8)};
            for (uint32_t g = 0; g < CG_X_TILE / 16; g++) {
                Cast(nzLocal[g * CG_X_TILE * 16], f32Local[g * 16], RoundMode::CAST_NONE, 16, CG_X_TILE, gather);
            }
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopy(xF16GM_[static_cast<uint64_t>(t) * CG_X_TILE * CG_X_TILE], nzLocal,
                     static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
        }
        SetWaitFlag<HardEvent::MTE3_V>(); // drain last tile write: CastWeightNz/ZeroDn reuse f16Buf_
        SetWaitFlag<HardEvent::V_MTE2>(); // drain last tile casts: CastWeightNz reloads inBuf_
    }

    // wkv/wgate (C,H) ioT -> f16 blocked-NZ tiles (R7). Same writer pipeline
    // as CastXNz; C % 128 == 0 so tiles are always full (no pad path). One
    // tile index space covers both tensors: t < kTilesC*nHm -> wkv, else
    // wgate; tile (kb, nb) lands at in-tensor index tIn = nb*kTilesC + kb.
    __aicore__ inline void CastWeightNz(uint32_t cid)
    {
        uint32_t kTilesC = cSize_ / CG_X_TILE; // C % 128 == 0 (host checks)
        uint32_t nHm = hSize_ / CG_X_TILE;     // H % 128 == 0
        uint32_t perTensor = kTilesC * nHm;
        uint32_t totalTiles = 2U * perTensor;
        uint32_t tilesPerCore = CeilDivU32(totalTiles, coreNum_);
        uint32_t t0 = cid * tilesPerCore;
        uint32_t t1 = MinU32(t0 + tilesPerCore, totalTiles);
        LocalTensor<ioT> inLocal = inBuf_.Get<ioT>();
        LocalTensor<float> f32Local = f32Buf_.Get<float>();
        LocalTensor<half> nzLocal = f16Buf_.Get<half>();
        for (uint32_t t = t0; t < t1; t++) {
            bool gate = (t >= perTensor);
            uint32_t tIn = gate ? (t - perTensor) : t;
            uint32_t nb = tIn / kTilesC;
            uint32_t kb = tIn - nb * kTilesC;
            const GlobalTensor<ioT>& src = gate ? wgateGM_ : wkvGM_;
            const GlobalTensor<half>& dst = gate ? wgateF16GM_ : wkvF16GM_;
            SetWaitFlag<HardEvent::V_MTE2>(); // prior tile's casts done reading inLocal
            DataCopyExtParams cp{static_cast<uint16_t>(CG_X_TILE), static_cast<uint32_t>(CG_X_TILE * sizeof(ioT)),
                                 static_cast<uint32_t>((hSize_ - CG_X_TILE) * sizeof(ioT)), 0, 0};
            DataCopyPadExtParams<ioT> pp{false, 0, 0, static_cast<ioT>(0)};
            DataCopyPad(inLocal, src[static_cast<uint64_t>(kb) * CG_X_TILE * hSize_ + nb * CG_X_TILE], cp, pp);
            SetWaitFlag<HardEvent::MTE2_V>();
            SetWaitFlag<HardEvent::MTE3_V>(); // prior tile's GM write of nzLocal drained
            Cast(f32Local, inLocal, RoundMode::CAST_NONE, static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
            UnaryRepeatParams gather{1, 1, 1, static_cast<uint8_t>(CG_X_TILE / 8)};
            for (uint32_t g = 0; g < CG_X_TILE / 16; g++) {
                Cast(nzLocal[g * CG_X_TILE * 16], f32Local[g * 16], RoundMode::CAST_NONE, 16, CG_X_TILE, gather);
            }
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopy(dst[static_cast<uint64_t>(tIn) * CG_X_TILE * CG_X_TILE], nzLocal,
                     static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
        }
        SetWaitFlag<HardEvent::MTE3_V>(); // drain last tile write: ZeroDn reuses f16Buf_
        SetWaitFlag<HardEvent::V_MTE2>(); // drain last tile casts
    }

    __aicore__ inline void ZeroDn(uint32_t cid)
    {
        // fp16 inputs use HiLo dual-plane dn workspaces (dn = hi + lo); both
        // planes must start zeroed. bf16 keeps the single-plane layout.
        constexpr uint32_t planes = std::is_same_v<ioT, half> ? 2U : 1U;
        // plane element count (tPad * cSize) can exceed 2^32 on extreme
        // shapes: all tile arithmetic stays in 64-bit
        const uint64_t planeElems = static_cast<uint64_t>(tPad_) * cSize_;
        const uint64_t hTiles = (planeElems + CG_CAST_VEC - 1) / CG_CAST_VEC;
        uint64_t totalTiles = 2ULL * planes * hTiles;
        uint32_t perCore = CeilDivU32(static_cast<uint32_t>(totalTiles), coreNum_);
        LocalTensor<half> f16Local = f16Buf_.Get<half>();
        for (uint32_t v = 0; v < perCore; v++) {
            uint64_t tile = static_cast<uint64_t>(cid) * perCore + v;
            if (tile >= totalTiles) {
                break;
            }
            uint32_t idx = static_cast<uint32_t>(tile / hTiles);
            uint64_t elem = (tile % hTiles) * CG_CAST_VEC;
            uint32_t cnt = static_cast<uint32_t>(MinU64(CG_CAST_VEC, planeElems - elem));
            Duplicate(f16Local, static_cast<half>(0), cnt);
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopyExtParams outCp{1, static_cast<uint32_t>(cnt * sizeof(half)), 0, 0, 0};
            uint64_t base = static_cast<uint64_t>(idx % planes) * planeElems + elem;
            if (idx < planes) {
                DataCopyPad(dnKvGM_[base], f16Local, outCp);
            } else {
                DataCopyPad(dnScGM_[base], f16Local, outCp);
            }
            SetWaitFlag<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void ZeroApe()
    {
        LocalTensor<float> f32Local = f32Buf_.Get<float>();
        for (uint32_t off = 0; off < crSize_ * cSize_; off += CG_CAST_VEC) {
            uint32_t cnt = MinU32(CG_CAST_VEC, crSize_ * cSize_ - off);
            Duplicate(f32Local, 0.0f, cnt);
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopyExtParams outCp{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
            DataCopyPad(dApeGM_[off], f32Local, outCp);
            SetWaitFlag<HardEvent::MTE3_V>();
        }
    }

    GlobalTensor<ioT> xGM_, wkvGM_, wgateGM_;
    GlobalTensor<half> xF16GM_, wkvF16GM_, wgateF16GM_, dnKvGM_, dnScGM_;
    GlobalTensor<float> dApeGM_;
    uint32_t tLen_, hSize_, cSize_, crSize_, tPad_, coreNum_;
    TBuf<TPosition::VECCALC> inBuf_;
    TBuf<TPosition::VECCALC> f32Buf_;
    TBuf<TPosition::VECCALC> f16Buf_;
};

// =========================================================================
// Phase 1 (AIV): scatter
//   Per (block, D-segment): dK = dC*sm; dSp = dC*kv; col = sum sm*dSp;
//   dZ = sm*(dSp - col); route cur/prev slots into dn_kv/dn_sc (fp16);
//   accumulate dZ rows into per-core d_ape accumulator, atomic-add at segment end.
//   Block-vectorized: sm/kv blocks bulk-loaded once per segment; colsum via 4
//   strided MulAddDst lanes (dC factored out); dK/dZ computed as whole-block
//   broadcast-stride ops; rows staged into (cr, seg) fp16 slot tiles (cur/prev
//   split is a 2-row-strided cast) and flushed with one 2D DataCopyPad per tile.
//   Per-block sync count is O(1) instead of O(N).
// =========================================================================
template <typename ioT>
class CompressorGradScatterKernel {
public:
    __aicore__ inline CompressorGradScatterKernel() {}

    __aicore__ inline void Init(GM_ADDR dCmpKv, GM_ADDR smScore, GM_ADDR kvIn, GM_ADDR dApe, GM_ADDR userWs,
                                const optiling::CompressorGradTilingDataA3* tilingData, TPipe* pipe)
    {
        dCkGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dCmpKv));
        smGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smScore));
        kvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kvIn));
        dnKvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnKvWsOff));
        dnScGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnScWsOff));
        dApeGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dApe));
        bSize_ = static_cast<uint32_t>(tilingData->batchSize);
        sSize_ = static_cast<uint32_t>(tilingData->seqSize);
        dSize_ = static_cast<uint32_t>(tilingData->headDim);
        crSize_ = static_cast<uint32_t>(tilingData->cmpRatio);
        coff_ = static_cast<uint32_t>(tilingData->coff);
        nblk_ = static_cast<uint32_t>(tilingData->nblk);
        nblkIn_ = static_cast<uint32_t>(tilingData->nblkIn);
        nRows_ = static_cast<uint32_t>(tilingData->cmpRowCnt);
        cSize_ = static_cast<uint32_t>(tilingData->totalHeadDim);
        segLen_ = static_cast<uint32_t>(tilingData->segLen);
        blocksPerCore_ = static_cast<uint32_t>(tilingData->blocksPerCore);
        scatterCoreNum_ = static_cast<uint32_t>(tilingData->scatterCoreNum);
        dnPlaneElems_ = static_cast<uint64_t>(tilingData->tPad) * static_cast<uint64_t>(cSize_);
        uint32_t seg = segLen_;
        // fp16: HiLo dual-plane dn staging; bf16: hi plane only.
        constexpr uint32_t planes = std::is_same_v<ioT, half> ? 2U : 1U;
        // prev slot tile only exists for coff==2.
        const uint32_t slots = (coff_ == 2) ? 2U : 1U;
        // dn rows are staged per (cur/prev) slot x plane as (cr, seg) fp16 tiles
        // and flushed with one 2D DataCopyPad per tile instead of per-row copies.
        // fp32 region: ape 2*cr + sm/kv/dkv 3*N + colAcc 4 + dC32 1 [+ hBTile cr for fp16]
        uint32_t f32Elems = (2U * crSize_ + 3U * nRows_ + 5U) * seg + (planes - 1U) * crSize_ * seg;
        uint32_t bufBytes =
            f32Elems * sizeof(float) + seg * sizeof(ioT) + slots * planes * crSize_ * seg * sizeof(half);
        pipe->InitBuffer(buf_, bufBytes);
        apeCur_ = buf_.Get<float>();
        apePrev_ = buf_.Get<float>()[crSize_ * seg];
        uint32_t off = 2 * crSize_ * seg; // in float elements
        sm_ = buf_.Get<float>()[off];
        kvBlk_ = buf_.Get<float>()[off + nRows_ * seg];
        dkvBlk_ = buf_.Get<float>()[off + 2 * nRows_ * seg];
        off += 3 * nRows_ * seg;
        colAcc_ = buf_.Get<float>()[off]; // 4 x seg multi-accumulator
        dC32_ = buf_.Get<float>()[off + 4 * seg];
        off += 5 * seg;
        if constexpr (std::is_same_v<ioT, half>) {
            hBTile_ = buf_.Get<float>()[off];
            off += crSize_ * seg;
        }
        uint32_t byteOff = off * sizeof(float);
        dCIn_ = buf_.Get<uint8_t>().ReinterpretCast<ioT>()[byteOff / sizeof(ioT)];
        uint32_t halfOff = (byteOff + seg * sizeof(ioT)) / sizeof(half);
        // tiles laid out plane-major: [hi cur][hi prev?] then [lo cur][lo prev?]
        stageHiCur_ = buf_.Get<half>()[halfOff];
        stageHiPrev_ = buf_.Get<half>()[halfOff + crSize_ * seg];
        stageLoCur_ = buf_.Get<half>()[halfOff + slots * crSize_ * seg];
        stageLoPrev_ = buf_.Get<half>()[halfOff + (slots + 1) * crSize_ * seg];
    }

    __aicore__ inline void Process()
    {
        uint32_t cid = GetBlockIdx();
        if (cid >= scatterCoreNum_) {
            return;
        }
        uint32_t nbTotal = bSize_ * nblk_;
        for (uint32_t d0 = 0; d0 < dSize_; d0 += segLen_) {
            Duplicate(apeCur_, 0.0f, crSize_ * segLen_);
            Duplicate(apePrev_, 0.0f, crSize_ * segLen_);

            for (uint32_t t = 0; t < blocksPerCore_; t++) {
                uint32_t blk = cid * blocksPerCore_ + t;
                if (blk >= nbTotal) {
                    break;
                }
                uint32_t b = blk / nblk_;
                uint32_t j = blk % nblk_;
                ProcessBlockSegment(b, j, d0);
            }
            // atomic accumulate this segment's d_ape partials into GM
            uint32_t curCol = (coff_ - 1) * dSize_ + d0;
            SetAtomicAdd<float>();
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopyExtParams cpApe{static_cast<uint16_t>(crSize_), static_cast<uint32_t>(segLen_ * sizeof(float)), 0,
                                    static_cast<uint32_t>((cSize_ - segLen_) * sizeof(float)), 0};
            DataCopyPad(dApeGM_[curCol], apeCur_, cpApe);
            if (coff_ == 2) {
                DataCopyPad(dApeGM_[d0], apePrev_, cpApe);
            }
            SetWaitFlag<HardEvent::MTE3_V>();
            DisableDmaAtomic();
            SetWaitFlag<HardEvent::V_MTE2>();
        }
    }

private:
    __aicore__ inline void ProcessBlockSegment(uint32_t b, uint32_t j, uint32_t d0)
    {
        const uint32_t seg = segLen_;
        const uint32_t n = nRows_;
        const uint64_t blkBase = static_cast<uint64_t>(b) * nblkIn_ + j;
        // one fp32 row / one fp16 tile row in 32B blocks (seg in {16, 32, 64})
        const uint8_t rowBlk = static_cast<uint8_t>(seg / 8);
        // stage-in: sm + kv blocks (N, SEG) + dC row (SEG) in one MTE2 batch
        {
            DataCopyExtParams cpBlk{static_cast<uint16_t>(n), static_cast<uint32_t>(seg * sizeof(float)),
                                    static_cast<uint32_t>((dSize_ - seg) * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> ppF{true, 0, 0, 0.0f};
            DataCopyExtParams cpDc{1, static_cast<uint32_t>(seg * sizeof(ioT)), 0, 0, 0};
            DataCopyPadExtParams<ioT> ppDc{true, 0, 0, static_cast<ioT>(0)};
            SetWaitFlag<HardEvent::V_MTE2>();
            DataCopyPad(sm_, smGM_[blkBase * n * dSize_ + d0], cpBlk, ppF);
            DataCopyPad(kvBlk_, kvGM_[blkBase * n * dSize_ + d0], cpBlk, ppF);
            DataCopyPad(dCIn_, dCkGM_[blkBase * dSize_ + d0], cpDc, ppDc);
            Duplicate(colAcc_, 0.0f, 4 * seg); // overlaps the in-flight loads
            SetWaitFlag<HardEvent::MTE2_V>();
            Cast(dC32_, dCIn_, RoundMode::CAST_NONE, seg);
        }
        // colraw lanes: colAcc_[k] = sum_{r = k mod 4} sm_r (*) kv_r (fp32 fma,
        // single-rounding; dC factored out, applied once by pass 2's Mul)
        SetWaitFlag<HardEvent::MTE3_V>(); // prev block's dn_sc flush drained (staging free)
        {
            BinaryRepeatParams lane{1, 1, 1, 0, static_cast<uint8_t>(4 * rowBlk), static_cast<uint8_t>(4 * rowBlk)};
            for (uint32_t k = 0; k < 4 && k < n; k++) {
                MulAddDst(colAcc_[k * seg], sm_[k * seg], kvBlk_[k * seg], seg, (n - k + 3) >> 2, lane);
            }
        }
        // pass 1: dkv = sm (*) dC (dC broadcast per row), staged into dn_kv tiles
        {
            BinaryRepeatParams bcast{1, 1, 1, rowBlk, rowBlk, 0};
            // repeatTime is uint8_t (<=255): slab N>255 (cr=128 x coff=2 -> N=256)
            // into <=255-repeat chunks; src1RepStride=0 keeps dC broadcast from
            // its base every chunk, so chunking is value-identical.
            for (uint32_t r0 = 0; r0 < n; r0 += 255) {
                const uint8_t reps = static_cast<uint8_t>(MinU32(n - r0, 255));
                Mul(dkvBlk_[r0 * seg], sm_[r0 * seg], dC32_, seg, reps, bcast);
            }
        }
        StageTiles(dkvBlk_);
        FlushDn(dnKvGM_, b, j, d0);
        // colsum finalize + pass 2 block math overlap the dn_kv flush
        // (none of them touch the staging tiles)
        Add(colAcc_, colAcc_, colAcc_[seg], seg);
        Add(colAcc_[2 * seg], colAcc_[2 * seg], colAcc_[3 * seg], seg);
        Add(colAcc_, colAcc_, colAcc_[2 * seg], seg);
        // pass 2 (in-place on kvBlk_): dz = sm * dC * (kv - colRaw)
        {
            BinaryRepeatParams bcast{1, 1, 1, rowBlk, rowBlk, 0};
            // same uint8_t repeatTime guard as pass 1: <=255-repeat row slabs;
            // per-row independent ops, chunking keeps values bit-identical.
            for (uint32_t r0 = 0; r0 < n; r0 += 255) {
                const uint8_t reps = static_cast<uint8_t>(MinU32(n - r0, 255));
                Sub(kvBlk_[r0 * seg], kvBlk_[r0 * seg], colAcc_, seg, reps, bcast);
            }
            for (uint32_t r0 = 0; r0 < n; r0 += 255) {
                const uint8_t reps = static_cast<uint8_t>(MinU32(n - r0, 255));
                Mul(kvBlk_[r0 * seg], kvBlk_[r0 * seg], dC32_, seg, reps, bcast);
            }
            Mul(kvBlk_, sm_, kvBlk_, static_cast<int32_t>(n * seg));
        }
        // ape accumulation: cur rows -> apeCur_, prev rows -> apePrev_
        if (coff_ == 1) {
            Add(apeCur_, apeCur_, kvBlk_, static_cast<int32_t>(n * seg));
        } else {
            BinaryRepeatParams apePrm{1, 1, 1, rowBlk, rowBlk, static_cast<uint8_t>(2 * rowBlk)};
            Add(apeCur_, apeCur_, kvBlk_[seg], seg, crSize_, apePrm); // odd rows (o=1)
            if (j >= 1) {
                Add(apePrev_, apePrev_, kvBlk_, seg, crSize_, apePrm); // even rows (o=0)
            }
        }
        SetWaitFlag<HardEvent::MTE3_V>(); // dn_kv flush drained (staging free for pass 2)
        StageTiles(kvBlk_);
        FlushDn(dnScGM_, b, j, d0);
        // no trailing MTE3_V: the dn_sc flush drains under the next block's
        // loads; it is awaited at the top of the next ProcessBlockSegment (and
        // at segment end by the d_ape flush's MTE3_V).
    }

    // stage all rows of a row-major fp32 block (N, seg) into the cur/prev dn
    // slot tiles; coff==1: rows map 1:1 onto the cur tile (contiguous cast);
    // coff==2: odd rows -> cur tile, even rows -> prev tile (strided casts).
    // fp16 additionally produces the lo tile: lo = fp16(row - fp32(hi)).
    __aicore__ inline void StageTiles(const LocalTensor<float>& blk)
    {
        const uint32_t seg = segLen_;
        const uint8_t rowBlk = static_cast<uint8_t>(seg / 8);
        const uint8_t rowBlkH = static_cast<uint8_t>(seg / 16);
        if (coff_ == 1) {
            Cast(stageHiCur_, blk, RoundMode::CAST_NONE, crSize_ * seg);
            if constexpr (std::is_same_v<ioT, half>) {
                Cast(hBTile_, stageHiCur_, RoundMode::CAST_NONE, crSize_ * seg);
                Sub(hBTile_, blk, hBTile_, static_cast<int32_t>(crSize_ * seg));
                Cast(stageLoCur_, hBTile_, RoundMode::CAST_NONE, crSize_ * seg);
            }
        } else {
            UnaryRepeatParams castPrm{1, 1, rowBlkH, static_cast<uint8_t>(2 * rowBlk)};
            Cast(stageHiCur_, blk[seg], RoundMode::CAST_NONE, seg, crSize_, castPrm); // odd rows
            Cast(stageHiPrev_, blk, RoundMode::CAST_NONE, seg, crSize_, castPrm);     // even rows
            if constexpr (std::is_same_v<ioT, half>) {
                BinaryRepeatParams subPrm{1, 1, 1, rowBlk, static_cast<uint8_t>(2 * rowBlk), rowBlk};
                Cast(hBTile_, stageHiCur_, RoundMode::CAST_NONE, crSize_ * seg);
                Sub(hBTile_, blk[seg], hBTile_, seg, crSize_, subPrm);
                Cast(stageLoCur_, hBTile_, RoundMode::CAST_NONE, crSize_ * seg);
                Cast(hBTile_, stageHiPrev_, RoundMode::CAST_NONE, crSize_ * seg);
                Sub(hBTile_, blk, hBTile_, seg, crSize_, subPrm);
                Cast(stageLoPrev_, hBTile_, RoundMode::CAST_NONE, crSize_ * seg);
            }
        }
    }

    // flush the staged slot tiles: cur tile -> tokens [j*cr, (j+1)*cr) cols
    // [(coff-1)*D+d0, +seg); prev tile (coff==2, j>=1) -> tokens [(j-1)*cr,
    // j*cr) cols [d0, +seg); one 2D DataCopyPad per tile per HiLo plane.
    __aicore__ inline void FlushDn(const GlobalTensor<half>& dn, uint32_t b, uint32_t j, uint32_t d0)
    {
        const uint32_t seg = segLen_;
        DataCopyExtParams cp{static_cast<uint16_t>(crSize_), static_cast<uint32_t>(seg * sizeof(half)), 0,
                             static_cast<uint32_t>((cSize_ - seg) * sizeof(half)), 0};
        uint64_t curOff = (static_cast<uint64_t>(b) * sSize_ + j * crSize_) * cSize_ + (coff_ - 1) * dSize_ + d0;
        SetWaitFlag<HardEvent::V_MTE3>();
        DataCopyPad(dn[curOff], stageHiCur_, cp);
        if constexpr (std::is_same_v<ioT, half>) {
            DataCopyPad(dn[curOff + dnPlaneElems_], stageLoCur_, cp);
        }
        if (coff_ == 2 && j >= 1) {
            uint64_t prevOff = (static_cast<uint64_t>(b) * sSize_ + (j - 1) * crSize_) * cSize_ + d0;
            DataCopyPad(dn[prevOff], stageHiPrev_, cp);
            if constexpr (std::is_same_v<ioT, half>) {
                DataCopyPad(dn[prevOff + dnPlaneElems_], stageLoPrev_, cp);
            }
        }
    }

    GlobalTensor<ioT> dCkGM_;
    GlobalTensor<float> smGM_, kvGM_, dApeGM_;
    GlobalTensor<half> dnKvGM_, dnScGM_;
    uint32_t bSize_, sSize_, dSize_, crSize_, coff_, nblk_, nblkIn_, nRows_, cSize_, segLen_;
    uint32_t blocksPerCore_, scatterCoreNum_;
    uint64_t dnPlaneElems_;
    TBuf<TPosition::VECCALC> buf_;
    LocalTensor<float> apeCur_, apePrev_, sm_, kvBlk_, dkvBlk_, colAcc_, dC32_, hBTile_;
    LocalTensor<ioT> dCIn_;
    LocalTensor<half> stageHiCur_, stageHiPrev_, stageLoCur_, stageLoPrev_;
};

#endif // COMPRESSOR_GRAD_CAST_SCATTER_H
