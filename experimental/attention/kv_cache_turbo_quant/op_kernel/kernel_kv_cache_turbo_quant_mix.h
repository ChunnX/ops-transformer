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
 * @file kernel_kv_cache_turbo_quant_mix.h
 *
 * TurboQuant encode kernel for the prefill path (large vector counts,
 * qjl_dim == head_dim), launched with KERNEL_TYPE_MIX_AIC_1_2.
 *
 * AIC/AIV role division (Wy/QSFA-style):
 *  - AIC blocks run only cube matmul: z = (Hh+Hl) @ x and p = (Sh+Sl) @ (rh+rl),
 *    with fp16 hi/lo splits of H and S (combined split error ~2^-22, fp32-grade).
 *  - AIV blocks run everything else (norms, MSE quantize, residual norms, sign,
 *    bit packing, output flush). Each logical tri-core group c handles one
 *    vector range R_c; AIC c processes all of R_c, its two partner AIVs each
 *    process half of R_c's vector work.
 *  - GM spin-flag handshakes (x ready, z ready, r_hat ready, p ready)
 *    synchronize the phases (the FFTS CrossCore channel never delivers tokens
 *    in this stack; the flags use volatile GM stores + dcci/dc_preload polls).
 *
 * The two fp32-precision matrix-vector products that bound the pure-vector
 * kernel (WholeReduceSum row reductions) are thus computed entirely on the
 * cube units at fp16 speed with fp32-grade precision.
 *
 * IMPORTANT: bare cube work in a MIX launch cannot repeat in this stack
 * (every 2nd launch of the same executor dies with 507015) - this kernel is
 * therefore driven through a fresh executor per launch by the op_api layer.
 */

#ifndef KV_CACHE_TURBO_QUANT_KERNEL_MIX_H_
#define KV_CACHE_TURBO_QUANT_KERNEL_MIX_H_

#include "kernel_operator.h"

using namespace AscendC;

class KernelKvCacheTurboQuantMix {
public:
    __aicore__ inline KernelKvCacheTurboQuantMix() {}

    // minimal bisect: one tile of the flow, stages dumped into the outputs
    __aicore__ inline void BisectTile0()
    {
        // stage 1: z read-back from GM into the OUTPUT norm tensor (visible)
        LocalTensor<float> zT = zTileBuf_.Get<float>();
        LocalTensor<bfloat16_t> xT = xTileBuf_.Get<bfloat16_t>();
        SyncPipe<HardEvent::V_MTE2>();
        DataCopy(xT, kvGm_[0], HEAD_DIM);
        DataCopy(zT, wsZ1Gm_[0], HEAD_DIM);
        SyncPipe<HardEvent::MTE2_V>();
        // dump the wsZ1 physical address and the role's user base for the host
        {
            GlobalTensor<int32_t> addrGm;
            addrGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flagRaw_ + (12UL * 1024 * 1024)) + 64,
                                   8);
            const uint64_t zAddr = reinterpret_cast<uint64_t>(wsZ1Gm_.GetPhyAddr());
            const uint64_t fAddr = reinterpret_cast<uint64_t>(flagRaw_);
            int32_t zlo = static_cast<int32_t>(zAddr & 0xFFFFFFFF);
            int32_t zhi = static_cast<int32_t>(zAddr >> 32);
            int32_t flo = static_cast<int32_t>(fAddr & 0xFFFFFFFF);
            int32_t fhi = static_cast<int32_t>(fAddr >> 32);
            SyncPipe<HardEvent::S_MTE3>();
            addrGm.SetValue(0, zlo);
            addrGm.SetValue(1, zhi);
            addrGm.SetValue(2, flo);
            addrGm.SetValue(3, fhi);
        }
        LocalTensor<bfloat16_t> zBF = normBFBuf_.Get<bfloat16_t>();
        Cast(zBF, zT, RoundMode::CAST_RINT, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(normGm_[0], zBF, HEAD_DIM);           // z1(bf16) -> norm out
        DataCopy(gammaGm_[0], xT, HEAD_DIM);           // x  -> gamma out
        SyncPipe<HardEvent::MTE3_V>();
    }



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
        coreNum_ = tiling->coreNum;   // logical tri-core groups used

        flagRaw_ = reinterpret_cast<__gm__ int32_t *>(workspace);
        flagGm_.SetGlobalBuffer(flagRaw_, FLAG_ELEMS);
        flagU64_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(workspace), FLAG_ELEMS / 8);
        isAic_ = false;
        int64_t logicalIdx = 0;
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
            }
        }
        if ASCEND_IS_AIC {
            isAic_ = true;
            logicalIdx = GetBlockIdx();
        } else {
            logicalIdx = GetBlockIdx() / 2;   // two AIVs share one logical group
        }
        logicalIdx_ = logicalIdx;

        // logical range assignment in chunks of CHUNK_VEC vectors (32B-aligned outputs)
        const int64_t chunkNum = (totalVec_ + CHUNK_VEC - 1) / CHUNK_VEC;
        const int64_t base = chunkNum / static_cast<int64_t>(coreNum_);
        const int64_t rem = chunkNum % static_cast<int64_t>(coreNum_);
        int64_t myChunks = 0;
        int64_t chunkBegin = 0;
        if (logicalIdx < rem) {
            myChunks = base + 1;
            chunkBegin = logicalIdx * (base + 1);
        } else {
            myChunks = base;
            chunkBegin = rem * (base + 1) + (logicalIdx - rem) * base;
        }
        rangeBegin_ = chunkBegin * CHUNK_VEC;
        int64_t rangeEnd = (chunkBegin + myChunks) * CHUNK_VEC;
        if (rangeEnd > totalVec_) {
            rangeEnd = totalVec_;
        }
        rangeLen_ = (myChunks > 0) ? (rangeEnd - rangeBegin_) : 0;

        // AIV sub-role within the group: even AIV takes the first half of the
        // vector work, odd AIV the second half. TILE-aligned split so every
        // 16-vec tile belongs to exactly one sub (per-chunk X/R flags).
        if (!isAic_) {
            const int64_t subIdx = GetBlockIdx() % 2;
            const int64_t halfTiles = (rangeLen_ / 2 + TILE_VEC - 1) / TILE_VEC;
            int64_t halfLen = halfTiles * TILE_VEC;
            if (halfLen > rangeLen_ || (rangeLen_ > 0 && halfTiles == 0)) {
                halfLen = rangeLen_;
            }
            vecBegin_ = rangeBegin_ + subIdx * halfLen;
            vecNum_ = rangeLen_ - subIdx * halfLen;
            if (vecNum_ > halfLen) {
                vecNum_ = halfLen;
            }
            aivSubIdx_ = subIdx;
        }

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kvVectors), totalVec_ * HEAD_DIM);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rotationMatrix), HEAD_DIM * HEAD_DIM);
        qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qjlMatrix), qjlDim_ * HEAD_DIM);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantIdx), totalVec_ * idxBytesPerVec_);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantQjl), totalVec_ * qjlBytesPerVec_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantNorm), totalVec_);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantGamma), totalVec_);

        // workspace layout (bytes):
        //   0..16K      cross-core GM spin flags (64B-strided int32 slots)
        //   16K..144K   Hh/Hl/Sh/Sl fp16 splits (4 x 32KB)
        //   144K        z1/z2 fp32 areas (2 x totalVec x 512B)
        //   then        rh/rl fp16 areas (2 x totalVec x 256B)
        //   then        p1..p4 fp32 areas (4 x totalVec x 512B)
        flagRaw_ = reinterpret_cast<__gm__ int32_t *>(workspace);
        dbgMark_ = flagRaw_ + 3584;   // marker block: cells 3584..3770 (real flags end at 3071; ws probes at 3776..3807)
        flagGm_.SetGlobalBuffer(flagRaw_, FLAG_ELEMS);
        flagU64_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(workspace), FLAG_ELEMS / 8);
        __gm__ half *wsH = reinterpret_cast<__gm__ half *>(workspace + WS_SPLIT_OFF);
        wsHhGm_.SetGlobalBuffer(wsH + 0 * SPLIT_ELEMS, SPLIT_ELEMS);
        wsHlGm_.SetGlobalBuffer(wsH + 1 * SPLIT_ELEMS, SPLIT_ELEMS);
        wsShGm_.SetGlobalBuffer(wsH + 2 * SPLIT_ELEMS, SPLIT_ELEMS);
        wsSlGm_.SetGlobalBuffer(wsH + 3 * SPLIT_ELEMS, SPLIT_ELEMS);
        __gm__ float *wsF = reinterpret_cast<__gm__ float *>(workspace);
        wsZ1Gm_.SetGlobalBuffer(wsF + WS_Z_OFF / 4 + 0 * totalVec_ * HEAD_DIM, totalVec_ * HEAD_DIM);
        wsZ2Gm_.SetGlobalBuffer(wsF + WS_Z_OFF / 4 + 1 * totalVec_ * HEAD_DIM, totalVec_ * HEAD_DIM);
        __gm__ half *wsR = reinterpret_cast<__gm__ half *>(workspace + WS_Z_OFF + 2 * totalVec_ * HEAD_DIM * 4);
        wsRhat2Gm_.SetGlobalBuffer(wsR, 2 * totalVec_ * HEAD_DIM);   // [vec][rh|rl] interleaved
        __gm__ float *wsP = reinterpret_cast<__gm__ float *>(workspace + WS_Z_OFF + 2 * totalVec_ * HEAD_DIM * 4 +
                                                              2 * totalVec_ * HEAD_DIM * 2);
        wsP1Gm_.SetGlobalBuffer(wsP + 0 * totalVec_ * HEAD_DIM, totalVec_ * HEAD_DIM);
        wsP2Gm_.SetGlobalBuffer(wsP + 1 * totalVec_ * HEAD_DIM, totalVec_ * HEAD_DIM);
        wsP3Gm_.SetGlobalBuffer(wsP + 2 * totalVec_ * HEAD_DIM, totalVec_ * HEAD_DIM);
        wsP4Gm_.SetGlobalBuffer(wsP + 3 * totalVec_ * HEAD_DIM, totalVec_ * HEAD_DIM);
        wsXhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wsP + 4 * totalVec_ * HEAD_DIM),
                                totalVec_ * HEAD_DIM);
        if (!isAic_ && GetBlockIdx() == 0) {
            // dump the kernel's view of the output tensor addresses
            const uint64_t na = reinterpret_cast<uint64_t>(normGm_.GetPhyAddr());
            const uint64_t ga = reinterpret_cast<uint64_t>(gammaGm_.GetPhyAddr());
            const uint64_t ia = reinterpret_cast<uint64_t>(idxGm_.GetPhyAddr());
            const uint64_t qa = reinterpret_cast<uint64_t>(qjlOutGm_.GetPhyAddr());
        }


        if (isAic_) {
            pipe_.InitBuffer(l1ABuf_, L1_A_BYTES);
            pipe_.InitBuffer(l1BBuf_, L1_B_BYTES);
            pipe_.InitBuffer(l0ABuf_, L0_A_BYTES);
            pipe_.InitBuffer(l0BBuf_, L0_B_BYTES);
            pipe_.InitBuffer(l0CBuf_, L0_C_BYTES);
            eventMte2Mte1_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1);
            eventMte1M_ = GetTPipePtr()->FetchEventID(HardEvent::MTE1_M);
            eventMMte1_ = GetTPipePtr()->FetchEventID(HardEvent::M_MTE1);
            eventMMte1B_ = GetTPipePtr()->FetchEventID(HardEvent::M_MTE1);
            eventMte1Mte2_ = GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2);
            eventFixS_ = GetTPipePtr()->FetchEventID(HardEvent::FIX_S);
            eventFixM_ = GetTPipePtr()->FetchEventID(HardEvent::FIX_M);
            eventMFix_ = GetTPipePtr()->FetchEventID(HardEvent::M_FIX);

        } else {
            eventVS_ = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            eventSV_ = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            eventMte2V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            eventVMte2_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            eventVMte3_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            eventMte3V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            eventMte3Mte2_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            eventMte3S_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_S);
            pipe_.InitBuffer(zTileBuf_, TILE_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(pTileBuf_, TILE_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(xTileBuf_, TILE_VEC * HEAD_DIM * sizeof(bfloat16_t));
            pipe_.InitBuffer(flagProbeBuf_, 64);
            pipe_.InitBuffer(yBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(yhatBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(idxFBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(pBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(bvBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(wtLBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(wtHBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(wordIdxLoBuf_, 2 * TILE_VEC * WORDS_PER_VEC * sizeof(float));
            pipe_.InitBuffer(wordIdxHiBuf_, 2 * TILE_VEC * WORDS_PER_VEC * sizeof(float));
            pipe_.InitBuffer(normVecBuf_, TILE_VEC * sizeof(float));
            pipe_.InitBuffer(rnVecBuf_, TILE_VEC * sizeof(float));
            pipe_.InitBuffer(gammaVecBuf_, TILE_VEC * sizeof(float));
            pipe_.InitBuffer(normBFBuf_, TILE_VEC * sizeof(bfloat16_t));
            pipe_.InitBuffer(gammaBFBuf_, TILE_VEC * sizeof(bfloat16_t));
            pipe_.InitBuffer(idxStageBuf_, 2 * TILE_VEC * MAX_IDX_BYTES);
            pipe_.InitBuffer(qjlStageBuf_, TILE_VEC * WORDS_PER_VEC);
            pipe_.InitBuffer(rhatRhBuf_, TILE_VEC * HEAD_DIM * sizeof(half));
            pipe_.InitBuffer(rhatRlBuf_, TILE_VEC * HEAD_DIM * sizeof(half));
            pipe_.InitBuffer(rhatBackBuf_, TILE_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(rhatSubBuf_, TILE_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(xUpF32Buf_, TILE_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(xUpHBuf_, TILE_VEC * HEAD_DIM * sizeof(half));
            pipe_.InitBuffer(redTmpBuf_, RED_TMP_ELEMS * sizeof(float));
            pipe_.InitBuffer(splitBuf_, SPLIT_STAGE_BYTES);
            if (GetBlockIdx() == 0) {
            }
        }

        if (!isAic_) {
            InitConstants();
        }
    }

    __aicore__ inline void Process()
    {
        if (totalVec_ <= 0 || rangeLen_ <= 0) {
            return;
        }
        if (isAic_) {
            const int32_t slotBase = static_cast<int32_t>(logicalIdx_) * 8 * SLOT_STRIDE;
            // ---- phase 1 (AIC): z1 = Hh @ x, z2 = Hl @ x for the whole range
            GmFlagWait(slotBase + 0 * SLOT_STRIDE, MAGIC_X_LO);
            GmFlagWait(slotBase + 1 * SLOT_STRIDE, MAGIC_X_HI);
            LoadSplitToL1(wsHhGm_, wsHlGm_);
            for (int64_t t0 = 0; t0 < rangeLen_; t0 += TILE_VEC) {
                const int64_t g = rangeBegin_ + t0;
                MmTile(wsXhGm_[g * HEAD_DIM], wsZ1Gm_[g * HEAD_DIM], wsZ2Gm_[g * HEAD_DIM], t0);
            }
            SyncPipe<HardEvent::FIX_S>();   // z tiles in L2 before the flag store
            // per-line flush of the z range through the ATOMIC cache domain
            // (the cube DMA write path) so the partner AIVs observe it
            {
                __gm__ float *zBase = const_cast<__gm__ float *>(wsZ1Gm_.GetPhyAddr()) +
                                      rangeBegin_ * HEAD_DIM;
                const int64_t lines = (rangeLen_ * HEAD_DIM * 2 * sizeof(float) + 63) / 64;
                GlobalTensor<int64_t> zLine;
                zLine.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(zBase), lines * 8);
                for (int64_t li = 0; li < lines; ++li) {
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ATOMIC>(
                        zLine[li * 8]);
                }
            }
            GmFlagSet(slotBase + 2 * SLOT_STRIDE, MAGIC_Z);   // Z_LO
            GmFlagSet(slotBase + 6 * SLOT_STRIDE, MAGIC_Z);   // Z_HI
            // ---- phase 3 (AIC): waits r_hat splits, then p = (Sh+Sl) @ (rh+rl)
            GmFlagWait(slotBase + 3 * SLOT_STRIDE, MAGIC_R_LO);
            GmFlagWait(slotBase + 4 * SLOT_STRIDE, MAGIC_R_HI);
            LoadSplitToL1(wsShGm_, wsSlGm_);
            for (int64_t t0 = 0; t0 < rangeLen_; t0 += TILE_VEC) {
                const int64_t g = rangeBegin_ + t0;
                // [rh|rl] plane-major per tile: rl follows the tile's rh block
                MmTile2(wsRhat2Gm_[g * 2 * HEAD_DIM], wsRhat2Gm_[g * 2 * HEAD_DIM + TILE_VEC * HEAD_DIM],
                        wsP1Gm_[g * HEAD_DIM], wsP2Gm_[g * HEAD_DIM], wsP3Gm_[g * HEAD_DIM], wsP4Gm_[g * HEAD_DIM],
                        t0);
            }
            SyncPipe<HardEvent::FIX_S>();
            {
                __gm__ float *pBase = const_cast<__gm__ float *>(wsP1Gm_.GetPhyAddr()) +
                                      rangeBegin_ * HEAD_DIM;
                const int64_t lines = (rangeLen_ * HEAD_DIM * 4 * sizeof(float) + 63) / 64;
                GlobalTensor<int64_t> pLine;
                pLine.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(pBase), lines * 8);
                for (int64_t li = 0; li < lines; ++li) {
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ATOMIC>(
                        pLine[li * 8]);
                }
            }
            GmFlagSet(slotBase + 5 * SLOT_STRIDE, MAGIC_P);   // P_LO
            GmFlagSet(slotBase + 7 * SLOT_STRIDE, MAGIC_P);   // P_HI
        } else {
            ProcessAiv();
        }
    }

private:
    static constexpr int32_t HEAD_DIM = 128;
    static constexpr int32_t TILE_VEC = 16;
    static constexpr int32_t CHUNK_VEC = 16;
    static constexpr int32_t BATCH_VEC = 16;
    static constexpr int32_t WORDS_PER_VEC = 16;
    static constexpr int32_t MAX_IDX_BYTES = 64;
    static constexpr int32_t RED_TMP_ELEMS = 1024;
    static constexpr int32_t SPLIT_ELEMS = HEAD_DIM * HEAD_DIM;
    static constexpr int32_t SPLIT_STAGE_BYTES = 56 * 1024;
    static constexpr int32_t L1_A_BYTES = 4 * 2 * TILE_VEC * HEAD_DIM * sizeof(half);   // up to 4 x 8K
    static constexpr int32_t L1_B_BYTES = 2 * HEAD_DIM * HEAD_DIM * sizeof(half);   // 64K: hi+lo splits
    static constexpr int32_t L0_A_BYTES = 4 * 2 * TILE_VEC * HEAD_DIM * sizeof(half);   // up to 4 x 8K
    static constexpr int32_t L0_B_BYTES = 2 * HEAD_DIM * HEAD_DIM * sizeof(half);   // 64K: hi+lo splits
    static constexpr int32_t L0_C_BYTES = TILE_VEC * HEAD_DIM * sizeof(float);      // 8K
    static constexpr int32_t WS_SPLIT_OFF = 16 * 1024;
    static constexpr int32_t WS_Z_OFF = 144 * 1024;
    static constexpr int32_t FLAG_ELEMS = 4096;
    static constexpr float MIN_DENOM = 1e-30f;
    static constexpr float IND_SCALE = 1.2676506002282294e30f;
    // GM spin-flag protocol (the FFTS CrossCore channel never delivers tokens
    // in this stack; volatile GM stores + dcci + dc_preload polls are proven).
    // Slots are 64B-strided int32 cells; group g owns slots [g*8, g*8+8):
    // 0=X_LO 1=X_HI 2=Z_LO 3=R_LO 4=R_HI 5=P_LO 6=Z_HI 7=P_HI.
    // Waiters reset their slot to 0 after consuming (stream-serialized
    // launches of this kernel start clean - verified via host peek).
    static constexpr int32_t CHUNK_TILES = 4;             // pipeline chunk = 4 tiles (64 vecs)
    static constexpr int32_t GROUP_FLAG_CELLS = 64;       // per-group flag block
    static constexpr int32_t MAGIC_X_LO = 0x5A5A0001;
    static constexpr int32_t MAGIC_X_HI = 0x5A5A0002;
    static constexpr int32_t MAGIC_Z = 0x5A5A0003;
    static constexpr int32_t MAGIC_R_LO = 0x5A5A0004;
    static constexpr int32_t MAGIC_R_HI = 0x5A5A0005;
    static constexpr int32_t MAGIC_P = 0x5A5A0006;
    static constexpr int32_t SLOT_STRIDE = 16;   // int32 elems per 64B slot

    template <HardEvent evt>
    __aicore__ inline void SyncPipe()
    {
        TEventID eventId = GetEventId<evt>();
        SetFlag<evt>(eventId);
        WaitFlag<evt>(eventId);
    }

    // ---- GM spin-flag primitives (KFC-comm proven recipe: raw GM store/load
    // + asc_dcci_single + asc_datacache_preload; the preload after the
    // invalidate is what makes the poller observe remote stores) ----
    // GM spin-flag primitives (KFC-comm proven recipe: volatile GM store/load
    // + asm fences + DataCacheCleanAndInvalid + DataCachePreload; the preload
    // after the invalidate is what makes a poller observe remote stores).
    __aicore__ inline void GmFlagSet(int32_t slotElem, int32_t magic)
    {
        if (isAic_) {
            volatile __gm__ int32_t *slot = flagRaw_ + slotElem;
            *slot = magic;
            __asm__ __volatile__("" ::: "memory");
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                flagGm_[slotElem]);
            __asm__ __volatile__("" ::: "memory");
        } else {
            // AIV: dcci on dav_m200 poisons the core's later GM writes (the
            // write-poison ghost - bisected 2026-09-26 PM: in-loop dcci'd flag
            // sets killed the x-upload's MTE3 landing). Ship the flag via a
            // DMA line update instead (MTE2-in / modify / MTE3-out).
            LocalTensor<int32_t> probe = flagProbeBuf_.Get<int32_t>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(probe, flagGm_[slotElem & ~15], 16);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::V_S>();
            probe.SetValue(slotElem & 15, magic);
            SyncPipe<HardEvent::S_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(flagGm_[slotElem & ~15], probe, 16);
            SyncPipe<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void GmFlagWait(int32_t slotElem, int32_t magic)
    {
        volatile __gm__ int32_t *slot = flagRaw_ + slotElem;
        if (isAic_) {
            // KFC RcvMessageImpl recipe: invalidate the stale local line, then
            // preload, then re-read. Without the invalidate the preload keeps
            // serving the core's original (zero) copy and the poll never sees
            // the AIV's store (bisected 2026-09-26: AIC stuck at X wait with
            // the flag host-visible).
            while (true) {
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    flagGm_[slotElem]);
                __asm__ __volatile__("" ::: "memory");
                DataCachePreload<int64_t>(flagU64_[slotElem / 16], int64_t(0));
                __asm__ __volatile__("" ::: "memory");
                if (*slot == magic) {
                    break;
                }
            }
        } else {
            // dcci on dav_m200 poisons later GM writes (the ghost, confirmed
            // 2026-09-26 PM) - poll the flag line via an MTE2 DMA read instead:
            // the copy snoops L2 and leaves this core's write path intact
            LocalTensor<int32_t> probe = flagProbeBuf_.Get<int32_t>();
            while (true) {
                SyncPipe<HardEvent::V_MTE2>();
                DataCopy(probe, flagGm_[slotElem & ~15], 16);
                SyncPipe<HardEvent::MTE2_V>();
                SyncPipe<HardEvent::V_S>();
                if (probe.GetValue(slotElem & 15) == magic) {
                    break;
                }
                SyncPipe<HardEvent::S_V>();
                for (int32_t d = 0; d < 256; ++d) {
                    __asm__ __volatile__("" ::: "memory");
                }
            }
        }
        if (isAic_) {
            *slot = 0;
            __asm__ __volatile__("" ::: "memory");
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                flagGm_[slotElem]);
            __asm__ __volatile__("" ::: "memory");
        } else {
            LocalTensor<int32_t> probe = flagProbeBuf_.Get<int32_t>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(probe, flagGm_[slotElem & ~15], 16);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::V_S>();
            probe.SetValue(slotElem & 15, 0);
            SyncPipe<HardEvent::S_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(flagGm_[slotElem & ~15], probe, 16);
            SyncPipe<HardEvent::MTE3_V>();
        }
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
        } else if constexpr (evt == HardEvent::MTE3_MTE2) {
            return eventMte3Mte2_;
        } else if constexpr (evt == HardEvent::FIX_S) {
            return eventFixS_;
        } else if constexpr (evt == HardEvent::MTE3_S) {
            return eventMte3S_;
        } else {
            return eventMte3V_;
        }
    }

    __aicore__ inline void InitConstants()
    {
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
        for (int32_t i = 0; i < nCent_ - 1; ++i) {
            bound_[i] = (cent_[i] + cent_[i + 1]) * 0.5f;
            delta_[i] = cent_[i + 1] - cent_[i];
        }
        LocalTensor<float> wtL = wtLBuf_.Get<float>();
        LocalTensor<float> wtH = wtHBuf_.Get<float>();
        for (int32_t d = 0; d < BATCH_VEC * HEAD_DIM; ++d) {
            const int32_t lane = d & 7;
            const float lo = (lane < 4) ? static_cast<float>(static_cast<int32_t>(1) << (mseBits_ * lane)) : 0.0f;
            const float hi = (lane >= 4) ? static_cast<float>(static_cast<int32_t>(1) << (mseBits_ * (lane - 4)))
                                         : 0.0f;
            wtL.SetValue(d, lo);
            wtH.SetValue(d, hi);
        }
        if (GetBlockIdx() == 0) {
        }
        SyncPipe<HardEvent::S_V>();
        if (GetBlockIdx() == 0) {
        }
    }

    // fp32 H/S -> fp16 hi/lo splits to GM (AIV block 0 only)
    __aicore__ inline void SplitMatrices()
    {
        LocalTensor<float> chunk = splitBuf_.Get<float>();                    // 16KB fp32
        LocalTensor<float> backC = splitBuf_.Get<float>()[4 * 1024];          // 16KB
        LocalTensor<float> subC = splitBuf_.Get<float>()[8 * 1024];           // 16KB
        LocalTensor<half> hC = splitBuf_.Get<half>()[24 * 1024];              // 8KB fp16
        for (int32_t m = 0; m < 2; ++m) {
            GlobalTensor<float> &srcGm = (m == 0) ? rotGm_ : qjlGm_;
            GlobalTensor<half> &ws0 = (m == 0) ? wsHhGm_ : wsShGm_;
            GlobalTensor<half> &ws1 = (m == 0) ? wsHlGm_ : wsSlGm_;
            for (int32_t c = 0; c < SPLIT_ELEMS / 4096; ++c) {
                SyncPipe<HardEvent::V_MTE2>();
                DataCopy(chunk, srcGm[c * 4096], 4096);
                SyncPipe<HardEvent::MTE2_V>();
                SyncPipe<HardEvent::MTE3_V>();   // hC free: previous ws1 copy drained
                Cast(hC, chunk, RoundMode::CAST_RINT, 4096);
                PipeBarrier<PIPE_V>();
                SyncPipe<HardEvent::V_MTE3>();
                DataCopy(ws0[c * 4096], hC, 4096);
                Cast(backC, hC, RoundMode::CAST_NONE, 4096);   // read-read vs MTE3: ok
                PipeBarrier<PIPE_V>();
                Sub(subC, chunk, backC, 4096);
                PipeBarrier<PIPE_V>();
                SyncPipe<HardEvent::MTE3_V>();   // ws0 copy done reading hC
                Cast(hC, subC, RoundMode::CAST_RINT, 4096);
                PipeBarrier<PIPE_V>();
                SyncPipe<HardEvent::V_MTE3>();
                DataCopy(ws1[c * 4096], hC, 4096);
            }
        }
        SyncPipe<HardEvent::MTE3_V>();
    }

    // AIV-side full flow: norms, quantize, idx pack, r_hat split upload, sign, flush
    __aicore__ inline void ProcessAiv()
    {
        LocalTensor<bfloat16_t> xTile = xTileBuf_.Get<bfloat16_t>();
        LocalTensor<uint16_t> xTileU16 = xTileBuf_.Get<uint16_t>();  // bit view: bisheng forbids bf16 scalar casts
        LocalTensor<float> zTile = zTileBuf_.Get<float>();
        LocalTensor<float> normVec = normVecBuf_.Get<float>();
        LocalTensor<float> rnVec = rnVecBuf_.Get<float>();
        LocalTensor<float> gammaVec = gammaVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> normBF = normBFBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> gammaBF = gammaBFBuf_.Get<bfloat16_t>();
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();


        // ---- phase 0 (even AIV of each group): fp16 hi/lo splits of H and S.
        // All groups write identical content to the shared splits area (benign
        // identical write-write race); the group's X_LO flag then implies the
        // splits are complete for this group's AIC.
        // every AIV computes the (identical) splits into the shared area; a
        // core's own writes guarantee the content its AIC reads is valid
        SplitMatrices();

        // ---- phase 0.5 (AIV): x bf16 -> fp16 tiles to GM for the cube A feed,
        // with a per-chunk X flag so the AIC starts z after the first chunk
        for (int64_t tileStart = 0; tileStart < vecNum_; tileStart += TILE_VEC) {
            const int32_t nvx = static_cast<int32_t>((vecNum_ - tileStart) < TILE_VEC ? (vecNum_ - tileStart)
                                                                                       : TILE_VEC);
            const int64_t gx = vecBegin_ + tileStart;
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(xTileBuf_.Get<bfloat16_t>(), kvGm_[gx * HEAD_DIM], nvx * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::MTE3_V>();   // xh free: previous tile's upload drained
            // dedicated full-tile buffers (the 1KB scratch buffers here sprayed
            // x-data over the idx pack weights every tile - same bug class as
            // the UploadRhat overflow bisected earlier today)
            LocalTensor<float> xf32 = xUpF32Buf_.Get<float>();
            LocalTensor<half> xh = xUpHBuf_.Get<half>();
            Cast(xf32, xTileBuf_.Get<bfloat16_t>(), RoundMode::CAST_NONE, nvx * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Cast(xh, xf32, RoundMode::CAST_RINT, nvx * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(wsXhGm_[gx * HEAD_DIM], xh, nvx * HEAD_DIM);
        }
        SyncPipe<HardEvent::MTE3_S>();   // x tiles in L2 before the flag store
        const int32_t slotBase = static_cast<int32_t>(logicalIdx_) * 8 * SLOT_STRIDE;
        GmFlagSet(slotBase + aivSubIdx_ * SLOT_STRIDE, aivSubIdx_ == 0 ? MAGIC_X_LO : MAGIC_X_HI);
        GmFlagWait(slotBase + (2 + 4 * aivSubIdx_) * SLOT_STRIDE, MAGIC_Z);
        if (GetBlockIdx() < 4) {
            // per-AIV V_S-covered x/z dumps at cells 140+blk*4
            volatile __gm__ int32_t *pv = flagRaw_ + 3968 + 140 + GetBlockIdx() * 4;
            *pv = static_cast<int32_t>(GetBlockIdx()) * 1000;
        }

        for (int64_t tileStart = 0; tileStart < vecNum_; tileStart += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((vecNum_ - tileStart) < TILE_VEC ? (vecNum_ - tileStart) : TILE_VEC);
            const int64_t g0 = vecBegin_ + tileStart;
            if (GetBlockIdx() == 0) {
            }

            // z = z1 + z2 (fp32 tiles from the cube)
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(xTile, kvGm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(zTile, wsZ1Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(pTileBuf_.Get<float>(), wsZ2Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            Add(zTile, zTile, pTileBuf_.Get<float>(), nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            if (GetBlockIdx() == 0 && tileStart == 0) {
                // bulk-landing test: copy z to a peekable workspace area via the
                // same UB->GM MTE3 path the output flush uses
                GlobalTensor<bfloat16_t> probeGm;
                probeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(flagRaw_ + (12UL * 1024 * 1024)),
                                        HEAD_DIM);
                LocalTensor<bfloat16_t> zb = normBFBuf_.Get<bfloat16_t>();
                Cast(zb, zTile, RoundMode::CAST_RINT, HEAD_DIM);
                PipeBarrier<PIPE_V>();
                SyncPipe<HardEvent::V_MTE3>();
                DataCopy(probeGm, zb, HEAD_DIM);
                SyncPipe<HardEvent::MTE3_V>();
                // and x likewise at +64B
                GlobalTensor<bfloat16_t> probeGm2;
                probeGm2.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(flagRaw_ + (12UL * 1024 * 1024)) +
                                             32,
                                         HEAD_DIM);
                SyncPipe<HardEvent::V_MTE3>();
                DataCopy(probeGm2, xTile, HEAD_DIM);
                SyncPipe<HardEvent::MTE3_V>();
            }
            if (GetBlockIdx() == 0 && tileStart == 0) {
                SyncPipe<HardEvent::V_S>();
                float z0 = zTile.GetValue(0);
                float z1v = zTile.GetValue(1);
                int32_t x0 = static_cast<int32_t>(xTileU16.GetValue(0));
                int32_t x1 = static_cast<int32_t>(xTileU16.GetValue(1));
                int32_t xi0 = *reinterpret_cast<int32_t *>(&x0);
                int32_t xi1 = *reinterpret_cast<int32_t *>(&x1);
                for (int32_t dz = 0; dz < 8; ++dz) {
                    const float zz = zTile.GetValue(dz);
                }
                SyncPipe<HardEvent::S_V>();
            }

            // ---- squared norms from x: full-tile ops + two 64-wide WRS halves
            LocalTensor<float> sqG = yBuf_.Get<float>();
            LocalTensor<float> sqTmp = yhatBuf_.Get<float>();
            Cast(sqG, xTile, RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Mul(sqG, sqG, sqG, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            // lo/hi 64-wide halves: each WRS repeat sums one vec's half-row
            WholeReduceSum(normVec, sqG, 64, nv, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(sqTmp, sqG[64], 64, nv, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            Add(normVec, normVec, sqTmp, nv);
            PipeBarrier<PIPE_V>();
            Sqrt(normVec, normVec, nv);
            PipeBarrier<PIPE_V>();
            if (GetBlockIdx() == 0 && tileStart == 0) {
                SyncPipe<HardEvent::V_S>();
                for (int32_t di = 0; di < 4; ++di) {
                    const float nv0 = normVec.GetValue(di);
                }
                SyncPipe<HardEvent::S_V>();
            }

            SyncPipe<HardEvent::V_S>();
            for (int32_t v = 0; v < nv; ++v) {
                const float norm = normVec.GetValue(v);
                inv_[v] = (norm > 0.0f) ? 1.0f / ((norm < MIN_DENOM) ? MIN_DENOM : norm) : 0.0f;
            }
            SyncPipe<HardEvent::S_V>();

            // ---- y = z * inv, batched MSE quantize (words into this tile's set)
            for (int32_t vb = 0; vb < nv; vb += BATCH_VEC) {
                const int32_t b = (nv - vb) < BATCH_VEC ? (nv - vb) : BATCH_VEC;
                for (int32_t i = 0; i < b; ++i) {
                    Muls(yBuf_.Get<float>()[i * HEAD_DIM], zTile[(vb + i) * HEAD_DIM], inv_[vb + i], HEAD_DIM);
                }
                PipeBarrier<PIPE_V>();
                QuantizeBatch(vb, b, zTile, rnVec);
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
            UnpackIdxAll(nv);
            SyncPipe<HardEvent::S_V>();

            // ---- r_hat (in place) and its fp16 hi/lo splits uploaded to GM
            for (int32_t v = 0; v < nv; ++v) {
                Muls(zTile[v * HEAD_DIM], zTile[v * HEAD_DIM], invR_[v], HEAD_DIM);
            }
            PipeBarrier<PIPE_V>();
            UploadRhat(g0, nv, zTile);

            SyncPipe<HardEvent::V_MTE3>();
            // bulk MTE3 flush for full tiles (retest after the staging-overflow
            // era's "bulk never lands" verdict was traced to clobbered UB, not
            // the copy); scalar path only for the tail
            const int32_t nvBulk = nv & ~(CHUNK_VEC - 1);
            if (nvBulk > 0) {
                DataCopy(normGm_[g0], normBF, nvBulk);
                DataCopy(gammaGm_[g0], gammaBF, nvBulk);
                DataCopy(idxGm_[g0 * idxBytesPerVec_], idxStage, nvBulk * idxBytesPerVec_);
            }
            if (nvBulk < nv) {
                FlushScalar(g0 + nvBulk, nvBulk, nv - nvBulk, normBF, gammaBF, idxStage, qjlStage, true, false);
                FlushOutLines(normGm_, g0 + nvBulk, (nv - nvBulk) * 2);
                FlushOutLines(gammaGm_, g0 + nvBulk, (nv - nvBulk) * 2);
                FlushOutLines(idxGm_, (g0 + nvBulk) * idxBytesPerVec_, (nv - nvBulk) * idxBytesPerVec_);
            }
            SyncPipe<HardEvent::MTE3_V>();

        }

        // signal r_hat ready (r splits already drained to L2 by the loop-end sync)
        if (GetBlockIdx() < 2) {
        }
        SyncPipe<HardEvent::MTE3_S>();
        if (GetBlockIdx() < 2) {
        }
        GmFlagSet(slotBase + (3 + aivSubIdx_) * SLOT_STRIDE, aivSubIdx_ == 0 ? MAGIC_R_LO : MAGIC_R_HI);

        // early norm/gamma visibility sweep: a handful of tiles lost their
        // phase-1 scalar stores to later cache activity before the final
        // sweep ran (bisected 2026-09-26: ~10 tiles of 0xEEEE poison survived)
        FlushOutLines(normGm_, vecBegin_, vecNum_ * 2);
        FlushOutLines(gammaGm_, vecBegin_, vecNum_ * 2);

        // ---- wait p, sign, qjl flush
        GmFlagWait(slotBase + (5 + 2 * aivSubIdx_) * SLOT_STRIDE, MAGIC_P);
        for (int64_t tileStart = 0; tileStart < vecNum_; tileStart += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((vecNum_ - tileStart) < TILE_VEC ? (vecNum_ - tileStart) : TILE_VEC);
            const int64_t g0 = vecBegin_ + tileStart;
            LocalTensor<float> pAll = zTileBuf_.Get<float>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(pAll, wsP1Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            DataCopy(pTileBuf_.Get<float>(), wsP2Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            Add(pAll, pAll, pTileBuf_.Get<float>(), nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(pTileBuf_.Get<float>(), wsP3Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            Add(pAll, pAll, pTileBuf_.Get<float>(), nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            // p4 = rl@Sl (~1e-9) is skipped: its sign-flip probability is zero

            LocalTensor<float> bv = bvBuf_.Get<float>();
            Duplicate(bv, 0.0f, BATCH_VEC * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            for (int32_t vb = 0; vb < nv; vb += BATCH_VEC) {
                const int32_t b = (nv - vb) < BATCH_VEC ? (nv - vb) : BATCH_VEC;
                Compare(qjlStageBuf_.Get<uint8_t>()[(vb / BATCH_VEC) * BATCH_VEC * qjlBytesPerVec_],
                        pAll[vb * HEAD_DIM], bv, CMPMODE::GE, b * HEAD_DIM);
                PipeBarrier<PIPE_V>();
            }

            SyncPipe<HardEvent::V_MTE3>();
            // CRITICAL: FlushScalar's SCALAR reads of the Compare (V-pipe)
            // bitmask need V_S - without it the stage reads one tile stale for
            // the first vecs of every tile (bisected 2026-09-26: qjl bad for
            // exactly the first BATCH_VEC vecs of each tile)
            SyncPipe<HardEvent::V_S>();
            const int32_t nvBulk = nv & ~(CHUNK_VEC - 1);
            if (nvBulk > 0) {
                DataCopy(qjlOutGm_[g0 * qjlBytesPerVec_], qjlStageBuf_.Get<uint8_t>(), nvBulk * qjlBytesPerVec_);
            }
            if (nvBulk < nv) {
                FlushScalar(g0 + nvBulk, nvBulk, nv - nvBulk, normBF, gammaBF, idxStage,
                            qjlStageBuf_.Get<uint8_t>(), false, true);
                FlushOutLines(qjlOutGm_, (g0 + nvBulk) * qjlBytesPerVec_, (nv - nvBulk) * qjlBytesPerVec_);
            }
            SyncPipe<HardEvent::MTE3_V>();
            SyncPipe<HardEvent::S_V>();
        }

    }

    // ---- low-level fp16 matmul service (AIC only): z = A @ B^T per 16x128 tile ----
    // ND (row-major) GM tile -> NZ L1
    template <typename T>
    __aicore__ inline void CopyGmNDToL1(const LocalTensor<T> &l1Tensor, const GlobalTensor<T> &gmTensor,
                                        uint32_t rowAct, uint32_t colStride)
    {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = rowAct;
        nd2nzPara.srcDValue = colStride;
        nd2nzPara.dValue = HEAD_DIM;
        nd2nzPara.dstNzC0Stride = rowAct;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.dstNzMatrixStride = 0;
        nd2nzPara.srcNdMatrixStride = 0;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    }

    // load one split matrix [128,128] into L1 B (B^T math needs straight fractals)
    __aicore__ inline void LoadSplitToL1(const GlobalTensor<half> &ws0, const GlobalTensor<half> &ws1)
    {
        CopyGmNDToL1(l1BBuf_.Get<half>(), ws0, HEAD_DIM, HEAD_DIM);
        CopyGmNDToL1(l1BBuf_.Get<half>()[SPLIT_ELEMS], ws1, HEAD_DIM, HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        // B (row-major H/S split in NZ L1) -> L0B. For C = A @ B^T the B
        // fractals load straight (ifTranspose=false) - the Wy/QSFA convention.
        LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = (HEAD_DIM + 15) / 16 * HEAD_DIM / (32 / sizeof(half));
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        LoadData(l0BBuf_.Get<half>(), l1BBuf_.Get<half>(), ld);
        LoadData(l0BBuf_.Get<half>()[SPLIT_ELEMS], l1BBuf_.Get<half>()[SPLIT_ELEMS], ld);
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
    }

    // one A tile through ONE split only (the p4 = rl@Sl term is negligible and skipped)
    __aicore__ inline void MmTile1(const GlobalTensor<half> &aGm, int32_t splitIdx, const GlobalTensor<float> &zGm,
                                   uint32_t aStride = HEAD_DIM)
    {
        const int32_t bufSel = mmCtr_ & 3;
        ++mmCtr_;
        LocalTensor<half> l1A = l1ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        LocalTensor<half> l0A = l0ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        CopyGmNDToL1(l1A, aGm, TILE_VEC, aStride);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        LoadData3DParamsV2<half> pa;
        pa.l1H = TILE_VEC / 16;
        pa.l1W = 16;
        pa.padList[0] = 0;
        pa.padList[1] = 0;
        pa.padList[2] = 0;
        pa.padList[3] = 255;
        pa.mExtension = TILE_VEC;
        pa.kExtension = HEAD_DIM;
        pa.mStartPt = 0;
        pa.kStartPt = 0;
        pa.strideH = 1;
        pa.strideW = 1;
        pa.filterW = 1;
        pa.filterSizeW = (1 >> 8) & 255;
        pa.filterH = 1;
        pa.filterSizeH = (1 >> 8) & 255;
        pa.dilationFilterH = 1;
        pa.dilationFilterW = 1;
        pa.fMatrixCtrl = 0;
        pa.channelSize = HEAD_DIM;
        pa.enTranspose = 0;
        LoadData<half>(l0A, l1A, pa);
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
        MmadParams mp;
        mp.m = TILE_VEC;
        mp.n = HEAD_DIM;
        mp.k = HEAD_DIM;
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = true;
        mp.unitFlag = 0b11;
        LocalTensor<float> cL0 = l0CBuf_.Get<float>();
        Mmad(cL0, l0A, l0BBuf_.Get<half>()[splitIdx * SPLIT_ELEMS], mp);
        SetFlag<HardEvent::M_MTE1>(eventMMte1_);
        WaitFlag<HardEvent::M_MTE1>(eventMMte1_);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOut(zGm, cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }
    }

    // one 32-row [rh|rl] A tile through both splits (p pass, 2 loads+4+4 -> 1+2+2)
    __aicore__ inline void MmTile32(const GlobalTensor<half> &aGm, int64_t tileIdx)
    {
        const int32_t bufSel = mmCtr_ & 1;
        ++mmCtr_;
        LocalTensor<half> l1A = l1ABuf_.Get<half>()[bufSel * 2 * TILE_VEC * HEAD_DIM];
        LocalTensor<half> l0A = l0ABuf_.Get<half>()[bufSel * 2 * TILE_VEC * HEAD_DIM];
        CopyGmNDToL1(l1A, aGm, 2 * TILE_VEC, 2 * HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        LoadData3DParamsV2<half> pa;
        pa.l1H = (2 * TILE_VEC) / 16;
        pa.l1W = 16;
        pa.padList[0] = 0;
        pa.padList[1] = 0;
        pa.padList[2] = 0;
        pa.padList[3] = 255;
        pa.mExtension = 2 * TILE_VEC;
        pa.kExtension = HEAD_DIM;
        pa.mStartPt = 0;
        pa.kStartPt = 0;
        pa.strideH = 1;
        pa.strideW = 1;
        pa.filterW = 1;
        pa.filterSizeW = (1 >> 8) & 255;
        pa.filterH = 1;
        pa.filterSizeH = (1 >> 8) & 255;
        pa.dilationFilterH = 1;
        pa.dilationFilterW = 1;
        pa.fMatrixCtrl = 0;
        pa.channelSize = HEAD_DIM;
        pa.enTranspose = 0;
        LoadData<half>(l0A, l1A, pa);
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
        MmadParams mp;
        mp.m = 2 * TILE_VEC;
        mp.n = HEAD_DIM;
        mp.k = HEAD_DIM;
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = true;
        mp.unitFlag = 0b11;
        LocalTensor<float> cL0 = l0CBuf_.Get<float>();
        Mmad(cL0, l0A, l0BBuf_.Get<half>(), mp);
        SetFlag<HardEvent::M_MTE1>(eventMMte1_);
        WaitFlag<HardEvent::M_MTE1>(eventMMte1_);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOut32(wsP1Gm_[tileIdx * 2 * TILE_VEC * HEAD_DIM], cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        Mmad(cL0, l0A, l0BBuf_.Get<half>()[SPLIT_ELEMS], mp);
        SetFlag<HardEvent::M_MTE1>(eventMMte1_);
        WaitFlag<HardEvent::M_MTE1>(eventMMte1_);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOut32(wsP3Gm_[tileIdx * 2 * TILE_VEC * HEAD_DIM], cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }
    }

    __aicore__ inline void FixpipeOut32(const GlobalTensor<float> &gmOut, const LocalTensor<float> &cL0)
    {
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = 2 * TILE_VEC;
        fp.srcStride = 2 * TILE_VEC;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.unitFlag = 0b11;
        Fixpipe(gmOut, cL0, fp);
    }

    // one A tile through both splits: z1 = A @ B0^T, z2 = A @ B1^T
    __aicore__ inline void MmTile(const GlobalTensor<half> &aGm, const GlobalTensor<float> &z1Gm,
                                  const GlobalTensor<float> &z2Gm, int64_t dbgT0 = -1, uint32_t aStride = HEAD_DIM)
    {
        // 4-deep A rotation driven by a monotonic MmTile counter: the loop-t0
        // parity collided for consecutive MmTiles within one MmTile2 (rl(t0)
        // and rh(t0+16) shared a slot -> p2 held Sh@rh of the next tile,
        // bisected 2026-09-26). A strict +1 counter guarantees distinct slots.
        const int32_t bufSel = mmCtr_ & 3;
        ++mmCtr_;
        const TEventID evAB = eventMMte1_;
        LocalTensor<half> l1A = l1ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        LocalTensor<half> l0A = l0ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        CopyGmNDToL1(l1A, aGm, TILE_VEC, HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);

        // L1(NZ) -> L0A via the QSFA fmatrix form (gathers the k fractals of
        // each m-fractal row)
        LoadData3DParamsV2<half> pa;
        pa.l1H = TILE_VEC / 16;
        pa.l1W = 16;
        pa.padList[0] = 0;
        pa.padList[1] = 0;
        pa.padList[2] = 0;
        pa.padList[3] = 255;
        pa.mExtension = TILE_VEC;
        pa.kExtension = HEAD_DIM;
        pa.mStartPt = 0;
        pa.kStartPt = 0;
        pa.strideH = 1;
        pa.strideW = 1;
        pa.filterW = 1;
        pa.filterSizeW = (1 >> 8) & 255;
        pa.filterH = 1;
        pa.filterSizeH = (1 >> 8) & 255;
        pa.dilationFilterH = 1;
        pa.dilationFilterW = 1;
        pa.fMatrixCtrl = 0;
        pa.channelSize = HEAD_DIM;
        pa.enTranspose = 0;
        LoadData<half>(l0A, l1A, pa);
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
        MmadParams mp;
        mp.m = TILE_VEC;
        mp.n = HEAD_DIM;
        mp.k = HEAD_DIM;
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = true;
        mp.unitFlag = 0b11;   // single-shot Mmad = last accumulation: allow fixpipe move-out (QSFA)
        LocalTensor<float> cL0 = l0CBuf_.Get<float>();
        LocalTensor<half> aL0 = l0A;
        Mmad(cL0, aL0, l0BBuf_.Get<half>(), mp);
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        // M_FIX: C result is in L0C before the fixpipe reads it (missing this
        // raced the CO1->GM copy against the Mmad L0C writes -> wedge)
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOut(z1Gm, cL0);
        // L0C is shared: the next Mmad must wait until the fixpipe drained it
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        Mmad(cL0, aL0, l0BBuf_.Get<half>()[SPLIT_ELEMS], mp);
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOut(z2Gm, cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        // coarse M-pipe barrier every 4th MmTile: bounds the load-ahead skew
        // at 3, which the 4-deep A rotation absorbs
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }
    }

    // r_hat tiles through both splits: p1 = rh @ Sh^T, p2 = rl @ Sh^T, p3 = rh @ Sl^T, p4 = rl @ Sl^T
    __aicore__ inline void MmTile2(const GlobalTensor<half> &rhGm, const GlobalTensor<half> &rlGm,
                                   const GlobalTensor<float> &p1Gm, const GlobalTensor<float> &p2Gm,
                                   const GlobalTensor<float> &p3Gm, const GlobalTensor<float> &p4Gm,
                                   int64_t dbgT0 = -1)
    {
        // alternate the A-rotation parity between the rh/rl sub-tiles so the
        // second MmTile's loads never touch the first's in-flight operands
        const int64_t t0 = (dbgT0 < 0) ? 0 : dbgT0;
        MmTile(rhGm, p1Gm, p3Gm, t0);
        MmTile1(rlGm, 0, p2Gm);
        (void)p4Gm;   // p4 = rl@Sl ~1e-9: not produced, not read
    }

    __aicore__ inline void FixpipeOut(const GlobalTensor<float> &gmOut, const LocalTensor<float> &cL0)
    {
        // Fixpipe-API L0C -> GM (the 2026-09-24 grid-completing form; the
        // DataCopyCO12DstParams replacement wedged the M pipe at tile ~4/5).
        // Its writes sit in the cube DMA (ATOMIC) coherency domain, so the
        // partner AIVs only observe them after the per-line CACHELINE_ATOMIC
        // flush loop that follows each phase's tile loop.
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = TILE_VEC;
        fp.srcStride = TILE_VEC;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.unitFlag = 0b11;
        Fixpipe(gmOut, cL0, fp);
    }

    // batched MSE quantization of b vectors starting at vb: y = z*inv already in
    // yBuf slots; writes residual into zTile, rnVec, and idx pack words
    __aicore__ inline void QuantizeBatch(int32_t vb, int32_t b, const LocalTensor<float> &zTile,
                                         const LocalTensor<float> &rnVec, int32_t wordSet = 0)
    {
        LocalTensor<float> y = yBuf_.Get<float>();
        LocalTensor<float> yhat = yhatBuf_.Get<float>();
        LocalTensor<float> idxF = idxFBuf_.Get<float>();
        LocalTensor<float> indF = pBuf_.Get<float>();
        const int32_t w = b * HEAD_DIM;
        const int32_t half = nCent_ / 2;

        // symmetric codebooks: quantize |y| with the positive half of the
        // bounds (3 indicators for mse3 instead of 7), then fold the sign back
        LocalTensor<float> sg = redTmpBuf_.Get<float>();
        // sg = sign(y) in {-1,0,+1} via the overflow indicator (before Abs)
        Muls(sg, y, IND_SCALE, w);
        PipeBarrier<PIPE_V>();
        Muls(sg, sg, IND_SCALE, w);
        PipeBarrier<PIPE_V>();
        Mins(sg, sg, 1.0f, w);
        PipeBarrier<PIPE_V>();
        Maxs(sg, sg, -1.0f, w);
        PipeBarrier<PIPE_V>();
        // quantize |y| in place; r = y - sg*yhat_pos = sg*(|y| - yhat_pos)
        Abs(y, y, w);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> ay = y;
        Duplicate(yhat, cent_[half], w);
        PipeBarrier<PIPE_V>();
        Duplicate(idxF, 0.0f, w);
        PipeBarrier<PIPE_V>();
        for (int32_t k = 1; k < half; ++k) {
            Adds(indF, ay, -bound_[half - 1 + k], w);
            PipeBarrier<PIPE_V>();
            Maxs(indF, indF, 0.0f, w);
            PipeBarrier<PIPE_V>();
            Muls(indF, indF, IND_SCALE, w);
            PipeBarrier<PIPE_V>();
            Muls(indF, indF, IND_SCALE, w);
            PipeBarrier<PIPE_V>();
            Mins(indF, indF, 1.0f, w);
            PipeBarrier<PIPE_V>();
            Add(idxF, idxF, indF, w);
            PipeBarrier<PIPE_V>();
            Axpy(yhat, indF, delta_[half - 1 + k], w);
            PipeBarrier<PIPE_V>();
        }
        // idx = idxP*sg + sg*0.5 + (half - 0.5); yhat stays UNSIGNED because
        // the residual folds the sign separately: r = sg*(|y| - yhat_pos)
        Mul(idxF, idxF, sg, w);
        PipeBarrier<PIPE_V>();
        Axpy(idxF, sg, 0.5f, w);
        PipeBarrier<PIPE_V>();
        Adds(idxF, idxF, static_cast<float>(half) - 0.5f, w);
        PipeBarrier<PIPE_V>();
        Sub(zTile[vb * HEAD_DIM], y, yhat, w);
        PipeBarrier<PIPE_V>();
        Mul(zTile[vb * HEAD_DIM], zTile[vb * HEAD_DIM], sg, w);
        PipeBarrier<PIPE_V>();
        Mul(y, zTile[vb * HEAD_DIM], zTile[vb * HEAD_DIM], w);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> rnTmp = pBuf_.Get<float>();
        WholeReduceSum(rnVec[vb], y, 64, b, 1, 1, 16);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(rnTmp, y[64], 64, b, 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Add(rnVec[vb], rnVec[vb], rnTmp, b);
        PipeBarrier<PIPE_V>();
        // idx bit-pack: lo/hi half-words so every group word is an exact fp32 integer
        LocalTensor<float> wtL = wtLBuf_.Get<float>();
        LocalTensor<float> wtH = wtHBuf_.Get<float>();
        LocalTensor<float> wordLo = wordIdxLoBuf_.Get<float>()[wordSet * TILE_VEC * WORDS_PER_VEC];
        LocalTensor<float> wordHi = wordIdxHiBuf_.Get<float>()[wordSet * TILE_VEC * WORDS_PER_VEC];
        Mul(y, idxF, wtL, w);
        PipeBarrier<PIPE_V>();
        // WRS repeat count caps at 255: with b=16 (256 group-words) split into
        // two 128-rep calls (bisected 2026-09-26: idx garbage at BATCH_VEC=16)
        WholeReduceSum(wordLo[vb * WORDS_PER_VEC], y, 8, 128, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        if (b * WORDS_PER_VEC > 128) {
            WholeReduceSum(wordLo[vb * WORDS_PER_VEC + 128], y[128 * 8], 8, b * WORDS_PER_VEC - 128, 1, 1, 1);
            PipeBarrier<PIPE_V>();
        }
        Mul(y, idxF, wtH, w);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(wordHi[vb * WORDS_PER_VEC], y, 8, 128, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        if (b * WORDS_PER_VEC > 128) {
            WholeReduceSum(wordHi[vb * WORDS_PER_VEC + 128], y[128 * 8], 8, b * WORDS_PER_VEC - 128, 1, 1, 1);
            PipeBarrier<PIPE_V>();
        }
    }

    // r_hat zTile tile -> rh/rl fp16 tiles into GM (A feed of the QJL matmul)
    __aicore__ inline void UploadRhat(int64_t g0, int32_t nv, const LocalTensor<float> &zTile)
    {
        // dedicated full-tile buffers: the 1KB BATCH_VEC scratch buffers here
        // overflowed by 4-8KB per tile and steamrolled normBF/gammaBF staging
        // (bisected 2026-09-26: fp32 residual bytes in the bf16 output slots)
        // rh|rl interleaved per-vec in ONE plane: a single 8KB upload copy and,
        // downstream, ONE 32-row A tile for the p matmul (halves the p pass)
        LocalTensor<half> rh = rhatRhBuf_.Get<half>();
        LocalTensor<half> rl = rhatRhBuf_.Get<half>()[TILE_VEC * HEAD_DIM];
        LocalTensor<float> back = rhatBackBuf_.Get<float>();
        LocalTensor<float> sub = rhatSubBuf_.Get<float>();
        const int32_t w = nv * HEAD_DIM;

        Cast(rh, zTile, RoundMode::CAST_RINT, w);
        PipeBarrier<PIPE_V>();
        Cast(back, rh, RoundMode::CAST_NONE, w);
        PipeBarrier<PIPE_V>();
        Sub(sub, zTile, back, w);
        PipeBarrier<PIPE_V>();
        Cast(rl, sub, RoundMode::CAST_RINT, w);
        PipeBarrier<PIPE_V>();
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(wsRhat2Gm_[g0 * 2 * HEAD_DIM], rh, 2 * w);
    }

    __aicore__ inline void UnpackIdxAll(int32_t nv, int32_t wordSet = 0, int32_t stageSet = 0)
    {
        LocalTensor<float> wordLo = wordIdxLoBuf_.Get<float>()[wordSet * TILE_VEC * WORDS_PER_VEC];
        LocalTensor<float> wordHi = wordIdxHiBuf_.Get<float>()[wordSet * TILE_VEC * WORDS_PER_VEC];
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>()[stageSet * TILE_VEC * MAX_IDX_BYTES];
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

    // per-line ATOMIC-domain flush for cube-DMA-written areas (z/p): the AIVs
    // only observe the fixpipe output after this
    __aicore__ inline void FlushCubeLines(const GlobalTensor<float> &gm, int64_t firstVec, int64_t bytes)
    {
        __gm__ float *base = const_cast<__gm__ float *>(gm.GetPhyAddr()) + firstVec * HEAD_DIM;
        const int64_t lines = (bytes + 63) / 64;
        GlobalTensor<int64_t> lineT;
        for (int64_t li = 0; li < lines; ++li) {
            lineT.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(base) + li * 8, 8);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ATOMIC>(lineT[0]);
        }
    }

    // dcci-flush a GM byte range: scalar stores (and MIX-mode DMA) sit in the
    // core-local cache and never become host-visible on this stack unless an
    // explicit clean+invalidate covers their lines (bisected 2026-09-26).
    template <typename T>
    __aicore__ inline void FlushOutLines(const GlobalTensor<T> &gm, int64_t elemOff, int64_t bytes)
    {
        __gm__ uint8_t *base =
            reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ T *>(gm.GetPhyAddr())) + elemOff * sizeof(T);
        const uint64_t aligned = reinterpret_cast<uint64_t>(base) & ~63ULL;
        const int64_t skip = reinterpret_cast<uint64_t>(base) - aligned;
        const int64_t lines = (bytes + skip + 63) / 64;
        GlobalTensor<int64_t> lineT;
        for (int64_t li = 0; li < lines; ++li) {
            lineT.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(aligned + li * 64), 8);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lineT[0]);
        }
    }

    __aicore__ inline void FlushScalar(int64_t g0, int32_t stageOff, int32_t nv,
                                       const LocalTensor<bfloat16_t> &normBF, const LocalTensor<bfloat16_t> &gammaBF,
                                       const LocalTensor<uint8_t> &idxStage, const LocalTensor<uint8_t> &qjlStage,
                                       bool phase1, bool phase2)
    {
        for (int32_t v = 0; v < nv; ++v) {
            const int64_t g = g0 + v;
            const int32_t s = stageOff + v;
            if (phase1) {
                normGm_.SetValue(g, normBF.GetValue(s));
                gammaGm_.SetValue(g, gammaBF.GetValue(s));
                for (int32_t b = 0; b < idxBytesPerVec_; ++b) {
                    idxGm_.SetValue(g * idxBytesPerVec_ + b, idxStage.GetValue(s * idxBytesPerVec_ + b));
                }
            }
            if (phase2) {
                for (int32_t b = 0; b < qjlBytesPerVec_; ++b) {
                    qjlOutGm_.SetValue(g * qjlBytesPerVec_ + b, qjlStage.GetValue(s * qjlBytesPerVec_ + b));
                }
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> zTileBuf_;
    TBuf<QuePosition::VECCALC> pTileBuf_;
    TBuf<QuePosition::VECCALC> xTileBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
    TBuf<QuePosition::VECCALC> yhatBuf_;
    TBuf<QuePosition::VECCALC> idxFBuf_;
    TBuf<QuePosition::VECCALC> pBuf_;
    TBuf<QuePosition::VECCALC> bvBuf_;
    TBuf<QuePosition::VECCALC> wtLBuf_;
    TBuf<QuePosition::VECCALC> wtHBuf_;
    TBuf<QuePosition::VECCALC> wordIdxLoBuf_;
    TBuf<QuePosition::VECCALC> wordIdxHiBuf_;
    TBuf<QuePosition::VECCALC> normVecBuf_;
    TBuf<QuePosition::VECCALC> rnVecBuf_;
    TBuf<QuePosition::VECCALC> gammaVecBuf_;
    TBuf<QuePosition::VECCALC> normBFBuf_;
    TBuf<QuePosition::VECCALC> gammaBFBuf_;
    TBuf<QuePosition::VECCALC> idxStageBuf_;
    TBuf<QuePosition::VECCALC> qjlStageBuf_;
    TBuf<QuePosition::VECCALC> rhatRhBuf_;
    TBuf<QuePosition::VECCALC> rhatRlBuf_;
    TBuf<QuePosition::VECCALC> rhatBackBuf_;
    TBuf<QuePosition::VECCALC> rhatSubBuf_;
    TBuf<QuePosition::VECCALC> xUpF32Buf_;
    TBuf<QuePosition::VECCALC> xUpHBuf_;
    TBuf<QuePosition::VECCALC> flagProbeBuf_;
    TBuf<QuePosition::VECCALC> redTmpBuf_;
    TBuf<QuePosition::VECCALC> splitBuf_;

    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;
    GlobalTensor<half> wsHhGm_;
    GlobalTensor<half> wsHlGm_;
    GlobalTensor<half> wsShGm_;
    GlobalTensor<half> wsSlGm_;
    GlobalTensor<float> wsZ1Gm_;
    GlobalTensor<float> wsZ2Gm_;
    GlobalTensor<half> wsRhat2Gm_;
    GlobalTensor<float> wsP1Gm_;
    GlobalTensor<float> wsP2Gm_;
    GlobalTensor<float> wsP3Gm_;
    GlobalTensor<float> wsP4Gm_;
    GlobalTensor<half> wsXhGm_;
    __gm__ int32_t *flagRaw_ = nullptr;
    volatile __gm__ int32_t *dbgMark_ = nullptr;
    GlobalTensor<int32_t> flagGm_;
    GlobalTensor<uint64_t> flagU64_;
    TBuf<TPosition::A1> l1ABuf_;
    TBuf<TPosition::A1> l1BBuf_;
    TBuf<TPosition::A2> l0ABuf_;
    TBuf<TPosition::B2> l0BBuf_;
    TBuf<TPosition::CO1> l0CBuf_;


    bool isAic_ = false;
    int64_t aivSubIdx_ = 0;
    int64_t logicalIdx_ = 0;
    int64_t totalVec_ = 0;
    int64_t qjlDim_ = 0;
    int32_t mseBits_ = 3;
    int32_t nCent_ = 8;
    int32_t idxBytesPerVec_ = 48;
    int32_t qjlBytesPerVec_ = 16;
    uint32_t coreNum_ = 1;
    int64_t rangeBegin_ = 0;
    int64_t rangeLen_ = 0;
    int64_t vecBegin_ = 0;
    int64_t vecNum_ = 0;

    TEventID eventVS_ = 0;
    TEventID eventSV_ = 0;
    TEventID eventMte2V_ = 0;
    TEventID eventVMte2_ = 0;
    TEventID eventVMte3_ = 0;
    TEventID eventMte3V_ = 0;
    TEventID eventMte3Mte2_ = 0;
    TEventID eventMte3S_ = 0;
    TEventID eventMte2Mte1_ = 0;
    TEventID eventMte1M_ = 0;
    TEventID eventMMte1_ = 0;
    TEventID eventMte1Mte2_ = 0;
    TEventID eventFixS_ = 0;
    TEventID eventFixM_ = 0;
    TEventID eventMFix_ = 0;
    TEventID eventMMte1B_ = 0;
    int32_t mmCtr_ = 0;
    int64_t prevG0_ = 0;
    int32_t prevNv_ = 0;

    float cent_[16] = {0.0f};
    float bound_[15] = {0.0f};
    float delta_[15] = {0.0f};
    float inv_[TILE_VEC] = {0.0f};
    float invR_[TILE_VEC] = {0.0f};
};

#endif // KV_CACHE_TURBO_QUANT_KERNEL_MIX_H_
