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
 * v6 streaming pipeline (2026-09-27):
 *  - Work is split into 128-row chunks assigned to tri-core groups round-robin
 *    (chunk c -> group c % 24), so every group streams fat contiguous bursts
 *    and all 24 AIC + 48 AIV stay concurrently active (the old whole-range
 *    phase barriers idled half the grid and throttled DMA to ~180GB/s).
 *  - AIC runs only cube matmul: z = (Hh+Hl) @ xh (L0C-accumulated, one fp32
 *    plane) and p1 = rh@Sh^T / p3 = rh@Sl^T / p2 = rl@Sh^T (three separate
 *    fp16 planes; L0C accumulation is NOT used for p - it loses the correction
 *    terms and flips QJL signs wholesale). H/S are fp16 hi/lo splits
 *    (combined split error ~2^-22, fp32-grade).
 *  - AIV runs everything else: norms + xh upload (fused pre-pass, x is read
 *    once), MSE quantize, residual norms, r_hat split upload, idx bit-pack via
 *    vcgadd block-reduce (replaces the 128-rep WholeReduceSum pack, ~5x less
 *    vector time), QJL sign, output flush.
 *  - GM spin-flag handshakes (XH/Z/R/P per chunk per group) synchronize the
 *    pipeline; waiters reset their cell after consuming so the next launch of
 *    a reused workspace starts clean.
 *  - No per-line dcci flushes: on this stack the fixpipe ATOMIC-domain writes
 *    are visible to partner AIV DataCopy (MTE2 snoop) after SyncPipe<FIX_S> +
 *    the flag store/load ordering (bisected 2026-09-27: byte-identical
 *    accuracy with and without the 16K-line flush loops, -35us).
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

    __aicore__ inline void Init(GM_ADDR kvVectors, GM_ADDR rotationMatrix, GM_ADDR qjlMatrix, GM_ADDR quantIdx,
                                GM_ADDR quantQjl, GM_ADDR quantNorm, GM_ADDR quantGamma, GM_ADDR workspace,
                                const KvCacheTurboQuantTilingData* tiling)
    {
        totalVec_ = tiling->totalVec;
        qjlDim_ = tiling->qjlDim;
        mseBits_ = tiling->mseBits;
        nCent_ = 1 << mseBits_;
        idxBytesPerVec_ = HEAD_DIM * mseBits_ / 8;
        qjlBytesPerVec_ = qjlDim_ / 8;
        coreNum_ = tiling->coreNum; // logical tri-core groups used

        flagRaw_ = reinterpret_cast<__gm__ int32_t*>(workspace);
        flagGm_.SetGlobalBuffer(flagRaw_, FLAG_ELEMS);
        flagU64_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(workspace), FLAG_ELEMS / 8);
        isAic_ = false;
        int64_t logicalIdx = 0;
        if ASCEND_IS_AIC {
            isAic_ = true;
            logicalIdx = GetBlockIdx();
        } else {
            logicalIdx = GetBlockIdx() / 2; // two AIVs share one logical group
        }
        logicalIdx_ = logicalIdx;
        aivSubIdx_ = isAic_ ? 0 : (GetBlockIdx() % 2);

        // round-robin chunk assignment: group g owns chunks {g, g+coreNum, ...}
        chunkNum_ = (totalVec_ + CHUNK_ROWS - 1) / CHUNK_ROWS;
        myChunks_ = (logicalIdx < chunkNum_) ? ((chunkNum_ - 1 - logicalIdx) / static_cast<int64_t>(coreNum_) + 1) : 0;

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(kvVectors), totalVec_ * HEAD_DIM);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotationMatrix), HEAD_DIM * HEAD_DIM);
        qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(qjlMatrix), qjlDim_ * HEAD_DIM);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(quantIdx), totalVec_ * idxBytesPerVec_);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(quantQjl), totalVec_ * qjlBytesPerVec_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(quantNorm), totalVec_);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(quantGamma), totalVec_);

        // workspace layout (bytes), wsVec = totalVec padded up to one chunk:
        //   0..264K     cross-core GM spin flags (64B-strided int32 cells)
        //   264K..424K  Hh/Hl/Hll bf16-triple (3 x 32KB) + Sh/Sl fp16 (2 x 32KB)
        //   424K        z fp32            wsVec x 512B   (cube plane)
        //   then        r_hat [rh|rl]     wsVec x 512B   (fp16, chunk-major)
        //   then        p1/p2/p3 fp16 x3  wsVec x 256B   (cube planes)
        // (v10: x is read DIRECTLY by the AIC in bf16 - no xh upload plane,
        // and bf16 splits capture fp32 H exactly (3 x 8-bit mantissa = 24))
        wsVec_ = chunkNum_ * CHUNK_ROWS;
        __gm__ uint8_t* wsSplit = workspace + WS_SPLIT_OFF;
        wsHhGmB_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wsSplit + 0 * SPLIT_BYTES), SPLIT_ELEMS);
        wsHlGmB_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wsSplit + 1 * SPLIT_BYTES), SPLIT_ELEMS);
        wsHllGmB_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wsSplit + 2 * SPLIT_BYTES), SPLIT_ELEMS);
        wsShGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wsSplit + 3 * SPLIT_BYTES), SPLIT_ELEMS);
        wsSlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wsSplit + 4 * SPLIT_BYTES), SPLIT_ELEMS);
        __gm__ uint8_t* wsB = workspace + WS_Z_OFF;
        wsZGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsB), wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 4;
        wsRhat2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wsB), 2 * wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 2 * 2;
        wsP1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wsB), wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 2;
        wsP2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wsB), wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 2;
        wsP3Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wsB), wsVec_ * HEAD_DIM);

        if (isAic_) {
            pipe_.InitBuffer(l1ABuf_, L1_A_BYTES);
            pipe_.InitBuffer(l1BBuf_, L1_B_BYTES);
            pipe_.InitBuffer(l0ABuf_, L0_A_BYTES);
            pipe_.InitBuffer(l0BBuf_, L0_B_BYTES);
            pipe_.InitBuffer(l0CBuf_, L0_C_BYTES);
            // static event ids: the MIX runtime + the wrapper's workspace wipe
            // use the low (0-2) dynamic range; QUE_MAX_EVENT is only 8 and the
            // pool is shared - per-launch FetchEventID contributes to the repeat
            // crash. Static ids in the free 4-7 range avoid both parties.
            eventMte2Mte1_ = 4;
            eventMte1M_ = 4;
            eventMMte1_ = 4;
            eventMMte1B_ = 5;
            eventMte1Mte2_ = 4;
            eventFixS_ = 4;
            eventMFix_ = 4;
            eventFixM_ = 4;
            eventFixM2_ = 5;
        } else {
            eventVS_ = 4;
            eventSV_ = 4;
            eventMte2V_ = 4;
            eventVMte2_ = 4;
            eventVMte3_ = 4;
            eventMte3V_ = 4;
            eventMte3Mte2_ = 4;
            eventMte3S_ = 4;
            eventSMte3_ = 4;
            eventMte2V2_ = 5;
            pipe_.InitBuffer(zTileBuf_, 2 * TILE_VEC * HEAD_DIM * sizeof(float)); // prefetch pair
            pipe_.InitBuffer(pTileBuf_, TILE_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(xTileBuf_, 2 * TILE_VEC * HEAD_DIM * sizeof(bfloat16_t));
            pipe_.InitBuffer(pRdBuf_, 2 * 3 * TILE_VEC * HEAD_DIM * sizeof(half)); // p1/2/3 x 2 sets
            pipe_.InitBuffer(flagProbeBuf_, 64);
            pipe_.InitBuffer(yBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(yhatBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(idxFBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
            pipe_.InitBuffer(pBuf_, BATCH_VEC * HEAD_DIM * sizeof(float));
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
            pipe_.InitBuffer(tableBuf_, 32 * sizeof(float));
            pipe_.InitBuffer(maskBuf_, TILE_VEC * HEAD_DIM / 8 + 32);
            pipe_.InitBuffer(packOffBuf_, 256 * sizeof(uint32_t));
            pipe_.InitBuffer(onesBuf_, TILE_VEC * sizeof(float));
            pipe_.InitBuffer(splitBuf_, SPLIT_STAGE_BYTES);
            // per-row norm / reciprocal persistence across the pipeline phases
            // (aliases the split staging area, dead after phase 0)
            invAll_ = splitBuf_.Get<float>()[0];
            normAll_ = splitBuf_.Get<float>()[MAX_ROWS_PER_AIV];
        }

        if (!isAic_) {
            InitConstants();
        }
    }

    __aicore__ inline void Process()
    {
        if (totalVec_ <= 0 || myChunks_ <= 0) {
            return;
        }
        if (isAic_) {
            const int32_t slotBase = static_cast<int32_t>(logicalIdx_) * GROUP_FLAG_CELLS * SLOT_STRIDE;
            // ---- phase Z (AIC): z = (Hh+Hl) @ xh per chunk; the fused z/p
            // loop (build252) was SLOWER - the per-chunk L0B H<->S reload churn
            // and tighter handoffs cost more than the boundary overlap saved
            WaitXh(slotBase, 0);   // splits done (single handshake for the launch)
            LoadSplitToL1Triple(); // H bf16-triple: [Hh|Hl] resident, Hll staged in L1
            for (int64_t l = 0; l < myChunks_; ++l) {
                const int64_t c = ChunkId(l);
                const int64_t rows = ChunkRows(c);
                const int64_t qtiles = (rows + QTILE_ROWS - 1) / QTILE_ROWS;
                for (int64_t q = 0; q < qtiles; ++q) {
                    const int64_t g = c * CHUNK_ROWS + q * QTILE_ROWS;
                    const int64_t rem = rows - q * QTILE_ROWS;
                    const int32_t nv = static_cast<int32_t>(rem < QTILE_ROWS ? rem : QTILE_ROWS);
                    MmTileQ(kvGm_[g * HEAD_DIM], wsZGm_[g * HEAD_DIM], nv); // x read DIRECTLY (bf16)
                }
                DrainFixPend();
                SyncPipe<HardEvent::FIX_S>(); // z chunk drained before the flag store
                GmFlagSet(FlagCell(slotBase, l, SLOT_Z_LO), MAGIC_Z);
                GmFlagSet(FlagCell(slotBase, l, SLOT_Z_HI), MAGIC_Z);
            }
            // ---- phase P (AIC): p1 = rh@Sh, p3 = rh@Sl, p2 = rl@Sh per chunk
            LoadSplitToL1(wsShGm_, wsSlGm_);
            for (int64_t l = 0; l < myChunks_; ++l) {
                GmFlagWait(FlagCell(slotBase, l, SLOT_R_LO), MAGIC_R);
                GmFlagWait(FlagCell(slotBase, l, SLOT_R_HI), MAGIC_R);
                const int64_t c = ChunkId(l);
                const int64_t rows = ChunkRows(c);
                const int64_t qtiles = (rows + QTILE_ROWS - 1) / QTILE_ROWS;
                const int64_t rhBase = c * 2 * CHUNK_ROWS * HEAD_DIM;
                for (int64_t q = 0; q < qtiles; ++q) {
                    const int64_t g = c * CHUNK_ROWS + q * QTILE_ROWS;
                    const int64_t rem = rows - q * QTILE_ROWS;
                    const int32_t nv = static_cast<int32_t>(rem < QTILE_ROWS ? rem : QTILE_ROWS);
                    MmTilePQ(wsRhat2Gm_[rhBase + q * QTILE_ROWS * HEAD_DIM],
                             wsRhat2Gm_[rhBase + (CHUNK_ROWS + q * QTILE_ROWS) * HEAD_DIM], wsP1Gm_[g * HEAD_DIM],
                             wsP2Gm_[g * HEAD_DIM], wsP3Gm_[g * HEAD_DIM], nv);
                }
                DrainFixPend();
                SyncPipe<HardEvent::FIX_S>();
                GmFlagSet(FlagCell(slotBase, l, SLOT_P_LO), MAGIC_P);
                GmFlagSet(FlagCell(slotBase, l, SLOT_P_HI), MAGIC_P);
            }
            // ---- AIC cube epilogue: drain every cube pipe and reset the
            // load/atomic state so the NEXT launch's MIX runtime init sees a
            // clean AIC. Without this the cube state persists across launches
            // and a reused (executor, workspace) pair dies at grid init
            // (507015/507057) before any user code - the repeat-launch crash.
            // Scoped to the cube path so the pure-vector decode path (idle AIC)
            // is untouched.
            // EXPERIMENT (build251): epilogue drain/reset removed - measuring
            // whether this block is the ~180us device-side overhead. Repeats
            // rely on the per-call executor contract regardless.
            SyncPipe<HardEvent::FIX_S>();
        } else {
            ProcessAiv();
        }
    }

private:
    static constexpr int32_t HEAD_DIM = 128;
    static constexpr int32_t TILE_VEC = 16;
    static constexpr int32_t BATCH_VEC = 16;
    static constexpr int32_t WORDS_PER_VEC = 16;
    static constexpr int32_t MAX_IDX_BYTES = 64;
    static constexpr int32_t RED_TMP_ELEMS = 2048; // sg scratch spans a full 16x128 batch
    static constexpr int32_t SPLIT_ELEMS = HEAD_DIM * HEAD_DIM;
    static constexpr int32_t SPLIT_STAGE_BYTES = 28 * 1024;                         // 2048-elem split chunks
    static constexpr int32_t CHUNK_TILES = 8;                                       // pipeline chunk = 8 tiles
    static constexpr int32_t CHUNK_ROWS = CHUNK_TILES * TILE_VEC;                   // 128 vectors
    static constexpr int32_t QTILE_ROWS = 64;                                       // AIC quad-tile (m=64 Mmad)
    static constexpr int32_t HALF_CHUNK_ROWS = CHUNK_ROWS / 2;                      // per partner AIV
    static constexpr int32_t MAX_LOCAL_CHUNKS = 22;                                 // covers totalVec <= 67584
    static constexpr int32_t MAX_ROWS_PER_AIV = MAX_LOCAL_CHUNKS * HALF_CHUNK_ROWS; // 1408
    static constexpr int32_t L1_A_BYTES = 4 * QTILE_ROWS * HEAD_DIM * sizeof(half); // 4 x 16K
    static constexpr int32_t L1_B_BYTES = 3 * HEAD_DIM * HEAD_DIM * sizeof(half);   // 96K: bf16-triple H (S pair fits)
    static constexpr int32_t L0_A_BYTES = 4 * QTILE_ROWS * HEAD_DIM * sizeof(half); // 4 x 16K
    static constexpr int32_t L0_B_BYTES = 2 * HEAD_DIM * HEAD_DIM * sizeof(half);   // 64K: hi+lo splits
    static constexpr int32_t L0_C_BYTES = QTILE_ROWS * HEAD_DIM * sizeof(float);    // 32K
    // flag cells: 24 groups x 22 chunks x 8 slots x 16-elem stride
    static constexpr int32_t SLOTS_PER_CHUNK = 8;
    static constexpr int32_t SLOT_XH_LO = 0;
    static constexpr int32_t SLOT_XH_HI = 1;
    static constexpr int32_t SLOT_Z_LO = 2;
    static constexpr int32_t SLOT_Z_HI = 3;
    static constexpr int32_t SLOT_R_LO = 4;
    static constexpr int32_t SLOT_R_HI = 5;
    static constexpr int32_t SLOT_P_LO = 6;
    static constexpr int32_t SLOT_P_HI = 7;
    static constexpr int32_t GROUP_FLAG_CELLS = MAX_LOCAL_CHUNKS * SLOTS_PER_CHUNK; // 176
    static constexpr int32_t SLOT_STRIDE = 16;                                      // int32 elems per 64B slot
    static constexpr int32_t FLAG_ELEMS = 24 * GROUP_FLAG_CELLS * SLOT_STRIDE;      // 67584 (264KB)
    static constexpr int32_t SPLIT_BYTES = HEAD_DIM * HEAD_DIM * 2;                 // 32KB per split plane
    static constexpr int32_t WS_SPLIT_OFF = 264 * 1024;
    static constexpr int32_t WS_Z_OFF = (264 + 160) * 1024; // 5 split planes
    static constexpr float MIN_DENOM = 1e-30f;
    static constexpr float IND_SCALE = 1.2676506002282294e30f;
    static constexpr int32_t MAGIC_XH = 0x5A5A0001;
    static constexpr int32_t MAGIC_Z = 0x5A5A0003;
    static constexpr int32_t MAGIC_R = 0x5A5A0004;
    static constexpr int32_t MAGIC_P = 0x5A5A0006;

    __aicore__ inline int64_t ChunkId(int64_t l) const
    {
        return logicalIdx_ + l * static_cast<int64_t>(coreNum_);
    }

    __aicore__ inline int64_t ChunkRows(int64_t c) const
    {
        const int64_t rem = totalVec_ - c * CHUNK_ROWS;
        return rem < CHUNK_ROWS ? rem : CHUNK_ROWS;
    }

    static __aicore__ inline int32_t FlagCell(int32_t slotBase, int64_t l, int32_t slot)
    {
        return slotBase + static_cast<int32_t>(l * SLOTS_PER_CHUNK + slot) * SLOT_STRIDE;
    }

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
    __aicore__ inline void GmFlagSet(int32_t slotElem, int32_t magic)
    {
        if (isAic_) {
            volatile __gm__ int32_t* slot = flagRaw_ + slotElem;
            *slot = magic;
            __asm__ __volatile__("" ::: "memory");
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(flagGm_[slotElem]);
            __asm__ __volatile__("" ::: "memory");
        } else {
            // AIV: dcci on dav_m200 poisons the core's later GM writes (the
            // write-poison ghost - bisected 2026-09-26). The flag cell owns
            // its whole 64B line (SLOT_STRIDE=16), so the set is a plain UB
            // line store + MTE3 DataCopy; MTE3 is FIFO-ordered after this
            // core's data uploads, so the flag implies data-visible with no
            // extra syncs (the old MTE2-in/modify/MTE3-out dance was ~2us).
            LocalTensor<int32_t> probe = flagProbeBuf_.Get<int32_t>();
            SyncPipe<HardEvent::MTE3_S>(); // prior probe read-back drained
            probe.SetValue(slotElem & 15, magic);
            SyncPipe<HardEvent::S_MTE3>();
            DataCopy(flagGm_[slotElem & ~15], probe, 16);
        }
    }

    __aicore__ inline void GmFlagWait(int32_t slotElem, int32_t magic)
    {
        volatile __gm__ int32_t* slot = flagRaw_ + slotElem;
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
            // 2026-09-26) - poll the flag line via an MTE2 DMA read instead:
            // the copy snoops L2 and leaves this core's write path intact
            LocalTensor<int32_t> probe = flagProbeBuf_.Get<int32_t>();
            SyncPipe<HardEvent::MTE3_MTE2>(); // prior fast flag-set's MTE3 read of probe drained
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
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(flagGm_[slotElem]);
            __asm__ __volatile__("" ::: "memory");
        } else {
            // reset after consume: same MTE3 line store (cell-owned line)
            LocalTensor<int32_t> probe = flagProbeBuf_.Get<int32_t>();
            SyncPipe<HardEvent::MTE3_S>();
            probe.SetValue(slotElem & 15, 0);
            SyncPipe<HardEvent::S_MTE3>();
            DataCopy(flagGm_[slotElem & ~15], probe, 16);
        }
    }

    __aicore__ inline void WaitXh(int32_t slotBase, int64_t l)
    {
        GmFlagWait(FlagCell(slotBase, l, SLOT_XH_LO), MAGIC_XH);
        GmFlagWait(FlagCell(slotBase, l, SLOT_XH_HI), MAGIC_XH);
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
        } else if constexpr (evt == HardEvent::S_MTE3) {
            return eventSMte3_;
        } else {
            return eventMte3V_;
        }
    }

    __aicore__ inline void InitConstants()
    {
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
        for (int32_t i = 0; i < nCent_ - 1; ++i) {
            bound_[i] = (cent_[i] + cent_[i + 1]) * 0.5f;
            delta_[i] = cent_[i + 1] - cent_[i];
        }
        // binary-search tables: centroids at [0,16), mid-bounds at [16,32)
        // (strict-greater midpoint walk - the same decisions as golden's
        // searchsorted; yhat comes from a table gather, not a gap sum)
        LocalTensor<float> table = tableBuf_.Get<float>();
        for (int32_t i = 0; i < 16; ++i) {
            table.SetValue(i, i < nCent_ ? cent_[i] : 0.0f);
            table.SetValue(16 + i, i < nCent_ - 1 ? bound_[i] : 0.0f);
        }
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
        // pack regroup offsets (byte offsets into the merged int32 words):
        //   mse2: evenOff[i]=8i, oddOff[i]=8i+4 (i<128)
        //   mse3: g0..g3[i]=16i+{0,4,8,12} (i<64)
        LocalTensor<uint32_t> po = packOffBuf_.Get<uint32_t>();
        if (mseBits_ == 2) {
            for (int32_t i = 0; i < TILE_VEC * WORDS_PER_VEC / 2; ++i) {
                po.SetValue(i, i * 8);
                po.SetValue(TILE_VEC * WORDS_PER_VEC / 2 + i, i * 8 + 4);
            }
        } else if (mseBits_ == 3) {
            for (int32_t i = 0; i < 64; ++i) {
                po.SetValue(i, i * 16);
                po.SetValue(64 + i, i * 16 + 4);
                po.SetValue(128 + i, i * 16 + 8);
                po.SetValue(192 + i, i * 16 + 12);
            }
        }
        SyncPipe<HardEvent::S_V>();
    }

    // fp32 H/S -> fp16 hi/lo splits to GM (every AIV writes identical content;
    // a core's own writes guarantee the content its partner AIC reads is valid)
    // fp32 H -> bf16 TRIPLE split (Hh+Hl+Hll captures fp32 exactly: 3x8-bit
    // mantissa = 24 bits; offline-verified 0.0 residual and 0 idx flips) and
    // fp32 S -> fp16 hi/lo pair. Every AIV writes identical content (benign
    // race); a pair's XH(0) flag then implies the splits are complete for its
    // AIC. The AIC reads x DIRECTLY as bf16 (dtype-matched) - no xh upload.
    __aicore__ inline void SplitMatrices()
    {
        LocalTensor<float> chunk = splitBuf_.Get<float>();                   // 8KB fp32
        LocalTensor<float> backC = splitBuf_.Get<float>()[2 * 1024];         // 8KB
        LocalTensor<float> subC = splitBuf_.Get<float>()[4 * 1024];          // 8KB
        LocalTensor<half> hC = splitBuf_.Get<half>()[12 * 1024];             // 4KB fp16
        LocalTensor<bfloat16_t> bC = splitBuf_.Get<bfloat16_t>()[14 * 1024]; // 4KB bf16
        // H: bf16-triple
        for (int32_t c = 0; c < SPLIT_ELEMS / 2048; ++c) {
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(chunk, rotGm_[c * 2048], 2048);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::MTE3_V>(); // bC free: previous Hll copy drained
            Cast(bC, chunk, RoundMode::CAST_RINT, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(wsHhGmB_[c * 2048], bC, 2048);
            Cast(backC, bC, RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();
            Sub(subC, chunk, backC, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::MTE3_V>(); // wsHh copy done reading bC
            Cast(bC, subC, RoundMode::CAST_RINT, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(wsHlGmB_[c * 2048], bC, 2048);
            Cast(backC, bC, RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();
            Sub(subC, subC, backC, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::MTE3_V>(); // wsHl copy done reading bC
            Cast(bC, subC, RoundMode::CAST_RINT, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(wsHllGmB_[c * 2048], bC, 2048);
        }
        // S: fp16 hi/lo pair (unchanged)
        for (int32_t c = 0; c < SPLIT_ELEMS / 2048; ++c) {
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(chunk, qjlGm_[c * 2048], 2048);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::MTE3_V>();
            Cast(hC, chunk, RoundMode::CAST_RINT, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(wsShGm_[c * 2048], hC, 2048);
            Cast(backC, hC, RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();
            Sub(subC, chunk, backC, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::MTE3_V>();
            Cast(hC, subC, RoundMode::CAST_RINT, 2048);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(wsSlGm_[c * 2048], hC, 2048);
        }
        SyncPipe<HardEvent::MTE3_V>();
    }

    // ---- phase 2 (AIV): y = z * inv, MSE quantize, residual norms, r_hat
    // split upload, idx pack, gamma output - per chunk as Z arrives
    // ---- phase 2 (AIV): norms from z (H orthogonal => ||z||==||x||), y = z*inv,
    // MSE quantize, residual norms, r_hat split upload, idx pack, gamma + norm
    // output - per chunk as Z arrives; z reads prefetched one tile ahead
    __aicore__ inline void QuantChunk(int64_t l)
    {
        const int64_t c = ChunkId(l);
        const int64_t rows = ChunkRows(c);
        const int64_t halfBegin = c * CHUNK_ROWS + aivSubIdx_ * HALF_CHUNK_ROWS;
        int64_t myRows = rows - aivSubIdx_ * HALF_CHUNK_ROWS;
        if (myRows > HALF_CHUNK_ROWS) {
            myRows = HALF_CHUNK_ROWS;
        }
        if (myRows <= 0) {
            GmFlagSet(FlagCell(SlotBase(), l, aivSubIdx_ == 0 ? SLOT_R_LO : SLOT_R_HI), MAGIC_R);
            return;
        }
        LocalTensor<float> zBase = zTileBuf_.Get<float>();
        LocalTensor<float> normVec = normVecBuf_.Get<float>();
        LocalTensor<float> rnVec = rnVecBuf_.Get<float>();
        LocalTensor<float> gammaVec = gammaVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> normBF = normBFBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> gammaBF = gammaBFBuf_.Get<bfloat16_t>();
        LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();
        LocalTensor<float> sqTmp = yhatBuf_.Get<float>();
        const int64_t nt = (myRows + TILE_VEC - 1) / TILE_VEC;

        // prefetch tile 0's z read; the loop keeps MTE2 one tile ahead
        {
            const int32_t nv0 = static_cast<int32_t>(myRows < TILE_VEC ? myRows : TILE_VEC);
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(zBase, wsZGm_[halfBegin * HEAD_DIM], nv0 * HEAD_DIM);
            SetFlag<HardEvent::MTE2_V>(eventMte2V_);
        }
        for (int64_t t = 0; t < myRows; t += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((myRows - t) < TILE_VEC ? (myRows - t) : TILE_VEC);
            const int64_t g0 = halfBegin + t;
            const int32_t bufSel = static_cast<int32_t>(t / TILE_VEC) & 1;
            if (t + TILE_VEC < myRows) {
                const int64_t gN = g0 + TILE_VEC;
                const int64_t remN = myRows - t - TILE_VEC;
                const int32_t nvN = static_cast<int32_t>(remN < TILE_VEC ? remN : TILE_VEC);
                SyncPipe<HardEvent::V_MTE2>();
                DataCopy(zBase[(bufSel ^ 1) * TILE_VEC * HEAD_DIM], wsZGm_[gN * HEAD_DIM], nvN * HEAD_DIM);
                SetFlag<HardEvent::MTE2_V>((bufSel ^ 1) == 0 ? eventMte2V_ : eventMte2V2_);
            }
            WaitFlag<HardEvent::MTE2_V>(bufSel == 0 ? eventMte2V_ : eventMte2V2_);
            LocalTensor<float> zTile = zBase[bufSel * TILE_VEC * HEAD_DIM];

            // row norms from z (fp32): ||z||_2 = ||x||_2 to ~2^-24
            LocalTensor<float> y = yBuf_.Get<float>();
            Mul(sqTmp, zTile, zTile, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(normVec, sqTmp, 64, nv, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(xBuf2(), sqTmp[64], 64, nv, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            Add(normVec, normVec, xBuf2(), nv);
            PipeBarrier<PIPE_V>();
            Sqrt(normVec, normVec, nv);
            PipeBarrier<PIPE_V>();
            // inv per row (scalar correctly-rounded reciprocal; error class 2^-24)
            SyncPipe<HardEvent::V_S>();
            for (int32_t v = 0; v < nv; ++v) {
                const float norm = normVec.GetValue(v);
                inv_[v] = (norm > 0.0f) ? 1.0f / ((norm < MIN_DENOM) ? MIN_DENOM : norm) : 0.0f;
            }
            SyncPipe<HardEvent::S_V>();
            for (int32_t i = 0; i < nv; ++i) {
                Muls(y[i * HEAD_DIM], zTile[i * HEAD_DIM], inv_[i], HEAD_DIM);
            }
            PipeBarrier<PIPE_V>();
            QuantizeBatch(0, nv, zTile, rnVec, y);
            Sqrt(rnVec, rnVec, nv);
            PipeBarrier<PIPE_V>();
            Mul(gammaVec, normVec, rnVec, nv);
            PipeBarrier<PIPE_V>();
            Cast(gammaBF, gammaVec, RoundMode::CAST_RINT, nv);
            PipeBarrier<PIPE_V>();
            Cast(normBF, normVec, RoundMode::CAST_RINT, nv);
            PipeBarrier<PIPE_V>();

            SyncPipe<HardEvent::V_S>();
            for (int32_t v = 0; v < nv; ++v) {
                const float rn = rnVec.GetValue(v);
                invR_[v] = (rn > 0.0f) ? 1.0f / ((rn < MIN_DENOM) ? MIN_DENOM : rn) : 0.0f;
            }
            PackIdxTile(nv, bufSel);
            SyncPipe<HardEvent::S_V>();

            // ---- r_hat (in place on the residual) -> fp16 hi/lo splits to GM
            for (int32_t v = 0; v < nv; ++v) {
                Muls(zTile[v * HEAD_DIM], zTile[v * HEAD_DIM], invR_[v], HEAD_DIM);
            }
            PipeBarrier<PIPE_V>();
            UploadRhat(c, aivSubIdx_ * HALF_CHUNK_ROWS + t, nv, zTile, bufSel);

            LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>()[bufSel * TILE_VEC * MAX_IDX_BYTES];
            SyncPipe<HardEvent::V_MTE3>();
            if (nv == TILE_VEC) {
                DataCopy(normGm_[g0], normBF, nv);
                DataCopy(gammaGm_[g0], gammaBF, nv);
                DataCopy(idxGm_[g0 * idxBytesPerVec_], idxStage, nv * idxBytesPerVec_);
            } else {
                SyncPipe<HardEvent::MTE3_V>();
                FlushScalar(g0, 0, nv, normBF, gammaBF, idxStage, qjlStage, true, false);
                FlushOutLines(normGm_, g0, nv * 2);
                FlushOutLines(gammaGm_, g0, nv * 2);
                FlushOutLines(idxGm_, g0 * idxBytesPerVec_, nv * idxBytesPerVec_);
            }
            SyncPipe<HardEvent::MTE3_V>();
        }
        SyncPipe<HardEvent::MTE3_S>();
        GmFlagSet(FlagCell(SlotBase(), l, aivSubIdx_ == 0 ? SLOT_R_LO : SLOT_R_HI), MAGIC_R);
    }

    // ---- phase 3 (AIV): p1+p2+p3 (fp32 sum) -> sign -> qjl pack, per chunk
    // ---- phase 3 (AIV): p1+p2+p3 (fp32 sum) -> sign -> qjl pack, per chunk
    // as P arrives; the three plane reads are prefetched one tile ahead
    __aicore__ inline void SignChunk(int64_t l)
    {
        const int64_t c = ChunkId(l);
        const int64_t rows = ChunkRows(c);
        const int64_t halfBegin = c * CHUNK_ROWS + aivSubIdx_ * HALF_CHUNK_ROWS;
        int64_t myRows = rows - aivSubIdx_ * HALF_CHUNK_ROWS;
        if (myRows > HALF_CHUNK_ROWS) {
            myRows = HALF_CHUNK_ROWS;
        }
        if (myRows <= 0) {
            return;
        }
        LocalTensor<float> pAll = zTileBuf_.Get<float>();
        LocalTensor<half> pRdBase = pRdBuf_.Get<half>();
        LocalTensor<float> pCorr = pTileBuf_.Get<float>();
        LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();
        LocalTensor<bfloat16_t> normBF = normBFBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> gammaBF = gammaBFBuf_.Get<bfloat16_t>();
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>();

        // prefetch tile 0's three planes
        {
            const int32_t nv0 = static_cast<int32_t>(myRows < TILE_VEC ? myRows : TILE_VEC);
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(pRdBase, wsP1Gm_[halfBegin * HEAD_DIM], nv0 * HEAD_DIM);
            DataCopy(pRdBase[TILE_VEC * HEAD_DIM], wsP2Gm_[halfBegin * HEAD_DIM], nv0 * HEAD_DIM);
            DataCopy(pRdBase[2 * TILE_VEC * HEAD_DIM], wsP3Gm_[halfBegin * HEAD_DIM], nv0 * HEAD_DIM);
            SetFlag<HardEvent::MTE2_V>(eventMte2V_);
        }

        for (int64_t t = 0; t < myRows; t += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((myRows - t) < TILE_VEC ? (myRows - t) : TILE_VEC);
            const int64_t g0 = halfBegin + t;
            const int32_t bufSel = static_cast<int32_t>(t / TILE_VEC) & 1;
            if (t + TILE_VEC < myRows) {
                const int64_t gN = g0 + TILE_VEC;
                const int64_t remN = myRows - t - TILE_VEC;
                const int32_t nvN = static_cast<int32_t>(remN < TILE_VEC ? remN : TILE_VEC);
                LocalTensor<half> pRdN = pRdBase[(bufSel ^ 1) * 3 * TILE_VEC * HEAD_DIM];
                SyncPipe<HardEvent::V_MTE2>();
                DataCopy(pRdN, wsP1Gm_[gN * HEAD_DIM], nvN * HEAD_DIM);
                DataCopy(pRdN[TILE_VEC * HEAD_DIM], wsP2Gm_[gN * HEAD_DIM], nvN * HEAD_DIM);
                DataCopy(pRdN[2 * TILE_VEC * HEAD_DIM], wsP3Gm_[gN * HEAD_DIM], nvN * HEAD_DIM);
                SetFlag<HardEvent::MTE2_V>((bufSel ^ 1) == 0 ? eventMte2V_ : eventMte2V2_);
            }
            WaitFlag<HardEvent::MTE2_V>(bufSel == 0 ? eventMte2V_ : eventMte2V2_);
            LocalTensor<half> pRd = pRdBase[bufSel * 3 * TILE_VEC * HEAD_DIM];

            // p = cast(p1 fp16) + cast(p2 fp16) + cast(p3 fp16), summed in fp32
            Cast(pAll, pRd, RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Cast(pCorr, pRd[TILE_VEC * HEAD_DIM], RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Add(pAll, pAll, pCorr, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Cast(pCorr, pRd[2 * TILE_VEC * HEAD_DIM], RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Add(pAll, pAll, pCorr, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();

            Compares(qjlStage, pAll, 0.0f, CMPMODE::GE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();

            SyncPipe<HardEvent::V_MTE3>();
            // CRITICAL: scalar reads of the Compare (V-pipe) bitmask need V_S
            SyncPipe<HardEvent::V_S>();
            if (nv == TILE_VEC) {
                DataCopy(qjlOutGm_[g0 * qjlBytesPerVec_], qjlStage, nv * qjlBytesPerVec_);
            } else {
                FlushScalar(g0, 0, nv, normBF, gammaBF, idxStage, qjlStage, false, true);
                FlushOutLines(qjlOutGm_, g0 * qjlBytesPerVec_, nv * qjlBytesPerVec_);
            }
            SyncPipe<HardEvent::MTE3_V>();
            SyncPipe<HardEvent::S_V>();
        }
    }

    // AIV-side full flow: splits -> quant -> sign. v11: the rotation matrix is
    // ORTHOGONAL (TurboQuant requires it), so ||z|| == ||H x|| == ||x|| to
    // ~2^-24 - row norms are computed from z inside the quant phase and the
    // pre-pass phase (x read, xh upload, norm persistence, XH handshake) is
    // DELETED: two phases total, no cross-phase state.
    __aicore__ inline void ProcessAiv()
    {
        SplitMatrices();
        // splits-done signal (this pair's AIC waits both cells once at start)
        if (myChunks_ > 0) {
            GmFlagSet(FlagCell(SlotBase(), 0, aivSubIdx_ == 0 ? SLOT_XH_LO : SLOT_XH_HI), MAGIC_XH);
        }
        const int32_t zSlot = aivSubIdx_ == 0 ? SLOT_Z_LO : SLOT_Z_HI;
        const int32_t pSlot = aivSubIdx_ == 0 ? SLOT_P_LO : SLOT_P_HI;
        for (int64_t l = 0; l < myChunks_; ++l) {
            GmFlagWait(FlagCell(SlotBase(), l, zSlot), MAGIC_Z);
            QuantChunk(l);
        }
        for (int64_t l = 0; l < myChunks_; ++l) {
            GmFlagWait(FlagCell(SlotBase(), l, pSlot), MAGIC_P);
            SignChunk(l);
        }
    }

    __aicore__ inline int32_t SlotBase() const
    {
        return static_cast<int32_t>(logicalIdx_) * GROUP_FLAG_CELLS * SLOT_STRIDE;
    }

    // scratch fp32 row buffer for the second WRS half (aliases redTmp)
    __aicore__ inline LocalTensor<float> xBuf2()
    {
        return redTmpBuf_.Get<float>();
    }

    // ---- low-level fp16 matmul service (AIC only): z = A @ B^T per 16x128 tile ----
    // ND (row-major) GM tile -> NZ L1; rowAct bounds the GM read so tail tiles
    // never touch input rows past totalVec (pad rows compute garbage in L0,
    // which the AIV never consumes)
    template <typename T>
    __aicore__ inline void CopyGmNDToL1(const LocalTensor<T>& l1Tensor, const GlobalTensor<T>& gmTensor,
                                        uint32_t rowAct, uint32_t colStride)
    {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = rowAct;
        nd2nzPara.srcDValue = colStride;
        nd2nzPara.dValue = HEAD_DIM;
        nd2nzPara.dstNzC0Stride = (rowAct + 15) / 16 * 16;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.dstNzMatrixStride = 0;
        nd2nzPara.srcNdMatrixStride = 0;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    }

    // load one split matrix pair [128,128] x 2 into L1 B and L0 B (resident)
    __aicore__ inline void LoadSplitToL1(const GlobalTensor<half>& ws0, const GlobalTensor<half>& ws1)
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

    // ---- L0C double-buffer service: alternate between two 32KB L0C tiles and
    // defer the FIX_M wait until the buffer's NEXT reuse, so fixpipe(t) drains
    // while tile t+1 loads/multiplies (the serial Mmad->fix->wait chain idled
    // the cube DMA ~half the time)
    __aicore__ inline int32_t L0CAlloc()
    {
        const int32_t cSel = fixCtr_ & 1;
        ++fixCtr_;
        if (fixPend_[cSel]) {
            WaitFlag<HardEvent::FIX_M>((cSel == 0) ? eventFixM_ : eventFixM2_);
            fixPend_[cSel] = false;
        }
        return cSel;
    }

    __aicore__ inline void L0CFixDone(int32_t cSel)
    {
        SetFlag<HardEvent::FIX_M>((cSel == 0) ? eventFixM_ : eventFixM2_);
        fixPend_[cSel] = true;
    }

    __aicore__ inline void DrainFixPend()
    {
        for (int32_t b = 0; b < 2; ++b) {
            if (fixPend_[b]) {
                WaitFlag<HardEvent::FIX_M>((b == 0) ? eventFixM_ : eventFixM2_);
                fixPend_[b] = false;
            }
        }
    }

    // load the H bf16-triple into L1B (3 planes resident); L0B holds [Hh|Hl]
    __aicore__ inline void LoadSplitToL1Triple()
    {
        CopyGmNDToL1(l1BBuf_.Get<bfloat16_t>(), wsHhGmB_, HEAD_DIM, HEAD_DIM);
        CopyGmNDToL1(l1BBuf_.Get<bfloat16_t>()[SPLIT_ELEMS], wsHlGmB_, HEAD_DIM, HEAD_DIM);
        CopyGmNDToL1(l1BBuf_.Get<bfloat16_t>()[2 * SPLIT_ELEMS], wsHllGmB_, HEAD_DIM, HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = (HEAD_DIM + 15) / 16 * HEAD_DIM / (32 / sizeof(half));
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        LocalTensor<bfloat16_t> l0B = l0BBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> l1B = l1BBuf_.Get<bfloat16_t>();
        LoadData(l0B, l1B, ld);                           // L0B[0] = Hh
        LoadData(l0B[SPLIT_ELEMS], l1B[SPLIT_ELEMS], ld); // L0B[1] = Hl
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
    }

    // one A quad-tile (m=64) of x (bf16, read DIRECTLY from the input) through
    // the H bf16-triple, L0C-ACCUMULATED to a single fp32 plane:
    //   z = x @ (Hh+Hl+Hll)^T  (the triple captures fp32 H exactly; x bf16 is
    //   already exact - 0 idx flips offline). Hll is staged in L1B and swapped
    //   into L0B[0] between the accumulating Mmads (Hh restored after), since
    //   L0B holds only two 32KB planes.
    __aicore__ inline void MmTileQ(const GlobalTensor<bfloat16_t>& aGm, const GlobalTensor<float>& outGm,
                                   int32_t rowAct)
    {
        const int32_t bufSel = mmCtr_ & 3;
        ++mmCtr_;
        const TEventID evAB = eventMMte1_;
        LocalTensor<bfloat16_t> l1A = l1ABuf_.Get<bfloat16_t>()[bufSel * QTILE_ROWS * HEAD_DIM];
        LocalTensor<bfloat16_t> l0A = l0ABuf_.Get<bfloat16_t>()[bufSel * QTILE_ROWS * HEAD_DIM];
        CopyGmNDToL1(l1A, aGm, rowAct, HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        LoadData3DParamsV2<bfloat16_t> pa;
        pa.l1H = QTILE_ROWS / 16;
        pa.l1W = 16;
        pa.padList[0] = 0;
        pa.padList[1] = 0;
        pa.padList[2] = 0;
        pa.padList[3] = 255;
        pa.mExtension = QTILE_ROWS;
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
        LoadData<bfloat16_t>(l0A, l1A, pa);
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
        MmadParams mp;
        mp.m = QTILE_ROWS;
        mp.n = HEAD_DIM;
        mp.k = HEAD_DIM;
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = true; // z = x @ Hh^T
        mp.unitFlag = 0b10;       // hold C in L0C, more accumulation to come
        const int32_t cSel = L0CAlloc();
        LocalTensor<float> cL0 = l0CBuf_.Get<float>()[cSel * QTILE_ROWS * HEAD_DIM];
        LocalTensor<bfloat16_t> aL0 = l0A;
        LocalTensor<bfloat16_t> l0B = l0BBuf_.Get<bfloat16_t>();
        Mmad(cL0, aL0, l0B, mp);
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        mp.cmatrixInitVal = false; // z += x @ Hl^T
        Mmad(cL0, aL0, l0B[SPLIT_ELEMS], mp);
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        // swap L0B[0]: Hh -> Hll (Hh's Mmad already consumed it)
        {
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = (HEAD_DIM + 15) / 16 * HEAD_DIM / (32 / sizeof(half));
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            LoadData(l0B, l1BBuf_.Get<bfloat16_t>()[2 * SPLIT_ELEMS], ld);
        }
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
        mp.unitFlag = 0b11;      // last Mmad of the chain: allow move-out
        Mmad(cL0, aL0, l0B, mp); // z += x @ Hll^T (last)
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        // restore L0B[0] = Hh for the next tile (overlaps the fixpipe)
        {
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = (HEAD_DIM + 15) / 16 * HEAD_DIM / (32 / sizeof(half));
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            LoadData(l0B, l1BBuf_.Get<bfloat16_t>(), ld);
        }
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        // M_FIX: C result is in L0C before the fixpipe reads it
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOutQ(outGm, cL0);
        L0CFixDone(cSel);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_); // L0B[0] = Hh restored
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }
    }

    // r_hat quad-tile (m=64) through the p matmul: p1 = rh@Sh, p3 = rh@Sl
    // (rh loaded ONCE, both Mmads share L0A - the proven v5 cadence), then
    // p2 = rl@Sh. All three planes fp16 fixpipe out (relative out-rounding
    // ~2^-12|p| never flips a sign; the flip floor is the fp16 correction
    // plane noise ~1e-7 sigma, same as fp32 p1). AIV sums in fp32.
    __aicore__ inline void MmTilePQ(const GlobalTensor<half>& rhGm, const GlobalTensor<half>& rlGm,
                                    const GlobalTensor<half>& p1Gm, const GlobalTensor<half>& p2Gm,
                                    const GlobalTensor<half>& p3Gm, int32_t rowAct)
    {
        const int32_t bufSel = mmCtr_ & 3;
        ++mmCtr_;
        const TEventID evAB = eventMMte1_;
        LocalTensor<half> l1A = l1ABuf_.Get<half>()[bufSel * QTILE_ROWS * HEAD_DIM];
        LocalTensor<half> l0A = l0ABuf_.Get<half>()[bufSel * QTILE_ROWS * HEAD_DIM];
        CopyGmNDToL1(l1A, rhGm, rowAct, HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        LoadData3DParamsV2<half> pa;
        pa.l1H = QTILE_ROWS / 16;
        pa.l1W = 16;
        pa.padList[0] = 0;
        pa.padList[1] = 0;
        pa.padList[2] = 0;
        pa.padList[3] = 255;
        pa.mExtension = QTILE_ROWS;
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
        mp.m = QTILE_ROWS;
        mp.n = HEAD_DIM;
        mp.k = HEAD_DIM;
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = true;
        mp.unitFlag = 0b11;
        const int32_t cSelA = L0CAlloc();
        LocalTensor<float> cL0 = l0CBuf_.Get<float>()[cSelA * QTILE_ROWS * HEAD_DIM];
        Mmad(cL0, l0A, l0BBuf_.Get<half>(), mp); // p1 = rh @ Sh^T
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOutQF16(p1Gm, cL0);
        L0CFixDone(cSelA);
        const int32_t cSelB = L0CAlloc();
        LocalTensor<float> cL0B = l0CBuf_.Get<float>()[cSelB * QTILE_ROWS * HEAD_DIM];
        Mmad(cL0B, l0A, l0BBuf_.Get<half>()[SPLIT_ELEMS], mp); // p3 = rh @ Sl^T (same L0A)
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOutQF16(p3Gm, cL0B);
        L0CFixDone(cSelB);
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }

        // p2 = rl @ Sh^T
        const int32_t bufSel2 = mmCtr_ & 3;
        ++mmCtr_;
        LocalTensor<half> l1A2 = l1ABuf_.Get<half>()[bufSel2 * QTILE_ROWS * HEAD_DIM];
        LocalTensor<half> l0A2 = l0ABuf_.Get<half>()[bufSel2 * QTILE_ROWS * HEAD_DIM];
        CopyGmNDToL1(l1A2, rlGm, rowAct, HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        LoadData<half>(l0A2, l1A2, pa);
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
        const int32_t cSelC = L0CAlloc();
        LocalTensor<float> cL0C = l0CBuf_.Get<float>()[cSelC * QTILE_ROWS * HEAD_DIM];
        Mmad(cL0C, l0A2, l0BBuf_.Get<half>(), mp); // p2 = rl @ Sh^T
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOutQF16(p2Gm, cL0C);
        L0CFixDone(cSelC);
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }
    }

    __aicore__ inline void FixpipeOut(const GlobalTensor<float>& gmOut, const LocalTensor<float>& cL0)
    {
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = TILE_VEC;
        fp.srcStride = TILE_VEC;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.unitFlag = 0b11;
        Fixpipe(gmOut, cL0, fp);
    }

    // quad-tile (m=64) variants: 64x128 fp32 (32KB) / fp16 (16KB) per op
    __aicore__ inline void FixpipeOutQ(const GlobalTensor<float>& gmOut, const LocalTensor<float>& cL0)
    {
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = QTILE_ROWS;
        fp.srcStride = QTILE_ROWS;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.unitFlag = 0b11;
        Fixpipe(gmOut, cL0, fp);
    }

    __aicore__ inline void FixpipeOutQF16(const GlobalTensor<half>& gmOut, const LocalTensor<float>& cL0)
    {
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = QTILE_ROWS;
        fp.srcStride = QTILE_ROWS;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.unitFlag = 0b11;
        fp.quantPre = QuantMode_t::F322F16;
        Fixpipe(gmOut, cL0, fp);
    }

    // fp32 L0C -> fp16 GM (F322F16, no scale): for the p2 correction plane
    __aicore__ inline void FixpipeOutF16(const GlobalTensor<half>& gmOut, const LocalTensor<float>& cL0)
    {
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = TILE_VEC;
        fp.srcStride = TILE_VEC;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.unitFlag = 0b11;
        fp.quantPre = QuantMode_t::F322F16;
        Fixpipe(gmOut, cL0, fp);
    }

    // batched MSE quantization of b vectors starting at vb: y = z*inv already in
    // yBuf slots; writes residual into zTile and rnVec (idx stays in idxFBuf
    // for PackIdxTile)
    // batched MSE quantization of b vectors: y already normalized in yBuf;
    // binary search over the strict-greater midpoints (bits rounds), yhat from
    // a centroid table gather - provably the same decisions as golden.
    // writes residual y-yhat into zTile and rn^2 into rnVec; idx stays in idxFBuf
    // batched MSE quantization of b vectors: y in yBuf (normalized); SYMMETRIC
    // indicator chain on |y| over the positive-half mid-bounds (strict-greater
    // = golden's decisions), yhat_pos from a centroid-table Gather (exact -
    // the Axpy delta fold differs by ulps and flips near-zero QJL signs),
    // residual r = sg*(|y|-yhat_pos) is bit-exact vs golden's y-cent[idx]
    // (mul by sg=+-1.0 is an exact sign flip). sg is forced to -1 at y==0
    // (golden: 0 > 0 == false -> negative side; without it sg=0 folds to
    // idx=half-0.5 garbage - the old mse2 lone idx flip)
    // batched MSE quantization of b vectors: y already normalized in yBuf;
    // binary search over the strict-greater midpoints (bits rounds), yhat from
    // a centroid table gather - provably the same decisions as golden.
    // writes residual y-yhat into zTile and rn^2 into rnVec; idx stays in idxFBuf
    __aicore__ inline void QuantizeBatch(int32_t vb, int32_t b, const LocalTensor<float>& zTile,
                                         const LocalTensor<float>& rnVec, const LocalTensor<float>& y)
    {
        LocalTensor<float> yhat = yhatBuf_.Get<float>();
        LocalTensor<float> idxF = idxFBuf_.Get<float>();
        LocalTensor<float> tmp = pBuf_.Get<float>();
        LocalTensor<float> table = tableBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        LocalTensor<int32_t> offI = redTmpBuf_.Get<int32_t>();
        LocalTensor<uint32_t> offU = redTmpBuf_.Get<uint32_t>();
        const int32_t w = b * HEAD_DIM;
        const int32_t bits = mseBits_;

        Duplicate(idxF, 0.0f, w);
        PipeBarrier<PIPE_V>();
        for (int32_t bit = bits - 1; bit >= 0; --bit) {
            const int32_t stepI = 1 << bit;
            const float step = static_cast<float>(stepI);
            // the vector mask register is sticky (WRS leaves 64) - reset to
            // norm mode before every compare or only 64 lanes participate
            set_mask_norm();
            if (bit == bits - 1) {
                Compares(mask, y, bound_[stepI - 1], CMPMODE::GT, w);
            } else {
                Adds(tmp, idxF, static_cast<float>(stepI - 1), w);
                PipeBarrier<PIPE_V>();
                Cast(offI, tmp, RoundMode::CAST_RINT, w);
                PipeBarrier<PIPE_V>();
                ShiftLeft(offI, offI, static_cast<int32_t>(2), w);
                PipeBarrier<PIPE_V>();
                Gather(tmp, table, offU, 16u * sizeof(float), static_cast<uint32_t>(w));
                PipeBarrier<PIPE_V>();
                Compare(mask, y, tmp, CMPMODE::GT, static_cast<uint32_t>(w));
            }
            PipeBarrier<PIPE_V>();
            Duplicate(yhat, step, w);
            PipeBarrier<PIPE_V>();
            Select(tmp, mask, yhat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<uint32_t>(w));
            PipeBarrier<PIPE_V>();
            Add(idxF, idxF, tmp, w);
            PipeBarrier<PIPE_V>();
        }
        // table lookup, not a sum of centroid gaps (exact golden bytes)
        Cast(offI, idxF, RoundMode::CAST_RINT, w);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offI, offI, static_cast<int32_t>(2), w);
        PipeBarrier<PIPE_V>();
        Gather(yhat, table, offU, 0u, static_cast<uint32_t>(w));
        PipeBarrier<PIPE_V>();

        Sub(zTile[vb * HEAD_DIM], y, yhat, w);
        PipeBarrier<PIPE_V>();
        // r^2 into yhat (dead after the gather) - NOT into y
        Mul(yhat, zTile[vb * HEAD_DIM], zTile[vb * HEAD_DIM], w);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> rnTmp = pBuf_.Get<float>();
        WholeReduceSum(rnVec[vb], yhat, 64, b, 1, 1, 16);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(rnTmp, yhat[64], 64, b, 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Add(rnVec[vb], rnVec[vb], rnTmp, b);
        PipeBarrier<PIPE_V>();
    }

    // idx bit-pack via vcgadd block-reduce: idx * 2^(bits*lane) sums 8 lanes
    // into one exact fp32 integer group word (replaces the 128-rep
    // WholeReduceSum pack - ~5x less vector time). Raw __ubuf__ pointer walk
    // for the byte split: GetValue/SetValue drain the scalar pipe per element.
    __aicore__ inline void PackIdxTile(int32_t nv, int32_t bufSel = 0)
    {
        LocalTensor<float> idxF = idxFBuf_.Get<float>();
        // binary-search tables: centroids at [0,16), mid-bounds at [16,32)
        // (strict-greater midpoint walk - the same decisions as golden's
        // searchsorted; yhat comes from a table gather, not a gap sum)
        LocalTensor<float> table = tableBuf_.Get<float>();
        for (int32_t i = 0; i < 16; ++i) {
            table.SetValue(i, i < nCent_ ? cent_[i] : 0.0f);
            table.SetValue(16 + i, i < nCent_ - 1 ? bound_[i] : 0.0f);
        }
        LocalTensor<float> wtL = wtLBuf_.Get<float>();
        LocalTensor<float> wtH = wtHBuf_.Get<float>();
        LocalTensor<float> y = yBuf_.Get<float>();
        LocalTensor<float> wordLo = wordIdxLoBuf_.Get<float>();
        LocalTensor<float> wordHi = wordIdxHiBuf_.Get<float>();
        const int32_t w = nv * HEAD_DIM;

        Mul(y, idxF, wtL, w);
        PipeBarrier<PIPE_V>();
        BlockReduceSum(wordLo, y, nv * 2, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        // lo = lanes 0-3, hi = lanes 4-7 for EVERY bit width (the pack emits
        // word = lo + hi<<4*bits) - hi is not optional for bits < 4
        Mul(y, idxF, wtH, w);
        PipeBarrier<PIPE_V>();
        BlockReduceSum(wordHi, y, nv * 2, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        // int32 casts into the (dead-after-quantize) indicator scratch: in-place
        // fp32->int32 Cast on the same UB lines is not safe to assume
        LocalTensor<int32_t> wordsILo = pBuf_.Get<int32_t>();
        LocalTensor<int32_t> wordsIHi = pBuf_.Get<int32_t>()[TILE_VEC * WORDS_PER_VEC];
        Cast(wordsILo, wordLo, RoundMode::CAST_RINT, ((nv * WORDS_PER_VEC) + 7) & ~7);
        PipeBarrier<PIPE_V>();
        Cast(wordsIHi, wordHi, RoundMode::CAST_RINT, ((nv * WORDS_PER_VEC) + 7) & ~7);
        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();
        // vectorized pack (the scalar walk was 40-86us/launch): merge lo/hi into
        // full words with wide ops, regroup with Gather, and write int32 (never
        // byte-at-a-time; the AIV scalar pipe runs ~5cyc/op).
        //   mse4: out32[i] = lo[i] | hi[i]<<16           (1:1, pure vector)
        //   mse2: out32[i] = W[2i] | W[2i+1]<<16         (even/odd gather)
        //   mse3: o0=w0|w1<<24, o1=w1>>8|w2<<16, o2=w2>>16|w3<<8 (4 gathers + 3 stores/group)
        LocalTensor<int32_t> wLo = pBuf_.Get<int32_t>();
        LocalTensor<int32_t> wHi = pBuf_.Get<int32_t>()[TILE_VEC * WORDS_PER_VEC];
        LocalTensor<int32_t> wHiS = redTmpBuf_.Get<int32_t>();
        const int32_t nW = nv * WORDS_PER_VEC;
        const int32_t hiShift = 4 * mseBits_;
        // W = lo + hi<<hiShift (fields never overlap: Add == Or here)
        ShiftLeft(wHiS, wHi, hiShift, nW);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> stageW = idxStageBuf_.Get<int32_t>()[bufSel * TILE_VEC * MAX_IDX_BYTES / 4];
        LocalTensor<int32_t> wM = pBuf_.Get<int32_t>()[2 * TILE_VEC * WORDS_PER_VEC]; // merged words
        Add(wM, wLo, wHiS, nW);
        PipeBarrier<PIPE_V>();
        if (mseBits_ == 4) {
            Add(stageW, wLo, wHiS, nW);
            PipeBarrier<PIPE_V>();
        } else if (mseBits_ == 2) {
            LocalTensor<uint32_t> evOff = packOffBuf_.Get<uint32_t>();
            LocalTensor<uint32_t> odOff = packOffBuf_.Get<uint32_t>()[TILE_VEC * WORDS_PER_VEC / 2];
            LocalTensor<int32_t> ev = pTileBuf_.Get<int32_t>();
            LocalTensor<int32_t> od = pTileBuf_.Get<int32_t>()[TILE_VEC * WORDS_PER_VEC / 2];
            const int32_t nP = nv * WORDS_PER_VEC / 2;
            Gather(ev, wM, evOff, 0u, static_cast<uint32_t>(nP));
            PipeBarrier<PIPE_V>();
            Gather(od, wM, odOff, 0u, static_cast<uint32_t>(nP));
            PipeBarrier<PIPE_V>();
            ShiftLeft(od, od, static_cast<int32_t>(16), nP);
            PipeBarrier<PIPE_V>();
            Add(stageW, ev, od, nP);
            PipeBarrier<PIPE_V>();
        } else {
            // 24-bit words -> 4 words = 3 int32 (o0=w0|w1<<24, o1=w1>>8|w2<<16,
            // o2=w2>>16|w3<<8), reading the VECTOR-merged wM. The Gather-based
            // regrouping measured SLOWER than this short walk (6 gathers/tile
            // ~ the walk's whole cost - build279 regression).
            PipeBarrier<PIPE_ALL>();
            __ubuf__ int32_t* wm = reinterpret_cast<__ubuf__ int32_t*>(wM.GetPhyAddr());
            __ubuf__ int32_t* ow = reinterpret_cast<__ubuf__ int32_t*>(
                idxStageBuf_.Get<uint8_t>()[bufSel * TILE_VEC * MAX_IDX_BYTES].GetPhyAddr());
            for (int32_t v = 0; v < nv; ++v) {
                __ubuf__ int32_t* owv = ow + v * 12;
                const __ubuf__ int32_t* wr = wm + v * WORDS_PER_VEC;
                for (int32_t g = 0; g < 4; ++g) {
                    const uint32_t w0 = static_cast<uint32_t>(wr[g * 4]);
                    const uint32_t w1 = static_cast<uint32_t>(wr[g * 4 + 1]);
                    const uint32_t w2 = static_cast<uint32_t>(wr[g * 4 + 2]);
                    const uint32_t w3 = static_cast<uint32_t>(wr[g * 4 + 3]);
                    owv[g * 3] = static_cast<int32_t>(w0 | (w1 << 24));
                    owv[g * 3 + 1] = static_cast<int32_t>((w1 >> 8) | (w2 << 16));
                    owv[g * 3 + 2] = static_cast<int32_t>((w2 >> 16) | (w3 << 8));
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // r_hat zTile tile -> rh/rl fp16 tiles into GM, chunk-major layout:
    // per chunk [rh: CHUNK_ROWS x 128 | rl: CHUNK_ROWS x 128] so the AIC reads
    // 64-row contiguous quad-tiles (Nd2Nz stride 128) from either section
    __aicore__ inline void UploadRhat(int64_t chunkId, int64_t localRow, int32_t nv, const LocalTensor<float>& zTile,
                                      int32_t bufSel = 0)
    {
        // dedicated full-tile buffers: the 1KB BATCH_VEC scratch buffers here
        // overflowed by 4-8KB per tile and steamrolled normBF/gammaBF staging
        // (bisected 2026-09-26). rh|rl interleaved per-vec in ONE plane: a
        // single upload copy and one 32-row A tile fetch pattern downstream.
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
        const int64_t cBase = chunkId * 2 * CHUNK_ROWS * HEAD_DIM;
        DataCopy(wsRhat2Gm_[cBase + localRow * HEAD_DIM], rh, w);
        DataCopy(wsRhat2Gm_[cBase + (CHUNK_ROWS + localRow) * HEAD_DIM], rl, w);
    }

    // dcci-flush a GM byte range: scalar stores (and MIX-mode DMA) sit in the
    // core-local cache and never become host-visible on this stack unless an
    // explicit clean+invalidate covers their lines (bisected 2026-09-26).
    // Used only for the scalar-written partial-tile tails.
    template <typename T>
    __aicore__ inline void FlushOutLines(const GlobalTensor<T>& gm, int64_t elemOff, int64_t bytes)
    {
        __gm__ uint8_t* base =
            reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(gm.GetPhyAddr())) + elemOff * sizeof(T);
        const uint64_t aligned = reinterpret_cast<uint64_t>(base) & ~63ULL;
        const int64_t skip = reinterpret_cast<uint64_t>(base) - aligned;
        const int64_t lines = (bytes + skip + 63) / 64;
        GlobalTensor<int64_t> lineT;
        for (int64_t li = 0; li < lines; ++li) {
            lineT.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(aligned + li * 64), 8);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lineT[0]);
        }
    }

    __aicore__ inline void FlushScalar(int64_t g0, int32_t stageOff, int32_t nv, const LocalTensor<bfloat16_t>& normBF,
                                       const LocalTensor<bfloat16_t>& gammaBF, const LocalTensor<uint8_t>& idxStage,
                                       const LocalTensor<uint8_t>& qjlStage, bool phase1, bool phase2)
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
    TBuf<QuePosition::VECCALC> pRdBuf_;
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
    TBuf<QuePosition::VECCALC> tableBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> packOffBuf_;
    TBuf<QuePosition::VECCALC> onesBuf_;
    TBuf<QuePosition::VECCALC> splitBuf_;

    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;
    GlobalTensor<bfloat16_t> wsHhGmB_;
    GlobalTensor<bfloat16_t> wsHlGmB_;
    GlobalTensor<bfloat16_t> wsHllGmB_;
    GlobalTensor<half> wsShGm_;
    GlobalTensor<half> wsSlGm_;
    GlobalTensor<float> wsZGm_;
    GlobalTensor<half> wsRhat2Gm_;
    GlobalTensor<half> wsP1Gm_;
    GlobalTensor<half> wsP2Gm_;
    GlobalTensor<half> wsP3Gm_;
    __gm__ int32_t* flagRaw_ = nullptr;
    GlobalTensor<int32_t> flagGm_;
    GlobalTensor<uint64_t> flagU64_;
    TBuf<TPosition::A1> l1ABuf_;
    TBuf<TPosition::A1> l1BBuf_;
    TBuf<TPosition::A2> l0ABuf_;
    TBuf<TPosition::B2> l0BBuf_;
    TBuf<TPosition::CO1> l0CBuf_;
    LocalTensor<float> invAll_;
    LocalTensor<float> normAll_;

    bool isAic_ = false;
    int64_t aivSubIdx_ = 0;
    int64_t logicalIdx_ = 0;
    int64_t totalVec_ = 0;
    int64_t wsVec_ = 0;
    int64_t qjlDim_ = 0;
    int32_t mseBits_ = 3;
    int32_t nCent_ = 8;
    int32_t idxBytesPerVec_ = 48;
    int32_t qjlBytesPerVec_ = 16;
    uint32_t coreNum_ = 1;
    int64_t chunkNum_ = 0;
    int64_t myChunks_ = 0;

    TEventID eventVS_ = 0;
    TEventID eventSV_ = 0;
    TEventID eventMte2V_ = 0;
    TEventID eventVMte2_ = 0;
    TEventID eventVMte3_ = 0;
    TEventID eventMte3V_ = 0;
    TEventID eventMte3Mte2_ = 0;
    TEventID eventMte3S_ = 0;
    TEventID eventSMte3_ = 0;
    TEventID eventMte2V2_ = 0;
    TEventID eventMte2Mte1_ = 0;
    TEventID eventMte1M_ = 0;
    TEventID eventMMte1_ = 0;
    TEventID eventMte1Mte2_ = 0;
    TEventID eventFixS_ = 0;
    TEventID eventFixM_ = 0;
    TEventID eventMFix_ = 0;
    TEventID eventMMte1B_ = 0;
    TEventID eventFixM2_ = 0;
    int32_t mmCtr_ = 0;
    int32_t fixCtr_ = 0;
    bool fixPend_[2] = {false, false};

    float cent_[16] = {0.0f};
    float bound_[15] = {0.0f};
    float delta_[15] = {0.0f};
    float inv_[TILE_VEC] = {0.0f};
    float invR_[TILE_VEC] = {0.0f};
};

#endif
