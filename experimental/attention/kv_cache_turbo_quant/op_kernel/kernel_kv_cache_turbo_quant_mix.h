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
        if ASCEND_IS_AIC {
            isAic_ = true;
            logicalIdx = GetBlockIdx();
        } else {
            logicalIdx = GetBlockIdx() / 2;   // two AIVs share one logical group
        }
        logicalIdx_ = logicalIdx;
        aivSubIdx_ = isAic_ ? 0 : (GetBlockIdx() % 2);

        // round-robin chunk assignment: group g owns chunks {g, g+coreNum, ...}
        chunkNum_ = (totalVec_ + CHUNK_ROWS - 1) / CHUNK_ROWS;
        myChunks_ = (logicalIdx < chunkNum_)
                        ? ((chunkNum_ - 1 - logicalIdx) / static_cast<int64_t>(coreNum_) + 1)
                        : 0;

        kvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kvVectors), totalVec_ * HEAD_DIM);
        rotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rotationMatrix), HEAD_DIM * HEAD_DIM);
        qjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qjlMatrix), qjlDim_ * HEAD_DIM);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantIdx), totalVec_ * idxBytesPerVec_);
        qjlOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantQjl), totalVec_ * qjlBytesPerVec_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantNorm), totalVec_);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(quantGamma), totalVec_);

        // workspace layout (bytes), wsVec = totalVec padded up to one chunk:
        //   0..264K     cross-core GM spin flags (64B-strided int32 cells)
        //   264K..392K  Hh/Hl/Sh/Sl fp16 splits (4 x 32KB)
        //   392K        z fp32            wsVec x 512B   (cube plane)
        //   then        r_hat [rh|rl]     wsVec x 512B   (fp16, interleaved/tile)
        //   then        p1/p2/p3 fp16 x3  wsVec x 256B   (cube planes)
        //   then        xh fp16           wsVec x 256B   (AIV upload)
        wsVec_ = chunkNum_ * CHUNK_ROWS;
        __gm__ half *wsH = reinterpret_cast<__gm__ half *>(workspace + WS_SPLIT_OFF);
        wsHhGm_.SetGlobalBuffer(wsH + 0 * SPLIT_ELEMS, SPLIT_ELEMS);
        wsHlGm_.SetGlobalBuffer(wsH + 1 * SPLIT_ELEMS, SPLIT_ELEMS);
        wsShGm_.SetGlobalBuffer(wsH + 2 * SPLIT_ELEMS, SPLIT_ELEMS);
        wsSlGm_.SetGlobalBuffer(wsH + 3 * SPLIT_ELEMS, SPLIT_ELEMS);
        __gm__ uint8_t *wsB = workspace + WS_Z_OFF;
        wsZGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsB), wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 4;
        wsRhat2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wsB), 2 * wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 2 * 2;
        wsP1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wsB), wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 2;
        wsP2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wsB), wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 2;
        wsP3Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wsB), wsVec_ * HEAD_DIM);
        wsB += static_cast<int64_t>(wsVec_) * HEAD_DIM * 2;
        wsXhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wsB), wsVec_ * HEAD_DIM);

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
        } else {
            eventVS_ = 4;
            eventSV_ = 4;
            eventMte2V_ = 4;
            eventVMte2_ = 4;
            eventVMte3_ = 4;
            eventMte3V_ = 4;
            eventMte3Mte2_ = 4;
            eventMte3S_ = 4;
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
            // ---- phase Z (AIC): z = (Hh+Hl) @ xh per chunk, pipelined against
            // the AIV pre-pass (XH) and quant pass (consumes Z)
            WaitXh(slotBase, 0);   // first XH implies this group's H/S splits done
            LoadSplitToL1(wsHhGm_, wsHlGm_);
            for (int64_t l = 0; l < myChunks_; ++l) {
                if (l > 0) {
                    WaitXh(slotBase, l);
                }
                const int64_t c = ChunkId(l);
                const int64_t rows = ChunkRows(c);
                const int64_t tiles = (rows + TILE_VEC - 1) / TILE_VEC;
                for (int64_t t = 0; t < tiles; ++t) {
                    const int64_t g = c * CHUNK_ROWS + t * TILE_VEC;
                    const int32_t nv = static_cast<int32_t>((rows - t * TILE_VEC) < TILE_VEC ? (rows - t * TILE_VEC)
                                                                                              : TILE_VEC);
                    MmTile(wsXhGm_[g * HEAD_DIM], wsZGm_[g * HEAD_DIM], nv);
                }
                SyncPipe<HardEvent::FIX_S>();   // z chunk drained before the flag store
                GmFlagSet(FlagCell(slotBase, l, SLOT_Z_LO), MAGIC_Z);
                GmFlagSet(FlagCell(slotBase, l, SLOT_Z_HI), MAGIC_Z);
            }
            // ---- phase P (AIC): p1 = rh@Sh, p3 = rh@Sl, p2 = rl@Sh - three
            // SEPARATE fp16 planes (no L0C accumulation for p - it loses the
            // correction-term precision and the QJL signs flip wholesale,
            // confirmed 2026-09-27 build244: 93% qjl wrong with accumulated p13)
            LoadSplitToL1(wsShGm_, wsSlGm_);
            for (int64_t l = 0; l < myChunks_; ++l) {
                GmFlagWait(FlagCell(slotBase, l, SLOT_R_LO), MAGIC_R);
                GmFlagWait(FlagCell(slotBase, l, SLOT_R_HI), MAGIC_R);
                const int64_t c = ChunkId(l);
                const int64_t rows = ChunkRows(c);
                const int64_t tiles = (rows + TILE_VEC - 1) / TILE_VEC;
                for (int64_t t = 0; t < tiles; ++t) {
                    const int64_t g = c * CHUNK_ROWS + t * TILE_VEC;
                    const int32_t nv = static_cast<int32_t>((rows - t * TILE_VEC) < TILE_VEC ? (rows - t * TILE_VEC)
                                                                                              : TILE_VEC);
                    // [rh|rl] plane-major per tile: rl follows the tile's rh block
                    MmTileP(wsRhat2Gm_[g * 2 * HEAD_DIM], wsRhat2Gm_[g * 2 * HEAD_DIM + TILE_VEC * HEAD_DIM],
                            wsP1Gm_[g * HEAD_DIM], wsP2Gm_[g * HEAD_DIM], wsP3Gm_[g * HEAD_DIM], nv);
                }
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
            SyncPipe<HardEvent::FIX_S>();
            SetFlag<HardEvent::FIX_M>(eventFixM_);
            WaitFlag<HardEvent::FIX_M>(eventFixM_);
            SetFlag<HardEvent::M_MTE1>(eventMMte1_);
            WaitFlag<HardEvent::M_MTE1>(eventMMte1_);
            SetFlag<HardEvent::MTE1_M>(eventMte1M_);
            WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
            PipeBarrier<PIPE_M>();
            SetAtomicNone();
            SetMaskNorm();
            SetLoadDataBoundary((uint64_t)0);
            SetLoadDataPaddingValue((uint64_t)0);
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
    static constexpr int32_t RED_TMP_ELEMS = 2048;   // sg scratch spans a full 16x128 batch
    static constexpr int32_t SPLIT_ELEMS = HEAD_DIM * HEAD_DIM;
    static constexpr int32_t SPLIT_STAGE_BYTES = 56 * 1024;
    static constexpr int32_t CHUNK_TILES = 8;                       // pipeline chunk = 8 tiles
    static constexpr int32_t CHUNK_ROWS = CHUNK_TILES * TILE_VEC;   // 128 vectors
    static constexpr int32_t HALF_CHUNK_ROWS = CHUNK_ROWS / 2;      // per partner AIV
    static constexpr int32_t MAX_LOCAL_CHUNKS = 22;                 // covers totalVec <= 67584
    static constexpr int32_t MAX_ROWS_PER_AIV = MAX_LOCAL_CHUNKS * HALF_CHUNK_ROWS;   // 1408
    static constexpr int32_t L1_A_BYTES = 4 * 2 * TILE_VEC * HEAD_DIM * sizeof(half);   // up to 4 x 8K
    static constexpr int32_t L1_B_BYTES = 2 * HEAD_DIM * HEAD_DIM * sizeof(half);   // 64K: hi+lo splits
    static constexpr int32_t L0_A_BYTES = 4 * 2 * TILE_VEC * HEAD_DIM * sizeof(half);   // up to 4 x 8K
    static constexpr int32_t L0_B_BYTES = 2 * HEAD_DIM * HEAD_DIM * sizeof(half);   // 64K: hi+lo splits
    static constexpr int32_t L0_C_BYTES = TILE_VEC * HEAD_DIM * sizeof(float);      // 8K
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
    static constexpr int32_t GROUP_FLAG_CELLS = MAX_LOCAL_CHUNKS * SLOTS_PER_CHUNK;   // 176
    static constexpr int32_t SLOT_STRIDE = 16;   // int32 elems per 64B slot
    static constexpr int32_t FLAG_ELEMS = 24 * GROUP_FLAG_CELLS * SLOT_STRIDE;        // 67584 (264KB)
    static constexpr int32_t WS_SPLIT_OFF = 264 * 1024;
    static constexpr int32_t WS_Z_OFF = (264 + 128) * 1024;
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
            volatile __gm__ int32_t *slot = flagRaw_ + slotElem;
            *slot = magic;
            __asm__ __volatile__("" ::: "memory");
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                flagGm_[slotElem]);
            __asm__ __volatile__("" ::: "memory");
        } else {
            // AIV: dcci on dav_m200 poisons the core's later GM writes (the
            // write-poison ghost - bisected 2026-09-26: in-loop dcci'd flag
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
            // 2026-09-26) - poll the flag line via an MTE2 DMA read instead:
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
        SyncPipe<HardEvent::S_V>();
    }

    // fp32 H/S -> fp16 hi/lo splits to GM (every AIV writes identical content;
    // a core's own writes guarantee the content its partner AIC reads is valid)
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

    // ---- phase 1 (AIV): fused pre-pass per chunk: row norms of x (persisted
    // for the quant phase), norm output, and the fp16 xh upload for the cube
    __aicore__ inline void PrePassChunk(int64_t l)
    {
        const int64_t c = ChunkId(l);
        const int64_t rows = ChunkRows(c);
        // this AIV's half of the chunk
        const int64_t halfBegin = c * CHUNK_ROWS + aivSubIdx_ * HALF_CHUNK_ROWS;
        int64_t myRows = rows - aivSubIdx_ * HALF_CHUNK_ROWS;
        if (myRows > HALF_CHUNK_ROWS) {
            myRows = HALF_CHUNK_ROWS;
        }
        if (myRows <= 0) {
            GmFlagSet(FlagCell(SlotBase(), l, aivSubIdx_ == 0 ? SLOT_XH_LO : SLOT_XH_HI), MAGIC_XH);
            return;
        }
        LocalTensor<bfloat16_t> xTile = xTileBuf_.Get<bfloat16_t>();
        LocalTensor<float> xf32 = xUpF32Buf_.Get<float>();
        LocalTensor<half> xh = xUpHBuf_.Get<half>();
        LocalTensor<float> normVec = normVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> normBF = normBFBuf_.Get<bfloat16_t>();
        LocalTensor<float> sqTmp = yhatBuf_.Get<float>();
        for (int64_t t = 0; t < myRows; t += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((myRows - t) < TILE_VEC ? (myRows - t) : TILE_VEC);
            const int64_t g = halfBegin + t;
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(xTile, kvGm_[g * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            SyncPipe<HardEvent::MTE3_V>();   // xh free: previous tile's upload drained
            Cast(xf32, xTile, RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            // norms: squared magnitude, two 64-wide WRS halves per row
            Mul(sqTmp, xf32, xf32, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(normVec, sqTmp, 64, nv, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(xBuf2(), sqTmp[64], 64, nv, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            Add(normVec, normVec, xBuf2(), nv);
            PipeBarrier<PIPE_V>();
            Sqrt(normVec, normVec, nv);
            PipeBarrier<PIPE_V>();
            // persist norm/inv for the quant phase; write the norm output now
            SyncPipe<HardEvent::V_S>();
            const int64_t rowBase = l * HALF_CHUNK_ROWS + t;
            for (int32_t v = 0; v < nv; ++v) {
                const float norm = normVec.GetValue(v);
                normAll_.SetValue(rowBase + v, norm);
                invAll_.SetValue(rowBase + v, (norm > 0.0f) ? 1.0f / ((norm < MIN_DENOM) ? MIN_DENOM : norm) : 0.0f);
            }
            SyncPipe<HardEvent::S_V>();
            Cast(normBF, normVec, RoundMode::CAST_RINT, nv);
            PipeBarrier<PIPE_V>();
            // fp16 upload for the cube A feed
            Cast(xh, xf32, RoundMode::CAST_RINT, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE3>();
            DataCopy(wsXhGm_[g * HEAD_DIM], xh, nv * HEAD_DIM);
            if (nv == TILE_VEC) {
                DataCopy(normGm_[g], normBF, nv);
            } else {
                SyncPipe<HardEvent::MTE3_V>();
                SyncPipe<HardEvent::V_S>();
                for (int32_t v = 0; v < nv; ++v) {
                    normGm_.SetValue(g + v, normBF.GetValue(v));
                }
                SyncPipe<HardEvent::S_V>();
                FlushOutLines(normGm_, g, nv * 2);
            }
        }
        SyncPipe<HardEvent::MTE3_S>();   // xh/norm in L2 before the flag store
        GmFlagSet(FlagCell(SlotBase(), l, aivSubIdx_ == 0 ? SLOT_XH_LO : SLOT_XH_HI), MAGIC_XH);
    }

    // ---- phase 2 (AIV): y = z * inv, MSE quantize, residual norms, r_hat
    // split upload, idx pack, gamma output - per chunk as Z arrives
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
        LocalTensor<float> zTile = zTileBuf_.Get<float>();
        LocalTensor<float> normVec = normVecBuf_.Get<float>();
        LocalTensor<float> rnVec = rnVecBuf_.Get<float>();
        LocalTensor<float> gammaVec = gammaVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> gammaBF = gammaBFBuf_.Get<bfloat16_t>();
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();

        for (int64_t t = 0; t < myRows; t += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((myRows - t) < TILE_VEC ? (myRows - t) : TILE_VEC);
            const int64_t g0 = halfBegin + t;
            const int64_t rowBase = l * HALF_CHUNK_ROWS + t;

            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(zTile, wsZGm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();

            // y = z * inv (per-row scalar, from the persisted pre-pass values)
            SyncPipe<HardEvent::V_S>();
            for (int32_t v = 0; v < nv; ++v) {
                inv_[v] = invAll_.GetValue(rowBase + v);
                normVec.SetValue(v, normAll_.GetValue(rowBase + v));
            }
            SyncPipe<HardEvent::S_V>();
            for (int32_t i = 0; i < nv; ++i) {
                Muls(yBuf_.Get<float>()[i * HEAD_DIM], zTile[i * HEAD_DIM], inv_[i], HEAD_DIM);
            }
            PipeBarrier<PIPE_V>();
            QuantizeBatch(0, nv, zTile, rnVec);
            Sqrt(rnVec, rnVec, nv);
            PipeBarrier<PIPE_V>();
            Mul(gammaVec, normVec, rnVec, nv);
            PipeBarrier<PIPE_V>();
            Cast(gammaBF, gammaVec, RoundMode::CAST_RINT, nv);
            PipeBarrier<PIPE_V>();

            SyncPipe<HardEvent::V_S>();
            for (int32_t v = 0; v < nv; ++v) {
                const float rn = rnVec.GetValue(v);
                invR_[v] = (rn > 0.0f) ? 1.0f / ((rn < MIN_DENOM) ? MIN_DENOM : rn) : 0.0f;
            }
            PackIdxTile(nv);
            SyncPipe<HardEvent::S_V>();

            // ---- r_hat (in place) and its fp16 hi/lo splits uploaded to GM
            for (int32_t v = 0; v < nv; ++v) {
                Muls(zTile[v * HEAD_DIM], zTile[v * HEAD_DIM], invR_[v], HEAD_DIM);
            }
            PipeBarrier<PIPE_V>();
            UploadRhat(g0, nv, zTile);

            SyncPipe<HardEvent::V_MTE3>();
            if (nv == TILE_VEC) {
                DataCopy(gammaGm_[g0], gammaBF, nv);
                DataCopy(idxGm_[g0 * idxBytesPerVec_], idxStage, nv * idxBytesPerVec_);
            } else {
                FlushScalar(g0, 0, nv, normBFBuf_.Get<bfloat16_t>(), gammaBF, idxStage, qjlStage, true, false);
                FlushOutLines(gammaGm_, g0, nv * 2);
                FlushOutLines(idxGm_, g0 * idxBytesPerVec_, nv * idxBytesPerVec_);
            }
            SyncPipe<HardEvent::MTE3_V>();
        }
        SyncPipe<HardEvent::MTE3_S>();
        GmFlagSet(FlagCell(SlotBase(), l, aivSubIdx_ == 0 ? SLOT_R_LO : SLOT_R_HI), MAGIC_R);
    }

    // ---- phase 3 (AIV): p1+p2+p3 (fp32 sum) -> sign -> qjl pack, per chunk
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
        LocalTensor<half> pHalf = bvBuf_.Get<half>();
        LocalTensor<float> pCorr = pTileBuf_.Get<float>();
        LocalTensor<float> bv = bvBuf_.Get<float>();
        LocalTensor<uint8_t> qjlStage = qjlStageBuf_.Get<uint8_t>();
        LocalTensor<bfloat16_t> normBF = normBFBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> gammaBF = gammaBFBuf_.Get<bfloat16_t>();
        LocalTensor<uint8_t> idxStage = idxStageBuf_.Get<uint8_t>();

        for (int64_t t = 0; t < myRows; t += TILE_VEC) {
            const int32_t nv = static_cast<int32_t>((myRows - t) < TILE_VEC ? (myRows - t) : TILE_VEC);
            const int64_t g0 = halfBegin + t;
            // p = cast(p1 fp16) + cast(p2 fp16) + cast(p3 fp16), summed in fp32
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(pHalf, wsP1Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            Cast(pAll, pHalf, RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(pHalf, wsP2Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            Cast(pCorr, pHalf, RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Add(pAll, pAll, pCorr, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            SyncPipe<HardEvent::V_MTE2>();
            DataCopy(pHalf, wsP3Gm_[g0 * HEAD_DIM], nv * HEAD_DIM);
            SyncPipe<HardEvent::MTE2_V>();
            Cast(pCorr, pHalf, RoundMode::CAST_NONE, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Add(pAll, pAll, pCorr, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();

            Duplicate(bv, 0.0f, nv * HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Compare(qjlStage, pAll, bv, CMPMODE::GE, nv * HEAD_DIM);
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

    // AIV-side full flow: splits -> per-chunk [pre | quant | sign]
    __aicore__ inline void ProcessAiv()
    {
        SplitMatrices();

        for (int64_t l = 0; l < myChunks_; ++l) {
            PrePassChunk(l);
        }
        for (int64_t l = 0; l < myChunks_; ++l) {
            GmFlagWait(FlagCell(SlotBase(), l, aivSubIdx_ == 0 ? SLOT_Z_LO : SLOT_Z_HI), MAGIC_Z);
            QuantChunk(l);
        }
        for (int64_t l = 0; l < myChunks_; ++l) {
            GmFlagWait(FlagCell(SlotBase(), l, aivSubIdx_ == 0 ? SLOT_P_LO : SLOT_P_HI), MAGIC_P);
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
    __aicore__ inline void CopyGmNDToL1(const LocalTensor<T> &l1Tensor, const GlobalTensor<T> &gmTensor,
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

    // one A tile through both B-splits, L0C-ACCUMULATED into a single plane:
    //   out = A @ (B0+B1)^T = A@B0^T + A@B1^T  (fp32-grade from fp16 splits,
    //   one fixpipe + one GM plane instead of two - halves the z/p DMA)
    __aicore__ inline void MmTile(const GlobalTensor<half> &aGm, const GlobalTensor<float> &outGm, int32_t rowAct)
    {
        // 4-deep A rotation driven by a monotonic MmTile counter: the loop-t0
        // parity collided for consecutive MmTiles (bisected 2026-09-26). A
        // strict +1 counter guarantees distinct slots.
        const int32_t bufSel = mmCtr_ & 3;
        ++mmCtr_;
        const TEventID evAB = eventMMte1_;
        LocalTensor<half> l1A = l1ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        LocalTensor<half> l0A = l0ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        CopyGmNDToL1(l1A, aGm, rowAct, HEAD_DIM);
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
        mp.cmatrixInitVal = true;    // first split: C = A @ B0^T
        mp.unitFlag = 0b10;          // UNITFLAG_ENABLE: hold C in L0C, more accumulation to come
        LocalTensor<float> cL0 = l0CBuf_.Get<float>();
        LocalTensor<half> aL0 = l0A;
        Mmad(cL0, aL0, l0BBuf_.Get<half>(), mp);
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        // second split accumulates onto the same L0C (same A operand, B1):
        // C += A @ B1^T. cmatrixInitVal=false uses the real C already in L0C.
        mp.cmatrixInitVal = false;
        mp.unitFlag = 0b11;          // UNITFLAG_EN_OUTER_LAST: last, allow fixpipe move-out
        Mmad(cL0, aL0, l0BBuf_.Get<half>()[SPLIT_ELEMS], mp);
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        // M_FIX: C result is in L0C before the fixpipe reads it
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOut(outGm, cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        // coarse M-pipe barrier every 4th MmTile: bounds the load-ahead skew
        // at 3, which the 4-deep A rotation absorbs
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }
    }

    // r_hat tile through the p matmul, in the proven v5 cadence: rh is loaded
    // ONCE per tile and drives BOTH rh Mmads (p1 = rh@Sh, p3 = rh@Sl), then rl
    // is loaded once for p2 = rl@Sh (2 A-loads per tile; the 3-load variant
    // densified the load stream past what the 4-deep rotation absorbs and
    // middle tiles came back scrambled). All three planes are fp16 fixpipe
    // out: relative out-rounding (~2^-12|p|) never flips a sign.
    __aicore__ inline void MmTileP(const GlobalTensor<half> &rhGm, const GlobalTensor<half> &rlGm,
                                   const GlobalTensor<half> &p1Gm, const GlobalTensor<half> &p2Gm,
                                   const GlobalTensor<half> &p3Gm, int32_t rowAct)
    {
        const int32_t bufSel = mmCtr_ & 3;
        ++mmCtr_;
        const TEventID evAB = eventMMte1_;
        LocalTensor<half> l1A = l1ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        LocalTensor<half> l0A = l0ABuf_.Get<half>()[bufSel * TILE_VEC * HEAD_DIM];
        CopyGmNDToL1(l1A, rhGm, rowAct, HEAD_DIM);   // rh rows contiguous in the tile block
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
        Mmad(cL0, l0A, l0BBuf_.Get<half>(), mp);   // p1 = rh @ Sh^T
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOutF16(p1Gm, cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        Mmad(cL0, l0A, l0BBuf_.Get<half>()[SPLIT_ELEMS], mp);   // p3 = rh @ Sl^T (same L0A)
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOutF16(p3Gm, cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }

        // p2 = rl @ Sh^T (separate A load, same cadence as the proven v5 MmTile1)
        const int32_t bufSel2 = mmCtr_ & 3;
        ++mmCtr_;
        LocalTensor<half> l1A2 = l1ABuf_.Get<half>()[bufSel2 * TILE_VEC * HEAD_DIM];
        LocalTensor<half> l0A2 = l0ABuf_.Get<half>()[bufSel2 * TILE_VEC * HEAD_DIM];
        CopyGmNDToL1(l1A2, rlGm, rowAct, HEAD_DIM);
        SetFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(eventMte2Mte1_);
        LoadData<half>(l0A2, l1A2, pa);
        SetFlag<HardEvent::MTE1_M>(eventMte1M_);
        WaitFlag<HardEvent::MTE1_M>(eventMte1M_);
        MmadParams mp2;
        mp2.m = TILE_VEC;
        mp2.n = HEAD_DIM;
        mp2.k = HEAD_DIM;
        mp2.cmatrixSource = false;
        mp2.cmatrixInitVal = true;
        mp2.unitFlag = 0b11;
        Mmad(cL0, l0A2, l0BBuf_.Get<half>(), mp2);
        SetFlag<HardEvent::M_MTE1>(evAB);
        WaitFlag<HardEvent::M_MTE1>(evAB);
        SetFlag<HardEvent::M_FIX>(eventMFix_);
        WaitFlag<HardEvent::M_FIX>(eventMFix_);
        FixpipeOutF16(p2Gm, cL0);
        SetFlag<HardEvent::FIX_M>(eventFixM_);
        WaitFlag<HardEvent::FIX_M>(eventFixM_);
        if ((mmCtr_ & 3) == 0) {
            PipeBarrier<PIPE_M>();
        }
    }

    __aicore__ inline void FixpipeOut(const GlobalTensor<float> &gmOut, const LocalTensor<float> &cL0)
    {
        // Fixpipe-API L0C -> GM. Its writes sit in the cube DMA (ATOMIC)
        // coherency domain; on this stack partner AIVs observe them after
        // SyncPipe<FIX_S> + the flag store ordering (no per-line dcci needed -
        // bisected 2026-09-27, byte-identical accuracy with/without).
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = TILE_VEC;
        fp.srcStride = TILE_VEC;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.unitFlag = 0b11;
        Fixpipe(gmOut, cL0, fp);
    }

    // fp32 L0C -> fp16 GM (F322F16, no scale): for the p2 correction plane
    __aicore__ inline void FixpipeOutF16(const GlobalTensor<half> &gmOut, const LocalTensor<float> &cL0)
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
    __aicore__ inline void QuantizeBatch(int32_t vb, int32_t b, const LocalTensor<float> &zTile,
                                         const LocalTensor<float> &rnVec)
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
    }

    // idx bit-pack via vcgadd block-reduce: idx * 2^(bits*lane) sums 8 lanes
    // into one exact fp32 integer group word (replaces the 128-rep
    // WholeReduceSum pack - ~5x less vector time). Raw __ubuf__ pointer walk
    // for the byte split: GetValue/SetValue drain the scalar pipe per element.
    __aicore__ inline void PackIdxTile(int32_t nv)
    {
        LocalTensor<float> idxF = idxFBuf_.Get<float>();
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
        __ubuf__ int32_t *lo = reinterpret_cast<__ubuf__ int32_t *>(wordsILo.GetPhyAddr());
        __ubuf__ int32_t *hi = reinterpret_cast<__ubuf__ int32_t *>(wordsIHi.GetPhyAddr());
        __ubuf__ uint8_t *out = reinterpret_cast<__ubuf__ uint8_t *>(idxStageBuf_.Get<uint8_t>().GetPhyAddr());
        const int32_t hiShift = 4 * mseBits_;
        const int32_t nWords = nv * WORDS_PER_VEC;
        const int32_t bits = mseBits_;
        if (bits == 4) {
            for (int32_t i = 0; i < nWords; ++i) {
                const uint32_t word = static_cast<uint32_t>(lo[i]) + (static_cast<uint32_t>(hi[i]) << hiShift);
                out[i * 4] = static_cast<uint8_t>(word);
                out[i * 4 + 1] = static_cast<uint8_t>(word >> 8);
                out[i * 4 + 2] = static_cast<uint8_t>(word >> 16);
                out[i * 4 + 3] = static_cast<uint8_t>(word >> 24);
            }
        } else if (bits == 3) {
            for (int32_t i = 0; i < nWords; ++i) {
                const uint32_t word = static_cast<uint32_t>(lo[i]) + (static_cast<uint32_t>(hi[i]) << hiShift);
                out[i * 3] = static_cast<uint8_t>(word);
                out[i * 3 + 1] = static_cast<uint8_t>(word >> 8);
                out[i * 3 + 2] = static_cast<uint8_t>(word >> 16);
            }
        } else {
            for (int32_t i = 0; i < nWords; ++i) {
                const uint32_t word = static_cast<uint32_t>(lo[i]) + (static_cast<uint32_t>(hi[i]) << hiShift);
                out[i * 2] = static_cast<uint8_t>(word);
                out[i * 2 + 1] = static_cast<uint8_t>(word >> 8);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // r_hat zTile tile -> rh/rl fp16 tiles into GM (A feed of the QJL matmul)
    __aicore__ inline void UploadRhat(int64_t g0, int32_t nv, const LocalTensor<float> &zTile)
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
        // two fixed-offset copies: the AIC reads rl at tile_base + 16*128 even
        // for partial tiles, so rl must NOT be packed compactly after rh
        DataCopy(wsRhat2Gm_[g0 * 2 * HEAD_DIM], rh, w);
        DataCopy(wsRhat2Gm_[g0 * 2 * HEAD_DIM + TILE_VEC * HEAD_DIM], rl, w);
    }

    // dcci-flush a GM byte range: scalar stores (and MIX-mode DMA) sit in the
    // core-local cache and never become host-visible on this stack unless an
    // explicit clean+invalidate covers their lines (bisected 2026-09-26).
    // Used only for the scalar-written partial-tile tails.
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
                // norm was already written by the pre-pass; only gamma + idx here
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
    GlobalTensor<float> wsZGm_;
    GlobalTensor<half> wsRhat2Gm_;
    GlobalTensor<half> wsP1Gm_;
    GlobalTensor<half> wsP2Gm_;
    GlobalTensor<half> wsP3Gm_;
    GlobalTensor<half> wsXhGm_;
    __gm__ int32_t *flagRaw_ = nullptr;
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
    TEventID eventMte2Mte1_ = 0;
    TEventID eventMte1M_ = 0;
    TEventID eventMMte1_ = 0;
    TEventID eventMte1Mte2_ = 0;
    TEventID eventFixS_ = 0;
    TEventID eventFixM_ = 0;
    TEventID eventMFix_ = 0;
    TEventID eventMMte1B_ = 0;
    int32_t mmCtr_ = 0;

    float cent_[16] = {0.0f};
    float bound_[15] = {0.0f};
    float delta_[15] = {0.0f};
    float inv_[TILE_VEC] = {0.0f};
    float invR_[TILE_VEC] = {0.0f};
};

#endif
