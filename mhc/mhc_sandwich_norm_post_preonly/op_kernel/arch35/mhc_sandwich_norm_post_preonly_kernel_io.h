/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

// ========================================================================
// mhc_sandwich_norm_post_preonly_kernel_io.h
// IO building blocks: 双核同步、workspace 读写、输出写入
// ========================================================================

// ---- Dual-core synchronization ----

/// Write my scalars to workspace (own data slot).
/// Caller must follow with SyncAndReadPartner to exchange.
__aicore__ inline void DualCoreSyncExchange(
    float mySumSqPre, const float (&myXHat)[N_MAX],
    float &totalSumSqPre, float (&xHatVals)[N_MAX])
{
    uint32_t mySync1Slot   = (coreRole_ == 0) ? WS_SYNC1_CORE0 : WS_SYNC1_CORE1;
    uint32_t partSync1Slot = (coreRole_ == 0) ? WS_SYNC1_CORE1 : WS_SYNC1_CORE0;
    uint32_t myBlockIdx      = GetBlockIdx();
    uint32_t partnerBlockIdx = (coreRole_ == 0) ? (myBlockIdx + 1) : (myBlockIdx - 1);

    LocalTensor<float> scBuf = bufScalar_.Get<float>();

    // Pack sync1 block
    scBuf.SetValue(0, mySumSqPre);
    for (uint32_t o = 0; o < phiRows_; ++o)
        scBuf.SetValue(1 + o, myXHat[o]);
    for (uint32_t i = 1 + phiRows_; i < WS_SYNC_BLOCK; ++i)
        scBuf.SetValue(i, 0.0f);
    PipeBarrier<PIPE_V>();

    // Write scalars
    DataCopyExtParams wp = {1, 0, 0, 0, 0};
    wp.blockLen = WS_SYNC_BLOCK * sizeof(float);
    DataCopyPad(workspaceGm_[pairWsBase_ + mySync1Slot], scBuf, wp);
}

/// Write x2 from UB to workspace GM (for partner to read)
__aicore__ inline void WriteX2ToWorkspace(uint32_t tD, uint32_t tOff,
    LocalTensor<float> x2Src, uint32_t numHeads, uint32_t wsSlot)
{
    for (uint32_t hi = 0; hi < numHeads; ++hi) {
        DataCopyExtParams wp = {1, 0, 0, 0, 0};
        wp.blockLen = tD * sizeof(float);
        DataCopyPad(workspaceGm_[pairWsBase_ + wsSlot + (uint64_t)hi * D_ + tOff],
                    x2Src[hi * dimTile_], wp);
    }
}

// ---- Sense-reversing barrier (write-own-only) ----
// Replaces IBSet/IBWait to eliminate cross-core GM writes (mssanitizer OOB).
// Each core only writes to its own flag; reads partner's flag via spin.
// Polarity flips each round (0→1→0→...) so no reset/clear phase is needed.

/// Sense-reversing barrier: signal own readiness + wait for partner.
/// Each core writes ONLY to syncGm_[ownBlockIdx], reads partner's slot.
__aicore__ inline void SyncBarrier(uint32_t partnerBlockIdx)
{
    int32_t syncVal = (syncRound_ & 1) ? 0 : 1;
    uint32_t mySlot   = GetBlockIdx() * WS_SYNC_BLOCK;
    uint32_t partSlot = partnerBlockIdx * WS_SYNC_BLOCK;

    // Signal: write syncVal to own slot
    LocalTensor<int32_t> syncUb = bufSyncUb_.AllocTensor<int32_t>();
    syncUb.SetValue(0, syncVal);
    for (uint32_t i = 1; i < WS_SYNC_BLOCK; ++i) syncUb.SetValue(i, 0);
    PipeBarrier<PIPE_V>();
    DataCopy(syncGm_[mySlot], syncUb, WS_SYNC_BLOCK);
    PipeBarrier<PIPE_ALL>();
    bufSyncUb_.FreeTensor(syncUb);

    // Wait: spin-read partner's slot until syncVal
    syncUb = bufSyncUb_.AllocTensor<int32_t>();
    // 一个手写Barrier，用来在处理一个token的两个核之间做同步：
    // 1. 每个核把 syncVal 写进GM上自己的slot；
    // 2.while循环不停地从GM读对方核的slot到UB，直到它等于本轮的syncVal；
    // 3. 一旦读到对方的值，说明对方达到屏障点，跳出，双核对齐。
    while (true) {
        DataCopy(syncUb, syncGm_[partSlot], WS_SYNC_BLOCK);
        PipeBarrier<PIPE_ALL>();
        if (syncUb.GetValue(0) == syncVal) {
            break;
        }
    }
    bufSyncUb_.FreeTensor(syncUb);

    syncRound_++;
}

/// Sync + read partner's sync block, combine partials.
__aicore__ inline void SyncAndReadPartner(
    float mySumSqPre, const float (&myXHat)[N_MAX],
    float &totalSumSqPre, float (&xHatVals)[N_MAX])
{
    uint32_t partnerBlockIdx = (coreRole_ == 0) ? (GetBlockIdx() + 1) : (GetBlockIdx() - 1);
    uint32_t partSync1Slot   = (coreRole_ == 0) ? WS_SYNC1_CORE1 : WS_SYNC1_CORE0;

    PipeBarrier<PIPE_ALL>();
    SyncBarrier(partnerBlockIdx);

    // Read partner's sync1 block
    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    DataCopyExtParams rp = {1, 0, 0, 0, 0};
    rp.blockLen = WS_SYNC_BLOCK * sizeof(float);
    DataCopyPadExtParams<float> rpad = {false, 0, 0, 0};
    DataCopyPad(scBuf, workspaceGm_[pairWsBase_ + partSync1Slot], rp, rpad);
    PipeBarrier<PIPE_ALL>();

    totalSumSqPre = mySumSqPre + scBuf.GetValue(0);
    for (uint32_t o = 0; o < phiRows_; ++o)
        xHatVals[o] = myXHat[o] + scBuf.GetValue(1 + o);
}


/// In-place quantize round-trip: fp32 → io_dtype → fp32
/// Simulates stage-boundary dtype truncation to match golden
__aicore__ inline void QuantizeRoundTrip(LocalTensor<float> buf, uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, buf, count);
    PipeBarrier<PIPE_V>();  // RAW on tmp: CastToIO must finish writing before Cast reads
    Cast(buf, tmp, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    bufHInPrimeBf16_.FreeTensor(tmp);
}

/// Read partner's x2[hi] for one tile from workspace into dst buffer
__aicore__ inline void ReadPartnerX2Tile(LocalTensor<float> dst,
    uint32_t hi, uint32_t tOff, uint32_t tD)
{
    uint32_t partnerX2WsSlot = WS_HIN_BASE + (1 - coreRole_) * myHeadCount_ * D_;
    DataCopyExtParams rp = {1, 0, 0, 0, 0};
    rp.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> rpad = {false, 0, 0, 0};
    DataCopyPad(dst, workspaceGm_[pairWsBase_ + partnerX2WsSlot +
                (uint64_t)hi * D_ + tOff], rp, rpad);
    PipeBarrier<PIPE_ALL>();
}

// ---- Output helpers ----

/// Cast fp32 → T with correct RoundMode
__aicore__ inline void CastToIO(LocalTensor<T> dst, LocalTensor<float> src,
    uint32_t count)
{
    if constexpr (IsSameT<T, half>::value)
        Cast(dst, src, RoundMode::CAST_NONE, count);
    else
        Cast(dst, src, RoundMode::CAST_RINT, count);
}

/// Cast + write one tile of h_in_prime to GM
__aicore__ inline void WriteHInPrimeTile(LocalTensor<float> src,
    uint64_t gmOffset, uint32_t tD)
{
    LocalTensor<T> hIpBf16 = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(hIpBf16, src, tD);
    bufHInPrimeBf16_.EnQue(hIpBf16);
    hIpBf16 = bufHInPrimeBf16_.DeQue<T>();
    DataCopy(hInPrimeGm_[gmOffset], hIpBf16, tD);
    bufHInPrimeBf16_.FreeTensor(hIpBf16);

    // src(32) copyout hInF32
    if (returnHInF32_) {
        // tbuf分配的local tensor可以做搬入搬出，但要手动加同步
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);

        DataCopy(hInF32Gm_[gmOffset], src, tD);

        SetFlag<HardEvent::MTE3_V>(evtMte3toV_);
        WaitFlag<HardEvent::MTE3_V>(evtMte3toV_);
    }
}

/// Cast + write one head's x2_out tile to GM
__aicore__ inline void WriteX2OutTile(LocalTensor<float> src,
    uint64_t gmOffset, uint32_t tD)
{
    LocalTensor<T> x2Bf16 = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(x2Bf16, src, tD);
    bufHInPrimeBf16_.EnQue(x2Bf16);
    x2Bf16 = bufHInPrimeBf16_.DeQue<T>();
    DataCopy(x2OutGm_[gmOffset], x2Bf16, tD);
    bufHInPrimeBf16_.FreeTensor(x2Bf16);
}

// ---- Misc ----

/// Get x2[head hi, tile t] from the appropriate UB buffer (multi-tile dual-core)
///   tile 0: head0 → bufGamma1Fp32_, head1 → bufGamma2Fp32_
///   tile 1: head0..1 → bufX2Fp32_
__aicore__ inline __attribute__((always_inline)) LocalTensor<float>
GetX2HeadTile(uint32_t tile, uint32_t hi)
{
    if (tile == 0) {
        return (hi == 0) ? bufGamma1Fp32_.Get<float>()
                         : bufGamma2Fp32_.Get<float>();
    }
    return bufX2Fp32_.Get<float>()[hi * dimTile_];
}

__aicore__ inline uint32_t GetTileD(uint32_t tile) const
{
    if (lastDimTile_ != 0 && tile == dimLoop_ - 1) return lastDimTile_;
    return dimTile_;
}

// sum of ||x2[h]||^2 across N heads (single-core path)
__aicore__ inline float ComputeSumSqAllHeads(uint32_t tD);
// phi matmul for one tile (single-core path)
__aicore__ inline void ComputePhiMatmulTile(uint32_t tD, uint32_t tOff,
    float (&xHatVals)[N_MAX]);
// N*D global RMSNorm_mid (single-core single-tile)
__aicore__ inline void ApplyRmsNormMidSingleTile(uint32_t tD);

