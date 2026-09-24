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
 * \file engram_fetch_local_arch35.h
 * \brief engram_fetch arch35 kernel, local full-table mode (fetch_mode=1, inference only):
 *        every rank holds the full table, indices are local row ids, pure local GM copies, no cross-NPU traffic.
 *        Small 32B-aligned rows are gathered in batched waves (rowsPerWave rows per 24KB relay round) when
 *        at least 8 rows pack into one relay tile; other shapes use the per-row legacy relay.
 */

#ifndef ENGRAM_FETCH_LOCAL_ARCH35_H
#define ENGRAM_FETCH_LOCAL_ARCH35_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_ENGRAM_FETCH_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_tiling_data.h"
#include "../engram_fetch_utils.h"
#include "engram_fetch_infer_base_arch35.h"

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_KERNEL)

// Inference local mode: fetched[t] = localTable[indices[t]].
// Table base = commBuffer[rankId] (device mapping of the host pinned window); entry is a global row id.
class EngramFetchLocalArch35 : public EngramFetchInferBase {
public:
    __aicore__ inline EngramFetchLocalArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR sfTable,
                                GM_ADDR fetchedSf, GM_ADDR workspaceGM, AscendC::TPipe *pipe,
                                const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void GatherRange(uint32_t start, uint32_t end);
    __aicore__ inline void GatherWave(const AscendC::LocalTensor<int32_t> &indicesLocal, uint32_t batchOffset,
                                      uint32_t waveLen, uint32_t tokenBase);
    __aicore__ inline void GatherRowsLegacy(const AscendC::LocalTensor<int32_t> &indicesLocal, uint32_t batchLen,
                                            uint32_t tokenBase);

    uint32_t totalBlocks_{1};
    uint64_t tableBase_{0};
    bool batched_{false};
    uint32_t rowsPerWave_{1};
};

__aicore__ inline void EngramFetchLocalArch35::Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                                    GM_ADDR sfTable, GM_ADDR fetchedSf, GM_ADDR workspaceGM,
                                                    AscendC::TPipe *pipe, const EngramFetchTilingData *tilingData)
{
    (void)workspaceGM;
    InitInferBase(commContext, indices, fetched, sfTable, fetchedSf, pipe, tilingData);
    totalBlocks_ = AscendC::GetBlockNum();

    AscendC::GlobalTensor<uint64_t> commBufferGm;
    commBufferGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->commBuffer[0]);
    tableBase_ = commBufferGm.GetValue(rankId_);

    // Batched-wave gather packs rowsPerWave_ consecutive rows into one relay buffer (all row loads
    // data + SF issued back-to-back, one EnQue/DeQue, one contiguous MTE3 store plus per-row SF
    // stores) instead of one full relay round trip per row. The infer base sizes each relay buffer
    // to a single row (tileBytes_ = AlignUp(hiddenBytes_, 32B), capped at TILE_BYTES_INFER_MAX),
    // which leaves no room to pack rows, so the relay is re-widened to the 24KB infer tile and every
    // wave keeps rowsPerWave_ * rowBytes <= tileBytes_.
    uint32_t sfBytes = 0U;
    if (HasSf()) {
        sfBytes = static_cast<uint32_t>(numSfPacks_) * static_cast<uint32_t>(sfElemSize_);
    }
    uint32_t sfSlot = (sfBytes + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; // 32B-aligned SF slot in UB
    uint64_t rowBytes = static_cast<uint64_t>(hiddenBytes_);
    if (sfBytes > 0U) {
        rowBytes += sfSlot;
    }
    batched_ = false;
    rowsPerWave_ = 1U;
    uint64_t relayUb = static_cast<uint64_t>(tileBytes_) * RELAY_BUFFER_NUM_INFER;
    // Batching only pays off when many small rows share one relay buffer: small rows expose the read
    // latency in the per-row relay, while large rows already keep MTE2 saturated and the wave
    // restructuring measured slower for them (h=9216 on the old 32KB base). Keep rows that are not
    // 32B-aligned or that pack fewer than minRowsPerWave rows per wave on the legacy path.
    constexpr uint32_t minRowsPerWave = 8U;
    bool rowAligned = hiddenBytes_ > 0 && static_cast<uint64_t>(hiddenBytes_) % UB_ALIGN == 0;
    bool wavePacks = rowAligned && rowBytes <= TILE_BYTES_INFER_MAX &&
                     static_cast<uint64_t>(TILE_BYTES_INFER_MAX) / rowBytes >= minRowsPerWave;
    if (wavePacks) {
        // Rebinding relayQue_ carves new 24KB tiles after the single-row tiles the base already
        // carved from the UB pool; the base tiles become dead space and must stay counted in the
        // indices-buffer UB budget. Skip wave packing when the widened relay would not fit.
        uint64_t waveRelayUb = (static_cast<uint64_t>(tileBytes_) + TILE_BYTES_INFER_MAX) * RELAY_BUFFER_NUM_INFER;
        if (waveRelayUb + UB_RESERVED_SIZE <= ubSize_) {
            tileBytes_ = TILE_BYTES_INFER_MAX;
            rowsPerWave_ = static_cast<uint32_t>(static_cast<uint64_t>(tileBytes_) / rowBytes);
            batched_ = true;
            tpipe_->InitBuffer(relayQue_, RELAY_BUFFER_NUM_INFER, tileBytes_);
            relayUb = waveRelayUb;
        }
    }

    InitIndicesBuffer(relayUb, sizeof(int32_t));
}

// One batched relay round for waveLen consecutive tokens: all row loads (data + SF) are issued into a
// single relay buffer back-to-back (deep memory-level parallelism), then drained by one EnQue/DeQue and
// shipped out with one contiguous data store plus per-row SF stores. Consecutive tokens keep the data
// destination contiguous, which is what makes the single big store possible. Invalid ids map to row 0
// so the contiguous store always writes well-defined bytes (the legacy path leaves the destination row
// untouched instead). The 6-deep relay queue lets consecutive waves overlap: MTE2 loads of wave i+1
// run while the MTE3 stores of earlier waves drain.
__aicore__ inline void EngramFetchLocalArch35::GatherWave(const AscendC::LocalTensor<int32_t> &indicesLocal,
                                                          uint32_t batchOffset, uint32_t waveLen, uint32_t tokenBase)
{
    const bool hasSf = HasSf();
    const uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    const uint32_t sfBytes = hasSf ? static_cast<uint32_t>(numSfPacks_) * static_cast<uint32_t>(sfElemSize_) : 0U;
    const uint32_t sfSlot = hasSf ? (sfBytes + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN : 0U;

    AscendC::GlobalTensor<uint8_t> srcGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)tableBase_);
    AscendC::LocalTensor<uint8_t> buf = relayQue_.AllocTensor<uint8_t>();

    // Wave layout inside one relay buffer: [waveLen rows of hiddenBytes data][waveLen 32B-aligned SF
    // slots]. waveLen * rowBytes <= tileBytes_ holds by construction, so every offset stays in-buffer
    // and the single data store never exceeds tileBytes_ (blockLen is uint16 in bytes).
    AscendC::DataCopyExtParams loadParams{1U, static_cast<uint32_t>(hiddenBytes), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> loadPad{false, 0, 0, 0};
    const uint32_t dataBytes = static_cast<uint32_t>(waveLen * hiddenBytes);
    for (uint32_t j = 0; j < waveLen; j++) {
        int32_t entry = indicesLocal.GetValue(batchOffset + j);
        if (entry < 0 || entry >= numEntriesPerRank_) {
            entry = 0;
        }
        AscendC::DataCopyPad(buf[static_cast<uint32_t>(j * hiddenBytes)],
                             srcGm[static_cast<uint64_t>(entry) * hiddenBytes], loadParams, loadPad);
    }
    if (sfBytes > 0U) {
        AscendC::GlobalTensor<uint8_t> sfSrcGm;
        sfSrcGm.SetGlobalBuffer(sfTableGM_);
        AscendC::DataCopyExtParams sfLoadParams{1U, sfBytes, 0U, 0U, 0U};
        for (uint32_t j = 0; j < waveLen; j++) {
            int32_t entry = indicesLocal.GetValue(batchOffset + j);
            if (entry < 0 || entry >= numEntriesPerRank_) {
                entry = 0;
            }
            AscendC::DataCopyPad(buf[dataBytes + j * sfSlot], sfSrcGm[static_cast<uint64_t>(entry) * sfBytes],
                                 sfLoadParams, loadPad);
        }
    }

    relayQue_.EnQue<uint8_t>(buf);
    buf = relayQue_.DeQue<uint8_t>();

    AscendC::GlobalTensor<uint8_t> dstGm;
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)fetchedGM_);
    AscendC::DataCopyParams dataStoreParams{1U, static_cast<uint16_t>(dataBytes), 0U, 0U};
    AscendC::DataCopyPad(dstGm[static_cast<uint64_t>(tokenBase) * hiddenBytes], buf, dataStoreParams);
    if (sfBytes > 0U) {
        AscendC::GlobalTensor<uint8_t> sfDstGm;
        sfDstGm.SetGlobalBuffer((__gm__ uint8_t *)fetchedSfGM_);
        AscendC::DataCopyParams sfStoreParams{1U, static_cast<uint16_t>(sfBytes), 0U, 0U};
        for (uint32_t j = 0; j < waveLen; j++) {
            AscendC::DataCopyPad(sfDstGm[static_cast<uint64_t>(tokenBase + j) * sfBytes], buf[dataBytes + j * sfSlot],
                                 sfStoreParams);
        }
    }
    relayQue_.FreeTensor<uint8_t>(buf);
}

// Legacy per-row relay path, kept for row strides that are not 32B-aligned or too large to pack a
// wave (rowsPerWave_ < 8): one LocalCopySlice relay round per row, invalid ids are skipped and the
// destination row is left untouched.
__aicore__ inline void EngramFetchLocalArch35::GatherRowsLegacy(const AscendC::LocalTensor<int32_t> &indicesLocal,
                                                                uint32_t batchLen, uint32_t tokenBase)
{
    const bool hasSf = HasSf();
    const uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    for (uint32_t i = 0; i < batchLen; i++) {
        int32_t entry = indicesLocal.GetValue(i);
        if (entry < 0 || entry >= numEntriesPerRank_) {
            continue;
        }
        uint64_t globalTokenIdx = static_cast<uint64_t>(tokenBase) + i;
        GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
        GM_ADDR src = (GM_ADDR)tableBase_ + static_cast<uint64_t>(entry) * hiddenBytes;
        LocalCopySlice(dst, src, hiddenBytes);
        if (hasSf) {
            GatherSf(entry, globalTokenIdx);
        }
    }
}

__aicore__ inline void EngramFetchLocalArch35::GatherRange(uint32_t start, uint32_t end)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = indicesBatchSize_;
        if (cur + batchLen > end) {
            batchLen = end - cur;
        }
        CopyIndicesToUb(cur, batchLen);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        if (batched_) {
            uint32_t done = 0;
            while (done < batchLen) {
                uint32_t waveLen = batchLen - done;
                if (waveLen > rowsPerWave_) {
                    waveLen = rowsPerWave_;
                }
                GatherWave(indicesLocal, done, waveLen, cur + done);
                done += waveLen;
            }
        } else {
            GatherRowsLegacy(indicesLocal, batchLen, cur);
        }
        cur += batchLen;
    }
}

__aicore__ inline void EngramFetchLocalArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }
        // Remainder-aware range split: all AIV cores participate and the per-core token count differs by <= 1
        // (same style as GetCoreAssignment in sliced mode; naive ceil split idles nearly half the cores
        // when numTokens % totalBlocks is small).
        auto assign = GetTokenRangeAssignment(totalBlocks_, aivId_, static_cast<uint32_t>(numTokens_));
        GatherRange(assign.start, assign.start + assign.count);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

#endif

} // namespace Mc2Kernel

#endif
