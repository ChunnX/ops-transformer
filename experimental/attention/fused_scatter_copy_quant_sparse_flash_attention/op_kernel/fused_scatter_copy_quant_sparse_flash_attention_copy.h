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
 * \file fused_scatter_copy_quant_sparse_flash_attention_copy.h
 * \brief
 */

#ifndef FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_COPY_H
#define FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_COPY_H

#include "kernel_operator.h"
#include "fused_scatter_copy_quant_sparse_flash_attention_contract.h"
#include "fused_scatter_copy_quant_sparse_flash_attention_schedule.h"

namespace fused_scatter_copy_qsfa {
using namespace AscendC;

constexpr uint32_t ROW_BYTES = 656;
constexpr uint32_t UB_ROW_BYTES = 672;

__aicore__ inline int64_t RowOffsetInRequest(GlobalTensor<int32_t> &table, uint32_t width,
    uint32_t blocks, uint64_t tableRow, int32_t token)
{
    if (token < 0 || static_cast<uint32_t>(token) / 128 >= width) return -1;
    const int32_t block = table.GetValue(tableRow + token / 128);
    if (block < 0 || static_cast<uint32_t>(block) >= blocks) return -1;
    return (static_cast<int64_t>(block) * 128 + token % 128) * ROW_BYTES;
}

__aicore__ inline int64_t RowOffset(GlobalTensor<int32_t> &table, uint32_t width,
    uint32_t blocks, uint32_t batch, int32_t token)
{
    return RowOffsetInRequest(table, width, blocks, static_cast<uint64_t>(batch) * width, token);
}

// The gate is evaluated from device metadata so graph replay may select a
// different path. A heterogeneous batch may contain any mixture of MTP0..15.
__aicore__ inline bool RequiresUnionCopy(GM_ADDR counts, GM_ADDR misses, GM_ADDR ends,
    const FusedScatterCopyQuantSparseFlashAttentionTilingData &td)
{
    GlobalTensor<int32_t> c, m, q;
    c.SetGlobalBuffer((__gm__ int32_t *)counts);
    m.SetGlobalBuffer((__gm__ int32_t *)misses);
    q.SetGlobalBuffer((__gm__ int32_t *)ends);
    bool fallback = false;
    int32_t previous = 0;
    for (uint32_t b = 0; b < td.baseParams.batchSize; ++b) {
        const int32_t n = c.GetValue(b);
        const int32_t end = q.GetValue(b);
        ASSERT(n >= 0 && n <= COPY_CAPACITY);
        ASSERT(IsValidQuerySegment(
            previous, end, static_cast<int32_t>(td.baseParams.qSeqSize)));
        fallback |= NeedsUnionCopy(n);
        for (int32_t t = previous; t < end; ++t) {
            const int32_t miss = m.GetValue(t);
            ASSERT(miss >= 0 && miss <= 2048 && (n != 0 || miss == 0));
        }
        previous = end;
    }
    ASSERT(previous == td.baseParams.qSeqSize);
    return fallback;
}

__aicore__ inline void UnionCopy(TPipe &pipe, GM_ADDR hbm, GM_ADDR dram,
    GM_ADDR ht, GM_ADDR dt, GM_ADDR src, GM_ADDR dst, GM_ADDR counts,
    const FusedScatterCopyQuantSparseFlashAttentionTilingData &td)
{
    TBuf<TPosition::VECIN> storage;
    pipe.InitBuffer(storage, 2 * UB_ROW_BYTES);
    if ASCEND_IS_AIV {
        GlobalTensor<int8_t> hg, dg;
        GlobalTensor<int32_t> htable, dtable, sources, dests, sizes;
        hg.SetGlobalBuffer((__gm__ int8_t *)hbm);
        dg.SetGlobalBuffer((__gm__ int8_t *)dram);
        htable.SetGlobalBuffer((__gm__ int32_t *)ht);
        dtable.SetGlobalBuffer((__gm__ int32_t *)dt);
        sources.SetGlobalBuffer((__gm__ int32_t *)src);
        dests.SetGlobalBuffer((__gm__ int32_t *)dst);
        sizes.SetGlobalBuffer((__gm__ int32_t *)counts);
        auto buffers = storage.Get<int8_t>();
        event_t readReady[2], writeDone[2];
        bool wrote[2] = {false, false};
        int64_t pendingDst[2];
        uint32_t issued = 0, finished = 0;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            readReady[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_MTE3>());
            writeDone[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());
        }
        const uint32_t lanes = td.singleCoreParams.usedCoreNum * 2;
        for (uint32_t b = 0; b < td.baseParams.batchSize; ++b) {
            const int32_t n = sizes.GetValue(b);
            for (uint32_t i = GetBlockIdx(); i < n; i += lanes) {
                const uint64_t p = static_cast<uint64_t>(b) * COPY_CAPACITY + i;
                const auto so = RowOffset(dtable, td.dramMaxBlocks,
                    td.dramPhysicalBlocks, b, sources.GetValue(p));
                const auto dest = RowOffset(htable, td.baseParams.maxBlockNumPerBatch,
                    td.hbmPhysicalBlocks, b, dests.GetValue(p));
                ASSERT(so >= 0 && dest >= 0);
                const uint32_t slot = issued % 2;
                if (wrote[slot]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(writeDone[slot]);
                    wrote[slot] = false;
                }
                auto buf = buffers[slot * UB_ROW_BYTES];
                DataCopyPad(buf, dg[so], DataCopyExtParams{1, ROW_BYTES, 0, 0, 0},
                    DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
                pendingDst[slot] = dest;
                SetFlag<HardEvent::MTE2_MTE3>(readReady[slot]);
                ++issued;
                if (issued - finished == 2) {
                    const uint32_t out = finished % 2;
                    WaitFlag<HardEvent::MTE2_MTE3>(readReady[out]);
                    DataCopyPad(hg[pendingDst[out]], buffers[out * UB_ROW_BYTES],
                        DataCopyExtParams{1, ROW_BYTES, 0, 0, 0});
                    SetFlag<HardEvent::MTE3_MTE2>(writeDone[out]);
                    wrote[out] = true;
                    ++finished;
                }
            }
        }
        while (finished < issued) {
            const uint32_t out = finished % 2;
            WaitFlag<HardEvent::MTE2_MTE3>(readReady[out]);
            DataCopyPad(hg[pendingDst[out]], buffers[out * UB_ROW_BYTES],
                DataCopyExtParams{1, ROW_BYTES, 0, 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(writeDone[out]);
            wrote[out] = true;
            ++finished;
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (wrote[slot]) WaitFlag<HardEvent::MTE3_MTE2>(writeDone[slot]);
            pipe.ReleaseEventID<HardEvent::MTE2_MTE3>(readReady[slot]);
            pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(writeDone[slot]);
        }
    }
    SyncAll<false>();
}

} // namespace fused_scatter_copy_qsfa

#endif
