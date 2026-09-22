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

    uint32_t totalBlocks_{1};
    uint64_t tableBase_{0};
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

    InitIndicesBuffer(tileBytes_ * RELAY_BUFFER_NUM_INFER, sizeof(int32_t));
}

__aicore__ inline void EngramFetchLocalArch35::GatherRange(uint32_t start, uint32_t end)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    bool hasSf = HasSf();
    uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = indicesBatchSize_;
        if (cur + batchLen > end) {
            batchLen = end - cur;
        }
        CopyIndicesToUb(cur, batchLen);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < batchLen; i++) {
            int32_t entry = indicesLocal.GetValue(i);
            if (entry < 0 || entry >= numEntriesPerRank_) {
                continue;
            }
            uint64_t globalTokenIdx = static_cast<uint64_t>(cur) + i;
            GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
            GM_ADDR src = (GM_ADDR)tableBase_ + static_cast<uint64_t>(entry) * hiddenBytes;
            LocalCopySlice(dst, src, hiddenBytes);
            if (hasSf) {
                GatherSf(entry, globalTokenIdx);
            }
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
