/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_MTE_DIAGNOSTICS_H
#define MEGA_MOE_MTE_DIAGNOSTICS_H

#include "mega_moe_utils.h"

// This investigation branch enables diagnostics by default in mega_moe_types.h.
#ifndef MEGA_MOE_MTE_DIAG_MAX_RECV
#define MEGA_MOE_MTE_DIAG_MAX_RECV 2310
#endif
#ifndef MEGA_MOE_MTE_DIAG_CORE
#define MEGA_MOE_MTE_DIAG_CORE 5
#endif
#ifndef MEGA_MOE_MTE_DIAG_RECORDS
#define MEGA_MOE_MTE_DIAG_RECORDS 64
#endif

namespace MegaMoeImpl {
#if MEGA_MOE_MTE_DIAG
__aicore__ inline bool MteDiagEnabled(const Params &p)
{
    return p.tilingData->topoType == TOPO_TYPE_MTE &&
           (MEGA_MOE_MTE_DIAG_MAX_RECV == 0 || p.tilingData->maxOutputSize == MEGA_MOE_MTE_DIAG_MAX_RECV);
}

__aicore__ inline bool MteDiagSelected(const Params &p)
{
    if constexpr (g_coreType != AIV) {
        return false;
    }
    return MteDiagEnabled(p) && (MEGA_MOE_MTE_DIAG_CORE < 0 || GetBlockIdx() == MEGA_MOE_MTE_DIAG_CORE);
}

// Device string literals live in GM; preserve that address space in label parameters.
// No added barriers: these are scalar execution markers, not completion timestamps.
__aicore__ inline void MteDiagStage(const Params &p, const __gm__ char *stage)
{
    if (MteDiagSelected(p)) {
        printf("[MM_DIAG] rank=%u core=%u sub=%u max=%u stage=%s cycle=%llu\n", p.diagRank, GetBlockIdx(),
               GetSubBlockIdx(), p.tilingData->maxOutputSize, stage, static_cast<unsigned long long>(GetSystemCycle()));
    }
}

__aicore__ inline void MteDiagRequire(const Params &p, bool valid, const __gm__ char *site, int64_t value,
                                      uint64_t limit)
{
    if (MteDiagEnabled(p) && !valid) {
        printf("[MM_DIAG_INVALID] rank=%u core=%u sub=%u site=%s value=%lld limit=%llu max=%u\n", p.diagRank,
               GetBlockIdx(), GetSubBlockIdx(), site, static_cast<long long>(value),
               static_cast<unsigned long long>(limit), p.tilingData->maxOutputSize);
        AscendC::Trap();
    }
}

// Subtraction avoids overflow in offset + bytes. Limits describe the current logical
// partition, not the size/liveness of the remote HCCL registration.
__aicore__ inline void MteDiagRange(const Params &p, const __gm__ char *site, GM_ADDR base, uint64_t offset,
                                    uint64_t bytes, uint64_t limit, uint32_t peer, bool combine = false)
{
    if (!MteDiagEnabled(p)) {
        return;
    }
    bool valid = base != nullptr && offset <= limit && bytes <= limit - offset;
    uint32_t &records = p.diagRecords[combine ? 1 : 0];
    if (!valid || (MteDiagSelected(p) && records < MEGA_MOE_MTE_DIAG_RECORDS)) {
        printf("[MM_DIAG_ADDR] rank=%u core=%u site=%s peer=%u base=%llx offset=%llu bytes=%llu limit=%llu valid=%u\n",
               p.diagRank, GetBlockIdx(), site, peer, reinterpret_cast<unsigned long long>(base),
               static_cast<unsigned long long>(offset), static_cast<unsigned long long>(bytes),
               static_cast<unsigned long long>(limit), static_cast<uint32_t>(valid));
        if (valid && ++records == MEGA_MOE_MTE_DIAG_RECORDS) {
            printf("[MM_DIAG] rank=%u core=%u address_log_limit=%u; invalid checks remain active\n", p.diagRank,
                   GetBlockIdx(), static_cast<uint32_t>(MEGA_MOE_MTE_DIAG_RECORDS));
        }
    }
    if (!valid) {
        AscendC::Trap();
    }
}

__aicore__ inline void MteDiagInit(const Params &p, GM_ADDR workspace, GM_ADDR context)
{
    if (!MteDiagSelected(p)) {
        return;
    }
    WorkspaceLayout layout(p.tilingData);
    printf("[MM_DIAG_INIT_V1] rank=%u core=%u sub=%u bs=%u capacity=%u h=%u n=%u k=%u experts=%u max=%u prefetch=%u "
           "combine=%u\n",
           p.diagRank, GetBlockIdx(), GetSubBlockIdx(), p.tilingData->bs, p.tilingData->numMaxTokensPerRank,
           p.tilingData->h, p.tilingData->hiddenDim, p.tilingData->topK, p.tilingData->moeExpertPerRank,
           p.tilingData->maxOutputSize, static_cast<uint32_t>(p.tilingData->topkWeightsPrefetch),
           static_cast<uint32_t>(p.tilingData->combineQuantMode));
    printf("[MM_DIAG_LAYOUT] context=%llx workspace=%llx workspace_bytes=%llu route=%llx count=%llx quant=%llx "
           "combine=%llx\n",
           reinterpret_cast<unsigned long long>(context), reinterpret_cast<unsigned long long>(workspace),
           static_cast<unsigned long long>(layout.workspaceSize),
           reinterpret_cast<unsigned long long>(p.peermemInfo.maskRecvPtr),
           reinterpret_cast<unsigned long long>(p.peermemInfo.expertCountRecvPtr),
           reinterpret_cast<unsigned long long>(p.peermemInfo.quantTokenScalePtr),
           reinterpret_cast<unsigned long long>(p.peermemInfo.combineSendPtr));
    for (uint32_t peer = 0; peer < p.tilingData->epWorldSize; ++peer) {
        printf("[MM_DIAG_PEER] rank=%u core=%u peer=%u window_after_dump_prefix=%llx\n", p.diagRank, GetBlockIdx(),
               peer, reinterpret_cast<unsigned long long>(g_winRankAddr_[peer]));
    }
}
#endif
} // namespace MegaMoeImpl
#endif
