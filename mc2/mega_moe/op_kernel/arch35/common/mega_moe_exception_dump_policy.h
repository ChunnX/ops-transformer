/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH35_EXCEPTION_DUMP_POLICY_H
#define MEGA_MOE_ARCH35_EXCEPTION_DUMP_POLICY_H

#include "../../../../common/op_kernel/mc2_exception_dump.h"
#include "mega_moe_workspace.h"
#include "../mega_moe_tiling.h"

namespace MC2MegaMoeA5Dump {

enum class Stage : uint32_t {
    INIT = 0,
    INPUT_PREPARE = 1,
    SHARED_EXPERT_GMM1 = 2,
    CROSS_RANK_SYNC_INPUT = 3,
    MOE_GMM1_ACTIVATION = 4,
    MOE_GMM2_COMBINE = 5,
    SHARED_EXPERT_GMM2 = 6,
    CROSS_RANK_SYNC_OUTPUT = 7,
    UNPERMUTE = 8,
    COMPLETE = 9,
    END = 10
};

struct MegaMoeExceptionDumpPolicy {
    using TilingDataT = MegaMoeTilingData;
    using StageEnumT = Stage;
    static constexpr MC2ExceptionDump::OpType OP_TYPE = MC2ExceptionDump::OpType::OP_TYPE_MEGA_MOE;
};

using ExceptionDumpEngine = MC2ExceptionDump::ExceptionDump<MegaMoeExceptionDumpPolicy>;

#if defined(__CCE_AICORE__)
// 普通模板与 Wave 模板的 Init 共用同一套 dump 登记（A8W4Wave 经继承复用普通模板 Init）。
// Dump 调用顺序即解析侧识别的条目编号（0~5），不能调整；maskAlignSize 直接消费调用方
// Init 已算好的 sendMaskConfig_.maskAlignSize，不在此重推公式。
__aicore__ inline void RegisterMegaMoeExceptionDump(
    ExceptionDumpEngine &engine, GM_ADDR dumpBase, GM_ADDR tilingGM, const MegaMoeTilingData *tilingData,
    const MegaMoeImpl::PeermemInfo &peermemInfo, uint32_t maskAlignSize, GM_ADDR epRankIdAddr)
{
    engine.Init(dumpBase, tilingGM);
    engine.UpdateStage(Stage::INIT);
    constexpr size_t syncSlotBytes = MegaMoeImpl::INT_CACHELINE * sizeof(int32_t);
    engine.Dump(peermemInfo.rankSyncInWorldPtr, tilingData->epWorldSize, sizeof(int32_t), syncSlotBytes);
    engine.Dump(peermemInfo.rankSyncInWorldPtr + 48U * 1024U, tilingData->blockAivNum, sizeof(int32_t),
                syncSlotBytes);
    const size_t maskSlotSize = maskAlignSize + MegaMoeImpl::ALIGN_32;
    engine.Dump(peermemInfo.maskRecvPtr + maskAlignSize,
                static_cast<size_t>(tilingData->moeExpertPerRank) * tilingData->epWorldSize, sizeof(int32_t),
                maskSlotSize);
    engine.Dump(epRankIdAddr, sizeof(uint32_t));
    if (tilingData->topoType == MegaMoeImpl::TOPO_TYPE_MTE) {
        const size_t maskRecvSize = static_cast<size_t>(Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->moeExpertPerRank) * static_cast<int64_t>(tilingData->epWorldSize) *
                static_cast<int64_t>(maskSlotSize),
            static_cast<int64_t>(MegaMoeImpl::ALIGN_512)));
        engine.Dump(peermemInfo.rankSyncInWorldPtr, static_cast<size_t>(MegaMoeImpl::PEERMEM_DATA_OFFSET));
        engine.Dump(peermemInfo.maskRecvPtr, maskRecvSize);
    }
}
#endif

static_assert(MC2ExceptionDump::DUMP_HEADER_REGION_SIZE + MC2ExceptionDump::DUMP_TILING_REGION_SIZE +
                      MC2ExceptionDump::DUMP_BLOCK_STAGE_REGION_SIZE +
                      MC2ExceptionDump::MAX_DUMP_ENTRIES * sizeof(MC2ExceptionDump::DumpParams) <=
                  static_cast<size_t>(MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE),
              "MegaMoe arch35 exception-dump metadata exceeds its reserved communication-window prefix");

} // namespace MC2MegaMoeA5Dump

#endif // MEGA_MOE_ARCH35_EXCEPTION_DUMP_POLICY_H
