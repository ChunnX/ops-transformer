/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_EXCEPTION_DUMP_WRITER_H
#define MOE_EP_EXCEPTION_DUMP_WRITER_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "mc2_moe_context.h"
#include "moe_ep_exception_dump_defs.h"

namespace MoeEpExceptionDump {

__aicore__ inline void WriteMetadata(GM_ADDR context, GM_ADDR metadataSource)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (AscendC::GetBlockIdx() != 0U || context == nullptr || metadataSource == nullptr) {
        return;
    }

    auto commContext = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    GM_ADDR localWin = reinterpret_cast<GM_ADDR>(commContext->epHcclBuffer[commContext->epRankId]);
    AscendC::LocalTensor<uint8_t> metadataLocal{
        AscendC::TPosition::LCM, 0, static_cast<uint32_t>(sizeof(MoeEpDumpMetadata))};
    AscendC::GlobalTensor<uint8_t> metadataSourceGlobal;
    AscendC::GlobalTensor<uint8_t> metadataWinGlobal;
    metadataSourceGlobal.SetGlobalBuffer(metadataSource, sizeof(MoeEpDumpMetadata));
    metadataWinGlobal.SetGlobalBuffer(localWin, sizeof(MoeEpDumpMetadata));

    AscendC::DataCopyPad(metadataLocal, metadataSourceGlobal,
                         {1U, static_cast<uint16_t>(sizeof(MoeEpDumpMetadata)), 0U, 0U, 0U},
                         {false, 0U, 0U, 0U});
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
    // Scalar diagnostic fields can be updated in metadataLocal before the GM write.
    AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(metadataWinGlobal, metadataLocal,
                         {1U, static_cast<uint16_t>(sizeof(MoeEpDumpMetadata)), 0U, 0U, 0U});
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
}

// 临时维测：在 AIV0 上触发一次非法 GM 访问（aicore error），用于端到端验证
// exception dump 回调落盘链路。仅限调试验证，验证完成后应回退。
__aicore__ inline void DebugTrap()
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (AscendC::GetBlockIdx() != 0U) {
        return;
    }
    AscendC::GlobalTensor<int32_t> trapGlobal;
    trapGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(static_cast<uintptr_t>(0x1)), 1);
    trapGlobal.SetValue(0, 0);
}

} // namespace MoeEpExceptionDump

#endif // MOE_EP_EXCEPTION_DUMP_WRITER_H