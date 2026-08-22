#ifndef MOE_EP_EXCEPTION_DUMP_WRITER_H
#define MOE_EP_EXCEPTION_DUMP_WRITER_H

#include "kernel_operator.h"
#include "moe_ep_exception_dump_defs.h"

using namespace AscendC;

__aicore__ inline void WriteMoeEpDumpMetadataToWin(GM_ADDR winBase, const MoeEpDumpMetadata *metaInTiling)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (GetBlockIdx() != 0U) {
        return;
    }
    if (winBase == nullptr || metaInTiling == nullptr) {
        return;
    }

    constexpr uint32_t copySize = static_cast<uint32_t>(
        (sizeof(MoeEpDumpMetadata) + MOE_EP_UB_ALIGN - 1UL) / MOE_EP_UB_ALIGN * MOE_EP_UB_ALIGN);

    LocalTensor<uint8_t> metaLocal{TPosition::LCM, 0, copySize};
    GlobalTensor<uint8_t> winMetaGlobal;
    GlobalTensor<uint8_t> tilingMetaGlobal;
    winMetaGlobal.SetGlobalBuffer(winBase, copySize);
    tilingMetaGlobal.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(metaInTiling), copySize);
    DataCopyPad(metaLocal, tilingMetaGlobal, {1, static_cast<uint16_t>(copySize), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopyPad(winMetaGlobal, metaLocal, {1, static_cast<uint16_t>(copySize), 0, 0, 0});
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

#endif
